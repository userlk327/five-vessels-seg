"""Tiled SAM with OVERLAPPING-tile inference: a secondary (sensitivity) variant, decided on val
after the tile-seam check (user decision, Phase 5/6). The main tiled SAM stays as frozen.

    python -m src.sam_overlap --split val                   # predict val, select its own threshold
    python -m src.sam_overlap --split test --final-test     # Phase 6: test, once

Same trained model as the main tiled SAM (outputs/sam_lora_tiled/seed0/best.pt, checked against
its saved config); only inference differs, so there is no training of its own:
  * tiles of 1024x1024 native px at stride 512 -> 3 x 3 = 9 tiles per 2048 image (the 4 tiles
    of the main variant are 4 of these 9);
  * each tile's 256x256 decoder logits go onto the same 512x512 grid as the main variant
    (1 grid px = 4 native px; tile origin = multiple of 128 grid px);
  * overlapping logits are averaged with a separable tent weight per tile,
    w(i) = 1 - |(i + 0.5) / 128 - 1| for i = 0..255 (about 0 at the tile edges, 1 at its centre).
    With 50 % overlap the tents sum to 1 inside the image, so interior pixels get a convex
    combination of the tiles that cover them and no tile edge produces a hard seam;
  * sigmoid of the blended logits -> float16 probability map at 512, scored exactly like the
    main tiled SAM and the CNNs (bilinear to 2048, threshold chosen on val).
"""
from __future__ import annotations

import argparse
import json
import os
import time

import numpy as np
import torch
import yaml

SOURCE_RUN = "sam_lora_tiled"
RUN_NAME = "sam_lora_tiled_overlap"
STRIDE = 512                                    # native px; tile = 1024 (sam_resolution)
LABEL = "SAM LoRA, tiled, overlapping-tile inference (3x3 tiles, stride 512, tent-blended logits; sensitivity)"


def tent(n: int) -> np.ndarray:
    i = np.arange(n) + 0.5
    return 1.0 - np.abs(i / (n / 2) - 1.0)


def tile_origins(native: int, tile: int, stride: int) -> list[int]:
    out = list(range(0, native - tile + 1, stride))
    if out[-1] != native - tile:
        raise ValueError(f"stride {stride} does not tile {native} with {tile}")
    return out


def blend_tiles(logits: np.ndarray, origins: list[tuple[int, int]], grid: int, scale: int) -> np.ndarray:
    """logits (T, h, w) of tiles whose native origins are ``origins`` -> (grid, grid) blended logits.
    ``scale`` = native px per grid px."""
    t, h, w = logits.shape
    wy, wx = tent(h), tent(w)
    weight = np.outer(wy, wx)
    num = np.zeros((grid, grid), np.float64)
    den = np.zeros((grid, grid), np.float64)
    for k, (y0, x0) in enumerate(origins):
        gy, gx = y0 // scale, x0 // scale
        num[gy:gy + h, gx:gx + w] += weight * logits[k]
        den[gy:gy + h, gx:gx + w] += weight
    if (den <= 0).any():
        raise ValueError("some grid pixels are not covered by any tile")
    return (num / den).astype(np.float32)


def load_source_model(seed: int, device):
    from src.checkpoint import FORMAT_VERSION
    from src.models import build_model
    from src.utils import config_hash, load_config, run_dir_for
    cfg = load_config("configs/sam_lora_tiled.yaml")
    run_dir = run_dir_for(cfg, seed)
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    if config_hash(run_cfg) != ckpt["config_hash"] or ckpt.get("format_version") != FORMAT_VERSION or ckpt["seed"] != seed:
        raise RuntimeError(f"{run_dir}: config hash, format or seed does not match best.pt")
    model = build_model(run_cfg)
    model.load_state_dict(ckpt["model"])
    return run_cfg, ckpt, model.to(device).eval()


def predict(seed: int, split: str, final_test: bool, chunk: int = 3) -> tuple[dict, dict]:
    from src.data import cache_image_dir, read_rgb, sha256_file, split_ids, to_sam_input, write_npz_archive
    from src.train import amp_settings
    from src.utils import configure_cudnn, environment_info, portable_path, prob_archive_path, project_path

    if split == "test" and not final_test:
        raise SystemExit("The test split is predicted once, in Phase 6: pass --final-test.")
    out_dir = project_path("outputs") / RUN_NAME / f"seed{seed}"
    archive = prob_archive_path(out_dir, split)
    if split == "test" and archive.exists():
        raise SystemExit(f"{portable_path(archive)} exists: the test split is predicted once; never overwritten.")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    cfg, ckpt, model = load_source_model(seed, device)
    configure_cudnn(cfg)
    native, tile = int(cfg["data"]["native_resolution"]), int(cfg["data"]["sam_resolution"])
    dec = int(cfg["data"]["sam_decoder_resolution"])
    scale = tile // dec                                              # native px per decoder px (4)
    grid = native // scale                                           # 512, as the main tiled SAM
    starts = tile_origins(native, tile, STRIDE)
    origins = [(y, x) for y in starts for x in starts]
    use_amp, amp_dtype = amp_settings(cfg, device)
    ids = split_ids(cfg, split)
    probs, t0 = {}, time.time()
    with torch.no_grad():
        for n, iid in enumerate(ids):
            img = read_rgb(cache_image_dir(cfg, native) / f"{iid}.png")
            tiles = np.stack([to_sam_input(np.ascontiguousarray(img[y:y + tile, x:x + tile])) for y, x in origins])
            low = []
            for j in range(0, len(tiles), chunk):
                with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                    low.append(model(torch.from_numpy(tiles[j:j + chunk]).to(device)).float().cpu().numpy()[:, 0])
            blended = blend_tiles(np.concatenate(low), origins, grid, scale)
            probs[iid] = (1.0 / (1.0 + np.exp(-blended))).astype(np.float16)
            if n % 40 == 0:
                print(f"[overlap] {n + 1}/{len(ids)} {split} images", flush=True)
    write_npz_archive(archive, {i: probs[i] for i in ids})
    info = {"label": LABEL, "split": split, "n_images": len(ids), "source_run": f"{SOURCE_RUN}/seed{seed}",
            "checkpoint": portable_path(project_path("outputs") / SOURCE_RUN / f"seed{seed}" / "best.pt"),
            "best_epoch": int(ckpt["epoch"]), "config_hash": ckpt["config_hash"],
            "tiles": {"size": tile, "stride": STRIDE, "origins_native": origins, "blend": "tent-weighted mean of logits"},
            "output": portable_path(archive), "output_sha256": sha256_file(archive)[:16], "dtype": "float16",
            "resolution": grid, "amp": use_amp, "test_time_augmentation": False, "post_processing": False,
            "seconds": round(time.time() - t0, 1),
            **{k: v for k, v in environment_info().items() if k in ("time", "host", "gpu", "slurm_job_id", "git")}}
    with open(archive.parent / "predict_info.json", "w") as f:
        json.dump(info, f, indent=1)
    print(f"[overlap] {len(ids)} {split} images -> {portable_path(archive)} ({grid}x{grid}, {info['seconds']}s)")
    return cfg, info


def main() -> None:
    from src.evaluate import score_run
    from src.utils import project_path
    ap = argparse.ArgumentParser(description="Tiled SAM with overlapping-tile inference (sensitivity variant)")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    a = ap.parse_args()
    cfg, info = predict(a.seed, a.split, a.final_test)
    score_run(cfg, model=RUN_NAME, seed=a.seed, split=a.split, archive=project_path(info["output"]),
              out_dir=project_path(cfg["paths"]["results_dir"]) / "eval" / RUN_NAME / f"seed{a.seed}",
              workers=a.workers, final_test=a.final_test,
              extra_meta={"label": LABEL, "source_run": info["source_run"], "tiles": info["tiles"]})


if __name__ == "__main__":
    main()
