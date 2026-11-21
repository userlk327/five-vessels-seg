"""Model 5a: zero-shot SAM ViT-B with ORACLE point prompts. Label everywhere:
"zero-shot, oracle point prompts". Reported in its own table, never ranked against the
automatic models (the prompts come from each image's own ground truth: the Rule 3 exception).

    python -m src.sam_oracle --split val                  # every K in prompts.k_grid; K chosen on val
    python -m src.sam_oracle --split test --final-test    # Phase 6 only, the chosen K

Standard SAM usage on whole images: ``SamPredictor`` on the CLAHE image at 1024,
``multimask_output=False``, ``return_logits=True`` -> logits at 1024 (SAM's postprocessing)
-> sigmoid -> float16 probability map at 1024, scored by the shared pipeline (threshold
chosen on val per K).

Prompts per image, from its 2048 ground truth and FOV:
  positives  K pixels drawn uniformly from the GT skeleton inside the FOV
  negatives  K pixels drawn uniformly from FOV background at least ``negative_margin_px``
             (px at 2048) from any vessel pixel
  One permutation per image, so the prompt sets are nested (K = 5 is the first 5 of K = 10).
  RNG: RandomState(seed * 1_000_003 + crc32(image_id)): independent of order and of K.
  Coordinates are converted to the 1024 input frame: (x + 0.5) / 2 - 0.5.
"""
from __future__ import annotations

import argparse
import json
import zlib

import numpy as np
from scipy import ndimage as ndi
from scipy.special import expit


def prompt_rng(seed: int, image_id: str) -> np.random.RandomState:
    return np.random.RandomState((seed * 1_000_003 + zlib.crc32(image_id.encode())) % 2**32)


def sample_prompts(gt: np.ndarray, fov: np.ndarray, k: int, rng, margin: float) -> tuple[np.ndarray, np.ndarray]:
    """(k, 2) positive and (k, 2) negative points as (x, y) in the 2048 frame."""
    from src.metrics import skeleton
    gt, fov = gt.astype(bool), fov.astype(bool)
    pos = np.argwhere(skeleton(gt) & fov)
    neg = np.argwhere(fov & (ndi.distance_transform_edt(~gt) > margin))
    if len(pos) < k or len(neg) < k:
        raise ValueError(f"not enough prompt candidates ({len(pos)} positive, {len(neg)} negative) for K={k}")
    pos = pos[rng.permutation(len(pos))[:k]]
    neg = neg[rng.permutation(len(neg))[:k]]
    return pos[:, ::-1].astype(np.float64), neg[:, ::-1].astype(np.float64)


def to_input_frame(xy: np.ndarray, native: int, size: int) -> np.ndarray:
    """Pixel-centre conversion from the native frame to the SAM input frame."""
    s = size / native
    return (xy + 0.5) * s - 0.5


def run(cfg: dict, split: str, ks, final_test: bool, workers: int) -> None:
    import torch
    from segment_anything import SamPredictor, sam_model_registry

    from src.data import cache_image_dir, raw_mask_path, read_rgb, read_unchanged, split_ids, write_npz_archive
    from src.evaluate import load_fov, score_run
    from src.utils import project_path, run_dir_for

    if split == "test" and not final_test:
        raise SystemExit("The test split is run once, in Phase 6: pass --final-test.")
    pc = cfg["prompts"]
    seed, margin = int(pc["seed"]), float(pc["negative_margin_px"])
    native, size = int(cfg["data"]["native_resolution"]), int(cfg["data"]["sam_resolution"])
    device = "cuda" if torch.cuda.is_available() else "cpu"
    sam = sam_model_registry["vit_b"](checkpoint=str(project_path(cfg["model"]["checkpoint"]))).to(device).eval()
    predictor = SamPredictor(sam)
    ids = split_ids(cfg, split)
    if split == "test":                                   # predicted once (Phase 6); never overwritten
        done = [k for k in ks if (run_dir_for({**cfg, "run_name": f"sam_oracle_k{k}"}, seed) / split / "probs.npz").exists()]
        if done:
            raise SystemExit(f"test predictions already exist for K = {done}; they are never overwritten.")
    kmax = max(ks)
    probs = {k: {} for k in ks}
    prompts = {}
    for n, iid in enumerate(ids):
        gt = read_unchanged(raw_mask_path(cfg, iid)) > 0
        pos, neg = sample_prompts(gt, load_fov(cfg, iid), kmax, prompt_rng(seed, iid), margin)
        prompts[iid] = np.concatenate([pos, neg]).astype(np.float32)       # 2048 frame: kmax pos, then kmax neg
        predictor.set_image(read_rgb(cache_image_dir(cfg, size) / f"{iid}.png"))
        for k in ks:
            coords = to_input_frame(np.concatenate([pos[:k], neg[:k]]), native, size)
            labels = np.r_[np.ones(k), np.zeros(k)]
            logits, _, _ = predictor.predict(point_coords=coords, point_labels=labels,
                                             multimask_output=bool(pc["multimask_output"]), return_logits=True)
            probs[k][iid] = expit(logits[0].astype(np.float32)).astype(np.float16)
        if n % 20 == 0:
            print(f"[oracle] {n + 1}/{len(ids)} {split} images", flush=True)
    choice = {}
    for k in ks:
        name = f"sam_oracle_k{k}"
        run_dir = run_dir_for({**cfg, "run_name": name}, seed)
        archive = run_dir / split / "probs.npz"
        write_npz_archive(archive, {i: probs[k][i] for i in ids})
        np.savez_compressed(run_dir / split / "prompts.npz",
                            **{i: np.concatenate([p[:k], p[kmax:kmax + k]]) for i, p in prompts.items()})
        m = score_run(cfg, model=name, seed=seed, split=split, archive=archive,
                      out_dir=project_path(cfg["paths"]["results_dir"]) / "eval" / name / f"seed{seed}",
                      workers=workers, final_test=final_test,
                      extra_meta={"label": cfg["label"], "k": k, "prompt_seed": seed, "negative_margin_px": margin})
        choice[k] = {"threshold": float(m.threshold.iloc[0]), "mean_dice": float(m.dice.mean()),
                     "mean_thin_recall": float(m.cl_recall_thin.mean())}
    if split == "val":
        best = max(ks, key=lambda k: (choice[k]["mean_dice"], -k))
        out = project_path(cfg["paths"]["results_dir"]) / "eval" / "sam_oracle_choice.json"
        with open(out, "w") as f:
            json.dump({"label": cfg["label"], "chosen_k": best, "rule": "highest mean val Dice at each K's own "
                       "val-selected threshold; ties -> smaller K", "per_k": choice}, f, indent=1)
        print(f"[oracle] chosen K = {best}: {json.dumps(choice)} -> {out}")


def main() -> None:
    import os

    from src.utils import load_config
    ap = argparse.ArgumentParser(description="Zero-shot SAM with oracle point prompts")
    ap.add_argument("--config", default="configs/sam_oracle.yaml")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--k", type=int, nargs="+", help="default: prompts.k_grid (val) / the chosen K (test)")
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.k:
        ks = a.k
    elif a.split == "test":
        from src.utils import project_path
        ks = [json.loads((project_path(cfg["paths"]["results_dir"]) / "eval" / "sam_oracle_choice.json").read_text())["chosen_k"]]
    else:
        ks = cfg["prompts"]["k_grid"]
    run(cfg, a.split, ks, a.final_test, a.workers)


if __name__ == "__main__":
    main()
