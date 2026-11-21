"""Phase 6 cost table (CLAUDE.md Phase 6, table 6): parameters, inference time per image,
peak GPU memory at inference, FLOPs, plus training time / memory from the saved run summaries.

    python -m src.benchmark          # one GPU job (1x L40S, the training GPU type)

Writes results/tables/cost.csv and results/checks/benchmark_info.json.

Timing (per model, seed 0 checkpoints; same procedure for every model):
  * batch 1 = one image per call, over the 200 TEST images (CLAUDE.md); 10 warm-up calls first;
  * the timed span is the model's own inference path exactly as used for its saved
    predictions: from the input tensor(s) on the GPU to the probability map at the model's
    output grid, including tiling/stitching/blending and the sigmoid, with
    torch.cuda.synchronize() before and after; disk I/O and CPU preprocessing are excluded
    (tiles of one image: 4 in one call for tiled SAM, 9 in calls of 3 for the overlap variant,
    as in their prediction code);
  * oracle SAM: SamPredictor.set_image + predict (K = 5 points, prompts sampled beforehand);
  * mixed precision as at inference (fp16 autocast for the trained models; the oracle runs
    SAM's default float32 path, as in src/sam_oracle.py).
The outputs are DISCARDED: nothing is saved or scored, so this is not a second prediction of
the test set. FLOPs: torch.utils.flop_counter.FlopCounterMode over one image's inference
(counts matmul / convolution FLOPs; 1 MAC = 2 FLOPs). Peak GPU memory at inference:
torch.cuda.max_memory_allocated over the timed loop (weights included).
"""
from __future__ import annotations

import json
import time

import numpy as np
import torch
import yaml

MODELS = (
    ("unet", "U-Net", "configs/unet.yaml", "cnn"),
    ("attention_unet", "Attention U-Net", "configs/attention_unet.yaml", "cnn"),
    ("multiscale_unet", "Multi-Scale U-Net", "configs/multiscale_unet.yaml", "cnn"),
    ("ms_attention_unet", "Multi-Scale Attention U-Net", "configs/ms_attention_unet.yaml", "cnn"),
    ("sam_lora_tiled", "SAM LoRA, tiled 2x2 (main)", "configs/sam_lora_tiled.yaml", "sam_tiled"),
    ("sam_lora_tiled_overlap", "SAM LoRA, tiled, overlapping-tile inference (sensitivity)", "configs/sam_lora_tiled.yaml",
     "sam_overlap"),
    ("sam_lora_whole", "SAM LoRA, whole image", "configs/sam_lora_whole.yaml", "sam_whole"),
    ("sam_oracle_k5", "SAM zero-shot, oracle point prompts (K=5; separate table)", "configs/sam_oracle.yaml", "oracle"),
)
SEED, WARMUP = 0, 10


def load_trained(config_path: str, seed: int, device):
    from src.checkpoint import FORMAT_VERSION
    from src.models import build_model
    from src.utils import config_hash, load_config, run_dir_for
    run_dir = run_dir_for(load_config(config_path), seed)
    ckpt = torch.load(run_dir / "best.pt", map_location="cpu", weights_only=True)
    cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    if config_hash(cfg) != ckpt["config_hash"] or ckpt.get("format_version") != FORMAT_VERSION or ckpt["seed"] != seed:
        raise RuntimeError(f"{run_dir}: config hash, format or seed does not match best.pt")
    model = build_model(cfg)
    model.load_state_dict(ckpt["model"])
    return cfg, model.to(device).eval()


def make_case(kind: str, config_path: str, ids: list[str], device):
    """-> (cfg, model, params_total, params_trainable, inputs(i) -> CPU data, run(data) -> output)."""
    from src.data import (build_cnn_dataset, cache_image_dir, raw_mask_path, read_rgb, read_unchanged,
                          to_sam_input)
    from src.train import amp_settings
    from src.utils import load_config

    if kind == "oracle":
        from segment_anything import SamPredictor, sam_model_registry
        from scipy.special import expit

        from src.evaluate import load_fov
        from src.sam_oracle import prompt_rng, sample_prompts, to_input_frame
        from src.utils import project_path
        cfg = load_config(config_path)
        sam = sam_model_registry["vit_b"](checkpoint=str(project_path(cfg["model"]["checkpoint"]))).to(device).eval()
        sam.requires_grad_(False)     # inference only; the FLOP counter's hooks fail on outputs derived from
        predictor = SamPredictor(sam)  # trainable parameters (the prompt encoder's no-mask embedding)
        native, size, pc = int(cfg["data"]["native_resolution"]), int(cfg["data"]["sam_resolution"]), cfg["prompts"]
        k = json.loads((project_path("results/eval") / "sam_oracle_choice.json").read_text())["chosen_k"]

        def inputs(i):
            iid = ids[i]
            gt = read_unchanged(raw_mask_path(cfg, iid)) > 0
            pos, neg = sample_prompts(gt, load_fov(cfg, iid), k, prompt_rng(int(pc["seed"]), iid),
                                      float(pc["negative_margin_px"]))
            coords = to_input_frame(np.concatenate([pos, neg]), native, size)
            return read_rgb(cache_image_dir(cfg, size) / f"{iid}.png"), coords, np.r_[np.ones(k), np.zeros(k)]

        def run(data):
            rgb, coords, labels = data
            predictor.set_image(rgb)
            logits, _, _ = predictor.predict(point_coords=coords, point_labels=labels,
                                             multimask_output=bool(pc["multimask_output"]), return_logits=True)
            return expit(logits[0])

        n = sum(p.numel() for p in sam.parameters())
        return cfg, sam, n, 0, inputs, run, {"k": k, "amp": False}

    cfg, model = load_trained(config_path, SEED, device)
    use_amp, amp_dtype = amp_settings(cfg, device)
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)

    def fwd(x):
        with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            return model(x)

    if kind == "cnn":
        ds = build_cnn_dataset(cfg, ids, train=False, with_masks=False)

        def inputs(i):
            return ds[i]["image"][None]

        def run(x):
            return torch.sigmoid(fwd(x.to(device)).float())
    elif kind in ("sam_tiled", "sam_whole"):
        from src.sam_train import prob_maps, sam_dataset, tiles_per_image
        ds, k = sam_dataset(cfg, ids, train=False, with_masks=False), tiles_per_image(cfg)

        def inputs(i):
            return torch.stack([ds[i * k + t]["image"] for t in range(k)])

        def run(x):
            return prob_maps(fwd(x.to(device)), cfg)
    elif kind == "sam_overlap":
        from src.sam_overlap import STRIDE, blend_tiles, tile_origins
        native, tile = int(cfg["data"]["native_resolution"]), int(cfg["data"]["sam_resolution"])
        scale = tile // int(cfg["data"]["sam_decoder_resolution"])
        starts = tile_origins(native, tile, STRIDE)
        origins = [(y, x) for y in starts for x in starts]

        def inputs(i):
            img = read_rgb(cache_image_dir(cfg, native) / f"{ids[i]}.png")
            return torch.from_numpy(np.stack([to_sam_input(np.ascontiguousarray(img[y:y + tile, x:x + tile]))
                                              for y, x in origins]))

        def run(x):
            x = x.to(device)
            low = np.concatenate([fwd(x[j:j + 3]).float().cpu().numpy()[:, 0] for j in range(0, len(x), 3)])
            return 1.0 / (1.0 + np.exp(-blend_tiles(low, origins, native // scale, scale)))
    else:
        raise ValueError(kind)
    return cfg, model, total, trainable, inputs, run, {"amp": use_amp}


def time_case(run, inputs, n: int, device) -> tuple[np.ndarray, float, float]:
    from torch.utils.flop_counter import FlopCounterMode
    sync = torch.cuda.synchronize if device.type == "cuda" else (lambda: None)
    first = inputs(0)
    with torch.no_grad():
        try:
            with FlopCounterMode(display=False) as fc:
                run(first)
            flops = float(fc.get_total_flops())
        except Exception as e:                     # record as missing rather than lose the timings
            print(f"[benchmark] FLOP counting failed ({type(e).__name__}: {e}); FLOPs recorded as missing", flush=True)
            flops = float("nan")
        for _ in range(WARMUP):
            run(first)
        sync()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats()
        times = []
        for i in range(n):
            data = inputs(i)                     # CPU preprocessing: not timed
            sync()
            t0 = time.perf_counter()
            run(data)
            sync()
            times.append(time.perf_counter() - t0)
    peak = torch.cuda.max_memory_allocated() / 1e9 if device.type == "cuda" else float("nan")
    return np.array(times), flops, peak


def main() -> None:
    import argparse

    import pandas as pd

    from src.data import split_ids
    from src.utils import environment_info, load_config, project_path
    ap = argparse.ArgumentParser(description="Phase 6 cost benchmark (outputs discarded)")
    ap.add_argument("--n-images", type=int, default=None, help="default: all 200 test images")
    ap.add_argument("--only", nargs="*", help="run names (default: all)")
    a = ap.parse_args()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ids = split_ids(load_config("configs/base.yaml"), "test")
    n = a.n_images or len(ids)
    rows = []
    out = project_path("results/tables") / "cost.csv"
    for run_name, label, config_path, kind in MODELS:
        if a.only and run_name not in a.only:
            continue
        cfg, model, total, trainable, inputs, run, extra = make_case(kind, config_path, ids, device)
        times, flops, peak = time_case(run, inputs, n, device)
        row = {"run": run_name, "model": label, "params_total": total, "params_trainable": trainable,
               "inference_ms_mean": 1e3 * times.mean(), "inference_ms_sd": 1e3 * times.std(ddof=1),
               "inference_ms_median": 1e3 * np.median(times), "n_images_timed": n, "gflops_per_image": flops / 1e9,
               "inference_peak_gpu_gb": peak, "amp": extra["amp"]}
        trained = {"sam_lora_tiled_overlap": "sam_lora_tiled"}.get(run_name, run_name)
        summ = project_path("results/train") / trained / f"seed{SEED}" / "summary.json"
        if kind != "oracle" and summ.exists():
            s = json.loads(summ.read_text())
            row.update({"trained_run": trained, "training_time_h": s["training_time_h"], "epochs_run": s["epochs_run"],
                        "best_epoch": s["best_epoch"], "epoch_time_s_median": s["epoch_time_s"]["median"],
                        "training_peak_gpu_gb": s["gpu_peak_alloc_gb"]})
        rows.append(row)
        print(f"[benchmark] {run_name}: {row['inference_ms_mean']:.1f} +- {row['inference_ms_sd']:.1f} ms/image, "
              f"{row['gflops_per_image']:.1f} GFLOPs, peak {peak:.2f} GB", flush=True)
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()
        pd.DataFrame(rows).to_csv(out, index=False)          # saved after every model
    pd.DataFrame(rows).to_csv(out, index=False)
    info = {"split": "test", "n_images": n, "warmup_calls": WARMUP, "seed_checkpoints": SEED,
            "timed_span": "input on GPU -> probability map at the model's output grid (tiling, stitching/blending, "
                          "sigmoid included); disk I/O and CPU preprocessing excluded; outputs discarded",
            "flops": "torch.utils.flop_counter.FlopCounterMode, one image",
            "environment": {k: v for k, v in environment_info().items()
                            if k in ("time", "host", "gpu", "slurm_job_id", "git", "torch", "cuda")}}
    with open(project_path("results/checks") / "benchmark_info.json", "w") as f:
        json.dump(info, f, indent=1)
    print(f"[benchmark] wrote {out} and results/checks/benchmark_info.json")


if __name__ == "__main__":
    main()
