"""Save probability maps from a run's best checkpoint.

    python -m src.predict --config configs/unet.yaml --seed 0 --split val
    python -m src.predict --config configs/unet.yaml --seed 0 --split test --final-test   # Phase 6 only

Output: outputs/<run>/seed<seed>/<split>/probs.npz, one float16 member per image_id
(sigmoid outputs at the model's output size, cnn_resolution for the CNNs), plus
predict_info.json. No test-time augmentation, no post-processing.

The model and data settings come from the run's own saved config (outputs/<run>/seed<seed>/
config.yaml), checked against the checkpoint's config hash, so later edits to unrelated
config sections cannot change how a trained model is run. Inference uses the same fp16
autocast as training-time validation, and the probabilities are the same float16 values
that the training-time val Dice used.
"""
from __future__ import annotations

import argparse
import json
import time

import torch
import yaml

from src.checkpoint import FORMAT_VERSION
from src.utils import (config_hash, configure_cudnn, environment_info, load_config, portable_path, prob_archive_path,
                       run_dir_for)


def predict(cfg_path: str, seed: int, split: str, final_test: bool = False, smoke: bool = False) -> None:
    from torch.utils.data import DataLoader

    from src.data import build_cnn_dataset, sha256_file, split_ids, write_npz_archive
    from src.models import build_model
    from src.train import _worker_init, amp_settings

    if split == "test" and not final_test:
        raise SystemExit("The test split is predicted once, in Phase 6: pass --final-test.")
    cfg = load_config(cfg_path)
    if smoke:
        cfg["run_name"] = (cfg.get("run_name") or cfg["model"]["name"]) + "_smoke"
    run_dir = run_dir_for(cfg, seed)
    if split == "test" and prob_archive_path(run_dir, split).exists():
        raise SystemExit(f"{portable_path(prob_archive_path(run_dir, split))} exists: each run predicts the "
                         "test split once (Phase 6); it is never overwritten.")
    best = run_dir / "best.pt"
    ckpt = torch.load(best, map_location="cpu", weights_only=True)
    run_cfg = yaml.safe_load((run_dir / "config.yaml").read_text())
    if config_hash(run_cfg) != ckpt["config_hash"]:
        raise RuntimeError(f"{run_dir / 'config.yaml'} does not match the checkpoint's config hash")
    if ckpt.get("format_version") != FORMAT_VERSION or ckpt["seed"] != seed:
        raise RuntimeError(f"{best}: unexpected format or seed")

    configure_cudnn(run_cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(run_cfg)
    model.load_state_dict(ckpt["model"])
    model.to(device).eval()
    ids = split_ids(run_cfg, split)[:4] if smoke else split_ids(run_cfg, split)
    use_amp, amp_dtype = amp_settings(run_cfg, device)
    probs, t0 = {}, time.time()
    if run_cfg["model"]["name"] == "sam_lora":              # 512 (tiled) or 1024 (whole), see src/sam_train.py
        from src.sam_train import predict_sam
        probs = predict_sam(run_cfg, model, ids, device)
    else:
        ds = build_cnn_dataset(run_cfg, ids, train=False, with_masks=False)
        loader = DataLoader(ds, batch_size=int(run_cfg["train"]["batch_size"]), shuffle=False,
                            num_workers=int(run_cfg["data"].get("num_workers", 0)), worker_init_fn=_worker_init,
                            pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(0))
        with torch.no_grad():
            for batch in loader:
                with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                    logits = model(batch["image"].to(device, non_blocking=True))
                p = torch.sigmoid(logits.float()).half().cpu().numpy()[:, 0]
                for k, iid in enumerate(batch["image_id"]):
                    probs[iid] = p[k]
    path = prob_archive_path(run_dir, split)
    write_npz_archive(path, {i: probs[i] for i in ids})
    info = {"split": split, "n_images": len(ids), "checkpoint": portable_path(best), "best_epoch": int(ckpt["epoch"]),
            "best_val_dice_at_0.5": float(ckpt["best_score"]), "config_hash": ckpt["config_hash"],
            "output": portable_path(path), "output_sha256": sha256_file(path)[:16], "dtype": "float16",
            "resolution": int(next(iter(probs.values())).shape[0]), "amp": use_amp,
            "test_time_augmentation": False, "post_processing": False, "seconds": round(time.time() - t0, 1),
            **{k: v for k, v in environment_info().items() if k in ("time", "host", "gpu", "slurm_job_id", "git")}}
    with open(path.parent / "predict_info.json", "w") as f:
        json.dump(info, f, indent=1)
    print(f"[predict] {len(ids)} {split} images -> {path} ({info['resolution']}x{info['resolution']} float16, "
          f"best epoch {info['best_epoch']}, {info['seconds']}s)")


def main() -> None:
    ap = argparse.ArgumentParser(description="Save probability maps from best.pt")
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--split", choices=["val", "test"], required=True)
    ap.add_argument("--final-test", action="store_true", help="Phase 6 only: allow the test split")
    ap.add_argument("--smoke", action="store_true", help="the --smoke training run, first 4 images")
    a = ap.parse_args()
    predict(a.config, a.seed, a.split, a.final_test, a.smoke)


if __name__ == "__main__":
    main()
