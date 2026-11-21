"""Train one model with one seed.

    python -m src.train --config configs/unet.yaml --seed 0

Run folder: outputs/<run_name>/seed<seed>/  (outputs/ is a symlink into $WORK)
    last.pt          full state after the last finished epoch (resume point)
    best.pt          weights of the best epoch by validation Dice (used by predict.py)
    train_log.csv    one row per epoch
    config.yaml      resolved config (saved once; a resume with a different config is refused)
    run_info.jsonl   one line per session: host, GPU, git commit, package versions

Resume: if last.pt exists, training continues from it (model, optimizer, scaler,
epoch, best score, early-stopping counter, RNG states). If that run already
finished, the script exits immediately. So a job that hit its time limit can be
resubmitted with the same command.

One epoch (``make_epoch_functions``, shared by the four CNNs):
  train  cached CLAHE images at cnn_resolution, train-split normalization, soft targets
         (Phase 2) used as is; random flips / 90-degree rotations; shuffle order and
         augmentation depend only on (seed, epoch); BCE + soft Dice in float32 (src/losses.py);
         fp16 autocast + GradScaler.
  val    the same loss on the val soft targets (val_loss), and val_dice: sigmoid ->
         float16 (as saved by predict.py) -> bilinear to 2048 on the GPU -> >= 0.5 ->
         Dice against the ORIGINAL 2048 mask inside the FOV, per image, mean over the
         120 val images. Early stopping and best.pt use val_dice.

After training: results/train/<run>/seed<seed>/ gets train_log.csv, curves.png and
summary.json (best epoch, training time, peak GPU memory).

Other modes:
    python -m src.train --config configs/unet.yaml --seed 0 --smoke        # 8 train / 4 val images, 2 epochs
    python -m src.train --config configs/unet.yaml --seed 0 --probe 512 1024  # memory / speed probe
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import shutil
import time
from pathlib import Path
from typing import Callable

import numpy as np
import torch

from src.checkpoint import (cpu_state_dict, load_checkpoint, restore_rng_state,
                            save_best_export, save_checkpoint)
from src.utils import (config_hash, configure_cudnn, epoch_seed, load_config, project_path,
                       run_dir_for, seed_worker, set_seed, write_run_metadata)

LOG_FIELDS = ["epoch", "train_loss", "train_bce", "train_dice_loss", "val_loss", "val_bce",
              "val_dice_loss", "val_dice", "best_val_dice", "improved", "epochs_without_improvement",
              "lr", "train_time_s", "val_time_s", "epoch_time_s", "gpu_peak_alloc_gb",
              "gpu_peak_reserved_gb"]

# train_epoch_fn(epoch) -> {"train_loss": float}
# val_fn(epoch)         -> {"val_loss": float, "val_dice": float, ...}
TrainEpochFn = Callable[[int], dict]
ValFn = Callable[[int], dict]


def _read_log(path: Path) -> list[dict]:
    if not path.exists():
        return []
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def _write_log(path: Path, rows: list[dict]) -> None:
    tmp = path.with_name(path.name + ".tmp")
    with open(tmp, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS, extrasaction="ignore")
        w.writeheader()
        w.writerows(rows)
    tmp.replace(path)


def fit(*, model: torch.nn.Module, optimizer: torch.optim.Optimizer,
        scaler: torch.amp.GradScaler | None, run_dir: Path, seed: int, cfg_hash: str,
        max_epochs: int, patience: int, train_epoch_fn: TrainEpochFn, val_fn: ValFn,
        monitor: str = "val_dice", device: str | torch.device = "cpu") -> dict:
    """Epoch loop with early stopping, checkpointing and exact resume.

    Order at the end of each epoch: (1) last.pt is written atomically, containing
    the new state *and* the best weights so far; (2) best.pt is exported from it;
    (3) the log row is written. If the job dies anywhere in between, last.pt is
    the single source of truth: on resume, best.pt is re-exported if stale and log
    rows from unfinished epochs are dropped.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    last_path, best_path, log_path = run_dir / "last.pt", run_dir / "best.pt", run_dir / "train_log.csv"

    start_epoch, best_score, best_epoch, bad_epochs, best_state = 0, None, None, 0, None
    if last_path.exists():
        ckpt = load_checkpoint(last_path, model=model, optimizer=optimizer, scaler=scaler,
                               map_location=device, restore_rng=False)
        if ckpt["config_hash"] != cfg_hash:
            raise RuntimeError(f"{last_path} was made with a different config "
                               f"({ckpt['config_hash']} != {cfg_hash}). Refusing to resume; "
                               "use a new run_name or delete the run folder.")
        if ckpt["seed"] != seed:
            raise RuntimeError(f"{last_path} has seed {ckpt['seed']}, not {seed}")
        start_epoch, best_score, best_epoch = ckpt["epoch"], ckpt["best_score"], ckpt["best_epoch"]
        bad_epochs = ckpt["epochs_without_improvement"]
        best_state = ({k: v.cpu() for k, v in ckpt["best_model"].items()}
                      if ckpt["best_model"] is not None else None)
        # Consistency repairs after a kill between steps (1)-(3).
        if best_state is not None:
            stale = (not best_path.exists() or
                     torch.load(best_path, map_location="cpu", weights_only=True)["epoch"] != best_epoch)
            if stale:
                save_best_export(best_path, best_model_state=best_state, best_epoch=best_epoch,
                                 best_score=best_score, seed=seed, config_hash=cfg_hash)
        _write_log(log_path, [r for r in _read_log(log_path) if int(r["epoch"]) < start_epoch])
        if ckpt["finished"]:
            print(f"[train] {run_dir} already finished ({ckpt['finish_reason']}); "
                  f"best epoch {best_epoch}, {monitor}={best_score}. Nothing to do.")
            return {"finished": True, "best_epoch": best_epoch, "best_score": best_score,
                    "resumed_from": start_epoch, "finish_reason": ckpt["finish_reason"]}
        restore_rng_state(ckpt["rng"])
        print(f"[train] resuming {run_dir} at epoch {start_epoch} "
              f"(best {monitor}={best_score} at epoch {best_epoch}, {bad_epochs} epochs without improvement)")
    else:
        print(f"[train] starting {run_dir} from scratch")

    rows = _read_log(log_path)
    finish_reason = None
    for epoch in range(start_epoch, max_epochs):
        t0 = time.time()
        train_stats = train_epoch_fn(epoch)
        val_stats = val_fn(epoch)
        score = float(val_stats[monitor])
        improved = (not math.isnan(score)) and (best_score is None or score > best_score)
        if improved:
            best_score, best_epoch, bad_epochs = score, epoch, 0
            best_state = cpu_state_dict(model)
        else:
            bad_epochs += 1
        if bad_epochs >= patience:
            finish_reason = f"early stopping: no improvement in {patience} epochs"
        elif epoch + 1 >= max_epochs:
            finish_reason = f"reached max_epochs={max_epochs}"

        save_checkpoint(last_path, model=model, optimizer=optimizer, scaler=scaler,
                        epoch=epoch + 1, best_score=best_score, best_epoch=best_epoch,
                        epochs_without_improvement=bad_epochs, best_model_state=best_state,
                        finished=finish_reason is not None, finish_reason=finish_reason,
                        seed=seed, config_hash=cfg_hash)
        if improved:
            save_best_export(best_path, best_model_state=best_state, best_epoch=best_epoch,
                             best_score=best_score, seed=seed, config_hash=cfg_hash)
        rows.append({"epoch": epoch, **train_stats, **val_stats, "best_val_dice": best_score,
                     "improved": int(improved), "epochs_without_improvement": bad_epochs,
                     "lr": optimizer.param_groups[0]["lr"], "epoch_time_s": round(time.time() - t0, 2)})
        _write_log(log_path, rows)
        print(f"[train] epoch {epoch}: " + ", ".join(f"{k}={v:.4f}" if isinstance(v, float) else f"{k}={v}"
                                                      for k, v in {**train_stats, **val_stats}.items())
              + (" *" if improved else ""))
        if finish_reason is not None:
            print(f"[train] finished: {finish_reason}; best epoch {best_epoch}, {monitor}={best_score}")
            break
    return {"finished": finish_reason is not None, "best_epoch": best_epoch,
            "best_score": best_score, "resumed_from": start_epoch, "finish_reason": finish_reason}


def _worker_init(worker_id: int) -> None:
    """Seed NumPy / random from the loader's generator (utils.seed_worker) and keep OpenCV
    single-threaded in each worker."""
    import cv2
    seed_worker(worker_id)
    cv2.setNumThreads(1)


def amp_settings(cfg: dict, device: torch.device) -> tuple[bool, torch.dtype]:
    a = cfg["train"].get("amp", {})
    return bool(a.get("enabled")) and device.type == "cuda", getattr(torch, a.get("dtype", "float16"))


def load_val_targets(cfg: dict, ids: list[str]) -> dict:
    """Original 2048 val masks restricted to the FOV, and the FOVs (bool, CPU)."""
    from concurrent.futures import ThreadPoolExecutor

    from src.data import raw_mask_path, read_unchanged
    from src.evaluate import load_fov

    def one(i):
        gt, fov = read_unchanged(raw_mask_path(cfg, i)) > 0, load_fov(cfg, i)
        return i, (torch.from_numpy(gt & fov), torch.from_numpy(fov))

    with ThreadPoolExecutor(8) as ex:
        return dict(ex.map(one, ids))


def val_dice_batch(prob: torch.Tensor, image_ids, targets: dict, native: int, threshold: float) -> list[float]:
    """Per-image Dice at 2048 inside the FOV for a batch of probabilities (N, 1, h, w),
    exactly as final scoring does it (float16 values, bilinear, >= threshold)."""
    from src.metrics import overlap_counts
    from src.resample import upsample_prob_torch
    pred = upsample_prob_torch(prob.half().float(), native) >= threshold
    out = []
    for k, iid in enumerate(image_ids):
        gt, fov = (a.to(prob.device, non_blocking=True) for a in targets[iid])
        pk = pred[k, 0] & fov
        out.append(overlap_counts(int((pk & gt).sum()), int(pk.sum()), int(gt.sum()))["dice"])
    return out


def make_epoch_functions(cfg: dict, seed: int, model, optimizer, scaler, device, *,
                         train_ids: list[str] | None = None, val_ids: list[str] | None = None):
    """Build the data loaders and return (train_epoch_fn, val_fn); see the module docstring."""
    from torch.utils.data import DataLoader

    from src.data import build_cnn_dataset, split_ids
    from src.losses import BCEDiceLoss

    if cfg["model"]["name"] == "sam_lora":
        from src.sam_train import make_sam_epoch_functions
        return make_sam_epoch_functions(cfg, seed, model, optimizer, scaler, device,
                                        train_ids=train_ids, val_ids=val_ids)
    tc = cfg["train"]
    train_ids = train_ids or split_ids(cfg, "train")
    val_ids = val_ids or split_ids(cfg, "val")
    train_ds = build_cnn_dataset(cfg, train_ids, train=True, with_masks=True)
    val_ds = build_cnn_dataset(cfg, val_ids, train=False, with_masks=True)
    criterion = BCEDiceLoss.from_config(cfg)
    bs, accum = int(tc["batch_size"]), int(tc.get("grad_accumulation") or 1)
    workers = int(cfg["data"].get("num_workers", 0))
    use_amp, amp_dtype = amp_settings(cfg, device)
    pin = device.type == "cuda"
    native = int(cfg["data"]["native_resolution"])
    threshold = float(tc["early_stopping"]["val_threshold"])
    print(f"[train] {len(train_ids)} train / {len(val_ids)} val images at {cfg['data']['cnn_resolution']}, "
          f"batch {bs} x accumulation {accum}, {workers} workers, amp {use_amp} ({amp_dtype})", flush=True)
    targets = load_val_targets(cfg, val_ids)
    val_loader = DataLoader(val_ds, batch_size=bs, shuffle=False, num_workers=workers, worker_init_fn=_worker_init,
                            pin_memory=pin, generator=torch.Generator().manual_seed(0))

    def train_epoch_fn(epoch: int) -> dict:
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        g = torch.Generator()
        g.manual_seed(epoch_seed(seed, epoch))            # shuffle + worker seeds: function of (seed, epoch)
        loader = DataLoader(train_ds, batch_size=bs, shuffle=True, generator=g, num_workers=workers,
                            worker_init_fn=_worker_init, pin_memory=pin, drop_last=False)
        t0, n, sums = time.time(), 0, {"loss": 0.0, "bce": 0.0, "dice_loss": 0.0}
        optimizer.zero_grad(set_to_none=True)
        for step, batch in enumerate(loader):
            x = batch["image"].to(device, non_blocking=True)
            t = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
            loss, parts = criterion(logits, t)              # float32, soft targets as is
            scaler.scale(loss / accum).backward()
            if (step + 1) % accum == 0 or step + 1 == len(loader):
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            k = x.shape[0]
            n += k
            sums["loss"] += loss.item() * k
            sums["bce"] += parts["bce"] * k
            sums["dice_loss"] += parts["dice_loss"] * k
        if not math.isfinite(sums["loss"]):
            raise FloatingPointError(f"non-finite training loss in epoch {epoch}")
        return {"train_loss": sums["loss"] / n, "train_bce": sums["bce"] / n,
                "train_dice_loss": sums["dice_loss"] / n, "train_time_s": round(time.time() - t0, 2)}

    def val_fn(epoch: int) -> dict:
        model.eval()
        t0, n, sums, dices = time.time(), 0, {"loss": 0.0, "bce": 0.0, "dice_loss": 0.0}, []
        with torch.no_grad():
            for batch in val_loader:
                x = batch["image"].to(device, non_blocking=True)
                t = batch["mask"].to(device, non_blocking=True)
                with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                    logits = model(x)
                loss, parts = criterion(logits, t)
                k = x.shape[0]
                n += k
                sums["loss"] += loss.item() * k
                sums["bce"] += parts["bce"] * k
                sums["dice_loss"] += parts["dice_loss"] * k
                dices += val_dice_batch(torch.sigmoid(logits.float()), batch["image_id"], targets, native, threshold)
        out = {"val_loss": sums["loss"] / n, "val_bce": sums["bce"] / n, "val_dice_loss": sums["dice_loss"] / n,
               "val_dice": float(np.mean(dices)), "val_time_s": round(time.time() - t0, 2)}
        if device.type == "cuda":
            out["gpu_peak_alloc_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
            out["gpu_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 2)
        return out

    return train_epoch_fn, val_fn


# --------------------------------------------------------------------------- after training
def results_dir_for(cfg: dict, seed: int) -> Path:
    name = cfg.get("run_name") or cfg["model"]["name"]
    return project_path(cfg["paths"]["results_dir"]) / "train" / name / f"seed{seed}"


def write_training_summary(cfg: dict, seed: int, run_dir: Path, model: torch.nn.Module, fit_result: dict) -> None:
    """Small, git-tracked copies: train_log.csv, curves.png, summary.json."""
    import pandas as pd

    from src.analysis import plot_training_curves
    out = results_dir_for(cfg, seed)
    out.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(run_dir / "train_log.csv", out / "train_log.csv")
    log = pd.read_csv(out / "train_log.csv")
    best = log.loc[log.val_dice.idxmax()]
    info = [json.loads(l) for l in (run_dir / "run_info.jsonl").read_text().splitlines() if l.strip()]
    summary = {
        "run": cfg.get("run_name") or cfg["model"]["name"], "seed": seed,
        "finish_reason": fit_result.get("finish_reason"), "epochs_run": int(len(log)),
        "best_epoch": int(best.epoch), "best_val_dice_at_0.5": float(best.val_dice),
        "training_time_h": round(float(log.epoch_time_s.sum()) / 3600, 3),
        "training_time_definition": "sum of epoch times (train + val), excluding start-up and resume overhead",
        "epoch_time_s": {"mean": float(log.epoch_time_s.mean()), "median": float(log.epoch_time_s.median()),
                         "train_mean": float(log.train_time_s.mean()), "val_mean": float(log.val_time_s.mean())},
        "gpu_peak_alloc_gb": float(log.gpu_peak_alloc_gb.max()) if "gpu_peak_alloc_gb" in log and log.gpu_peak_alloc_gb.notna().any() else None,
        "gpu_peak_reserved_gb": float(log.gpu_peak_reserved_gb.max()) if "gpu_peak_reserved_gb" in log and log.gpu_peak_reserved_gb.notna().any() else None,
        "parameters": {"total": int(sum(p.numel() for p in model.parameters())),
                       "trainable": int(sum(p.numel() for p in model.parameters() if p.requires_grad))},
        "cnn_resolution": int(cfg["data"]["cnn_resolution"]), "batch_size": int(cfg["train"]["batch_size"]),
        "sessions": [{k: r.get(k) for k in ("time", "host", "gpu", "slurm_job_id", "slurm_restart_count", "torch", "cuda")}
                     for r in info],
        "config_hash": config_hash(cfg),
    }
    with open(out / "summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    plot_training_curves(out / "train_log.csv", out / "curves.png",
                         f"{summary['run']}, seed {seed}: best epoch {summary['best_epoch']} "
                         f"(val Dice {summary['best_val_dice_at_0.5']:.4f})")
    print(f"[train] wrote {out}/train_log.csv, curves.png, summary.json")


# --------------------------------------------------------------------------- probe
def _probe_once(cfg: dict, seed: int, ids: list[str], res: int, bs: int, steps: int, device: torch.device) -> dict:
    from torch.utils.data import DataLoader

    from src.data import build_cnn_dataset
    from src.losses import BCEDiceLoss
    from src.models import build_model, build_optimizer
    from src.resample import upsample_prob_torch

    use_amp, amp_dtype = amp_settings(cfg, device)
    rec = {"model": cfg.get("run_name") or cfg["model"]["name"], "resolution": res, "batch_size": bs,
           "steps": steps, "gpu": torch.cuda.get_device_name(device), "amp": use_amp}
    try:
        set_seed(seed)
        model = build_model(cfg).to(device)
        opt = build_optimizer(cfg, model)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        crit = BCEDiceLoss.from_config(cfg)
        ds = build_cnn_dataset(cfg, ids, train=True, with_masks=True, resolution=res)
        loader = DataLoader(ds, batch_size=bs, shuffle=True, generator=torch.Generator().manual_seed(seed),
                            num_workers=int(cfg["data"].get("num_workers", 0)), worker_init_fn=_worker_init,
                            pin_memory=True, drop_last=True)
        torch.cuda.reset_peak_memory_stats(device)
        times, t_last = [], None
        model.train()
        it = iter(loader)
        for step in range(steps):
            b = next(it)
            x, t = b["image"].to(device, non_blocking=True), b["mask"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                logits = model(x)
            loss, _ = crit(logits, t)
            scaler.scale(loss).backward()
            scaler.step(opt)
            scaler.update()
            opt.zero_grad(set_to_none=True)
            torch.cuda.synchronize(device)
            now = time.time()
            if t_last is not None and step >= 5:
                times.append(now - t_last)
            t_last = now
        train_peak = torch.cuda.max_memory_allocated(device) / 2**30
        model.eval()
        with torch.no_grad(), torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
            p = torch.sigmoid(model(x).float())
        upsample_prob_torch(p.half().float(), int(cfg["data"]["native_resolution"]))
        torch.cuda.synchronize(device)
        rec.update(status="ok", step_time_s_median=float(np.median(times)), step_time_s_mean=float(np.mean(times)),
                   steps_timed=len(times), train_peak_alloc_gb=round(train_peak, 2),
                   peak_alloc_gb=round(torch.cuda.max_memory_allocated(device) / 2**30, 2),
                   peak_reserved_gb=round(torch.cuda.max_memory_reserved(device) / 2**30, 2),
                   batches_per_epoch=len(ds) // bs + (len(ds) % bs > 0),
                   note="step time = wall time between consecutive optimizer steps (includes data loading), "
                        "after 5 warm-up steps; peak_alloc includes one validation forward + upsampling to 2048")
    except torch.cuda.OutOfMemoryError as e:
        rec.update(status="out_of_memory", error=str(e).split("\n")[0])
    return rec


def run_probe(cfg: dict, seed: int, resolutions: list[int], batch: int, steps: int, device: torch.device) -> None:
    """Peak GPU memory and time per training step at given input resolutions (no
    checkpoint, nothing kept): answers whether e.g. a 1024 pilot is affordable. On
    out-of-memory the batch is halved and tried again."""
    import gc

    from src.data import CHECKS_DIR, split_ids
    ids = split_ids(cfg, "train")
    name = cfg.get("run_name") or cfg["model"]["name"]
    for res in resolutions:
        bs = batch
        while bs >= 1:
            rec = _probe_once(cfg, seed, ids, res, bs, steps, device)
            gc.collect()
            torch.cuda.empty_cache()
            print(f"[probe] {json.dumps(rec)}", flush=True)
            with open(CHECKS_DIR / f"probe_{name}_r{res}_b{bs}.json", "w") as f:
                json.dump(rec, f, indent=1)
            if rec["status"] == "ok":
                break
            bs //= 2


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--smoke", action="store_true", help="8 train / 4 val images, 2 epochs, run name + '_smoke'")
    ap.add_argument("--probe", type=int, nargs="+", metavar="RES", help="memory / speed probe at these resolutions")
    ap.add_argument("--probe-batch", type=int, default=None)
    ap.add_argument("--probe-steps", type=int, default=30)
    args = ap.parse_args()

    cfg = load_config(args.config)
    train_ids = val_ids = None
    if args.smoke:
        from src.data import split_ids
        cfg["run_name"] = (cfg.get("run_name") or cfg["model"]["name"]) + "_smoke"
        cfg["train"]["max_epochs"] = 2
        cfg["data"]["num_workers"] = min(2, int(cfg["data"].get("num_workers", 0)))
        train_ids, val_ids = split_ids(cfg, "train")[:8], split_ids(cfg, "val")[:4]
    train_cfg = cfg["train"]
    set_seed(args.seed)
    configure_cudnn(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    if args.probe:
        if device.type != "cuda":
            raise SystemExit("the probe needs a GPU")
        run_probe(cfg, args.seed, args.probe, args.probe_batch or int(train_cfg["batch_size"]), args.probe_steps, device)
        return

    cfg_h = config_hash(cfg)
    run_dir = run_dir_for(cfg, args.seed)

    from src.models import build_model, build_optimizer
    model = build_model(cfg).to(device)
    optimizer = build_optimizer(cfg, model)
    use_amp, _ = amp_settings(cfg, device)
    use_scaler = use_amp and train_cfg.get("amp", {}).get("dtype") == "float16"
    scaler = torch.amp.GradScaler("cuda", enabled=use_scaler)

    write_run_metadata(run_dir, cfg, vars(args), {"config_hash": cfg_h,
                                                 "resume": (run_dir / "last.pt").exists()})
    train_epoch_fn, val_fn = make_epoch_functions(cfg, args.seed, model, optimizer, scaler, device,
                                                  train_ids=train_ids, val_ids=val_ids)
    res = fit(model=model, optimizer=optimizer, scaler=scaler, run_dir=run_dir, seed=args.seed,
              cfg_hash=cfg_h, max_epochs=int(train_cfg["max_epochs"]),
              patience=int(train_cfg["early_stopping"]["patience"]), train_epoch_fn=train_epoch_fn,
              val_fn=val_fn, monitor=train_cfg["early_stopping"]["metric"], device=device)
    write_training_summary(cfg, args.seed, run_dir, model, res)


if __name__ == "__main__":
    main()
