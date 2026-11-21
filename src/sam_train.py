"""Fine-tuned SAM (model 5b): data, one epoch of training / validation, prediction, the
Phase 5 gradient check and a memory / speed probe. Training itself runs through
src/train.py (same fit loop, checkpointing, early stopping and resume as the CNNs).

Two variants (``sam.variant`` in the config), both with no prompts:
  tiled  MAIN (user decision). Each 2048 image -> 2x2 native-resolution 1024x1024 tiles;
         one sample = one tile, so an epoch covers all image content once (as for the CNNs).
         Loss: the decoder's 256x256 logits upsampled x4 to 1024 (bilinear, as SAM's
         postprocessing) against the native binary mask tile (nothing resized).
         Output: the four 256x256 logit grids stitched into one 512x512 grid -> sigmoid ->
         probability map at 512: the same grid and evaluation path as the CNNs.
  whole  secondary. The CLAHE image at 1024; one sample = one image. Loss: 256 logits
         upsampled x4 to 1024 against the cached soft 1024 mask (Phase 2 method).
         Output: logits upsampled to 1024 (SAM's postprocessing) -> sigmoid -> probability map
         at 1024 (decoder grid 256: see the mask-resampling reference).
The x4 upsampling uses blocks.upsample_bilinear_int (identical to F.interpolate, but with
a deterministic CUDA backward).

    python -m src.sam_train gradcheck --config configs/sam_lora_tiled.yaml --seed 0
    python -m src.sam_train probe     --config configs/sam_lora_tiled.yaml --micro 1 2 4
"""
from __future__ import annotations

import argparse
import json
import time

import numpy as np
import torch

from src.models.blocks import upsample_bilinear_int


# --------------------------------------------------------------------------- variant helpers
def sam_variant(cfg: dict) -> str:
    v = (cfg.get("sam") or {}).get("variant")
    if v not in ("tiled", "whole"):
        raise ValueError(f"sam.variant must be 'tiled' or 'whole', got {v!r}")
    return v


def tiles_per_image(cfg: dict) -> int:
    n = int(cfg["data"]["native_resolution"]) // int(cfg["data"]["sam_resolution"])
    return n * n if sam_variant(cfg) == "tiled" else 1


def sam_dataset(cfg: dict, ids, *, train: bool, with_masks: bool):
    from src.data import FIVESDataset, SamTileDataset, cache_image_dir, mask_cache_path
    if sam_variant(cfg) == "tiled":
        return SamTileDataset(cfg, ids, train=train, with_masks=with_masks)
    res = int(cfg["data"]["sam_resolution"])
    return FIVESDataset(ids, image_dir=cache_image_dir(cfg, res),
                        mask_source=mask_cache_path(cfg, res) if with_masks else None,
                        normalization="sam", train=train,
                        augmentation=cfg["data"]["augmentation"] if train else None)


def logits_at_input(low: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Decoder logits (N, 1, 256, 256) -> (N, 1, 1024, 1024), float32 (SAM postprocessing)."""
    f = int(cfg["data"]["sam_resolution"]) // int(cfg["data"]["sam_decoder_resolution"])
    return upsample_bilinear_int(low.float(), f)


def prob_maps(low: torch.Tensor, cfg: dict) -> torch.Tensor:
    """Decoder logits of whole images (whole) or of consecutive complete tile sets (tiled) ->
    the probability maps that are saved and scored: (images, 1, R, R), R = 512 (tiled) or 1024."""
    if sam_variant(cfg) == "whole":
        return torch.sigmoid(logits_at_input(low, cfg))
    from src.data import stitch_tiles
    k = tiles_per_image(cfg)
    n = int(round(k ** 0.5))
    if low.shape[0] % k:
        raise ValueError("tiled output needs complete tile sets")
    return torch.sigmoid(torch.stack([stitch_tiles(low[j:j + k].float(), n) for j in range(0, low.shape[0], k)]))


def check_tile_batch(batch: dict, k: int) -> list[str]:
    """Image ids of a batch that must hold complete tile sets (tile order 0..k-1 per image);
    raises otherwise, so a stitched map can never mix tiles of different images."""
    ids = list(batch["image_id"])
    if k > 1:
        tiles = [int(t) for t in batch["tile"]]
        if len(ids) % k or any(len(set(ids[j:j + k])) != 1 or tiles[j:j + k] != list(range(k))
                               for j in range(0, len(ids), k)):
            raise RuntimeError(f"batch is not complete tile sets: {list(zip(ids, tiles))}")
    return ids[::k]


def _group_weights(n_samples: int, micro: int, accum: int) -> list[tuple[float, bool]]:
    """(loss weight, step after this micro-batch?) for each micro-batch of an epoch, so that
    every optimizer step averages over the samples of its group, also for a short last group."""
    sizes = [micro] * (n_samples // micro) + ([n_samples % micro] if n_samples % micro else [])
    out = []
    for g in range(0, len(sizes), accum):
        grp = sizes[g:g + accum]
        out += [(s / sum(grp), j == len(grp) - 1) for j, s in enumerate(grp)]
    return out


# --------------------------------------------------------------------------- epochs
def make_sam_epoch_functions(cfg: dict, seed: int, model, optimizer, scaler, device, *,
                             train_ids=None, val_ids=None):
    from torch.utils.data import DataLoader

    from src.data import split_ids
    from src.losses import BCEDiceLoss
    from src.train import _worker_init, amp_settings, load_val_targets, val_dice_batch
    from src.utils import epoch_seed

    tc = cfg["train"]
    micro, accum = int(tc["batch_size"]), int(tc.get("grad_accumulation") or 1)
    if micro * accum != 4:
        raise ValueError(f"effective batch must be 4 (micro {micro} x accumulation {accum})")
    train_ids = train_ids or split_ids(cfg, "train")
    val_ids = val_ids or split_ids(cfg, "val")
    train_ds = sam_dataset(cfg, train_ids, train=True, with_masks=True)
    val_ds = sam_dataset(cfg, val_ids, train=False, with_masks=True)
    k = tiles_per_image(cfg)
    val_bs = k if k > 1 else micro                     # tiled: one image (its k tiles) per batch
    criterion = BCEDiceLoss.from_config(cfg)
    workers = int(cfg["data"].get("num_workers", 0))
    use_amp, amp_dtype = amp_settings(cfg, device)
    pin = device.type == "cuda"
    native = int(cfg["data"]["native_resolution"])
    threshold = float(tc["early_stopping"]["val_threshold"])
    weights = _group_weights(len(train_ds), micro, accum)
    print(f"[sam] variant {sam_variant(cfg)}: {len(train_ids)} train images -> {len(train_ds)} samples, "
          f"{len(val_ids)} val images -> {len(val_ds)} samples; micro-batch {micro} x accumulation {accum} "
          f"= effective 4; {sum(1 for _, s in weights if s)} optimizer steps per epoch; amp {use_amp}", flush=True)
    targets = load_val_targets(cfg, val_ids)
    val_loader = DataLoader(val_ds, batch_size=val_bs, shuffle=False, num_workers=workers, worker_init_fn=_worker_init,
                            pin_memory=pin, generator=torch.Generator().manual_seed(0))

    def train_epoch_fn(epoch: int) -> dict:
        model.train()
        if device.type == "cuda":
            torch.cuda.reset_peak_memory_stats(device)
        g = torch.Generator()
        g.manual_seed(epoch_seed(seed, epoch))
        loader = DataLoader(train_ds, batch_size=micro, shuffle=True, generator=g, num_workers=workers,
                            worker_init_fn=_worker_init, pin_memory=pin, drop_last=False)
        t0, n, sums = time.time(), 0, {"loss": 0.0, "bce": 0.0, "dice_loss": 0.0}
        optimizer.zero_grad(set_to_none=True)
        for j, batch in enumerate(loader):
            x = batch["image"].to(device, non_blocking=True)
            t = batch["mask"].to(device, non_blocking=True)
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                low = model(x)
            loss, parts = criterion(logits_at_input(low, cfg), t)      # float32
            w, step = weights[j]
            scaler.scale(loss * w).backward()
            if step:
                scaler.step(optimizer)
                scaler.update()
                optimizer.zero_grad(set_to_none=True)
            b = x.shape[0]
            n += b
            sums["loss"] += loss.item() * b
            sums["bce"] += parts["bce"] * b
            sums["dice_loss"] += parts["dice_loss"] * b
        if not np.isfinite(sums["loss"]):
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
                    low = model(x)
                loss, parts = criterion(logits_at_input(low, cfg), t)
                b = x.shape[0]
                n += b
                sums["loss"] += loss.item() * b
                sums["bce"] += parts["bce"] * b
                sums["dice_loss"] += parts["dice_loss"] * b
                ids = check_tile_batch(batch, k)            # one image per batch (tiled): 4 tiles, order 0-3
                dices += val_dice_batch(prob_maps(low, cfg), ids, targets, native, threshold)
        out = {"val_loss": sums["loss"] / n, "val_bce": sums["bce"] / n, "val_dice_loss": sums["dice_loss"] / n,
               "val_dice": float(np.mean(dices)), "val_time_s": round(time.time() - t0, 2)}
        if device.type == "cuda":
            out["gpu_peak_alloc_gb"] = round(torch.cuda.max_memory_allocated(device) / 2**30, 2)
            out["gpu_peak_reserved_gb"] = round(torch.cuda.max_memory_reserved(device) / 2**30, 2)
        return out

    return train_epoch_fn, val_fn


def predict_sam(cfg: dict, model, ids, device) -> dict:
    """Probability maps (float16) of fine-tuned SAM: 512 (tiled) or 1024 (whole), per image."""
    from torch.utils.data import DataLoader

    from src.train import _worker_init, amp_settings
    k = tiles_per_image(cfg)
    ds = sam_dataset(cfg, ids, train=False, with_masks=False)
    loader = DataLoader(ds, batch_size=k if k > 1 else int(cfg["train"]["batch_size"]), shuffle=False,
                        num_workers=int(cfg["data"].get("num_workers", 0)), worker_init_fn=_worker_init,
                        pin_memory=device.type == "cuda", generator=torch.Generator().manual_seed(0))
    use_amp, amp_dtype = amp_settings(cfg, device)
    out = {}
    model.eval()
    with torch.no_grad():
        for batch in loader:
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                low = model(batch["image"].to(device, non_blocking=True))
            p = prob_maps(low, cfg).half().cpu().numpy()[:, 0]
            for j, iid in enumerate(check_tile_batch(batch, k)):
                out[iid] = p[j]
    return out


# --------------------------------------------------------------------------- gradient check
def _group(name: str) -> str:
    if any(f".{m}." in name for m in ("a_q", "a_v")):
        return "lora_A"
    if any(f".{m}." in name for m in ("b_q", "b_v")):
        return "lora_B"
    if name.startswith("sam.mask_decoder."):
        return "mask_decoder"
    if name.startswith("sam.prompt_encoder."):
        return "prompt_encoder (frozen)"
    return "image_encoder (frozen)"


def gradcheck(cfg: dict, seed: int, steps: int = 3) -> dict:
    """A few optimizer steps on ONE fixed batch (effective size 4, the first training samples,
    no augmentation), with the training setup (AMP, GradScaler, Adam). Records per step which
    parameters have no / zero / non-zero gradients, and at the end which ones changed."""
    from src.data import CHECKS_DIR, split_ids
    from src.losses import BCEDiceLoss
    from src.models import build_model, build_optimizer
    from src.train import amp_settings
    from src.utils import configure_cudnn, set_seed

    set_seed(seed)
    configure_cudnn(cfg)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = build_model(cfg).to(device)
    opt = build_optimizer(cfg, model)
    use_amp, amp_dtype = amp_settings(cfg, device)
    scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
    crit = BCEDiceLoss.from_config(cfg)
    micro = int(cfg["train"]["batch_size"])
    ds = sam_dataset(cfg, split_ids(cfg, "train"), train=False, with_masks=True)
    samples = [ds[i] for i in range(4)]
    x = torch.stack([s["image"] for s in samples]).to(device)
    t = torch.stack([s["mask"] for s in samples]).to(device)
    names = dict(model.named_parameters())
    groups = {n: _group(n) for n in names}
    before = {n: p.detach().clone() for n, p in names.items()}
    steps_log, applied, it = [], 0, 0
    while applied < steps and it < 10:
        it += 1
        opt.zero_grad(set_to_none=True)
        for j in range(0, 4, micro):
            with torch.autocast(device.type, dtype=amp_dtype, enabled=use_amp):
                low = model(x[j:j + micro])
            loss = crit(logits_at_input(low, cfg), t[j:j + micro])[0]
            scaler.scale(loss * (low.shape[0] / 4)).backward()
        scaler.unscale_(opt)
        state = {}
        for n, p in names.items():
            g = p.grad
            s = "none" if g is None else ("nonfinite" if not torch.isfinite(g).all() else
                                         ("zero" if not g.any() else "nonzero"))
            state[n] = (s, 0.0 if g is None or s == "nonfinite" else float(g.norm()))
        scale_before = scaler.get_scale() if use_amp else 1.0
        scaler.step(opt)
        scaler.update()
        skipped = use_amp and scaler.get_scale() < scale_before
        applied += not skipped
        summary = {}
        for grp in sorted(set(groups.values())):
            ns = [n for n in names if groups[n] == grp]
            cnt = {k: sum(state[n][0] == k for n in ns) for k in ("none", "zero", "nonzero", "nonfinite")}
            summary[grp] = {"tensors": len(ns), **cnt,
                            "grad_norm": float(np.sqrt(sum(state[n][1] ** 2 for n in ns)))}
        steps_log.append({"iteration": it, "loss": float(loss), "optimizer_step_skipped_by_gradscaler": bool(skipped),
                          "groups": summary,
                          "mask_decoder_no_grad": [n for n in names if groups[n] == "mask_decoder" and state[n][0] == "none"],
                          "mask_decoder_zero_grad": [n for n in names if groups[n] == "mask_decoder" and state[n][0] == "zero"]})
    changed = {n: float((p.detach() - before[n]).abs().max()) for n, p in names.items()}
    res = {"run": cfg.get("run_name"), "variant": sam_variant(cfg), "seed": seed, "device": str(device),
           "gpu": torch.cuda.get_device_name(device) if device.type == "cuda" else None, "amp": use_amp,
           "batch": "first 4 training samples, no augmentation, the same batch at every step",
           "lr": float(cfg["train"]["optimizer"]["lr"]), "optimizer_steps_applied": applied, "iterations": it,
           "parameters": {grp: {"tensors": sum(groups[n] == grp for n in names),
                                "elements": int(sum(names[n].numel() for n in names if groups[n] == grp)),
                                "requires_grad": sorted({names[n].requires_grad for n in names if groups[n] == grp})}
                          for grp in sorted(set(groups.values()))},
           "steps": steps_log,
           "after_all_steps": {grp: {"tensors_changed": sum(changed[n] > 0 for n in names if groups[n] == grp),
                                     "tensors_unchanged": sum(changed[n] == 0 for n in names if groups[n] == grp),
                                     "max_abs_change": max(changed[n] for n in names if groups[n] == grp)}
                               for grp in sorted(set(groups.values()))},
           "mask_decoder_unchanged": [n for n in names if groups[n] == "mask_decoder" and changed[n] == 0],
           "frozen_bit_identical": all(torch.equal(names[n].detach(), before[n]) for n in names
                                       if groups[n].endswith("(frozen)"))}
    path = CHECKS_DIR / f"sam_gradcheck_{cfg.get('run_name')}.json"
    with open(path, "w") as f:
        json.dump(res, f, indent=1)
    print(json.dumps({k: v for k, v in res.items() if k != "steps"}, indent=1))
    for s in steps_log:
        print(f"[gradcheck] iteration {s['iteration']}: loss {s['loss']:.4f}, skipped {s['optimizer_step_skipped_by_gradscaler']}; "
              + "; ".join(f"{g}: {v['nonzero']} nonzero / {v['zero']} zero / {v['none']} none / {v['nonfinite']} nonfinite "
                          f"(norm {v['grad_norm']:.3g})" for g, v in s["groups"].items()))
        print(f"            decoder without grad: {s['mask_decoder_no_grad']}\n            decoder zero grad: {s['mask_decoder_zero_grad']}")
    print(f"[gradcheck] wrote {path}")
    return res


# --------------------------------------------------------------------------- probe
def probe(cfg: dict, seed: int, micros=(1, 2, 4), steps: int = 12) -> None:
    """Peak GPU memory and time per training sample for each micro-batch size, time per
    validation sample, and the resulting time-per-epoch estimate. Nothing is kept."""
    import gc

    from torch.utils.data import DataLoader

    from src.data import CHECKS_DIR, split_ids
    from src.losses import BCEDiceLoss
    from src.models import build_model, build_optimizer
    from src.train import _worker_init, amp_settings
    from src.utils import set_seed

    device = torch.device("cuda")
    use_amp, amp_dtype = amp_settings(cfg, device)
    crit = BCEDiceLoss.from_config(cfg)
    train_ds = sam_dataset(cfg, split_ids(cfg, "train"), train=True, with_masks=True)
    val_ds = sam_dataset(cfg, split_ids(cfg, "val"), train=False, with_masks=True)
    k = tiles_per_image(cfg)
    rows = []

    def once(micro):
        set_seed(seed)
        model = build_model(cfg).to(device)
        opt = build_optimizer(cfg, model)
        scaler = torch.amp.GradScaler("cuda", enabled=use_amp)
        loader = DataLoader(train_ds, batch_size=micro, shuffle=True, generator=torch.Generator().manual_seed(seed),
                            num_workers=int(cfg["data"].get("num_workers", 0)), worker_init_fn=_worker_init,
                            pin_memory=True, drop_last=True)
        rec = {"micro_batch": micro}
        try:
            torch.cuda.empty_cache()
            torch.cuda.reset_peak_memory_stats(device)
            it, times, last = iter(loader), [], None
            model.train()
            for s in range(steps):
                b = next(it)
                x, t = b["image"].to(device, non_blocking=True), b["mask"].to(device, non_blocking=True)
                with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                    low = model(x)
                loss = crit(logits_at_input(low, cfg), t)[0]
                scaler.scale(loss).backward()
                scaler.step(opt)
                scaler.update()
                opt.zero_grad(set_to_none=True)
                torch.cuda.synchronize(device)
                now = time.time()
                if last is not None and s >= 3:
                    times.append(now - last)
                last = now
            train_peak = torch.cuda.max_memory_allocated(device) / 2**30
            model.eval()
            vl = DataLoader(val_ds, batch_size=k if k > 1 else micro, shuffle=False,
                            num_workers=int(cfg["data"].get("num_workers", 0)), worker_init_fn=_worker_init, pin_memory=True)
            vt, vn, vlast = [], 0, None
            with torch.no_grad():
                for s, b in enumerate(vl):
                    with torch.autocast("cuda", dtype=amp_dtype, enabled=use_amp):
                        low = model(b["image"].to(device, non_blocking=True))
                    prob_maps(low, cfg)
                    torch.cuda.synchronize(device)
                    now = time.time()
                    if vlast is not None:
                        vt.append(now - vlast)
                        vn += b["image"].shape[0]
                    vlast = now
                    if s >= 6:
                        break
            rec.update(status="ok", train_s_per_micro_batch=float(np.median(times)),
                       train_s_per_sample=float(np.median(times)) / micro, train_peak_alloc_gb=round(train_peak, 2),
                       peak_reserved_gb=round(torch.cuda.max_memory_reserved(device) / 2**30, 2),
                       val_s_per_sample=float(sum(vt) / vn))
        except torch.cuda.OutOfMemoryError as e:
            rec.update(status="out_of_memory", error=str(e).split("\n")[0])
        return rec

    for micro in micros:
        rec = once(micro)
        gc.collect()
        torch.cuda.empty_cache()
        print(f"[probe] {json.dumps(rec)}", flush=True)
        rows.append(rec)
    ok = [r for r in rows if r["status"] == "ok"]
    best = min(ok, key=lambda r: r["train_s_per_sample"]) if ok else None
    est = None
    if best:
        tr = len(train_ds) * best["train_s_per_sample"]
        va = len(val_ds) * best["val_s_per_sample"]
        est = {"micro_batch": best["micro_batch"], "train_samples": len(train_ds), "val_samples": len(val_ds),
               "train_s": round(tr), "val_s": round(va), "epoch_s": round(tr + va),
               "hours_per_50_epochs": round(50 * (tr + va) / 3600, 2),
               "hours_at_max_150_epochs": round(150 * (tr + va) / 3600, 2)}
    out = {"run": cfg.get("run_name"), "variant": sam_variant(cfg), "gpu": torch.cuda.get_device_name(device),
           "amp": use_amp, "steps": steps, "rows": rows, "estimate_fastest_micro_batch": est,
           "note": "time per sample = median wall time per optimizer step (includes data loading) / micro-batch, "
                   "after 3 warm-up steps; the epoch estimate ignores start-up and checkpointing"}
    path = CHECKS_DIR / f"sam_probe_{cfg.get('run_name')}.json"
    with open(path, "w") as f:
        json.dump(out, f, indent=1)
    print(json.dumps(est, indent=1))
    print(f"[probe] wrote {path}")


def main() -> None:
    from src.utils import load_config
    ap = argparse.ArgumentParser(description="Fine-tuned SAM checks")
    ap.add_argument("command", choices=["gradcheck", "probe"])
    ap.add_argument("--config", required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--steps", type=int, default=None)
    ap.add_argument("--micro", type=int, nargs="+", default=[1, 2, 4])
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.command == "gradcheck":
        gradcheck(cfg, a.seed, a.steps or 3)
    else:
        probe(cfg, a.seed, a.micro, a.steps or 12)


if __name__ == "__main__":
    main()
