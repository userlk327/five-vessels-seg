"""Training loss shared by every trained model (the four CNNs and fine-tuned SAM, 5b).

    loss = bce_weight * BCEWithLogits(logits, target) + dice_weight * (1 - soft Dice)

* **Float32.** Logits and targets are cast to float32 and autocast is switched off inside
  the loss, so fp16 mixed precision never touches the loss arithmetic.
* **Targets are used exactly as given.** With the Phase 2 decision (soft targets: area
  average, no threshold) both terms see values in [0, 1], e.g. 0.25 for a 1-px vessel at
  512. Nothing in the training path thresholds or rounds them (tests/test_losses.py).
* **BCE:** mean over every pixel of the batch.
* **Soft Dice:** per image, D_i = (2 * sum(p * t) + eps) / (sum(p) + sum(t) + eps) over the
  image's pixels, p = sigmoid(logits), eps = 1; the Dice term is the mean of (1 - D_i) over
  the batch.

Command line (a check, not part of training):

    python -m src.losses optimum   # where BCE + soft Dice puts each soft-target value at its
                                   # optimum (free per-pixel logits) -> results/checks/loss_optimum_*
"""
from __future__ import annotations

import argparse
import json
import os

import numpy as np
import torch
import torch.nn.functional as F


class BCEDiceLoss(torch.nn.Module):
    def __init__(self, bce_weight: float = 1.0, dice_weight: float = 1.0, eps: float = 1.0):
        super().__init__()
        self.bce_weight, self.dice_weight, self.eps = float(bce_weight), float(dice_weight), float(eps)

    @classmethod
    def from_config(cls, cfg: dict) -> "BCEDiceLoss":
        c = cfg["train"]["loss"]
        if c.get("dtype", "float32") != "float32":
            raise ValueError("the loss is always computed in float32")
        return cls(c["bce_weight"], c["dice_weight"], c.get("dice_eps", 1.0))

    def forward(self, logits: torch.Tensor, target: torch.Tensor) -> tuple[torch.Tensor, dict]:
        """logits, target: (N, 1, H, W). Returns (loss, {"bce": float, "dice_loss": float})."""
        if logits.shape != target.shape:
            raise ValueError(f"logits {tuple(logits.shape)} vs target {tuple(target.shape)}")
        if not target.is_floating_point():
            raise TypeError("targets must be floating point (soft targets are used as is)")
        with torch.autocast(device_type=logits.device.type, enabled=False):
            z, t = logits.float(), target.float()
            bce = F.binary_cross_entropy_with_logits(z, t)
            p = torch.sigmoid(z)
            dims = tuple(range(1, z.ndim))
            inter = (p * t).sum(dims)
            dice = (2 * inter + self.eps) / (p.sum(dims) + t.sum(dims) + self.eps)
            dice_loss = (1 - dice).mean()
            loss = self.bce_weight * bce + self.dice_weight * dice_loss
        return loss, {"bce": float(bce.detach()), "dice_loss": float(dice_loss.detach())}


# --------------------------------------------------------------------------- check: loss optimum
def free_logit_optimum(target: torch.Tensor, loss_fn, steps: int = 4000, lr: float = 0.1) -> dict:
    """Minimize ``loss_fn`` over free per-pixel logits (no network): where the loss itself
    wants each pixel to be. Adam with a cosine learning-rate decay to 0."""
    z = torch.zeros_like(target, requires_grad=True)
    opt = torch.optim.Adam([z], lr=lr)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, steps)
    for _ in range(steps):
        opt.zero_grad()
        loss = loss_fn(z, target)[0]
        loss.backward()
        opt.step()
        sched.step()
    opt.zero_grad()
    loss = loss_fn(z, target)[0]
    loss.backward()
    return {"p": torch.sigmoid(z).detach(), "loss": float(loss), "grad_max": float(z.grad.abs().max())}


def cmd_optimum(cfg: dict, n_images: int, steps: int) -> None:
    import pandas as pd

    from src.data import CHECKS_DIR, raw_mask_path, read_unchanged
    from src.evaluate import eval_settings, load_fov, maskresize_sample, sweep_metrics, threshold_grid
    from src.metrics import prepare_gt
    from src.resample import downsample_mask, upsample_prob

    torch.manual_seed(0)
    res = int(cfg["data"]["cnn_resolution"])
    method = cfg["data"]["mask_resize"]
    s = eval_settings(cfg)
    grid = threshold_grid(cfg)
    # first image of each category in the Phase 2 mask-resizing sample (training split)
    sample = maskresize_sample(cfg)
    ids = [next(i for i in sample if i.endswith(c)) for c in "ADGN"][:n_images]
    full = BCEDiceLoss.from_config(cfg)
    bce_only = BCEDiceLoss(full.bce_weight, 0.0, full.eps)
    levels = sorted({k / 16 for k in range(17)})
    rows, sweeps = [], []
    for image_id in ids:
        gt = read_unchanged(raw_mask_path(cfg, image_id)) > 0
        t_np = downsample_mask(gt, res, method)
        t = torch.from_numpy(t_np)[None, None]
        g = prepare_gt(gt, load_fov(cfg, image_id))
        maps = {"target": t_np}
        for name, fn in (("bce_only", bce_only), ("bce_dice", full)):
            out = free_logit_optimum(t, fn, steps)
            p = out["p"][0, 0].numpy()
            maps[name] = p
            d = float((2 * (p * t_np).sum() + full.eps) / (p.sum() + t_np.sum() + full.eps))
            for lv in levels:
                sel = np.isclose(t_np, lv)
                if sel.any():
                    rows.append({"image_id": image_id, "loss": name, "target": lv, "n_px": int(sel.sum()),
                                 "p_mean": float(p[sel].mean()), "p_min": float(p[sel].min()),
                                 "p_max": float(p[sel].max()), "dice_at_optimum": d,
                                 "final_loss": out["loss"], "final_grad_max": out["grad_max"]})
        for name, m in maps.items():
            up = upsample_prob(m.astype(np.float16).astype(np.float32), s["native"])
            for r in sweep_metrics(up, g, grid, s["cutoffs"], s["center_tol"]):
                sweeps.append({"image_id": image_id, "map": name, **r})
        print(f"[optimum] {image_id} done", flush=True)
    df, sw = pd.DataFrame(rows), pd.DataFrame(sweeps)
    df.to_csv(CHECKS_DIR / "loss_optimum_levels.csv", index=False)
    sw.to_csv(CHECKS_DIR / "loss_optimum_sweep.csv", index=False)
    by_level = {}
    for (name, lv), x in df.groupby(["loss", "target"]):
        by_level.setdefault(f"{lv:g}", {})[name] = round(float(np.average(x.p_mean, weights=x.n_px)), 4)
    b = df[df.loss == "bce_only"]
    best = {}
    for name, x in sw.groupby("map"):
        m = x.groupby("threshold")[["dice", "cl_recall_thin", "cl_recall_medium", "cl_recall_thick"]].mean()
        tb = float(m.dice.idxmax())
        best[name] = {"at_0.5": m.loc[0.5].round(4).to_dict(), "best_dice_threshold": tb,
                      "at_best_dice_threshold": m.loc[tb].round(4).to_dict()}
    summary = {"images": ids, "resolution": res, "target_method": method, "steps": steps,
               "optimizer": "Adam lr 0.1, cosine decay to 0, free per-pixel logits (no network), batch of 1",
               "loss": {"bce_weight": full.bce_weight, "dice_weight": full.dice_weight, "dice_eps": full.eps},
               "mean_p_at_optimum_by_target_value": by_level,
               "max_abs_p_minus_t_bce_only": float(max((b.p_max - b.target).abs().max(), (b.p_min - b.target).abs().max())),
               "final_grad_max": df.groupby("loss").final_grad_max.max().to_dict(),
               "dice_at_optimum": df.groupby(["loss", "image_id"]).dice_at_optimum.first().round(4).unstack(0).to_dict(),
               "evaluation_at_2048_mean_over_images": best,
               "note": "float16 storage and bilinear upsampling to 2048 as for predictions; metrics inside the FOV"}
    with open(CHECKS_DIR / "loss_optimum_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))


def main() -> None:
    from src.utils import load_config
    ap = argparse.ArgumentParser(description="Loss checks")
    ap.add_argument("command", choices=["optimum"])
    ap.add_argument("--config", default="configs/train_shared.yaml")
    ap.add_argument("--images", type=int, default=4)
    ap.add_argument("--steps", type=int, default=4000)
    a = ap.parse_args()
    torch.set_num_threads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    cmd_optimum(load_config(a.config), a.images, a.steps)


if __name__ == "__main__":
    main()
