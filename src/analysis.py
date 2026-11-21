"""Analysis: tables (main, width-stratified, disease breakdown, 2x2 ablation with bootstrap CIs,
vessel measures, cost), Wilcoxon signed-rank tests with Holm correction, figures (Phase 6).

Available now:
    plot_training_curves   training / validation curves of one run (Phase 3 onwards)

    python -m src.analysis curves --config configs/unet.yaml --seed 0   # redraw from the saved log
    python -m src.analysis params    # parameter table of the four CNNs -> results/tables/model_params.csv
    python -m src.analysis runs      # one row per trained run: threshold, epochs, time, memory, val metrics
    python -m src.analysis ablation  # 2x2 effects on VAL (exploratory; the test analysis is Phase 6)
    python -m src.analysis seam      # tile-seam check of tiled SAM on VAL (CPU job; results/checks/seam_check_*)
    python -m src.analysis sam       # Phase 5 first look on VAL: SAM runs, SAM vs each CNN per image, oracle table

Seeds (CLAUDE.md, Phase 6 rules, used here on val): model-level numbers are computed per seed,
then mean +- sd across seeds; per-image analyses first average each image's metric over
that model's seeds, so each image counts once.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

# Reference categorical palette (dataviz skill, light mode): slot 1 blue, slot 2 orange.
SERIES = ("#2a78d6", "#eb6834")
INK, MUTED, GRID, SURFACE = "#0b0b0b", "#898781", "#e1e0d9", "#fcfcfb"


def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    ax.grid(True, color=GRID, lw=0.6)
    ax.set_axisbelow(True)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color("#c3c2b7")
    ax.tick_params(colors=MUTED, labelcolor=INK, labelsize=8)


def plot_training_curves(log_csv: Path, out_png: Path, title: str) -> None:
    """Two panels, one measure each: the loss (train vs val) and the validation Dice used for
    early stopping (2048, threshold 0.5, inside the FOV), with the best epoch marked."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = pd.read_csv(log_csv)
    best = log.loc[log.val_dice.idxmax()]
    fig, ax = plt.subplots(1, 2, figsize=(11, 3.8), facecolor=SURFACE)
    ax[0].plot(log.epoch, log.train_loss, color=SERIES[0], lw=2, label="train")
    ax[0].plot(log.epoch, log.val_loss, color=SERIES[1], lw=2, label="validation")
    ax[0].set_ylabel("BCE + soft Dice loss", color=INK, fontsize=9)
    ax[0].legend(frameon=False, fontsize=8, labelcolor=INK)
    ax[1].plot(log.epoch, log.val_dice, color=SERIES[0], lw=2)
    ax[1].set_ylabel("validation Dice (2048, threshold 0.5, FOV)", color=INK, fontsize=9)
    from matplotlib.ticker import MaxNLocator
    for a in ax:
        _style(a)
        a.xaxis.set_major_locator(MaxNLocator(integer=True))
        a.set_xlabel("epoch", color=INK, fontsize=9)
        a.axvline(best.epoch, color=MUTED, lw=1, ls="--")
    ax[1].annotate(f"best: epoch {int(best.epoch)}, {best.val_dice:.4f}", (best.epoch, best.val_dice),
                   xytext=(6, -14), textcoords="offset points", fontsize=8, color=INK)
    fig.suptitle(title, fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out_png, dpi=120, facecolor=SURFACE)
    plt.close(fig)


CNN_CONFIGS = {"U-Net": "configs/unet.yaml", "Attention U-Net": "configs/attention_unet.yaml",
               "Multi-Scale U-Net": "configs/multiscale_unet.yaml",
               "Multi-Scale Attention U-Net": "configs/ms_attention_unet.yaml"}


def conv_macs(model, size: int) -> int:
    """Multiply-accumulates of all Conv2d / ConvTranspose2d layers for one size x size
    image (BN, activations, pooling, interpolation and element-wise ops not counted)."""
    import torch
    total = [0]

    def hook(m, inp, out):
        if isinstance(m, torch.nn.ConvTranspose2d):
            total[0] += inp[0].numel() * m.out_channels * m.kernel_size[0] * m.kernel_size[1] // m.groups
        else:
            total[0] += out.numel() * (m.in_channels // m.groups) * m.kernel_size[0] * m.kernel_size[1]

    hs = [m.register_forward_hook(hook) for m in model.modules()
          if isinstance(m, (torch.nn.Conv2d, torch.nn.ConvTranspose2d))]
    model.eval()
    with torch.no_grad():
        model(torch.zeros(1, 3, size, size))
    for h in hs:
        h.remove()
    return total[0]


def cmd_params() -> None:
    import torch

    from src.models import build_model
    from src.models.blocks import AttentionGate, MultiScaleConv
    from src.utils import load_config, project_path
    rows = []
    for name, path in CNN_CONFIGS.items():
        cfg = load_config(path)
        torch.manual_seed(0)
        m = build_model(cfg)
        att = sum(p.numel() for g in m.modules() if isinstance(g, AttentionGate) for p in g.parameters())
        fuse = sum(p.numel() for g in m.modules() if isinstance(g, MultiScaleConv)
                   for p in list(g.fuse.parameters()) + list(g.bn.parameters()))
        rows.append({"model": name, "config": path,
                     "params_total": sum(p.numel() for p in m.parameters()),
                     "params_trainable": sum(p.numel() for p in m.parameters() if p.requires_grad),
                     "attention_gate_params": att, "multiscale_fusion_and_bn_params": fuse,
                     "gmacs_conv_at_cnn_resolution": round(conv_macs(m, int(cfg["data"]["cnn_resolution"])) / 1e9, 2),
                     "cnn_resolution": int(cfg["data"]["cnn_resolution"])})
    df = pd.DataFrame(rows)
    base = df.loc[df.model == "U-Net"].iloc[0]
    df["params_vs_unet_pct"] = (100 * (df.params_total / base.params_total - 1)).round(2)
    df["gmacs_vs_unet_pct"] = (100 * (df.gmacs_conv_at_cnn_resolution / base.gmacs_conv_at_cnn_resolution - 1)).round(1)
    out = project_path("results/tables")
    out.mkdir(parents=True, exist_ok=True)
    df.to_csv(out / "model_params.csv", index=False)
    print(df.drop(columns=["config"]).to_string(index=False))
    print(f"[analysis] wrote {out / 'model_params.csv'}")


ABLATION = {"unet": "U-Net", "attention_unet": "Attention U-Net", "multiscale_unet": "Multi-Scale U-Net",
            "ms_attention_unet": "Multi-Scale Attention U-Net"}
MAIN_METRICS = ["dice", "iou", "precision", "recall", "cl_recall_thin", "cl_recall_medium", "cl_recall_thick",
                "cldice", "nsd"]


def load_eval(split: str, models=ABLATION, root: Path | None = None) -> pd.DataFrame:
    """Per-image metrics of every available (model, seed) run for one split."""
    from src.utils import project_path
    root = root or project_path("results/eval")
    frames = [pd.read_csv(f) for m in models for f in sorted((root / m).glob(f"seed*/{split}_metrics.csv"))]
    if not frames:
        raise FileNotFoundError(f"no {split}_metrics.csv under {root}")
    return pd.concat(frames, ignore_index=True)


def bootstrap_ci(x, n_boot: int = 10000, seed: int = 0, level: float = 0.95) -> tuple[float, float]:
    """Percentile bootstrap CI of the mean, resampling the per-image values."""
    import numpy as np
    x = np.asarray(x, float)
    x = x[np.isfinite(x)]
    idx = np.random.RandomState(seed).randint(0, len(x), (n_boot, len(x)))
    means = x[idx].mean(1)
    a = (1 - level) / 2
    return float(np.quantile(means, a)), float(np.quantile(means, 1 - a))


def ablation_effects(per_image: pd.DataFrame, metrics=MAIN_METRICS, n_boot: int = 10000, seed: int = 0) -> pd.DataFrame:
    """2x2 effects per image (CLAUDE.md Phase 6), from a table indexed by image_id with one
    column per (model, metric) holding the image's metric averaged over that model's seeds:
      attention   = mean[(Att - U), (MSAtt - MS)]
      multi-scale = mean[(MS - U), (MSAtt - Att)]
      interaction = (MSAtt - MS) - (Att - U)
    Mean over images with a 95% bootstrap CI (images resampled)."""
    rows = []
    for k in metrics:
        u, a, m, ma = (per_image[(name, k)] for name in ABLATION)
        effects = {"attention": ((a - u) + (ma - m)) / 2, "multi-scale": ((m - u) + (ma - a)) / 2,
                   "interaction": (ma - m) - (a - u)}
        for e, v in effects.items():
            lo, hi = bootstrap_ci(v, n_boot, seed)
            rows.append({"metric": k, "effect": e, "mean": float(v.mean()), "ci_low": lo, "ci_high": hi,
                         "n_images": int(v.notna().sum())})
    return pd.DataFrame(rows)


def run_rows(split: str, models: dict) -> list[dict]:
    """One row per trained (model, seed) run: threshold, epochs, time, memory, mean metrics."""
    import json

    from src.utils import project_path
    rows = []
    for m, label in models.items():
        for f in sorted(project_path("results/eval").glob(f"{m}/seed*/{split}_metrics.csv")):
            seed = int(f.parent.name[4:])
            df = pd.read_csv(f)
            s = json.loads((project_path("results/train") / m / f"seed{seed}" / "summary.json").read_text())
            rows.append({"model": label, "seed": seed, "threshold": float(df.threshold.iloc[0]),
                         "best_epoch": s["best_epoch"], "epochs_run": s["epochs_run"],
                         "epoch_time_s_median": s["epoch_time_s"]["median"], "training_time_h": s["training_time_h"],
                         "gpu_peak_alloc_gb": s["gpu_peak_alloc_gb"], "host": s["sessions"][-1]["host"],
                         "slurm_job_id": s["sessions"][-1]["slurm_job_id"], "n_images": len(df),
                         **df[MAIN_METRICS].mean().to_dict()})
    return rows


def cmd_runs(split: str = "val") -> pd.DataFrame:
    from src.utils import project_path
    out = pd.DataFrame(run_rows(split, ABLATION))
    path = project_path("results/tables") / f"{split}_runs.csv"
    path.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(path, index=False)
    print(out.round(4).to_string(index=False))
    print(f"[analysis] wrote {path}")
    return out


def load_sweep_at(split: str, threshold: float, models=ABLATION) -> pd.DataFrame:
    """Per-image threshold-dependent metrics of every run at one fixed grid threshold (from
    the saved sweeps): a sensitivity check that removes the per-run threshold choice."""
    from src.utils import project_path
    frames = [pd.read_csv(f) for m in models for f in sorted(project_path("results/eval").glob(f"{m}/seed*/{split}_sweep.csv"))]
    df = pd.concat(frames, ignore_index=True)
    out = df[df.threshold.round(4) == round(threshold, 4)]
    if out.empty:
        raise SystemExit(f"threshold {threshold} is not on the sweep grid")
    return out


def cmd_ablation(split: str = "val", final_test: bool = False, fixed_threshold: float | None = None) -> None:
    from src.utils import project_path
    if split == "test" and not final_test:
        raise SystemExit("Test-set analysis happens in Phase 6: pass --final-test.")
    metrics = MAIN_METRICS
    if fixed_threshold is None:
        df, tag = load_eval(split), ""
    else:
        df, tag = load_sweep_at(split, fixed_threshold), f"_fixed_t{fixed_threshold:g}"
        metrics = ["dice", "iou", "precision", "recall", "cl_recall_thin", "cl_recall_medium", "cl_recall_thick"]
    missing = [m for m in ABLATION if m not in set(df.model)]
    if missing:
        raise SystemExit(f"no {split} results yet for {missing}")
    seeds = df.groupby("model").seed.apply(lambda s: sorted(s.unique())).to_dict()
    # model level: per-seed means, then mean +- sd across seeds
    per_seed = df.groupby(["model", "seed"])[metrics].mean()
    model_level = per_seed.groupby("model").agg(["mean", "std"])
    model_level.columns = [f"{a}_{b}" for a, b in model_level.columns]
    model_level.insert(0, "n_seeds", per_seed.groupby("model").size())
    model_level = model_level.loc[list(ABLATION)].rename(index=ABLATION)
    # per image: average over each model's seeds first, then the 2x2 effects
    per_image = df.groupby(["image_id", "model"])[metrics].mean().unstack("model")
    per_image.columns = per_image.columns.swaplevel()
    effects = ablation_effects(per_image, metrics)
    effects.insert(0, "split", split)
    effects.insert(1, "threshold", "val-selected per run" if fixed_threshold is None else fixed_threshold)
    out = project_path("results/tables")
    out.mkdir(parents=True, exist_ok=True)
    model_level.to_csv(out / f"{split}_model_level{tag}.csv")
    effects.to_csv(out / f"{split}_ablation_effects{tag}.csv", index=False)
    print(f"seeds per model: { {k: [int(x) for x in v] for k, v in seeds.items()} }")
    show = [k for k in ("dice", "cl_recall_thin", "cldice", "nsd") if k in metrics]
    print(model_level[[f"{k}_{s}" for k in show for s in ("mean", "std")]].round(4).to_string())
    print(effects.round(4).to_string(index=False))
    print(f"[analysis] wrote {out / f'{split}_model_level{tag}.csv'} and {out / f'{split}_ablation_effects{tag}.csv'}")


SAM_AUTO = {"sam_lora_tiled": "SAM LoRA, tiled 2x2 (main)", "sam_lora_whole": "SAM LoRA, whole image"}
SAM_REFERENCE = {"sam_lora_tiled": "reference_soft_512", "sam_lora_whole": "reference_soft_256"}
SAM_ORACLE_LABEL = "zero-shot, oracle point prompts"


def cmd_sam(split: str = "val") -> None:
    """Phase 5 first look, VAL only (the test analysis is Phase 6). Writes
      {split}_sam_runs.csv          the fine-tuned (automatic) SAM runs, plus each one's
                                    mask-resampling reference as a context row (not an upper bound)
      {split}_sam_vs_cnn_paired.csv per image: SAM minus each CNN (CNN metric averaged over its
                                    seeds first), mean, 95% bootstrap CI, share of images SAM > CNN
      {split}_sam_oracle.csv        zero-shot SAM with ORACLE point prompts, per K (separate table)
    """
    import json

    import numpy as np

    from src.utils import project_path
    if split != "val":
        raise SystemExit("The SAM first look is on val only; the test analysis is Phase 6.")
    root, tables = project_path("results/eval"), project_path("results/tables")
    tables.mkdir(parents=True, exist_ok=True)

    rows = run_rows(split, SAM_AUTO)
    for m, ref in SAM_REFERENCE.items():
        df = pd.read_csv(root / ref / f"{split}_metrics.csv")
        rows.append({"model": f"{ref} (context for {m}; not an upper bound)", "threshold": float(df.threshold.iloc[0]),
                     "n_images": len(df), **df[MAIN_METRICS].mean().to_dict()})
    runs = pd.DataFrame(rows)
    runs.to_csv(tables / f"{split}_sam_runs.csv", index=False)

    cnn = load_eval(split).groupby(["model", "image_id"])[MAIN_METRICS].mean()
    paired = []
    for m, label in SAM_AUTO.items():
        sam_df = load_eval(split, models={m: label}).set_index("image_id")[MAIN_METRICS]
        for c, clabel in ABLATION.items():
            other = cnn.loc[c]
            for k in MAIN_METRICS:
                d = (sam_df[k] - other[k].reindex(sam_df.index)).to_numpy(float)
                d = d[np.isfinite(d)]
                lo, hi = bootstrap_ci(d)
                paired.append({"sam_model": m, "cnn_model": clabel, "metric": k, "mean_diff": float(d.mean()),
                               "ci_low": lo, "ci_high": hi, "share_sam_better": float((d > 0).mean()),
                               "n_images": len(d)})
    paired = pd.DataFrame(paired)
    paired.to_csv(tables / f"{split}_sam_vs_cnn_paired.csv", index=False)

    choice = json.loads((root / "sam_oracle_choice.json").read_text())
    orows = []
    for f in sorted(root.glob(f"sam_oracle_k*/seed*/{split}_metrics.csv"), key=lambda f: int(f.parts[-3][12:])):
        df = pd.read_csv(f)
        k = int(f.parts[-3][12:])
        orows.append({"label": SAM_ORACLE_LABEL, "k": k, "chosen": k == choice["chosen_k"],
                      "threshold": float(df.threshold.iloc[0]), "n_images": len(df),
                      **df[MAIN_METRICS + ["density_pred", "density_gt"]].mean().to_dict()})
    oracle = pd.DataFrame(orows)
    oracle.to_csv(tables / f"{split}_sam_oracle.csv", index=False)

    print(runs.drop(columns=["host", "slurm_job_id"], errors="ignore").round(4).to_string(index=False))
    show = paired[paired.metric.isin(["dice", "cl_recall_thin", "cldice", "nsd"])]
    print(show.round(4).to_string(index=False))
    print(oracle.round(4).to_string(index=False))
    print(f"[analysis] wrote {split}_sam_runs.csv, {split}_sam_vs_cnn_paired.csv, {split}_sam_oracle.csv in {tables}")


SEAM_RUNS = (("sam_lora_tiled", 0), ("sam_lora_whole", 0), ("unet", 0), ("unet", 1), ("unet", 2))
SEAM_BAND_PX = (16, 32, 64)        # half-widths at 2048 (16 px = 4 px of the stitched 512 grid)


def _line_distance(native: int, positions) -> "np.ndarray":
    """Distance of every pixel centre to the nearest vertical or horizontal line at ``positions``."""
    import numpy as np
    c = np.arange(native) + 0.5
    d = np.min(np.abs(c[:, None] - np.asarray(positions, float)[None, :]), axis=1)
    return np.minimum(d[:, None], d[None, :])


def _grid_jumps(prob: "np.ndarray", fov: "np.ndarray") -> dict:
    """Mean |p[k+1] - p[k]| between adjacent columns / rows of the model's own output grid
    (pixels fully inside the FOV), at the central seam (k = n/2 - 1) and, as reference, the
    median over the central half of the grid excluding the seam +- 2."""
    import cv2
    import numpy as np
    n = prob.shape[0]
    inside = cv2.resize(fov.astype(np.float32), (n, n), interpolation=cv2.INTER_AREA) >= 0.999
    out = {"grid": n}
    for name, axis in (("col", 1), ("row", 0)):
        p, m = (prob, inside) if axis == 1 else (prob.T, inside.T)
        jump, both = np.abs(np.diff(p, axis=1)), m[:, 1:] & m[:, :-1]
        prof = np.array([jump[both[:, k], k].mean() if both[:, k].any() else np.nan for k in range(n - 1)])
        k = n // 2 - 1
        ref = np.r_[prof[n // 4:k - 2], prof[k + 3:3 * n // 4]]
        out[f"jump_{name}_seam"], out[f"jump_{name}_ref"] = float(prof[k]), float(np.nanmedian(ref))
    out["jump_ratio"] = (out["jump_col_seam"] + out["jump_row_seam"]) / (out["jump_col_ref"] + out["jump_row_ref"])
    return out


def _seam_one(args) -> list[dict]:
    import cv2
    import numpy as np
    from scipy import ndimage as ndi

    from src.data import raw_mask_path, read_npz_member, read_unchanged
    from src.evaluate import eval_settings, load_fov
    from src.metrics import prepare_gt, width_class
    from src.resample import binarize, upsample_prob
    image_id, cfg, runs = args
    cv2.setNumThreads(1)
    s = eval_settings(cfg)
    native = s["native"]
    g = prepare_gt(read_unchanged(raw_mask_path(cfg, image_id)) > 0, load_fov(cfg, image_id))
    thin = np.zeros_like(g.skel)
    thin[np.nonzero(g.skel)] = width_class(g.skel_radius, s["cutoffs"]) == 0
    bands = {"seam": _line_distance(native, [native / 2]),
             "midline": _line_distance(native, [native / 4, 3 * native / 4])}
    rows = []
    for run, seed, archive, threshold in runs:
        prob = read_npz_member(archive, image_id).astype(np.float32)
        pred = binarize(upsample_prob(prob, native), threshold) & g.fov
        hit = ndi.binary_dilation(pred, structure=np.ones((3, 3), bool), iterations=s["center_tol"])
        base = {"image_id": image_id, "model": run, "seed": seed, "threshold": threshold, **_grid_jumps(prob, g.fov)}
        for band, dist in bands.items():
            for w in SEAM_BAND_PX:
                b = (dist < w) & g.fov
                sk, skt = g.skel & b, thin & b
                rows.append({**base, "band": band, "half_width_px": w,
                             "tp": int((pred & g.gt & b).sum()), "n_pred": int((pred & b).sum()),
                             "n_gt": int((g.gt & b).sum()), "n_skel": int(sk.sum()), "skel_hit": int((hit & sk).sum()),
                             "n_thin": int(skt.sum()), "thin_hit": int((hit & skt).sum())})
    return rows


def cmd_seam(split: str = "val", workers: int = 1) -> None:
    """Tile-seam check for tiled SAM, VAL only. Tiled SAM's 2x2 tiles meet at x = 1024 and
    y = 1024 (native px); the four 256 decoder grids are stitched without overlap. Two measures:
      1. discontinuity: probability jump across the seam on the model's own grid, relative to
         the other adjacent columns / rows (jump_ratio; ~1 if no seam effect);
      2. accuracy near the seam: Dice and centreline recall (all, thin) inside bands of
         half-width 16 / 32 / 64 px around the seam lines vs around the tile MIDLINES
         (x, y = 512, 1536), as a difference-in-differences against models without seams
         (U-Net seeds 0-2, averaged per image; whole-image SAM):
           DiD = (tiled - control)[seam band] - (tiled - control)[midline band]
         < 0 means tiled SAM loses more near its seams than elsewhere. 95% bootstrap CIs
         over images. Writes results/checks/seam_check_per_image.csv, seam_check_summary.json."""
    import json

    import numpy as np

    from src.data import split_ids
    from src.evaluate import _pool
    from src.utils import load_config, prob_archive_path, project_path
    if split != "val":
        raise SystemExit("The seam check runs on val only.")
    cfg = load_config("configs/base.yaml")
    runs = []
    for run, seed in SEAM_RUNS:
        thr = json.loads((project_path("results/eval") / run / f"seed{seed}" / "val_threshold.json").read_text())["threshold"]
        archive = prob_archive_path(project_path(cfg["paths"]["outputs_dir"]) / run / f"seed{seed}", split)
        runs.append((run, seed, archive, float(thr)))
    ids = split_ids(cfg, split)
    with _pool(workers) as ex:
        rows = [r for part in ex.map(_seam_one, [(i, cfg, runs) for i in ids], chunksize=2) for r in part]
    df = pd.DataFrame(rows)
    out_dir = project_path("results/checks")
    df.to_csv(out_dir / "seam_check_per_image.csv", index=False)

    with np.errstate(invalid="ignore", divide="ignore"):         # per run and image; NaN if nothing in the band
        df["dice"] = 2 * df.tp / (df.n_pred + df.n_gt)
        df["cl_recall"] = df.skel_hit / df.n_skel
        df["cl_recall_thin"] = df.thin_hit / df.n_thin
    df = df.assign(model=df.model.where(df.model != "unet", "unet_mean_s012"))   # U-Net: mean over seeds per image
    per = df.groupby(["model", "image_id", "band", "half_width_px"], as_index=False)[
        ["tp", "n_pred", "n_gt", "n_skel", "skel_hit", "n_thin", "thin_hit", "jump_ratio",
         "dice", "cl_recall", "cl_recall_thin"]].mean()
    from src.utils import environment_info
    env = {k: v for k, v in environment_info().items() if k in ("time", "host", "slurm_job_id", "git")}
    summary = {"split": split, "n_images": len(ids), "environment": env, "runs": [[r, s_, t] for r, s_, _, t in runs],
               "band_half_widths_px": list(SEAM_BAND_PX), "pooled": [], "did": [], "jump_ratio": {}}
    for (m, band, w), d in per.groupby(["model", "band", "half_width_px"]):
        sums = d[["tp", "n_pred", "n_gt", "n_skel", "skel_hit", "n_thin", "thin_hit"]].sum()
        summary["pooled"].append({"model": m, "band": band, "half_width_px": int(w),
                                  "dice": float(2 * sums.tp / (sums.n_pred + sums.n_gt)),
                                  "cl_recall": float(sums.skel_hit / sums.n_skel),
                                  "cl_recall_thin": float(sums.thin_hit / sums.n_thin), "n_thin": int(sums.n_thin)})
    wide = per.pivot_table(index=["image_id", "half_width_px"], columns=["model", "band"],
                           values=["dice", "cl_recall", "cl_recall_thin"])
    for control in ("unet_mean_s012", "sam_lora_whole"):
        for w in SEAM_BAND_PX:
            sub = wide.xs(w, level="half_width_px")
            for k in ("dice", "cl_recall", "cl_recall_thin"):
                did = ((sub[(k, "sam_lora_tiled", "seam")] - sub[(k, control, "seam")])
                       - (sub[(k, "sam_lora_tiled", "midline")] - sub[(k, control, "midline")])).dropna()
                lo, hi = bootstrap_ci(did)
                summary["did"].append({"control": control, "half_width_px": w, "metric": k, "did_mean": float(did.mean()),
                                       "ci_low": lo, "ci_high": hi, "n_images": int(len(did))})
    jr = per[(per.band == "seam") & (per.half_width_px == SEAM_BAND_PX[0])].pivot(index="image_id", columns="model",
                                                                                 values="jump_ratio")
    for m in jr.columns:
        summary["jump_ratio"][m] = {"mean": float(jr[m].mean()), "median": float(jr[m].median())}
    d = (jr["sam_lora_tiled"] - jr["unet_mean_s012"]).dropna()
    lo, hi = bootstrap_ci(d)
    summary["jump_ratio"]["tiled_minus_unet"] = {"mean": float(d.mean()), "ci_low": lo, "ci_high": hi}
    with open(out_dir / "seam_check_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(pd.DataFrame(summary["pooled"]).round(4).to_string(index=False))
    print(pd.DataFrame(summary["did"]).round(4).to_string(index=False))
    print(json.dumps(summary["jump_ratio"], indent=1))
    print(f"[analysis] wrote {out_dir / 'seam_check_per_image.csv'}, seam_check_summary.json")


def cmd_curves_all() -> None:
    """Validation Dice per epoch for every run of the four CNNs: one panel per model,
    one line per seed (results/figures/val_dice_curves_cnns.png)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.ticker import MaxNLocator

    from src.utils import project_path
    seed_colors = ("#2a78d6", "#eb6834", "#1baf7a")          # reference categorical slots 1-3
    fig, axes = plt.subplots(1, 4, figsize=(16, 3.6), sharey=True, facecolor=SURFACE)
    for ax, (m, label) in zip(axes, ABLATION.items()):
        for seed, color in enumerate(seed_colors):
            f = project_path("results/train") / m / f"seed{seed}" / "train_log.csv"
            if not f.exists():
                continue
            log = pd.read_csv(f)
            ax.plot(log.epoch, log.val_dice, color=color, lw=1.5, label=f"seed {seed}")
            b = log.loc[log.val_dice.idxmax()]
            ax.plot([b.epoch], [b.val_dice], "o", ms=5, color=color, markeredgecolor=SURFACE, markeredgewidth=1)
        _style(ax)
        ax.xaxis.set_major_locator(MaxNLocator(integer=True))
        ax.set_title(label, fontsize=9, color=INK)
        ax.set_xlabel("epoch", color=INK, fontsize=9)
        ax.set_ylim(0.85, 0.925)
    axes[0].set_ylabel("val Dice (2048, threshold 0.5, FOV)", color=INK, fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, labelcolor=INK, loc="lower right")
    fig.suptitle("Validation Dice per epoch (dots: best epoch, kept as best.pt); y-axis cut at 0.85", fontsize=10, color=INK)
    fig.tight_layout()
    out = project_path("results/figures")
    out.mkdir(parents=True, exist_ok=True)
    fig.savefig(out / "val_dice_curves_cnns.png", dpi=120, facecolor=SURFACE)
    plt.close(fig)
    print(f"[analysis] wrote {out / 'val_dice_curves_cnns.png'}")


def main() -> None:
    from src.train import results_dir_for
    from src.utils import load_config
    ap = argparse.ArgumentParser(description="Analysis")
    ap.add_argument("command", choices=["curves", "curves_all", "params", "runs", "ablation", "sam", "seam"])
    ap.add_argument("--config")
    ap.add_argument("--seed", type=int)
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final-test", action="store_true", help="Phase 6 only")
    ap.add_argument("--workers", type=int, default=int(__import__("os").environ.get("SLURM_CPUS_PER_TASK", "1")))
    ap.add_argument("--fixed-threshold", type=float, help="ablation: use this grid threshold for every run (sensitivity check)")
    a = ap.parse_args()
    if a.command == "params":
        cmd_params()
        return
    if a.command == "runs":
        cmd_runs(a.split)
        return
    if a.command == "ablation":
        cmd_ablation(a.split, a.final_test, a.fixed_threshold)
        return
    if a.command == "seam":
        cmd_seam(a.split, a.workers)
        return
    if a.command == "sam":
        cmd_sam(a.split)
        return
    if a.command == "curves_all":
        cmd_curves_all()
        return
    if a.config is None or a.seed is None:
        ap.error("curves needs --config and --seed")
    cfg = load_config(a.config)
    out = results_dir_for(cfg, a.seed)
    plot_training_curves(out / "train_log.csv", out / "curves.png",
                         f"{cfg.get('run_name') or cfg['model']['name']}, seed {a.seed}")
    print(f"[analysis] wrote {out / 'curves.png'}")


if __name__ == "__main__":
    main()
