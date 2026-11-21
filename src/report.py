"""Phase 6: test-set tables, statistics and figures (CLAUDE.md Phase 6, items 2-5).

    python -m src.report --final-test           # CPU job; every number from the saved per-image CSVs

Rules (CLAUDE.md Phase 6; fixed in docs/protocol.md before any test prediction):
* model level: each metric per seed (mean over the 200 images), then mean +- sd ACROSS SEEDS;
  image-level spread: sd and median over images of the seed-averaged per-image values;
* per-image analyses (Wilcoxon, bootstrap): each image's metric is first averaged over that
  model's seeds, so each image counts once (3 seeds x 200 images is still 200 images);
* PRIMARY: each run at its own val-selected threshold. SECONDARY: every run at a fixed 0.5
  (threshold-dependent metrics from the saved sweeps; clDice and NSD exist only at the
  selected threshold);
* statistics: paired Wilcoxon signed-rank test (two-sided) per metric, each automatic model vs
  the U-Net baseline; Holm correction within each metric across those comparisons;
* oracle-prompted SAM ("zero-shot, oracle point prompts"): its own table, never ranked against
  or tested with the automatic models;
* mask-resampling references: context rows only, never upper bounds.

Tables -> results/tables/test_*.csv; figures -> results/figures/. Fundus images in figures are
pixelated outside the FOV (burned-in text is never legible).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from src.analysis import INK, MUTED, SURFACE, _style, bootstrap_ci

AUTO = {"unet": "U-Net", "attention_unet": "Attention U-Net", "multiscale_unet": "Multi-Scale U-Net",
        "ms_attention_unet": "Multi-Scale Attention U-Net", "sam_lora_tiled": "SAM LoRA, tiled (main SAM)",
        "sam_lora_whole": "SAM LoRA, whole image",
        "sam_lora_tiled_overlap": "SAM LoRA, tiled, overlapping-tile inference (sensitivity)"}
SHORT = {"unet": "U-Net", "attention_unet": "Att U-Net", "multiscale_unet": "MS U-Net",
         "ms_attention_unet": "MS-Att U-Net", "sam_lora_tiled": "SAM tiled", "sam_lora_whole": "SAM whole",
         "sam_lora_tiled_overlap": "SAM tiled-overlap"}
# Categorical palette, fixed order (dataviz reference palette, light mode; validated as a set:
# all hard checks pass; slots 3-5 sit below 3:1 on the surface, so every chart has value labels or a table).
COLOR = dict(zip(AUTO, ("#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6")))
BASELINE = "unet"
METRICS = ["dice", "iou", "precision", "recall", "cl_recall_thin", "cl_recall_medium", "cl_recall_thick",
           "cldice", "nsd"]
SENS = ["cl_recall_thin_r4_8", "cl_recall_medium_r4_8", "cl_recall_thick_r4_8"]
SWEEP_METRICS = ["dice", "iou", "precision", "recall", "cl_recall_thin", "cl_recall_medium", "cl_recall_thick"] + SENS
CATEGORIES = ("A", "D", "G", "N")
REFERENCES = ("reference_soft_512", "reference_soft_256")
LABEL = {"dice": "Dice", "iou": "IoU", "precision": "Precision", "recall": "Recall",
         "cl_recall_thin": "Thin recall", "cl_recall_medium": "Medium recall", "cl_recall_thick": "Thick recall",
         "cldice": "clDice", "nsd": "NSD"}
# error maps: TP neutral, FP / FN the first two categorical slots (validated all-pairs)
ERR_COLORS = {"bg": (252, 252, 251), "tp": (150, 149, 143), "fp": (235, 104, 52), "fn": (42, 120, 214)}


def root() -> Path:
    from src.utils import project_path
    return project_path("results")


def load(split: str, kind: str = "metrics", models=AUTO) -> pd.DataFrame:
    frames = [pd.read_csv(f) for m in models for f in sorted((root() / "eval" / m).glob(f"seed*/{split}_{kind}.csv"))]
    if not frames:
        raise FileNotFoundError(f"no {split}_{kind}.csv for {list(models)}")
    return pd.concat(frames, ignore_index=True)


def present(df: pd.DataFrame) -> list[str]:
    return [m for m in AUTO if m in set(df.model)]


def seed_avg(df: pd.DataFrame, metrics) -> pd.DataFrame:
    return df.groupby(["model", "image_id", "category"], as_index=False)[metrics].mean()


def model_level(df: pd.DataFrame, metrics, labels=AUTO) -> pd.DataFrame:
    per_seed = df.groupby(["model", "seed"])[metrics].mean()
    pim = seed_avg(df, metrics).set_index("model")
    rows = []
    for m in [x for x in labels if x in set(df.model)]:
        ps = per_seed.loc[m]
        for k in metrics:
            v = pim.loc[[m], k].dropna()
            rows.append({"run": m, "model": labels[m], "metric": k, "n_seeds": len(ps), "mean": float(ps[k].mean()),
                         "sd_seeds": float(ps[k].std(ddof=1)) if len(ps) > 1 else np.nan,
                         "sd_images": float(v.std(ddof=1)), "median": float(v.median()), "n_images": len(v)})
    return pd.DataFrame(rows)


def holm(p) -> np.ndarray:
    p = np.asarray(p, float)
    order, m = np.argsort(p), len(p)
    adj, running = np.empty(m), 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj


def wilcoxon_vs_baseline(df: pd.DataFrame, metrics) -> pd.DataFrame:
    from scipy.stats import wilcoxon
    pim = seed_avg(df, metrics).set_index(["model", "image_id"])
    base = pim.loc[BASELINE]
    rows = []
    for k in metrics:
        block = []
        for m in present(df):
            if m == BASELINE:
                continue
            pair = pd.concat([pim.loc[m][k], base[k]], axis=1, keys=["model", "base"]).dropna()
            d = pair["model"] - pair["base"]
            res = wilcoxon(pair["model"], pair["base"]) if (d != 0).any() else None
            block.append({"metric": k, "run": m, "model": AUTO[m], "vs": AUTO[BASELINE], "n_images": len(d),
                          "mean_diff": float(d.mean()), "median_diff": float(d.median()),
                          "share_better": float((d > 0).mean()),
                          "statistic": float(res.statistic) if res else np.nan, "p": float(res.pvalue) if res else 1.0})
        adj = holm([b["p"] for b in block])
        for b, a in zip(block, adj):
            b["p_holm"] = float(a)
        rows += block
    out = pd.DataFrame(rows)
    out["holm_family"] = "per metric, across the models compared with U-Net"
    return out


def disease(df: pd.DataFrame, metrics) -> pd.DataFrame:
    rows = []
    for c in CATEGORIES:
        sub = df[df.category == c]
        t = model_level(sub, metrics)
        t.insert(2, "category", c)
        rows.append(t)
    return pd.concat(rows, ignore_index=True)


def vessel_measures(df: pd.DataFrame) -> pd.DataFrame:
    from scipy.stats import pearsonr
    cols = ["density_pred", "density_gt", "fd_pred", "fd_gt"]
    pim = seed_avg(df, cols)
    rows = []
    for m in present(df):
        s = pim[pim.model == m]
        for k in ("density", "fd"):
            pair = s[[f"{k}_pred", f"{k}_gt"]].dropna()
            d = pair[f"{k}_pred"] - pair[f"{k}_gt"]
            r, p = pearsonr(pair[f"{k}_pred"], pair[f"{k}_gt"])
            rows.append({"run": m, "model": AUTO[m], "measure": "vessel density" if k == "density" else "fractal dimension",
                         "n_images": len(pair), "mean_pred": float(pair[f"{k}_pred"].mean()),
                         "mean_gt": float(pair[f"{k}_gt"].mean()), "bias_mean_diff": float(d.mean()),
                         "mean_abs_diff": float(d.abs().mean()), "sd_diff": float(d.std(ddof=1)),
                         "loa_low": float(d.mean() - 1.96 * d.std(ddof=1)), "loa_high": float(d.mean() + 1.96 * d.std(ddof=1)),
                         "pearson_r": float(r), "pearson_p": float(p)})
    return pd.DataFrame(rows)


def display(table: pd.DataFrame, tests: pd.DataFrame | None, metrics) -> pd.DataFrame:
    """Wide table: 'mean +- sd_images (median)' per metric; seed sd and Holm p in their own columns."""
    rows = []
    for m, g in table.groupby("run", sort=False):
        g = g.set_index("metric")
        row = {"model": g.model.iloc[0], "n_seeds": int(g.n_seeds.iloc[0]), "n_images": int(g.n_images.max())}
        for k in metrics:
            r = g.loc[k]
            row[LABEL.get(k, k)] = f"{r['mean']:.4f} ± {r['sd_images']:.4f} ({r['median']:.4f})"
            if row["n_seeds"] > 1:
                row[f"{LABEL.get(k, k)} sd seeds"] = f"{r['sd_seeds']:.4f}"
            if tests is not None and m != BASELINE:
                t = tests[(tests.run == m) & (tests.metric == k)]
                if len(t):
                    row[f"{LABEL.get(k, k)} p_holm"] = f"{t.p_holm.iloc[0]:.2g}"
        rows.append(row)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- figures
def _bars(ax, groups, series, values, errors=None, fmt="{:.3f}", label_rot=90):
    """Grouped vertical bars from zero; value labels in ink (not series colour)."""
    n = len(series)
    width = 0.8 / n
    for j, s in enumerate(series):
        xs = np.arange(len(groups)) + (j - (n - 1) / 2) * width
        vals = [values[g][s] for g in groups]
        err = [errors[g][s] for g in groups] if errors is not None else None
        ax.bar(xs, vals, width=width * 0.92, color=COLOR[s], label=SHORT[s], zorder=2,
               yerr=err, error_kw={"elinewidth": 1, "ecolor": INK, "capsize": 0})
        for x, v in zip(xs, vals):
            ax.text(x, v + 0.01, fmt.format(v), ha="center", va="bottom", rotation=label_rot, fontsize=6.5, color=INK)
    ax.set_xticks(np.arange(len(groups)))


def fig_main(table: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    show = ["dice", "cldice", "nsd", "cl_recall_thin"]
    runs = [m for m in AUTO if m in set(table.run)]
    fig, axes = plt.subplots(1, len(show), figsize=(3.4 * len(show), 3.8), facecolor=SURFACE)
    for ax, k in zip(axes, show):
        t = table[table.metric == k].set_index("run")
        vals = {k: {m: t.loc[m, "mean"] for m in runs}}
        errs = {k: {m: (t.loc[m, "sd_seeds"] if t.loc[m, "n_seeds"] > 1 else 0.0) for m in runs}}
        _style(ax)
        _bars(ax, [k], runs, vals, errs)
        ax.set_xticklabels([])
        ax.set_ylim(0, 1.12)
        ax.set_title(LABEL[k], fontsize=10, color=INK)
    axes[0].set_ylabel("test, mean over images (whiskers: sd across seeds)", fontsize=8, color=INK)
    axes[-1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Main metrics on the test set (each run at its val-selected threshold)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_width(table: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    ks = ["cl_recall_thin", "cl_recall_medium", "cl_recall_thick"]
    runs = [m for m in AUTO if m in set(table.run)]
    t = table.set_index(["metric", "run"])
    vals = {k: {m: t.loc[(k, m), "mean"] for m in runs} for k in ks}
    errs = {k: {m: (t.loc[(k, m), "sd_seeds"] if t.loc[(k, m), "n_seeds"] > 1 else 0.0) for m in runs} for k in ks}
    fig, ax = plt.subplots(figsize=(9, 4), facecolor=SURFACE)
    _style(ax)
    _bars(ax, ks, runs, vals, errs)
    ax.set_xticklabels(["thin (r < 3 px)", "medium (3-7 px)", "thick (r > 7 px)"])
    ax.set_ylim(0, 1.15)
    ax.set_ylabel("centreline recall, 1-px tolerance (test)", fontsize=8, color=INK)
    ax.legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    ax.set_title("Width-stratified centreline recall (radius at 2048; whiskers: sd across seeds)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_disease(dis: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    show = ["dice", "cl_recall_thin"]
    runs = [m for m in AUTO if m in set(dis.run)]
    fig, axes = plt.subplots(1, 2, figsize=(13, 4), facecolor=SURFACE)
    for ax, k in zip(axes, show):
        t = dis[dis.metric == k].set_index(["category", "run"])
        vals = {c: {m: t.loc[(c, m), "mean"] for m in runs} for c in CATEGORIES}
        _style(ax)
        _bars(ax, list(CATEGORIES), runs, vals)
        n = {c: int(t.loc[(c, runs[0]), "n_images"]) for c in CATEGORIES}
        names = {"A": "AMD", "D": "DR", "G": "glaucoma", "N": "normal"}
        ax.set_xticklabels([f"{names[c]} ({c}, n={n[c]})" for c in CATEGORIES], fontsize=8)
        ax.set_ylim(0, 1.15)
        ax.set_title(f"{LABEL[k]} by disease category (test)", fontsize=10, color=INK)
    axes[-1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def fig_threshold(out: Path, splits=("val", "test")) -> pd.DataFrame:
    """Thin-vessel recall vs threshold (0.05-0.95) per model, from the saved sweeps
    (per seed: mean over images; then mean over seeds)."""
    import matplotlib.pyplot as plt
    rows = []
    fig, axes = plt.subplots(1, len(splits), figsize=(6 * len(splits), 4), facecolor=SURFACE, sharey=True)
    for ax, split in zip(np.atleast_1d(axes), splits):
        sw = load(split, "sweep")
        curve = sw.groupby(["model", "seed", "threshold"]).cl_recall_thin.mean().groupby(["model", "threshold"]).mean()
        _style(ax)
        for m in present(sw):
            c = curve.loc[m]
            ax.plot(c.index, c.values, color=COLOR[m], lw=2, label=SHORT[m], zorder=3)
            rows += [{"split": split, "run": m, "threshold": float(t), "cl_recall_thin": float(v)} for t, v in c.items()]
        ax.set_xlabel("threshold", fontsize=8, color=INK)
        ax.set_title(f"{split}", fontsize=10, color=INK)
        ax.set_xlim(0.05, 0.95)
    np.atleast_1d(axes)[0].set_ylabel("thin-vessel centreline recall", fontsize=8, color=INK)
    np.atleast_1d(axes)[-1].legend(fontsize=7, frameon=False, loc="upper left", bbox_to_anchor=(1.0, 1.0))
    fig.suptitle("Thin-vessel recall vs threshold (mean over images, then over seeds)", fontsize=10, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)
    return pd.DataFrame(rows)


def fig_bland_altman(df: pd.DataFrame, out: Path) -> None:
    import matplotlib.pyplot as plt
    pim = seed_avg(df, ["density_pred", "density_gt", "fd_pred", "fd_gt"])
    runs = present(df)
    fig, axes = plt.subplots(2, len(runs), figsize=(2.6 * len(runs), 5.4), facecolor=SURFACE, squeeze=False,
                             sharey="row")
    for j, m in enumerate(runs):
        s = pim[pim.model == m]
        for i, (k, name) in enumerate((("density", "vessel density"), ("fd", "fractal dimension"))):
            ax = axes[i, j]
            _style(ax)
            mean = (s[f"{k}_pred"] + s[f"{k}_gt"]) / 2
            diff = s[f"{k}_pred"] - s[f"{k}_gt"]
            ax.scatter(mean, diff, s=9, color=COLOR[m], edgecolors=SURFACE, linewidths=0.4, zorder=3)
            b, sd = diff.mean(), diff.std(ddof=1)
            for y, ls in ((b, "-"), (b - 1.96 * sd, "--"), (b + 1.96 * sd, "--")):
                ax.axhline(y, color=MUTED, lw=1, ls=ls, zorder=2)
            ax.axhline(0, color=INK, lw=0.8, ls=":", zorder=1)
            ax.set_title(f"{SHORT[m]}: {name}" if i == 0 else name, fontsize=8, color=INK)
            ax.tick_params(labelsize=6)
            if j == 0:
                ax.set_ylabel("predicted - ground truth", fontsize=7, color=INK)
            if i == 1:
                ax.set_xlabel("mean of predicted and ground truth", fontsize=7, color=INK)
    fig.suptitle("Bland-Altman, test (solid: bias; dashed: bias +- 1.96 sd; dotted: zero). Image-derived measures, "
                 "not clinically validated", fontsize=9, color=INK)
    fig.tight_layout()
    fig.savefig(out, dpi=130, facecolor=SURFACE)
    plt.close(fig)


def qualitative_ids(cfg: dict, per_category: int = 2, seed: int = 0) -> list[str]:
    from src.data import load_splits
    sp = load_splits(cfg)
    rng = np.random.RandomState(seed)
    ids = []
    for c in CATEGORIES:
        pool = sorted(sp[(sp.split == "test") & (sp.category == c)].image_id)
        ids += list(rng.choice(pool, per_category, replace=False))
    return ids


def thin_window(g, cutoffs, win: int = 256, step: int = 8) -> tuple[int, int]:
    """Top-left corner of the win x win window with the most thin-class GT skeleton pixels."""
    from src.metrics import width_class
    thin = np.zeros(g.skel.shape, np.float64)
    ys, xs = np.nonzero(g.skel)
    thin[ys, xs] = width_class(g.skel_radius, cutoffs) == 0
    ii = np.pad(thin.cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    best, pos = -1.0, (0, 0)
    for y in range(0, thin.shape[0] - win + 1, step):
        for x in range(0, thin.shape[1] - win + 1, step):
            s = ii[y + win, x + win] - ii[y, x + win] - ii[y + win, x] + ii[y, x]
            if s > best:
                best, pos = s, (y, x)
    return pos


def error_rgb(pred: np.ndarray, gt: np.ndarray, fov: np.ndarray) -> np.ndarray:
    out = np.empty(pred.shape + (3,), np.uint8)
    out[:] = ERR_COLORS["bg"]
    p, g = pred & fov, gt & fov
    out[p & g], out[p & ~g], out[~p & g] = ERR_COLORS["tp"], ERR_COLORS["fp"], ERR_COLORS["fn"]
    return out


def fig_qualitative(cfg: dict, out_dir: Path, runs: list[str]) -> list[str]:
    import cv2
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch, Rectangle

    from src.data import _redacted, raw_image_path, raw_mask_path, read_npz_member, read_rgb, read_unchanged
    from src.evaluate import eval_settings, load_fov
    from src.metrics import prepare_gt
    from src.resample import binarize, upsample_prob
    from src.utils import project_path
    s = eval_settings(cfg)
    thr = {m: json.loads((root() / "eval" / m / "seed0" / "val_threshold.json").read_text())["threshold"] for m in runs}
    ids = qualitative_ids(cfg)
    small = 512
    written = []
    for iid in ids:
        gt = read_unchanged(raw_mask_path(cfg, iid)) > 0
        fov = load_fov(cfg, iid)
        g = prepare_gt(gt, fov)
        y0, x0 = thin_window(g, s["cutoffs"])
        crop = (slice(y0, y0 + 256), slice(x0, x0 + 256))
        img = _redacted(cfg, read_rgb(raw_image_path(cfg, iid)))          # pixelated outside the FOV
        panels = [("image (outside FOV pixelated)", img), ("ground truth", error_rgb(gt, gt, fov))]
        for m in runs:
            prob = read_npz_member(project_path("outputs") / m / "seed0" / "test" / "probs.npz", iid).astype(np.float32)
            pred = binarize(upsample_prob(prob, s["native"]), thr[m])
            panels.append((f"{SHORT[m]} (t={thr[m]:g})", error_rgb(pred, gt, fov)))
        fig, axes = plt.subplots(2, len(panels), figsize=(2.05 * len(panels), 4.6), facecolor=SURFACE)
        for j, (title, arr) in enumerate(panels):
            axes[0, j].imshow(cv2.resize(arr, (small, small), interpolation=cv2.INTER_AREA))
            axes[1, j].imshow(arr[crop], interpolation="nearest")
            axes[0, j].set_title(title, fontsize=7, color=INK)
            for a in axes[:, j]:
                a.set_xticks([])
                a.set_yticks([])
        sc = small / s["native"]
        axes[0, 0].add_patch(Rectangle((x0 * sc, y0 * sc), 256 * sc, 256 * sc, fill=False, ec=INK, lw=1))
        axes[1, 0].set_ylabel(f"native 256x256\n(y{y0} x{x0}, most\nthin-vessel pixels)", fontsize=7, color=INK)
        axes[0, 0].set_ylabel(f"{iid}\n(seed-0 runs)", fontsize=7, color=INK)
        handles = [Patch(color=np.array(ERR_COLORS[k]) / 255, label=lab) for k, lab in
                   (("tp", "true positive"), ("fp", "false positive"), ("fn", "false negative"))]
        fig.legend(handles=handles, loc="lower center", ncol=3, fontsize=7, frameon=False)
        fig.tight_layout(rect=(0, 0.05, 1, 1))
        path = out_dir / f"test_qualitative_{iid}.png"
        fig.savefig(path, dpi=120, facecolor=SURFACE)
        plt.close(fig)
        written.append(path.name)
    return written


# --------------------------------------------------------------------------- main
def main() -> None:
    from src.analysis import cmd_ablation
    from src.utils import environment_info, load_config
    ap = argparse.ArgumentParser(description="Phase 6 test-set report")
    ap.add_argument("--final-test", action="store_true")
    ap.add_argument("--no-qualitative", action="store_true")
    a = ap.parse_args()
    if not a.final_test:
        raise SystemExit("Phase 6 test analysis: pass --final-test.")
    cfg = load_config("configs/base.yaml")
    tables, figs = root() / "tables", root() / "figures"
    tables.mkdir(parents=True, exist_ok=True)
    figs.mkdir(parents=True, exist_ok=True)

    df = load("test")
    runs = present(df)
    missing = [m for m in AUTO if m not in runs]
    # primary
    main_t = model_level(df, METRICS + SENS)
    tests = wilcoxon_vs_baseline(df, METRICS + SENS)
    main_t.to_csv(tables / "test_main.csv", index=False)
    tests.to_csv(tables / "test_wilcoxon.csv", index=False)
    display(main_t, tests, METRICS).to_csv(tables / "test_main_display.csv", index=False)
    width = main_t[main_t.metric.isin(["cl_recall_thin", "cl_recall_medium", "cl_recall_thick"] + SENS)].copy()
    width.insert(3, "cutoffs_radius_px", np.where(width.metric.str.endswith("_r4_8"), "4/8 (sensitivity)", "3/7"))
    width.to_csv(tables / "test_width.csv", index=False)
    dis = disease(df, METRICS)
    dis.to_csv(tables / "test_disease.csv", index=False)
    vm = vessel_measures(df)
    vm.to_csv(tables / "test_vessel_measures.csv", index=False)
    # secondary: fixed 0.5 for every run
    sw = load("test", "sweep")
    sw05 = sw[sw.threshold.round(4) == 0.5]
    fixed_t = model_level(sw05, SWEEP_METRICS)
    fixed_tests = wilcoxon_vs_baseline(sw05, SWEEP_METRICS)
    fixed_t.to_csv(tables / "test_main_fixed_t0.5.csv", index=False)
    fixed_tests.to_csv(tables / "test_wilcoxon_fixed_t0.5.csv", index=False)
    # 2x2 ablation (CNNs), primary and fixed 0.5
    cmd_ablation("test", final_test=True)
    cmd_ablation("test", final_test=True, fixed_threshold=0.5)
    # oracle (separate) and references (context)
    choice = json.loads((root() / "eval" / "sam_oracle_choice.json").read_text())
    k = choice["chosen_k"]
    orc = pd.read_csv(root() / "eval" / f"sam_oracle_k{k}" / "seed0" / "test_metrics.csv")
    o = model_level(orc, METRICS + ["density_pred", "density_gt"], labels={f"sam_oracle_k{k}": f"SAM zero-shot, oracle point prompts (K={k})"})
    o.to_csv(tables / "test_sam_oracle.csv", index=False)
    refs = pd.concat([pd.read_csv(root() / "eval" / r / "test_metrics.csv").assign(seed=0) for r in REFERENCES])
    model_level(refs, METRICS + SENS, labels={r: f"{r} (context; not an upper bound)" for r in REFERENCES}).to_csv(
        tables / "test_references.csv", index=False)
    # figures
    fig_main(main_t, figs / "test_main_metrics.png")
    fig_width(main_t, figs / "test_width_recall.png")
    fig_disease(dis, figs / "test_disease.png")
    fig_threshold(figs / "thin_recall_vs_threshold.png").to_csv(tables / "thin_recall_vs_threshold.csv", index=False)
    fig_bland_altman(df, figs / "test_bland_altman.png")
    qual = [] if a.no_qualitative else fig_qualitative(cfg, figs, [m for m in runs])
    info = {"runs": {m: sorted(int(s) for s in df[df.model == m].seed.unique()) for m in runs}, "missing": missing,
            "oracle_k": k, "qualitative_images": qualitative_ids(cfg), "qualitative_figures": qual,
            "environment": {kk: v for kk, v in environment_info().items() if kk in ("time", "host", "slurm_job_id", "git")}}
    with open(tables / "test_report_info.json", "w") as f:
        json.dump(info, f, indent=1)
    show = main_t[main_t.metric.isin(["dice", "cl_recall_thin", "cldice", "nsd"])]
    print(show.pivot(index="model", columns="metric", values="mean").round(4).to_string())
    print(tests[tests.metric.isin(["dice", "cl_recall_thin"])][["metric", "model", "mean_diff", "p_holm"]].round(4).to_string(index=False))
    print(f"[report] runs {info['runs']}; missing {missing}; wrote tables to {tables} and figures to {figs}")


if __name__ == "__main__":
    main()
