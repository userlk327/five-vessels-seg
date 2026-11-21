"""Evaluation (Phase 2 onwards).

Scoring saved probability maps (every model, every seed):

    python -m src.evaluate score --config configs/unet.yaml --seed 0 --split val
    python -m src.evaluate score --config configs/unet.yaml --seed 0 --split test --final-test  # Phase 6 only
    python -m src.evaluate reference --split val     # mask-resampling reference (context, NOT an upper bound)

  1. Load outputs/<run>/seed<seed>/<split>/probs.npz (float16 sigmoid outputs at the model's
     output size, one member per image; exactly the split's image ids, checked).
  2. Upsample each map bilinearly (half-pixel centres) to 2048.
  3. val: sweep the threshold grid (0.05-0.95, step 0.05) and choose the threshold with
     the highest mean Dice over the val images (ties: the lowest) -> val_threshold.json.
     test: use that run's own val_threshold.json; nothing is chosen on test.
  4. Score every image at that threshold against the original 2048 ground truth, inside
     its FOV (src/metrics.py) -> one CSV row per (model, seed, split, image).
  The per-threshold sweep (Dice, precision, recall, width-stratified recall, also with the
  4 / 8 cut-offs) is saved for val and test as well. It is reporting only (e.g. the
  Phase 6 figure of thin-vessel recall against the threshold); the test sweep is never
  used to choose anything.
  Output: results/eval/<run>/seed<seed>/{val,test}_{metrics,sweep}.csv, val_threshold.json,
  {val,test}_summary.json.

Phase 2 checks:

    python -m src.evaluate widths       # radius distribution of the TRAINING ground truths (inside the
                                        # FOV) -> results/checks/width_radius_*; cut-offs approved: 3 / 7
    python -m src.evaluate widths_plot  # redraw that figure from the saved files
    python -m src.evaluate maskresize   # training-mask resizing comparison (30 training masks, 512 and 1024)
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import cv2
import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from src.data import (CHECKS_DIR, load_splits, npz_keys, raw_mask_path, read_npz_member, read_unchanged,
                      sha256_file, write_npz_archive)
from src.metrics import WIDTH_CLASSES, overlap_counts, skeleton, width_class
from src.utils import (config_hash, environment_info, load_config, portable_path, prob_archive_path, project_path,
                       run_dir_for)


def fov_cache_dir(cfg: dict) -> Path:
    return project_path(cfg["paths"]["cache_dir"]) / "fov_2048"


def load_fov(cfg: dict, image_id: str) -> np.ndarray:
    """Cached FOV mask (``python -m src.fov all``) at 2048."""
    m = cv2.imread(str(fov_cache_dir(cfg) / f"{image_id}.png"), cv2.IMREAD_UNCHANGED)
    if m is None:
        raise FileNotFoundError(f"no cached FOV for {image_id}; run python -m src.fov all")
    return m > 0


def _pool(workers: int) -> ProcessPoolExecutor:
    return ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn"))


# --------------------------------------------------------------------------- widths
def _skeleton_radii(args) -> tuple[str, np.ndarray]:
    """EDT radius (px at 2048) at every GT skeleton pixel inside the FOV, as in
    metrics.prepare_gt (skeleton and EDT of the full GT, restricted to the FOV)."""
    image_id, cfg = args
    cv2.setNumThreads(1)
    gt = read_unchanged(raw_mask_path(cfg, image_id)) > 0
    sk = skeleton(gt) & load_fov(cfg, image_id)
    return image_id, ndi.distance_transform_edt(gt)[sk].astype(np.float32)


def cmd_widths(cfg: dict, workers: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    sp = load_splits(cfg)
    ids = sp.loc[sp.split == "train", "image_id"].tolist()
    with _pool(workers) as ex:
        res = dict(ex.map(_skeleton_radii, [(i, cfg) for i in ids], chunksize=4))
    r = np.concatenate([res[i] for i in ids])
    lo, hi = cfg["evaluation"]["width_classes_radius_px"]["thin_below"], cfg["evaluation"]["width_classes_radius_px"]["thick_above"]

    def shares(a, b):
        return {"thin": float((r < a).mean()), "medium": float(((r >= a) & (r <= b)).mean()), "thick": float((r > b).mean())}

    per_image = pd.DataFrame([{"image_id": i, "category": i.split("_")[2], "n_skel": len(res[i]),
                               **{f"share_{k}": v for k, v in
                                  {"thin": (res[i] < lo).mean(), "medium": ((res[i] >= lo) & (res[i] <= hi)).mean(),
                                   "thick": (res[i] > hi).mean()}.items()}} for i in ids])
    per_image.to_csv(CHECKS_DIR / "width_radius_per_image.csv", index=False)
    vals, counts = np.unique(np.round(r, 3), return_counts=True)
    pd.DataFrame({"radius_px": vals, "skeleton_px": counts}).to_csv(CHECKS_DIR / "width_radius_hist.csv", index=False)
    alternatives = {f"{a}/{b}": shares(a, b) for a, b in [(lo, hi), (2, 5), (2.5, 6), (3, 6), (4, 8)]}
    summary = {"images": len(ids), "split": "train", "skeleton_px": int(len(r)),
               "definition": "radius = Euclidean distance transform of the full GT at each GT skeleton pixel inside the FOV (px at 2048); "
                             "thin: r < lo, medium: lo <= r <= hi, thick: r > hi",
               "proposed_cutoffs": {"thin_below": lo, "thick_above": hi},
               "shares_proposed": shares(lo, hi),
               "shares_alternatives": alternatives,
               "radius_quantiles": {str(p): round(float(np.quantile(r, p)), 3) for p in (0.1, 0.25, 0.5, 0.75, 0.9, 0.99)},
               "per_image_share_quantiles": {k: {str(p): round(float(per_image[f"share_{k}"].quantile(p)), 4)
                                                 for p in (0.05, 0.5, 0.95)} for k in ("thin", "medium", "thick")},
               "shares_by_category": {c: {k: round(float(np.average(g[f"share_{k}"], weights=g.n_skel)), 4)
                                          for k in ("thin", "medium", "thick")} for c, g in per_image.groupby("category")},
               "approx_width_px": "a straight vessel of width w has centre radius (w + 1) / 2, so r < 3 ~ w <= 4 px at 2048 "
                                  "(<= 1 px at 512), r > 7 ~ w >= 14 px"}
    with open(CHECKS_DIR / "width_radius_summary.json", "w") as f:
        json.dump(summary, f, indent=1)
    print(json.dumps(summary, indent=1))

    cmd_widths_plot(cfg)
    print("[widths] wrote width_radius_summary.json, width_radius_hist.csv/.png, width_radius_per_image.csv")


# --------------------------------------------------------------------------- training-mask resizing comparison
def maskresize_sample(cfg: dict, counts=(("A", 8), ("D", 7), ("G", 8), ("N", 7)), seed: int = 0) -> list[str]:
    """30 training images, stratified by category, fixed seed."""
    sp = load_splits(cfg)
    rng = np.random.RandomState(seed)
    ids = []
    for cat, n in counts:
        pool = sorted(sp[(sp.split == "train") & (sp.category == cat)].image_id, key=lambda i: int(i.split("_")[1]))
        ids += sorted(rng.choice(pool, n, replace=False).tolist(), key=lambda i: int(i.split("_")[1]))
    return ids


def _maskresize_one(args) -> list[dict]:
    from src.metrics import centerline_recall, cldice, overlap, prepare_gt
    from src.resample import MASK_METHODS, binarize, downsample_mask, upsample_prob
    image_id, cfg, cutoffs, resolutions = args
    cv2.setNumThreads(1)
    gt = read_unchanged(raw_mask_path(cfg, image_id)) > 0
    fov = load_fov(cfg, image_id)
    g = prepare_gt(gt, fov)
    rows = []
    for res in resolutions:
        for method in MASK_METHODS:
            target = downsample_mask(gt, res, method)
            up = upsample_prob(target, gt.shape[0])
            for thr in ((0.5, 0.25) if method == "soft" else (0.5,)):
                pred = binarize(up, thr)
                m = overlap(pred & fov, g.gt)
                m.update(centerline_recall(pred, g, cutoffs))
                m.update(cldice(pred, g))
                rows.append({"image_id": image_id, "resolution": res, "method": method,
                             "threshold_after_upsampling": thr,
                             "target_mean": float(target.mean()), **m})
    return rows


def cmd_maskresize(cfg: dict, workers: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from src.metrics import prepare_gt
    from src.resample import MASK_METHODS, binarize, downsample_mask, upsample_prob

    ev = cfg["evaluation"]["width_classes_radius_px"]
    cutoffs = (ev["thin_below"], ev["thick_above"])
    resolutions = (512, 1024)
    ids = maskresize_sample(cfg)
    with _pool(workers) as ex:
        rows = [r for rs in ex.map(_maskresize_one, [(i, cfg, cutoffs, resolutions) for i in ids]) for r in rs]
    df = pd.DataFrame(rows)
    df["category"] = df.image_id.str[-1]
    df.to_csv(CHECKS_DIR / "mask_resize_per_image.csv", index=False)
    cols = ["dice", "precision", "recall", "cl_recall_thin", "cl_recall_medium", "cl_recall_thick", "cldice"]
    keys = ["resolution", "method", "threshold_after_upsampling"]
    order = {m: k for k, m in enumerate(MASK_METHODS)}
    mean = df.groupby(keys)[cols].mean()
    sd = df.groupby(keys)[cols].std()
    summ = mean.round(4).astype(str) + " ± " + sd.round(4).astype(str)
    summ = summ.reset_index().sort_values(["resolution", "method", "threshold_after_upsampling"],
                                          key=lambda c: c.map(order) if c.name == "method" else (-c if c.name == "threshold_after_upsampling" else c))
    summ.to_csv(CHECKS_DIR / "mask_resize_summary.csv", index=False)
    mean.reset_index().to_csv(CHECKS_DIR / "mask_resize_means.csv", index=False)
    with open(CHECKS_DIR / "mask_resize_sample.json", "w") as f:
        json.dump({"images": ids, "n": len(ids), "categories": pd.Series([i[-1] for i in ids]).value_counts().to_dict(),
                   "selection": "training split, stratified A 8 / D 7 / G 8 / N 7, RandomState(0)",
                   "reconstruction": "downsample with the method, bilinear upsampling to 2048 (half-pixel centres), "
                                     ">= threshold (0.5; soft also 0.25 as an illustration), scored inside the FOV",
                   "cutoffs": cutoffs}, f, indent=1)
    pd.set_option("display.width", 250)
    print(mean.round(3).to_string())

    # crop figure: the 256 window with most thin skeleton pixels in the first sampled image
    i0 = ids[0]
    gt = read_unchanged(raw_mask_path(cfg, i0)) > 0
    fov = load_fov(cfg, i0)
    g = prepare_gt(gt, fov)
    thin = np.zeros(gt.shape, np.float64)
    ys, xs = np.nonzero(g.skel)
    sel = g.skel_radius < cutoffs[0]
    thin[ys[sel], xs[sel]] = 1
    ii = thin.cumsum(0).cumsum(1)
    best = max(((ii[y + 255, x + 255] - (ii[y - 1, x + 255] if y else 0) - (ii[y + 255, x - 1] if x else 0)
                 + (ii[y - 1, x - 1] if x and y else 0), y, x) for y in range(0, 1793, 32) for x in range(0, 1793, 32)))
    _, y0, x0 = best
    win = (slice(y0, y0 + 256), slice(x0, x0 + 256))
    variants = [(m, 0.5) for m in MASK_METHODS] + [("soft", 0.25)]
    fig, axes = plt.subplots(4, len(variants) + 1, figsize=(2.6 * (len(variants) + 1), 11.2))
    for ax in axes.flat:
        ax.axis("off")
    for r, res in enumerate(resolutions):
        f = gt.shape[0] // res
        axes[2 * r, 0].imshow(gt[win], cmap="gray", interpolation="nearest")
        axes[2 * r, 0].set_title(f"2048 ground truth\n(row: targets at {res})", fontsize=8)
        axes[2 * r + 1, 0].set_title(f"reconstructions from {res}\nwhite TP, red missed, cyan extra", fontsize=8)
        for c, (m, thr) in enumerate(variants, start=1):
            t = downsample_mask(gt, res, m)
            tw = t[y0 // f:(y0 + 256) // f, x0 // f:(x0 + 256) // f]
            axes[2 * r, c].imshow(np.kron(tw, np.ones((f, f))), cmap="gray", vmin=0, vmax=1, interpolation="nearest")
            axes[2 * r, c].set_title(f"{m} target at {res}" + (" (same as soft)" if (m, thr) == ("soft", 0.25) else ""), fontsize=8)
            pred = binarize(upsample_prob(t, gt.shape[0]), thr)[win]
            gw, fw = g.gt[win], fov[win]
            err = np.zeros((256, 256, 3), np.uint8)
            err[pred & gw] = (255, 255, 255); err[gw & ~pred] = (255, 0, 0); err[pred & ~gw & fw] = (0, 255, 255)
            row = df[(df.image_id == i0) & (df.resolution == res) & (df.method == m) & (df.threshold_after_upsampling == thr)].iloc[0]
            axes[2 * r + 1, c].imshow(err, interpolation="nearest")
            axes[2 * r + 1, c].set_title(f"{m}, thr {thr}: Dice {row.dice:.3f}\nthin recall {row.cl_recall_thin:.3f}", fontsize=8)
    fig.suptitle(f"{i0}: native 256x256 crop (y{y0} x{x0}, the window with most thin-class skeleton pixels). Targets are "
                 "drawn as blocks; reconstructions = bilinear upsampling to 2048, then threshold.", fontsize=9)
    fig.tight_layout(rect=(0, 0, 1, 0.97))
    fig.savefig(CHECKS_DIR / "mask_resize_crop.png", dpi=100)
    plt.close(fig)
    print(f"[maskresize] {len(ids)} images; crop {i0} y{y0} x{x0}; wrote mask_resize_per_image.csv, "
          "mask_resize_summary.csv, mask_resize_means.csv, mask_resize_sample.json, mask_resize_crop.png")


def cmd_widths_plot(cfg: dict, workers: int = 1) -> None:
    """Figure from the saved width files (no recomputation)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    summ = json.loads((CHECKS_DIR / "width_radius_summary.json").read_text())
    hist = pd.read_csv(CHECKS_DIR / "width_radius_hist.csv")
    per_image = pd.read_csv(CHECKS_DIR / "width_radius_per_image.csv")
    lo, hi = summ["proposed_cutoffs"]["thin_below"], summ["proposed_cutoffs"]["thick_above"]
    s = summ["shares_proposed"]
    share = hist.skeleton_px / hist.skeleton_px.sum()
    fig, ax = plt.subplots(1, 2, figsize=(14, 4.6), gridspec_kw={"width_ratios": [1.6, 1]})
    colors = np.where(hist.radius_px < lo, "tab:blue", np.where(hist.radius_px <= hi, "tab:orange", "tab:green"))
    ax[0].bar(hist.radius_px, share, width=0.1, color=colors)
    for x in (lo, hi):
        ax[0].axvline(x, ls="--", color="k", lw=0.8)
    ax[0].set_xlim(0, 20)
    ax[0].set_xlabel("local radius r at a skeleton pixel (px at 2048; EDT of the ground truth)")
    ax[0].set_ylabel("share of skeleton pixels")
    ax[0].set_title(f"Training ground truth: {summ['images']} images, {summ['skeleton_px']:,} skeleton pixels inside the FOV\n"
                    f"thin r<{lo}: {s['thin']:.1%}   medium {lo}-{hi}: {s['medium']:.1%}   thick r>{hi}: {s['thick']:.1%}",
                    fontsize=9)
    ax[1].boxplot([per_image[f"share_{k}"] for k in ("thin", "medium", "thick")], tick_labels=["thin", "medium", "thick"])
    ax[1].set_ylabel("share of the image's skeleton pixels")
    ax[1].set_title(f"Per image (cut-offs {lo} / {hi})", fontsize=9)
    fig.tight_layout()
    fig.savefig(CHECKS_DIR / "width_radius_hist.png", dpi=110)
    plt.close(fig)
    print(f"[widths] wrote {CHECKS_DIR / 'width_radius_hist.png'}")


# --------------------------------------------------------------------------- evaluation pipeline
def threshold_grid(cfg: dict) -> list[float]:
    """0.05, 0.10, ..., 0.95 as Python floats, so every comparison is the same ``prob >= t``."""
    g = cfg["evaluation"]["threshold_grid"]
    n = int(round((g["stop"] - g["start"]) / g["step"])) + 1
    return [round(g["start"] + k * g["step"], 6) for k in range(n)]


def eval_settings(cfg: dict) -> dict:
    ev = cfg["evaluation"]
    w, ws = ev["width_classes_radius_px"], ev.get("width_sensitivity_radius_px")
    return {"native": int(cfg["data"]["native_resolution"]),
            "cutoffs": (w["thin_below"], w["thick_above"]),
            "sens_cutoffs": (ws["thin_below"], ws["thick_above"]) if ws else None,
            "center_tol": int(ev["centerline_tolerance_px"]), "nsd_tol": float(ev["nsd_tolerance_px"]),
            "box_sizes": tuple(ev["fractal_box_sizes"])}


def sweep_metrics(up: np.ndarray, g, thresholds, cutoffs, tol: int = 1, sens_cutoffs=None) -> list[dict]:
    """Threshold-dependent metrics of one upsampled probability map at every threshold,
    identical to ``compute_all(up >= t, ...)`` for Dice, IoU, precision, recall and the
    width-stratified centerline recall (tests/test_evaluate.py). Centerline hits come from
    a (2 tol + 1)^2 max filter of the probabilities inside the FOV, which equals
    thresholding first and dilating by tol px afterwards."""
    if up.shape != g.fov.shape:
        raise ValueError(f"probability map {up.shape} vs ground truth {g.fov.shape}")
    up = up.astype(np.float32, copy=False)
    inside, p_gt, n_gt = up[g.fov], up[g.gt], int(g.gt.sum())
    masked = np.where(g.fov, up, np.float32(-1.0))
    mx = ndi.maximum_filter(masked, size=2 * tol + 1, mode="constant", cval=-1.0) if tol > 0 else masked
    m_skel = mx[g.skel]
    classes = {"": width_class(g.skel_radius, cutoffs)}
    if sens_cutoffs is not None:
        classes["_r{:g}_{:g}".format(*sens_cutoffs)] = width_class(g.skel_radius, sens_cutoffs)
    rows = []
    for t in thresholds:
        t = float(t)
        row = {"threshold": t, **overlap_counts(int((p_gt >= t).sum()), int((inside >= t).sum()), n_gt)}
        hit = m_skel >= t
        for tag, cls in classes.items():
            for k, name in enumerate(WIDTH_CLASSES):
                sel = cls == k
                row[f"cl_recall_{name}{tag}"] = float(hit[sel].mean()) if sel.any() else float("nan")
        row["cl_recall_all"] = float(hit.mean()) if hit.size else float("nan")
        rows.append(row)
    return rows


def select_threshold(sweep: pd.DataFrame) -> tuple[float, pd.Series]:
    """Highest mean Dice over the images; on an exact tie the lowest threshold."""
    curve = sweep.groupby("threshold").dice.mean().sort_index()
    return float(curve.idxmax()), curve


def check_prob_map(prob: np.ndarray, image_id: str) -> None:
    if prob.dtype != np.float16 or prob.ndim != 2 or prob.shape[0] != prob.shape[1]:
        raise ValueError(f"{image_id}: probability map must be a square float16 array, got {prob.dtype} {prob.shape}")
    if not np.isfinite(prob).all() or prob.min() < 0 or prob.max() > 1:
        raise ValueError(f"{image_id}: probabilities outside [0, 1]")


def _score_one(args) -> tuple[str, dict]:
    from src.metrics import compute_all, prepare_gt
    from src.resample import binarize, upsample_prob
    image_id, cfg, archive, threshold, do_sweep = args
    cv2.setNumThreads(1)
    s = eval_settings(cfg)
    prob = read_npz_member(archive, image_id)
    check_prob_map(prob, image_id)
    gt = read_unchanged(raw_mask_path(cfg, image_id))
    if gt.shape != (s["native"], s["native"]):
        raise ValueError(f"{image_id}: ground truth {gt.shape}")
    g = prepare_gt(gt > 0, load_fov(cfg, image_id))
    up = upsample_prob(prob.astype(np.float32), s["native"])
    out = {"prob_resolution": int(prob.shape[0])}
    if do_sweep:
        out["sweep"] = sweep_metrics(up, g, threshold_grid(cfg), s["cutoffs"], s["center_tol"], s["sens_cutoffs"])
    if threshold is not None:
        out["metrics"] = compute_all(binarize(up, threshold), g, s["cutoffs"], s["center_tol"], s["nsd_tol"],
                                     s["box_sizes"], extras=True, sensitivity_cutoffs=s["sens_cutoffs"])
    return image_id, out


def _summary(df: pd.DataFrame) -> dict:
    skip = {"seed", "threshold", "prob_resolution"}
    cols = [c for c in df.columns if c not in skip and not c.startswith("n_skel") and pd.api.types.is_numeric_dtype(df[c])]
    return {c: {"mean": float(df[c].mean()), "sd": float(df[c].std()), "median": float(df[c].median()),
                "n": int(df[c].notna().sum())} for c in cols}


def score_run(cfg: dict, *, model: str, seed: int | None, split: str, archive: Path, out_dir: Path,
              workers: int, final_test: bool = False, extra_meta: dict | None = None) -> pd.DataFrame:
    """Steps 2-4 of the module docstring for one run (model + seed) and one split."""
    if split not in ("val", "test"):
        raise ValueError("split must be val or test")
    if split == "test" and not final_test:
        raise SystemExit("Test predictions are scored once, in Phase 6: pass --final-test.")
    archive = Path(archive)
    sp = load_splits(cfg)
    expected = sp.loc[sp.split == split, "image_id"].tolist()
    keys = npz_keys(archive)
    missing, extra = sorted(set(expected) - set(keys)), sorted(set(keys) - set(expected))
    if missing or extra:
        raise ValueError(f"{archive}: {len(missing)} {split} images missing {missing[:5]}, "
                         f"{len(extra)} not in the {split} split {extra[:5]}")
    out_dir.mkdir(parents=True, exist_ok=True)
    category = dict(zip(sp.image_id, sp.category))
    prov = {"model": model, "seed": seed, "split": split, "n_images": len(expected),
            "probabilities": portable_path(archive), "probabilities_sha256": sha256_file(archive)[:16],
            "config_hash": config_hash(cfg), "evaluation": cfg["evaluation"],
            "environment": {k: v for k, v in environment_info().items() if k in ("time", "host", "git", "slurm_job_id")},
            **(extra_meta or {})}

    def run(threshold, do_sweep):
        with _pool(workers) as ex:
            res = dict(ex.map(_score_one, [(i, cfg, archive, threshold, do_sweep) for i in expected]))
        return [(i, res[i]) for i in expected]

    def frame(rows):
        df = pd.DataFrame(rows)
        df.insert(0, "category", [category[i] for i in df.image_id])
        for k, v in (("split", split), ("seed", seed), ("model", model)):
            df.insert(0, k, v)
        return df

    print(f"[score] {model} seed {seed} {split}: {len(expected)} images from {archive}", flush=True)
    if split == "val":
        res = run(None, True)
        sweep = frame([{"image_id": i, "prob_resolution": r["prob_resolution"], **row} for i, r in res for row in r["sweep"]])
        threshold, curve = select_threshold(sweep)
        with open(out_dir / "val_threshold.json", "w") as f:
            json.dump({**prov, "threshold": threshold, "mean_dice_at_threshold": float(curve.loc[threshold]),
                       "rule": "highest mean Dice over the val images on the grid; ties -> lowest threshold",
                       "mean_dice_by_threshold": {f"{t:g}": float(d) for t, d in curve.items()}}, f, indent=1)
        res = run(threshold, False)
    else:
        tfile = out_dir / "val_threshold.json"
        if not tfile.exists():
            raise SystemExit(f"{tfile} not found: score the val split of this run first")
        tinfo = json.loads(tfile.read_text())
        if (tinfo["model"], tinfo["seed"]) != (model, seed):
            raise ValueError(f"{tfile} belongs to {tinfo['model']} seed {tinfo['seed']}")
        threshold = float(tinfo["threshold"])
        res = run(threshold, True)
        sweep = frame([{"image_id": i, "prob_resolution": r["prob_resolution"], **row} for i, r in res for row in r["sweep"]])
    sweep.to_csv(out_dir / f"{split}_sweep.csv", index=False)
    metrics = frame([{"image_id": i, "threshold": threshold, "prob_resolution": r["prob_resolution"], **r["metrics"]}
                     for i, r in res])
    metrics.to_csv(out_dir / f"{split}_metrics.csv", index=False)
    sweep_means = sweep.drop(columns=["model", "seed", "split", "category", "image_id", "prob_resolution"]) \
                       .groupby("threshold").mean()
    with open(out_dir / f"{split}_summary.json", "w") as f:
        json.dump({**prov, "threshold": threshold, "at_threshold": _summary(metrics),
                   "sweep_means_by_threshold": {f"{t:g}": {k: float(v) for k, v in r.items()}
                                                for t, r in sweep_means.iterrows()}}, f, indent=1)
    show = ["dice", "iou", "precision", "recall", "cl_recall_thin", "cl_recall_medium", "cl_recall_thick", "cldice", "nsd"]
    print(f"[score] threshold {threshold:g}; means over {len(metrics)} images:\n"
          + metrics[show].mean().round(4).to_string())
    print(f"[score] wrote {out_dir}/{split}_metrics.csv, {split}_sweep.csv, {split}_summary.json"
          + (", val_threshold.json" if split == "val" else ""), flush=True)
    return metrics


def cmd_score(cfg: dict, workers: int, split: str, seed: int, final_test: bool) -> None:
    run_dir = run_dir_for(cfg, seed)
    model = cfg.get("run_name") or cfg["model"]["name"]
    score_run(cfg, model=model, seed=seed, split=split, archive=prob_archive_path(run_dir, split),
              out_dir=project_path(cfg["paths"]["results_dir"]) / "eval" / model / f"seed{seed}",
              workers=workers, final_test=final_test)


# --------------------------------------------------------------------------- mask-resampling reference
def reference_specs(cfg: dict) -> list[dict]:
    """The ground truth pushed through each model family's resolution path. Context for
    thin-vessel results only: it shows what resizing alone destroys. NOT an upper bound."""
    d = cfg["data"]
    method, cnn, dec, sam = d["mask_resize"], int(d["cnn_resolution"]), int(d["sam_decoder_resolution"]), int(d["sam_resolution"])
    return [{"name": f"reference_{method}_{cnn}", "size": cnn, "stored_at": cnn,
             "path": f"2048 GT -> {method} at {cnn} -> float16 -> bilinear to 2048 -> threshold (as CNN predictions)"},
            {"name": f"reference_{method}_{dec}", "size": dec, "stored_at": sam,
             "path": f"2048 GT -> {method} at {dec} -> bilinear to {sam} (SAM postprocessing) -> float16 -> "
                     f"bilinear to 2048 -> threshold (as SAM predictions)"}]


def _reference_map(args) -> tuple[str, np.ndarray]:
    from src.resample import downsample_mask, upsample_prob
    image_id, cfg, size, stored_at, method = args
    cv2.setNumThreads(1)
    t = downsample_mask(read_unchanged(raw_mask_path(cfg, image_id)) > 0, size, method)
    if stored_at != size:
        t = upsample_prob(t, stored_at)
    return image_id, np.clip(t, 0, 1).astype(np.float16)


def cmd_reference(cfg: dict, workers: int, split: str, final_test: bool, only: str = "all") -> None:
    if split == "test" and not final_test:
        raise SystemExit("The test-set reference is computed in Phase 6: pass --final-test.")
    sp = load_splits(cfg)
    ids = sp.loc[sp.split == split, "image_id"].tolist()
    specs = reference_specs(cfg)
    specs = {"all": specs, "cnn": specs[:1], "sam_decoder": specs[1:]}[only]
    for spec in specs:
        with _pool(workers) as ex:
            maps = dict(ex.map(_reference_map, [(i, cfg, spec["size"], spec["stored_at"], cfg["data"]["mask_resize"])
                                                for i in ids]))
        archive = project_path(cfg["paths"]["outputs_dir"]) / "reference" / spec["name"] / split / "probs.npz"
        write_npz_archive(archive, {i: maps[i] for i in ids})
        score_run(cfg, model=spec["name"], seed=None, split=split, archive=archive,
                  out_dir=project_path(cfg["paths"]["results_dir"]) / "eval" / spec["name"], workers=workers,
                  final_test=final_test,
                  extra_meta={"reference": spec, "note": "mask-resampling reference: context only, not an upper bound"})


def main() -> None:
    ap = argparse.ArgumentParser(description="Evaluation")
    ap.add_argument("command", choices=["score", "reference", "widths", "widths_plot", "maskresize"])
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--seed", type=int, help="score: the run's training seed")
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--final-test", action="store_true", help="Phase 6 only: allow scoring the test split")
    ap.add_argument("--only", choices=["all", "cnn", "sam_decoder"], default="all",
                    help="reference: just the cnn_resolution path or just SAM's 256 decoder path")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    a = ap.parse_args()
    cfg = load_config(a.config)
    cv2.setNumThreads(int(os.environ.get("OMP_NUM_THREADS", "1")))
    if a.command == "score":
        if a.seed is None:
            ap.error("score needs --seed")
        cmd_score(cfg, a.workers, a.split, a.seed, a.final_test)
    elif a.command == "reference":
        cmd_reference(cfg, a.workers, a.split, a.final_test, a.only)
    else:
        {"widths": cmd_widths, "widths_plot": cmd_widths_plot, "maskresize": cmd_maskresize}[a.command](cfg, a.workers)


if __name__ == "__main__":
    main()
