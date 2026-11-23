"""Post hoc analyses added after the referee report (manuscript revision; decisions D2 and D4).
Both were added AFTER the main test evaluation. Neither predicts anything again, and nothing is
chosen on the test set: D2 re-scores the saved probability maps, D4 re-analyses the saved
per-image CSVs.

    python -m src.posthoc score --split val  --workers 16    # D2, fine threshold sweep (val)
    python -m src.posthoc score --split test --workers 16    # D2, fine sweep + extra metrics (test)
    python -m src.posthoc tables                             # D2 tables + D4 statistics
    (CPU job: jobs/cpu.sbatch src.posthoc score --split val :: src.posthoc score --split test :: src.posthoc tables)

D2, per image and automatic run (7 configurations x 3 seeds):
  * fine threshold sweep 0.01 ... 0.99 (step 0.01): Dice, IoU, precision, recall and the
    width-stratified centreline recall, exactly as src.evaluate.sweep_metrics
    -> results/eval/<run>/seed<s>/<split>_sweep_fine.csv
  * test only, at the run's own val-selected threshold and at the fixed 0.5:
      - width-stratified centreline PRECISION, the mirror of the recall: skeleton of the binary
        prediction (inside the FOV); each predicted skeleton pixel is classed by the prediction's
        own EDT radius with the same cut-offs (thin < 3 px, medium 3-7, thick > 7 at 2048);
        precision of a class = share of its predicted skeleton pixels where the ground truth,
        dilated by 1 px (3 x 3), is vessel. A class with no predicted skeleton pixels is NaN.
      - clDice and NSD (in the main analysis only at the selected threshold)
      - spur-pruned recall and precision (sensitivity check of the thin class): terminal
        skeleton branches (one end free, the other at a junction) no longer than 1.5 x the
        largest vessel radius along the branch + 2 px are removed before classing; one pass.
        These are medial-axis spurs running from the centre line of a wider vessel to a bump in
        its boundary, whose tips have small radii; a real side branch survives unless it
        extends less than about 0.5 R + 2 px beyond its parent vessel.
    -> results/eval/<run>/seed<s>/test_posthoc.csv; ground-truth pruning counts per image
       -> results/tables/posthoc_gt_pruning.csv
  * matched operating point: per run, the threshold whose mean VAL precision (fine grid) is
    closest to a target precision (ties: lower threshold), applied to test. Targets: the
    U-Net's mean val precision at its val-selected thresholds (primary), 0.90 and 0.95.

D4 (results/tables/posthoc_stats.csv), for the main comparisons (each model vs the U-Net, and the
2 x 2 ablation effects), at the val-selected thresholds, at 0.5 and at matched precision:
  * image bootstrap: seed-averaged per-image values, images resampled (as the main analysis;
    reproduces its intervals exactly, checked);
  * cluster bootstrap: test images grouped into possible same-patient clusters with the split
    rules (byte-identical image files, identical masks, identical marking + adjacent numbers,
    isolated identical-marking pairs; test-test pairs only) and clusters resampled
    (results/checks/test_groups.csv: ids only);
  * seed x cluster bootstrap: clusters resampled as above and, in the same replicate, the three
    seeds of each model resampled with replacement - jointly for comparisons among the four
    CNNs (their seed s shares the data order; paired), independently for SAM vs the U-Net;
  * seed-paired values: the statistic computed from seed s of every model, for s = 0, 1, 2.
  10,000 replicates, percentile 95% intervals.
"""
from __future__ import annotations

import argparse
import json
import multiprocessing
import os
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
import pandas as pd
from scipy import ndimage as ndi

from src.metrics import WIDTH_CLASSES, centerline_recall, cldice, nsd, skeleton, width_class

AUTO = ("unet", "attention_unet", "multiscale_unet", "ms_attention_unet", "sam_lora_tiled", "sam_lora_whole",
        "sam_lora_tiled_overlap")
CNNS = AUTO[:4]
SEEDS = (0, 1, 2)
FINE = [round(0.01 * k, 2) for k in range(1, 100)]
TARGETS_FIXED = (0.90, 0.95)
PRUNE_FACTOR, PRUNE_MARGIN = 1.5, 2.0
N_BOOT = 10000
_SQ = np.ones((3, 3), bool)


# --------------------------------------------------------------------------- building blocks
def prune_spurs(skel: np.ndarray, radius_map: np.ndarray, factor: float = PRUNE_FACTOR,
                margin: float = PRUNE_MARGIN) -> np.ndarray:
    """Remove terminal branches (an end point at one end, a junction at the other) whose pixel
    count is at most ``factor`` x the largest radius along the branch + ``margin``. One pass."""
    sk = skel.astype(bool)
    if not sk.any():
        return sk.copy()
    nb = ndi.convolve(sk.astype(np.uint8), np.ones((3, 3), np.uint8), mode="constant", cval=0).astype(np.int16) - 1
    junction = sk & (nb >= 3)
    endpoint = sk & (nb == 1)
    branches = sk & ~junction
    lab, n = ndi.label(branches, structure=_SQ)
    if n == 0:
        return sk.copy()
    idx = np.arange(1, n + 1)
    length = np.asarray(ndi.sum(branches, lab, idx))
    has_end = np.asarray(ndi.maximum(endpoint, lab, idx)) > 0
    touches = np.asarray(ndi.maximum(ndi.binary_dilation(junction, _SQ), lab, idx)) > 0
    rmax = np.asarray(ndi.maximum(radius_map, lab, idx))
    spur = has_end & touches & (length <= factor * rmax + margin)
    return sk & ~np.concatenate([[False], spur])[lab]


def width_precision(pred_full: np.ndarray, fov: np.ndarray, gt_dil: np.ndarray, cutoffs, pred_skel=None) -> dict:
    """Width-stratified centreline precision (module docstring), unpruned and spur-pruned."""
    p = pred_full.astype(bool)
    sp = skeleton(p) if pred_skel is None else pred_skel
    edt = ndi.distance_transform_edt(p) if p.any() else np.zeros(p.shape)
    out = {}
    for tag, sk in (("", sp), ("_pruned", prune_spurs(sp, edt))):
        s = sk & fov
        hit, cls = gt_dil[s], width_class(edt[s], cutoffs)
        for k, name in enumerate(WIDTH_CLASSES):
            sel = cls == k
            out[f"w_prec_{name}{tag}"] = float(hit[sel].mean()) if sel.any() else float("nan")
            out[f"n_pskel_{name}{tag}"] = int(sel.sum())
        out[f"w_prec_all{tag}"] = float(hit.mean()) if hit.size else float("nan")
    return out


def pruned_recall(pred_full: np.ndarray, fov: np.ndarray, skel_pruned: np.ndarray, radius: np.ndarray, cutoffs,
                  tol: int = 1) -> dict:
    """centerline_recall on the spur-pruned GT skeleton (skel_pruned & FOV, radii in the same order)."""
    p = pred_full.astype(bool) & fov
    if tol > 0:
        p = ndi.binary_dilation(p, structure=_SQ, iterations=tol)
    hit, cls = p[skel_pruned], width_class(radius, cutoffs)
    out = {}
    for k, name in enumerate(WIDTH_CLASSES):
        sel = cls == k
        out[f"cl_recall_{name}_pruned"] = float(hit[sel].mean()) if sel.any() else float("nan")
    return out


def runs() -> list[tuple[str, int]]:
    from src.utils import project_path
    return [(m, s) for m in AUTO for s in SEEDS
            if (project_path("outputs") / m / f"seed{s}" / "val" / "probs.npz").exists()]


def selected_threshold(model: str, seed: int) -> float:
    from src.utils import project_path
    j = json.loads((project_path("results/eval") / model / f"seed{seed}" / "val_threshold.json").read_text())
    assert (j["model"], j["seed"]) == (model, seed)
    return float(j["threshold"])


# --------------------------------------------------------------------------- D2 scoring
def _score_image(args) -> tuple[str, dict]:
    import cv2

    from src.data import raw_mask_path, read_npz_member, read_unchanged
    from src.evaluate import check_prob_map, eval_settings, load_fov, sweep_metrics
    from src.metrics import prepare_gt
    from src.resample import upsample_prob
    from src.utils import project_path
    image_id, cfg, split, run_list, thresholds = args
    cv2.setNumThreads(1)
    s = eval_settings(cfg)
    gt_full = read_unchanged(raw_mask_path(cfg, image_id)) > 0
    fov = load_fov(cfg, image_id)
    g = prepare_gt(gt_full, fov)
    out = {"sweep": {}, "posthoc": {}}
    if split == "test":
        edt_gt = ndi.distance_transform_edt(gt_full)
        sk_pr = prune_spurs(skeleton(gt_full), edt_gt) & fov
        radius_pr = edt_gt[sk_pr]
        gt_dil = ndi.binary_dilation(g.gt, structure=_SQ, iterations=s["center_tol"])
        c0, c1 = width_class(g.skel_radius, s["cutoffs"]), width_class(radius_pr, s["cutoffs"])
        out["gt"] = {**{f"n_skel_{n}": int((c0 == k).sum()) for k, n in enumerate(WIDTH_CLASSES)},
                     **{f"n_skel_{n}_pruned": int((c1 == k).sum()) for k, n in enumerate(WIDTH_CLASSES)}}
    for model, seed in run_list:
        arch = project_path("outputs") / model / f"seed{seed}" / split / "probs.npz"
        prob = read_npz_member(arch, image_id)
        check_prob_map(prob, image_id)
        up = upsample_prob(prob.astype(np.float32), s["native"])
        out["sweep"][(model, seed)] = sweep_metrics(up, g, FINE, s["cutoffs"], s["center_tol"])
        if split != "test":
            continue
        rows, done = [], {}
        for rule, t in (("selected", thresholds[(model, seed)]), ("fixed_0.5", 0.5)):
            if t not in done:
                pred = up >= t
                sp = skeleton(pred)
                r = {**centerline_recall(pred, g, s["cutoffs"], s["center_tol"]),
                     **pruned_recall(pred, fov, sk_pr, radius_pr, s["cutoffs"], s["center_tol"]),
                     **width_precision(pred, fov, gt_dil, s["cutoffs"], pred_skel=sp),
                     **cldice(pred, g, pred_skel=sp), "nsd": nsd(pred, g, s["nsd_tol"])}
                done[t] = {k: v for k, v in r.items() if not k.startswith("n_skel")}
            rows.append({"rule": rule, "threshold": t, **done[t]})
        out["posthoc"][(model, seed)] = rows
    return image_id, out


def cmd_score(cfg: dict, split: str, workers: int) -> None:
    from src.data import load_splits
    from src.utils import project_path
    sp = load_splits(cfg)
    ids = sp.loc[sp.split == split, "image_id"].tolist()
    cat = dict(zip(sp.image_id, sp.category))
    run_list = runs()
    thresholds = {r: selected_threshold(*r) for r in run_list}
    print(f"[posthoc] {split}: {len(ids)} images x {len(run_list)} runs", flush=True)
    with ProcessPoolExecutor(workers, mp_context=multiprocessing.get_context("spawn")) as ex:
        res = dict(ex.map(_score_image, [(i, cfg, split, run_list, thresholds) for i in ids], chunksize=1))
    ev = project_path("results/eval")
    checks = {}
    for model, seed in run_list:
        d = ev / model / f"seed{seed}"
        sw = pd.DataFrame([{"model": model, "seed": seed, "split": split, "image_id": i, "category": cat[i], **row}
                           for i in ids for row in res[i]["sweep"][(model, seed)]])
        sw.to_csv(d / f"{split}_sweep_fine.csv", index=False)
        old = pd.read_csv(d / f"{split}_sweep.csv")                      # consistency with the main sweep
        m = old.merge(sw, on=["image_id", "threshold"], suffixes=("_old", ""))
        chk = {c: float((m[c] - m[f"{c}_old"]).abs().max()) for c in ("dice", "precision", "cl_recall_thin")}
        chk["n_grid_rows_matched"] = int(len(m))
        if split == "test":
            ph = pd.DataFrame([{"model": model, "seed": seed, "split": split, "image_id": i, "category": cat[i], **row}
                               for i in ids for row in res[i]["posthoc"][(model, seed)]])
            ph.to_csv(d / "test_posthoc.csv", index=False)
            main = pd.read_csv(d / "test_metrics.csv")
            mm = main.merge(ph[ph.rule == "selected"], on="image_id", suffixes=("_main", ""))
            for c in ("cldice", "nsd", "cl_recall_thin", "cl_recall_thick"):
                chk[f"selected_{c}"] = float((mm[c] - mm[f"{c}_main"]).abs().max())
        checks[f"{model}/seed{seed}"] = chk
    if split == "test":
        gt = pd.DataFrame([{"image_id": i, "category": cat[i], **res[i]["gt"]} for i in ids])
        gt.to_csv(project_path("results/tables") / "posthoc_gt_pruning.csv", index=False)
    worst = {k: max(v for c, v in chk.items() if not c.startswith("n_")) for k, chk in checks.items()}
    (project_path("results/checks") / f"posthoc_consistency_{split}.json").write_text(
        json.dumps({"max_abs_diff_vs_main_analysis": checks, "worst_per_run": worst}, indent=1))
    print(f"[posthoc] {split}: largest difference from the main analysis over all runs: {max(worst.values()):.3g}")


# --------------------------------------------------------------------------- per-image long table
def test_groups(cfg: dict) -> pd.DataFrame:
    """Possible same-patient clusters among the TEST images (split rules, test-test pairs only;
    identical masks added). Ids only."""
    from src.data import CHECKS_DIR, _groups_to_components, id_sort_key, load_splits, marking_group_sizes
    from src.utils import project_path
    ids = sorted(load_splits(cfg).query("split == 'test'").image_id, key=id_sort_key)
    qa = pd.read_csv(CHECKS_DIR / "image_qa.csv")
    pairs = pd.read_csv(CHECKS_DIR / "marking_pairs.csv")
    t = cfg["data"]["split"]["group_marking_iou"]
    edges = []
    for col, why in (("image_sha256_parquet", "identical image file"), ("mask_sha256_parquet", "identical mask")):
        for _, g in qa[qa[col].duplicated(keep=False)].groupby(col):
            m = sorted(g.image_id, key=id_sort_key)
            edges += [(m[0], b, why) for b in m[1:]]
    adj = pairs[(pairs.iou >= t) & pairs.adjacent_numbers]
    edges += [(a, b, f"identical marking (IoU >= {t}) + adjacent numbers") for a, b in zip(adj.image_a, adj.image_b)]
    gsize = marking_group_sizes(pairs, t)
    iso = pairs[(pairs.iou >= t) & (pairs.image_a.map(gsize) == 2)]
    edges += [(a, b, f"identical marking (IoU >= {t}), isolated pair") for a, b in zip(iso.image_a, iso.image_b)]
    edges = [e for e in edges if e[0] in ids and e[1] in ids]
    comps = _groups_to_components(ids, [(a, b) for a, b, _ in edges])
    why = {}
    for a, b, r in edges:
        for i in (a, b):
            why.setdefault(i, set()).add(r)
    df = pd.DataFrame([{"image_id": i, "cluster": f"t{k:03d}", "cluster_size": len(c),
                        "reasons": "; ".join(sorted(why.get(i, [])))} for k, c in enumerate(comps) for i in c])
    df.to_csv(project_path("results/checks") / "test_groups.csv", index=False)
    return df


def matched_thresholds(target_unet: float) -> pd.DataFrame:
    """Per run and target: the fine-grid threshold whose mean val precision is closest to the target."""
    from src.utils import project_path
    ev, rows = project_path("results/eval"), []
    for model, seed in runs():
        curve = pd.read_csv(ev / model / f"seed{seed}" / "val_sweep_fine.csv").groupby("threshold").precision.mean()
        for name, target in (("unet_val_precision", target_unet), *((f"{p:.2f}", p) for p in TARGETS_FIXED)):
            gap = (curve - target).abs()
            t = float(gap[gap == gap.min()].index.min())
            rows.append({"model": model, "seed": seed, "target": name, "target_precision": target, "threshold": t,
                         "val_precision": float(curve.loc[t])})
    return pd.DataFrame(rows)


def long_table(cfg: dict) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    """Per-image values: rule x metric x run x image."""
    from src.utils import project_path
    ev = project_path("results/eval")
    unet_val_prec = float(np.mean([pd.read_csv(ev / "unet" / f"seed{s}" / "val_metrics.csv").precision.mean()
                                   for s in SEEDS]))
    mt = matched_thresholds(unet_val_prec)
    parts = []
    keep = ["model", "seed", "image_id", "category"]
    for model, seed in runs():
        d = ev / model / f"seed{seed}"
        main = pd.read_csv(d / "test_metrics.csv")
        sel = main[keep + ["dice", "precision", "recall", "cl_recall_thin", "cl_recall_medium", "cl_recall_thick",
                           "cldice", "nsd"]].assign(rule="selected")
        ph = pd.read_csv(d / "test_posthoc.csv")
        sel = sel.merge(ph[ph.rule == "selected"].drop(columns=["rule", "threshold", "split", "category", "model", "seed",
                                                                "cldice", "nsd", "cl_recall_thin", "cl_recall_medium",
                                                                "cl_recall_thick", "cl_recall_all"]), on="image_id")
        fine = pd.read_csv(d / "test_sweep_fine.csv")
        fx = fine[np.isclose(fine.threshold, 0.5)][keep + ["dice", "precision", "recall", "cl_recall_thin",
                                                             "cl_recall_medium", "cl_recall_thick"]]
        fx = fx.merge(ph[ph.rule == "fixed_0.5"].drop(columns=["rule", "threshold", "split", "category", "model", "seed",
                                                               "cl_recall_thin", "cl_recall_medium", "cl_recall_thick",
                                                               "cl_recall_all"]), on="image_id").assign(rule="fixed_0.5")
        frames = [sel, fx]
        for _, r in mt[(mt.model == model) & (mt.seed == seed)].iterrows():
            f = fine[np.isclose(fine.threshold, r.threshold)][keep + ["dice", "precision", "recall", "cl_recall_thin",
                                                                       "cl_recall_medium", "cl_recall_thick"]]
            frames.append(f.assign(rule=f"matched_{r.target}"))
        parts += frames
    wide = pd.concat(parts, ignore_index=True)
    return wide, mt, unet_val_prec


# --------------------------------------------------------------------------- D2 tables
def model_level(wide: pd.DataFrame, metrics) -> pd.DataFrame:
    """Per seed: mean over images (NaN skipped); then mean and SD across seeds."""
    per_seed = wide.groupby(["rule", "model", "seed"])[metrics].mean().reset_index()
    agg = per_seed.groupby(["rule", "model"])[metrics].agg(["mean", "std"])
    agg.columns = [f"{m}_{s}" for m, s in agg.columns]
    n = wide.groupby(["rule", "model"]).seed.nunique().rename("n_seeds")
    return agg.join(n).reset_index()


# --------------------------------------------------------------------------- D4 statistics
def _matrix(wide: pd.DataFrame, rule: str, metric: str, ids: list[str]) -> dict[str, np.ndarray]:
    sub = wide[wide.rule == rule]
    X = {}
    for m in AUTO:
        piv = sub[sub.model == m].pivot(index="seed", columns="image_id", values=metric)
        if len(piv):
            X[m] = piv.loc[list(SEEDS), ids].to_numpy(float)
            if not np.isfinite(X[m]).all():
                raise ValueError(f"{rule} {metric} {m}: non-finite per-image values")
    return X


COMPARISONS = {f"{m} - unet": {m: 1.0, "unet": -1.0} for m in AUTO[1:]}
COMPARISONS.update({
    "attention effect": {"attention_unet": .5, "unet": -.5, "ms_attention_unet": .5, "multiscale_unet": -.5},
    "multi-scale effect": {"multiscale_unet": .5, "unet": -.5, "ms_attention_unet": .5, "attention_unet": -.5},
    "interaction": {"ms_attention_unet": 1.0, "multiscale_unet": -1.0, "attention_unet": -1.0, "unet": 1.0}})


def cmd_stats(wide: pd.DataFrame, groups: pd.DataFrame, plan) -> pd.DataFrame:
    ids = sorted(groups.image_id)
    clusters = groups.set_index("image_id").loc[ids, "cluster"].to_numpy()
    cl_codes, cl_index = np.unique(clusters, return_inverse=True)
    K, n = len(cl_codes), len(ids)
    member = np.zeros((K, n))
    member[cl_index, np.arange(n)] = 1.0
    rs_img = np.random.RandomState(0).randint(0, n, (N_BOOT, n))            # = src.analysis.bootstrap_ci draws
    W_img = np.zeros((N_BOOT, n))
    np.add.at(W_img, (np.repeat(np.arange(N_BOOT), n), rs_img.ravel()), 1.0)
    rs_cl = np.random.RandomState(0).randint(0, K, (N_BOOT, K))
    C = np.zeros((N_BOOT, K))
    np.add.at(C, (np.repeat(np.arange(N_BOOT), K), rs_cl.ravel()), 1.0)
    W_cl = C @ member                                                      # image weights per replicate
    seed_rng = np.random.RandomState(1)

    def seed_weights():
        d = seed_rng.randint(0, len(SEEDS), (N_BOOT, len(SEEDS)))
        V = np.zeros((N_BOOT, len(SEEDS)))
        np.add.at(V, (np.repeat(np.arange(N_BOOT), len(SEEDS)), d.ravel()), 1.0)
        return V

    def q(v):
        return float(np.quantile(v, 0.025)), float(np.quantile(v, 0.975))

    rows = []
    for rule, metric in plan:
        X = _matrix(wide, rule, metric, ids)
        V_shared = seed_weights()
        V_own = {m: seed_weights() for m in AUTO}
        for name, coef in COMPARISONS.items():
            if not all(m in X for m in coef):
                continue
            img_vals = sum(c * X[m].mean(0) for m, c in coef.items())        # seed-averaged per-image statistic
            est = float(img_vals.mean())
            b_img = (W_img @ img_vals) / W_img.sum(1)
            b_cl = (W_cl @ img_vals) / W_cl.sum(1)
            cnn_only = all(m in CNNS for m in coef)
            b_sc = np.zeros(N_BOOT)
            for m, c in coef.items():
                V = V_shared if cnn_only else V_own[m]
                b_sc += c * ((V * (W_cl @ X[m].T)).sum(1) / (V.sum(1) * W_cl.sum(1)))
            seed_vals = [float(sum(c * X[m][k].mean() for m, c in coef.items())) for k in range(len(SEEDS))]
            rows.append({"rule": rule, "metric": metric, "comparison": name, "estimate": est,
                         **dict(zip(("ci_image_low", "ci_image_high"), q(b_img))),
                         **dict(zip(("ci_cluster_low", "ci_cluster_high"), q(b_cl))),
                         **dict(zip(("ci_seed_cluster_low", "ci_seed_cluster_high"), q(b_sc))),
                         "seed_pairing": "joint (CNN seeds paired)" if cnn_only else "independent per model",
                         **{f"seed{s}": v for s, v in zip(SEEDS, seed_vals)},
                         "seeds_same_sign": int(max(sum(v > 0 for v in seed_vals), sum(v < 0 for v in seed_vals))),
                         "n_images": n, "n_clusters": K})
    return pd.DataFrame(rows)


def cmd_tables(cfg: dict) -> None:
    from src.utils import project_path
    tab = project_path("results/tables")
    wide, mt, unet_val_prec = long_table(cfg)
    mt.to_csv(tab / "posthoc_matched_thresholds.csv", index=False)
    wp = [f"w_prec_{c}{t}" for t in ("", "_pruned") for c in WIDTH_CLASSES]
    pr = [f"cl_recall_{c}_pruned" for c in WIDTH_CLASSES]
    base = ["dice", "precision", "recall", "cl_recall_thin", "cl_recall_medium", "cl_recall_thick"]
    ml = model_level(wide[wide.rule.isin(["selected", "fixed_0.5"])], base + ["cldice", "nsd"] + wp + pr)
    ml.to_csv(tab / "posthoc_model_level.csv", index=False)
    mm = model_level(wide[wide.rule.str.startswith("matched_")], base)
    mm.to_csv(tab / "posthoc_matched_model_level.csv", index=False)
    gtp = pd.read_csv(tab / "posthoc_gt_pruning.csv")
    prune = {f"share_{c}_skeleton_removed": float(1 - gtp[f"n_skel_{c}_pruned"].sum() / gtp[f"n_skel_{c}"].sum())
             for c in WIDTH_CLASSES}
    groups = test_groups(cfg)
    plan = [("selected", m) for m in ("dice", "precision", "cl_recall_thin", "cldice", "nsd", "w_prec_thin",
                                      "cl_recall_thin_pruned")] + \
           [("fixed_0.5", m) for m in ("dice", "precision", "cl_recall_thin", "cldice", "nsd", "w_prec_thin")] + \
           [("matched_unet_val_precision", m) for m in ("dice", "precision", "cl_recall_thin")]
    # w_prec_thin can be NaN only for an image without predicted thin skeleton; checked in _matrix
    st = cmd_stats(wide, groups, plan)
    st.to_csv(tab / "posthoc_stats.csv", index=False)
    # check: the image bootstrap reproduces the main ablation intervals
    chk = {}
    for rule, f in (("selected", "test_ablation_effects.csv"), ("fixed_0.5", "test_ablation_effects_fixed_t0.5.csv")):
        old = pd.read_csv(tab / f)
        for eff, name in (("attention", "attention effect"), ("multi-scale", "multi-scale effect"),
                          ("interaction", "interaction")):
            for metric in ("dice", "cl_recall_thin"):
                o = old[(old.metric == metric) & (old.effect == eff)].iloc[0]
                s = st[(st.rule == rule) & (st.metric == metric) & (st.comparison == name)].iloc[0]
                chk[f"{rule}/{metric}/{eff}"] = max(abs(o["mean"] - s.estimate), abs(o.ci_low - s.ci_image_low),
                                                    abs(o.ci_high - s.ci_image_high))
    info = {"unet_mean_val_precision_target": unet_val_prec, "spur_pruning_rule": f"terminal branch length <= {PRUNE_FACTOR} x max radius + {PRUNE_MARGIN} px",
            "gt_pruning": prune, "n_boot": N_BOOT,
            "n_test_clusters": int(groups.cluster.nunique()),
            "n_test_images_in_multi_image_clusters": int((groups.cluster_size > 1).sum()),
            "n_multi_image_clusters": int(groups.loc[groups.cluster_size > 1, "cluster"].nunique()),
            "largest_cluster": int(groups.cluster_size.max()),
            "check_image_bootstrap_vs_main_ablation_max_abs_diff": max(chk.values()), "checks": chk}
    (tab / "posthoc_info.json").write_text(json.dumps(info, indent=1))
    show = st[st.metric.isin(["cl_recall_thin", "dice"])][["rule", "metric", "comparison", "estimate", "ci_image_low",
                                                            "ci_image_high", "ci_cluster_low", "ci_cluster_high",
                                                            "ci_seed_cluster_low", "ci_seed_cluster_high",
                                                            "seeds_same_sign"]]
    with pd.option_context("display.width", 250, "display.max_columns", 30):
        print(show.round(4).to_string(index=False))
        print(json.dumps({k: v for k, v in info.items() if k != "checks"}, indent=1))
    print(f"[posthoc] wrote {tab}/posthoc_*.csv, posthoc_info.json, results/checks/test_groups.csv")


def main() -> None:
    from src.utils import load_config
    ap = argparse.ArgumentParser(description="Post hoc analyses D2 and D4 (after the referee report)")
    ap.add_argument("command", choices=["score", "tables"])
    ap.add_argument("--split", choices=["val", "test"], default="val")
    ap.add_argument("--config", default="configs/base.yaml")
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", "1")))
    a = ap.parse_args()
    cfg = load_config(a.config)
    if a.command == "score":
        cmd_score(cfg, a.split, a.workers)
    else:
        cmd_tables(cfg)


if __name__ == "__main__":
    main()
