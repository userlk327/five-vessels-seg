"""Agreement between two available annotations of the same image (post hoc, descriptive; labels only).

    python -m src.agreement

Added during manuscript writing (user decision, 2026-11-22). It uses ground-truth masks only, no
model predictions, and changes no earlier result.

FIVES contains pairs of byte-identical image files whose masks differ (data audit:
results/checks/integrity.json, "identical_images" groups with masks_identical == False). The dataset
does not say who drew either mask, so this is agreement between two available annotations of the
same image, not inter-rater agreement. For each pair, each mask is scored against the other with
the evaluation metrics and settings used for the models (src.metrics.compute_all, native 2048
resolution, inside the image's FOV, no resampling), in both directions, because centreline recall
and precision depend on which mask is the reference. Symmetric metrics (Dice, IoU, clDice, NSD) are
the same in both directions.

Writes
  results/tables/annotation_agreement_directions.csv  one row per pair and direction
  results/tables/annotation_agreement.csv             one row per pair: mean of its two directions
  results/checks/annotation_agreement_info.json       pairs, settings, environment
"""
from __future__ import annotations

import json

import numpy as np
import pandas as pd

METRICS = ["dice", "iou", "precision", "recall", "cl_recall_thin", "cl_recall_medium", "cl_recall_thick",
           "cldice", "nsd", "cl_recall_thin_r4_8"]


def score_pair(mask_ref: np.ndarray, mask_other: np.ndarray, fov: np.ndarray, s: dict) -> dict:
    """Metrics of ``mask_other`` scored against ``mask_ref`` (both binary, native resolution)."""
    from src.metrics import compute_all, prepare_gt
    g = prepare_gt(mask_ref.astype(bool), fov)
    return compute_all(mask_other.astype(bool), g, s["cutoffs"], s["center_tol"], s["nsd_tol"], s["box_sizes"],
                       extras=False, sensitivity_cutoffs=s["sens_cutoffs"])


def main() -> None:
    from src.data import load_splits, raw_image_path, raw_mask_path, read_unchanged, sha256_file
    from src.evaluate import eval_settings, load_fov
    from src.utils import environment_info, load_config, project_path
    cfg = load_config("configs/base.yaml")
    s = eval_settings(cfg)
    split = load_splits(cfg).set_index("image_id").split
    groups = json.loads((project_path("results/checks") / "integrity.json").read_text())["identical_images"]["groups"]
    pairs = [g["images"] for g in groups if not g["masks_identical"]]
    rows = []
    for a, b in pairs:
        if sha256_file(raw_image_path(cfg, a)) != sha256_file(raw_image_path(cfg, b)):
            raise RuntimeError(f"{a}, {b}: image files are not identical")
        fov = load_fov(cfg, a)
        if not np.array_equal(fov, load_fov(cfg, b)):
            raise RuntimeError(f"{a}, {b}: FOV masks differ for identical images")
        ma, mb = read_unchanged(raw_mask_path(cfg, a)) > 0, read_unchanged(raw_mask_path(cfg, b)) > 0
        for ref, oth, mr, mo in ((a, b, ma, mb), (b, a, mb, ma)):
            m = score_pair(mr, mo, fov, s)
            rows.append({"pair": f"{a}|{b}", "reference": ref, "other": oth, "split": split[a], "category": a[-1],
                         **{k: m[k] for k in METRICS}, "n_skel_thin": m["n_skel_thin"]})
            print(f"[agreement] {oth} vs {ref}: Dice {m['dice']:.4f}, thin {m['cl_recall_thin']:.4f}", flush=True)
    d = pd.DataFrame(rows)
    out = project_path("results/tables")
    d.to_csv(out / "annotation_agreement_directions.csv", index=False)
    per_pair = d.groupby(["pair", "split", "category"], as_index=False)[METRICS].mean()
    per_pair.to_csv(out / "annotation_agreement.csv", index=False)
    info = {"description": "agreement between two available annotations of the same image; post hoc, descriptive; "
                           "labels only (no model predictions)",
            "n_pairs": len(pairs), "pairs": pairs,
            "settings": {"resolution": s["native"], "width_cutoffs_radius_px": list(s["cutoffs"]),
                         "centerline_tolerance_px": s["center_tol"], "nsd_tolerance_px": s["nsd_tol"], "fov": "image's FOV mask"},
            "per_pair": "mean of the two directions (each mask as reference once)",
            "environment": {k: v for k, v in environment_info().items() if k in ("time", "host", "git")}}
    (project_path("results/checks") / "annotation_agreement_info.json").write_text(json.dumps(info, indent=1))
    print(per_pair[["pair", "split"] + METRICS].round(4).to_string(index=False))


if __name__ == "__main__":
    main()
