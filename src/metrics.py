"""Per-image segmentation metrics at native 2048x2048 resolution.

Every metric is evaluated inside the image's FOV mask (``src.fov``), applied identically
to the prediction and the ground truth; the ground truth itself is never edited.
Structures (skeletons, boundaries, vessel radii) are computed on the full masks and then
restricted to the FOV, so a vessel cut by the FOV edge does not get an artificial end or
boundary there.

Metrics (CLAUDE.md, Phase 2):
  overlap        Dice, IoU, precision, recall
  centerline     width-stratified centerline recall, 1-px tolerance: GT skeleton pixels
                 grouped by local radius (Euclidean distance transform of the GT) into
                 thin / medium / thick; recall = share of a class's skeleton pixels where
                 the prediction, dilated by 1 px (3x3), is vessel
  clDice         2 Tprec Tsens / (Tprec + Tsens), Tprec = |skel(P) & G| / |skel(P)|,
                 Tsens = |skel(G) & P| / |skel(G)|
  NSD            boundary-pixel approximation of normalized surface Dice: boundary = mask
                 minus its 1-px (3x3) erosion; tolerance tau = 2 px;
                 NSD = (|Bp within tau of Bg| + |Bg within tau of Bp|) / (|Bp| + |Bg|)
  density        vessel pixels in the FOV / FOV pixels
  fractal dim.   box counting on mask & FOV, box sizes 2..512 (powers of 2), slope of
                 log N(s) against log(1/s)

Empty sets (nothing divides by zero):
  * A ratio whose denominator is empty is 1.0 if its numerator's set is also empty in
    the other mask (both empty = perfect agreement), else 0.0. E.g. an empty prediction
    gets precision 0 against a non-empty GT, and 1 against an empty GT.
  * A width class with no GT skeleton pixels in the image gets NaN (the class is absent;
    averages over images skip it and report n).
  * The fractal dimension of an empty mask is NaN.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy import ndimage as ndi
from skimage.morphology import skeletonize

WIDTH_CLASSES = ("thin", "medium", "thick")
_SQUARE = np.ones((3, 3), bool)


def _ratio(num: int, den: int, other_empty: bool) -> float:
    if den == 0:
        return 1.0 if other_empty else 0.0
    return float(num) / float(den)


# --------------------------------------------------------------------------- building blocks
def skeleton(mask: np.ndarray) -> np.ndarray:
    return skeletonize(mask.astype(bool)) if mask.any() else np.zeros(mask.shape, bool)


def boundary(mask: np.ndarray) -> np.ndarray:
    """Mask minus its 1-px erosion (3x3); image-edge pixels of the mask count as boundary."""
    m = mask.astype(bool)
    return m & ~ndi.binary_erosion(m, structure=_SQUARE, border_value=0)


def width_class(radius: np.ndarray, cutoffs: tuple[float, float]) -> np.ndarray:
    """0 = thin (r < lo), 1 = medium (lo <= r <= hi), 2 = thick (r > hi); radius in px at 2048."""
    lo, hi = cutoffs
    return np.where(radius < lo, 0, np.where(radius <= hi, 1, 2))


@dataclass
class GroundTruth:
    """Everything derived from one GT mask and its FOV; computed once per image and
    reused for every model, seed and threshold."""
    gt: np.ndarray            # GT & FOV
    fov: np.ndarray
    skel: np.ndarray          # skeleton(full GT) & FOV
    skel_radius: np.ndarray   # EDT(full GT) at each skeleton pixel (same order as np.nonzero(skel))
    bnd: np.ndarray           # boundary(full GT) & FOV
    dist_to_bnd: np.ndarray   # distance of every pixel to the nearest GT-boundary pixel (in FOV)


def prepare_gt(gt_full: np.ndarray, fov: np.ndarray) -> GroundTruth:
    gt_full = gt_full.astype(bool)
    fov = fov.astype(bool)
    skel = skeleton(gt_full) & fov
    radius = ndi.distance_transform_edt(gt_full)[skel]
    bnd = boundary(gt_full) & fov
    dist = ndi.distance_transform_edt(~bnd) if bnd.any() else np.full(gt_full.shape, np.inf)
    return GroundTruth(gt=gt_full & fov, fov=fov, skel=skel, skel_radius=radius, bnd=bnd, dist_to_bnd=dist)


# --------------------------------------------------------------------------- metrics
def overlap(pred: np.ndarray, gt: np.ndarray) -> dict:
    p, g = pred.astype(bool), gt.astype(bool)
    return overlap_counts(int((p & g).sum()), int(p.sum()), int(g.sum()))


def overlap_counts(tp: int, np_: int, ng: int) -> dict:
    """Dice, IoU, precision, recall from true positives, predicted and GT pixel counts."""
    both_empty = np_ == 0 and ng == 0
    return {"dice": _ratio(2 * tp, np_ + ng, both_empty), "iou": _ratio(tp, np_ + ng - tp, both_empty),
            "precision": _ratio(tp, np_, ng == 0), "recall": _ratio(tp, ng, np_ == 0)}


def centerline_recall(pred: np.ndarray, g: GroundTruth, cutoffs: tuple[float, float], tol: int = 1) -> dict:
    """Recall of GT skeleton pixels per width class; prediction dilated by ``tol`` px (square)."""
    p = pred.astype(bool) & g.fov
    if tol > 0:
        p = ndi.binary_dilation(p, structure=_SQUARE, iterations=tol)
    hit = p[g.skel]
    cls = width_class(g.skel_radius, cutoffs)
    out = {}
    for k, name in enumerate(WIDTH_CLASSES):
        sel = cls == k
        out[f"cl_recall_{name}"] = float(hit[sel].mean()) if sel.any() else float("nan")
        out[f"n_skel_{name}"] = int(sel.sum())
    out["cl_recall_all"] = float(hit.mean()) if hit.size else float("nan")
    return out


def cldice(pred: np.ndarray, g: GroundTruth, pred_skel: np.ndarray | None = None) -> dict:
    p = pred.astype(bool)
    sp = (skeleton(p) if pred_skel is None else pred_skel) & g.fov
    p = p & g.fov
    t_prec = _ratio(int((sp & g.gt).sum()), int(sp.sum()), not g.skel.any())
    t_sens = _ratio(int((g.skel & p).sum()), int(g.skel.sum()), not sp.any())
    cl = 0.0 if t_prec + t_sens == 0 else 2 * t_prec * t_sens / (t_prec + t_sens)
    return {"cldice": cl, "cl_tprec": t_prec, "cl_tsens": t_sens}


def nsd(pred: np.ndarray, g: GroundTruth, tol: float = 2.0) -> float:
    bp = boundary(pred.astype(bool)) & g.fov
    n_p, n_g = int(bp.sum()), int(g.bnd.sum())
    if n_p == 0 or n_g == 0:
        return 1.0 if n_p == n_g == 0 else 0.0
    p_ok = int((g.dist_to_bnd[bp] <= tol).sum())
    g_ok = int((ndi.distance_transform_edt(~bp)[g.bnd] <= tol).sum())
    return (p_ok + g_ok) / (n_p + n_g)


def vessel_density(mask: np.ndarray, fov: np.ndarray) -> float:
    f = fov.astype(bool)
    return float((mask.astype(bool) & f).sum()) / float(f.sum())


def fractal_dimension(mask: np.ndarray, sizes=(2, 4, 8, 16, 32, 64, 128, 256, 512)) -> float:
    """Box-counting dimension: slope of log N(s) against log(1/s)."""
    m = mask.astype(bool)
    if not m.any():
        return float("nan")
    sizes = [s for s in sizes if s <= min(m.shape)]                 # boxes larger than the image are meaningless
    counts = []
    for s in sizes:
        h, w = (m.shape[0] // s) * s, (m.shape[1] // s) * s
        counts.append(m[:h, :w].reshape(h // s, s, w // s, s).any(axis=(1, 3)).sum())
    slope, _ = np.polyfit(np.log(1.0 / np.asarray(sizes, float)), np.log(np.asarray(counts, float)), 1)
    return float(slope)


def compute_all(pred: np.ndarray, g: GroundTruth, cutoffs: tuple[float, float], center_tol: int = 1,
                nsd_tol: float = 2.0, box_sizes=(2, 4, 8, 16, 32, 64, 128, 256, 512), extras: bool = True,
                sensitivity_cutoffs: tuple[float, float] | None = None) -> dict:
    """Every metric for one binary prediction (2048x2048) against one prepared GT.
    ``sensitivity_cutoffs`` adds the width-stratified recall with other cut-offs, with
    keys suffixed ``_r{lo}_{hi}`` (e.g. cl_recall_thin_r4_8; Phase 6 secondary table)."""
    p_full = pred.astype(bool)
    p = p_full & g.fov
    out = overlap(p, g.gt)
    out.update(centerline_recall(p_full, g, cutoffs, center_tol))
    if sensitivity_cutoffs is not None:
        tag = "_r{:g}_{:g}".format(*sensitivity_cutoffs)
        out.update({k + tag: v for k, v in centerline_recall(p_full, g, sensitivity_cutoffs, center_tol).items()
                    if k != "cl_recall_all" and k != "n_skel_all"})
    out.update(cldice(p_full, g))
    out["nsd"] = nsd(p_full, g, nsd_tol)
    if extras:
        out["density_pred"] = vessel_density(p, g.fov)
        out["density_gt"] = vessel_density(g.gt, g.fov)
        out["fd_pred"] = fractal_dimension(p, box_sizes)
        out["fd_gt"] = fractal_dimension(g.gt, box_sizes)
    return out
