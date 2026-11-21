"""Unit tests for src/metrics.py on synthetic masks (CLAUDE.md, Phase 2)."""
import numpy as np
import pytest

from src.metrics import (centerline_recall, cldice, compute_all, fractal_dimension, nsd, overlap,
                         prepare_gt, width_class)

CUT = (3, 7)          # thin < 3, medium 3-7, thick > 7 (radius, px)
N = 256


def full_fov(n=N):
    return np.ones((n, n), bool)


def vline(width, x0=100, n=N, y=(20, 236)):
    m = np.zeros((n, n), bool)
    m[y[0]:y[1], x0:x0 + width] = True
    return m


def tree():
    """GT with one thick (15 px) and one thin (2 px) vessel of the same length."""
    return vline(15, x0=60) | vline(2, x0=180)


def test_identical_masks_score_one():
    gt = tree(); g = prepare_gt(gt, full_fov())
    m = compute_all(gt, g, CUT, extras=False)
    for k in ("dice", "iou", "precision", "recall", "cldice", "nsd"):
        assert m[k] == pytest.approx(1.0), k
    for k in ("cl_recall_thin", "cl_recall_thick", "cl_recall_all"):
        assert m[k] == pytest.approx(1.0), k


def test_empty_prediction_scores_zero():
    gt = tree(); g = prepare_gt(gt, full_fov())
    m = compute_all(np.zeros_like(gt), g, CUT, extras=False)
    for k in ("dice", "iou", "precision", "recall", "cldice", "nsd", "cl_recall_thin", "cl_recall_thick", "cl_recall_all"):
        assert m[k] == 0.0, k


def test_absent_width_class_is_nan():
    gt = vline(2)                                                             # thin only
    m = centerline_recall(gt, prepare_gt(gt, full_fov()), CUT)
    assert m["cl_recall_thin"] == 1.0 and np.isnan(m["cl_recall_medium"]) and np.isnan(m["cl_recall_thick"])


def test_both_empty_is_perfect_and_no_division_by_zero():
    z = np.zeros((N, N), bool); g = prepare_gt(z, full_fov())
    m = compute_all(z, g, CUT, extras=False)
    assert m["dice"] == m["iou"] == m["precision"] == m["recall"] == m["cldice"] == m["nsd"] == 1.0
    assert np.isnan(m["cl_recall_all"])


def test_two_px_line_shifted_one_px_dice_drops_nsd_stays_one():
    gt = vline(2, x0=100); pred = vline(2, x0=101)
    g = prepare_gt(gt, full_fov())
    assert overlap(pred, gt)["dice"] == pytest.approx(0.5)                   # clearly lower
    assert nsd(pred, g, tol=2) == pytest.approx(1.0)                         # within tau = 2
    assert nsd(vline(2, x0=104), g, tol=2) < 1                               # 4 px off: not within tolerance


def test_missing_thin_vessel_hurts_cldice_much_more_than_dice():
    gt = tree(); pred = vline(15, x0=60)                                     # thick only
    g = prepare_gt(gt, full_fov())
    dice = overlap(pred, gt)["dice"]; cl = cldice(pred, g)["cldice"]
    assert dice > 0.9 and cl < 0.75
    assert (1 - cl) > 3 * (1 - dice)
    r = centerline_recall(pred, g, CUT)
    assert r["cl_recall_thin"] == 0.0 and r["cl_recall_thick"] == pytest.approx(1.0)


@pytest.mark.parametrize("width,expected", [(1, "thin"), (3, "thin"), (5, "medium"), (9, "medium"),
                                            (13, "medium"), (15, "thick"), (21, "thick")])
def test_line_of_known_width_lands_in_correct_class(width, expected):
    """Straight vertical line: the skeleton radius is (width + 1) / 2 (EDT to background).
    A few skeleton pixels at the line ends (end spurs) can fall in a neighbouring class."""
    g = prepare_gt(vline(width), full_fov())
    counts = centerline_recall(vline(width), g, CUT)
    total = sum(counts[f"n_skel_{c}"] for c in ("thin", "medium", "thick"))
    assert counts[f"n_skel_{expected}"] >= 0.95 * total


def test_width_class_boundaries():
    assert width_class(np.array([2.9, 3.0, 7.0, 7.1]), CUT).tolist() == [0, 1, 1, 2]


def test_fractal_dimension_square_two_line_one():
    sq = np.zeros((2048, 2048), bool); sq[100:1900, 100:1900] = True       # real image size: boxes 2..512
    ln = np.zeros((2048, 2048), bool); ln[1024, 24:2024] = True
    assert fractal_dimension(sq) == pytest.approx(2.0, abs=0.1)
    assert fractal_dimension(ln) == pytest.approx(1.0, abs=0.1)
    assert np.isnan(fractal_dimension(np.zeros((1024, 1024), bool)))


def test_metrics_only_count_inside_fov():
    gt = tree()
    gt_with_artifact = gt.copy(); gt_with_artifact[0, :] = True               # border-line annotation
    fov = np.zeros((N, N), bool); fov[10:246, 10:246] = True
    g = prepare_gt(gt_with_artifact, fov)
    pred = gt.copy(); pred[250:, :] = True                                    # false positives outside FOV
    m = compute_all(pred, g, CUT, extras=False)
    for k in ("dice", "precision", "recall", "cldice", "nsd", "cl_recall_all"):
        assert m[k] == pytest.approx(1.0), k


def test_vessel_density_uses_the_fov():
    gt = tree(); fov = np.zeros((N, N), bool); fov[:, :128] = True
    m = compute_all(gt, prepare_gt(gt, fov), CUT)
    assert m["density_gt"] == pytest.approx((gt & fov).sum() / fov.sum())


def test_sensitivity_cutoffs_add_secondary_columns():
    gt = vline(5, x0=60) | vline(2, x0=180)                                  # r = 3: medium at 3/7, thin at 4/8
    g = prepare_gt(gt, full_fov())
    m = compute_all(vline(2, x0=180), g, CUT, extras=False, sensitivity_cutoffs=(4, 8))
    assert m["cl_recall_medium"] == 0.0 and m["cl_recall_thin"] > 0.95         # main 3/7 (end spurs aside)
    assert 0.3 < m["cl_recall_thin_r4_8"] < 0.7                                              # 4/8: both lines thin
    assert m["n_skel_thin_r4_8"] > m["n_skel_thin"]
