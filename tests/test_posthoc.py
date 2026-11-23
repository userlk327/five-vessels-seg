"""Synthetic checks of the post hoc metrics (src/posthoc.py): spur pruning and width-stratified precision."""
import numpy as np
from scipy import ndimage as ndi

from src.metrics import skeleton, width_class
from src.posthoc import prune_spurs, width_precision

CUT = (3, 7)


def test_prune_removes_short_terminal_branch_only():
    sk = np.zeros((100, 100), bool)
    sk[50, 10:91] = True                      # main line
    sk[45:50, 30] = True                      # 5-px terminal branch at a junction -> spur (radius 10)
    sk[20:50, 70] = True                      # 30-px terminal branch -> kept
    sk[90, 10:14] = True                      # isolated short segment (no junction) -> kept
    r = np.full(sk.shape, 10.0)
    out = prune_spurs(sk, r)
    assert not out[45:49, 30].any()
    assert out[20:49, 70].all() and out[90, 10:14].all() and out[50, 12:89].all()


def test_prune_removes_boundary_spurs_of_a_thick_vessel():
    m = np.zeros((120, 260), bool)
    m[40:61, 30:230] = True                   # 21 px wide bar: radius about 10.5
    for x in range(50, 220, 25):
        m[37:40, x:x + 3] = True              # small bumps on the boundary -> medial-axis spurs with thin tips
    sk, edt = skeleton(m), ndi.distance_transform_edt(m)
    thin_before = int((width_class(edt[sk], CUT) == 0).sum())
    pr = prune_spurs(sk, edt)
    assert thin_before > 0
    assert int((width_class(edt[pr], CUT) == 0).sum()) == 0
    assert pr.sum() > 150                      # the medial line itself stays


def test_prune_keeps_a_thin_vessel_leaving_a_thick_one():
    m = np.zeros((200, 260), bool)
    m[40:61, 30:230] = True                   # thick vessel
    m[61:140, 129:132] = True                 # thin vessel (3 px wide), 79 px long, attached below
    sk, edt = skeleton(m), ndi.distance_transform_edt(m)
    pr = prune_spurs(sk, edt)
    assert pr[70:135, 129:132].sum() >= 60


def test_width_precision_inside_and_outside():
    fov = np.ones((100, 100), bool)
    gt = np.zeros((100, 100), bool)
    gt[48:53, 10:90] = True
    gt_dil = ndi.binary_dilation(gt, np.ones((3, 3), bool))
    pred_in = np.zeros_like(gt)
    pred_in[49:52, 15:85] = True               # 3-px band inside the GT: thin class, all hits
    out = width_precision(pred_in, fov, gt_dil, CUT)
    assert out["w_prec_thin"] == 1.0 and np.isnan(out["w_prec_thick"])
    pred_out = np.zeros_like(gt)
    pred_out[20:23, 15:85] = True              # 3-px band far from the GT: thin class, all misses
    out = width_precision(pred_out, fov, gt_dil, CUT)
    assert out["w_prec_thin"] == 0.0 and out["n_pskel_thin"] > 50


def test_width_precision_empty_prediction_is_nan():
    z = np.zeros((50, 50), bool)
    out = width_precision(z, ~z, z, CUT)
    assert all(np.isnan(out[f"w_prec_{c}"]) for c in ("thin", "medium", "thick", "all"))
