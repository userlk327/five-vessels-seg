import numpy as np

from src.agreement import score_pair

S = {"cutoffs": (3, 7), "center_tol": 1, "nsd_tol": 2.0, "box_sizes": (2, 4, 8, 16, 32), "sens_cutoffs": (4, 8)}


def _masks():
    m = np.zeros((96, 96), bool)
    m[20:22, 10:86] = True          # thin vessel (radius < 3)
    m[50:66, 10:86] = True          # thick vessel
    return m, np.ones_like(m)


def test_identical_annotations_agree_perfectly():
    m, fov = _masks()
    r = score_pair(m, m.copy(), fov, S)
    for k in ("dice", "iou", "precision", "recall", "cl_recall_thin", "cl_recall_thick", "cldice", "nsd"):
        assert np.isclose(r[k], 1.0), k


def test_direction_matters_for_recall_but_not_dice():
    a, fov = _masks()
    b = a.copy()
    b[20:22, :] = False             # the other annotation misses the thin vessel
    ab, ba = score_pair(a, b, fov, S), score_pair(b, a, fov, S)
    assert np.isclose(ab["dice"], ba["dice"])
    assert ab["cl_recall_thin"] == 0.0 and np.isclose(ba["precision"], ab["recall"])
