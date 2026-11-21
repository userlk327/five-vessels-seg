"""Evaluation pipeline: the threshold sweep equals full scoring at every threshold, the
threshold is chosen on val only, and test scoring is guarded. Small synthetic images."""
import json

import cv2
import numpy as np
import pandas as pd
import pytest
import yaml

from src.data import write_npz_archive
from src.evaluate import score_run, select_threshold, sweep_metrics, threshold_grid
from src.metrics import compute_all, prepare_gt
from src.resample import binarize, block_mean, upsample_prob
from src.utils import load_config

N = 256


def synthetic(seed=0):
    """GT with thin (w 2), medium (w 8) and thick (w 16) vessels, a disk FOV, and a noisy
    low-resolution probability map upsampled to N."""
    rng = np.random.RandomState(seed)
    gt = np.zeros((N, N), np.uint8)
    cv2.line(gt, (20, 30), (230, 60), 1, 2)
    cv2.line(gt, (40, 200), (220, 120), 1, 8)
    cv2.line(gt, (128, 10), (110, 250), 1, 16)
    yy, xx = np.mgrid[:N, :N]
    fov = (yy - 128) ** 2 + (xx - 128) ** 2 < 120 ** 2
    low = np.clip(block_mean(gt > 0, N // 4) + rng.normal(0, 0.15, (N // 4, N // 4)), 0, 1).astype(np.float16)
    return gt > 0, fov, low


def test_threshold_grid():
    g = threshold_grid(load_config("configs/base.yaml"))
    assert len(g) == 19 and g[0] == 0.05 and g[-1] == 0.95 and 0.5 in g
    assert all(isinstance(t, float) for t in g)


@pytest.mark.parametrize("tol", [1, 2])
def test_sweep_equals_full_scoring(tol):
    gt, fov, low = synthetic()
    g = prepare_gt(gt, fov)
    up = upsample_prob(low.astype(np.float32), N)
    grid = [0.05, 0.25, 0.5, 0.75, 0.95]
    rows = sweep_metrics(up, g, grid, (3, 7), tol, (4, 8))
    for t, row in zip(grid, rows):
        full = compute_all(binarize(up, t), g, (3, 7), tol, 2.0, (2, 4, 8), extras=False, sensitivity_cutoffs=(4, 8))
        for k, v in row.items():
            if k != "threshold":
                assert v == pytest.approx(full[k], nan_ok=True), (t, k)


def test_select_threshold_highest_mean_dice_lowest_on_tie():
    df = pd.DataFrame({"threshold": [0.3, 0.3, 0.4, 0.4, 0.5, 0.5],
                       "dice": [0.7, 0.9, 0.85, 0.75, 0.8, 0.8]})                # means 0.8, 0.8, 0.8
    assert select_threshold(df)[0] == 0.3
    df.loc[5, "dice"] = 0.81
    assert select_threshold(df)[0] == 0.5


def test_score_run_end_to_end(tmp_path):
    """val: sweep -> threshold -> per-image metrics; test: refused without --final-test,
    then scored at the val threshold, which is never re-chosen on test."""
    cfg = load_config("configs/base.yaml")
    cfg["data"]["native_resolution"] = N
    cfg["evaluation"]["fractal_box_sizes"] = [2, 4, 8, 16]
    cfg["paths"] = {k: str(tmp_path / k) for k in ("fives_dir", "cache_dir", "outputs_dir", "results_dir")}
    cfg["paths"]["splits_csv"] = str(tmp_path / "splits.csv")
    ids = {"val": ["train_1_A", "train_2_D"], "test": ["test_1_G"]}
    pd.DataFrame([{"image_id": i, "category": i[-1], "split": s} for s, l in ids.items() for i in l] +
                 [{"image_id": "train_3_N", "category": "N", "split": "train"}]).to_csv(cfg["paths"]["splits_csv"], index=False)
    probs = {}
    for k, i in enumerate(ids["val"] + ids["test"]):
        gt, fov, low = synthetic(k)
        for sub, arr in ((f"fives_dir/{i.split('_')[0]}/masks", gt), ("cache_dir/fov_2048", fov)):
            (tmp_path / sub).mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(tmp_path / sub / f"{i}.png"), arr.astype(np.uint8) * 255)
        probs[i] = low
    for s in ("val", "test"):
        write_npz_archive(tmp_path / f"{s}.npz", {i: probs[i] for i in ids[s]})
    out = tmp_path / "eval"
    val = score_run(cfg, model="m", seed=0, split="val", archive=tmp_path / "val.npz", out_dir=out, workers=1)
    thr = json.loads((out / "val_threshold.json").read_text())
    sweep = pd.read_csv(out / "val_sweep.csv")
    assert len(sweep) == 2 * 19 and thr["threshold"] == select_threshold(sweep)[0]
    assert (val.threshold == thr["threshold"]).all() and list(val.image_id) == ids["val"]
    assert list(val.columns[:6]) == ["model", "seed", "split", "category", "image_id", "threshold"]
    # the per-image metrics at the chosen threshold agree with the sweep row at that threshold
    s = sweep[sweep.threshold == thr["threshold"]].set_index("image_id")
    for c in ("dice", "precision", "recall", "cl_recall_thin", "cl_recall_thin_r4_8"):
        assert np.allclose(val.set_index("image_id")[c], s.loc[val.image_id, c], equal_nan=True), c
    with pytest.raises(SystemExit):
        score_run(cfg, model="m", seed=0, split="test", archive=tmp_path / "test.npz", out_dir=out, workers=1)
    test = score_run(cfg, model="m", seed=0, split="test", archive=tmp_path / "test.npz", out_dir=out, workers=1,
                     final_test=True)
    assert (test.threshold == thr["threshold"]).all()
    assert json.loads((out / "val_threshold.json").read_text()) == thr                # unchanged by the test run
    # an archive that is not exactly the split's images is refused
    write_npz_archive(tmp_path / "short.npz", {ids["val"][0]: probs[ids["val"][0]]})
    with pytest.raises(ValueError):
        score_run(cfg, model="m", seed=0, split="val", archive=tmp_path / "short.npz", out_dir=out, workers=1)


def test_portable_path_never_writes_personal_absolute_paths(monkeypatch, tmp_path):
    from src.utils import PROJECT_ROOT, portable_path
    assert portable_path(PROJECT_ROOT / "outputs" / "unet" / "seed0" / "val" / "probs.npz") == "outputs/unet/seed0/val/probs.npz"
    monkeypatch.setenv("WORK", str(tmp_path))
    assert portable_path(tmp_path / "fives-vessel-seg" / "x.npz") == "$WORK/fives-vessel-seg/x.npz"
