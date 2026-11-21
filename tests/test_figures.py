"""Smoke tests for figure code, so a plotting-API change fails here and not in a job."""
import json
import warnings

import numpy as np
import pandas as pd


def test_widths_plot_runs_without_deprecations(tmp_path, monkeypatch):
    import src.evaluate as E
    monkeypatch.setattr(E, "CHECKS_DIR", tmp_path)
    (tmp_path / "width_radius_summary.json").write_text(json.dumps({
        "images": 3, "skeleton_px": 600, "proposed_cutoffs": {"thin_below": 3, "thick_above": 7},
        "shares_proposed": {"thin": 0.1, "medium": 0.5, "thick": 0.4}}))
    pd.DataFrame({"radius_px": [1.0, 2.0, 3.0, 5.0, 8.0], "skeleton_px": [50, 50, 100, 200, 200]}).to_csv(
        tmp_path / "width_radius_hist.csv", index=False)
    rng = np.random.RandomState(0)
    pd.DataFrame({"image_id": ["a", "b", "c"], "category": list("ADG"), "n_skel": [200] * 3,
                  "share_thin": rng.rand(3), "share_medium": rng.rand(3), "share_thick": rng.rand(3)}).to_csv(
        tmp_path / "width_radius_per_image.csv", index=False)
    with warnings.catch_warnings():
        warnings.simplefilter("error")                                       # deprecations become failures
        E.cmd_widths_plot({})
    assert (tmp_path / "width_radius_hist.png").stat().st_size > 10_000
