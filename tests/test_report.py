import numpy as np
import pandas as pd

import src.report as report


def _sweep(split: str) -> pd.DataFrame:
    """Synthetic sweep: per-seed mean over the 3 images = 0.1 * seed + 0.01 + threshold."""
    rows = []
    for m, seeds in (("unet", (0, 1, 2)), ("sam_lora_tiled", (0, 1, 2)), ("sam_lora_whole", (0,))):
        for s in seeds:
            for i in range(3):
                for t in (0.25, 0.5, 0.75):
                    rows.append({"model": m, "seed": s, "image_id": f"{split}_{i}", "threshold": t,
                                 "cl_recall_thin": 0.1 * s + 0.01 * i + t})
    return pd.DataFrame(rows)


def test_threshold_figure_reports_seed_mean_and_range(monkeypatch, tmp_path):
    monkeypatch.setattr(report, "load", lambda split, kind="metrics", models=report.AUTO: _sweep(split))
    d = report.fig_threshold(tmp_path / "f.png")
    assert (tmp_path / "f.png").exists()
    assert not d.duplicated(["split", "run", "threshold"]).any()          # U-Net drawn twice, listed once
    r = d[(d.split == "test") & (d.run == "unet") & (d.threshold == 0.5)].iloc[0]
    assert r.n_seeds == 3                                                  # seeds 0, 1, 2 -> 0.51, 0.61, 0.71
    assert np.isclose(r.cl_recall_thin, 0.61) and np.isclose(r.seed_min, 0.51) and np.isclose(r.seed_max, 0.71)
    w = d[d.run == "sam_lora_whole"].iloc[0]
    assert w.n_seeds == 1 and w.seed_min == w.seed_max == w.cl_recall_thin


def test_seed_label_names_the_number_of_seeds():
    assert report.seed_label("sam_lora_tiled", 3) == "SAM tiled (3 seeds)"
    assert report.seed_label("sam_lora_whole", 1) == "SAM whole (1 seed)"
