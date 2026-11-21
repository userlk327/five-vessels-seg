"""2x2 ablation effects and the bootstrap CI on synthetic per-image values with known effects."""
import numpy as np
import pandas as pd
import pytest

from src.analysis import ABLATION, ablation_effects, bootstrap_ci


def test_effects_recover_known_values():
    rng = np.random.RandomState(0)
    n = 200
    base = rng.uniform(0.5, 0.7, n)
    # attention +0.04, multi-scale +0.02, interaction +0.01 (only when both)
    vals = {"unet": base, "attention_unet": base + 0.04, "multiscale_unet": base + 0.02,
            "ms_attention_unet": base + 0.04 + 0.02 + 0.01}
    per_image = pd.DataFrame({(m, "dice"): v for m, v in vals.items()}, index=[f"i{k}" for k in range(n)])
    e = ablation_effects(per_image, metrics=["dice"], n_boot=500).set_index("effect")["mean"]
    assert e["attention"] == pytest.approx(0.045) and e["multi-scale"] == pytest.approx(0.025)
    assert e["interaction"] == pytest.approx(0.01)
    assert list(ABLATION) == list(vals)


def test_bootstrap_ci_covers_the_mean_and_shrinks_with_n():
    rng = np.random.RandomState(1)
    small, large = rng.normal(0.1, 1, 50), rng.normal(0.1, 1, 5000)
    lo, hi = bootstrap_ci(small, 2000)
    assert lo < small.mean() < hi
    lo2, hi2 = bootstrap_ci(large, 2000)
    assert (hi2 - lo2) < (hi - lo) / 5
    assert bootstrap_ci(np.r_[small, np.nan], 2000) == bootstrap_ci(small, 2000)
