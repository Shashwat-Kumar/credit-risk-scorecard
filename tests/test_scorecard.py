import numpy as np
import pandas as pd
from scorecard.woe import fit_bins, information_value, transform
from scorecard.train import ks_stat, run


def test_woe_direction_and_iv():
    rng = np.random.default_rng(0)
    x = pd.Series(rng.normal(size=5000))
    y = pd.Series((rng.random(5000) < 1 / (1 + np.exp(-2 * x))).astype(int))
    t = fit_bins(x, y)
    assert t["woe"].iloc[0] > t["woe"].iloc[-1]   # higher x = riskier = lower WoE
    assert information_value(t) > 0.3
    assert len(transform(x, t)) == 5000


def test_noise_has_low_iv():
    rng = np.random.default_rng(1)
    t = fit_bins(pd.Series(rng.normal(size=5000)), pd.Series(rng.integers(0, 2, 5000)))
    assert information_value(t) < 0.02


def test_ks_perfect():
    assert ks_stat([0, 0, 1, 1], [0.1, 0.2, 0.8, 0.9]) == 1.0


def test_pipeline(tmp_path):
    m = run(n=6000, seed=3, out=tmp_path)
    assert m["auc"] > 0.7 and m["ks"] > 0.3
    assert (tmp_path / "scorecard_points.csv").exists()
