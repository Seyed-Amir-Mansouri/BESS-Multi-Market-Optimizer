"""Conformal recalibration of quantile forecasts."""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from bess.calibration import calibrate, coverage


def _frames(days: int = 200, seed: int = 0, overconfident: float = 0.5):
    """Point forecasts with noise whose spread varies by day, and q10/q90 that are too narrow."""
    rng = np.random.default_rng(seed)
    rows_p, rows_q = [], []
    for k in range(days):
        d = str(date(2025, 1, 1) + timedelta(days=k))
        sd = 5.0 + 10.0 * (k % 3)
        pred = 50.0 + rng.normal(0, 5, 96)
        act = pred + rng.normal(0, sd, 96)
        half = 1.2816 * sd * overconfident
        for s in range(96):
            rows_p.append(("da", "da", d, s, act[s], pred[s]))
            rows_q.append(("da", "da", d, s, pred[s] - half, pred[s] + half))
    pred = pd.DataFrame(rows_p, columns=["target", "gate", "day", "slot", "actual", "pred"])
    quant = pd.DataFrame(rows_q, columns=["target", "gate", "day", "slot", "q10", "q90"])
    return pred, quant


def test_calibrated_quantiles_hit_their_levels(cfg):
    pred, quant = _frames()
    raw = ((pred["actual"] < quant["q10"]).mean(), (pred["actual"] > quant["q90"]).mean())
    assert raw[0] > 0.2 and raw[1] > 0.2
    cal = calibrate(cfg, pred, quant)
    cov = coverage(cal, sorted(pred["day"].unique())[60:]).iloc[0]
    assert abs(cov["below_q10"] - 10) < 2 and abs(cov["below_q90"] - 90) < 2
    assert abs(cov["below_q1"] - 1) < 1 and abs(cov["below_q99"] - 99) < 1


def test_calibration_uses_only_days_published_before(cfg):
    pred, quant = _frames(days=120)
    a = calibrate(cfg, pred, quant)[("da", "da")]
    lag = int(cfg["forecast"]["conformal"]["lag_days"])
    late = pred["day"] > str(date(2025, 4, 16) - timedelta(days=lag))
    pred.loc[late, "actual"] += 1000.0
    b = calibrate(cfg, pred, quant)[("da", "da")]
    i = a.pos["2025-04-16"]
    assert np.allclose(a.zq[i], b.zq[i])


def test_quantile_errors_rise_with_rank(cfg):
    pred, quant = _frames(days=100)
    c = calibrate(cfg, pred, quant)[("da", "da")]
    i = c.pos["2025-04-01"]
    u = np.tile(np.array([[0.02], [0.3], [0.5], [0.7], [0.98]]), (1, 96))
    e = c.errors(u, i)
    assert (np.diff(e, axis=0) >= 0).all() and e[0].mean() < 0 < e[-1].mean()
