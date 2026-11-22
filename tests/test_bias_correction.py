"""Hourly bias correction of the price forecasts."""
from datetime import date, timedelta

import numpy as np
import pandas as pd

from bess.forecast import bias_corrected


def _pred(days: int = 40, drift: float = 10.0) -> pd.DataFrame:
    """DA forecasts at the DA gate that are too low by a fixed amount in the morning slots."""
    rows = []
    for k in range(days):
        d = str(date(2026, 6, 1) + timedelta(days=k))
        for s in range(96):
            gap = drift if 24 <= s < 36 else 0.0
            rows.append(("da", "da", d, s, 100.0 + gap, 100.0))
    return pd.DataFrame(rows, columns=["target", "gate", "day", "slot", "actual", "pred"])


def test_shifts_by_share_of_recent_same_slot_error(cfg):
    out = bias_corrected(cfg, _pred()).set_index(["day", "slot"])["pred"]
    share = float(cfg["forecast"]["bias_correction"]["share"])
    assert np.isclose(out[("2026-07-01", 30)], 100.0 + share * 10.0)
    assert np.isclose(out[("2026-07-01", 60)], 100.0)


def test_uses_only_days_before_the_lag(cfg):
    p = _pred()
    a = bias_corrected(cfg, p).set_index(["day", "slot"])["pred"]
    lag = int(cfg["forecast"]["bias_correction"]["lag_days"])
    late = p["day"] > str(date(2026, 7, 1) - timedelta(days=lag))
    p.loc[late, "actual"] += 500.0
    b = bias_corrected(cfg, p).set_index(["day", "slot"])["pred"]
    assert np.allclose(a.xs("2026-07-01").to_numpy(), b.xs("2026-07-01").to_numpy())
