"""Gate features, walk-forward forecasts and the LightGBM strategy."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from bess.backtest import run
from bess.data.pubtime import market_time
from bess.features import NS, Panel
from bess.forecast import accuracy, walk_forward
from bess.market_simulator import MarketData
from bess.timeutils import QH, day_bounds_utc

DAYS = [date(2025, 11, 20) + timedelta(days=i) for i in range(24)]


def _write(cfg, store, days, shift=None):
    rows = []
    for d in days:
        a, b = day_bounds_utc(d, cfg.tz)
        t = pd.date_range(a, b, freq=QH, inclusive="left")
        i = np.arange(len(t))
        base = 40 + 60 * (i >= 72) + 3 * (d.weekday() >= 5) + (shift(d) if shift else 0.0)
        for series, market, v in (("da_price_es", "da", base), ("ida1_price_es", "ida1", base + 4 * np.sin(i / 5)),
                                  ("ida2_price_es", "ida2", base + 2), ("ida3_price_es", "ida3", base - 2)):
            rows.append(pd.DataFrame({
                "source": "test", "dataset": "test", "series": series, "delivery_start": t, "delivery_end": t + QH,
                "resolution_min": 15, "value": v, "unit": "EUR/MWh",
                "published_at": market_time(cfg, market, "results", d), "pub_source": "rule",
                "fetched_at": pd.Timestamp("2026-01-01", tz="UTC")}))
    store.write(pd.concat(rows, ignore_index=True))


def _small(cfg):
    cfg.raw["data"]["start"] = str(DAYS[0])
    cfg.raw["forecast"].update(min_train_days=5, retrain_months=1, lookback_days=10)
    cfg.raw["forecast"]["lgbm"].update(n_estimators=20, min_child_samples=5)
    return cfg


def test_gate_features_use_only_published_values(cfg, store):
    _write(cfg, store, DAYS)
    md = MarketData(store, cfg, DAYS[0], DAYS[-1])
    p = Panel(md, store, cfg)
    i = p.pos[DAYS[5]]
    at_da, at_ida1 = p.features("da"), p.features("ida1")
    rows = slice(i * NS, (i + 1) * NS)
    assert at_da["da_d0"].iloc[rows].isna().all()
    assert np.allclose(at_ida1["da_d0"].iloc[rows], p.val["da"][i])
    assert np.allclose(at_da["da_lag1"].iloc[rows], p.val["da"][i - 1])
    assert at_da["ida1_d0"].iloc[rows].isna().all() and at_ida1["ida1_d0"].iloc[rows].isna().all()


def test_walk_forward_ignores_future_values(cfg, store, tmp_path):
    cfg = _small(cfg)
    _write(cfg, store, DAYS)
    a = walk_forward(cfg, store, date(2025, 12, 1), DAYS[-1], ["da", "ida1"])
    assert set(a["day"]) == {str(d) for d in DAYS if d >= date(2025, 12, 1)}
    assert a["pred"].notna().any()
    from bess.data.store import Store
    other = Store(tmp_path / "other")
    _write(cfg, other, DAYS, shift=lambda d: 500.0 if d >= date(2025, 12, 3) else 0.0)
    b = walk_forward(cfg, other, date(2025, 12, 1), DAYS[-1], ["da", "ida1"])
    early = lambda x: x[x["day"] < "2025-12-03"].sort_values(["target", "gate", "day", "slot"])["pred"].to_numpy()
    assert np.allclose(early(a), early(b), equal_nan=True)
    acc = accuracy(a)
    assert {"mae_model", "mae_naive", "skill"} <= set(acc.columns)


def test_lightgbm_strategy_runs_in_backtest(cfg, store):
    cfg = _small(cfg)
    _write(cfg, store, DAYS)
    walk_forward(cfg, store, date(2025, 12, 1), DAYS[-1], ["da", "ida1", "ida2", "ida3"])
    ml = run(cfg, store, "ml", date(2025, 12, 2), DAYS[-1])
    pf = run(cfg, store, "pf", date(2025, 12, 2), DAYS[-1])
    assert len(ml) == len(pf) > 0
    assert pf["profit"].sum() >= ml["profit"].sum() - 1e-6


def test_shrink_keeps_only_gaps_above_margin():
    from bess.strategies.deterministic import shrink
    ref = np.array([100.0, 100.0, 100.0, np.nan])
    out = shrink(np.array([95.0, 88.0, 125.0, 80.0]), ref, 8.0)
    assert np.allclose(out, [100.0, 96.0, 117.0, 80.0])


def test_error_margins_use_past_days_only():
    from bess.strategies.deterministic import error_margins
    days = [str(date(2025, 12, 1) + timedelta(days=i)) for i in range(5)]
    p = pd.DataFrame({"target": "ida1", "gate": "ida1", "day": days, "slot": 0,
                      "actual": 100.0, "pred": [104.0, 106.0, 102.0, 200.0, 200.0], "naive": 100.0})
    m = error_margins(p, 28)[("ida1", "ida1")]
    assert m.asof(pd.Timestamp("2025-12-03")) == pytest.approx(4.0)


def test_model_edge_compares_with_naive_on_past_days():
    from bess.strategies.deterministic import model_edge
    days = [str(date(2025, 12, 1) + timedelta(days=i)) for i in range(4)]
    p = pd.DataFrame({"target": "da", "gate": "da", "day": days, "slot": 0, "actual": 100.0,
                      "pred": [102.0, 103.0, 120.0, 120.0], "naive": [110.0, 110.0, 101.0, 101.0]})
    e = model_edge(p, 56)[("da", "da")]
    assert e.asof(pd.Timestamp("2025-12-02")) == pytest.approx(7.5)
    assert e.asof(pd.Timestamp("2025-12-04")) < 0
