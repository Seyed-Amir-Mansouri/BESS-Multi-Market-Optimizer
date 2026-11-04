"""Battery model, bids, settlement, optimizer, naive forecasts and backtest."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from bess.backtest import run
from bess.battery import Battery, Fees
from bess.bids import band_offers, clear, price_taking
from bess.data.pubtime import market_time
from bess.forecast import naive
from bess.market_simulator import MarketData, settle
from bess.optim import STAGES, DayInput, solve_day
from bess.timeutils import QH, day_bounds_utc

TZ = "Europe/Madrid"
NO_FEES = Fees(0.0, 0.0, 0.0, 0.0)


def _idx(n=4):
    return pd.date_range("2025-11-20 00:00", periods=n, freq="15min", tz="UTC")


def test_clearing_price_taking_and_missing_session():
    idx = _idx()
    bids = price_taking("da", idx, np.array([2.0, -1.0, 0.0, 3.0]), {"min": -500.0, "max": 4000.0})
    prices = pd.Series([50.0, 60.0, 70.0, np.nan], index=idx)
    assert clear(bids, prices, "sell").tolist() == [2.0, 0.0, 0.0, 0.0]
    assert clear(bids, prices, "buy").tolist() == [0.0, 1.0, 0.0, 0.0]
    offers = band_offers("afrr_capacity", idx, np.array([1.0, 0, 0, 0]), np.array([0, 2.0, 0, 0]))
    assert clear(offers, pd.Series(5.0, index=idx), "down").tolist() == [0.0, 2.0, 0.0, 0.0]


def test_settlement_matches_afrr_worked_example(cfg):
    bat = Battery.from_config(cfg)
    act = pd.DataFrame({"da": 50.0, "ida1": np.nan, "ida2": np.nan, "ida3": np.nan, "cap_up": 10.0, "cap_dn": 6.0,
                        "share_up": 0.2, "share_dn": 0.15, "e_up": 150.0, "e_dn": 50.0}, index=_idx())
    out, soc = settle(act, {}, np.full(4, 2.0), np.full(4, 1.0), 5.0, bat, NO_FEES, 100.0)
    assert out["afrr_capacity"] == pytest.approx(26.0)
    assert out["afrr_energy"] == pytest.approx(52.5)
    assert out["degradation"] == pytest.approx(-1.65)
    assert out["profit"] == pytest.approx(76.85)
    assert soc[-1] - soc[0] == pytest.approx(-0.40 / 0.95 + 0.15 * 0.95)


def _day_input(n, da, groups=None, cap=np.nan, soc0=5.0):
    nan = np.full(n, np.nan)
    price = {st: nan for st in STAGES}
    price["da"] = da
    return DayInput(n=n, price=price, open={st: np.ones(n, bool) for st in STAGES}, groups=groups or {},
                    cap_up=np.full(n, cap), cap_dn=np.full(n, cap), share_up=np.zeros(n), share_dn=np.zeros(n),
                    e_up=np.zeros(n), e_dn=np.zeros(n), soc0=soc0)


def test_optimizer_arbitrage_respects_battery(cfg):
    bat = Battery.from_config(cfg)
    n = 96
    da = np.where(np.arange(n) < 48, 10.0, 100.0)
    plan = solve_day(_day_input(n, da), bat, NO_FEES)
    x = plan.position["ida3"]
    assert x[:48].min() < -4.9 and x[48:].max() > 4.9
    assert np.all(np.abs(x) <= bat.power + 1e-6)
    assert plan.soc.min() >= bat.soc_min - 1e-6 and plan.soc.max() <= bat.soc_max + 1e-6
    assert plan.soc[-1] == pytest.approx(bat.soc_target, abs=1e-6)
    assert plan.objective > 0


def test_daily_cycle_cap(cfg):
    bat = Battery.from_config(cfg)
    n = 96
    da = np.tile(np.r_[np.full(6, 10.0), np.full(6, 200.0)], 8)
    plan = solve_day(_day_input(n, da), bat, NO_FEES)
    x = plan.position["ida3"]
    assert np.maximum(x, 0).sum() * 0.25 <= bat.max_cycles * bat.energy + 1e-6


def test_hourly_products_trade_whole_hours(cfg):
    bat = Battery.from_config(cfg)
    n = 96
    da = 50 + 40 * np.sin(np.arange(n) / 7.0)
    groups = {"da": np.repeat(np.arange(24), 4)}
    plan = solve_day(_day_input(n, da, groups), bat, NO_FEES)
    t = plan.trade["da"].reshape(24, 4)
    assert np.allclose(t, t[:, :1], atol=1e-6)


def test_afrr_band_keeps_full_activation_headroom(cfg):
    bat = Battery.from_config(cfg)
    n = 96
    plan = solve_day(_day_input(n, np.full(n, 50.0), cap=20.0), bat, NO_FEES)
    assert plan.r_up.max() > 0 and plan.r_dn.max() > 0
    x = plan.position["ida3"]
    assert np.all(x + plan.r_up <= bat.power + 1e-6) and np.all(x - plan.r_dn >= -bat.power - 1e-6)
    assert np.all(plan.soc[1:] - plan.r_up * bat.reserve_h / bat.eta_d >= bat.soc_min - 1e-6)


def _records(cfg, days, series, market, price):
    rows = []
    for d in days:
        a, b = day_bounds_utc(d, TZ)
        t = pd.date_range(a, b, freq=QH, inclusive="left")
        rows.append(pd.DataFrame({
            "source": "test", "dataset": "test", "series": series, "delivery_start": t, "delivery_end": t + QH,
            "resolution_min": 15, "value": price(d, np.arange(len(t))), "unit": "EUR/MWh",
            "published_at": market_time(cfg, market, "results", d), "pub_source": "rule",
            "fetched_at": pd.Timestamp("2026-01-01", tz="UTC")}))
    return rows


DAYS = [date(2025, 11, 10) + timedelta(days=i) for i in range(6)]


def _market_store(cfg, store):
    da = lambda d, i: 40 + 60 * (i >= 72) + d.day
    ida = lambda d, i: 40 + 60 * (i >= 72) + d.day + 5 * np.sin(i)
    recs = _records(cfg, DAYS, "da_price_es", "da", da)
    for k in (1, 2, 3):
        recs += _records(cfg, DAYS, f"ida{k}_price_es", f"ida{k}", ida)
    store.write(pd.concat(recs, ignore_index=True))
    return store


def test_naive_forecast_uses_only_published_data(cfg, store):
    s = _market_store(cfg, store)
    md = MarketData(s, cfg, DAYS[0], DAYS[-1])
    d = DAYS[3]
    before = naive(md, d, market_time(cfg, "da", "gate", d))
    after = naive(md, d, market_time(cfg, "ida1", "gate", d))
    actual = md.day(d)["da"].to_numpy()
    prev = md.day(DAYS[2])["da"].to_numpy()
    assert np.allclose(before["da"], prev) and not np.allclose(before["da"], actual)
    assert np.allclose(after["da"], actual) and np.allclose(after["ida1"], actual)


def test_backtest_perfect_foresight_is_upper_bound(cfg, store):
    s = _market_store(cfg, store)
    pf = run(cfg, s, "pf", DAYS[1], DAYS[-1])
    det = run(cfg, s, "det", DAYS[1], DAYS[-1])
    assert len(pf) == len(det) == 5
    assert pf["profit"].sum() >= det["profit"].sum() - 1e-6
    assert pf["profit"].sum() > 0
    assert (pf["soc_end"].between(0, 10)).all() and (pf["shortfall_mwh"] < 1e-6).all()
