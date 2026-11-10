"""Error model, scenario reduction, stochastic LP and bid curves."""
from datetime import date, timedelta

import numpy as np
import pandas as pd
import pytest

from bess.battery import Battery, Fees
from bess.bids import clear, curve
from bess.optim import STAGES, DayInput, solve_day, solve_scenarios
from bess.scenarios import kmedoids
from bess.uncertainty import ErrorModel

NO_FEES = Fees(0.0, 0.0, 0.0, 0.0)


def _input(n, da, cap=np.nan):
    nan = np.full(n, np.nan)
    price = {st: nan for st in STAGES}
    price["da"] = da
    return DayInput(n=n, price=price, open={st: np.ones(n, bool) for st in STAGES}, groups={},
                    cap_up=np.full(n, cap), cap_dn=np.full(n, cap), share_up=np.zeros(n), share_dn=np.zeros(n),
                    e_up=np.zeros(n), e_dn=np.zeros(n), soc0=5.0)


def test_kmedoids_returns_weighted_representatives():
    rng = np.random.default_rng(1)
    X = np.vstack([rng.normal(0, 0.1, (30, 4)), rng.normal(5, 0.1, (10, 4))])
    med, prob = kmedoids(X, np.ones(40), 2)
    assert prob.sum() == pytest.approx(1.0)
    assert sorted(np.round(prob, 2)) == [0.25, 0.75]
    assert set(med) <= set(range(40))


def test_identical_scenarios_match_deterministic(cfg):
    bat = Battery.from_config(cfg)
    n = 96
    inp = _input(n, np.where(np.arange(n) < 48, 10.0, 100.0))
    one = solve_day(inp, bat, NO_FEES)
    many = solve_scenarios([inp, inp, inp], [0.2, 0.3, 0.5], bat, NO_FEES, curve_stage="da")
    assert many.objective == pytest.approx(one.objective, rel=1e-6)


def test_curve_is_monotone_and_clears_each_scenario(cfg):
    bat = Battery.from_config(cfg)
    n = 8
    base = np.r_[np.full(4, 20.0), np.full(4, 120.0)]
    inps = [_input(n, base + d) for d in (-30.0, 0.0, 30.0)]
    plan = solve_scenarios(inps, [1 / 3] * 3, bat, NO_FEES, curve_stage="da")
    order = np.argsort(plan.scen_price, axis=0)
    q = np.take_along_axis(plan.scen_trade, order, 0)
    assert np.all(np.diff(q, axis=0) >= -1e-6)
    idx = pd.date_range("2025-11-20", periods=n, freq="15min", tz="UTC")
    bids = curve("da", idx, plan.scen_price, np.round(plan.scen_trade, 3), {"min": -500.0, "max": 4000.0})
    for s in range(3):
        p = pd.Series(plan.scen_price[s], index=idx)
        got = clear(bids, p, "sell") - clear(bids, p, "buy")
        assert np.allclose(got, np.round(plan.scen_trade[s], 3), atol=1e-3)


def test_error_pool_uses_only_published_days(cfg):
    days = [date(2025, 1, 1) + timedelta(days=i) for i in range(60)]
    p = pd.DataFrame([{"target": "da", "gate": "da", "day": str(d), "slot": s, "actual": 50.0, "pred": 40.0, "naive": 45.0}
                      for d in days for s in range(96)])
    em = ErrorModel(p, cfg)
    day = date(2025, 2, 15)
    idx, w = em.pool(day)
    assert max(em.days[idx]) <= day - timedelta(days=int(cfg["scenarios"]["lag_days"]))
    assert w.sum() == pytest.approx(1.0)
    assert np.allclose(em.matrix("da", "da", idx, True), 10.0)
    assert np.allclose(em.matrix("da", "da", idx, False), 5.0)


def test_soc_estimate_uses_metered_charge_at_gate(cfg):
    from bess.market_simulator import Simulator

    class _MD:
        pass

    sim = Simulator(_MD(), cfg)
    idx = pd.date_range("2025-11-20", periods=96, freq="15min", tz="UTC")
    actual = np.linspace(2.0, 3.0, 97)
    plan = np.linspace(5.0, 6.0, 97)
    sim.prev = (idx, actual, plan)
    gate = idx[48]
    assert sim.soc_estimate(gate, 9.0) == pytest.approx(actual[48] + plan[-1] - plan[48])
    sim.prev = None
    assert sim.soc_estimate(gate, 9.0) == 9.0


def test_nonanticipative_later_markets_share_one_plan(cfg):
    from bess import optim
    bat = Battery.from_config(cfg)
    n = 8
    base = np.r_[np.full(4, 20.0), np.full(4, 120.0)]
    inps = []
    for d in (-30.0, 30.0):
        i = _input(n, base)
        i.price = dict(i.price, ida1=base + d)
        inps.append(i)
    captured = {}
    orig = optim._scenario

    def spy(*a, **k):
        out = orig(*a, **k)
        captured.setdefault("trades", []).append(out[1]["ida1"])
        return out

    optim._scenario = spy
    try:
        solve_scenarios(inps, [0.5, 0.5], bat, NO_FEES, curve_stage="da", nonanticipative=True)
    finally:
        optim._scenario = orig
    a, b = captured["trades"]
    assert np.array_equal(a[0], b[0]) and np.array_equal(a[1], b[1])


def test_quick_set_has_one_week_per_window_outside_exclusions(cfg):
    from bess.backtest import excluded, test_set
    w = test_set(cfg, "quick")
    assert len(w) == 8
    for a, b in w:
        assert (b - a).days == 6
        assert not excluded(cfg, a) and not excluded(cfg, b)
    assert all(w[i][1] < w[i + 1][0] for i in range(len(w) - 1))


def test_compare_uses_common_days_and_share_of_foresight():
    from bess.backtest import compare
    days = ["2025-01-01", "2025-01-02", "2025-01-03"]
    mk = lambda p, ex: pd.DataFrame({"day": days, "excluded": ex, "profit": p, "penalty": [0.0] * 3, "shortfall_mwh": [0.0] * 3})
    t = compare({"pf": mk([100.0, 100.0, 100.0], [False, False, False]), "ml": mk([50.0, 40.0, 999.0], [False, False, True])})
    assert list(t["days"]) == [2, 2]
    assert t.set_index("strategy").loc["ml", "pct_of_pf"] == pytest.approx(45.0)


def _rebal_cfg():
    return {"lead_min": 60, "deadband_mwh": 1.0, "hours": 1.0}


def test_rebalance_idle_when_charge_follows_plan(cfg):
    from bess.market_simulator import rebalance
    bat = Battery.from_config(cfg)
    n = 96
    idx = pd.date_range("2025-11-20", periods=n, freq="15min", tz="UTC")
    z = np.zeros(n)
    plan = np.full(n + 1, 5.0)
    q = rebalance(idx, z, z, z, z, z, 5.0, [(idx[0] - pd.Timedelta(hours=12), plan)], bat, _rebal_cfg())
    assert np.allclose(q, 0.0)


def test_rebalance_sells_surplus_after_lead_and_stops_inside_deadband(cfg):
    from bess.market_simulator import rebalance
    bat = Battery.from_config(cfg)
    n = 96
    idx = pd.date_range("2025-11-20", periods=n, freq="15min", tz="UTC")
    z = np.zeros(n)
    a_dn = np.zeros(n)
    a_dn[:4] = 4.0
    r = np.full(n, 1.0)
    plan = np.full(n + 1, 5.0)
    q = rebalance(idx, z, z, a_dn, r, r, 5.0, [(idx[0] - pd.Timedelta(hours=12), plan)], bat, _rebal_cfg())
    assert np.allclose(q[:6], 0.0)
    assert q[6:16].sum() > 0 and np.all(q >= -1e-9)
    assert np.all(q <= bat.power - 1.0 + 1e-9)
    gained = 4.0 * 4 * 0.25 * bat.eta_c
    sold = q.sum() * 0.25 / bat.eta_d
    assert sold == pytest.approx(gained - 1.0, abs=0.05)


def test_headroom_keeps_power_free_next_to_the_band(cfg):
    bat = Battery.from_config(cfg)
    n = 8
    inp = _input(n, np.full(n, 50.0), cap=80.0)
    free = solve_day(inp, bat, NO_FEES)
    inp.headroom = 1.0
    kept = solve_day(inp, bat, NO_FEES)
    assert free.r_up.max() > bat.power - 1.0 + 0.1
    assert np.all(kept.position["ida3"] + kept.r_up <= bat.power - 1.0 + 1e-6)
    assert np.all(kept.position["ida3"] - kept.r_dn >= -bat.power + 1.0 - 1e-6)


def test_rebalance_sells_stay_inside_the_daily_discharge_budget(cfg):
    from bess.market_simulator import rebalance
    bat = Battery.from_config(cfg)
    n = 96
    idx = pd.date_range("2025-11-20", periods=n, freq="15min", tz="UTC")
    z = np.zeros(n)
    a_dn = np.zeros(n)
    a_dn[:4] = 4.0
    r = np.full(n, 1.0)
    plan = np.full(n + 1, 5.0)
    x = np.zeros(n)
    x[40:44] = 4.0
    q = rebalance(idx, x, z, a_dn, r, r, 5.0, [(idx[0] - pd.Timedelta(hours=12), plan)], bat, _rebal_cfg(), budget=4.5)
    assert np.maximum(q, 0).sum() * 0.25 <= 0.5 + 1e-6


def test_settle_stops_discharge_at_the_cycle_budget_and_counts_shortfall(cfg):
    from bess.market_simulator import settle
    bat = Battery.from_config(cfg)
    n = 8
    idx = pd.date_range("2025-11-20", periods=n, freq="15min", tz="UTC")
    act = pd.DataFrame({c: 50.0 for c in ["da", "ida1", "ida2", "ida3", "cap_up", "cap_dn", "e_up", "e_dn"]}, index=idx)
    act["share_up"] = 0.0
    act["share_dn"] = 0.0
    trades = {"da": np.full(n, 4.0)}
    z = np.zeros(n)
    free, _ = settle(act, trades, z, z, 9.0, bat, NO_FEES, 100.0)
    capped, _ = settle(act, trades, z, z, 9.0, bat, NO_FEES, 100.0, budget=5.0)
    assert free["discharge_mwh"] == pytest.approx(8.0)
    assert capped["discharge_mwh"] == pytest.approx(5.0)
    assert capped["shortfall_mwh"] == pytest.approx(3.0)


def test_settle_prices_rebalancing_at_continuous_trades_where_known(cfg):
    from bess.market_simulator import settle
    bat = Battery.from_config(cfg)
    n = 4
    idx = pd.date_range("2025-11-20", periods=n, freq="15min", tz="UTC")
    act = pd.DataFrame({c: 50.0 for c in ["da", "ida1", "ida2", "ida3", "cap_up", "cap_dn", "e_up", "e_dn"]}, index=idx)
    act["share_up"] = 0.0
    act["share_dn"] = 0.0
    z = np.zeros(n)
    q = np.array([1.0, -1.0, 1.0, 0.0])
    mic = np.array([80.0, 80.0, np.nan, 80.0])
    out, _ = settle(act, {}, z, z, 5.0, bat, NO_FEES, 100.0, rebal=q, spread=10.0, mic=mic, slippage=2.0)
    assert out["rebalance"] == pytest.approx((78.0 - 82.0 + 40.0) * 0.25)


def test_planned_charge_keeps_the_margin_from_both_limits(cfg):
    bat = Battery.from_config(cfg)
    n = 96
    da = np.where((np.arange(n) >= 24) & (np.arange(n) < 72), 10.0, 200.0)
    free = solve_day(_input(n, da), bat, NO_FEES)
    inp = _input(n, da)
    inp.soc_margin = np.full(n + 1, 1.5)
    kept = solve_day(inp, bat, NO_FEES)
    assert free.soc.min() < 0.5 and free.soc.max() > 9.5
    assert kept.soc[1:].min() >= 1.5 - 1e-6 and kept.soc.max() <= 8.5 + 1e-6
