"""Next-day value, deliverable curves and range-scaled scenarios."""
import numpy as np
import pandas as pd

from bess.battery import Battery, Fees
from bess.optim import STAGES, DayInput, next_day_value, solve_scenarios
from bess.uncertainty import ErrorModel


def _inp(cfg, n=96, price=None, curve=None, soc0=5.0):
    p = np.full(n, 50.0) if price is None else price
    nan = np.full(n, np.nan)
    return DayInput(n=n, price={st: p.copy() for st in STAGES}, open={st: np.ones(n, bool) for st in STAGES},
                    groups={}, cap_up=nan, cap_dn=nan, share_up=nan, share_dn=nan, e_up=nan, e_dn=nan,
                    soc0=soc0, end_curve=curve)


def test_next_day_value_is_concave_in_the_start_charge(cfg):
    bat, fees = Battery.from_config(cfg), Fees.from_config(cfg)
    price = np.r_[np.full(48, 20.0), np.full(48, 150.0)]
    socs = np.linspace(bat.soc_min, bat.soc_max, 5)
    v = next_day_value(price, bat, fees, socs)
    slope = np.diff(v) / np.diff(socs)
    assert np.all(np.diff(slope) <= 1e-6)
    assert slope[0] > 0


def test_end_curve_keeps_energy_when_tomorrow_pays_more(cfg):
    bat, fees = Battery.from_config(cfg), Fees.from_config(cfg)
    socs = np.linspace(bat.soc_min, bat.soc_max, 5)
    high = (socs, socs * 500.0)
    low = (socs, socs * 1.0)
    end_high = solve_scenarios([_inp(cfg, curve=high)], [1.0], bat, fees).soc[-1]
    end_low = solve_scenarios([_inp(cfg, curve=low)], [1.0], bat, fees).soc[-1]
    assert end_high > end_low


def test_deliverable_curves_keep_every_mix_within_limits(cfg):
    bat, fees = Battery.from_config(cfg), Fees.from_config(cfg)
    n = 8
    a = np.r_[np.full(4, 10.0), np.full(4, 200.0)]
    b = np.r_[np.full(4, 200.0), np.full(4, 10.0)]
    inps = [_inp(cfg, n, a), _inp(cfg, n, b)]
    for i in inps:
        i.open = {st: np.ones(n, bool) if st == "ida3" else np.zeros(n, bool) for st in STAGES}
        i.fixed = {st: np.zeros(n) for st in STAGES[:-1]}
    plan = solve_scenarios(inps, [0.5, 0.5], bat, fees, curve_stage="ida3", light=True, deliverable=True)
    hi, lo = plan.scen_trade.max(0), plan.scen_trade.min(0)
    down = bat.soc_target - np.cumsum(np.maximum(hi, 0) / bat.eta_d - np.maximum(-hi, 0) * bat.eta_c) * 0.25
    up = bat.soc_target - np.cumsum(lo * bat.eta_c) * 0.25
    assert down.min() >= bat.soc_min - 1e-6
    assert up.max() <= bat.soc_max + 1e-6


def test_scaled_errors_follow_todays_range(cfg):
    days = pd.date_range("2025-01-01", periods=40).strftime("%Y-%m-%d")
    rows = []
    for i, d in enumerate(days):
        for s in range(96):
            rows.append(("ida2", "ida1", d, s, 50.0 + (5 if i % 2 else -5), 50.0, 50.0))
    pred = pd.DataFrame(rows, columns=["target", "gate", "day", "slot", "actual", "pred", "naive"])
    q = pred[["target", "gate", "day", "slot"]].assign(q10=45.0, q90=55.0)
    q.loc[q["day"] == days[-1], ["q10", "q90"]] = [30.0, 70.0]
    em = ErrorModel(pred, cfg, q)
    e = em.scaled("ida2", "ida1", np.arange(10), pd.Timestamp(days[-1]).date())
    assert np.allclose(np.abs(e), 20.0)
