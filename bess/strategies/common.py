"""Shared steps of the strategies."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..battery import Battery, Fees
from ..bids import COLUMNS, band_offers, curve, price_taking
from ..config import Config
from ..market_simulator import Context, coverage, hour_groups
from ..optim import STAGES, DayInput, Plan, next_day_value, solve_day


class Strategy:
    name = "base"
    perfect = False

    def __init__(self, md, cfg: Config):
        self.md, self.cfg = md, cfg
        self.bat, self.fees = Battery.from_config(cfg), Fees.from_config(cfg)
        sim = cfg["simulator"]
        self.soft = float(sim["soft_penalty_eur_per_mwh"])
        self.time_limit = float(sim["solver_time_limit_s"])
        self.plan_end = None
        self.plan_soc = None
        rb = sim.get("rebalance") or {}
        self.headroom = float(rb.get("headroom_mw", 0.0)) if rb.get("enabled") and not self.perfect else 0.0
        self.cycle_reserve = 0.0 if self.perfect else float(sim.get("cycle_reserve_mwh", 0.0))
        self.cycle_activation = 1.0 if self.perfect else float(sim.get("cycle_activation_factor", 1.0))
        m = None if self.perfect else sim.get("soc_margin_mwh")
        self.soc_margin = None if m is None else np.resize(np.asarray(m, float), 24)
        self.afrr = bool(cfg["markets"]["afrr_capacity"].get("enabled", True))
        self.interim_cost = float(sim.get("interim_cycle_cost_eur_per_mwh", 0.0))
        self.interim_exclusive = bool(sim.get("interim_exclusive", False))

    def decide(self, ctx: Context) -> pd.DataFrame:
        raise NotImplementedError

    def day_input(self, ctx: Context, v: dict) -> DayInput:
        idx = ctx.index
        if ctx.own:
            v = dict(v)
            for k, o in ctx.own.items():
                v[k] = np.where(np.isnan(o), np.asarray(v[k], float), o)
        if not self.afrr:
            v = dict(v, cap_up=np.full(len(idx), np.nan), cap_dn=np.full(len(idx), np.nan))
        split = date.fromisoformat(self.cfg["regimes"]["afrr_split"]["date"])
        return DayInput(
            n=len(idx), price={st: v[st] for st in STAGES},
            open={st: coverage(self.cfg, st, idx) for st in STAGES},
            groups=hour_groups(self.cfg, ctx.day, idx),
            cap_up=v["cap_up"], cap_dn=v["cap_dn"], share_up=v["share_up"], share_dn=v["share_dn"],
            e_up=v["e_up"], e_dn=v["e_dn"], soc0=ctx.soc0, fixed=ctx.fixed, band_fixed=ctx.band,
            ratio_up=v["ratio_up"] if ctx.day < split else None, end_price=self.end_price(v),
            end_curve=self.end_curve(ctx, v),
            headroom=self.headroom, cycle_reserve=self.cycle_reserve,
            cycle_activation=self.cycle_activation, soc_margin=self.margin_profile(idx), interim_cost=self.interim_cost,
            interim_exclusive=self.interim_exclusive)

    def margin_profile(self, idx: pd.DatetimeIndex) -> np.ndarray | None:
        """Planned charge margin from each limit per quarter-hour boundary, by local hour."""
        if self.soc_margin is None:
            return None
        m = self.soc_margin[idx.tz_convert(self.cfg.tz).hour]
        return np.r_[m, m[-1]]

    def end_price(self, v: dict) -> float | None:
        """Value of stored energy at midnight, or None for a fixed end-of-day target."""
        if self.cfg["simulator"].get("end_of_day", "target") != "value":
            return None
        da = np.asarray(v["da"], float)
        return float(max(np.nanmean(da), 0.0)) if np.isfinite(da).any() else None

    def end_curve(self, ctx: Context, v: dict) -> tuple | None:
        """Value of the energy left at midnight from a one-day plan on today's day-ahead prices."""
        sim = self.cfg["simulator"]
        if sim.get("end_of_day", "target") != "curve":
            return None
        da = np.asarray(v["da"], float)
        if not np.isfinite(da).any():
            return None
        key = (ctx.day, ctx.gate, float(np.nansum(da)))
        if getattr(self, "_curve_key", None) != key:
            da = np.where(np.isnan(da), np.nanmean(da), da)
            socs = np.linspace(self.bat.soc_min, self.bat.soc_max, int(sim.get("end_curve_points", 5)))
            self._curve = (socs, next_day_value(da, self.bat, self.fees.on(ctx.day), socs))
            self._curve_key = key
        return self._curve

    def solve(self, ctx: Context, v: dict) -> Plan:
        plan = solve_day(self.day_input(ctx, v), self.bat, self.fees.on(ctx.day), self.soft, self.time_limit)
        self.plan_end = float(plan.soc[-1])
        self.plan_soc = plan.soc
        return plan

    def bids(self, ctx: Context, plan: Plan) -> pd.DataFrame:
        if ctx.gate == "afrr_capacity":
            return band_offers(ctx.gate, ctx.index, np.round(plan.r_up, 3), np.round(plan.r_dn, 3),
                               price=float(self.cfg["markets"]["afrr_capacity"]["price_limits"]["min"]))
        if ctx.gate not in plan.trade:
            return pd.DataFrame(columns=COLUMNS)
        cov = coverage(self.cfg, ctx.gate, ctx.index)
        limits = self.cfg["markets"][ctx.gate]["price_limits"]
        if plan.scen_trade is not None:
            return curve(ctx.gate, ctx.index, plan.scen_price, np.round(plan.scen_trade, 3) * cov, limits)
        return price_taking(ctx.gate, ctx.index, np.round(plan.trade[ctx.gate], 3) * cov, limits)
