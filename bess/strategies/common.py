"""Shared steps of the strategies."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..battery import Battery, Fees
from ..bids import COLUMNS, band_offers, curve, price_taking
from ..config import Config
from ..market_simulator import Context, coverage, hour_groups
from ..optim import STAGES, DayInput, Plan, solve_day


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

    def decide(self, ctx: Context) -> pd.DataFrame:
        raise NotImplementedError

    def day_input(self, ctx: Context, v: dict) -> DayInput:
        idx = ctx.index
        if ctx.own:
            v = dict(v)
            for k, o in ctx.own.items():
                v[k] = np.where(np.isnan(o), np.asarray(v[k], float), o)
        split = date.fromisoformat(self.cfg["regimes"]["afrr_split"]["date"])
        return DayInput(
            n=len(idx), price={st: v[st] for st in STAGES},
            open={st: coverage(self.cfg, st, idx) for st in STAGES},
            groups=hour_groups(self.cfg, ctx.day, idx),
            cap_up=v["cap_up"], cap_dn=v["cap_dn"], share_up=v["share_up"], share_dn=v["share_dn"],
            e_up=v["e_up"], e_dn=v["e_dn"], soc0=ctx.soc0, fixed=ctx.fixed, band_fixed=ctx.band,
            ratio_up=v["ratio_up"] if ctx.day < split else None, end_price=self.end_price(v),
            headroom=self.headroom)

    def end_price(self, v: dict) -> float | None:
        """Value of stored energy at midnight, or None for a fixed end-of-day target."""
        if self.cfg["simulator"].get("end_of_day", "target") != "value":
            return None
        da = np.asarray(v["da"], float)
        return float(max(np.nanmean(da), 0.0)) if np.isfinite(da).any() else None

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
