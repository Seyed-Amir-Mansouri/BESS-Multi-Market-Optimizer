"""Shared steps of the strategies."""
from __future__ import annotations

from datetime import date

import numpy as np
import pandas as pd

from ..battery import Battery, Fees
from ..bids import COLUMNS, band_offers, price_taking
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

    def decide(self, ctx: Context) -> pd.DataFrame:
        raise NotImplementedError

    def solve(self, ctx: Context, v: dict) -> Plan:
        idx = ctx.index
        split = date.fromisoformat(self.cfg["regimes"]["afrr_split"]["date"])
        inp = DayInput(
            n=len(idx), price={st: v[st] for st in STAGES},
            open={st: coverage(self.cfg, st, idx) for st in STAGES},
            groups=hour_groups(self.cfg, ctx.day, idx),
            cap_up=v["cap_up"], cap_dn=v["cap_dn"], share_up=v["share_up"], share_dn=v["share_dn"],
            e_up=v["e_up"], e_dn=v["e_dn"], soc0=ctx.soc0, fixed=ctx.fixed, band_fixed=ctx.band,
            ratio_up=v["ratio_up"] if ctx.day < split else None)
        plan = solve_day(inp, self.bat, self.fees, self.soft, self.time_limit)
        self.plan_end = float(plan.soc[-1])
        return plan

    def bids(self, ctx: Context, plan: Plan) -> pd.DataFrame:
        if ctx.gate == "afrr_capacity":
            return band_offers(ctx.gate, ctx.index, np.round(plan.r_up, 3), np.round(plan.r_dn, 3),
                               price=float(self.cfg["markets"]["afrr_capacity"]["price_limits"]["min"]))
        if ctx.gate not in plan.trade:
            return pd.DataFrame(columns=COLUMNS)
        trade = np.round(plan.trade[ctx.gate], 3) * coverage(self.cfg, ctx.gate, ctx.index)
        return price_taking(ctx.gate, ctx.index, trade, self.cfg["markets"][ctx.gate]["price_limits"])
