"""Perfect foresight: actual prices and activations of the whole day."""
from __future__ import annotations

from datetime import timedelta

import numpy as np

from ..market_simulator import Context
from ..optim import next_day_value
from .common import Strategy


class PerfectForesight(Strategy):
    name = "pf"
    perfect = True

    def decide(self, ctx: Context):
        if getattr(self, "plan_day", None) != ctx.day:
            self.plan_day = ctx.day
            act = self.md.day(ctx.day)
            v = {c: act[c].to_numpy(dtype=float) for c in act.columns}
            self.plan = self.solve(ctx, v)
        return self.bids(ctx, self.plan)

    def end_curve(self, ctx: Context, v: dict) -> tuple | None:
        """Value of the energy left at midnight from a one-day plan on the next day's actual day-ahead prices."""
        sim = self.cfg["simulator"]
        if sim.get("end_of_day", "target") != "curve":
            return None
        nxt = self.md.day(ctx.day + timedelta(days=1))
        da = nxt["da"].to_numpy(float) if len(nxt) else np.array([np.nan])
        if not np.isfinite(da).any():
            da = np.asarray(v["da"], float)
        if not np.isfinite(da).any():
            return None
        da = np.where(np.isnan(da), np.nanmean(da), da)
        socs = np.linspace(self.bat.soc_min, self.bat.soc_max, int(sim.get("end_curve_points", 5)))
        return socs, next_day_value(da, self.bat, self.fees.on(ctx.day), socs)
