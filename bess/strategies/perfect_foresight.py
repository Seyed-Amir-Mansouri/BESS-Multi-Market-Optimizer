"""Perfect foresight: actual prices and activations of the whole day."""
from __future__ import annotations

from ..market_simulator import Context
from .common import Strategy


class PerfectForesight(Strategy):
    name = "pf"
    perfect = True

    def decide(self, ctx: Context):
        if ctx.gate == "da":
            act = self.md.day(ctx.day)
            v = {c: act[c].to_numpy(dtype=float) for c in act.columns}
            self.plan = self.solve(ctx, v)
        return self.bids(ctx, self.plan)
