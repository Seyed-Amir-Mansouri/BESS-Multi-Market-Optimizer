"""Deterministic MILP re-run at each gate with naive forecasts."""
from __future__ import annotations

from ..forecast import naive
from ..market_simulator import Context
from .common import Strategy


class Deterministic(Strategy):
    name = "det"

    def decide(self, ctx: Context):
        v = naive(self.md, ctx.day, ctx.as_of, int(self.cfg["simulator"]["activation_days"]))
        return self.bids(ctx, self.solve(ctx, v))
