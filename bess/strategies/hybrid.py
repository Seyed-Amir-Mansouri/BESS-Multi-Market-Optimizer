"""Benchmarks mixing actual values for some inputs with LightGBM forecasts for the rest."""
from __future__ import annotations

import numpy as np

from ..market_simulator import Context
from .deterministic import LightGBM


class Hybrid(LightGBM):
    """LightGBM strategy with chosen inputs replaced by actual values and chosen markets closed."""
    name = "hyb"

    def __init__(self, md, cfg):
        super().__init__(md, cfg)
        h = cfg.get("hybrid") or {}
        self.perfect_inputs = list(h.get("perfect") or [])
        self.closed = list(h.get("closed") or [])

    def forecasts(self, ctx: Context) -> dict:
        v = super().forecasts(ctx)
        act = self.md.day(ctx.day)
        for t in self.perfect_inputs:
            v[t] = act[t].to_numpy(dtype=float)
        for m in self.closed:
            v[m] = np.full(len(ctx.index), np.nan)
        return v
