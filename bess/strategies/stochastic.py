"""Two-stage stochastic LP over reduced error scenarios with current-market bid curves."""
from __future__ import annotations

import numpy as np

from ..forecast import known, load_predictions, load_quantiles
from ..market_simulator import Context
from ..optim import STAGES, solve_scenarios
from ..scenarios import ScenarioGenerator
from ..uncertainty import ErrorModel
from .deterministic import LightGBM


class Stochastic(LightGBM):
    """LightGBM point forecasts plus bootstrapped errors of similar past days."""
    name = "sto"

    def __init__(self, md, cfg):
        super().__init__(md, cfg)
        quant = load_quantiles(cfg) if cfg["scenarios"].get("conditional") else None
        self.gen = ScenarioGenerator(ErrorModel(load_predictions(cfg), cfg, quant), cfg)
        self.shortfall = float(cfg["simulator"]["shortfall_penalty_eur_per_mwh"])
        self.n_scen = []
        self.fallbacks = 0
        sc = cfg.get("stochastic") or {}
        self.band_mode = str(sc.get("band", "deterministic"))
        self.nonant = bool(sc.get("nonanticipative", True))

    def scenarios(self, ctx: Context, v: dict) -> tuple[list, np.ndarray]:
        slots = self.md.slot.loc[ctx.index].to_numpy()
        kn = {t: known(self.md, t, ctx.day, ctx.as_of) for t in self.source}
        return self.gen.generate(ctx.day, v, kn, slots, self.source)

    def decide(self, ctx: Context):
        v = self.forecasts(ctx)
        band = ctx.band
        if self.band_mode == "deterministic" and band is None:
            det = self.solve(ctx, v)
            if ctx.gate == "afrr_capacity":
                return self.bids(ctx, det)
            band = (det.r_up, det.r_dn)
        scen, prob = self.scenarios(ctx, v)
        self.n_scen.append(len(scen))
        inps = [self.day_input(ctx, s) for s in scen]
        for i in inps:
            i.band_fixed = band
        cur = ctx.gate if ctx.gate in STAGES else None
        try:
            plan = solve_scenarios(inps, prob, self.bat, self.fees.on(ctx.day), self.soft, self.time_limit,
                                   shortfall=self.shortfall, curve_stage=cur, light=True,
                                   nonanticipative=self.nonant)
        except RuntimeError:
            self.fallbacks += 1
            return self.bids(ctx, self.solve(ctx, v))
        self.plan_end = float(plan.soc[-1])
        self.plan_soc = plan.soc
        return self.bids(ctx, plan)
