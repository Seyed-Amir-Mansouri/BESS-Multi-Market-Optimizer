"""Two-stage stochastic LP over reduced error scenarios with current-market bid curves."""
from __future__ import annotations

import numpy as np
import pandas as pd

from ..calibration import calibrate
from ..forecast import known, load_predictions, load_quantiles
from ..bids import price_taking
from ..market_simulator import Context, coverage
from ..optim import STAGES, solve_scenarios
from ..scenarios import ScenarioGenerator
from ..uncertainty import ErrorModel
from .deterministic import LightGBM


class Stochastic(LightGBM):
    """LightGBM point forecasts plus bootstrapped errors of similar past days."""
    name = "sto"

    def __init__(self, md, cfg):
        super().__init__(md, cfg)
        pred = load_predictions(cfg, bias=True)
        quant = load_quantiles(cfg) if cfg["scenarios"].get("conditional") or cfg["scenarios"].get("method") == "quantile" else None
        cal = calibrate(cfg, pred, quant) if cfg["scenarios"].get("method") == "quantile" else None
        self.gen = ScenarioGenerator(ErrorModel(pred, cfg, quant if cfg["scenarios"].get("conditional") else None), cfg, cal)
        self.shortfall = float(cfg["simulator"]["shortfall_penalty_eur_per_mwh"])
        self.n_scen = []
        self.fallbacks = 0
        sc = cfg.get("stochastic") or {}
        self.band_mode = str(sc.get("band", "deterministic"))
        na = sc.get("nonanticipative", True)
        self.nonant = na if isinstance(na, (bool, str)) else int(na)
        self.deliverable = bool(sc.get("deliverable", False))
        self.gates = sc.get("gates")
        risk = sc.get("risk") or {}
        self.risk_weight = float(risk.get("weight", 0.0))
        self.risk_alpha = float(risk.get("alpha", 0.9))
        self.single = str(sc.get("single_bid", "off"))
        self.single_margin = float(sc.get("single_bid_margin_eur", 5.0))

    def scenarios(self, ctx: Context, v: dict) -> tuple[list, np.ndarray]:
        slots = self.md.slot.loc[ctx.index].to_numpy()
        kn = {t: known(self.md, t, ctx.day, ctx.as_of) for t in self.source}
        return self.gen.generate(ctx.day, v, kn, slots, self.source, ctx.gate)

    def decide(self, ctx: Context):
        if self.gates and ctx.gate not in self.gates:
            return super().decide(ctx)
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
                                   nonanticipative=self.nonant, deliverable=self.deliverable,
                                   risk_weight=self.risk_weight, risk_alpha=self.risk_alpha,
                                   single=self.single != "off")
        except RuntimeError:
            self.fallbacks += 1
            return self.bids(ctx, self.solve(ctx, v))
        self.plan_end = float(plan.soc[-1])
        self.plan_soc = plan.soc
        if self.single != "off" and plan.scen_trade is not None:
            return self.single_bids(ctx, plan)
        return self.bids(ctx, plan)

    def single_bids(self, ctx: Context, plan) -> pd.DataFrame:
        """One quantity per quarter-hour, price-taking or limited at the mean scenario price less a margin."""
        q = np.round(plan.scen_trade[0], 3) * coverage(self.cfg, ctx.gate, ctx.index)
        limits = self.cfg["markets"][ctx.gate]["price_limits"]
        b = price_taking(ctx.gate, ctx.index, q, limits)
        if self.single == "limit" and not b.empty:
            mean = pd.Series(np.nanmean(plan.scen_price, axis=0), index=ctx.index)
            m = mean.reindex(pd.DatetimeIndex(b["delivery_start"])).to_numpy()
            sell = b["side"].to_numpy() == "sell"
            b["price"] = np.round(np.clip(np.where(sell, m - self.single_margin, m + self.single_margin),
                                          limits["min"], limits["max"]), 2)
        return b
