"""Deterministic LP re-run at each gate with point forecasts."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from ..forecast import known, load_predictions, naive
from ..market_simulator import GATES, Context
from ..optim import STAGES
from .common import Strategy


class Deterministic(Strategy):
    name = "det"

    def forecasts(self, ctx: Context) -> dict:
        return naive(self.md, ctx.day, ctx.as_of, int(self.cfg["simulator"]["activation_days"]))

    def decide(self, ctx: Context):
        return self.bids(ctx, self.solve(ctx, self.forecasts(ctx)))


def error_margins(pred: pd.DataFrame, window_days: int) -> dict:
    """Rolling daily mean absolute error of each intraday price gap to the day-ahead price."""
    p = pred[pred["target"].isin(STAGES[1:])]
    da = pred[(pred["target"] == "da") & (pred["gate"] == "da")].set_index(["day", "slot"])
    da_err = (da["pred"] - da["actual"]).rename("da_err")
    p = p.join(da_err, on=["day", "slot"])
    p["da_err"] = np.where(p["gate"] == "da", p["da_err"], 0.0)
    p = p.dropna(subset=["pred", "actual", "da_err"])
    err = (p["pred"] - p["actual"] - p["da_err"]).abs().groupby([p["target"], p["gate"], p["day"]]).mean()
    out = {}
    for (t, g), s in err.groupby(level=[0, 1]):
        s = s.droplevel([0, 1])
        s.index = pd.to_datetime(s.index)
        out[(t, g)] = s.sort_index().rolling(f"{window_days}D").mean()
    return out


def model_edge(pred: pd.DataFrame, window_days: int) -> dict:
    """Rolling daily mean of naive error minus model error for each model."""
    p = pred.dropna(subset=["pred", "naive", "actual"])
    edge = ((p["naive"] - p["actual"]).abs() - (p["pred"] - p["actual"]).abs()).groupby([p["target"], p["gate"], p["day"]]).mean()
    out = {}
    for (t, g), s in edge.groupby(level=[0, 1]):
        s = s.droplevel([0, 1])
        s.index = pd.to_datetime(s.index)
        out[(t, g)] = s.sort_index().rolling(f"{window_days}D").mean()
    return out


def shrink(price: np.ndarray, ref: np.ndarray, margin: float) -> np.ndarray:
    """Move a forecast towards the reference price by the error margin."""
    gap = price - ref
    out = ref + np.sign(gap) * np.maximum(np.abs(gap) - margin, 0.0)
    return np.where(np.isnan(ref), price, out)


class LightGBM(Deterministic):
    """Deterministic LP with walk-forward LightGBM forecasts, naive where no model exists."""
    name = "ml"

    def __init__(self, md, cfg):
        super().__init__(md, cfg)
        p = load_predictions(cfg)
        self.pred = {k: g.set_index(["day", "slot"])["pred"] for k, g in p.groupby(["target", "gate"])} if not p.empty else {}
        sm = cfg["forecast"].get("spread_margin") or {}
        self.scale = float(sm.get("scale", 0.0))
        self.lag = int(sm.get("lag_days", 2))
        self.margins = error_margins(p, int(sm.get("window_days", 28))) if self.scale > 0 and not p.empty else {}
        ms = cfg["forecast"].get("model_select") or {}
        self.select = bool(ms.get("enabled", False)) and not p.empty
        self.edges = model_edge(p, int(ms.get("window_days", 56))) if self.select else {}

    def margin(self, target: str, gate: str, day) -> float:
        s = self.margins.get((target, gate))
        if s is None:
            return 0.0
        m = s.asof(pd.Timestamp(day - timedelta(days=self.lag)))
        return 0.0 if pd.isna(m) else self.scale * float(m)

    def trusted(self, target: str, gate: str, day) -> bool:
        """Whether the model beat the naive forecast on recent published days."""
        if not self.select:
            return True
        s = self.edges.get((target, gate))
        if s is None:
            return False
        e = s.asof(pd.Timestamp(day - timedelta(days=self.lag)))
        return bool(pd.notna(e) and e > 0)

    def forecasts(self, ctx: Context) -> dict:
        v = super().forecasts(ctx)
        k = GATES.index(ctx.gate)
        slots = self.md.slot.loc[ctx.index].to_numpy()
        day = str(ctx.day)
        used, model_f = {}, {}
        self.source = {}
        for t, gates in self.cfg["forecast"]["gates"].items():
            usable = [g for g in gates if GATES.index(g) <= k]
            if usable:
                self.source[t] = (usable[-1], False)
            s = self.pred.get((t, usable[-1])) if usable else None
            if s is None or day not in s.index.get_level_values(0) or not self.trusted(t, usable[-1], ctx.day):
                continue
            f = s.loc[day].reindex(slots).to_numpy(float)
            kn = known(self.md, t, ctx.day, ctx.as_of)
            v[t] = np.where(~np.isnan(kn), kn, np.where(np.isnan(f), v[t], f))
            used[t] = usable[-1]
            model_f[t] = f
            self.source[t] = (usable[-1], True)
        latest = self.cfg["forecast"].get("ida_fallback", "da") == "latest"
        fallback = np.asarray(v["da"], float).copy()
        for t in STAGES[1:]:
            kn = known(self.md, t, ctx.day, ctx.as_of)
            model = ~np.isnan(model_f[t]) if t in model_f else np.zeros(len(kn), bool)
            v[t] = np.where(~np.isnan(kn), kn, np.where(model, v[t], fallback))
            if latest:
                fallback = np.where(np.isnan(kn), fallback, kn)
        if self.margins:
            ref = np.asarray(v["da"], float).copy()
            for t in STAGES[1:]:
                kn = known(self.md, t, ctx.day, ctx.as_of)
                if t in used:
                    m = self.margin(t, used[t], ctx.day)
                    if m > 0:
                        v[t] = np.where(np.isnan(kn), shrink(np.asarray(v[t], float), ref, m), kn)
                ref = np.where(np.isnan(kn), ref, kn)
        return v
