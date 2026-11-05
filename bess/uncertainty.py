"""Out-of-sample forecast errors of whole historical days."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

NS = 96


class ErrorModel:
    """Day by slot error matrices of each model and naive forecast from the walk-forward predictions."""

    def __init__(self, pred: pd.DataFrame, cfg):
        sc = cfg["scenarios"]
        self.window = int(sc["window_days"])
        self.lag = int(sc["lag_days"])
        self.season = float(sc["season_days"])
        self.recency = float(sc["recency_days"])
        self.min_days = int(sc["min_days"])
        self.err = {}
        if pred.empty:
            self.days = np.array([], dtype=object)
            return
        p = pred.dropna(subset=["actual"])
        days = sorted({date.fromisoformat(d) for d in p["day"].unique()})
        self.days = np.array(days, dtype=object)
        pos = {str(d): i for i, d in enumerate(days)}
        for (t, g), grp in p.groupby(["target", "gate"]):
            r = grp["day"].map(pos).to_numpy()
            c = grp["slot"].to_numpy().astype(int)
            for kind, col in (("model", "pred"), ("naive", "naive")):
                m = np.full((len(days), NS), np.nan)
                m[r, c] = (grp["actual"] - grp[col]).to_numpy()
                self.err[(t, g, kind)] = m

    def pool(self, day: date) -> tuple[np.ndarray, np.ndarray]:
        """Indices and weights of past days whose errors are known before the day."""
        if not len(self.days):
            return np.array([], int), np.array([])
        hi, lo = day - timedelta(days=self.lag), day - timedelta(days=self.window)
        idx = np.array([i for i, d in enumerate(self.days) if lo <= d <= hi], int)
        if len(idx) < self.min_days:
            return np.array([], int), np.array([])
        d = self.days[idx]
        age = np.array([(day - x).days for x in d], float)
        gap = np.abs(np.array([x.timetuple().tm_yday for x in d]) - day.timetuple().tm_yday)
        gap = np.minimum(gap, 365 - gap)
        weekend = np.array([x.weekday() >= 5 for x in d]) == (day.weekday() >= 5)
        w = np.exp(-(gap / self.season) ** 2) * np.exp(-age / self.recency) * np.where(weekend, 1.0, 0.5)
        return idx, w / w.sum()

    def matrix(self, target: str, gate: str, idx: np.ndarray, model: bool) -> np.ndarray:
        """Errors of the chosen forecast on the given days, naive errors where the model has none."""
        nv = self.err.get((target, gate, "naive"))
        if nv is None:
            return np.zeros((len(idx), NS))
        out = nv[idx]
        if model:
            m = self.err[(target, gate, "model")][idx]
            out = np.where(np.isnan(m), out, m)
        return np.nan_to_num(out)
