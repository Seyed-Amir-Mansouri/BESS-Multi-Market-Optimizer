"""Out-of-sample forecast errors of whole historical days."""
from __future__ import annotations

from datetime import date, timedelta

import numpy as np
import pandas as pd

NS = 96


class ErrorModel:
    """Day by slot error matrices of each model and naive forecast from the walk-forward predictions."""

    def __init__(self, pred: pd.DataFrame, cfg, quant: pd.DataFrame | None = None):
        sc = cfg["scenarios"]
        self.window = int(sc["window_days"])
        self.lag = int(sc["lag_days"])
        self.season = float(sc["season_days"])
        self.recency = float(sc["recency_days"])
        self.min_days = int(sc["min_days"])
        self.err = {}
        self.half = {}
        self.floor = float(sc.get("min_half_width", 1.0))
        self.zmax = float(sc.get("z_max", 8.0))
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
        if quant is not None and not quant.empty:
            self._widths(pred, quant)

    def _widths(self, pred: pd.DataFrame, quant: pd.DataFrame) -> None:
        """Upper and lower half widths of the 10 to 90 percent range around the point forecast."""
        q = quant.merge(pred[["target", "gate", "day", "slot", "pred"]], on=["target", "gate", "day", "slot"], suffixes=("_q", ""))
        q = q.dropna(subset=["pred", "q10", "q90"])
        self.qdays = {d: i for i, d in enumerate(sorted(q["day"].unique()))}
        n = len(self.qdays)
        for (t, g), grp in q.groupby(["target", "gate"]):
            r = grp["day"].map(self.qdays).to_numpy()
            c = grp["slot"].to_numpy().astype(int)
            up, dn = np.full((n, NS), np.nan), np.full((n, NS), np.nan)
            up[r, c] = np.maximum((grp["q90"] - grp["pred"]).to_numpy(), self.floor)
            dn[r, c] = np.maximum((grp["pred"] - grp["q10"]).to_numpy(), self.floor)
            self.half[(t, g)] = (up, dn)

    def width(self, target: str, gate: str, day: date) -> tuple | None:
        """Today's upper and lower half widths, or None without a range forecast."""
        h = self.half.get((target, gate))
        i = self.qdays.get(str(day)) if h is not None else None
        if i is None:
            return None
        return h[0][i], h[1][i]

    def scaled(self, target: str, gate: str, idx: np.ndarray, day: date) -> np.ndarray | None:
        """Past model errors in units of their own range, rescaled to today's range."""
        today = self.width(target, gate, day)
        m = self.err.get((target, gate, "model"))
        if today is None or m is None:
            return None
        up, dn = self.half[(target, gate)]
        r = np.array([self.qdays.get(str(d), -1) for d in self.days[idx]])
        ok = r >= 0
        hu = np.where(ok[:, None], up[np.maximum(r, 0)], np.nan)
        hd = np.where(ok[:, None], dn[np.maximum(r, 0)], np.nan)
        e = m[idx]
        z = np.clip(np.where(e >= 0, e / hu, e / hd), -self.zmax, self.zmax)
        out = np.where(z >= 0, z * today[0][None, :], z * today[1][None, :])
        return np.where(np.isnan(out), np.nan_to_num(e), out)

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
