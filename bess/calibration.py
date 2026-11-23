"""Conformal recalibration of quantile forecasts with a quantile set per market."""
from __future__ import annotations

from dataclasses import dataclass
from datetime import date, timedelta

import numpy as np
import pandas as pd

NS = 96
DEFAULT_LEVELS = (0.1, 0.5, 0.9)


def levels(cfg, target: str) -> np.ndarray:
    """Quantile levels used for a market."""
    sets = (cfg["forecast"].get("conformal") or {}).get("levels") or {}
    return np.asarray(sorted(sets.get(target) or DEFAULT_LEVELS), float)


@dataclass
class Calibrated:
    """Calibrated quantiles and the rank of each actual value for one market and gate."""

    days: list
    lv: np.ndarray
    point: np.ndarray
    scale: np.ndarray
    zq: np.ndarray
    pit: np.ndarray
    actual: np.ndarray

    def __post_init__(self):
        self.pos = {d: i for i, d in enumerate(self.days)}

    def ready(self, i: int) -> bool:
        return bool(np.isfinite(self.zq[i]).all(axis=1).any())

    def quantiles(self, i: int) -> np.ndarray:
        """Slot by level calibrated quantiles of one day."""
        return self.point[i][:, None] + self.scale[i][:, None] * self.zq[i]

    def errors(self, u: np.ndarray, i: int) -> np.ndarray:
        """Rows of errors around the point forecast of day i at the given slot ranks, interpolated between the levels."""
        z = np.column_stack([np.interp(u[:, j], self.lv, self.zq[i, j]) for j in range(u.shape[1])])
        return z * self.scale[i][None, :]


def _matrix(df: pd.DataFrame, col: str, pos: dict) -> np.ndarray:
    m = np.full((len(pos), NS), np.nan)
    m[df["day"].map(pos).to_numpy(int), df["slot"].to_numpy(int)] = df[col].to_numpy(float)
    return m


def calibrate(cfg, pred: pd.DataFrame, quant: pd.DataFrame) -> dict:
    """Calibrated quantiles per market and gate from scaled errors of the days in a rolling window."""
    c = cfg["forecast"].get("conformal") or {}
    window, lag = int(c.get("window_days", 56)), int(c.get("lag_days", 2))
    min_days, floor = int(c.get("min_days", 28)), float(c.get("min_half_width", 1.0))
    blocks, gamma = int(c.get("hour_blocks", 1)), float(c.get("adapt_rate", 0.0))
    native = bool(c.get("native", False))
    block = (np.arange(NS) // (NS // blocks)).clip(max=blocks - 1)
    if pred.empty or quant.empty:
        return {}
    keys = ["target", "gate", "day", "slot"]
    cols = [c for c in quant.columns if c.startswith("q") and c[1:].isdigit()]
    q = quant[keys + cols].merge(pred[keys + ["pred", "actual"]], on=keys).dropna(subset=["pred", "q10", "q90"])
    out = {}
    for (t, g), grp in q.groupby(["target", "gate"]):
        days = sorted(grp["day"].astype(str).unique())
        pos = {d: i for i, d in enumerate(days)}
        grp = grp.assign(day=grp["day"].astype(str))
        P, A = _matrix(grp, "pred", pos), _matrix(grp, "actual", pos)
        S = np.maximum((_matrix(grp, "q90", pos) - _matrix(grp, "q10", pos)) / 2.0, floor)
        Z = (A - P) / S
        lv = levels(cfg, t)
        off = np.zeros((len(days), NS, len(lv)))
        if native:
            for j, l in enumerate(lv):
                name = f"q{round(l * 100)}"
                if name in grp.columns and grp[name].notna().any():
                    off[..., j] = np.nan_to_num((_matrix(grp, name, pos) - P) / S)
        dd = np.array([date.fromisoformat(d) for d in days])
        zq = np.full((len(days), NS, len(lv)), np.nan)
        pit = np.full((len(days), NS), np.nan)
        tau, seen = np.tile(lv, (blocks, 1)), 0
        for i, d in enumerate(dd):
            while gamma > 0 and seen < i and dd[seen] <= d - timedelta(days=lag):
                for b in range(blocks):
                    zk, qk = Z[seen, block == b], zq[seen, block == b]
                    ok = np.isfinite(zk) & np.isfinite(qk).all(axis=1)
                    if ok.any():
                        tau[b] += gamma * (lv - (zk[ok][:, None] < qk[ok]).mean(axis=0))
                seen += 1
            sel = (dd >= d - timedelta(days=window)) & (dd <= d - timedelta(days=lag))
            if sel.sum() < min_days:
                continue
            for b in range(blocks):
                cols = block == b
                z = Z[sel][:, cols].ravel()
                z = np.sort(z[np.isfinite(z)])
                if len(z) < 10:
                    continue
                tb = np.clip(tau[b] if native else np.sort(tau[b]), 0.0005, 0.9995)
                if native:
                    zw = Z[sel][:, cols][..., None] - off[sel][:, cols]
                    zl = np.array([np.nanquantile(zw[..., j], tb[j]) for j in range(len(lv))])
                    zq[i, cols] = np.sort(off[i, cols] + zl, axis=-1)
                else:
                    zq[i, cols] = np.quantile(z, tb)
                zi = Z[i, cols]
                ok = np.isfinite(zi)
                r = np.full(zi.shape, np.nan)
                r[ok] = (np.searchsorted(z, zi[ok], side="left") + np.searchsorted(z, zi[ok], side="right")) / (2.0 * len(z))
                pit[i, cols] = r
        out[(t, g)] = Calibrated(days, lv, P, S, zq, np.clip(pit, 0.5 / NS, 1 - 0.5 / NS), A)
    return out


def coverage(cal: dict, days=None) -> pd.DataFrame:
    """Share of actual values below each calibrated quantile, per market and gate, over the chosen days."""
    rows = []
    for (t, g), c in cal.items():
        idx = [c.pos[d] for d in (days if days is not None else c.days) if d in c.pos]
        idx = [i for i in idx if c.ready(i)]
        if not idx:
            continue
        a = c.actual[idx]
        qs = np.stack([c.quantiles(i) for i in idx])
        ok = np.isfinite(a) & np.isfinite(qs).all(axis=-1)
        rec = {"target": t, "gate": g, "n": int(ok.sum())}
        for j, l in enumerate(c.lv):
            rec[f"below_q{round(l * 100)}"] = 100.0 * float((a[ok] < qs[..., j][ok]).mean())
        rows.append(rec)
    return pd.DataFrame(rows)
