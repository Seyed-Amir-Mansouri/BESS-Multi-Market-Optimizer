"""Naive forecasts from data published before the gate."""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from .market_simulator import MarketData


def known(md: MarketData, col: str, day, as_of: pd.Timestamp) -> np.ndarray:
    """Actual values of the day that were published by `as_of`."""
    idx = md.index(day)
    v = md.values.loc[idx, col].to_numpy(dtype=float)
    p = md.pubs.loc[idx, col]
    return np.where((p <= as_of).to_numpy(), v, np.nan)


def profile(md: MarketData, col: str, day, as_of: pd.Timestamp, n_days: int = 1, lookback: int = 21) -> np.ndarray:
    """Mean time-of-day profile of the last `n_days` days fully published by `as_of`."""
    idx = md.index(day)
    picked = []
    for i in range(1, lookback + 1):
        d = day - timedelta(days=i)
        if d not in md.day_count.index or md.day_count.at[d, col] == 0:
            continue
        if md.day_pub.at[d, col] > as_of:
            continue
        picked.append(d)
        if len(picked) == n_days:
            break
    if not picked:
        return np.full(len(idx), np.nan)
    mask = md.day_of.isin(picked).to_numpy()
    prof = md.values.loc[mask, col].groupby(md.slot[mask].to_numpy()).mean()
    out = prof.reindex(range(96)).ffill().bfill()
    return out.reindex(md.slot.loc[idx].to_numpy()).to_numpy(dtype=float)


def latest(md: MarketData, col: str, day, as_of: pd.Timestamp, n_days: int = 1) -> np.ndarray:
    """Published actual values of the day where available, otherwise the naive profile."""
    k = known(md, col, day, as_of)
    if not np.isnan(k).any():
        return k
    p = profile(md, col, day, as_of, n_days)
    return np.where(np.isnan(k), p, k)


def naive(md: MarketData, day, as_of: pd.Timestamp, activation_days: int = 7) -> dict:
    """Point forecasts for every input of the daily model."""
    da = latest(md, "da", day, as_of)
    out = {"da": da, "ida1": da.copy(), "ida2": da.copy(), "ida3": da.copy()}
    for c in ("cap_up", "cap_dn", "ratio_up"):
        out[c] = latest(md, c, day, as_of)
    for c in ("share_up", "share_dn", "e_up", "e_dn"):
        out[c] = profile(md, c, day, as_of, activation_days)
    return out
