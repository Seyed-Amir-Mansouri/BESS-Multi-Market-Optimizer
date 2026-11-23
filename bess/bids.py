"""Price-quantity step bids and pay-as-clear clearing."""
from __future__ import annotations

import numpy as np
import pandas as pd

COLUMNS = ["market", "delivery_start", "side", "price", "quantity_mw"]


def price_taking(market: str, index: pd.DatetimeIndex, trade: np.ndarray, limits: dict, tol: float = 1e-6) -> pd.DataFrame:
    """One step per period at the price limit: sell at the minimum, buy at the maximum."""
    trade = np.asarray(trade, dtype=float)
    keep = np.abs(trade) > tol
    sell = trade[keep] > 0
    return pd.DataFrame({
        "market": market, "delivery_start": index[keep],
        "side": np.where(sell, "sell", "buy"),
        "price": np.where(sell, limits["min"], limits["max"]),
        "quantity_mw": np.abs(trade[keep]),
    }, columns=COLUMNS)


def _lines(q: np.ndarray, tol: float) -> int:
    """Bid lines the curve writes for one rising quantity path."""
    up, dn = np.maximum(q, 0), np.maximum(-q, 0)
    return int(up[0] > tol) + int(dn[-1] > tol) + int((np.diff(up) > tol).sum()) + int((-np.diff(dn) > tol).sum())


def limit_steps(Q: np.ndarray, steps: int, tol: float = 1e-6) -> np.ndarray:
    """Rising quantity paths with at most `steps` bid lines per column; smaller steps join the next kept step."""
    out = Q.copy()
    for j in range(Q.shape[1]):
        q = Q[:, j]
        if _lines(q, tol) <= steps:
            continue
        d = np.diff(q)
        nz = np.flatnonzero(d > tol)
        order = nz[np.argsort(-d[nz], kind="stable")]
        for n in range(min(steps, len(nz)), 0, -1):
            keep = np.sort(order[:n])
            nd = np.zeros_like(d)
            for k in nz:
                nxt = keep[keep >= k]
                nd[nxt[0] if len(nxt) else keep[-1]] += d[k]
            r = q[0] + np.concatenate([[0.0], np.cumsum(nd)])
            if _lines(r, tol) <= steps:
                out[:, j] = r
                break
    return out


def curve(market: str, index: pd.DatetimeIndex, prices: np.ndarray, qty: np.ndarray, limits: dict,
          tick: float = 0.01, tol: float = 1e-6, max_steps: int | None = None) -> pd.DataFrame:
    """Step bids from scenario prices and net sell quantities that rise with the price, at most max_steps lines per period."""
    P = np.clip(np.nan_to_num(np.asarray(prices, float)), limits["min"], limits["max"])
    o = np.argsort(P, axis=0, kind="stable")
    P = np.take_along_axis(P, o, 0)
    Q = np.maximum.accumulate(np.take_along_axis(np.asarray(qty, float), o, 0), axis=0)
    if max_steps:
        Q = limit_steps(Q, int(max_steps), tol)
    up, dn = np.maximum(Q, 0), np.maximum(-Q, 0)
    S, n = P.shape
    cols = np.arange(n)
    parts = [("sell", np.full(n, limits["min"]), up[0]), ("buy", np.full(n, limits["max"]), dn[-1])]
    for k in range(1, S):
        parts.append(("sell", P[k], up[k] - up[k - 1]))
        parts.append(("buy", np.maximum(P[k] - tick, limits["min"]), dn[k - 1] - dn[k]))
    rows = []
    for side, price, q in parts:
        keep = q > tol
        rows.append(pd.DataFrame({"market": market, "delivery_start": index[cols[keep]], "side": side,
                                  "price": price[keep], "quantity_mw": q[keep]}, columns=COLUMNS))
    return pd.concat(rows, ignore_index=True)


def band_offers(market: str, index: pd.DatetimeIndex, up: np.ndarray, down: np.ndarray, price: float = 0.0,
                tol: float = 1e-6) -> pd.DataFrame:
    """Capacity offers for upward and downward aFRR band."""
    parts = []
    for side, q in (("up", np.asarray(up, float)), ("down", np.asarray(down, float))):
        keep = q > tol
        parts.append(pd.DataFrame({"market": market, "delivery_start": index[keep], "side": side,
                                   "price": price, "quantity_mw": q[keep]}, columns=COLUMNS))
    return pd.concat(parts, ignore_index=True)


def clear(bids: pd.DataFrame, prices: pd.Series, side: str) -> np.ndarray:
    """Cleared quantity per period for one side; a missing clearing price clears nothing."""
    q = pd.Series(0.0, index=prices.index)
    b = bids[bids["side"] == side]
    if not b.empty:
        p = prices.reindex(pd.DatetimeIndex(b["delivery_start"])).to_numpy()
        ok = (b["price"].to_numpy() >= p) if side == "buy" else (b["price"].to_numpy() <= p)
        ok &= ~np.isnan(p)
        q = q.add(b.loc[ok].groupby("delivery_start")["quantity_mw"].sum(), fill_value=0.0)
    return q.reindex(prices.index).fillna(0.0).to_numpy()
