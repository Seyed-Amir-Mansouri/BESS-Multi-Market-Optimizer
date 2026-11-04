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
