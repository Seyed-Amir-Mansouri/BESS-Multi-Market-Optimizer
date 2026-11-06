"""Day by quarter-hour panels and gate features for the forecasting models."""
from __future__ import annotations

import warnings
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .config import Config
from .data.calendar_es import national_holidays
from .data.pubtime import market_time
from .data.store import Store
from .market_simulator import COLS, MarketData
from .timeutils import local_midnight_utc

NS = 96
NEVER = np.iinfo(np.int64).max
PRICES = ["da", "ida1", "ida2", "ida3"]


def _ns(p) -> np.ndarray:
    p = pd.to_datetime(pd.Series(p), utc=True)
    out = p.dt.tz_convert(None).to_numpy(dtype="datetime64[ns]").astype(np.int64)
    return np.where(p.isna().to_numpy(), NEVER, out)


def _nanmean(a: np.ndarray, axis: int) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        return np.nanmean(a, axis=axis)


class Panel:
    """Values and publication times as day by slot matrices."""

    def __init__(self, md: MarketData, store: Store, cfg: Config):
        self.cfg = cfg
        self.days = np.array(sorted(set(md.day_of)))
        self.pos = {d: i for i, d in enumerate(self.days)}
        di = md.day_of.map(self.pos).to_numpy()
        sl = md.slot.to_numpy()
        self.val, self.pub, self.day_pub = {}, {}, {}
        for c in COLS:
            self._put(c, di, sl, md.values[c].to_numpy(float), md.pubs[c])
        ex = cfg["forecast"].get("exogenous") or {}
        names = sorted({s for v in ex.values() for s in v})
        a = local_midnight_utc(self.days[0], cfg.tz)
        b = local_midnight_utc(self.days[-1] + timedelta(days=1), cfg.tz)
        df = store.load(series=names, start=a, end=b, qh=True) if names else pd.DataFrame()
        self.exog = []
        if df.empty:
            return
        idx = md.values.index
        val = df.pivot_table(index="delivery_start", columns="series", values="value", aggfunc="last").reindex(idx)
        pub = df.pivot_table(index="delivery_start", columns="series", values="published_at", aggfunc="max").reindex(idx)
        for k, ss in ex.items():
            ss = [s for s in ss if s in val]
            if not ss:
                continue
            p = pub[ss].apply(lambda c: pd.to_datetime(c, utc=True)).max(axis=1)
            self._put(k, di, sl, val[ss].mean(axis=1).to_numpy(float), p)
            self.exog.append(k)

    def _put(self, key, di, sl, v, p) -> None:
        pi = _ns(p)
        ok = ~np.isnan(v)
        g = pd.DataFrame({"d": di[ok], "s": sl[ok], "v": v[ok], "p": pi[ok]}).groupby(["d", "s"]).agg(v=("v", "mean"), p=("p", "max"))
        V = np.full((len(self.days), NS), np.nan)
        P = np.full((len(self.days), NS), NEVER, dtype=np.int64)
        r, c = g.index.get_level_values(0).to_numpy(), g.index.get_level_values(1).to_numpy()
        V[r, c] = g["v"].to_numpy()
        P[r, c] = g["p"].to_numpy()
        self.val[key], self.pub[key] = V, P
        has = ~np.isnan(V)
        self.day_pub[key] = np.where(has.any(axis=1), np.where(has, P, 0).max(axis=1), NEVER)

    def as_of(self, gate: str) -> np.ndarray:
        return np.array([market_time(self.cfg, gate, "gate", d).value for d in self.days], dtype=np.int64)

    def calendar(self) -> dict:
        d = pd.to_datetime(self.days)
        years = sorted({x.year for x in self.days})
        hol = {h for y in years for h in national_holidays(y)}
        reg = sorted(date.fromisoformat(v["date"]) for k, v in self.cfg["regimes"].items() if k != "ida_start")
        r = np.array([sum(x >= q for q in reg) for x in self.days])
        doy = d.day_of_year.to_numpy()
        cols = {"dow": d.day_of_week.to_numpy(), "month": d.month.to_numpy(),
                "doy_sin": np.sin(2 * np.pi * doy / 365.25), "doy_cos": np.cos(2 * np.pi * doy / 365.25),
                "holiday": np.array([x in hol for x in self.days], dtype=float), "regime": r}
        out = {k: np.repeat(v, NS) for k, v in cols.items()}
        out["slot"] = np.tile(np.arange(NS), len(self.days))
        return out

    def features(self, gate: str) -> pd.DataFrame:
        """One row per day and slot with the values published by the gate."""
        n = len(self.days)
        t = self.as_of(gate)
        lb = int(self.cfg["forecast"]["lookback_days"])
        F = self.calendar()
        for c in self.exog:
            F[c] = np.where(self.pub[c] <= t[:, None], self.val[c], np.nan).ravel()
        for c in COLS:
            V, P, dp = self.val[c], self.pub[c], self.day_pub[c]
            known = np.where(P <= t[:, None], V, np.nan)
            prev = np.vstack([np.full((1, NS), np.nan), np.where(P[:-1] <= t[1:, None], V[:-1], np.nan)])
            both = np.hstack([prev, known])
            filled = pd.DataFrame(both).ffill(axis=1).to_numpy()[:, -1]
            lag1 = np.full((n, NS), np.nan)
            lag7 = np.full((n, NS), np.nan)
            wk = np.full((n, NS), np.nan)
            for i in range(n):
                picked = [j for j in range(i - 1, max(i - lb, 0) - 1, -1) if dp[j] <= t[i]][:7]
                if picked:
                    lag1[i] = V[picked[0]]
                    lag7[i] = _nanmean(V[picked], 0)
                if i >= 7 and dp[i - 7] <= t[i]:
                    wk[i] = V[i - 7]
            F[f"{c}_d0"] = known.ravel()
            F[f"{c}_d0_mean"] = np.repeat(_nanmean(known, 1), NS)
            F[f"{c}_prev_mean"] = np.repeat(_nanmean(prev, 1), NS)
            F[f"{c}_last"] = np.repeat(filled, NS)
            F[f"{c}_lag1"] = lag1.ravel()
            F[f"{c}_lag1_mean"] = np.repeat(_nanmean(lag1, 1), NS)
            F[f"{c}_lag7"] = lag7.ravel()
            F[f"{c}_wk"] = wk.ravel()
        for c in PRICES[1:]:
            F[f"{c}_spread_lag7"] = F[f"{c}_lag7"] - F["da_lag7"]
            F[f"{c}_spread_lag1"] = F[f"{c}_lag1"] - F["da_lag1"]
            F[f"{c}_spread_d0"] = F[f"{c}_d0"] - F["da_d0"]
        fc = self.cfg["forecast"]
        k = int(fc.get("nowcast_slots", 12))
        for name, (a, f) in (fc.get("nowcast") or {}).items():
            if a in self.val and f in self.val:
                F[name] = np.repeat(self.recent_error(a, f, t, k), NS)
        if all(c in F for c in ("load_fc", "wind_fc", "solar_fc")):
            F["resid_fc"] = F["load_fc"] - F["wind_fc"] - F["solar_fc"]
        return pd.DataFrame(F)

    def recent_error(self, actual: str, forecast: str, t: np.ndarray, k: int) -> np.ndarray:
        """Mean actual minus forecast over the last k quarter-hours published by each day's gate."""
        A, P = self.val[actual].ravel(), self.pub[actual].ravel()
        e = A - self.val[forecast].ravel()
        out = np.full(len(self.days), np.nan)
        for i in range(len(self.days)):
            lo, hi = max(i - 1, 0) * NS, (i + 1) * NS
            ok = (P[lo:hi] <= t[i]) & ~np.isnan(e[lo:hi])
            if ok.any():
                out[i] = e[lo:hi][ok][-k:].mean()
        return out

    def naive(self, target: str, X: pd.DataFrame) -> np.ndarray:
        """Phase 2 naive forecast of a target from the gate features."""
        if target in PRICES:
            return X["da_d0"].fillna(X["da_lag1"]).to_numpy()
        if target in ("cap_up", "cap_dn", "ratio_up"):
            return X[f"{target}_d0"].fillna(X[f"{target}_lag1"]).to_numpy()
        return X[f"{target}_lag7"].to_numpy()
