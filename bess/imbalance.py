"""Walk-forward forecast of the gain from a deliberate long or short imbalance position."""
from __future__ import annotations

import logging

import numpy as np
import pandas as pd

from .config import Config

log = logging.getLogger(__name__)

KEYS = ("da", "ida1", "ida2", "ida3", "act_up", "act_dn", "imb_short", "imb_long")


def features(cfg: Config, w: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Inputs known at the start of each quarter-hour and the realised gains of being long and short."""
    c = cfg["simulator"].get("imbalance") or {}
    lag, plag = int(c.get("activation_lag_qh", 1)), int(c.get("price_lag_qh", 5))
    ref = w.ida3.fillna(w.ida2).fillna(w.ida1).fillna(w.da)
    y = pd.DataFrame({"gl": w.imb_long - ref, "gs": ref - w.imb_short})
    net = w.act_up - w.act_dn
    X = pd.DataFrame(index=w.index)
    for k in range(lag, lag + 4):
        X[f"n{k}"] = net.shift(k)
        X[f"u{k}"] = w.act_up.shift(k)
        X[f"d{k}"] = w.act_dn.shift(k)
    for k in range(plag, plag + 2):
        X[f"l{k}"] = (w.imb_long - w.da).shift(k)
        X[f"s{k}"] = (w.imb_short - w.da).shift(k)
    X["ref"] = ref
    X["rd"] = ref - w.da
    X["i1d"] = w.ida1 - w.da
    loc = w.index.tz_convert(cfg.tz)
    X["h"] = loc.hour + loc.minute / 60
    X["dow"] = np.asarray(loc.dayofweek)
    return X, y


def train_predict(cfg: Config, w: pd.DataFrame, min_days: int = 90) -> pd.DataFrame:
    """Monthly walk-forward predictions of the long and short gains, each month trained only on earlier months."""
    from sklearn.ensemble import HistGradientBoostingRegressor
    X, y = features(cfg, w)
    ok = y.notna().all(axis=1)
    loc = w.index.tz_convert(cfg.tz)
    months = pd.PeriodIndex(loc.tz_localize(None), freq="M").unique()
    first = w.index[ok][0] if ok.any() else None
    out = []
    for m in months:
        a = pd.Timestamp(m.start_time, tz=cfg.tz).tz_convert("UTC")
        b = pd.Timestamp(m.end_time, tz=cfg.tz).tz_convert("UTC")
        tr = (X.index < a) & ok
        te = (X.index >= a) & (X.index <= b)
        if not te.any() or first is None or (a - first).days < min_days:
            continue
        p = pd.DataFrame(index=X.index[te])
        for g in ("gl", "gs"):
            mdl = HistGradientBoostingRegressor(max_iter=300, learning_rate=0.05, min_samples_leaf=200, random_state=0)
            p[g] = mdl.fit(X[tr], y.loc[tr, g]).predict(X[te])
        out.append(p)
    return pd.concat(out) if out else pd.DataFrame(columns=["gl", "gs"])


def load_signal(cfg: Config, store) -> pd.DataFrame:
    """Forecast gains from the forecast folder, rebuilt when missing or older than the data."""
    c = cfg["simulator"].get("imbalance") or {}
    lag, plag = int(c.get("activation_lag_qh", 1)), int(c.get("price_lag_qh", 5))
    path = cfg.path("forecast_dir") / f"imbalance_signal_a{lag}_p{plag}.parquet"
    names = cfg["simulator"]["series"]
    w = store.wide([names[k] for k in KEYS]).rename(columns={names[k]: k for k in KEYS})
    w = w.reindex(pd.date_range(w.index.min(), w.index.max(), freq="15min", name=w.index.name))
    w = w[w.da.notna().cumsum() > 0]
    if path.exists():
        s = pd.read_parquet(path)
        if len(s) and s.index.max() >= w.index[w.imb_long.notna()].max():
            return s
    log.info("imbalance signal: training walk-forward models")
    s = train_predict(cfg, w)
    path.parent.mkdir(parents=True, exist_ok=True)
    s.to_parquet(path)
    return s


def imbalance_sign(cfg: Config, store) -> pd.Series:
    """+1 to go long, -1 to go short, 0 to follow the schedule, per quarter-hour."""
    th = float((cfg["simulator"].get("imbalance") or {}).get("threshold_eur_per_mwh", 30.0))
    s = load_signal(cfg, store)
    gl, gs = s["gl"], s["gs"]
    return pd.Series(np.where((gl > th) & (gl >= gs), 1.0, np.where((gs > th) & (gs > gl), -1.0, 0.0)), index=s.index)
