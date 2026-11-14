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


def predictions_path(cfg):
    p = cfg.path("forecast_dir")
    p.mkdir(parents=True, exist_ok=True)
    return p / str(cfg["forecast"].get("predictions") or "predictions.parquet")


def load_predictions(cfg) -> pd.DataFrame:
    p = predictions_path(cfg)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def quantiles_path(cfg):
    p = cfg.path("forecast_dir")
    p.mkdir(parents=True, exist_ok=True)
    return p / str(cfg["forecast"].get("quantile_predictions") or "quantiles.parquet")


def load_quantiles(cfg) -> pd.DataFrame:
    p = quantiles_path(cfg)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()


def _folds(start, end, months: int) -> list:
    out = []
    a = pd.Timestamp(start).to_period("M").to_timestamp().date()
    while a <= end:
        b = (pd.Timestamp(a) + pd.DateOffset(months=months)).date()
        out.append((max(a, start), min(b, end + timedelta(days=1))))
        a = b
    return out


def _da_predictions(cfg, panel) -> np.ndarray:
    """Saved out-of-sample day-ahead predictions at the day-ahead gate aligned to the panel rows."""
    from .features import NS
    out = np.full(len(panel.days) * NS, np.nan)
    p = load_predictions(cfg)
    if p.empty:
        return out
    p = p[(p["target"] == "da") & (p["gate"] == "da") & p["pred"].notna()]
    pos = {str(d): i for i, d in enumerate(panel.days)}
    r = p["day"].astype(str).map(pos)
    ok = r.notna().to_numpy()
    out[(r[ok].astype(int) * NS + p["slot"][ok]).to_numpy(int)] = p["pred"][ok].to_numpy()
    return out


def walk_forward(cfg, store, start=None, end=None, targets=None, log=None, save=True, folds=None,
                 quantiles=None) -> pd.DataFrame:
    """Train LightGBM per target and gate on past days only and predict each following block."""
    from datetime import date

    from lightgbm import LGBMRegressor

    from .features import NS, PRICES, Panel
    from .market_simulator import GATES

    fc = cfg["forecast"]
    start = start or date.fromisoformat(cfg["backtest"]["start"])
    end = end or cfg.data_end()
    md = MarketData(store, cfg, cfg.data_start(), end)
    panel = Panel(md, store, cfg)
    end = min(end, panel.days[-1])
    day_col = np.repeat(panel.days, NS)
    slot_col = np.tile(np.arange(NS), len(panel.days))
    folds = folds or _folds(start, end, int(fc["retrain_months"]))
    embargo = timedelta(days=int(fc["embargo_days"]))
    params = dict(fc["lgbm"])
    targets = targets or list(fc["gates"])
    if quantiles:
        targets = [t for t in targets if t in PRICES]
    surprise = bool(fc.get("da_surprise"))
    da_pred = _da_predictions(cfg, panel) if surprise else None
    rows = []
    for gate in GATES:
        pairs = [t for t in targets if gate in fc["gates"][t]]
        if not pairs:
            continue
        full = panel.features(gate)
        if surprise and gate != "da":
            full["da_surprise"] = full["da_d0"].to_numpy() - da_pred
        X = full.loc[:, full.notna().any()]
        for target in pairs:
            y = panel.val[target].ravel()
            nv = panel.naive(target, full)
            base = full["da_d0"].to_numpy() if target in PRICES[1:] and gate != "da" else None
            yt = y - base if base is not None else y
            lo, hi = (fc.get("bounds") or {}).get(target, [None, None])
            for f0, f1 in folds:
                te = (day_col >= f0) & (day_col < f1)
                if not te.any():
                    continue
                tr = (day_col < f0 - embargo) & ~np.isnan(yt)
                pred = np.full(te.sum(), np.nan)
                qs = {}
                if quantiles and len(np.unique(day_col[tr])) >= int(fc["min_train_days"]):
                    off = base[te] if base is not None else 0.0
                    for q in quantiles:
                        m = LGBMRegressor(**{**params, "objective": "quantile", "alpha": q}).fit(X.loc[tr], yt[tr])
                        qs[f"q{round(q * 100)}"] = m.predict(X.loc[te]) + off
                elif len(np.unique(day_col[tr])) >= int(fc["min_train_days"]):
                    m = LGBMRegressor(**params).fit(X.loc[tr], yt[tr])
                    pred = m.predict(X.loc[te])
                    if base is not None:
                        pred = pred + base[te]
                    pred = np.clip(pred, lo if lo is not None else -np.inf, hi if hi is not None else np.inf)
                    if surprise and target == "da" and gate == "da":
                        da_pred[te] = pred
                rows.append(pd.DataFrame({"target": target, "gate": gate, "day": day_col[te].astype(str),
                                          "slot": slot_col[te], "actual": y[te], "pred": pred, "naive": nv[te], **qs}))
            if log:
                log(f"{target} @ {gate} done")
    out = pd.concat(rows, ignore_index=True)
    if save:
        out.to_parquet(quantiles_path(cfg) if quantiles else predictions_path(cfg), index=False)
    return out


def train_once(cfg, store, start, end, log=None) -> tuple:
    """Models trained once on all days before `start` predict start to end; earlier days keep the walk-forward predictions."""
    from .config import Config, _deep_merge
    new = walk_forward(cfg, store, start, end, log=log, save=False, folds=[(start, end + timedelta(days=1))])
    old = load_predictions(cfg)
    if not old.empty:
        old = old[pd.to_datetime(old["day"]).dt.date < start]
    out = pd.concat([old, new], ignore_index=True)
    live = Config(raw=_deep_merge(cfg.raw, {"forecast": {"predictions": "predictions_live.parquet"}}), base_dir=cfg.base_dir)
    out.to_parquet(predictions_path(live), index=False)
    return out, live


def accuracy(pred: pd.DataFrame) -> pd.DataFrame:
    """Mean absolute error of the model and the naive forecast on common rows."""
    p = pred.dropna(subset=["actual", "pred", "naive"])
    g = p.assign(e_m=(p["pred"] - p["actual"]).abs(), e_n=(p["naive"] - p["actual"]).abs()).groupby(["target", "gate"], sort=False)
    out = g.agg(days=("day", "nunique"), mae_naive=("e_n", "mean"), mae_model=("e_m", "mean"),
                rmse_naive=("e_n", lambda e: float(np.sqrt((e ** 2).mean()))),
                rmse_model=("e_m", lambda e: float(np.sqrt((e ** 2).mean())))).reset_index()
    out["skill"] = 1 - out["mae_model"] / out["mae_naive"]
    return out
