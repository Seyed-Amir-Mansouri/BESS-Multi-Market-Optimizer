"""Data-quality checks on the clean store.

Cleaning policy (applied at parse/write time): rows without a value are dropped, exact
duplicates (series, delivery_start, published_at) keep the latest fetch, and corrected
OMIE file versions supersede earlier ones. Nothing is dropped for being extreme: limit
breaches, outliers and flat runs are FLAGGED here for review, because real price spikes
are exactly what a battery earns from.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from ..config import Config
from ..timeutils import delivery_day, local_time_utc
from .store import Store

PRICE_SERIES = ["da_price_es", "ida1_price_es", "ida2_price_es", "ida3_price_es",
                "afrr_band_price", "afrr_band_price_up", "afrr_band_price_down",
                "afrr_energy_price_up", "afrr_energy_price_down"]


def market_of(series: str) -> str | None:
    base = series.split(".")[0]
    for m in ("ida1", "ida2", "ida3", "da"):
        if base.startswith(f"{m}_price"):
            return m
    if base.startswith("afrr_band_price"):
        return "afrr_capacity"
    return None


def _expected_periods(cfg: Config, series: str, day, res_min: int) -> int:
    m = market_of(series)
    cov = cfg["markets"][m]["coverage"] if m else {"from": "00:00", "to": "24:00"}
    a = local_time_utc(day, cov["from"], cfg.tz)
    b = local_time_utc(day, cov["to"], cfg.tz)
    return int((b - a) / pd.Timedelta(minutes=res_min))


def _ranges(days: list) -> list[str]:
    """Collapse sorted dates into 'a..b' ranges."""
    out, start, prev = [], None, None
    for d in days:
        if start is None:
            start = prev = d
        elif d == prev + timedelta(days=1):
            prev = d
        else:
            out.append(f"{start}" if start == prev else f"{start}..{prev}")
            start = prev = d
    if start is not None:
        out.append(f"{start}" if start == prev else f"{start}..{prev}")
    return out


def _longest_flat_h(g: pd.DataFrame) -> float:
    v = g["value"].to_numpy()
    if len(v) < 2:
        return 0.0
    contiguous = (g["delivery_start"].to_numpy()[1:] == g["delivery_end"].to_numpy()[:-1])
    same = (v[1:] == v[:-1]) & contiguous
    best = run = 0.0
    dur = (g["resolution_min"].to_numpy()[1:]) / 60.0
    for s, d in zip(same, dur):
        run = run + d if s else 0.0
        best = max(best, run)
    return best + (g["resolution_min"].iloc[0] / 60.0 if best else 0.0)


def check_series(cfg: Config, series: str, g: pd.DataFrame, revisions: int = 0) -> dict:
    g = g.sort_values("delivery_start")
    days = delivery_day(g["delivery_start"], cfg.tz)
    days.index = g.index
    g = g.assign(day=days)
    per_day = g.groupby("day").agg(n=("value", "size"), res=("resolution_min", lambda s: int(s.mode().iloc[0])))
    first, last = per_day.index.min(), per_day.index.max()
    all_days = pd.date_range(first, last, freq="D").date
    res_by_day = per_day["res"].reindex(all_days).ffill().astype(int)
    expected = {d: _expected_periods(cfg, series, d, int(r)) for d, r in res_by_day.items()}
    present = per_day["n"].reindex(all_days).fillna(0).astype(int)
    missing_days = [d for d in all_days if present[d] == 0]
    partial_days = [d for d in all_days if 0 < present[d] < expected[d]]
    exp_total = sum(expected.values())
    pres_total = int(min(present.sum(), exp_total)) if exp_total else int(present.sum())
    changes = []
    prev = None
    for d, r in per_day["res"].items():
        if prev is not None and r != prev:
            changes.append(f"{d}: {prev}->{r} min")
        prev = r
    m = market_of(series)
    lim = cfg["markets"][m]["price_limits"] if m else {}
    lo, hi = lim.get("min"), lim.get("max")
    none = pd.Series(False, index=g.index)
    oob = g[((g["value"] < lo) if lo is not None else none) | ((g["value"] > hi) if hi is not None else none)]
    at_limit = g[((g["value"] <= lo) if lo is not None else none) | ((g["value"] >= hi) if hi is not None else none)]
    month = g["delivery_start"].dt.strftime("%Y-%m")
    med = g.groupby(month)["value"].transform("median")
    mad = g.groupby(month)["value"].transform(lambda s: (s - s.median()).abs().median()) * 1.4826
    z = ((g["value"] - med) / mad.replace(0, np.nan)).abs()
    outliers = int((z > 8).sum())
    return {
        "series": series, "dataset": g["dataset"].iloc[0], "unit": g["unit"].iloc[0],
        "first_day": first, "last_day": last, "n_days": len(all_days),
        "resolution_changes": "; ".join(changes),
        "expected_periods": exp_total, "present_periods": int(present.sum()),
        "completeness_pct": round(100.0 * pres_total / exp_total, 2) if exp_total else None,
        "missing_days": len(missing_days), "missing_day_ranges": ", ".join(_ranges(missing_days)[:8]),
        "partial_days": len(partial_days), "partial_day_examples": ", ".join(str(d) for d in partial_days[:5]),
        "revised_periods": revisions,
        "out_of_limits": len(oob), "at_limit": len(at_limit),
        "outliers_z8": outliers, "longest_flat_h": round(_longest_flat_h(g), 2),
        "min": float(g["value"].min()), "mean": float(g["value"].mean()), "max": float(g["value"].max()),
    }


def run_checks(store: Store, cfg: Config) -> pd.DataFrame:
    con = store.connect()
    rev = con.execute("""
        SELECT series, count(*) AS revised FROM (
            SELECT series, delivery_start FROM records GROUP BY ALL HAVING count(DISTINCT value) > 1)
        GROUP BY series""").df()
    con.close()
    rev = dict(zip(rev["series"], rev["revised"])) if not rev.empty else {}
    rows = []
    for ds in store.datasets():
        df = store.load(dataset=ds)
        for series, g in df.groupby("series"):
            rows.append(check_series(cfg, series, g, int(rev.get(series, 0))))
    return pd.DataFrame(rows)


def monthly_stats(store: Store, cfg: Config, series: list[str] | None = None) -> pd.DataFrame:
    series = series or PRICE_SERIES
    df = store.load(series=series, qh=True)
    if df.empty:
        return pd.DataFrame(columns=["series", "month", "mean", "min", "max", "std", "n"])
    df["month"] = df["delivery_start"].dt.tz_convert(cfg.tz).dt.strftime("%Y-%m")
    return (df.groupby(["series", "month"])["value"]
              .agg(["mean", "min", "max", "std", "count"]).rename(columns={"count": "n"}).reset_index())


def period_label(ts_local: pd.Series) -> pd.Series:
    return ts_local.dt.year.astype(str) + "-H" + np.where(ts_local.dt.month <= 6, "1", "2")


def by_period(store: Store, cfg: Config, series: list[str]) -> pd.DataFrame:
    """Mean value per half-year and per market regime."""
    df = store.load(series=series, qh=True)
    if df.empty:
        return pd.DataFrame()
    local = df["delivery_start"].dt.tz_convert(cfg.tz)
    df["half_year"] = period_label(local)
    bounds = sorted((pd.Timestamp(v["date"]).tz_localize(cfg.tz), k) for k, v in cfg["regimes"].items())
    names = np.array([f"before {bounds[0][1]}"] + [f"after {k}" for _, k in bounds])
    idx = np.searchsorted(np.array([b.value for b, _ in bounds]), df["delivery_start"].astype("int64").to_numpy(), side="right")
    df["regime"] = names[idx]
    a = df.groupby(["series", "half_year"])["value"].mean().unstack("series")
    b = df.groupby(["series", "regime"])["value"].mean().unstack("series")
    a.index.name = b.index.name = "period"
    return pd.concat([a, b])


def cross_check(store: Store, a: str, b: str, tol: float = 0.01) -> dict | None:
    w = store.wide([a, b])
    if a not in w or b not in w:
        return None
    both = w[[a, b]].dropna()
    if both.empty:
        return {"a": a, "b": b, "overlap_periods": 0}
    diff = (both[a] - both[b]).abs()
    worst = diff.idxmax()
    return {"a": a, "b": b, "overlap_periods": len(both), "match_share": float((diff <= tol).mean()),
            "max_abs_diff": float(diff.max()), "worst_period_utc": str(worst), "mean_abs_diff": float(diff.mean())}
