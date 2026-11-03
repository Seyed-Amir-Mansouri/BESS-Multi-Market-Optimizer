"""Gap filling from alternative sources, physical limits and market availability."""
from __future__ import annotations

import logging
from datetime import timedelta

import pandas as pd

from ..config import Config
from ..timeutils import delivery_day
from .store import Store

log = logging.getLogger(__name__)


def _days(df: pd.DataFrame, tz: str) -> set:
    return set(delivery_day(df["delivery_start"], tz))


def fill_from_sources(store: Store, cfg: Config) -> pd.DataFrame:
    """Copy missing days inside a series' own range from the first source that has them."""
    rows = []
    for target, sources in cfg["fill"]["from"].items():
        t = store.load(series=target)
        if t.empty:
            continue
        own = t[~t["pub_source"].str.startswith("fill:")]
        have = _days(t, cfg.tz)
        first, last = min(_days(own, cfg.tz)), max(_days(own, cfg.tz))
        done = set()
        for src in sources:
            s = store.load(series=src)
            if s.empty:
                continue
            s = s.assign(day=delivery_day(s["delivery_start"], cfg.tz).to_numpy())
            s = s[(s["day"] >= first) & (s["day"] <= last) & ~s["day"].isin(have | done)]
            if s.empty:
                continue
            store.write(s.assign(source="fill", dataset=t["dataset"].iloc[0], series=target, unit=t["unit"].iloc[0],
                                 pub_source=f"fill:{src}", fetched_at=pd.Timestamp.now(tz="UTC")))
            done |= set(s["day"].unique())
        f = store.load(series=target)
        f = f[f["pub_source"].str.startswith("fill:")]
        days = delivery_day(f["delivery_start"], cfg.tz).to_numpy()
        rows += [{"series": target, "day": str(d), "source": src.removeprefix("fill:")}
                 for d, src in sorted(set(zip(days, f["pub_source"])))]
    out = pd.DataFrame(rows, columns=["series", "day", "source"])
    store.write_events("fills", out, replace=True)
    return out


def apply_physical_limits(store: Store, cfg: Config) -> int:
    """Replace values outside physical limits: below min by min, above max by the previous valid value."""
    n = 0
    lims = cfg["fill"]["physical_limits"]
    for ds in store.datasets():
        df = store.load(dataset=ds)
        for var, lim in lims.items():
            part = df[df["series"].str.contains(f".{var}.", regex=False)]
            for series, g in part.groupby("series"):
                g = g.sort_values("delivery_start")
                low, high = g["value"] < lim["min"], g["value"] > lim["max"]
                if not (low.any() or high.any()):
                    continue
                v = g["value"].mask(high).ffill().fillna(lim["min"]).clip(lower=lim["min"])
                fixed = g[low | high].assign(value=v[low | high], pub_source="fill:physical_limit")
                store.write(fixed)
                n += len(fixed)
                log.info("%s: %d values outside [%s, %s]", series, len(fixed), lim["min"], lim["max"])
    return n


def unavailable_days(store: Store, cfg: Config) -> pd.DataFrame:
    """Delivery days on which a market has no published result in any source."""
    rows = []
    for market, series in cfg["fill"]["unavailable"].items():
        missing = {}
        for s in series:
            df = store.load(series=s)
            if df.empty:
                continue
            days = _days(df, cfg.tz)
            a, b = min(days), max(days)
            for i in range((b - a).days + 1):
                d = a + timedelta(days=i)
                if d not in days:
                    missing.setdefault(d, []).append(s)
        rows += [{"market": market, "day": str(d), "missing": ", ".join(v)} for d, v in sorted(missing.items())]
    out = pd.DataFrame(rows, columns=["market", "day", "missing"])
    store.write_events("market_unavailable", out, replace=True)
    return out


def run_fill(store: Store, cfg: Config) -> dict:
    fills = fill_from_sources(store, cfg)
    clipped = apply_physical_limits(store, cfg)
    unavailable = unavailable_days(store, cfg)
    return {"fills": fills, "clipped": clipped, "unavailable": unavailable}
