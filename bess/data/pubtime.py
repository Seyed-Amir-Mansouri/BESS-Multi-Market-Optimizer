"""Publication-time rules: when each record became public (UTC)."""
from __future__ import annotations

import pandas as pd

from ..config import Config
from ..timeutils import delivery_day, local_time_utc


def market_time(cfg: Config, market: str, kind: str, day) -> pd.Timestamp:
    """UTC time of `kind` ('gate' or 'results') of `market` for delivery day `day`."""
    spec = cfg["markets"][market][kind]
    return local_time_utc(day, spec["time"], cfg.tz, spec.get("day_offset", 0))


def _per_day(days: pd.Series, fn) -> pd.Series:
    uniq = {d: fn(d) for d in pd.unique(days)}
    return days.map(uniq)


def apply_rule(cfg: Config, rule: dict, delivery_start: pd.Series, delivery_end: pd.Series,
               anchor_day: pd.Series | None = None) -> tuple[pd.Series, str]:
    """Return (published_at UTC series, pub_source label) for records under `rule`."""
    days = anchor_day if anchor_day is not None else delivery_day(delivery_start, cfg.tz)
    days = pd.Series(days.to_numpy(), index=delivery_start.index)
    kind = rule["type"]
    if kind == "market_results":
        pub = _per_day(days, lambda d: market_time(cfg, rule["market"], "results", d))
    elif kind == "day_offset":
        pub = _per_day(days, lambda d: local_time_utc(d, rule["time"], cfg.tz, rule.get("day_offset", 0)))
    elif kind == "after_delivery":
        pub = pd.to_datetime(delivery_end, utc=True) + pd.Timedelta(minutes=rule["minutes"])
    elif kind == "lead":
        pub = pd.to_datetime(delivery_start, utc=True) + pd.Timedelta(hours=rule["hours"])
    else:
        raise ValueError(f"unknown pub_rule type {kind!r}")
    source = "rule:assumed" if rule.get("assumed") else "rule"
    return pd.to_datetime(pub, utc=True), source
