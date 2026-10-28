"""Publication-time timeline: what is known, and up to which delivery time, at each gate.

For every series and every market gate, `gate_availability` answers: at the gate for
delivery day D, how far does published data of this series reach (in hours relative to
D 00:00 local)? E.g. +24 = all of D is known (a forecast for D, or D's own DA price at the
IDA1 gate); -11 = data only up to D-1 13:00 (a real-time measurement at the DA gate).
This is the contract the Phase 3 feature builder must respect.
"""
from __future__ import annotations

from datetime import timedelta

import numpy as np
import pandas as pd

from ..config import Config
from ..timeutils import local_midnight_utc
from .pubtime import market_time
from .store import Store

GATES = ["da", "ida1", "afrr_capacity", "ida2", "ida3"]


def series_summary(store: Store) -> pd.DataFrame:
    con = store.connect()
    df = con.execute("""
        SELECT source, dataset, series, any_value(unit) AS unit,
               min(delivery_start) AS first_delivery, max(delivery_end) AS last_delivery,
               count(*) AS n_records,
               string_agg(DISTINCT CAST(resolution_min AS VARCHAR), ',') AS resolutions_min,
               string_agg(DISTINCT pub_source, ',') AS pub_sources,
               median(epoch(published_at) - epoch(delivery_start)) / 3600.0 AS median_pub_lag_h,
               min(epoch(published_at) - epoch(delivery_start)) / 3600.0 AS min_pub_lag_h,
               max(epoch(published_at) - epoch(delivery_start)) / 3600.0 AS max_pub_lag_h
        FROM records GROUP BY ALL ORDER BY source, series
    """).df()
    con.close()
    return df


def gate_availability(store: Store, cfg: Config, days: list | None = None, sample: int = 120) -> pd.DataFrame:
    """Median horizon (h from D 00:00 local) of published data per series at each gate.

    days: delivery days to evaluate (default: the last `sample` days present in the store).
    Returns columns: series, gate, horizon_h_median, full_day_share, n_days.
    """
    con = store.connect()
    data = con.execute("SELECT series, delivery_end, published_at FROM records").df()
    con.close()
    if data.empty:
        return pd.DataFrame(columns=["series", "gate", "horizon_h_median", "full_day_share", "n_days"])
    for c in ("delivery_end", "published_at"):
        data[c] = pd.to_datetime(data[c], utc=True).astype("datetime64[ns, UTC]")
    if days is None:
        last = data["delivery_end"].max().tz_convert(cfg.tz).date()
        days = list(pd.date_range(end=pd.Timestamp(last), periods=sample, freq="D").date)
    d0 = np.array([local_midnight_utc(d, cfg.tz).value for d in days])
    d1 = np.array([local_midnight_utc(d + timedelta(days=1), cfg.tz).value for d in days])
    rows = []
    for series, g in data.groupby("series"):
        g = g.sort_values("published_at")
        pub = g["published_at"].astype("int64").to_numpy()
        reach = np.maximum.accumulate(g["delivery_end"].astype("int64").to_numpy())
        for gate in GATES:
            if gate not in cfg["markets"]:
                continue
            t = np.array([market_time(cfg, gate, "gate", d).value for d in days])
            idx = np.searchsorted(pub, t, side="right") - 1
            ok = idx >= 0
            if not ok.any():
                rows.append((series, gate, np.nan, 0.0, 0))
                continue
            known = reach[idx[ok]]
            horizon = (known - d0[ok]) / 3.6e12
            rows.append((series, gate, float(np.median(horizon)), float(np.mean(known >= d1[ok])), int(ok.sum())))
    return pd.DataFrame(rows, columns=["series", "gate", "horizon_h_median", "full_day_share", "n_days"])


def gate_schedule(cfg: Config, day) -> pd.DataFrame:
    """Gate and results times (local) for one delivery day - the decision sequence."""
    rows = []
    for m, spec in cfg["markets"].items():
        rows.append({
            "market": m, "label": spec.get("label", m),
            "gate_local": market_time(cfg, m, "gate", day).tz_convert(cfg.tz),
            "results_local": market_time(cfg, m, "results", day).tz_convert(cfg.tz),
            "covers": f"{spec['coverage']['from']}-{spec['coverage']['to']}",
        })
    return pd.DataFrame(rows).sort_values("gate_local").reset_index(drop=True)
