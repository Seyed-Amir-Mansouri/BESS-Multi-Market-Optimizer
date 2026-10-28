"""Open-Meteo Previous Runs API: archived weather FORECASTS as they were issued (not actuals).

Variable `<var>_previous_dayN` holds, for each target hour, the value forecast by the model
run about N days earlier. Publication time = target - N days + model_delay_h (an assumption,
labelled `rule:assumed`). With lead 2 every hour of D is public before the DA gate (D-1
12:00); lead 1 is only usable at the intraday gates.
"""
from __future__ import annotations

import json
import logging
from datetime import date, timedelta

import pandas as pd

from ..config import Config
from ..timeutils import chunks
from .http import Http
from .pubtime import apply_rule
from .store import Store

log = logging.getLogger(__name__)

SOURCE = "openmeteo"
DATASET = "weather"


def series_name(loc: str, var: str, lead: int) -> str:
    return f"wx.{loc}.{var}.d{lead}"


def parse_previous_runs(payload: list | dict, locations: list[str], cfg: Config,
                        fetched_at: pd.Timestamp | None = None) -> pd.DataFrame:
    """Previous Runs API JSON (one object per location, same order as requested) -> records."""
    objs = payload if isinstance(payload, list) else [payload]
    if len(objs) != len(locations):
        raise ValueError(f"expected {len(locations)} location blocks, got {len(objs)}")
    wcfg = cfg["weather"]
    out = []
    for loc, obj in zip(locations, objs):
        hourly = obj.get("hourly") or {}
        units = obj.get("hourly_units") or {}
        times = pd.to_datetime(pd.Series(hourly.get("time", [])), utc=True)
        for col, vals in hourly.items():
            if col == "time" or "_previous_day" not in col:
                continue
            var, lead = col.rsplit("_previous_day", 1)
            lead = int(lead)
            start = times
            end = start + pd.Timedelta(hours=1)
            rule = {"type": "lead", "hours": -24 * lead + wcfg["model_delay_h"], "assumed": True}
            pub, src = apply_rule(cfg, rule, start, end)
            out.append(pd.DataFrame({
                "source": SOURCE, "dataset": DATASET, "series": series_name(loc, var, lead),
                "delivery_start": start, "delivery_end": end, "resolution_min": 60,
                "value": pd.to_numeric(pd.Series(vals), errors="coerce"),
                "unit": units.get(col, ""), "published_at": pub, "pub_source": src,
                "fetched_at": fetched_at if fetched_at is not None else pd.Timestamp.now(tz="UTC"),
            }))
    if not out:
        return pd.DataFrame()
    return pd.concat(out, ignore_index=True).dropna(subset=["value"])


class WeatherDownloader:
    def __init__(self, cfg: Config, store: Store, http: Http | None = None):
        self.cfg = cfg
        self.store = store
        self.http = http or Http(cfg)
        self.wcfg = cfg["weather"]

    def params(self, a: date, b: date) -> dict:
        locs = self.wcfg["locations"]
        hourly = [f"{v}_previous_day{n}" for v in self.wcfg["variables"] for n in self.wcfg["lead_days"]]
        return {
            "latitude": ",".join(str(l["lat"]) for l in locs.values()),
            "longitude": ",".join(str(l["lon"]) for l in locs.values()),
            "hourly": ",".join(hourly),
            "start_date": a.isoformat(), "end_date": b.isoformat(),
            "timezone": "GMT", "wind_speed_unit": "ms", "models": self.wcfg["model"],
        }

    def download(self, start: date, end: date, force: bool = False) -> dict:
        locations = list(self.wcfg["locations"])
        recs = []
        for a, b in chunks(start, end, self.wcfg["chunk_days"]):
            path = self.store.raw_path(SOURCE, f"previous_runs_{a:%Y%m%d}_{b:%Y%m%d}.json")
            settled = b < date.today() - timedelta(days=7)
            if not (path.exists() and settled and not force):
                r = self.http.get(self.wcfg["base_url"], params=self.params(a, b))
                if r.status_code != 200:
                    log.error("Open-Meteo %s..%s: HTTP %s %s", a, b, r.status_code, r.text[:300])
                    continue
                path.write_text(r.text, encoding="utf-8")
            recs.append(parse_previous_runs(json.loads(path.read_text(encoding="utf-8")), locations, self.cfg))
        recs = [r for r in recs if not r.empty]
        return {"weather": self.store.write(pd.concat(recs, ignore_index=True)) if recs else 0}

    def parse(self) -> dict:
        locations = list(self.wcfg["locations"])
        recs = [parse_previous_runs(json.loads(p.read_text(encoding="utf-8")), locations, self.cfg)
                for p in sorted((self.store.raw_dir / SOURCE).glob("previous_runs_*.json"))]
        recs = [r for r in recs if not r.empty]
        return {"weather": self.store.write(pd.concat(recs, ignore_index=True)) if recs else 0}
