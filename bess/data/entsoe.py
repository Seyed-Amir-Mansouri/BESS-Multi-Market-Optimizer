"""ENTSO-E Transparency Platform client."""
from __future__ import annotations

import logging
import re
from datetime import date, timedelta

import pandas as pd

from ..config import Config, secret
from ..timeutils import chunks, local_midnight_utc, per_day_resolution
from .pubtime import apply_rule
from .store import Store

log = logging.getLogger(__name__)

SOURCE = "entsoe"
DATASET = "entsoe"


def _slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", str(s).lower()).strip("_")


def frame_to_records(obj: pd.Series | pd.DataFrame, key: str, spec: dict, cfg: Config,
                     fetched_at: pd.Timestamp | None = None) -> pd.DataFrame:
    """entsoe-py result (tz-aware index) -> long-format records."""
    if obj is None or len(obj) == 0:
        return pd.DataFrame()
    df = obj.to_frame(name=key) if isinstance(obj, pd.Series) else obj.copy()
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = ["|".join(str(x) for x in col if str(x)) for col in df.columns]
    single = df.shape[1] == 1
    df.index = pd.to_datetime(df.index, utc=True)
    df = df[~df.index.duplicated(keep="last")].sort_index()
    out = []
    for col in df.columns:
        s = pd.to_numeric(df[col], errors="coerce").dropna()
        if s.empty:
            continue
        start = pd.Series(s.index, index=range(len(s)))
        res = per_day_resolution(start, cfg.tz)
        end = start + pd.to_timedelta(res, unit="min")
        out.append(pd.DataFrame({
            "series": key if single else f"{key}.{_slug(col)}",
            "delivery_start": start, "delivery_end": end, "resolution_min": res,
            "value": s.to_numpy(),
        }))
    if not out:
        return pd.DataFrame()
    rec = pd.concat(out, ignore_index=True)
    pub, pub_src = apply_rule(cfg, spec["pub_rule"], rec["delivery_start"], rec["delivery_end"])
    rec["published_at"] = pub
    rec["pub_source"] = pub_src
    rec["source"] = SOURCE
    rec["dataset"] = DATASET
    rec["unit"] = spec.get("unit", "")
    rec["fetched_at"] = fetched_at if fetched_at is not None else pd.Timestamp.now(tz="UTC")
    return rec


class EntsoeDownloader:
    def __init__(self, cfg: Config, store: Store, offline: bool = False):
        self.cfg = cfg
        self.store = store
        self.ecfg = cfg["entsoe"]
        self.client = None
        if not offline:
            key = secret("entsoe")
            if not key:
                raise RuntimeError("ENTSOE_API_KEY is not set in the environment")
            from entsoe import EntsoePandasClient
            h = cfg["data"]["http"]
            self.client = EntsoePandasClient(api_key=key, retry_count=h["retries"], retry_delay=int(h["backoff_s"]) * 5,
                                             timeout=h["timeout_s"])

    def _cache(self, key: str, a: date, b: date):
        return self.store.raw_path(SOURCE, key, f"{a:%Y%m%d}_{b:%Y%m%d}.parquet")

    def _query(self, spec: dict, a: date, b: date):
        start = local_midnight_utc(a, self.cfg.tz).tz_convert(self.cfg.tz)
        end = local_midnight_utc(b + timedelta(days=1), self.cfg.tz).tz_convert(self.cfg.tz)
        fn = getattr(self.client, f"query_{spec['query']}")
        return fn(self.ecfg["country_code"], start=start, end=end)

    def download(self, start: date, end: date, keys: list[str] | None = None, force: bool = False) -> dict:
        from entsoe.exceptions import NoMatchingDataError
        counts = {}
        for key, spec in self.ecfg["datasets"].items():
            if keys and key not in keys:
                continue
            for a, b in chunks(start, end, self.ecfg["chunk_days"]):
                path = self._cache(key, a, b)
                settled = b < date.today() - timedelta(days=7)
                if path.exists() and settled and not force:
                    continue
                try:
                    obj = self._query(spec, a, b)
                except NoMatchingDataError:
                    log.info("ENTSO-E %s %s..%s: no data", key, a, b)
                    continue
                except Exception as e:
                    log.error("ENTSO-E %s %s..%s failed: %s", key, a, b, e)
                    continue
                frame = obj.to_frame(name=key) if isinstance(obj, pd.Series) else obj
                if isinstance(frame.columns, pd.MultiIndex):
                    frame.columns = ["|".join(str(x) for x in c if str(x)) for c in frame.columns]
                frame.columns = [str(c) for c in frame.columns]
                frame.to_parquet(path)
        counts.update(self.parse(keys))
        if self.ecfg.get("download_outages") and (not keys or "outages" in keys):
            counts["outages"] = self.download_outages(start, end)
        return counts

    def parse(self, keys: list[str] | None = None) -> dict:
        counts = {}
        for key, spec in self.ecfg["datasets"].items():
            if keys and key not in keys:
                continue
            recs = [frame_to_records(pd.read_parquet(p), key, spec, self.cfg)
                    for p in sorted((self.store.raw_dir / SOURCE / key).glob("*.parquet"))]
            recs = [r for r in recs if not r.empty]
            counts[key] = self.store.write(pd.concat(recs, ignore_index=True)) if recs else 0
        return counts

    def download_outages(self, start: date, end: date) -> int:
        """Unavailability of generation units -> events table `entsoe_outages`."""
        from entsoe.exceptions import NoMatchingDataError
        frames = []
        for a, b in chunks(start, end, self.ecfg["chunk_days"]):
            try:
                df = self.client.query_unavailability_of_generation_units(
                    self.ecfg["country_code"],
                    start=local_midnight_utc(a, self.cfg.tz).tz_convert(self.cfg.tz),
                    end=local_midnight_utc(b + timedelta(days=1), self.cfg.tz).tz_convert(self.cfg.tz))
            except NoMatchingDataError:
                continue
            except Exception as e:
                log.error("ENTSO-E outages %s..%s failed: %s", a, b, e)
                continue
            df = df.reset_index()
            frames.append(df)
        if not frames:
            return 0
        df = pd.concat(frames, ignore_index=True)
        df.columns = [_slug(c) for c in df.columns]
        for c in df.columns:
            if df[c].dtype == object:
                df[c] = df[c].astype(str)
        if "created_doc_time" in df.columns:
            df["published_at"] = pd.to_datetime(df["created_doc_time"], utc=True, errors="coerce")
        key = [c for c in ("mrid", "revision", "start", "end") if c in df.columns] or None
        return self.store.write_events("entsoe_outages", df, key=key)
