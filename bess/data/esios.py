"""REE eSIOS API (https://api.esios.ree.es). Requires ESIOS_API_KEY in the environment.

Each configured indicator becomes one series (suffixed `.<geo_id>` when the indicator has
several geographies and none is selected). Resolution is inferred per local day from the
timestamps, so changes such as hourly -> 15-min are detected rather than assumed.
"""
from __future__ import annotations

import json
import logging
import unicodedata
from datetime import date, timedelta

import pandas as pd

from ..config import Config, secret
from ..timeutils import chunks, delivery_day, local_midnight_utc, per_day_resolution
from .http import Http
from .pubtime import apply_rule
from .store import Store

log = logging.getLogger(__name__)

SOURCE = "esios"
DATASET = "esios"
TIEMPO_MIN = {"cinco minutos": 5, "diez minutos": 10, "quince minutos": 15, "quinceminutal": 15,
              "hora": 60, "horario": 60, "dia": 1440, "día": 1440}


def strip_accents(s: str) -> str:
    return "".join(c for c in unicodedata.normalize("NFKD", s) if not unicodedata.combining(c)).lower()


def name_matches(name: str, expect: list[str]) -> bool:
    n = strip_accents(name or "")
    return all(strip_accents(w) in n for w in expect)


def _tiempo_minutes(indicator: dict) -> int | None:
    for t in indicator.get("tiempo") or []:
        m = TIEMPO_MIN.get(strip_accents(str(t.get("name", ""))))
        if m:
            return m
    return None


def parse_indicator(payload: dict, key: str, spec: dict, cfg: Config,
                    fetched_at: pd.Timestamp | None = None) -> pd.DataFrame:
    """eSIOS /indicators/{id} JSON -> long-format records."""
    ind = payload.get("indicator", payload)
    values = ind.get("values") or []
    if not values:
        return pd.DataFrame()
    v = pd.DataFrame(values)
    ts_col = "datetime_utc" if "datetime_utc" in v.columns else "datetime"
    v["delivery_start"] = pd.to_datetime(v[ts_col], utc=True)
    v["value"] = pd.to_numeric(v["value"], errors="coerce")
    if "geo_id" not in v.columns:
        v["geo_id"] = 0
    wanted = spec.get("geo_ids")
    if wanted:
        v = v[v["geo_id"].isin(wanted)]
    multi_geo = v["geo_id"].nunique() > 1
    fallback = _tiempo_minutes(ind)
    out = []
    for geo, g in v.groupby("geo_id"):
        g = g.sort_values("delivery_start").drop_duplicates("delivery_start", keep="last").copy()
        g["day"] = delivery_day(g["delivery_start"], cfg.tz).to_numpy()
        g["resolution_min"] = per_day_resolution(g["delivery_start"], cfg.tz, fallback or 60).to_numpy()
        daily = g["resolution_min"] >= 1440
        g["delivery_end"] = g["delivery_start"] + pd.to_timedelta(g["resolution_min"], unit="min")
        if daily.any():
            g.loc[daily, "delivery_end"] = [local_midnight_utc(d + timedelta(days=1), cfg.tz) for d in g.loc[daily, "day"]]
        g["series"] = f"{key}.{geo}" if multi_geo else key
        out.append(g)
    df = pd.concat(out, ignore_index=True)
    pub, pub_src = apply_rule(cfg, spec["pub_rule"], df["delivery_start"], df["delivery_end"])
    return pd.DataFrame({
        "source": SOURCE, "dataset": DATASET, "series": df["series"],
        "delivery_start": df["delivery_start"], "delivery_end": df["delivery_end"],
        "resolution_min": df["resolution_min"], "value": df["value"],
        "unit": spec.get("unit", ""), "published_at": pub, "pub_source": pub_src,
        "fetched_at": fetched_at if fetched_at is not None else pd.Timestamp.now(tz="UTC"),
    }).dropna(subset=["value"])


class EsiosClient:
    def __init__(self, cfg: Config, store: Store, http: Http | None = None, offline: bool = False):
        self.cfg = cfg
        self.store = store
        self.base = cfg["esios"]["base_url"]
        if offline:
            self.http = None
            return
        key = secret("esios")
        if not key:
            raise RuntimeError("ESIOS_API_KEY is not set in the environment")
        self.http = http or Http(cfg, headers={
            "Accept": "application/json; application/vnd.esios-api-v1+json",
            "Content-Type": "application/json",
            "x-api-key": key,
        })

    def _get(self, path: str, params=None) -> dict:
        r = self.http.get(f"{self.base}{path}", params=params)
        if r.status_code in (401, 403):
            raise RuntimeError(f"eSIOS refused the API key (HTTP {r.status_code}) for {path}")
        r.raise_for_status()
        return r.json()

    def catalog(self) -> pd.DataFrame:
        """All indicators (id, name, short_name) - stored as events table `esios_catalog`."""
        data = self._get("/indicators")
        df = pd.DataFrame(data.get("indicators", []))
        keep = [c for c in ("id", "name", "short_name", "description") if c in df.columns]
        df = df[keep].copy()
        if "description" in df.columns:
            df["description"] = df["description"].astype(str).str.replace(r"<[^>]+>", " ", regex=True).str.slice(0, 400)
        df["fetched_at"] = pd.Timestamp.now(tz="UTC")
        self.store.write_events("esios_catalog", df, key=["id"])
        return df

    def metadata(self, indicator_id: int) -> dict:
        today = date.today()
        params = {"start_date": f"{today - timedelta(days=2)}T00:00:00Z", "end_date": f"{today - timedelta(days=1)}T00:00:00Z"}
        ind = self._get(f"/indicators/{indicator_id}", params).get("indicator", {})
        ind.pop("values", None)
        return ind

    def fetch_raw(self, key: str, spec: dict, a: date, b: date, force: bool = False) -> dict:
        path = self.store.raw_path(SOURCE, key, f"{spec['id']}_{a:%Y%m%d}_{b:%Y%m%d}.json")
        settled = b < date.today() - timedelta(days=7)
        if path.exists() and settled and not force:
            return json.loads(path.read_text(encoding="utf-8"))
        start = local_midnight_utc(a, self.cfg.tz)
        end = local_midnight_utc(b + timedelta(days=1), self.cfg.tz) - pd.Timedelta(seconds=1)
        params = [("start_date", start.strftime("%Y-%m-%dT%H:%M:%SZ")), ("end_date", end.strftime("%Y-%m-%dT%H:%M:%SZ"))]
        for g in spec.get("geo_ids") or []:
            params.append(("geo_ids[]", g))
        if spec.get("time_trunc"):
            params.append(("time_trunc", spec["time_trunc"]))
            params.append(("time_agg", spec.get("time_agg", "average")))
        payload = self._get(f"/indicators/{spec['id']}", params)
        path.write_text(json.dumps(payload), encoding="utf-8")
        return payload

    def download(self, start: date, end: date, keys: list[str] | None = None, force: bool = False) -> dict:
        counts = {}
        indicators = self.cfg["esios"]["indicators"]
        for key, spec in indicators.items():
            if keys and key not in keys:
                continue
            if spec.get("id") is None:
                log.warning("eSIOS %s: no indicator id configured - skipped (see `bess esios-catalog`)", key)
                continue
            recs = []
            for a, b in chunks(start, end, self.cfg["esios"]["chunk_days"]):
                try:
                    payload = self.fetch_raw(key, spec, a, b, force)
                except Exception as e:
                    log.error("eSIOS %s (%s) %s..%s failed: %s", key, spec["id"], a, b, e)
                    continue
                recs.append(parse_indicator(payload, key, spec, self.cfg))
            recs = [r for r in recs if not r.empty]
            counts[key] = self.store.write(pd.concat(recs, ignore_index=True)) if recs else 0
            log.info("eSIOS %s: %s rows", key, counts[key])
        return counts

    def parse(self, start: date, end: date) -> dict:
        """Rebuild clean records from cached raw JSON (no network)."""
        counts = {}
        for key, spec in self.cfg["esios"]["indicators"].items():
            recs = []
            for p in sorted((self.store.raw_dir / SOURCE / key).glob("*.json")):
                recs.append(parse_indicator(json.loads(p.read_text(encoding="utf-8")), key, spec, self.cfg))
            recs = [r for r in recs if not r.empty]
            counts[key] = self.store.write(pd.concat(recs, ignore_index=True)) if recs else 0
        return counts
