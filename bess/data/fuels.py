"""Daily Iberian gas and EU carbon prices as values known before each delivery day."""
from __future__ import annotations

import io
import logging
from datetime import date, timedelta

import pandas as pd

from ..config import Config
from ..timeutils import local_midnight_utc
from .http import Http
from .store import Store

log = logging.getLogger(__name__)

SOURCE = "fuels"
QH = pd.Timedelta(minutes=15)


def parse_mibgas(text: str) -> pd.Series:
    """Day-ahead gas index by delivery day and its trading day."""
    df = pd.read_csv(io.StringIO(text), sep=";", skiprows=1, dtype=str)
    df.columns = [c.strip() for c in df.columns]
    price = next(c for c in df.columns if "Price" in c)
    df = df[df["Product"].str.strip() == "GDAES_D+1"]
    out = pd.DataFrame({
        "delivery": pd.to_datetime(df["First Day Delivery"].str.strip(), format="%d/%m/%Y", errors="coerce").dt.date,
        "traded": pd.to_datetime(df["Trading day"].str.strip(), format="%d/%m/%Y", errors="coerce").dt.date,
        "value": pd.to_numeric(df[price].str.replace(",", "."), errors="coerce"),
    }).dropna()
    return out.drop_duplicates("delivery", keep="last").set_index("delivery").sort_index()


def parse_eua(text: str) -> pd.Series:
    """EUA closing price by trading day."""
    df = pd.read_csv(io.StringIO(text), sep=";", dtype=str)
    d = pd.to_datetime(df.iloc[:, 0].str.strip(), format="%d-%m-%Y", errors="coerce").dt.date
    v = pd.to_numeric(df["EUA"].str.replace(",", "."), errors="coerce")
    s = pd.Series(v.to_numpy(), index=d).dropna()
    return s[s > 0].groupby(level=0).last().sort_index()


def known_before(cfg: Config, days: list, gas: pd.DataFrame, eua: pd.Series) -> pd.DataFrame:
    """Gas price of the previous gas day and the last EUA close two days before each delivery day."""
    rows = []
    for d in days:
        prev = d - timedelta(days=1)
        if prev in gas.index:
            g = gas.loc[prev]
            rows.append(("gas_es_prev", d, float(g["value"]), g["traded"] + timedelta(days=1)))
        e = eua[eua.index <= d - timedelta(days=2)]
        if len(e):
            rows.append(("eua_prev", d, float(e.iloc[-1]), e.index[-1] + timedelta(days=1)))
    parts = []
    for series, d, v, pub in rows:
        a, b = local_midnight_utc(d, cfg.tz), local_midnight_utc(d + timedelta(days=1), cfg.tz)
        t = pd.date_range(a, b, freq=QH, inclusive="left")
        parts.append(pd.DataFrame({
            "source": SOURCE, "dataset": SOURCE, "series": series, "delivery_start": t, "delivery_end": t + QH,
            "resolution_min": 15, "value": v, "unit": "EUR/MWh" if series.startswith("gas") else "EUR/t",
            "published_at": local_midnight_utc(pub, cfg.tz), "pub_source": "rule:assumed",
            "fetched_at": pd.Timestamp.now(tz="UTC")}))
    return pd.concat(parts, ignore_index=True) if parts else pd.DataFrame()


class FuelDownloader:
    """Yearly MIBGAS and SendeCO2 files."""

    def __init__(self, cfg: Config, store: Store, http: Http | None = None, offline: bool = False):
        self.cfg, self.store, self.offline = cfg, store, offline
        self.http = http or Http(cfg)
        self.dir = store.raw_dir / SOURCE
        self.dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, name: str, url: str, force: bool) -> str | None:
        path = self.dir / name
        if not self.offline and (not path.exists() or force):
            r = self.http.get(url)
            if r.status_code != 200 or not r.content:
                log.info("no file %s (HTTP %s)", name, r.status_code)
                return None
            path.write_bytes(r.content)
        if not path.exists():
            return None
        return path.read_bytes().decode("latin-1")

    def download(self, start: date, end: date, force: bool = False) -> dict:
        f = self.cfg["fuels"]
        gas, eua = [], []
        for y in range(start.year - 1, end.year + 1):
            now = y >= date.today().year
            t = self.fetch(f"mibgas_{y}.csv", f["mibgas_url"].format(year=y), force or now)
            if t:
                gas.append(parse_mibgas(t))
            t = self.fetch(f"eua_{y}.csv", f["eua_url"].format(year=y), force or now)
            if t:
                eua.append(parse_eua(t))
        if not gas or not eua:
            return {"fuels": 0}
        g = pd.concat(gas)
        g = g[~g.index.duplicated(keep="last")].sort_index()
        e = pd.concat(eua).groupby(level=0).last().sort_index()
        days = [start + timedelta(days=i) for i in range((end - start).days + 1)]
        rec = known_before(self.cfg, days, g, e)
        n = self.store.write(rec) if not rec.empty else 0
        log.info("fuels: %d records", len(rec))
        return {"fuels": n}
