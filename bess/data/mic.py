"""OMIE continuous intraday trades aggregated to quarter-hour prices known at each gate."""
from __future__ import annotations

import io
import logging
import zipfile
from datetime import date, timedelta

import numpy as np
import pandas as pd

from ..config import Config
from .http import Http
from .pubtime import market_time
from .store import Store

log = logging.getLogger(__name__)

SOURCE = "omie"
PARENT = "trades"
QH = pd.Timedelta(minutes=15)


def _months(start: date, end: date) -> list[date]:
    out, m = [], date(start.year, start.month, 1)
    while m <= end:
        out.append(m)
        m = date(m.year + (m.month == 12), m.month % 12 + 1, 1)
    return out


def parse_trades(text: str, tz: str) -> pd.DataFrame:
    """Trades of one daily file as UTC contract start, end, match time, price and quantity."""
    df = pd.read_csv(io.StringIO(text), sep=";", skiprows=2, dtype=str)
    t_col = next(c for c in df.columns if c.startswith("Momento"))
    df = df[df["Contrato"].notna() & df[t_col].fillna("").str.contains(":")]
    c = df["Contrato"].str.strip()
    a = pd.to_datetime(c.str[:14], format="%Y%m%d %H:%M", errors="coerce")
    b = pd.to_datetime(c.str[15:29], format="%Y%m%d %H:%M", errors="coerce")
    t = pd.to_datetime(df[t_col].str.strip(), format="%d/%m/%Y %H:%M:%S", errors="coerce")
    out = pd.DataFrame({
        "start": a.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC"),
        "end": b.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC"),
        "t": t.dt.tz_localize(tz, ambiguous="NaT", nonexistent="NaT").dt.tz_convert("UTC"),
        "price": pd.to_numeric(df["Precio"].str.replace(",", "."), errors="coerce"),
        "mwh": pd.to_numeric(df["Cantidad"].str.replace(",", "."), errors="coerce"),
    }).dropna()
    length = out["end"] - out["start"]
    return out[(length >= QH) & (length <= pd.Timedelta(hours=1)) & (length % QH == pd.Timedelta(0))]


def to_quarter_hours(tr: pd.DataFrame) -> pd.DataFrame:
    """Hourly and other multi-quarter contracts split into quarter-hours at the same price."""
    k = ((tr["end"] - tr["start"]) // QH).astype(int).to_numpy()
    rep = tr.loc[tr.index.repeat(k)].copy()
    rep["start"] = rep["start"] + QH * np.concatenate([np.arange(n) for n in k])
    rep["mwh"] = rep["mwh"] / np.repeat(k, k)
    return rep[["start", "t", "price", "mwh"]].reset_index(drop=True)


def _vwap(tr: pd.DataFrame, mask: np.ndarray) -> pd.DataFrame:
    s = tr[mask]
    g = s.assign(pq=s["price"] * s["mwh"]).groupby("start")[["pq", "mwh"]].sum()
    g = g[g["mwh"] > 0]
    return pd.DataFrame({"value": g["pq"] / g["mwh"], "mwh": g["mwh"]})


def aggregate(tr: pd.DataFrame, cfg: Config) -> pd.DataFrame:
    """Gate and rebalancing windows of quarter-hour trades as store records."""
    if tr.empty:
        return pd.DataFrame()
    days = tr["start"].dt.tz_convert(cfg.tz).dt.date
    gates = {g: days.map({d: market_time(cfg, g, "gate", d) for d in days.unique()}) for g in ("ida2", "ida3")}
    lead = pd.Timedelta(minutes=float(cfg["mic"]["rebalance_lead_min"]))
    half = pd.Timedelta(minutes=float(cfg["mic"]["rebalance_window_min"]) / 2)
    lo, hi = tr["start"] - lead - half, tr["start"] - lead + half
    parts = []
    for name, mask, pub in (
        ("mic_ida2", (tr["t"] < gates["ida2"]).to_numpy(), "ida2"),
        ("mic_ida3", (tr["t"] < gates["ida3"]).to_numpy(), "ida3"),
        ("mic_rebal", ((tr["t"] >= lo) & (tr["t"] < hi)).to_numpy(), "rebal"),
        ("mic_all", np.ones(len(tr), bool), "end"),
    ):
        v = _vwap(tr, mask)
        if v.empty:
            continue
        start = v.index.to_series()
        if pub in ("ida2", "ida3"):
            d = start.dt.tz_convert(cfg.tz).dt.date
            published = d.map({x: market_time(cfg, pub, "gate", x) for x in d.unique()})
        elif pub == "rebal":
            published = start - lead + half
        else:
            published = start + QH
        for series, col, unit in ((name, "value", "EUR/MWh"), (f"{name}_mwh", "mwh", "MWh")):
            parts.append(pd.DataFrame({
                "source": SOURCE, "dataset": "omie_mic_trades", "series": series,
                "delivery_start": start.to_numpy(), "delivery_end": (start + QH).to_numpy(), "resolution_min": 15,
                "value": v[col].to_numpy(), "unit": unit, "published_at": published.to_numpy(),
                "pub_source": "rule:assumed", "fetched_at": pd.Timestamp.now(tz="UTC")}))
    return pd.concat(parts, ignore_index=True)


class MicDownloader:
    """Monthly OMIE trade archives of the continuous intraday market."""

    def __init__(self, cfg: Config, store: Store, http: Http | None = None):
        self.cfg, self.store = cfg, store
        self.http = http or Http(cfg)
        self.dir = store.raw_dir / SOURCE / PARENT
        self.dir.mkdir(parents=True, exist_ok=True)

    def fetch(self, month: date, force: bool = False) -> bool:
        name = f"{PARENT}_{month:%Y%m}.zip"
        path = self.dir / name
        if path.exists() and not force:
            return True
        r = self.http.get(f"{self.cfg['omie']['base_url']}?parents={PARENT}&filename={name}")
        if r.status_code != 200 or not r.content.startswith(b"PK"):
            log.info("no trade archive %s (HTTP %s)", name, r.status_code)
            return False
        path.write_bytes(r.content)
        return True

    def month_records(self, month: date) -> pd.DataFrame:
        path = self.dir / f"{PARENT}_{month:%Y%m}.zip"
        if not path.exists():
            return pd.DataFrame()
        with zipfile.ZipFile(path) as z:
            tr = [parse_trades(z.read(n).decode("latin-1"), self.cfg.tz) for n in sorted(z.namelist())]
        tr = [t for t in tr if not t.empty]
        if not tr:
            return pd.DataFrame()
        return aggregate(to_quarter_hours(pd.concat(tr, ignore_index=True)), self.cfg)

    def download(self, start: date, end: date, force: bool = False) -> dict:
        n = 0
        for m in _months(start, end):
            if m >= date.today().replace(day=1):
                continue
            if not self.fetch(m, force):
                continue
            rec = self.month_records(m)
            if not rec.empty:
                n = self.store.write(rec)
            log.info("mic %s: %d records", f"{m:%Y-%m}", len(rec))
        return {"mic": n}
