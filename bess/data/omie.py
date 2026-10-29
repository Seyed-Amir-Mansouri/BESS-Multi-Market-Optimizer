"""OMIE public files: day-ahead (marginalpdbc) and intraday-auction (marginalpibc) prices."""
from __future__ import annotations

import io
import json
import logging
import re
import zipfile
from datetime import date, datetime, timedelta
from pathlib import Path

import pandas as pd

from ..config import Config
from ..timeutils import daterange, local_midnight_utc, resolution_from_max_period
from .http import Http
from .pubtime import apply_rule
from .store import Store

log = logging.getLogger(__name__)

SOURCE = "omie"
HEADERS = {"marginalpdbc": "MARGINALPDBC", "marginalpibc": "MARGINALPIBC"}
FNAME_RE = re.compile(r"^(?P<parent>[a-z_]+?)_(?P<date>\d{8})(?P<session>\d{2})?\.(?P<version>\d+)$")


# ------------------------------------------------------------------------------ parsing
def parse_marginal(text: str) -> pd.DataFrame:
    """Parse a marginalpdbc / marginalpibc file into rows (day, period, price_pt, price_es)."""
    rows = []
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("*"):
            continue
        parts = [p.strip() for p in line.split(";")]
        if not parts[0].isdigit():
            continue
        try:
            y, m, d, p = (int(x) for x in parts[:4])
            pt = float(parts[4].replace(",", ".")) if parts[4] else float("nan")
            es = float(parts[5].replace(",", ".")) if len(parts) > 5 and parts[5] else float("nan")
        except (ValueError, IndexError) as e:
            raise ValueError(f"unparseable OMIE row {line!r}") from e
        rows.append((date(y, m, d), p, pt, es))
    return pd.DataFrame(rows, columns=["day", "period", "price_pt", "price_es"])


def marginal_to_records(rows: pd.DataFrame, market: str, cfg: Config, file_date: date,
                        fetched_at: pd.Timestamp | None = None) -> pd.DataFrame:
    """Long-format records for one parsed marginal file (series <market>_price_es / _pt)."""
    if rows.empty:
        return pd.DataFrame()
    tz = cfg.tz
    rows = rows.copy()
    res = rows.groupby("day")["period"].transform("max").map(resolution_from_max_period)
    midnight = rows["day"].map({d: local_midnight_utc(d, tz) for d in rows["day"].unique()})
    rows["delivery_start"] = pd.to_datetime(midnight, utc=True) + pd.to_timedelta((rows["period"] - 1) * res, unit="min")
    rows["delivery_end"] = rows["delivery_start"] + pd.to_timedelta(res, unit="min")
    rows["resolution_min"] = res
    anchor = pd.Series([file_date] * len(rows), index=rows.index)
    pub, pub_src = apply_rule(cfg, {"type": "market_results", "market": market},
                              rows["delivery_start"], rows["delivery_end"], anchor_day=anchor)
    fetched = fetched_at if fetched_at is not None else pd.Timestamp.now(tz="UTC")
    out = []
    for zone in ("es", "pt"):
        r = pd.DataFrame({
            "source": SOURCE,
            "dataset": f"omie_{'da' if market == 'da' else 'ida'}",
            "series": f"{market}_price_{zone}",
            "delivery_start": rows["delivery_start"],
            "delivery_end": rows["delivery_end"],
            "resolution_min": rows["resolution_min"],
            "value": rows[f"price_{zone}"],
            "unit": "EUR/MWh",
            "published_at": pub,
            "pub_source": pub_src,
            "fetched_at": fetched,
        })
        out.append(r.dropna(subset=["value"]))
    return pd.concat(out, ignore_index=True)


def _num_es(s: str) -> float:
    """Spanish number format '1.234,5' -> 1234.5."""
    s = s.strip().replace(".", "").replace(",", ".")
    return float(s) if s else float("nan")


def parse_curve(text: str) -> tuple[pd.DataFrame, pd.Timestamp | None]:
    """Parse a curva_pbc_uof file into steps and issue time."""
    lines = text.splitlines()
    issued = None
    m = re.search(r"Fecha Emisi[oó]n\s*:\s*(\d{2}/\d{2}/\d{4})\s*-\s*(\d{2}:\d{2})", text[:500])
    if m:
        issued = pd.Timestamp(datetime.strptime(" ".join(m.groups()), "%d/%m/%Y %H:%M"))
    header_idx = next((i for i, l in enumerate(lines) if ("Hora" in l or "Periodo" in l) and "Precio" in l), None)
    if header_idx is None:
        return pd.DataFrame(columns=["period", "day", "side", "energy", "price", "status"]), issued
    header = [h.strip() for h in lines[header_idx].split(";")]

    def col(*names):
        for i, h in enumerate(header):
            if any(n.lower() in h.lower() for n in names):
                return i
        raise ValueError(f"curve header lacks {names}: {header}")
    ip, idate, iside = col("Hora", "Periodo"), col("Fecha"), col("Tipo Oferta")
    ie, ipr, ist = col("Energ"), col("Precio"), col("Ofertada", "Casada")
    rows = []
    for l in lines[header_idx + 1:]:
        parts = l.split(";")
        if len(parts) <= max(ip, idate, iside, ie, ipr, ist) or not parts[ip].strip().isdigit():
            continue
        rows.append((int(parts[ip]), datetime.strptime(parts[idate].strip(), "%d/%m/%Y").date(),
                     parts[iside].strip(), _num_es(parts[ie]), _num_es(parts[ipr]), parts[ist].strip()))
    return pd.DataFrame(rows, columns=["period", "day", "side", "energy", "price", "status"]), issued


def curve_to_volume_records(steps: pd.DataFrame, issued_local: pd.Timestamp | None, cfg: Config,
                            file_date: date) -> pd.DataFrame:
    """Matched DA sell volume per period (sum of matched sell steps) as a time series."""
    if steps.empty:
        return pd.DataFrame()
    sold = steps[(steps["side"] == "V") & (steps["status"] == "C")]
    vol = sold.groupby(["day", "period"], as_index=False)["energy"].sum()
    rows = vol.rename(columns={"energy": "price_es"})
    rows["price_pt"] = float("nan")
    rec = marginal_to_records(rows[["day", "period", "price_pt", "price_es"]], "da", cfg, file_date)
    rec = rec[rec["series"] == "da_price_es"].copy()
    rec["series"] = "da_matched_sell_volume"
    rec["dataset"] = "omie_da_volume"
    rec["unit"] = "MWh"
    if issued_local is not None:
        rec["published_at"] = issued_local.tz_localize(cfg.tz).tz_convert("UTC")
        rec["pub_source"] = "file_header"
    return rec


# ------------------------------------------------------------------------------ downloading
class OmieDownloader:
    def __init__(self, cfg: Config, store: Store, http: Http | None = None):
        self.cfg = cfg
        self.store = store
        self.http = http or Http(cfg)
        self.ocfg = cfg["omie"]

    def url(self, parent: str, filename: str) -> str:
        return f"{self.ocfg['base_url']}?parents={parent}&filename={filename}"

    def _dir(self, parent: str) -> Path:
        d = self.store.raw_dir / SOURCE / parent
        d.mkdir(parents=True, exist_ok=True)
        return d

    def _missing(self, parent: str) -> dict:
        p = self._dir(parent) / "_missing.json"
        return json.loads(p.read_text()) if p.exists() else {}

    def _save_missing(self, parent: str, missing: dict) -> None:
        (self._dir(parent) / "_missing.json").write_text(json.dumps(missing, indent=0, sort_keys=True))

    def _valid(self, parent: str, content: bytes) -> bool:
        head = content[:200].decode("latin-1", errors="replace").lstrip().upper()
        expected = HEADERS.get(parent)
        if expected:
            return head.startswith(expected)
        return not head.startswith("<")

    def fetch(self, parent: str, filename: str, file_date: date, missing: dict, force: bool = False) -> bool:
        path = self._dir(parent) / filename
        if path.exists() and not force:
            return True
        checked = missing.get(filename)
        if checked and not force and (date.fromisoformat(checked) - file_date).days > 14:
            return False
        r = self.http.get(self.url(parent, filename))
        if r.status_code == 200 and r.content and self._valid(parent, r.content):
            path.write_bytes(r.content)
            missing.pop(filename, None)
            return True
        missing[filename] = date.today().isoformat()
        return False

    def fetch_year_zip(self, parent: str, year: int) -> int:
        marker = self._dir(parent) / f"_zip_{year}.done"
        if marker.exists():
            return 0
        r = self.http.get(self.url(parent, f"{parent}_{year}.zip"))
        if r.status_code != 200 or not r.content.startswith(b"PK"):
            log.info("no yearly zip for %s %s (HTTP %s), falling back to daily files", parent, year, r.status_code)
            return 0
        n = 0
        with zipfile.ZipFile(io.BytesIO(r.content)) as z:
            for name in z.namelist():
                base = Path(name).name
                if FNAME_RE.match(base):
                    (self._dir(parent) / base).write_bytes(z.read(name))
                    n += 1
        marker.write_text(str(n))
        log.info("%s %s: %d files from yearly zip", parent, year, n)
        return n

    def _latest_files(self, parent: str, start: date, end: date) -> dict[tuple[date, int | None], Path]:
        """Highest-version raw file per (day, session) within [start, end]."""
        best: dict[tuple[date, int | None], tuple[int, Path]] = {}
        for p in self._dir(parent).glob(f"{parent}_*"):
            m = FNAME_RE.match(p.name)
            if not m or m["parent"] != parent:
                continue
            d = datetime.strptime(m["date"], "%Y%m%d").date()
            if not (start <= d <= end):
                continue
            key = (d, int(m["session"]) if m["session"] else None)
            v = int(m["version"])
            if key not in best or v > best[key][0]:
                best[key] = (v, p)
        return {k: v[1] for k, v in best.items()}

    def _download_parent(self, parent: str, names: list[tuple[date, str]], start: date, end: date, force: bool):
        if self.ocfg.get("use_yearly_zip", True):
            for year in range(start.year, end.year + 1):
                if year < date.today().year:
                    self.fetch_year_zip(parent, year)
        missing = self._missing(parent)
        have = {(k[0], k[1]) for k in self._latest_files(parent, start, end)}
        todo = 0
        try:
            for d, stem in names:
                session = int(stem[-2:]) if parent == "marginalpibc" else None
                if (d, session) in have and not force:
                    continue
                for v in range(1, self.ocfg["max_version"] + 1):
                    if not self.fetch(parent, f"{stem}.{v}", d, missing, force):
                        break
                    todo += 1
        finally:
            self._save_missing(parent, missing)
        log.info("%s: %d files downloaded", parent, todo)

    def download(self, start: date, end: date, force: bool = False) -> dict:
        end = min(end, date.today() + timedelta(days=1))
        da_names = [(d, f"marginalpdbc_{d:%Y%m%d}") for d in daterange(start, end)]
        self._download_parent("marginalpdbc", da_names, start, end, force)
        ida_start = max(start, date.fromisoformat(self.ocfg["ida_first_day"]))
        ida_names = [(d, f"marginalpibc_{d:%Y%m%d}{s:02d}") for d in daterange(ida_start, end)
                     for s in self.ocfg["ida_sessions"]]
        if ida_start <= end:
            self._download_parent("marginalpibc", ida_names, ida_start, end, force)
        if self.ocfg.get("download_curves"):
            missing = self._missing("curva_pbc_uof")
            try:
                for d in daterange(start, end):
                    self.fetch("curva_pbc_uof", f"curva_pbc_uof_{d:%Y%m%d}.1", d, missing, force)
            finally:
                self._save_missing("curva_pbc_uof", missing)
        return self.parse(start, end)

    def parse(self, start: date, end: date) -> dict:
        """Rebuild clean records from raw files in [start, end]."""
        counts = {}
        fetched = pd.Timestamp.now(tz="UTC")
        recs = []
        for (d, _), path in sorted(self._latest_files("marginalpdbc", start, end).items()):
            rows = parse_marginal(path.read_bytes().decode("latin-1"))
            recs.append(marginal_to_records(rows, "da", self.cfg, d, fetched))
        counts["omie_da"] = self.store.write(pd.concat(recs, ignore_index=True)) if recs else 0
        recs = []
        first = date.fromisoformat(self.ocfg["ida_first_day"])
        for (d, s), path in sorted(self._latest_files("marginalpibc", max(start, first), end).items()):
            if s not in self.ocfg["ida_sessions"]:
                continue
            rows = parse_marginal(path.read_bytes().decode("latin-1"))
            recs.append(marginal_to_records(rows, f"ida{s}", self.cfg, d, fetched))
        counts["omie_ida"] = self.store.write(pd.concat(recs, ignore_index=True)) if recs else 0
        recs = []
        for (d, _), path in sorted(self._latest_files("curva_pbc_uof", start, end).items()):
            try:
                steps, issued = parse_curve(path.read_bytes().decode("latin-1"))
                recs.append(curve_to_volume_records(steps, issued, self.cfg, d))
            except ValueError as e:
                log.warning("curve parse failed for %s: %s", path.name, e)
        if recs:
            counts["omie_da_volume"] = self.store.write(pd.concat(recs, ignore_index=True))
        return counts
