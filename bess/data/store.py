"""Parquet + DuckDB store.

Layout under data_dir:
    raw/<source>/...                 original downloaded files (re-parseable, makes downloads resumable)
    clean/<dataset>/<YYYY-MM>.parquet long-format time series, partitioned by delivery month (UTC)
    events/<name>.parquet            non-time-series tables (outages, eSIOS catalog, ...)

Every time-series record carries its delivery period AND its publication time:

    source, dataset, series, delivery_start, delivery_end, resolution_min,
    value, unit, published_at, pub_source, fetched_at

A (series, delivery_start) can have several versions with different published_at (e.g. a
forecast re-issued later); `load(as_of=t)` returns, per period, the latest version
published at or before t - the no-lookahead view used by the backtest clock.
"""
from __future__ import annotations

import os
from pathlib import Path

import duckdb
import pandas as pd

from ..timeutils import to_quarter_hour

COLUMNS = ["source", "dataset", "series", "delivery_start", "delivery_end", "resolution_min",
           "value", "unit", "published_at", "pub_source", "fetched_at"]
TS_COLS = ["delivery_start", "delivery_end", "published_at", "fetched_at"]
KEY = ["series", "delivery_start", "published_at"]


def normalize(df: pd.DataFrame) -> pd.DataFrame:
    """Enforce the record schema and dtypes."""
    missing = [c for c in COLUMNS if c not in df.columns]
    if missing:
        raise ValueError(f"records missing columns: {missing}")
    out = df[COLUMNS].copy()
    for c in TS_COLS:
        out[c] = pd.to_datetime(out[c], utc=True).astype("datetime64[ns, UTC]")
    for c in ("source", "dataset", "series", "unit", "pub_source"):
        out[c] = out[c].astype(str)
    out["resolution_min"] = out["resolution_min"].astype("int32")
    out["value"] = pd.to_numeric(out["value"], errors="coerce").astype("float64")
    return out


def _to_us(df: pd.DataFrame) -> pd.DataFrame:
    """Parquet files hold microsecond UTC timestamps (DuckDB's native TIMESTAMPTZ)."""
    out = df.copy()
    for c in TS_COLS:
        out[c] = out[c].astype("datetime64[us, UTC]")
    return out


def _sql_path(p: Path) -> str:
    return p.as_posix().replace("'", "''")


class Store:
    def __init__(self, data_dir: str | os.PathLike):
        self.root = Path(data_dir)
        self.clean_dir = self.root / "clean"
        self.events_dir = self.root / "events"
        self.raw_dir = self.root / "raw"

    # ---------------------------------------------------------------- raw files
    def raw_path(self, source: str, *parts: str) -> Path:
        p = self.raw_dir.joinpath(source, *parts)
        p.parent.mkdir(parents=True, exist_ok=True)
        return p

    # ---------------------------------------------------------------- time series
    def write(self, df: pd.DataFrame) -> int:
        """Upsert records; returns number of rows written. Idempotent."""
        if df is None or df.empty:
            return 0
        df = normalize(df)
        n = 0
        month = df["delivery_start"].dt.strftime("%Y-%m")
        for (dataset, ym), part in df.groupby([df["dataset"], month]):
            path = self.clean_dir / dataset / f"{ym}.parquet"
            path.parent.mkdir(parents=True, exist_ok=True)
            if path.exists():
                part = pd.concat([normalize(pd.read_parquet(path)), part], ignore_index=True)
            part = (part.drop_duplicates(KEY, keep="last")
                        .sort_values(["series", "delivery_start", "published_at"])
                        .reset_index(drop=True))
            tmp = path.with_suffix(".tmp")
            _to_us(part).to_parquet(tmp, index=False)
            os.replace(tmp, path)
            n += len(part)
        return n

    def datasets(self) -> list[str]:
        if not self.clean_dir.exists():
            return []
        return sorted(p.name for p in self.clean_dir.iterdir() if p.is_dir() and any(p.glob("*.parquet")))

    def _scan(self, dataset: str | None = None) -> str | None:
        if dataset:
            files = sorted((self.clean_dir / dataset).glob("*.parquet"))
        else:
            files = sorted(self.clean_dir.glob("*/*.parquet"))
        if not files:
            return None
        lst = ", ".join(f"'{_sql_path(f)}'" for f in files)
        return f"read_parquet([{lst}], union_by_name=true)"

    def connect(self) -> duckdb.DuckDBPyConnection:
        """In-memory DuckDB connection with a `records` view over all clean parquet files."""
        con = duckdb.connect()
        con.execute("SET TimeZone='UTC'")
        scan = self._scan()
        if scan:
            con.execute(f"CREATE VIEW records AS SELECT * FROM {scan}")
        else:
            empty = ", ".join(f"NULL::{t} AS {c}" for c, t in [
                ("source", "VARCHAR"), ("dataset", "VARCHAR"), ("series", "VARCHAR"),
                ("delivery_start", "TIMESTAMPTZ"), ("delivery_end", "TIMESTAMPTZ"),
                ("resolution_min", "INTEGER"), ("value", "DOUBLE"), ("unit", "VARCHAR"),
                ("published_at", "TIMESTAMPTZ"), ("pub_source", "VARCHAR"), ("fetched_at", "TIMESTAMPTZ")])
            con.execute(f"CREATE VIEW records AS SELECT {empty} WHERE false")
        for p in sorted(self.events_dir.glob("*.parquet")) if self.events_dir.exists() else []:
            con.execute(f"CREATE VIEW events_{p.stem} AS SELECT * FROM read_parquet('{_sql_path(p)}')")
        return con

    def export_duckdb(self, path: str | os.PathLike) -> Path:
        """Persistent DuckDB file with views (for interactive SQL, e.g. DBeaver / duckdb CLI)."""
        path = Path(path)
        if path.exists():
            path.unlink()
        con = duckdb.connect(str(path))
        scan = self._scan()
        if scan:
            con.execute(f"CREATE VIEW records AS SELECT * FROM {scan}")
        if self.events_dir.exists():
            for p in sorted(self.events_dir.glob("*.parquet")):
                con.execute(f"CREATE VIEW events_{p.stem} AS SELECT * FROM read_parquet('{_sql_path(p)}')")
        con.close()
        return path

    def load(self, series: str | list[str] | None = None, dataset: str | None = None,
             start: pd.Timestamp | str | None = None, end: pd.Timestamp | str | None = None,
             as_of: pd.Timestamp | str | None = None, latest: bool = True, qh: bool = False) -> pd.DataFrame:
        """Load records with delivery_start in [start, end).

        as_of:  only records published at or before this instant (no-lookahead).
        latest: keep one version per (series, delivery_start): the latest published one.
        qh:     convert to 15-min resolution (hourly expanded, sub-15-min aggregated).
        """
        scan = self._scan(dataset)
        if scan is None:
            return pd.DataFrame(columns=COLUMNS)
        where, params = [], []
        if series is not None:
            series = [series] if isinstance(series, str) else list(series)
            where.append(f"series IN ({', '.join('?' * len(series))})")
            params += series
        if start is not None:
            where.append("delivery_start >= ?")
            params.append(pd.Timestamp(start, tz="UTC") if pd.Timestamp(start).tzinfo is None else pd.Timestamp(start))
        if end is not None:
            where.append("delivery_start < ?")
            params.append(pd.Timestamp(end, tz="UTC") if pd.Timestamp(end).tzinfo is None else pd.Timestamp(end))
        if as_of is not None:
            t = pd.Timestamp(as_of)
            where.append("published_at <= ?")
            params.append(t if t.tzinfo else t.tz_localize("UTC"))
        sql = f"SELECT * FROM {scan}"
        if where:
            sql += " WHERE " + " AND ".join(where)
        if latest:
            sql += " QUALIFY row_number() OVER (PARTITION BY series, delivery_start ORDER BY published_at DESC, fetched_at DESC) = 1"
        sql += " ORDER BY series, delivery_start, published_at"
        con = duckdb.connect()
        con.execute("SET TimeZone='UTC'")
        df = con.execute(sql, [p.to_pydatetime() if isinstance(p, pd.Timestamp) else p for p in params]).df()
        con.close()
        if df.empty:
            return pd.DataFrame(columns=COLUMNS)
        df = normalize(df)
        return to_quarter_hour(df) if qh else df

    def wide(self, series: list[str], start=None, end=None, as_of=None) -> pd.DataFrame:
        """Quarter-hour panel: index delivery_start (UTC), one column per series."""
        df = self.load(series=series, start=start, end=end, as_of=as_of, qh=True)
        if df.empty:
            return pd.DataFrame(columns=series)
        return df.pivot_table(index="delivery_start", columns="series", values="value", aggfunc="last")

    # ---------------------------------------------------------------- event tables
    def write_events(self, name: str, df: pd.DataFrame, key: list[str] | None = None) -> int:
        if df is None or df.empty:
            return 0
        self.events_dir.mkdir(parents=True, exist_ok=True)
        path = self.events_dir / f"{name}.parquet"
        if path.exists():
            df = pd.concat([pd.read_parquet(path), df], ignore_index=True)
        df = df.drop_duplicates(subset=key, keep="last") if key else df.drop_duplicates(keep="last")
        tmp = path.with_suffix(".tmp")
        df.reset_index(drop=True).to_parquet(tmp, index=False)
        os.replace(tmp, path)
        return len(df)

    def read_events(self, name: str) -> pd.DataFrame:
        path = self.events_dir / f"{name}.parquet"
        return pd.read_parquet(path) if path.exists() else pd.DataFrame()
