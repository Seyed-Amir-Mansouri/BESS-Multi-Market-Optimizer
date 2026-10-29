"""Time helpers."""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd

QH_MIN = 15
QH = pd.Timedelta(minutes=QH_MIN)

EXTENSIVE_UNITS = {"MWh", "GWh", "kWh"}


def as_date(d: date | str | pd.Timestamp) -> date:
    if isinstance(d, str):
        return date.fromisoformat(d)
    if isinstance(d, pd.Timestamp):
        return d.date()
    if isinstance(d, datetime):
        return d.date()
    return d


def local_midnight_utc(day: date | str, tz: str) -> pd.Timestamp:
    d = as_date(day)
    return pd.Timestamp(d.year, d.month, d.day).tz_localize(tz).tz_convert("UTC")


def day_bounds_utc(day: date | str, tz: str) -> tuple[pd.Timestamp, pd.Timestamp]:
    d = as_date(day)
    return local_midnight_utc(d, tz), local_midnight_utc(d + timedelta(days=1), tz)


def periods_in_day(day: date | str, resolution_min: int, tz: str) -> int:
    start, end = day_bounds_utc(day, tz)
    return int((end - start) / pd.Timedelta(minutes=resolution_min))


def period_to_utc(day: date | str, period: int, resolution_min: int, tz: str) -> pd.Timestamp:
    return local_midnight_utc(day, tz) + (int(period) - 1) * pd.Timedelta(minutes=resolution_min)


def local_time_utc(day: date | str, hhmm: str, tz: str, day_offset: int = 0) -> pd.Timestamp:
    """UTC instant of local clock time `hhmm` on `day + day_offset` ("24:00" = next midnight)."""
    d = as_date(day) + timedelta(days=day_offset)
    hh, mm = (int(x) for x in hhmm.split(":"))
    if hh == 24:
        return local_midnight_utc(d + timedelta(days=1), tz) + pd.Timedelta(minutes=mm)
    naive = pd.Timestamp(d.year, d.month, d.day, hh, mm)
    return naive.tz_localize(tz, ambiguous=True, nonexistent="shift_forward").tz_convert("UTC")


def delivery_day(ts_utc: pd.Series | pd.DatetimeIndex, tz: str) -> pd.Series:
    """Local delivery date of each UTC timestamp."""
    s = pd.Series(pd.to_datetime(ts_utc, utc=True))
    return s.dt.tz_convert(tz).dt.date


def resolution_from_max_period(max_period: int) -> int:
    """Resolution in minutes from the highest period number of a day."""
    return 15 if max_period > 25 else 60


def infer_resolution_min(ts_utc: pd.Series) -> int | None:
    """Most common spacing (minutes) between consecutive distinct timestamps."""
    s = pd.Series(pd.to_datetime(ts_utc, utc=True)).drop_duplicates().sort_values()
    if len(s) < 2:
        return None
    diffs = s.diff().dropna()
    diffs = diffs[diffs > pd.Timedelta(0)]
    if diffs.empty:
        return None
    return int(diffs.mode().iloc[0] / pd.Timedelta(minutes=1))


def per_day_resolution(ts_utc: pd.Series, tz: str, fallback: int = 60) -> pd.Series:
    """Resolution (minutes) of each timestamp, inferred from the spacing within its local day."""
    ts = pd.Series(pd.to_datetime(ts_utc, utc=True))
    days = delivery_day(ts, tz)
    days.index = ts.index
    res = {}
    for day, g in ts.groupby(days):
        diffs = g.drop_duplicates().sort_values().diff().dropna()
        diffs = diffs[diffs > pd.Timedelta(0)]
        res[day] = int(diffs.mode().iloc[0] / pd.Timedelta(minutes=1)) if len(diffs) else fallback
    return days.map(res).astype(int)


def to_quarter_hour(df: pd.DataFrame) -> pd.DataFrame:
    """Bring long-format records to 15-min resolution."""
    if df.empty:
        return df.copy()
    out = []
    res = df["resolution_min"].astype(int)
    same = df[res == QH_MIN]
    if not same.empty:
        out.append(same)
    coarse = df[res > QH_MIN]
    if not coarse.empty:
        n = (coarse["resolution_min"].astype(int) // QH_MIN).clip(lower=1)
        rep = coarse.loc[coarse.index.repeat(n)].copy()
        k = rep.groupby(level=0).cumcount()
        rep["delivery_start"] = rep["delivery_start"] + k * QH
        rep["delivery_end"] = rep["delivery_start"] + QH
        ext = rep["unit"].isin(EXTENSIVE_UNITS)
        n_rep = n.loc[rep.index].to_numpy()
        rep.loc[ext, "value"] = rep.loc[ext, "value"] / n_rep[ext.to_numpy()]
        rep["resolution_min"] = QH_MIN
        out.append(rep)
    fine = df[res < QH_MIN]
    if not fine.empty:
        fine = fine.copy()
        fine["delivery_start"] = fine["delivery_start"].dt.floor(f"{QH_MIN}min")
        keys = [c for c in df.columns if c not in ("value", "delivery_end", "resolution_min", "published_at", "fetched_at")]
        ext = fine["unit"].isin(EXTENSIVE_UNITS)
        parts = []
        for is_ext, grp in fine.groupby(ext):
            agg = grp.groupby(keys, dropna=False).agg(
                value=("value", "sum" if is_ext else "mean"),
                published_at=("published_at", "max"),
                fetched_at=("fetched_at", "max"),
            ).reset_index()
            parts.append(agg)
        fine = pd.concat(parts, ignore_index=True)
        fine["delivery_end"] = fine["delivery_start"] + QH
        fine["resolution_min"] = QH_MIN
        out.append(fine[df.columns])
    result = pd.concat(out, ignore_index=True)
    return result.sort_values(["series", "delivery_start", "published_at"]).reset_index(drop=True)


def daterange(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def chunks(start: date, end: date, days: int):
    """Inclusive [a, b] date chunks of at most `days` days."""
    a = start
    while a <= end:
        b = min(end, a + timedelta(days=days - 1))
        yield a, b
        a = b + timedelta(days=1)
