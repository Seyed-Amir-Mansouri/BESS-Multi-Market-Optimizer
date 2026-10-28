
import pandas as pd
import pytest

from bess.timeutils import (local_time_utc, period_to_utc, periods_in_day, per_day_resolution,
                            resolution_from_max_period, to_quarter_hour)

TZ = "Europe/Madrid"


@pytest.mark.parametrize("day,res,n", [
    ("2025-07-01", 15, 96), ("2025-07-01", 60, 24),
    ("2025-03-30", 15, 92), ("2025-03-30", 60, 23),
    ("2025-10-26", 15, 100), ("2025-10-26", 60, 25),
])
def test_periods_in_day_dst(day, res, n):
    assert periods_in_day(day, res, TZ) == n


def test_period_to_utc_across_fall_back():
    assert period_to_utc("2025-10-26", 1, 60, TZ) == pd.Timestamp("2025-10-25 22:00", tz="UTC")
    assert period_to_utc("2025-10-26", 4, 60, TZ) - period_to_utc("2025-10-26", 3, 60, TZ) == pd.Timedelta(hours=1)
    assert period_to_utc("2025-10-26", 25, 60, TZ) == pd.Timestamp("2025-10-26 22:00", tz="UTC")
    assert period_to_utc("2025-10-26", 100, 15, TZ) == pd.Timestamp("2025-10-26 22:45", tz="UTC")


def test_local_time_utc():
    assert local_time_utc("2025-07-01", "12:00", TZ, -1) == pd.Timestamp("2025-06-30 10:00", tz="UTC")
    assert local_time_utc("2025-01-15", "12:00", TZ, -1) == pd.Timestamp("2025-01-14 11:00", tz="UTC")
    assert local_time_utc("2025-01-15", "24:00", TZ) == pd.Timestamp("2025-01-15 23:00", tz="UTC")


def test_resolution_from_max_period():
    assert [resolution_from_max_period(n) for n in (23, 24, 25, 92, 96, 100)] == [60, 60, 60, 15, 15, 15]


def _rec(starts, res, values, unit):
    s = pd.to_datetime(pd.Series(starts), utc=True)
    return pd.DataFrame({"source": "t", "dataset": "t", "series": "x", "delivery_start": s,
                         "delivery_end": s + pd.Timedelta(minutes=res), "resolution_min": res, "value": values,
                         "unit": unit, "published_at": pd.Timestamp("2025-01-01", tz="UTC"),
                         "pub_source": "rule", "fetched_at": pd.Timestamp("2025-01-01", tz="UTC")})


def test_to_quarter_hour_expands_hourly_prices_equally():
    q = to_quarter_hour(_rec(["2025-01-02 00:00", "2025-01-02 01:00"], 60, [40.0, 80.0], "EUR/MWh"))
    assert len(q) == 8
    assert (q["resolution_min"] == 15).all()
    assert q["value"].tolist() == [40.0] * 4 + [80.0] * 4
    assert q["delivery_start"].iloc[1] == pd.Timestamp("2025-01-02 00:15", tz="UTC")


def test_to_quarter_hour_splits_energy():
    q = to_quarter_hour(_rec(["2025-01-02 00:00"], 60, [10.0], "MWh"))
    assert q["value"].tolist() == [2.5] * 4


def test_to_quarter_hour_aggregates_5min():
    starts = pd.date_range("2025-01-02 00:00", periods=6, freq="5min", tz="UTC")
    q = to_quarter_hour(_rec(starts, 5, [1, 2, 3, 10, 20, 30], "MW"))
    assert q["value"].tolist() == [2.0, 20.0]


def test_per_day_resolution_detects_switch():
    a = pd.date_range("2025-09-30 00:00", periods=24, freq="h", tz=TZ)
    b = pd.date_range("2025-10-01 00:00", periods=96, freq="15min", tz=TZ)
    ts = pd.Series(a.append(b)).dt.tz_convert("UTC")
    r = per_day_resolution(ts, TZ)
    assert set(r.iloc[:24]) == {60} and set(r.iloc[24:]) == {15}
