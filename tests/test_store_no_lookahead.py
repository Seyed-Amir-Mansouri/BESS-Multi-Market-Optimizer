"""Store semantics and no-lookahead guarantees."""
from datetime import date, timedelta

import pandas as pd
import pytest

from bess.data.omie import marginal_to_records, parse_marginal
from bess.data.pubtime import market_time
from bess.data.timeline import gate_availability
from bess.data.weather import parse_previous_runs
from conftest import marginal_text

TZ = "Europe/Madrid"


def _rec(series, start, value, published, res=60):
    s = pd.Timestamp(start, tz="UTC")
    return {"source": "t", "dataset": "t", "series": series, "delivery_start": s,
            "delivery_end": s + pd.Timedelta(minutes=res), "resolution_min": res, "value": value, "unit": "EUR/MWh",
            "published_at": pd.Timestamp(published, tz="UTC"), "pub_source": "rule",
            "fetched_at": pd.Timestamp("2026-01-01", tz="UTC")}


def test_write_is_idempotent_and_partitioned(store):
    df = pd.DataFrame([_rec("x", "2025-01-31 23:00", 1.0, "2025-01-30"), _rec("x", "2025-02-01 00:00", 2.0, "2025-01-31")])
    store.write(df)
    store.write(df)
    assert len(store.load()) == 2
    assert sorted(p.name for p in (store.clean_dir / "t").glob("*.parquet")) == ["2025-01.parquet", "2025-02.parquet"]


def test_as_of_hides_future_publications_and_picks_latest_revision(store):
    store.write(pd.DataFrame([
        _rec("fc", "2025-03-02 10:00", 100.0, "2025-03-01 08:00"),
        _rec("fc", "2025-03-02 10:00", 120.0, "2025-03-01 20:00"),
    ]))
    assert store.load(series="fc", as_of="2025-03-01 07:59").empty
    assert store.load(series="fc", as_of="2025-03-01 12:00")["value"].tolist() == [100.0]
    assert store.load(series="fc", as_of="2025-03-02 00:00")["value"].tolist() == [120.0]
    assert store.load(series="fc")["value"].tolist() == [120.0]
    assert len(store.load(series="fc", latest=False)) == 2


@pytest.mark.parametrize("day", [date(2025, 1, 15), date(2025, 3, 30), date(2025, 7, 1), date(2025, 10, 26)])
def test_gate_sequence_is_chronological(cfg, day):
    """Each market's results must be public before the next gate that may use them."""
    t = {m: (market_time(cfg, m, "gate", day), market_time(cfg, m, "results", day)) for m in cfg["markets"]}
    order = ["da", "ida1", "afrr_capacity", "ida2", "ida3"]
    for a, b in zip(order, order[1:]):
        assert t[a][0] < t[a][1] <= t[b][0], f"{a} results must precede the {b} gate"


def test_da_prices_of_d_invisible_at_da_gate_visible_at_ida1(cfg, store):
    days = [date(2025, 7, 1) + timedelta(days=i) for i in range(3)]
    store.write(pd.concat([marginal_to_records(parse_marginal(marginal_text(d, resolution=15)), "da", cfg, d) for d in days]))
    d = days[2]
    d0 = pd.Timestamp(d).tz_localize(TZ).tz_convert("UTC")
    at_da_gate = store.load(series="da_price_es", as_of=market_time(cfg, "da", "gate", d))
    assert at_da_gate["delivery_start"].max() < d0
    at_ida1 = store.load(series="da_price_es", as_of=market_time(cfg, "ida1", "gate", d))
    assert (at_ida1["delivery_start"] >= d0).sum() == 96


def test_gate_availability_horizons(cfg, store):
    days = [date(2025, 7, 1) + timedelta(days=i) for i in range(5)]
    store.write(pd.concat([marginal_to_records(parse_marginal(marginal_text(d, resolution=15)), "da", cfg, d) for d in days]))
    av = gate_availability(store, cfg, days=days[1:])
    h = av[av["series"] == "da_price_es"].set_index("gate")["horizon_h_median"]
    assert h["da"] == pytest.approx(0.0)
    assert h["ida1"] == pytest.approx(24.0)


def test_weather_lead2_covers_d_at_da_gate_lead1_does_not(cfg, store):
    d = date(2025, 7, 2)
    times = pd.date_range(pd.Timestamp(d).tz_localize(TZ), periods=24, freq="h").tz_convert("UTC")
    block = {"hourly": {"time": times.strftime("%Y-%m-%dT%H:%M").tolist(),
                        "wind_speed_100m_previous_day1": [5.0] * 24, "wind_speed_100m_previous_day2": [6.0] * 24},
             "hourly_units": {}}
    store.write(parse_previous_runs([block], ["zaragoza"], cfg))
    gate = market_time(cfg, "da", "gate", d)
    known = store.load(as_of=gate)
    assert (known["series"] == "wx.zaragoza.wind_speed_100m.d2").sum() == 24
    assert (known["series"] == "wx.zaragoza.wind_speed_100m.d1").sum() < 24
