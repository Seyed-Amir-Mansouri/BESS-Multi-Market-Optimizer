from datetime import date

import pandas as pd

from bess.data.entsoe import frame_to_records
from bess.data.esios import name_matches, parse_indicator
from bess.data.pubtime import market_time
from bess.data.weather import parse_previous_runs

TZ = "Europe/Madrid"


def esios_payload(starts, values, geo_id=3, name="Precio mercado spot diario"):
    return {"indicator": {"id": 600, "name": name, "tiempo": [{"name": "Hora", "id": 4}],
                          "values": [{"value": v, "datetime_utc": t.strftime("%Y-%m-%dT%H:%M:%SZ"),
                                      "datetime": t.tz_convert(TZ).isoformat(), "geo_id": geo_id, "geo_name": "España"}
                                     for t, v in zip(starts, values)]}}


def test_esios_resolution_per_day_and_pub_rule(cfg):
    a = pd.date_range("2025-09-30", periods=24, freq="h", tz=TZ).tz_convert("UTC")
    b = pd.date_range("2025-10-01", periods=96, freq="15min", tz=TZ).tz_convert("UTC")
    starts = a.append(b)
    spec = cfg["esios"]["indicators"]["da_price_esios"]
    rec = parse_indicator(esios_payload(starts, range(len(starts))), "da_price_esios", spec, cfg)
    assert len(rec) == 120
    assert rec.groupby(rec["delivery_start"].dt.tz_convert(TZ).dt.date)["resolution_min"].first().tolist() == [60, 15]
    d = date(2025, 10, 1)
    day = rec[rec["delivery_start"].dt.tz_convert(TZ).dt.date == d]
    assert set(day["published_at"]) == {market_time(cfg, "da", "results", d)}


def test_esios_geo_filter(cfg):
    t = pd.date_range("2025-01-01", periods=3, freq="h", tz="UTC")
    p = esios_payload(t, [1, 2, 3])
    p["indicator"]["values"] += esios_payload(t, [9, 9, 9], geo_id=1)["indicator"]["values"]
    rec = parse_indicator(p, "da_price_esios", cfg["esios"]["indicators"]["da_price_esios"], cfg)
    assert rec["value"].tolist() == [1, 2, 3] and set(rec["series"]) == {"da_price_esios"}


def test_name_matching_is_accent_insensitive():
    assert name_matches("Previsión de la producción eólica nacional peninsular", ["eolica"])
    assert name_matches("Precio Reserva de regulación secundaria a subir", ["secundaria", "subir"])
    assert not name_matches("Precio Reserva de regulación secundaria a bajar", ["secundaria", "subir"])


def test_entsoe_frame_to_records_and_forecast_not_known_at_da_gate(cfg):
    idx = pd.date_range("2025-07-01", periods=96, freq="15min", tz=TZ)
    df = pd.DataFrame({"Solar": range(96), "Wind Onshore": range(96)}, index=idx)
    spec = cfg["entsoe"]["datasets"]["wind_solar_forecast_entsoe"]
    rec = frame_to_records(df, "wind_solar_forecast_entsoe", spec, cfg)
    assert set(rec["series"]) == {"wind_solar_forecast_entsoe.solar", "wind_solar_forecast_entsoe.wind_onshore"}
    assert set(rec["resolution_min"]) == {15}
    assert (rec["published_at"] > market_time(cfg, "da", "gate", date(2025, 7, 1))).all()


def test_weather_previous_runs_publication(cfg):
    times = pd.date_range("2025-07-01", periods=24, freq="h", tz="UTC").strftime("%Y-%m-%dT%H:%M").tolist()
    block = {"hourly": {"time": times, "temperature_2m_previous_day1": [20.0] * 24,
                        "temperature_2m_previous_day2": [21.0] * 24},
             "hourly_units": {"time": "iso8601", "temperature_2m_previous_day1": "°C", "temperature_2m_previous_day2": "°C"}}
    rec = parse_previous_runs([block, block], ["madrid", "sevilla"], cfg)
    assert len(rec) == 24 * 2 * 2
    d2 = rec[rec["series"] == "wx.madrid.temperature_2m.d2"]
    assert ((d2["delivery_start"] - d2["published_at"]) == pd.Timedelta(hours=48 - 6)).all()
    assert set(rec["pub_source"]) == {"rule:assumed"}
