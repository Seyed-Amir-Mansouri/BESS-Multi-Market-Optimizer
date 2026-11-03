"""Gap filling, physical limits and unavailable market days."""
from datetime import date, timedelta

import pandas as pd

from bess.data.fill import run_fill
from bess.data.omie import marginal_to_records, parse_marginal
from conftest import marginal_text

DAYS = [date(2025, 11, 20) + timedelta(days=i) for i in range(5)]


def _store_with_gaps(cfg, store):
    recs = []
    for d in DAYS:
        da = marginal_to_records(parse_marginal(marginal_text(d, resolution=15)), "da", cfg, d)
        recs.append(da.assign(source="esios", dataset="esios", series="da_price_esios"))
        if d != DAYS[2]:
            recs.append(da)
        if d != DAYS[3]:
            recs.append(marginal_to_records(parse_marginal(marginal_text(d, "MARGINALPIBC", resolution=15)), "ida1", cfg, d))
    t = pd.date_range("2025-11-20", periods=4, freq="h", tz="UTC")
    recs.append(pd.DataFrame({
        "source": "openmeteo", "dataset": "weather", "series": "wx.burgos.shortwave_radiation.d1",
        "delivery_start": t, "delivery_end": t + pd.Timedelta(hours=1), "resolution_min": 60,
        "value": [100.0, 6310.0, -0.8, 200.0], "unit": "W/m2", "published_at": t - pd.Timedelta(hours=18),
        "pub_source": "rule:assumed", "fetched_at": pd.Timestamp("2026-01-01", tz="UTC")}))
    store.write(pd.concat(recs, ignore_index=True))
    return store


def test_fill_copies_missing_day_from_esios(cfg, store):
    s = _store_with_gaps(cfg, store)
    out = run_fill(s, cfg)
    assert out["fills"][["series", "day", "source"]].values.tolist() == [["da_price_es", str(DAYS[2]), "da_price_esios"]]
    da = s.load(series="da_price_es")
    filled = da[da["pub_source"] == "fill:da_price_esios"]
    assert len(filled) == 96 and (filled["dataset"] == "omie_da").all()
    esios = s.load(series="da_price_esios")
    assert filled["published_at"].min() == esios[esios["delivery_start"].isin(filled["delivery_start"])]["published_at"].min()


def test_physical_limits_and_unavailable(cfg, store):
    s = _store_with_gaps(cfg, store)
    out = run_fill(s, cfg)
    wx = s.load(series="wx.burgos.shortwave_radiation.d1")
    assert wx["value"].tolist() == [100.0, 100.0, 0.0, 200.0] and out["clipped"] == 2
    un = out["unavailable"]
    assert un[un["market"] == "ida1"]["day"].tolist() == [str(DAYS[3])]
    assert len(run_fill(s, cfg)["fills"]) == 1
