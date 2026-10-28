from datetime import date

import pandas as pd

from bess.data.omie import OmieDownloader, marginal_to_records, parse_marginal
from conftest import FakeHttp, FakeResponse, marginal_text, zip_bytes


def test_parse_hourly_da_and_publication_time(cfg):
    d = date(2025, 1, 15)
    rows = parse_marginal(marginal_text(d))
    assert len(rows) == 24 and rows["period"].max() == 24
    rec = marginal_to_records(rows, "da", cfg, d)
    es = rec[rec["series"] == "da_price_es"]
    assert len(es) == 24 and set(es["resolution_min"]) == {60}
    assert es["delivery_start"].min() == pd.Timestamp("2025-01-14 23:00", tz="UTC")
    assert set(es["published_at"]) == {pd.Timestamp("2025-01-14 12:30", tz="UTC")}
    assert set(rec["series"]) == {"da_price_es", "da_price_pt"}


def test_quarter_hour_detection_and_dst(cfg):
    d = date(2025, 10, 26)
    rec = marginal_to_records(parse_marginal(marginal_text(d, resolution=15)), "da", cfg, d)
    es = rec[rec["series"] == "da_price_es"]
    assert len(es) == 100 and set(es["resolution_min"]) == {15}
    assert es["delivery_end"].max() == pd.Timestamp("2025-10-26 23:00", tz="UTC")
    assert es["delivery_start"].is_unique


def test_ida3_partial_day_and_session_publication(cfg):
    d = date(2025, 7, 1)
    text = marginal_text(d, "MARGINALPIBC", resolution=15, periods=range(49, 97))
    rec = marginal_to_records(parse_marginal(text), "ida3", cfg, d)
    es = rec[rec["series"] == "ida3_price_es"]
    assert len(es) == 48
    assert es["delivery_start"].min().tz_convert("Europe/Madrid").strftime("%H:%M") == "12:00"
    assert set(es["published_at"]) == {pd.Timestamp("2025-07-01 08:45", tz="UTC")}
    assert (es["published_at"] < es["delivery_start"]).all()


def test_ida1_published_day_before(cfg):
    d = date(2025, 7, 1)
    rec = marginal_to_records(parse_marginal(marginal_text(d, "MARGINALPIBC", resolution=15)), "ida1", cfg, d)
    assert set(rec["published_at"]) == {pd.Timestamp("2025-06-30 13:45", tz="UTC")}


def test_highest_file_version_wins(cfg, store):
    d = date(2025, 2, 3)
    raw = store.raw_dir / "omie" / "marginalpdbc"
    raw.mkdir(parents=True)
    (raw / "marginalpdbc_20250203.1").write_text(marginal_text(d, price=lambda p: 10.0), encoding="latin-1")
    (raw / "marginalpdbc_20250203.2").write_text(marginal_text(d, price=lambda p: 20.0), encoding="latin-1")
    OmieDownloader(cfg, store, http=FakeHttp()).parse(d, d)
    df = store.load(series="da_price_es")
    assert len(df) == 24 and set(df["value"]) == {20.0}


def test_download_skips_html_error_pages_and_caches(cfg, store):
    d = date(2025, 2, 3)
    good = marginal_text(d).encode("latin-1")
    http = FakeHttp({"marginalpdbc_20250203.1": FakeResponse(200, good),
                     "marginalpdbc_20250203.2": FakeResponse(200, b"<html>error</html>")})
    cfg.raw["omie"]["use_yearly_zip"] = False
    cfg.raw["omie"]["ida_first_day"] = "2030-01-01"
    dl = OmieDownloader(cfg, store, http=http)
    dl.download(d, d)
    files = sorted(p.name for p in (store.raw_dir / "omie" / "marginalpdbc").glob("marginalpdbc_*"))
    assert files == ["marginalpdbc_20250203.1"]
    n_calls = len(http.calls)
    dl.download(d, d)
    assert len(http.calls) == n_calls
    assert len(store.load(series="da_price_es")) == 24


def test_yearly_zip(cfg, store):
    days = [date(2024, 1, 1), date(2024, 1, 2)]
    z = zip_bytes({f"marginalpdbc_{d:%Y%m%d}.1": marginal_text(d) for d in days})
    http = FakeHttp({"marginalpdbc_2024.zip": FakeResponse(200, z)})
    cfg.raw["omie"]["ida_first_day"] = "2030-01-01"
    OmieDownloader(cfg, store, http=http).download(days[0], days[1])
    assert len(store.load(series="da_price_es")) == 48
    assert not any("marginalpdbc_20240101.1" in u for u in http.calls)
