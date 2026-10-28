"""End-to-end over a synthetic store: quality checks, Phase-1 verifications, HTML report."""
from datetime import date, timedelta

import pandas as pd
import pytest

from bess.data.calendar_es import build_calendar, easter, national_holidays
from bess.data.omie import marginal_to_records, parse_marginal
from bess.data.quality import run_checks
from bess.data.verify import run_verifications
from bess.report import build_quality_report
from conftest import marginal_text


@pytest.fixture
def filled(cfg, store):
    """DA hourly in late Sep 2025, 15-min from 1 Oct; one day missing; one price above the limit;
    IDA1/IDA3 15-min with IDA3 covering only 12:00-24:00."""
    recs = []
    for i in range(20):
        d = date(2025, 9, 22) + timedelta(days=i)
        if d == date(2025, 9, 25):
            continue
        res = 15 if d >= date(2025, 10, 1) else 60
        price = (lambda p: 5000.0 if p == 10 else 60.0 + p % 7) if d == date(2025, 10, 5) else (lambda p: 60.0 + p % 7)
        recs.append(marginal_to_records(parse_marginal(marginal_text(d, resolution=res, price=price)), "da", cfg, d))
        recs.append(marginal_to_records(parse_marginal(marginal_text(d, "MARGINALPIBC", resolution=15)), "ida1", cfg, d))
        recs.append(marginal_to_records(parse_marginal(marginal_text(d, "MARGINALPIBC", resolution=15, periods=range(49, 97))),
                                        "ida3", cfg, d))
    store.write(pd.concat(recs, ignore_index=True))
    return store


def test_quality_flags_gap_and_limit(cfg, filled):
    chk = run_checks(filled, cfg).set_index("series")
    da = chk.loc["da_price_es"]
    assert da["missing_days"] == 1 and "2025-09-25" in da["missing_day_ranges"]
    assert da["out_of_limits"] == 1
    assert "2025-10-01: 60->15 min" in da["resolution_changes"]
    assert chk.loc["ida3_price_es"]["completeness_pct"] == pytest.approx(100.0 * 19 / 20, abs=0.01)
    assert chk.loc["ida3_price_es"]["partial_days"] == 0


def test_verifications_detect_switch_and_coverage(cfg, filled):
    v = run_verifications(filled, cfg).set_index("item")
    assert v.loc["qh_switch_da", "status"] == "data" and "2025-10-01" in v.loc["qh_switch_da", "finding"]
    assert v.loc["coverage_ida3", "status"] == "data" and "12:00-24:00" in v.loc["coverage_ida3", "finding"]
    assert v.loc["coverage_ida1", "status"] == "data"
    assert v.loc["limits_da", "status"] == "mismatch"
    assert v.loc["qh_switch_ida2", "status"] == "no data"
    assert v.loc["fee_ivpee_rate", "status"] == "unverified"


def test_coverage_mismatch_is_reported(cfg, filled):
    cfg.raw["markets"]["ida3"]["coverage"]["from"] = "00:00"
    v = run_verifications(filled, cfg).set_index("item")
    assert v.loc["coverage_ida3", "status"] == "mismatch"


def test_report_is_self_contained(cfg, filled, tmp_path):
    out = build_quality_report(filled, cfg, tmp_path / "r" / "q.html")
    html = out.read_text(encoding="utf-8")
    assert "Phase-1 verifications" in html and "da_price_es" in html and "<svg" in html
    assert "<script" not in html and "http://" not in html and "https://" not in html
    assert (tmp_path / "r" / "verifications.csv").exists()


def test_report_on_empty_store(cfg, store, tmp_path):
    html = build_quality_report(store, cfg, tmp_path / "e.html").read_text(encoding="utf-8")
    assert "The store is empty" in html


def test_calendar():
    assert easter(2025) == date(2025, 4, 20) and easter(2026) == date(2026, 4, 5)
    assert national_holidays(2025)[date(2025, 4, 18)] == "Viernes Santo"


def test_calendar_table(cfg):
    cal = build_calendar(cfg, date(2025, 3, 29), date(2025, 3, 31)).set_index("day")
    assert cal.loc[date(2025, 3, 30), "n_quarter_hours"] == 92 and cal.loc[date(2025, 3, 30), "dst_change"]
    assert cal.loc[date(2025, 3, 31), "da_gate_utc"] == pd.Timestamp("2025-03-30 10:00", tz="UTC")
