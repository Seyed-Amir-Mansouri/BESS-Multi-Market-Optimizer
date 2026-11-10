"""OMIE continuous intraday trade parsing and gate aggregation."""
import pandas as pd
import pytest

from bess.data.mic import aggregate, parse_trades, to_quarter_hours

TEXT = """OMIE - Mercado de electricidad;Fecha Emision :02/10/2025 - 00:25;;01/10/2025;Transacciones realizadas en XBID;;;;

Fecha;Contrato;Agente compra;Unidad compra;Zona compra;Agente venta;Unidad venta;Zona venta;Precio;Cantidad;Momento casacion;
01/10/2025;20251001 20:00-20251001 20:15;A;A1;Z;B;B1;Z;100,00;1,00;30/09/2025 21:00:00;
01/10/2025;20251001 20:00-20251001 20:15;A;A1;Z;B;B1;Z;200,00;3,00;01/10/2025 09:00:00;
01/10/2025;20251001 20:00-20251001 20:15;A;A1;Z;B;B1;Z;160,00;1,00;01/10/2025 18:30:00;
01/10/2025;20251001 21:00-20251001 22:00;A;A1;Z;B;B1;Z;90,00;4,00;30/09/2025 23:00:00;
01/10/2025;20251001;A;A1;Z;B;B1;Z;90,00;4,00;01/10/2025;
;;;;;;;;;;;
"""


def test_parse_trades_keeps_valid_contracts_in_utc(cfg):
    tr = parse_trades(TEXT, cfg.tz)
    assert len(tr) == 4
    assert tr["start"].iloc[0] == pd.Timestamp("2025-10-01 18:00", tz="UTC")
    assert tr["t"].iloc[0] == pd.Timestamp("2025-09-30 19:00", tz="UTC")


def test_hourly_contracts_split_into_quarter_hours(cfg):
    q = to_quarter_hours(parse_trades(TEXT, cfg.tz))
    hourly = q[q["price"] == 90.0]
    assert len(hourly) == 4 and hourly["mwh"].sum() == pytest.approx(4.0)


def test_gate_prices_use_only_trades_matched_before_the_gate(cfg):
    rec = aggregate(to_quarter_hours(parse_trades(TEXT, cfg.tz)), cfg)
    at = rec[rec["delivery_start"] == pd.Timestamp("2025-10-01 18:00", tz="UTC")].set_index("series")
    assert at.loc["mic_ida2", "value"] == pytest.approx(100.0)
    assert at.loc["mic_ida3", "value"] == pytest.approx((100.0 + 600.0) / 4.0)
    assert at.loc["mic_all", "value"] == pytest.approx((100.0 + 600.0 + 160.0) / 5.0)
    assert at.loc["mic_rebal", "value"] == pytest.approx(160.0)
    assert at.loc["mic_ida2", "published_at"] == pd.Timestamp("2025-09-30 20:00", tz="UTC")
    assert at.loc["mic_rebal", "published_at"] == pd.Timestamp("2025-10-01 17:00", tz="UTC")
