"""Fuel price parsing and publication before delivery."""
from datetime import date

import pandas as pd

from bess.data.fuels import known_before, parse_eua, parse_mibgas

MIB = """Fecha Emisión :31/12/2025 - 21:39;
Trading day;Product;Place of delivery;Area;First Day Delivery;Last Day Delivery;"MIBGAS Daily Price [EUR/MWh]";
02/01/2026;GDAES_D+1;PVB;ES;03/01/2026;03/01/2026;30.5;
02/01/2026;GMAES;PVB;ES;01/02/2026;28/02/2026;29.0;
01/01/2026;GDAES_D+1;PVB;ES;02/01/2026;02/01/2026;29.5;
"""
EUA = """Fecha;EUA;CER;SPREAD (EUA - CER)
02-01-2026;72.84;0.00;"72,84"
05-01-2026;73.81;0.00;"73,81"
"""


def test_parsers_keep_day_ahead_gas_and_eua_closes():
    g = parse_mibgas(MIB)
    assert list(g.index) == [date(2026, 1, 2), date(2026, 1, 3)]
    assert g.loc[date(2026, 1, 3), "value"] == 30.5
    e = parse_eua(EUA)
    assert e.loc[date(2026, 1, 5)] == 73.81


def test_values_are_published_before_the_day_ahead_gate(cfg):
    g, e = parse_mibgas(MIB), parse_eua(EUA)
    rec = known_before(cfg, [date(2026, 1, 4), date(2026, 1, 7)], g, e)
    gate = pd.Timestamp("2026-01-06 11:00", tz="UTC")
    day = rec[rec["delivery_start"] >= pd.Timestamp("2026-01-06 23:00", tz="UTC")]
    assert (day["published_at"] <= gate).all()
    gas = rec[(rec["series"] == "gas_es_prev") & (rec["delivery_start"] < pd.Timestamp("2026-01-04 23:00", tz="UTC"))]
    assert (gas["value"] == 30.5).all() and len(gas) == 96
    eua7 = day[day["series"] == "eua_prev"]["value"].unique()
    assert list(eua7) == [73.81]
