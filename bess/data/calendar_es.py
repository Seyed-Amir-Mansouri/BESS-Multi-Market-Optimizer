"""Calendar features: Spanish national holidays, weekday, DST, day length and gate times.

Only nationwide holidays are included (regional ones vary by autonomous community and
matter little for the national price). Good Friday is national; Easter Monday is regional.
"""
from __future__ import annotations

from datetime import date, timedelta

import pandas as pd

from ..config import Config
from ..timeutils import daterange, periods_in_day
from .pubtime import market_time

FIXED = {(1, 1): "Año Nuevo", (1, 6): "Epifanía", (5, 1): "Fiesta del Trabajo",
         (8, 15): "Asunción", (10, 12): "Fiesta Nacional", (11, 1): "Todos los Santos",
         (12, 6): "Día de la Constitución", (12, 8): "Inmaculada Concepción", (12, 25): "Navidad"}


def easter(year: int) -> date:
    """Gregorian Easter Sunday (anonymous Gregorian algorithm)."""
    a, b, c = year % 19, year // 100, year % 100
    d, e = b // 4, b % 4
    f = (b + 8) // 25
    g = (b - f + 1) // 3
    h = (19 * a + b - d - g + 15) % 30
    i, k = c // 4, c % 4
    l = (32 + 2 * e + 2 * i - h - k) % 7
    m = (a + 11 * h + 22 * l) // 451
    month = (h + l - 7 * m + 114) // 31
    day = ((h + l - 7 * m + 114) % 31) + 1
    return date(year, month, day)


def national_holidays(year: int) -> dict[date, str]:
    hol = {date(year, m, d): name for (m, d), name in FIXED.items()}
    hol[easter(year) - timedelta(days=2)] = "Viernes Santo"
    return hol


def build_calendar(cfg: Config, start: date, end: date) -> pd.DataFrame:
    tz = cfg.tz
    hol = {}
    for y in range(start.year, end.year + 1):
        hol.update(national_holidays(y))
    rows = []
    for d in daterange(start, end):
        n_qh = periods_in_day(d, 15, tz)
        row = {
            "day": d, "weekday": d.weekday(), "is_weekend": d.weekday() >= 5,
            "holiday": hol.get(d), "is_holiday": d in hol,
            "n_quarter_hours": n_qh, "dst_change": n_qh != 96,
        }
        for m in cfg["markets"]:
            row[f"{m}_gate_utc"] = market_time(cfg, m, "gate", d)
            row[f"{m}_results_utc"] = market_time(cfg, m, "results", d)
        rows.append(row)
    return pd.DataFrame(rows)
