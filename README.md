# Integrated Battery Trading and Revenue Optimization Tool for Day-Ahead, Intraday, and Balancing Markets

Optimizes a battery's participation in the Spanish day-ahead, intraday-auction (IDA1–3) and aFRR capacity markets.

**This delivery is Phase 1 (Data):** downloaders, cleaning, the publication-time timeline, the data quality report and the Phase-1 verifications.

---

## Windows setup (PowerShell)

1. Install **Python 3.11 or newer** from python.org and tick "Add python.exe to PATH".
2. Unzip the delivery, open PowerShell in the unzipped folder and run:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
pip install -e ".[dev]"
```

If `Activate.ps1` is blocked, run `Set-ExecutionPolicy -Scope CurrentUser RemoteSigned` once.

3. Set the API keys. `setx` makes them permanent; open a new PowerShell window afterwards:

```powershell
setx ESIOS_API_KEY "your-esios-token"
setx ENTSOE_API_KEY "your-entsoe-token"
```

4. Check that the data sources are reachable and run the tests:

```powershell
bess check-network
pytest
```

On macOS/Linux the steps are the same, except you use `python3 -m venv .venv` and `source .venv/bin/activate`, and set the keys with `export`.

## Running Phase 1

```powershell
bess download all                 # OMIE, eSIOS, ENTSO-E, Open-Meteo from data.start (2024-01-01) to today
bess calendar                     # holidays / DST / gate-time table
bess timeline                     # what is known, up to when, at each gate
bess verify                       # Phase-1 verifications (prints a table)
bess quality                      # reports/data_quality.html + series_checks.csv + verifications.csv
```

Other useful commands:

| Command | What it does |
|---|---|
| `bess download omie --start 2025-01-01 --end 2025-01-31` | one source, one window |
| `bess parse all` | rebuild the clean store from cached raw files (offline) |
| `bess esios-catalog --search "secundaria subir"` | find eSIOS indicator ids |
| `bess status` | what is in the store |
| `bess export-duckdb` | `data_store/bess.duckdb` with views `records` and `events_*` for SQL tools |
| `bess --config my.yaml ...` | your YAML deep-merged over `bess/config/default.yaml` |

Downloads are resumable and idempotent. Raw files are cached under `data_store/raw/`, so a re-run only fetches what is missing or still recent (the last 7 days). The first full download takes a while (roughly 30–60 min), mostly because OMIE daily files are fetched politely, one by one. Completed years come as one zip each.

## Data model

All series share one long-format Parquet schema under `data_store/clean/<dataset>/<YYYY-MM>.parquet`:

| column | meaning |
|---|---|
| `series` | e.g. `da_price_es`, `ida3_price_es`, `afrr_band_price_up`, `wx.zaragoza.wind_speed_100m.d2` |
| `delivery_start`, `delivery_end` | delivery period (UTC) |
| `resolution_min` | 15 or 60 (detected per day, never assumed) |
| `value`, `unit` | |
| `published_at` | when the value became public (UTC) |
| `pub_source` | `rule` (market timetable), `rule:assumed` (undocumented assumption), `file_header` |
| `fetched_at` | when we downloaded it |

**No lookahead.** `Store.load(as_of=t)` returns only records published at or before `t`. If a value has several versions (e.g. a re-issued forecast), you get the latest one published by `t`. `Store.load(qh=True)` brings everything to 15-min resolution: hourly prices and MW values are repeated, MWh values are split in four, and sub-15-min data is averaged.

Publication times are deliberately conservative (late). For example, DA results are taken as public at D-1 13:30 even though SDAC publishes around 12:45–13:00.

## Phase-1 verifications

`bess verify` checks these from the data, and the results also appear in section 1 of the quality report:

* **15-min switch dates** for DA and each IDA, detected from the number of periods in each OMIE file.
* **IDA coverage:** which delivery periods each session actually contains.
* **aFRR capacity resolution**, and the date of the separate upward/downward price.
* **eSIOS indicator identity:** each configured id's real name is checked against expected keywords.
* **Price limits:** observed min/max compared with the configured limits.
* **Cross-source agreement:** OMIE DA price vs eSIOS and vs ENTSO-E.

Facts that cannot be read from data (gate times, fees and taxes) are recorded in `bess/config/default.yaml` with a `verified` field: `docs`, `secondary` or `no`. They appear in the report as open items until they are confirmed.

### Findings from documents (2026-10-28)

| Item | Finding | Status |
|---|---|---|
| IDA gates and coverage | IDA1 D-1 15:00 and IDA2 D-1 22:00 cover all of D; IDA3 D 10:00 covers 12:00–24:00 of D | docs (SIDC / NEMO Committee) |
| IDA 15-min products | first quarter-hour IDA held 18 Mar 2025 | secondary; confirm with `bess verify` |
| DA 15-min | 1 Oct 2025 | docs |
| DA price limits | SDAC −500 / +4000 €/MWh (reference values; automatic increase mechanism, amended by ACER Decisions 02/2026 and 03/2026) | docs; observed range checked |
| IDA price limits | SIDC ±9999 €/MWh | docs |
| aFRR capacity gate | 16:00 CET D-1 | secondary; confirm in REE P.O. 7.2 |
| aFRR up/down split | REE: two independent upward/downward reserve auctions (terms approved Apr 2024) | the exact first day is detected from data |
| eSIOS ids | 632/633 band up/down, 634 band price, 682/683 aFRR energy price up/down (REE API docs); 2130/2131 separate band prices are candidates only | name-checked by `bess verify` |
| Storage fees and taxes | generation toll 0.5 €/MWh, IVPEE 7% and charging tolls all unconfirmed for standalone BESS; CNMC is revising Circular 3/2020 to remove double charging of storage | **open**: needs BOE / CNMC texts |

## Project layout

```
bess/
  config/default.yaml      markets, gate times, regimes, fees, sources (all configurable)
  config.py                YAML loading and merging; secrets from env vars only
  timeutils.py             DST-safe period <-> UTC conversion, 15-min conversion
  data/omie.py             OMIE DA/IDA marginal prices (+ optional DA aggregate curves)
  data/esios.py            REE eSIOS indicators + catalog
  data/entsoe.py           ENTSO-E load, generation, forecasts, outages
  data/weather.py          Open-Meteo archived forecasts (Previous Runs API)
  data/calendar_es.py      national holidays, DST, gate times per day
  data/pubtime.py          publication-time rules
  data/store.py            Parquet + DuckDB store, as-of queries
  data/timeline.py         what is known at each gate
  data/quality.py          completeness, gaps, limits, outliers, monthly stats, cross-checks
  data/verify.py           Phase-1 verifications
  report.py                self-contained HTML report
  cli.py                   `bess` command
tests/                     pytest: parsers, DST, store, no-lookahead, quality, report
```

## Known limitations (Phase 1)

* eSIOS, ENTSO-E and Open-Meteo do not stamp historical values with their real publication time, so times come from documented timetables (`rule`) or assumptions (`rule:assumed`). The report lists every series that relies on an assumption.
* eSIOS forecast indicators are overwritten by later issues. Only the last version is downloadable, so they may hold small lookahead. Prefer the ENTSO-E forecasts or the weather forecasts for the DA gate.
* The DA aggregate-curve parser (`omie.download_curves: true`) is experimental. It is not needed for the price-taker V1.
* Only national holidays are included.
