# Integrated Battery Trading and Revenue Optimization Tool for Day-Ahead, Intraday, and Balancing Markets

Optimizes a battery's participation in the Spanish day-ahead, intraday-auction (IDA1–3) and aFRR capacity markets.

**Phase 1 (Data):** downloaders, cleaning, gap filling, the publication-time timeline, the data quality report and the Phase-1 verifications.

**Phase 2 (Simulator and baselines):** battery model, step bids and clearing, market simulator, perfect foresight and a deterministic optimizer with naive forecasts.

**Phase 3 (Forecasting):** walk-forward LightGBM forecasts for every market input, plugged into the deterministic optimizer.

**Phase 4 (Uncertainty and operation):** scenarios from past forecast errors, a stochastic optimizer with bid curves, metered charge tracking between gates, continuous intraday rebalancing, benchmarks and quick and live tests.

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
bess download all
bess fill
bess calendar
bess timeline
bess verify
bess quality
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

Downloads are resumable and idempotent. Raw files are cached under `data_store/raw/`, so a re-run only fetches what is missing or still recent (the last 7 days). The first full download takes about 1.5 hours, mostly because OMIE has no yearly zips and its daily files are fetched politely, one by one.

## Running Phase 2

```powershell
bess backtest
bess report
```

| Command | What it does |
|---|---|
| `bess backtest --strategy pf` | perfect foresight only |
| `bess backtest --strategy det --start 2025-01-01 --end 2025-03-31` | deterministic strategy, one window |

Results are written to `data_store/backtest/` and the report to `reports/backtest.html`.

## Running Phase 3

```powershell
bess forecast
bess backtest --strategy ml
bess report
```

| Command | What it does |
|---|---|
| `bess forecast --target da --target ida1` | selected targets only |
| `bess forecast --start 2025-01-01 --end 2025-06-30` | one prediction window |

`bess forecast` trains one model per target and gate on past days only, retrains every `forecast.retrain_months` and writes out-of-sample predictions to `data_store/forecasts/predictions.parquet`. Strategy `ml` uses them and falls back to the naive forecast where no model exists yet. The report adds forecast accuracy against the naive forecasts.

## Running Phase 4

```powershell
bess backtest --strategy sto
bess backtest --set quick
bess live
```

| Command | What it does |
|---|---|
| `bess backtest --strategy sto` | stochastic optimizer over scenarios of past forecast errors |
| `bess backtest --set quick` | pf, det, ml and sto on the 8 test weeks in `backtest.sets.quick` |
| `bess backtest --set quick --tag test1 --strategy ml` | same weeks, results saved under a separate name |
| `bess live` | models trained once on all data before the last 14 days, then all strategies on those days |
| `bess live --days 7 --end 2026-10-24` | other window |
| `bess --config bench.yaml backtest --strategy hyb --tag b1` | benchmark with actual values for the inputs listed in `hybrid.perfect` |

The quick set has two one-week windows per season. Its capture rates come within about two points of the full backtest and rank the strategies the same way, so use it to compare ideas and confirm the winner on the full period. Two weeks alone are too noisy to choose between strategies.

### Operation between gates

* **Charge tracking** (`simulator.soc_tracking`): each gate starts from the metered charge at the gate plus the rest of the previous plan.
* **Rebalancing** (`simulator.rebalance`): when activations move the metered charge more than `deadband_mwh` from the latest plan, the battery trades in the continuous intraday market to steer back. Trades are decided `lead_min` before delivery, use only power left free by the market position and the aFRR band, and are priced at the day-ahead price plus or minus `spread_eur_per_mwh`. The spread and lead time are assumptions.
* **Headroom** (`simulator.rebalance.headroom_mw`): power the optimizer keeps free next to the band for rebalancing. Off by default because the lost band income is larger than the penalty it saves.

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

Facts that cannot be read from data (gate times, fees and taxes) are recorded in `bess/config/default.yaml` with a `verified` field: `docs`, `secondary`, `partial` or `no`. They appear in the report as open items until they are confirmed.

### Findings from documents (2026-10-28)

| Item | Finding | Status |
|---|---|---|
| IDA gates and coverage | IDA1 D-1 15:00 and IDA2 D-1 22:00 cover all of D; IDA3 D 10:00 covers 12:00–24:00 of D | docs (SIDC / NEMO Committee) |
| IDA 15-min products | first quarter-hour delivery day 19 Mar 2025 (IDA1/IDA2 held 18 Mar) | data |
| DA 15-min | 1 Oct 2025 | docs |
| DA price limits | SDAC −500 / +4000 €/MWh (reference values; automatic increase mechanism, amended by ACER Decisions 02/2026 and 03/2026) | docs; observed range checked |
| IDA price limits | SIDC ±9999 €/MWh | docs |
| aFRR capacity gate | offers D-1 17:00, assignment D-1 17:30 (P.O. 3.1, CNMC resolution of 19 Jan 2026, BOE-A-2026-1377); 16:00 / 16:30 before | docs |
| aFRR up/down split | first day with a separate upward price: 20 Nov 2024 | data |
| eSIOS ids | 2130 band price up, 634 band price down (single price before 20 Nov 2024), 632/633 band up/down, 682/683 energy price up/down, 680/681 activated energy up/down | data |
| Storage fees and taxes | IVPEE 7% (3.5% in 2024 Q1, 5.25% in 2024 Q2) on net energy income per quarter-hour plus aFRR income, following RDL 8/2023 and DGT V1602-13; no ruling specific to storage. Generation toll 0.5 €/MWh and charging tolls unconfirmed; CNMC is revising Circular 3/2020 | **partial**: IVPEE base for storage and tolls need BOE / CNMC texts |

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
  data/fill.py             gap filling and unavailable market days
  battery.py               battery and fee parameters
  bids.py                  step bids and pay-as-clear clearing
  optim.py                 daily LP for positions and aFRR band
  market_simulator.py      market data, gate sequence, settlement
  features.py              gate features from published data
  forecast.py              naive and walk-forward LightGBM forecasts
  uncertainty.py           whole-day forecast errors of past days
  scenarios.py             error bootstrap and k-medoids reduction
  strategies/              perfect foresight, naive, LightGBM, stochastic, hybrid benchmarks
  backtest.py              chronological backtest
  report.py                HTML reports
  cli.py                   `bess` command
tests/                     pytest: parsers, DST, store, no-lookahead, quality, report, simulator
```

## Known limitations (Phase 1)

* eSIOS, ENTSO-E and Open-Meteo do not stamp historical values with their real publication time, so times come from documented timetables (`rule`) or assumptions (`rule:assumed`). The report lists every series that relies on an assumption.
* eSIOS forecast indicators are overwritten by later issues. Only the last version is downloadable, so they may hold small lookahead. Prefer the ENTSO-E forecasts or the weather forecasts for the DA gate.
* The DA aggregate-curve parser (`omie.download_curves: true`) is experimental. It is not needed for the price-taker V1.
* Only national holidays are included.
