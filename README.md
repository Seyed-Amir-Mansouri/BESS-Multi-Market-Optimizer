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

### Market participation

Each market has its own switch under `participation` in the config; `false` keeps the battery out of it:

| Switch | Market | Default | Last change in reality | Decided in the model |
|---|---|---|---|---|
| `da` | Day-ahead auction (OMIE) | on | 12:00 D-1 | 12:00 D-1 |
| `ida1` | Intraday auction 1 | on | 15:00 D-1 | 15:00 D-1 |
| `ida2` | Intraday auction 2 | on | 22:00 D-1 | 22:00 D-1 |
| `ida3` | Intraday auction 3 (12:00 to 24:00) | on | 10:00 D | 10:00 D |
| `afrr` | aFRR band and its energy offers (REE) | off | band 17:00 D-1, energy about 25 min before delivery | band 17:00 D-1, energy offer from the charge metered 30 min before |
| `continuous` | Continuous intraday trades that steer the charge back to plan | on | 60 min before delivery | 60 min before delivery |
| `imbalance` | Deliberate imbalance settled at the REE imbalance price | off | no gate | start of each quarter-hour |

The switches are linked so that only real combinations are possible: the aFRR band and its energy offers are one switch because REE requires both, and deliberate imbalance only runs with `continuous` on so that the charge can be restored. Perfect foresight never rebalances or trades imbalance.

**Deliberate imbalance** (`simulator.imbalance`): at the start of each quarter-hour the battery discharges extra when a walk-forward model expects being long to beat the latest auction price by more than `threshold_eur_per_mwh`, or charges extra when being short is expected to pay. The model uses the previous quarter's aFRR activation and imbalance prices from about 75 minutes back, and is retrained every month on earlier data only. The deviation is limited to the power left free by the auctions and the band, and keeps one hour of band energy in the battery. Public data has no aFRR or imbalance values finer than 15 minutes, so decisions inside the quarter-hour cannot be tested.

Results on the quick weeks up to 8 August 2026 (the last day with every input complete), ml with aFRR on:

| Signal | Threshold | Profit | Change | Deliberate energy |
|---|---|---|---|---|
| none | | €99,303 | | 0 |
| forecast | 30 €/MWh | €98,563 | -€740 | 25 MWh |
| forecast | 50 €/MWh | €99,315 | +€12 | 3.5 MWh |
| forecast | 80 €/MWh | €99,363 | +€60 | 0 |
| actual direction known | 30 €/MWh | €99,605 | +€302 | 52 MWh |
| actual direction known | 50 €/MWh | €100,231 | +€928 | 23 MWh |
| actual direction known | 80 €/MWh | €100,156 | +€853 | 8 MWh |

Even knowing the direction in advance adds under 1%: in about 42% of quarter-hours neither direction beats the auction price, and restoring the charge costs the round-trip losses and wear. The forecast captures none of it, so `imbalance` stays off.

All switches on, eval weeks up to 8 August 2026 (9 weeks), rebalancing at real continuous trade prices: ml €104,327 (53.3% of perfect foresight €195,789), det €96,508, uplift 8.1% (90% range +€2.5k to +€13.8k), ml ahead in 7 of 9 weeks, shortfall cost 0.5%. With `imbalance` off ml earns the same. Priced at the day-ahead price plus or minus €10 instead, ml earns €103,483: the real trades make rebalancing cost €8.8/MWh instead of €12.2/MWh over 255 MWh.

### Operation between gates

* **Charge tracking** (`simulator.soc_tracking`): each gate starts from the metered charge at the gate plus the rest of the previous plan.
* **Rebalancing** (`simulator.rebalance`): when activations move the metered charge more than `deadband_mwh` from the latest plan, the battery trades in the continuous intraday market to steer back. Trades are decided `lead_min` before delivery, use only power left free by the market position and the aFRR band, and are priced (`price: mic`, default) at the volume-weighted OMIE continuous trades of the same quarter-hour executed 60 to 120 minutes before delivery, worse by `mic_slippage_eur_per_mwh` (2 €/MWh, an assumption for crossing the bid-ask spread). Where no trade exists, or with `price: spread`, the day-ahead price plus or minus `spread_eur_per_mwh` is used. Real trade prices exist for almost every quarter-hour up to 15 August 2026; before March 2025 the continuous products were hourly.
* **Headroom** (`simulator.rebalance.headroom_mw`, default 0): buying power the optimizer can keep free next to the down band, so rebalancing can still recharge when the band covers the full power. With the 2 MWh energy offer guard it is not needed. `headroom_sell_mw` (default 0) does the same for selling; shortfalls come from an empty battery, so selling headroom costs profit without lowering them.
* **Margin** (`simulator.soc_margin_mwh`, 2 MWh): charge kept away from empty and full while aFRR is on.
* **Shortfall settlement** (`simulator.shortfall_price: imbalance`): energy the battery cannot deliver is bought at the eSIOS short imbalance price (687), energy it cannot absorb is sold at the long imbalance price (686), and undelivered aFRR activation energy is not paid. `fixed` charges the day-ahead price plus `shortfall_penalty_eur_per_mwh` instead.
* **Energy offer guard** (`simulator.energy_offer`, 2 MWh, 2 quarter-hours): aFRR energy offers are submitted about 25 minutes before delivery. When the charge metered two quarter-hours earlier is within `guard_mwh` of empty, the upward energy offer is priced out of merit for that quarter-hour, and likewise the downward offer near full. The band is still offered and paid; only the activation energy of that quarter-hour is lost. This assumes REE accepts high-priced energy offers for awarded band. `guard_mwh: null` keeps every offer in merit.
* **aFRR activation** is taken pro rata: the unit's share of its band equals Spain's activated aFRR energy (680/681) over Spain's awarded band (632/633). A unit activated 1.5 times its pro-rata share misses about six times more energy, so check this against the unit's own history.

Headroom, margin and the energy offer guard apply only when `participation.afrr` is on.

### Eval results with aFRR

Eval set (83 days, one week per month, Nov 2025 to Oct 2026), defaults with aFRR on:

| Strategy | Profit | Share of perfect foresight | Shortfall cost |
|---|---|---|---|
| Perfect foresight | €293,950 | 100% | 0% |
| ml | €158,885 | 54.1% | 0.3% |
| det | €149,674 | 50.9% | 0.3% |

ml earns 6.2% more than det (90% range +€3.4k to +€15.6k over weekly resamples) and is ahead in 10 of 12 weeks. Undelivered energy is 4.7 MWh. Earlier settings on the same eval: no guard and 0.5 MW headroom €156,827 (29.4 MWh, shortfall cost 1.2%); 1 MWh guard and 0.5 MW headroom €156,077 (6.1 MWh, 0.6%); no guard and no headroom €161,328 but shortfall cost 2.8% of profit.

Tested and not adopted: deliberate imbalance positions from a forecast of the system direction (loses money once the energy is bought back, as about 70% of quarters settle at a dual price), and rolling intrinsic trading on the continuous intraday tape after IDA3 (under €1k gain on the quick set with higher shortfall cost).

Where the gap to perfect foresight sits: on the quick set ml makes €100.3k; with the true IDA2 and IDA3 prices given only at their own gates it makes €100.5k, and with them given from the aFRR gate it makes €136.3k. The gap is decided when the aFRR band is sized at 17:00 D-1, before the information that moves IDA2 and IDA3 is published. Also tested and not adopted:

- Intraday moves forecast from the latest auction price plus the ENTSO-E intraday wind and solar forecast (published overnight, before the IDA3 gate). Direction accuracy rises from 49.5% to 55.3% at IDA3, but eval profit only from €156.8k to €157.1k.
- Band prices cut by recent intraday price volatility at the aFRR gate (€82k to €97k on quick).
- Continuous intraday trades before 17:00 D-1 (no information beyond IDA1) and newly published ENTSO-E unavailability between gates (correlation about 0.03 with the IDA moves).
- ECMWF IFS forecast runs (Open-Meteo single runs, 9 locations, each run used from 7 h after its start) as gate-timed IDA1 to IDA3 features: IDA2 direction unchanged at 54.9%, quick €97.9k against €100.3k.
- Two-stage Monte Carlo band sizing at 17:00 D-1 (€97.4k and €95.2k on quick) and deliverable scenario bid curves at IDA2 and IDA3 (€95.9k).
- A daily policy learned walk-forward from 693 days of backtests that picks the band price factor (0.8, 1.0, 1.25) and the trust in the IDA forecasts before the LP runs (fitted Q with LightGBM and a recent-winner bandit): eval €155.5k to €156.5k against €156.1k. Choosing the best action each day with hindsight gives only €162.2k, so the gap to perfect foresight is missing price information, not the optimisation.
- Rebalancing closer than 60 minutes: Spain keeps the 60-minute continuous intraday gate closure until 2029 and the OMIE trades show none later.

ML combined with the optimizer, judged on eval against ml (€158,885, 54.1% of perfect foresight); none is adopted:

| Approach | Eval profit | vs ml | Weeks ahead | Shortfall | Solve per gate |
|---|---|---|---|---|---|
| ml (LP on LightGBM forecasts) | €158,885 | | | 4.7 MWh | 0.5 s median |
| Fault model correcting day-ahead-gate intraday forecasts | €157,928 | -€957 | 3/12 | 4.5 MWh | 0.5 s |
| Monte Carlo, 5 of 40 scenarios picked by a learned relevance model | €159,896 | +€1,011 (90% -€1.1k to +€3.2k) | 7/12 | 9.7 MWh | 10 s |
| Two-stage stochastic LP, 10 k-medoids scenarios | €154,185 | -€4,700 | 4/12 | 83.8 MWh | |

- Fault-based learning: a walk-forward LightGBM learns where the intraday forecasts point the wrong way (AUC 0.50 to 0.64) and corrects or withholds those moves before the LP. It helps only at the day-ahead gate on quick (+0.2%) and loses on eval.
- Spike risk: a classifier of activation share above 40% from what is known at 17:00 D-1 reaches AUC 0.63; keeping rebalancing headroom only in the riskiest 30% of quarter-hours earned less than dropping headroom and raising the energy offer guard to 2 MWh, which became the default.
- Learned band sizing: a model of the perfect-foresight band from 17:00 D-1 features, blended into or capping the LP band, cuts shortfall by up to 80% but costs 3% to 9% of profit on quick.
- Monte Carlo with a relevance model: 52,960 bootstrapped scenarios over 662 days labelled by their distance to the realised IDA2 and IDA3 prices; the model ranks them with Spearman 0.69 out of sample. Labelling by the LP plan's realised profit takes about 8 s per scenario and was only used to check the labels. The gain over ml is inside the noise and each gate takes about 20 times longer.

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
