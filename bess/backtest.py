"""Chronological backtest loop."""
from __future__ import annotations

import logging
import time
from datetime import date, timedelta

import pandas as pd

from .battery import Battery
from .config import Config
from .data.store import Store
from .market_simulator import MarketData, Simulator
from .strategies.deterministic import Deterministic, LightGBM
from .strategies.hybrid import Hybrid
from .strategies.perfect_foresight import PerfectForesight
from .strategies.stochastic import Stochastic
from .timeutils import daterange

log = logging.getLogger(__name__)

STRATEGIES = {"pf": PerfectForesight, "det": Deterministic, "ml": LightGBM, "sto": Stochastic, "hyb": Hybrid}


def end_cols(cfg: Config) -> list:
    """Series every simulated day needs given the enabled markets."""
    cols = ["da", "ida2"]
    if (cfg["markets"].get("afrr_capacity") or {}).get("enabled", True):
        cols.append("cap_dn")
    rb = cfg["simulator"].get("rebalance") or {}
    if rb.get("enabled") and rb.get("price") == "mic":
        cols.append("mic")
    return cols


def excluded(cfg: Config, day: date) -> bool:
    return any(date.fromisoformat(e["from"]) <= day <= date.fromisoformat(e["to"]) for e in cfg["backtest"].get("exclude") or [])


def regime(cfg: Config, day: date) -> str:
    """Label of the regime period the day falls in."""
    dates = sorted((date.fromisoformat(v["date"]), k) for k, v in cfg["regimes"].items() if k != "ida_start")
    label = "start"
    for d, k in dates:
        if day >= d:
            label = k
    return label


def results_path(cfg: Config, strategy: str, kind: str = "daily", tag: str | None = None):
    p = cfg.path("backtest_dir")
    p.mkdir(parents=True, exist_ok=True)
    return p / (f"{strategy}_{tag}_{kind}.parquet" if tag else f"{strategy}_{kind}.parquet")


def test_set(cfg: Config, name: str) -> list[tuple[date, date]]:
    """Windows of a named test set from the config."""
    return [(date.fromisoformat(w["from"]), date.fromisoformat(w["to"])) for w in cfg["backtest"]["sets"][name]]


def simulate(cfg: Config, store: Store, strategy: str, start: date, end: date | None) -> tuple[pd.DataFrame, list]:
    """Daily results and bids of one continuous window starting at the target charge."""
    md = MarketData(store, cfg, start - timedelta(days=30), end or cfg.data_end())
    cols = end_cols(cfg)
    end = min(end or md.last_day(cols), md.last_day(cols))
    sim = Simulator(md, cfg)
    strat = STRATEGIES[strategy](md, cfg)
    soc_plan = soc_actual = Battery.from_config(cfg).soc_target
    rows, bids = [], []
    t0 = time.time()
    for i, day in enumerate(daterange(start, end)):
        res, b, soc_actual, soc_plan = sim.run_day(day, strat, soc_plan, soc_actual)
        res["excluded"] = excluded(cfg, day)
        res["regime"] = regime(cfg, day)
        rows.append(res)
        if not b.empty:
            bids.append(b)
        if (i + 1) % 30 == 0:
            log.info("%s: %s done, %.0f s", strategy, day, time.time() - t0)
    return pd.DataFrame(rows), bids


def run(cfg: Config, store: Store, strategy: str, start: date | None = None, end: date | None = None,
        windows: list | None = None, tag: str | None = None) -> pd.DataFrame:
    """Simulate every delivery day of the period or of each window and store daily results and bids."""
    windows = windows or [(start or date.fromisoformat(cfg["backtest"]["start"]), end)]
    rows, bids = [], []
    for a, b in windows:
        r, bb = simulate(cfg, store, strategy, a, b)
        rows.append(r)
        bids += bb
    out = pd.concat(rows, ignore_index=True)
    out = out[["day", "regime", "excluded"] + [c for c in out.columns if c not in ("day", "regime", "excluded")]]
    out.to_parquet(results_path(cfg, strategy, tag=tag), index=False)
    if bids:
        b = pd.concat(bids, ignore_index=True)
        b["delivery_start"] = pd.to_datetime(b["delivery_start"], utc=True)
        b.to_parquet(results_path(cfg, strategy, "bids", tag), index=False)
    return out


def compare(results: dict) -> pd.DataFrame:
    """Profit of each strategy and its share of the perfect-foresight profit on common days."""
    days = None
    for r in results.values():
        d = set(r.loc[~r["excluded"], "day"].astype(str))
        days = d if days is None else days & d
    rows = []
    for name, r in results.items():
        k = r[r["day"].astype(str).isin(days)]
        rows.append({"strategy": name, "days": len(k), "profit": k["profit"].sum(), "penalty": k["penalty"].sum(),
                     "shortfall_mwh": k["shortfall_mwh"].sum()})
    out = pd.DataFrame(rows)
    if "pf" in results:
        ref = out.loc[out["strategy"] == "pf", "profit"].iloc[0]
        out["pct_of_pf"] = 100 * out["profit"] / ref if ref else float("nan")
    return out


def load_results(cfg: Config, strategy: str, tag: str | None = None) -> pd.DataFrame:
    p = results_path(cfg, strategy, tag=tag)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()
