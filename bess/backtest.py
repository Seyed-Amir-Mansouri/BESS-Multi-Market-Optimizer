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
from .strategies.deterministic import Deterministic
from .strategies.perfect_foresight import PerfectForesight
from .timeutils import daterange

log = logging.getLogger(__name__)

STRATEGIES = {"pf": PerfectForesight, "det": Deterministic}
END_COLS = ["da", "ida2", "cap_dn"]


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


def results_path(cfg: Config, strategy: str, kind: str = "daily"):
    p = cfg.path("backtest_dir")
    p.mkdir(parents=True, exist_ok=True)
    return p / f"{strategy}_{kind}.parquet"


def run(cfg: Config, store: Store, strategy: str, start: date | None = None, end: date | None = None) -> pd.DataFrame:
    """Simulate every delivery day from start to end and store daily results and bids."""
    start = start or date.fromisoformat(cfg["backtest"]["start"])
    md = MarketData(store, cfg, start - timedelta(days=30), end or cfg.data_end())
    end = min(end or md.last_day(END_COLS), md.last_day(END_COLS))
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
    out = pd.DataFrame(rows)
    out = out[["day", "regime", "excluded"] + [c for c in out.columns if c not in ("day", "regime", "excluded")]]
    out.to_parquet(results_path(cfg, strategy), index=False)
    if bids:
        b = pd.concat(bids, ignore_index=True)
        b["delivery_start"] = pd.to_datetime(b["delivery_start"], utc=True)
        b.to_parquet(results_path(cfg, strategy, "bids"), index=False)
    return out


def load_results(cfg: Config, strategy: str) -> pd.DataFrame:
    p = results_path(cfg, strategy)
    return pd.read_parquet(p) if p.exists() else pd.DataFrame()
