"""Market data panel, gate sequence, clearing and settlement."""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import date, timedelta

import numpy as np
import pandas as pd

from .battery import Battery, Fees, ivpee_base
from .bids import clear
from .config import Config
from .data.pubtime import market_time
from .data.store import Store
from .optim import STAGES
from .timeutils import QH, as_date, day_bounds_utc, local_midnight_utc

GATES = ["da", "ida1", "afrr_capacity", "ida2", "ida3"]
COLS = ["da", "ida1", "ida2", "ida3", "cap_up", "cap_dn", "share_up", "share_dn", "e_up", "e_dn", "ratio_up"]


class MarketData:
    """Quarter-hour panel of actual values and their publication times."""

    def __init__(self, store: Store, cfg: Config, start: date, end: date):
        self.cfg = cfg
        self.tz = cfg.tz
        names = cfg["simulator"]["series"]
        a, b = local_midnight_utc(start, self.tz), local_midnight_utc(end + timedelta(days=1), self.tz)
        df = store.load(series=list(names.values()), start=a, end=b, qh=True)
        index = pd.date_range(a, b, freq=QH, inclusive="left", name="delivery_start")
        inv = {v: k for k, v in names.items()}
        df["key"] = df["series"].map(inv)
        val = df.pivot_table(index="delivery_start", columns="key", values="value", aggfunc="last").reindex(index)
        pub = df.pivot_table(index="delivery_start", columns="key", values="published_at", aggfunc="max").reindex(index)
        for k in names:
            if k not in val:
                val[k] = np.nan
                pub[k] = pd.NaT
        pub = pub.apply(lambda c: pd.to_datetime(c, utc=True))
        local = index.tz_convert(self.tz)
        days = pd.Series(local.date, index=index)
        split = date.fromisoformat(cfg["regimes"]["afrr_split"]["date"])
        pre = (days < split).to_numpy()
        f, fp = pd.DataFrame(index=index), pd.DataFrame(index=index)
        for k in ("da", "ida1", "ida2", "ida3", "cap_dn", "e_up", "e_dn"):
            f[k], fp[k] = val[k], pub[k]
        f["cap_up"] = np.where(pre, val["cap_dn"], val["cap_up"])
        fp["cap_up"] = pub["cap_dn"].where(pre, pub["cap_up"])
        for side, sys_ in (("up", "sys_up"), ("dn", "sys_dn")):
            f[f"share_{side}"] = (val[f"act_{side}"] / (val[sys_] * QH.total_seconds() / 3600)).clip(0, 1)
            fp[f"share_{side}"] = pub[[f"act_{side}", sys_]].max(axis=1)
        f["ratio_up"] = val["sys_up"] / (val["sys_up"] + val["sys_dn"])
        fp["ratio_up"] = pub[["sys_up", "sys_dn"]].max(axis=1)
        self.values, self.pubs = f[COLS], fp[COLS]
        self.day_of = days
        self.slot = pd.Series((local.hour * 60 + local.minute) // 15, index=index)
        ok = self.values.notna()
        g = days.to_numpy()
        self.day_count = ok.groupby(g).sum()
        self.day_pub = self.pubs.where(ok).groupby(g).max()

    def index(self, day) -> pd.DatetimeIndex:
        a, b = day_bounds_utc(as_date(day), self.tz)
        return self.values.loc[(self.values.index >= a) & (self.values.index < b)].index

    def day(self, day) -> pd.DataFrame:
        return self.values.loc[self.index(day)]

    def last_day(self, cols: list[str]) -> date:
        return min(self.day_count.index[self.day_count[c] > 0].max() for c in cols if (self.day_count[c] > 0).any())


@dataclass
class Context:
    day: date
    gate: str
    as_of: pd.Timestamp
    index: pd.DatetimeIndex
    soc0: float
    fixed: dict = field(default_factory=dict)
    band: tuple | None = None
    own: dict = field(default_factory=dict)


def coverage(cfg: Config, market: str, index: pd.DatetimeIndex) -> np.ndarray:
    """Periods of the delivery day a market covers."""
    cov = cfg["markets"][market]["coverage"]
    local = index.tz_convert(cfg.tz)
    m = local.hour * 60 + local.minute
    lo = sum(int(x) * w for x, w in zip(cov["from"].split(":"), (60, 1)))
    hi = sum(int(x) * w for x, w in zip(cov["to"].split(":"), (60, 1)))
    return np.asarray((m >= lo) & (m < hi))


def hour_groups(cfg: Config, day: date, index: pd.DatetimeIndex) -> dict:
    """Hourly product groups per stage on days before quarter-hour products."""
    reg = cfg["regimes"]
    hours = np.asarray(index.floor("h").asi8)
    out = {}
    for st in STAGES:
        qh_from = date.fromisoformat(reg["da_qh" if st == "da" else "ida_qh"]["date"])
        if day < qh_from:
            out[st] = hours
    if day < date.fromisoformat(reg["ida_qh"]["date"]):
        out["afrr"] = hours
    return out


def settle(act: pd.DataFrame, trades: dict, r_up: np.ndarray, r_dn: np.ndarray, soc0: float, bat: Battery, fees: Fees,
           penalty: float, dt: float = 0.25, tol: float = 1e-3) -> tuple[dict, np.ndarray]:
    """Realised cash flows and state of charge of one day."""
    n = len(act)
    out = {}
    x = np.zeros(n)
    revenue = np.zeros(n)
    traded = 0.0
    for st in STAGES:
        t = np.asarray(trades.get(st, np.zeros(n)), float)
        p = act[st].to_numpy()
        t = np.where(np.isnan(p), 0.0, t)
        out[st] = float(np.nansum(p * t) * dt)
        revenue += np.nan_to_num(p * t)
        traded += float(np.abs(t).sum() * dt)
        x += t
    cap_up, cap_dn = act["cap_up"].to_numpy(), act["cap_dn"].to_numpy()
    r_up = np.where(np.isnan(cap_up), 0.0, r_up)
    r_dn = np.where(np.isnan(cap_dn), 0.0, r_dn)
    out["afrr_capacity"] = float(np.nansum(cap_up * r_up + cap_dn * r_dn) * dt)
    a_up = np.nan_to_num(act["share_up"].to_numpy()) * r_up
    a_dn = np.nan_to_num(act["share_dn"].to_numpy()) * r_dn
    e_up, e_dn = np.nan_to_num(act["e_up"].to_numpy()), np.nan_to_num(act["e_dn"].to_numpy())
    out["afrr_energy"] = float((e_up * a_up - e_dn * a_dn).sum() * dt)
    out["tax"] = -fees.ivpee * ivpee_base(revenue, np.nan_to_num(cap_up), np.nan_to_num(cap_dn), r_up, r_dn,
                                          a_up, a_dn, e_up, e_dn, dt)
    ch = np.maximum(-x, 0) + a_dn
    dis = np.maximum(x, 0) + a_up
    soc = np.empty(n + 1)
    soc[0] = soc0
    miss = np.zeros(n)
    for i in range(n):
        s = soc[i] + (ch[i] * bat.eta_c - dis[i] / bat.eta_d) * dt
        if s < bat.soc_min - tol:
            miss[i] = (bat.soc_min - s) * bat.eta_d
            dis[i] -= miss[i] / dt
            s = bat.soc_min
        elif s > bat.soc_max + tol:
            miss[i] = (s - bat.soc_max) / bat.eta_c
            ch[i] -= miss[i] / dt
            s = bat.soc_max
        soc[i + 1] = min(max(s, bat.soc_min), bat.soc_max)
    da = np.nan_to_num(act["da"].to_numpy())
    out["degradation"] = -float((ch + dis).sum() * dt * bat.degradation)
    out["fees"] = -float(dis.sum() * dt * fees.toll_discharge + ch.sum() * dt * fees.toll_charge
                         + traded * fees.market)
    out["penalty"] = -float((miss * (np.abs(da) + penalty)).sum())
    out["profit"] = sum(out[k] for k in STAGES + ["afrr_capacity", "afrr_energy", "degradation", "fees", "tax", "penalty"])
    out["charge_mwh"] = float(ch.sum() * dt)
    out["discharge_mwh"] = float(dis.sum() * dt)
    out["cycles"] = out["discharge_mwh"] / bat.energy
    out["band_up_mw"] = float(r_up.mean())
    out["band_dn_mw"] = float(r_dn.mean())
    out["shortfall_mwh"] = float(miss.sum())
    out["soc_start"] = float(soc0)
    out["soc_end"] = float(soc[-1])
    return out, soc


class Simulator:
    """Runs the gate sequence of each delivery day for one strategy."""

    def __init__(self, md: MarketData, cfg: Config):
        self.md, self.cfg = md, cfg
        self.bat, self.fees = Battery.from_config(cfg), Fees.from_config(cfg)
        self.penalty = float(cfg["simulator"]["shortfall_penalty_eur_per_mwh"])
        self.tracking = bool(cfg["simulator"].get("soc_tracking", False))
        self.prev = None

    def soc_estimate(self, as_of: pd.Timestamp, fallback: float) -> float:
        """Start-of-day charge from the metered charge at the gate plus the rest of the previous day's plan."""
        if self.prev is None:
            return fallback
        idx, soc, plan = self.prev
        if plan is None or len(plan) != len(soc):
            return fallback
        k = int((idx < as_of).sum())
        return float(np.clip(soc[k] + plan[-1] - plan[k], self.bat.soc_min, self.bat.soc_max))

    def run_day(self, day: date, strategy, soc_plan: float, soc_actual: float):
        idx = self.md.index(day)
        act = self.md.day(day)
        n = len(idx)
        fixed, trades, band = {}, {}, None
        prev = np.zeros(n)
        bids = []
        for gate in GATES:
            as_of = market_time(self.cfg, gate, "gate", day)
            own = {}
            if gate == "ida3" or strategy.perfect:
                soc0 = soc_actual
                if self.tracking and not strategy.perfect:
                    past = (idx < as_of)
                    own = {c: np.where(past, act[c].to_numpy(float), np.nan) for c in ("share_up", "share_dn")}
            else:
                soc0 = self.soc_estimate(as_of, soc_plan) if self.tracking else soc_plan
            ctx = Context(day=day, gate=gate, as_of=as_of, index=idx, soc0=soc0, fixed=dict(fixed), band=band, own=own)
            b = strategy.decide(ctx)
            if not b.empty:
                bids.append(b.assign(day=str(day), gate=gate))
            if gate == "afrr_capacity":
                band = (clear(b, act["cap_up"], "up"), clear(b, act["cap_dn"], "down"))
                continue
            p = act[gate]
            t = (clear(b, p, "sell") - clear(b, p, "buy")) * coverage(self.cfg, gate, idx)
            trades[gate] = t
            prev = prev + t
            fixed[gate] = prev.copy()
        if band is None:
            band = (np.zeros(n), np.zeros(n))
        res, soc = settle(act, trades, band[0], band[1], soc_actual, self.bat, self.fees.on(day), self.penalty)
        plan_soc = getattr(strategy, "plan_soc", None)
        self.prev = (idx, soc, None if plan_soc is None else np.array(plan_soc, float))
        res["day"] = str(day)
        res["markets_held"] = ",".join(m for m in STAGES if act[m].notna().any())
        plan_end = strategy.plan_end if getattr(strategy, "plan_end", None) is not None else self.bat.soc_target
        bids = pd.concat(bids, ignore_index=True) if bids else pd.DataFrame()
        return res, bids, res["soc_end"], plan_end
