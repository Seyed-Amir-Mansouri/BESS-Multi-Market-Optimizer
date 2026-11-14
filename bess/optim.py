"""Daily LP for energy positions and aFRR band."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, milp
from scipy.sparse import coo_matrix

from .battery import Battery, Fees

STAGES = ["da", "ida1", "ida2", "ida3"]
BAND_FROM = 1


class LP:
    """Sparse LP builder solved with HiGHS."""

    def __init__(self):
        self.lb, self.ub, self.c, self.integ = [], [], [], []
        self.r, self.k, self.v, self.rlb, self.rub = [], [], [], [], []
        self.extra = []
        self.n = 0
        self.m = 0

    def var(self, n: int, lb=0.0, ub=np.inf, cost=0.0, integer: bool = False) -> np.ndarray:
        idx = np.arange(self.n, self.n + n)
        self.lb.append(np.broadcast_to(np.asarray(lb, float), n))
        self.ub.append(np.broadcast_to(np.asarray(ub, float), n))
        self.c.append(np.broadcast_to(np.asarray(cost, float), n))
        self.integ.append(np.full(n, 1 if integer else 0))
        self.n += n
        return idx

    def add_cost(self, idx: np.ndarray, cost) -> None:
        self.extra.append((idx, np.broadcast_to(np.asarray(cost, float), len(idx))))

    def con(self, terms: list, lb, ub) -> None:
        m = max(np.size(i) for _, i in terms)
        rows = self.m + np.arange(m)
        for coef, idx in terms:
            idx = np.broadcast_to(np.asarray(idx), m)
            coef = np.broadcast_to(np.asarray(coef, float), m)
            self.r.append(rows)
            self.k.append(idx)
            self.v.append(coef)
        self.rlb.append(np.broadcast_to(np.asarray(lb, float), m))
        self.rub.append(np.broadcast_to(np.asarray(ub, float), m))
        self.m += m

    def row(self, terms: list, lb: float, ub: float) -> None:
        """One constraint summing all terms."""
        for coef, idx in terms:
            idx = np.atleast_1d(np.asarray(idx))
            self.r.append(np.full(len(idx), self.m))
            self.k.append(idx)
            self.v.append(np.broadcast_to(np.asarray(coef, float), len(idx)))
        self.rlb.append(np.array([lb], float))
        self.rub.append(np.array([ub], float))
        self.m += 1

    def solve(self, time_limit: float = 30.0, gap: float = 1e-4) -> tuple[np.ndarray, float]:
        c = np.concatenate(self.c)
        for idx, cost in self.extra:
            np.add.at(c, idx, cost)
        a = coo_matrix((np.concatenate(self.v), (np.concatenate(self.r), np.concatenate(self.k))), shape=(self.m, self.n))
        res = milp(c, integrality=np.concatenate(self.integ), bounds=Bounds(np.concatenate(self.lb), np.concatenate(self.ub)),
                   constraints=LinearConstraint(a.tocsr(), np.concatenate(self.rlb), np.concatenate(self.rub)),
                   options={"time_limit": time_limit, "mip_rel_gap": gap, "disp": False})
        if res.x is None:
            raise RuntimeError(f"MILP failed: {res.message}")
        return res.x, float(res.fun)


@dataclass
class DayInput:
    """Prices, availability and commitments for one delivery day."""

    n: int
    price: dict
    open: dict
    groups: dict
    cap_up: np.ndarray
    cap_dn: np.ndarray
    share_up: np.ndarray
    share_dn: np.ndarray
    e_up: np.ndarray
    e_dn: np.ndarray
    soc0: float
    fixed: dict = field(default_factory=dict)
    band_fixed: tuple | None = None
    ratio_up: np.ndarray | None = None
    end_price: float | None = None
    headroom: float = 0.0
    cycle_reserve: float = 0.0
    cycle_activation: float = 1.0
    soc_margin: np.ndarray | None = None
    interim_cost: float = 0.0
    interim_exclusive: bool = False
    dt: float = 0.25


@dataclass
class Plan:
    position: dict
    trade: dict
    r_up: np.ndarray
    r_dn: np.ndarray
    soc: np.ndarray
    objective: float
    scen_price: np.ndarray | None = None
    scen_trade: np.ndarray | None = None


def _clean(a) -> np.ndarray:
    return np.nan_to_num(np.asarray(a, float), nan=0.0)


def _first_of_group(g) -> tuple[np.ndarray, np.ndarray]:
    g = np.asarray(g)
    first = np.array([np.flatnonzero(g == v)[0] for v in g])
    return first, np.flatnonzero(first != np.arange(len(g)))


def _band(lp: LP, inps: list, probs: np.ndarray, P: float) -> tuple[np.ndarray, np.ndarray]:
    """Band variables shared by all scenarios."""
    inp = inps[0]
    n = inp.n
    if inp.band_fixed is not None:
        return lp.var(n, inp.band_fixed[0], inp.band_fixed[0]), lp.var(n, inp.band_fixed[1], inp.band_fixed[1])
    band_ok = ~(np.isnan(np.asarray(inp.cap_up, float)) | np.isnan(np.asarray(inp.cap_dn, float)))
    ub = np.where(band_ok, P, 0.0)
    r_up, r_dn = lp.var(n, 0.0, ub), lp.var(n, 0.0, ub)
    if inp.ratio_up is not None:
        q = np.clip(sum(w * _clean(i.ratio_up) for w, i in zip(probs, inps)), 0, 1)
        if "afrr" in inp.groups:
            q = pd.Series(q).groupby(np.asarray(inp.groups["afrr"])).transform("mean").to_numpy()
        lp.con([(1 - q, r_up), (-q, r_dn)], 0.0, 0.0)
    g = inp.groups.get("afrr")
    if g is not None:
        first, rows = _first_of_group(g)
        if len(rows):
            lp.con([(1.0, r_up[rows]), (-1.0, r_up[first[rows]])], 0.0, 0.0)
            lp.con([(1.0, r_dn[rows]), (-1.0, r_dn[first[rows]])], 0.0, 0.0)
    return r_up, r_dn


def _scenario(lp: LP, inp: DayInput, bat: Battery, fees: Fees, w: float, r_up, r_dn, soft_penalty: float,
              shortfall: float | None, light: bool = False, shared: dict | None = None) -> tuple[dict, dict, dict]:
    """Stage blocks of one scenario with costs weighted by its probability."""
    n, dt, P = inp.n, inp.dt, bat.power
    a_up, a_dn = np.clip(_clean(inp.share_up), 0, 1), np.clip(_clean(inp.share_dn), 0, 1)
    cap_up, cap_dn = _clean(inp.cap_up), _clean(inp.cap_dn)
    e_up, e_dn, tax = _clean(inp.e_up), _clean(inp.e_dn), fees.ivpee
    band_value = ((cap_up - tax * np.maximum(cap_up, 0) + a_up * (e_up - tax * np.maximum(e_up, 0))) * dt,
                  (cap_dn - tax * np.maximum(cap_dn, 0) - a_dn * (e_dn + tax * np.maximum(-e_dn, 0))) * dt)
    lp.add_cost(r_up, -w * band_value[0])
    lp.add_cost(r_dn, -w * band_value[1])
    if shortfall is None:
        final_slack = np.full(n + 1, soft_penalty)
    else:
        da = np.abs(_clean(inp.price["da"]))
        final_slack = shortfall + np.r_[da[:1], da]

    prev = None
    prev_val = np.zeros(n)
    rev_terms, rev_const = [], np.zeros(n)
    pos, trades, socs = {}, {}, {}
    last = len(STAGES) - 1
    current = next((st for st in STAGES if st not in inp.fixed), None)
    for k, st in enumerate(STAGES):
        if st in inp.fixed:
            x = lp.var(n, inp.fixed[st], inp.fixed[st])
            fx = np.asarray(inp.fixed[st], float)
            rev_const += _clean(inp.price[st]) * (fx - prev_val) * dt
            prev_val = fx
            pos[st] = x
            prev = x
            continue
        price = np.asarray(inp.price[st], float)
        open_ = np.asarray(inp.open[st], bool) & ~np.isnan(price)
        p = _clean(price)
        tub = np.where(open_, 2 * P, 0.0)
        if shared is not None and st != current and st in shared:
            s, b = shared[st]
            lp.add_cost(s, w * (-p * dt + fees.market * dt))
            lp.add_cost(b, w * (p * dt + fees.market * dt))
        else:
            s = lp.var(n, 0.0, tub, cost=w * (-p * dt + fees.market * dt))
            b = lp.var(n, 0.0, tub, cost=w * (p * dt + fees.market * dt))
            if shared is not None and st != current:
                shared[st] = (s, b)
        rev_terms += [(-p * dt, s), (p * dt, b)]
        x = lp.var(n, -P, P)
        terms = [(1.0, x), (-1.0, s), (1.0, b)]
        if prev is not None:
            terms.append((-1.0, prev))
        lp.con(terms, 0.0, 0.0)
        g = inp.groups.get(st)
        if g is not None:
            first, rows = _first_of_group(g)
            if len(rows):
                lp.con([(1.0, s[rows]), (-1.0, b[rows]), (-1.0, s[first[rows]]), (1.0, b[first[rows]])], 0.0, 0.0)
        final = k == last
        if light and not final and st != current:
            trades[st] = (s, b)
            pos[st] = x
            prev = x
            continue
        c = lp.var(n, 0.0, P, cost=w * ((bat.degradation + fees.toll_charge) * dt if final else inp.interim_cost * dt))
        d = lp.var(n, 0.0, P, cost=w * ((bat.degradation + fees.toll_discharge) * dt if final else inp.interim_cost * dt))
        lp.con([(1.0, d), (-1.0, c), (-1.0, x)], 0.0, 0.0)
        if inp.interim_exclusive and not final:
            z = lp.var(n, 0.0, 1.0, integer=True)
            lp.con([(1.0, c), (-P, z)], -np.inf, 0.0)
            lp.con([(1.0, d), (P, z)], -np.inf, P)
        soc = lp.var(n + 1, -np.inf, np.inf)
        slack = final_slack if final else np.full(n + 1, soft_penalty)
        lo = lp.var(n + 1, 0.0, np.inf, cost=w * slack)
        hi = lp.var(n + 1, 0.0, np.inf, cost=w * slack)
        lp.con([(1.0, soc[:1])], inp.soc0, inp.soc0)
        band = k >= BAND_FROM
        dyn = [(1.0, soc[1:]), (-1.0, soc[:-1]), (-bat.eta_c * dt, c), (dt / bat.eta_d, d)]
        if band:
            dyn += [(-bat.eta_c * dt * a_dn, r_dn), (dt / bat.eta_d * a_up, r_up)]
        lp.con(dyn, 0.0, 0.0)
        for sl in (slice(0, n), slice(1, n + 1)):
            lower = [(1.0, soc[sl]), (1.0, lo[sl])]
            upper = [(1.0, soc[sl]), (-1.0, hi[sl])]
            if band:
                lower.append((-bat.reserve_h / bat.eta_d, r_up))
                upper.append((bat.reserve_h * bat.eta_c, r_dn))
            lp.con(lower, bat.soc_min, np.inf)
            lp.con(upper, -np.inf, bat.soc_max)
            if inp.soc_margin is not None:
                m = inp.soc_margin[sl]
                lp.con([(1.0, soc[sl]), (1.0, lo[sl])], bat.soc_min + m, np.inf)
                lp.con([(1.0, soc[sl]), (-1.0, hi[sl])], -np.inf, bat.soc_max - m)
        if band:
            hu = lp.var(n, 0.0, np.inf, cost=w * soft_penalty)
            hd = lp.var(n, 0.0, np.inf, cost=w * soft_penalty)
            lp.con([(1.0, x), (1.0, r_up), (-1.0, hu)], -np.inf, P - inp.headroom)
            lp.con([(1.0, x), (-1.0, r_dn), (1.0, hd)], -P + inp.headroom, np.inf)
        if inp.end_price is None:
            tp = lp.var(1, 0.0, np.inf, cost=w * soft_penalty)
            tn = lp.var(1, 0.0, np.inf, cost=w * soft_penalty)
        else:
            tp = lp.var(1, 0.0, np.inf, cost=-w * inp.end_price * bat.eta_d)
            tn = lp.var(1, 0.0, np.inf, cost=w * inp.end_price / bat.eta_c)
        lp.con([(1.0, soc[n:]), (-1.0, tp), (1.0, tn)], bat.soc_target, bat.soc_target)
        if final:
            lp.add_cost(r_up, w * (bat.degradation + fees.toll_discharge) * a_up * dt)
            lp.add_cost(r_dn, w * (bat.degradation + fees.toll_charge) * a_dn * dt)
            cy = lp.var(1, 0.0, np.inf, cost=w * soft_penalty)
            lp.row([(dt, d), (dt * a_up * inp.cycle_activation, r_up), (-1.0, cy)], -np.inf, bat.max_cycles * bat.energy - inp.cycle_reserve)
        pos[st], trades[st], socs[st] = x, (s, b), soc
        prev = x

    if tax > 0:
        z = lp.var(n, 0.0, np.inf, cost=w * tax)
        lp.con([(1.0, z)] + rev_terms, rev_const, np.inf)
    return pos, trades, socs


def _deliverable(lp: LP, inp: DayInput, xs: list, stage: str, bat: Battery, penalty: float) -> None:
    """Any mix of scenario positions in the last open market keeps the state of charge within limits."""
    n, dt, P = inp.n, inp.dt, bat.power
    fin = np.ones(n, bool)
    for st in STAGES[STAGES.index(stage) + 1:]:
        fin &= ~np.asarray(inp.open[st], bool)
    k = n if fin.all() else int(np.argmin(fin))
    if k == 0:
        return
    hi, lo = lp.var(k, -P, P), lp.var(k, -P, P)
    for x in xs:
        lp.con([(1.0, hi), (-1.0, x[:k])], 0.0, np.inf)
        lp.con([(1.0, lo), (-1.0, x[:k])], -np.inf, 0.0)
    c, d = lp.var(k, 0.0, P), lp.var(k, 0.0, P)
    lp.con([(1.0, d), (-1.0, c), (-1.0, hi)], 0.0, 0.0)
    down, up = lp.var(k + 1, -np.inf, np.inf), lp.var(k + 1, -np.inf, np.inf)
    lp.con([(1.0, down[:1])], inp.soc0, inp.soc0)
    lp.con([(1.0, up[:1])], inp.soc0, inp.soc0)
    lp.con([(1.0, down[1:]), (-1.0, down[:-1]), (-bat.eta_c * dt, c), (dt / bat.eta_d, d)], 0.0, 0.0)
    lp.con([(1.0, up[1:]), (-1.0, up[:-1]), (bat.eta_c * dt, lo)], 0.0, 0.0)
    a, b = lp.var(k, 0.0, np.inf, cost=penalty), lp.var(k, 0.0, np.inf, cost=penalty)
    lp.con([(1.0, down[1:]), (1.0, a)], bat.soc_min, np.inf)
    lp.con([(1.0, up[1:]), (-1.0, b)], -np.inf, bat.soc_max)


def solve_scenarios(inps: list, probs, bat: Battery, fees: Fees, soft_penalty: float = 1e4, time_limit: float = 30.0,
                    shortfall: float | None = None, curve_stage: str | None = None, light: bool = False,
                    nonanticipative: bool = False, deliverable: bool = False) -> Plan:
    """Maximise expected profit over scenarios with a shared band and a price-monotone current-market curve."""
    probs = np.asarray(probs, float) / np.sum(probs)
    n = inps[0].n
    lp = LP()
    r_up, r_dn = _band(lp, inps, probs, bat.power)
    curve = curve_stage if curve_stage in STAGES and curve_stage not in inps[0].fixed else None
    g = inps[0].groups.get(curve) if curve is not None else None
    if g is not None and len(inps) > 1:
        for i in inps:
            pr = pd.Series(np.asarray(i.price[curve], float)).groupby(np.asarray(g)).transform("mean").to_numpy()
            i.price = dict(i.price, **{curve: pr})
    shared = {} if nonanticipative else None
    blocks = [_scenario(lp, inp, bat, fees, w, r_up, r_dn, soft_penalty, shortfall, light, shared)
              for w, inp in zip(probs, inps)]
    if curve is not None and len(inps) > 1:
        price = np.vstack([_clean(i.price[curve]) for i in inps])
        S = np.vstack([b[1][curve][0] for b in blocks])
        B = np.vstack([b[1][curve][1] for b in blocks])
        order = np.argsort(price, axis=0, kind="stable")
        cols = np.arange(n)
        for k in range(len(inps) - 1):
            ia, ib = order[k], order[k + 1]
            eq = np.isclose(price[ia, cols], price[ib, cols])
            lp.con([(1.0, S[ia, cols]), (-1.0, B[ia, cols]), (-1.0, S[ib, cols]), (1.0, B[ib, cols])],
                   np.where(eq, 0.0, -np.inf), 0.0)
        if deliverable:
            _deliverable(lp, inps[0], [b[0][curve] for b in blocks], curve, bat,
                         shortfall if shortfall is not None else soft_penalty)

    sol, fun = lp.solve(time_limit=time_limit)
    position = {st: sum(w * sol[b[0][st]] for w, b in zip(probs, blocks)) for st in blocks[0][0]}
    trade = {st: sum(w * (sol[b[1][st][0]] - sol[b[1][st][1]]) for w, b in zip(probs, blocks)) for st in blocks[0][1]}
    last = STAGES[-1]
    soc = sum(w * sol[b[2][last]] for w, b in zip(probs, blocks)) if last in blocks[0][2] else np.full(n + 1, np.nan)
    plan = Plan(position=position, trade=trade, r_up=sol[r_up], r_dn=sol[r_dn], soc=soc, objective=-fun)
    if curve is not None:
        plan.scen_price = np.vstack([np.asarray(i.price[curve], float) for i in inps])
        plan.scen_trade = np.vstack([sol[b[1][curve][0]] - sol[b[1][curve][1]] for b in blocks])
    return plan


def solve_day(inp: DayInput, bat: Battery, fees: Fees, soft_penalty: float = 1e4, time_limit: float = 30.0) -> Plan:
    """Maximise profit of the day over all stages not yet cleared."""
    return solve_scenarios([inp], [1.0], bat, fees, soft_penalty, time_limit)

