"""Daily LP for energy positions and aFRR band."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
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
    dt: float = 0.25


@dataclass
class Plan:
    position: dict
    trade: dict
    r_up: np.ndarray
    r_dn: np.ndarray
    soc: np.ndarray
    objective: float


def _clean(a) -> np.ndarray:
    return np.nan_to_num(np.asarray(a, float), nan=0.0)


def solve_day(inp: DayInput, bat: Battery, fees: Fees, soft_penalty: float = 1e4, time_limit: float = 30.0) -> Plan:
    """Maximise expected profit of the day over all stages not yet cleared."""
    n, dt, P = inp.n, inp.dt, bat.power
    lp = LP()
    a_up, a_dn = np.clip(_clean(inp.share_up), 0, 1), np.clip(_clean(inp.share_dn), 0, 1)
    cap_up, cap_dn = _clean(inp.cap_up), _clean(inp.cap_dn)
    band_ok = ~(np.isnan(np.asarray(inp.cap_up, float)) | np.isnan(np.asarray(inp.cap_dn, float)))
    if inp.band_fixed is not None:
        r_up = lp.var(n, inp.band_fixed[0], inp.band_fixed[0])
        r_dn = lp.var(n, inp.band_fixed[1], inp.band_fixed[1])
    else:
        ub = np.where(band_ok, P, 0.0)
        r_up = lp.var(n, 0.0, ub)
        r_dn = lp.var(n, 0.0, ub)
    band_value = (cap_up + a_up * _clean(inp.e_up)) * dt, (cap_dn - a_dn * _clean(inp.e_dn)) * dt
    lp.add_cost(r_up, -band_value[0])
    lp.add_cost(r_dn, -band_value[1])
    if inp.ratio_up is not None and inp.band_fixed is None:
        q = np.clip(_clean(inp.ratio_up), 0, 1)
        lp.con([(1 - q, r_up), (-q, r_dn)], 0.0, 0.0)
    g = inp.groups.get("afrr")
    if g is not None and inp.band_fixed is None:
        g = np.asarray(g)
        first = np.array([np.flatnonzero(g == v)[0] for v in g])
        rows = np.flatnonzero(first != np.arange(n))
        if len(rows):
            lp.con([(1.0, r_up[rows]), (-1.0, r_up[first[rows]])], 0.0, 0.0)
            lp.con([(1.0, r_dn[rows]), (-1.0, r_dn[first[rows]])], 0.0, 0.0)

    prev = None
    pos, trades, socs = {}, {}, {}
    last = len(STAGES) - 1
    for k, st in enumerate(STAGES):
        if st in inp.fixed:
            x = lp.var(n, inp.fixed[st], inp.fixed[st])
            pos[st] = x
            prev = x
            continue
        price = np.asarray(inp.price[st], float)
        open_ = np.asarray(inp.open[st], bool) & ~np.isnan(price)
        p = _clean(price)
        tub = np.where(open_, 2 * P, 0.0)
        s = lp.var(n, 0.0, tub, cost=-p * dt + fees.market * dt + fees.ivpee * np.maximum(p, 0) * dt)
        b = lp.var(n, 0.0, tub, cost=p * dt + fees.market * dt)
        x = lp.var(n, -P, P)
        terms = [(1.0, x), (-1.0, s), (1.0, b)]
        if prev is not None:
            terms.append((-1.0, prev))
        lp.con(terms, 0.0, 0.0)
        g = inp.groups.get(st)
        if g is not None:
            g = np.asarray(g)
            first = np.array([np.flatnonzero(g == v)[0] for v in g])
            rows = np.flatnonzero(first != np.arange(n))
            if len(rows):
                lp.con([(1.0, s[rows]), (-1.0, b[rows]), (-1.0, s[first[rows]]), (1.0, b[first[rows]])], 0.0, 0.0)
        final = k == last
        c = lp.var(n, 0.0, P, cost=(bat.degradation + fees.toll_charge) * dt if final else 0.01 * dt)
        d = lp.var(n, 0.0, P, cost=(bat.degradation + fees.toll_discharge) * dt if final else 0.01 * dt)
        lp.con([(1.0, d), (-1.0, c), (-1.0, x)], 0.0, 0.0)
        soc = lp.var(n + 1, -np.inf, np.inf)
        lo = lp.var(n + 1, 0.0, np.inf, cost=soft_penalty)
        hi = lp.var(n + 1, 0.0, np.inf, cost=soft_penalty)
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
        if band:
            lp.con([(1.0, x), (1.0, r_up)], -np.inf, P)
            lp.con([(1.0, x), (-1.0, r_dn)], -P, np.inf)
        tp = lp.var(1, 0.0, np.inf, cost=soft_penalty)
        tn = lp.var(1, 0.0, np.inf, cost=soft_penalty)
        lp.con([(1.0, soc[n:]), (-1.0, tp), (1.0, tn)], bat.soc_target, bat.soc_target)
        if final:
            lp.add_cost(r_up, (bat.degradation + fees.toll_discharge) * a_up * dt)
            lp.add_cost(r_dn, (bat.degradation + fees.toll_charge) * a_dn * dt)
            cy = lp.var(1, 0.0, np.inf, cost=soft_penalty)
            lp.row([(dt, d), (dt * a_up, r_up), (-1.0, cy)], -np.inf, bat.max_cycles * bat.energy)
        pos[st], trades[st], socs[st] = x, (s, b), soc
        prev = x

    sol, fun = lp.solve(time_limit=time_limit)
    position = {st: sol[i] for st, i in pos.items()}
    trade = {st: sol[s] - sol[b] for st, (s, b) in trades.items()}
    final_soc = sol[socs[STAGES[-1]]] if STAGES[-1] in socs else np.full(n + 1, np.nan)
    return Plan(position=position, trade=trade, r_up=sol[r_up], r_dn=sol[r_dn], soc=final_soc, objective=-fun)
