"""Daily LP for energy positions and aFRR band."""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.optimize import Bounds, LinearConstraint, linprog, milp
from scipy.sparse import coo_matrix, vstack

from .battery import Battery, Fees

STAGES = ["da", "ida1", "ida2", "ida3"]
BAND_FROM = 2
LAZY_ROUNDS = 4
FLOW_TOL = 1e-6


_ENV = {}


def _gurobi_env():
    """Shared quiet Gurobi environment, or None when gurobipy or its licence is unavailable."""
    if "env" not in _ENV:
        try:
            import gurobipy as gp
            env = gp.Env(empty=True)
            env.setParam("OutputFlag", 0)
            env.start()
            _ENV["env"] = env
        except Exception:
            _ENV["env"] = None
    return _ENV["env"]


class LP:
    """Sparse LP builder solved with Gurobi when selected and licensed, otherwise HiGHS."""

    solver = "highs"
    deterministic = False
    method = "highs"
    gap = 1e-4

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

    def solve(self, time_limit: float = 30.0, gap: float | None = None) -> tuple[np.ndarray, float]:
        gap = self.gap if gap is None else gap
        c = np.concatenate(self.c)
        for idx, cost in self.extra:
            np.add.at(c, idx, cost)
        a = coo_matrix((np.concatenate(self.v), (np.concatenate(self.r), np.concatenate(self.k))), shape=(self.m, self.n)).tocsr()
        lo, hi = np.concatenate(self.rlb), np.concatenate(self.rub)
        lb, ub = np.concatenate(self.lb), np.concatenate(self.ub)
        integ = np.concatenate(self.integ)
        if self.solver == "gurobi" and _gurobi_env() is not None:
            return self._gurobi(c, a, lo, hi, lb, ub, integ, time_limit, gap)
        if integ.any():
            res = milp(c, integrality=integ, bounds=Bounds(lb, ub), constraints=LinearConstraint(a, lo, hi),
                       options={"time_limit": time_limit, "mip_rel_gap": gap, "disp": False})
            if res.x is None:
                raise RuntimeError(f"MILP failed: {res.message}")
            return res.x, float(res.fun)
        eq = np.isfinite(lo) & (lo == hi)
        up, dn = ~eq & np.isfinite(hi), ~eq & np.isfinite(lo)
        res = linprog(c, A_ub=vstack([a[up], -a[dn]]).tocsr(), b_ub=np.concatenate([hi[up], -lo[dn]]),
                      A_eq=a[eq], b_eq=lo[eq], bounds=np.column_stack([lb, ub]), method=self.method,
                      options={"time_limit": time_limit})
        if res.status != 0 or res.x is None:
            raise RuntimeError(f"LP failed: {res.message}")
        return res.x, float(res.fun)

    def _gurobi(self, c, a, lo, hi, lb, ub, integ, time_limit, gap) -> tuple[np.ndarray, float]:
        import gurobipy as gp
        inf = gp.GRB.INFINITY
        m = gp.Model(env=_gurobi_env())
        if self.deterministic:
            m.Params.WorkLimit = time_limit
        else:
            m.Params.TimeLimit = time_limit
        m.Params.MIPGap = gap
        x = m.addMVar(self.n, lb=np.clip(lb, -inf, inf), ub=np.clip(ub, -inf, inf),
                      vtype=np.where(integ > 0, gp.GRB.INTEGER, gp.GRB.CONTINUOUS))
        eq = np.isfinite(lo) & (lo == hi)
        up, dn = ~eq & np.isfinite(hi), ~eq & np.isfinite(lo)
        for rows, sense, rhs in ((eq, "=", lo), (up, "<", hi), (dn, ">", lo)):
            if rows.any():
                m.addMConstr(a[rows], x, sense, rhs[rows])
        m.setObjective(c @ x, gp.GRB.MINIMIZE)
        m.optimize()
        if m.SolCount == 0:
            raise RuntimeError(f"Gurobi failed: status {m.Status}")
        return x.X, float(m.ObjVal)


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
    trade_cap: dict = field(default_factory=dict)
    band_fixed: tuple | None = None
    ratio_up: np.ndarray | None = None
    end_price: float | None = None
    end_curve: tuple | None = None
    headroom: float = 0.0
    headroom_sell: float = 0.0
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
              shortfall: float | None, light: bool = False, shared: dict | None = None,
              binaries: dict | None = None) -> tuple[dict, dict, dict, dict]:
    """Stage blocks of one scenario with costs weighted by its probability.

    binaries maps an interim stage to a direction per period: 1 discharge only, -1 charge only, 0 free.
    None gives every period an integer switch instead.
    """
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
    pos, trades, socs, flows = {}, {}, {}, {}
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
        cap = inp.trade_cap.get(st)
        tub = np.where(open_, 2 * P if cap is None else np.minimum(2 * P, cap), 0.0)
        if shared is not None and st != current and "dev" in shared:
            s = lp.var(n, 0.0, tub, cost=w * (-p * dt + fees.market * dt))
            b = lp.var(n, 0.0, tub, cost=w * (p * dt + fees.market * dt))
            rule = shared["rule"]
            if st not in rule:
                rule[st] = (lp.var(n, -np.inf, np.inf), lp.var(n, -np.inf, np.inf))
            base, slope = rule[st]
            lp.con([(1.0, s), (-1.0, b), (-1.0, base), (-shared["dev"], slope)], 0.0, 0.0)
        elif shared is not None and st != current and st in shared:
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
        c = lp.var(n, 0.0, P, cost=w * ((bat.deg_c + fees.toll_charge) * dt if final else inp.interim_cost * dt))
        d = lp.var(n, 0.0, P, cost=w * ((bat.deg_d + fees.toll_discharge) * dt if final else inp.interim_cost * dt))
        lp.con([(1.0, d), (-1.0, c), (-1.0, x)], 0.0, 0.0)
        if inp.interim_exclusive and not final:
            flows[st] = (c, d)
            if binaries is None:
                z = lp.var(n, 0.0, 1.0, integer=True)
                lp.con([(1.0, c), (-P, z)], -np.inf, 0.0)
                lp.con([(1.0, d), (P, z)], -np.inf, P)
            else:
                way = np.asarray(binaries.get(st, np.zeros(n, int)), int)
                if (way > 0).any():
                    lp.con([(1.0, c[way > 0])], -np.inf, 0.0)
                if (way < 0).any():
                    lp.con([(1.0, d[way < 0])], -np.inf, 0.0)
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
            lp.con([(1.0, x), (1.0, r_up), (-1.0, hu)], -np.inf, P - inp.headroom_sell)
            lp.con([(1.0, x), (-1.0, r_dn), (1.0, hd)], -P + inp.headroom, np.inf)
        if inp.end_curve is not None:
            xs, vs = (np.asarray(a, float) for a in inp.end_curve)
            seg = lp.var(len(xs) - 1, 0.0, np.diff(xs), cost=-w * np.diff(vs) / np.diff(xs))
            lp.row([(1.0, soc[n:]), (-1.0, seg)], xs[0], xs[0])
        elif inp.end_price is None:
            tp = lp.var(1, 0.0, np.inf, cost=w * soft_penalty)
            tn = lp.var(1, 0.0, np.inf, cost=w * soft_penalty)
        else:
            tp = lp.var(1, 0.0, np.inf, cost=-w * inp.end_price * bat.eta_d)
            tn = lp.var(1, 0.0, np.inf, cost=w * inp.end_price / bat.eta_c)
        if inp.end_curve is None:
            lp.con([(1.0, soc[n:]), (-1.0, tp), (1.0, tn)], bat.soc_target, bat.soc_target)
        if final:
            lp.add_cost(r_up, w * (bat.deg_d + fees.toll_discharge) * a_up * dt)
            lp.add_cost(r_dn, w * (bat.deg_c + fees.toll_charge) * a_dn * dt)
            if np.isfinite(bat.max_cycles):
                cy = lp.var(1, 0.0, np.inf, cost=w * soft_penalty)
                lp.row([(dt, d), (dt * a_up * inp.cycle_activation, r_up), (-1.0, cy)], -np.inf, bat.max_cycles * bat.energy - inp.cycle_reserve)
        pos[st], trades[st], socs[st] = x, (s, b), soc
        prev = x

    if tax > 0:
        z = lp.var(n, 0.0, np.inf, cost=w * tax)
        lp.con([(1.0, z)] + rev_terms, rev_const, np.inf)
    return pos, trades, socs, flows


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


def _groups(inps: list, curve: str | None, nonanticipative) -> np.ndarray | None:
    """Group index per scenario for the shared later-auction plan, by mean current-auction price."""
    if nonanticipative is False or nonanticipative is None:
        return None
    k = 1 if nonanticipative is True else int(nonanticipative)
    if k <= 0 or k >= len(inps):
        return None
    if k == 1 or curve is None:
        return np.zeros(len(inps), int)
    level = np.array([np.nanmean(_clean(i.price[curve])) for i in inps])
    rank = np.empty(len(inps), int)
    rank[np.argsort(level, kind="stable")] = np.arange(len(inps))
    return rank * k // len(inps)


def _cvar(lp: LP, spans: list, probs: np.ndarray, weight: float, alpha: float) -> None:
    """Add weight times the conditional value at risk of scenario profit to the objective."""
    starts = np.cumsum([0] + [len(c) for c in lp.c])
    eta = lp.var(1, -np.inf, np.inf, cost=-weight)
    u = lp.var(len(spans), 0.0, np.inf, cost=weight * probs / (1.0 - alpha))
    for j, (c0, c1, e0, e1) in enumerate(spans):
        w = probs[j]
        idx = [np.arange(starts[c0], starts[c1])] + [i for i, _ in lp.extra[e0:e1]]
        val = [np.concatenate(lp.c[c0:c1]) / w if c1 > c0 else np.zeros(0)] + [c / w for _, c in lp.extra[e0:e1]]
        idx, val = np.concatenate(idx), np.concatenate(val)
        keep = val != 0
        lp.row([(1.0, u[j:j + 1]), (-1.0, eta), (-val[keep], idx[keep])], 0.0, np.inf)


def solve_scenarios(inps: list, probs, bat: Battery, fees: Fees, soft_penalty: float = 1e4, time_limit: float = 30.0,
                    shortfall: float | None = None, curve_stage: str | None = None, light: bool = False,
                    nonanticipative: bool | int | str = False, deliverable: bool = False,
                    risk_weight: float = 0.0, risk_alpha: float = 0.9, single: bool = False) -> Plan:
    """Maximise expected profit over scenarios with a shared band and a price-monotone current-market curve."""
    probs = np.asarray(probs, float) / np.sum(probs)
    n = inps[0].n
    curve = curve_stage if curve_stage in STAGES and curve_stage not in inps[0].fixed else None
    g = inps[0].groups.get(curve) if curve is not None else None
    if g is not None and len(inps) > 1:
        for i in inps:
            pr = pd.Series(np.asarray(i.price[curve], float)).groupby(np.asarray(g)).transform("mean").to_numpy()
            i.price = dict(i.price, **{curve: pr})

    affine = nonanticipative == "affine" and curve is not None and len(inps) > 1
    group = None if affine else _groups(inps, curve, nonanticipative)
    dev = None
    if affine:
        price = np.vstack([_clean(i.price[curve]) for i in inps])
        dev = (price - probs @ price) / 100.0

    def build(binaries):
        lp = LP()
        r_up, r_dn = _band(lp, inps, probs, bat.power)
        shared = [{} for _ in range(int(group.max()) + 1)] if group is not None else [None]
        if dev is not None:
            rule = {}
            shared = [{"rule": rule, "dev": d} for d in dev]
        blocks, spans = [], []
        for j, (w, inp) in enumerate(zip(probs, inps)):
            n0, e0 = len(lp.c), len(lp.extra)
            blocks.append(_scenario(lp, inp, bat, fees, w, r_up, r_dn, soft_penalty, shortfall, light,
                                    shared[j] if dev is not None else shared[group[j]] if group is not None else None,
                                    None if binaries is None else binaries[j]))
            spans.append((n0, len(lp.c), e0, len(lp.extra)))
        if risk_weight > 0 and len(inps) > 1:
            _cvar(lp, spans, probs, risk_weight, risk_alpha)
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
                       0.0 if single else np.where(eq, 0.0, -np.inf), 0.0)
            if deliverable:
                _deliverable(lp, inps[0], [b[0][curve] for b in blocks], curve, bat,
                             shortfall if shortfall is not None else soft_penalty)
        return lp, blocks, r_up, r_dn

    lazy = LP.solver != "gurobi" and any(i.interim_exclusive for i in inps)
    binaries = [{} for _ in inps] if lazy else None
    for _ in range(LAZY_ROUNDS + 1):
        lp, blocks, r_up, r_dn = build(binaries)
        sol, fun = lp.solve(time_limit=time_limit)
        if binaries is None:
            break
        added = False
        for j, b in enumerate(blocks):
            for st, (c, d) in b[3].items():
                both = (sol[c] > FLOW_TOL) & (sol[d] > FLOW_TOL)
                if both.any():
                    way = binaries[j].get(st, np.zeros(n, int)).copy()
                    way[both] = np.where(sol[d][both] >= sol[c][both], 1, -1)
                    binaries[j][st] = way
                    added = True
        if not added:
            break
    else:
        lp, blocks, r_up, r_dn = build(None)
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


def next_day_value(price: np.ndarray, bat: Battery, fees: Fees, socs: np.ndarray, dt: float = 0.25) -> np.ndarray:
    """Best one-market profit of the following day from each starting state of charge to the usual end target."""
    p = _clean(price)
    n = len(p)
    out = []
    for s0 in socs:
        lp = LP()
        c = lp.var(n, 0.0, bat.power, cost=(p + bat.deg_c + fees.toll_charge + fees.market) * dt)
        d = lp.var(n, 0.0, bat.power, cost=(-p + bat.deg_d + fees.toll_discharge + fees.market) * dt)
        soc = lp.var(n + 1, bat.soc_min, bat.soc_max)
        lp.con([(1.0, soc[:1])], s0, s0)
        lp.con([(1.0, soc[1:]), (-1.0, soc[:-1]), (-bat.eta_c * dt, c), (dt / bat.eta_d, d)], 0.0, 0.0)
        lp.con([(1.0, soc[n:])], bat.soc_target, bat.soc_target)
        if np.isfinite(bat.max_cycles):
            lp.row([(dt, d)], -np.inf, bat.max_cycles * bat.energy)
        out.append(-lp.solve()[1])
    return np.array(out)
