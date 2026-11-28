"""Scenario generation by joint whole-day error bootstrap and k-medoids reduction."""
from __future__ import annotations

import numpy as np

from .uncertainty import ErrorModel


def kmedoids(X: np.ndarray, w: np.ndarray, k: int, iters: int = 30, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """Weighted k-medoids; returns medoid rows and the weight of each cluster."""
    m = len(X)
    if m <= k:
        return np.arange(m), w / w.sum()
    D = np.sqrt(np.maximum(((X[:, None, :] - X[None, :, :]) ** 2).sum(-1), 0.0))
    rng = np.random.default_rng(seed)
    med = [int(np.argmin((D * w[None, :]).sum(1)))]
    for _ in range(1, k):
        near = D[:, med].min(1)
        p = w * near ** 2
        med.append(int(rng.choice(m, p=p / p.sum())) if p.sum() > 0 else int(rng.integers(m)))
    med = np.array(med)
    for _ in range(iters):
        lab = np.argmin(D[:, med], axis=1)
        new = med.copy()
        for j in range(k):
            mem = np.flatnonzero(lab == j)
            if len(mem):
                new[j] = mem[np.argmin((D[np.ix_(mem, mem)] * w[mem][None, :]).sum(1))]
        if np.array_equal(new, med):
            break
        med = new
    lab = np.argmin(D[:, med], axis=1)
    cw = np.array([w[lab == j].sum() for j in range(k)])
    keep = cw > 0
    return med[keep], cw[keep] / cw.sum()


class ScenarioGenerator:
    """Point forecasts plus errors of similar past days, reduced to a few weighted scenarios."""

    def __init__(self, errors: ErrorModel, cfg, calibrated: dict | None = None):
        sc = cfg["scenarios"]
        self.errors = errors
        self.samples = int(sc["samples"])
        self.count = int(sc["count"])
        self.by_gate = {k: int(v) for k, v in (sc.get("count_by_gate") or {}).items()}
        self.seed = int(sc["seed"])
        self.bounds = cfg["forecast"].get("bounds") or {}
        self.only = sc.get("targets")
        self.cal = calibrated or {}

    def generate(self, day, point: dict, known: dict, slots: np.ndarray, source: dict,
                 gate: str | None = None) -> tuple[list, np.ndarray]:
        """Scenario dicts and probabilities; one scenario equal to the point forecast when no history exists."""
        count = self.by_gate.get(gate, self.count)
        idx, w = self.errors.pool(day)
        targets = [t for t in source if np.isnan(known[t]).any() and (not self.only or t in self.only)]
        if not len(idx) or not targets or count <= 1:
            return [point], np.array([1.0])
        rng = np.random.default_rng(self.seed + day.toordinal())
        draw = rng.choice(len(idx), size=self.samples, p=w)
        uniq, cnt = np.unique(draw, return_counts=True)
        rows = idx[uniq]
        err = {}
        for t in targets:
            g, model = source[t]
            q = self.quantile_errors(t, g, rows, day) if model else None
            if q is not None:
                err[t] = np.where(np.isnan(known[t])[None, :], q[:, slots], 0.0)
                continue
            e = self.errors.scaled(t, g, rows, day) if model else None
            e = (e if e is not None else self.errors.matrix(t, g, rows, model))[:, slots]
            err[t] = np.where(np.isnan(known[t])[None, :], e, 0.0)
        feats = []
        for t in targets:
            s = err[t].std()
            if s > 0:
                feats.append(err[t] / s)
        X = np.hstack(feats) if feats else np.zeros((len(rows), 1))
        med, prob = kmedoids(X, cnt.astype(float), count, seed=self.seed)
        out = []
        for m in med:
            v = dict(point)
            for t in targets:
                x = np.asarray(point[t], float) + err[t][m]
                lo, hi = self.bounds.get(t, [None, None])
                v[t] = np.clip(x, -np.inf if lo is None else lo, np.inf if hi is None else hi)
            out.append(v)
        return out, prob

    def quantile_errors(self, target: str, gate: str, rows: np.ndarray, day) -> np.ndarray | None:
        """Errors at today's calibrated quantiles, read at the ranks the sampled past days had."""
        c = self.cal.get((target, gate))
        i = c.pos.get(str(day)) if c is not None else None
        if i is None or not c.ready(i):
            return None
        past = [c.pos.get(str(d)) for d in self.errors.days[rows]]
        u = np.array([c.pit[j] if j is not None else np.full(c.pit.shape[1], 0.5) for j in past])
        return np.nan_to_num(c.errors(np.where(np.isnan(u), 0.5, u), i))
