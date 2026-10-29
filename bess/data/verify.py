"""Phase-1 verifications.

Each check returns a row: item, question, finding, status, evidence. Status values:
    data       - established from the downloaded data
    docs       - from an official document (recorded in config)
    secondary  - from a secondary source only
    mismatch   - data contradicts the config: fix the config
    unverified - no evidence yet
    no data    - the series needed for the check is not in the store
"""
from __future__ import annotations

import logging
from collections import Counter

import pandas as pd

from ..config import Config
from ..timeutils import delivery_day
from .esios import name_matches
from .quality import cross_check
from .store import Store

log = logging.getLogger(__name__)


def _row(item, question, finding, status, evidence=""):
    return {"item": item, "question": question, "finding": finding, "status": status, "evidence": evidence}


def _first_day_with_resolution(df: pd.DataFrame, tz: str, res: int):
    d = df[df["resolution_min"] == res]
    if d.empty:
        return None
    return delivery_day(d["delivery_start"], tz).min()


def check_qh_switch(store: Store, cfg: Config) -> list[dict]:
    rows = []
    for m in ("da", "ida1", "ida2", "ida3"):
        df = store.load(series=f"{m}_price_es")
        q = f"From which delivery day does {m.upper()} trade 15-min products?"
        if df.empty:
            rows.append(_row(f"qh_switch_{m}", q, "", "no data"))
            continue
        first_qh = _first_day_with_resolution(df, cfg.tz, 15)
        last_h = (delivery_day(df.loc[df["resolution_min"] == 60, "delivery_start"], cfg.tz).max()
                  if (df["resolution_min"] == 60).any() else None)
        key = "da_qh" if m == "da" else "ida_qh"
        expected = cfg["regimes"][key]["date"]
        if first_qh is None:
            rows.append(_row(f"qh_switch_{m}", q, f"only hourly data up to {last_h}", "data",
                             f"config {key}={expected}"))
            continue
        status = "data" if str(first_qh) == expected or m != "da" else "mismatch"
        rows.append(_row(f"qh_switch_{m}", q, f"first 15-min delivery day {first_qh} (last hourly day {last_h})",
                         status, f"config regimes.{key} = {expected}"))
    return rows


def _clock(ts: pd.Series, tz: str) -> pd.Series:
    return ts.dt.tz_convert(tz).dt.strftime("%H:%M")


def check_ida_coverage(store: Store, cfg: Config) -> list[dict]:
    rows = []
    for s in (1, 2, 3):
        m = f"ida{s}"
        df = store.load(series=f"{m}_price_es")
        q = f"Which delivery periods does {m.upper()} cover?"
        if df.empty:
            rows.append(_row(f"coverage_{m}", q, "", "no data"))
            continue
        df = df.assign(day=delivery_day(df["delivery_start"], cfg.tz).to_numpy())
        found = []
        for res, g in df.groupby("resolution_min"):
            per_day = g.groupby("day").agg(a=("delivery_start", "min"), b=("delivery_end", "max"), n=("value", "size"))
            starts = Counter(_clock(per_day["a"], cfg.tz))
            ends = Counter(_clock(per_day["b"], cfg.tz).replace("00:00", "24:00"))
            a, b = starts.most_common(1)[0][0], ends.most_common(1)[0][0]
            share = starts.most_common(1)[0][1] / len(per_day)
            found.append((res, a, b, share, len(per_day)))
        cov = cfg["markets"][m]["coverage"]
        ok = all(a == cov["from"] and b == cov["to"] for _, a, b, _, _ in found)
        txt = "; ".join(f"{r}-min: {a}-{b} on {sh:.0%} of {n} days" for r, a, b, sh, n in found)
        rows.append(_row(f"coverage_{m}", q, txt, "data" if ok else "mismatch",
                         f"config coverage {cov['from']}-{cov['to']}"))
    return rows


def check_afrr(store: Store, cfg: Config) -> list[dict]:
    rows = []
    keys = ["afrr_band_price_up", "afrr_band_price_down", "afrr_band_up_mw", "afrr_band_down_mw",
            "afrr_energy_price_up", "afrr_energy_price_down", "afrr_energy_up_mwh", "afrr_energy_down_mwh"]
    found = {}
    for k in keys:
        df = store.load(series=k)
        if df.empty:
            continue
        df = df.assign(day=delivery_day(df["delivery_start"], cfg.tz).to_numpy())
        res = df.groupby("resolution_min")["day"].agg(["min", "max"])
        found[k] = (df["day"].min(), df["day"].max(), res)
    q = "aFRR capacity: time resolution (hourly vs 15-min) per period"
    if not any(k in found for k in keys[:4]):
        rows.append(_row("afrr_resolution", q, "", "no data"))
    else:
        parts = []
        for k in keys[:4]:
            if k in found:
                r = found[k][2]
                parts.append(f"{k}: " + ", ".join(f"{int(i)}-min {row['min']}..{row['max']}" for i, row in r.iterrows()))
        rows.append(_row("afrr_resolution", q, " | ".join(parts), "data"))
    q = "aFRR capacity: date of the separate upward/downward price"
    if "afrr_band_price_up" in found:
        first = found["afrr_band_price_up"][0]
        expected = cfg["regimes"]["afrr_split"]["date"]
        rows.append(_row("afrr_split", q, f"first day with a separate upward price: {first}",
                         "data" if str(first) == expected else "mismatch", f"config regimes.afrr_split = {expected}"))
    else:
        rows.append(_row("afrr_split", q, "", "no data"))
    q = "aFRR activation share inputs available (activated energy and awarded band, up and down)?"
    need = ["afrr_energy_up_mwh", "afrr_energy_down_mwh", "afrr_band_up_mw", "afrr_band_down_mw"]
    have = [k for k in need if k in found]
    rows.append(_row("afrr_activation_inputs", q, f"present: {', '.join(have) or 'none'}",
                     "data" if len(have) == 4 else ("no data" if not have else "unverified")))
    return rows


def check_esios_indicators(store: Store, cfg: Config, client=None) -> list[dict]:
    """Compare each configured indicator's real name with the expected keywords."""
    meta = store.read_events("esios_indicator_meta")
    if client is not None:
        rec = []
        for key, spec in cfg["esios"]["indicators"].items():
            if spec.get("id") is None:
                continue
            try:
                m = client.metadata(spec["id"])
                rec.append({"key": key, "id": spec["id"], "name": m.get("name", ""), "short_name": m.get("short_name", ""),
                            "tiempo": ",".join(str(t.get("name")) for t in m.get("tiempo") or []),
                            "fetched_at": pd.Timestamp.now(tz="UTC")})
            except Exception as e:
                rec.append({"key": key, "id": spec["id"], "name": f"ERROR {e}", "short_name": "", "tiempo": "",
                            "fetched_at": pd.Timestamp.now(tz="UTC")})
        if rec:
            store.write_events("esios_indicator_meta", pd.DataFrame(rec), key=["key"])
            meta = store.read_events("esios_indicator_meta")
    catalog = store.read_events("esios_catalog")
    rows = []
    for key, spec in cfg["esios"]["indicators"].items():
        q = f"eSIOS {key}: is indicator {spec.get('id')} the right one?"
        name = None
        if not meta.empty and key in set(meta["key"]):
            name = meta.loc[meta["key"] == key, "name"].iloc[-1]
        elif not catalog.empty and spec.get("id") in set(catalog["id"]):
            name = catalog.loc[catalog["id"] == spec["id"], "name"].iloc[-1]
        if spec.get("id") is None:
            rows.append(_row(f"esios_{key}", q, "no id configured", "unverified", "use `bess esios-catalog --search`"))
        elif name is None:
            rows.append(_row(f"esios_{key}", q, "metadata not fetched", "unverified",
                             f"expected words: {spec.get('expect')}"))
        else:
            ok = name_matches(name, spec.get("expect", []))
            rows.append(_row(f"esios_{key}", q, f"{spec['id']} = '{name}'", "data" if ok else "mismatch",
                             f"expected words: {spec.get('expect')}"))
    return rows


def check_price_limits(store: Store, cfg: Config) -> list[dict]:
    rows = []
    for m in ("da", "ida1", "ida2", "ida3", "afrr_capacity"):
        series = [f"{m}_price_es"] if m != "afrr_capacity" else ["afrr_band_price_up", "afrr_band_price_down"]
        df = store.load(series=series)
        lim = cfg["markets"][m]["price_limits"]
        q = f"{m.upper()} price limits (config min {lim.get('min')}, max {lim.get('max')})"
        if df.empty:
            rows.append(_row(f"limits_{m}", q, "", "no data"))
            continue
        lo, hi = df["value"].min(), df["value"].max()
        breach = (lim.get("min") is not None and lo < lim["min"]) or (lim.get("max") is not None and hi > lim["max"])
        rows.append(_row(f"limits_{m}", q, f"observed min {lo:.2f}, max {hi:.2f}",
                         "mismatch" if breach else cfg["markets"][m].get("verified", {}).get("price_limits", "unverified"),
                         "observed range inside the configured limits" if not breach else "observed value outside configured limits"))
    return rows


def check_config_facts(cfg: Config) -> list[dict]:
    rows = []
    for m, spec in cfg["markets"].items():
        v = spec.get("verified", {})
        rows.append(_row(f"gate_{m}", f"{spec.get('label', m)} gate closure / results time",
                         f"gate D{spec['gate']['day_offset']:+d} {spec['gate']['time']}, results D{spec['results']['day_offset']:+d} {spec['results']['time']} (local)",
                         v.get("gate", "unverified") if v.get("gate") != "no" else "unverified",
                         f"results time: {v.get('results', 'unverified')}"))
    for k, spec in cfg["fees"].items():
        rows.append(_row(f"fee_{k}", f"Storage fee/tax: {k}", f"{spec['value']} ({spec.get('applies_to')})",
                         "unverified" if spec.get("verified") in (None, "no", False) else spec["verified"], spec.get("note", "")))
    for k, spec in cfg["regimes"].items():
        rows.append(_row(f"regime_{k}", f"Regime boundary {k}", spec["date"],
                         "unverified" if spec.get("verified") in (None, "no", False) else spec["verified"], spec.get("note", "")))
    return rows


def check_cross_sources(store: Store) -> list[dict]:
    rows = []
    for a, b in (("da_price_es", "da_price_esios"), ("da_price_es", "da_price_entsoe")):
        q = f"Do {a} (OMIE) and {b} agree?"
        r = cross_check(store, a, b)
        if not r or not r.get("overlap_periods"):
            rows.append(_row(f"xcheck_{b}", q, "", "no data"))
            continue
        ok = r["match_share"] > 0.99
        rows.append(_row(f"xcheck_{b}", q,
                         f"{r['match_share']:.2%} of {r['overlap_periods']} QH within 0.01 EUR/MWh; max |diff| {r['max_abs_diff']:.2f} at {r['worst_period_utc']}",
                         "data" if ok else "mismatch"))
    return rows


def run_verifications(store: Store, cfg: Config, esios_client=None) -> pd.DataFrame:
    rows = []
    for fn in (check_qh_switch, check_ida_coverage, check_afrr, check_price_limits):
        try:
            rows += fn(store, cfg)
        except Exception as e:
            log.exception("verification %s failed", fn.__name__)
            rows.append(_row(fn.__name__, "", f"check crashed: {e}", "unverified"))
    rows += check_esios_indicators(store, cfg, esios_client)
    rows += check_cross_sources(store)
    rows += check_config_facts(cfg)
    return pd.DataFrame(rows)
