"""Self-contained HTML reports (no external assets; inline CSS and SVG charts)."""
from __future__ import annotations

import html
import math
from datetime import date
from pathlib import Path

import pandas as pd

from .config import Config
from .data import quality, timeline, verify
from .data.store import Store

PALETTE = ["#2f6fdf", "#d9622b", "#1f9e74", "#a347c9", "#c9a227", "#5b6b7f", "#d6457a", "#3fa7c4"]

CSS = """
:root{--bg:#f7f7f5;--panel:#fff;--ink:#1d2330;--muted:#5d6675;--line:#e1e3e8;--accent:#2f6fdf;
--ok:#1f8a5b;--okbg:#e3f4ec;--warn:#9a6a00;--warnbg:#fbf1d9;--bad:#b3261e;--badbg:#fbe4e2;--na:#5d6675;--nabg:#eceef2}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#14171c;--panel:#1c2027;--ink:#e6e8ec;--muted:#9aa3b2;
--line:#2c323c;--accent:#7aa7ff;--ok:#5fd19c;--okbg:#173527;--warn:#e7c066;--warnbg:#3a3015;--bad:#ff8a80;--badbg:#3d1f1d;--na:#9aa3b2;--nabg:#262b33}}
:root[data-theme="dark"]{--bg:#14171c;--panel:#1c2027;--ink:#e6e8ec;--muted:#9aa3b2;--line:#2c323c;--accent:#7aa7ff;
--ok:#5fd19c;--okbg:#173527;--warn:#e7c066;--warnbg:#3a3015;--bad:#ff8a80;--badbg:#3d1f1d;--na:#9aa3b2;--nabg:#262b33}
*{box-sizing:border-box}body{margin:0;background:var(--bg);color:var(--ink);font:14px/1.5 system-ui,-apple-system,Segoe UI,Roboto,sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}h1{font-size:24px;margin:0 0 4px}h2{font-size:18px;margin:36px 0 8px}
h3{font-size:15px;margin:20px 0 6px}p{margin:6px 0}.muted{color:var(--muted)}
.tiles{display:grid;grid-template-columns:repeat(auto-fit,minmax(170px,1fr));gap:10px;margin:16px 0}
.tile{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px}
.tile .v{font-size:22px;font-weight:600}.tile .l{color:var(--muted);font-size:12px}
.card{background:var(--panel);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto}
table{border-collapse:collapse;width:100%;font-size:12.5px}th,td{padding:5px 8px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top}
th{color:var(--muted);font-weight:600;position:sticky;top:0;background:var(--panel)}td.n{text-align:right;font-variant-numeric:tabular-nums}
.b{display:inline-block;padding:1px 7px;border-radius:10px;font-size:11.5px;font-weight:600;white-space:nowrap}
.b.ok{color:var(--ok);background:var(--okbg)}.b.warn{color:var(--warn);background:var(--warnbg)}
.b.bad{color:var(--bad);background:var(--badbg)}.b.na{color:var(--na);background:var(--nabg)}
svg text{fill:var(--muted);font-size:11px}svg .grid{stroke:var(--line)}.legend{display:flex;flex-wrap:wrap;gap:12px;font-size:12px;margin-top:4px}
.legend span::before{content:"";display:inline-block;width:10px;height:10px;border-radius:2px;margin-right:5px;background:var(--c)}
code{font-size:12px}
"""

STATUS_CLASS = {"data": "ok", "docs": "ok", "secondary": "warn", "unverified": "warn", "mismatch": "bad", "no data": "na"}


def esc(x) -> str:
    if x is None or (isinstance(x, float) and math.isnan(x)):
        return ""
    return html.escape(str(x))


def badge(status: str) -> str:
    return f'<span class="b {STATUS_CLASS.get(status, "na")}">{esc(status)}</span>'


def table(df: pd.DataFrame, fmt: dict | None = None, raw_cols: set | None = None, max_rows: int = 500) -> str:
    if df is None or df.empty:
        return '<p class="muted">No data.</p>'
    fmt = fmt or {}
    raw_cols = raw_cols or set()
    head = "".join(f"<th>{esc(c)}</th>" for c in df.columns)
    body = []
    for _, r in df.head(max_rows).iterrows():
        cells = []
        for c in df.columns:
            v = r[c]
            if c in raw_cols:
                cells.append(f"<td>{v}</td>")
            elif isinstance(v, (int, float)) and not isinstance(v, bool):
                f = fmt.get(c, "{:,.2f}" if isinstance(v, float) else "{:,}")
                cells.append(f'<td class="n">{"" if (isinstance(v, float) and math.isnan(v)) else f.format(v)}</td>')
            else:
                cells.append(f"<td>{esc(v)}</td>")
        body.append("<tr>" + "".join(cells) + "</tr>")
    more = f'<p class="muted">{len(df) - max_rows} more rows not shown.</p>' if len(df) > max_rows else ""
    return f'<div class="card"><table><thead><tr>{head}</tr></thead><tbody>{"".join(body)}</tbody></table></div>{more}'


def line_chart(data: pd.DataFrame, x: str, series: list[str], y_label: str = "", width: int = 1100, height: int = 300) -> str:
    """Inline SVG line chart; x is categorical (e.g. months), one line per column in `series`."""
    series = [s for s in series if s in data.columns]
    if data.empty or not series:
        return '<p class="muted">No data.</p>'
    xs = list(data[x])
    vals = data[series].to_numpy(dtype=float)
    finite = vals[~pd.isna(vals)]
    if finite.size == 0:
        return '<p class="muted">No data.</p>'
    lo, hi = float(min(0.0, finite.min())), float(finite.max())
    hi = hi if hi > lo else lo + 1
    ml, mr, mt, mb = 56, 12, 12, 40
    W, H = width - ml - mr, height - mt - mb
    px = lambda i: ml + (W * i / max(1, len(xs) - 1))
    py = lambda v: mt + H - H * (v - lo) / (hi - lo)
    parts = [f'<svg viewBox="0 0 {width} {height}" width="100%" role="img" aria-label="{esc(y_label)}">']
    for k in range(5):
        v = lo + (hi - lo) * k / 4
        y = py(v)
        parts.append(f'<line class="grid" x1="{ml}" x2="{width - mr}" y1="{y:.1f}" y2="{y:.1f}"/>'
                     f'<text x="{ml - 6}" y="{y + 4:.1f}" text-anchor="end">{v:,.0f}</text>')
    step = max(1, len(xs) // 12)
    for i in range(0, len(xs), step):
        parts.append(f'<text x="{px(i):.1f}" y="{height - 18}" text-anchor="middle">{esc(xs[i])}</text>')
    parts.append(f'<text x="12" y="{mt + H / 2:.0f}" transform="rotate(-90 12 {mt + H / 2:.0f})" text-anchor="middle">{esc(y_label)}</text>')
    for j, s in enumerate(series):
        color = PALETTE[j % len(PALETTE)]
        pts, segs = [], []
        for i, v in enumerate(vals[:, j]):
            if pd.isna(v):
                if pts:
                    segs.append(pts)
                pts = []
            else:
                pts.append(f"{px(i):.1f},{py(v):.1f}")
        if pts:
            segs.append(pts)
        for seg in segs:
            if len(seg) == 1:
                cx, cy = seg[0].split(",")
                parts.append(f'<circle cx="{cx}" cy="{cy}" r="2.5" fill="{color}"/>')
            else:
                parts.append(f'<polyline fill="none" stroke="{color}" stroke-width="2" points="{" ".join(seg)}"/>')
    parts.append("</svg>")
    legend = "".join(f'<span style="--c:{PALETTE[j % len(PALETTE)]}">{esc(s)}</span>' for j, s in enumerate(series))
    return f'<div class="card">{"".join(parts)}<div class="legend">{legend}</div></div>'


def page(title: str, body: str) -> str:
    return (f'<!doctype html><html lang="en"><head><meta charset="utf-8">'
            f'<meta name="viewport" content="width=device-width,initial-scale=1"><title>{esc(title)}</title>'
            f"<style>{CSS}</style></head><body><main>{body}</main></body></html>")


def _completeness_badge(p) -> str:
    if p is None or (isinstance(p, float) and math.isnan(p)):
        return badge("no data")
    cls = "ok" if p >= 99.5 else ("warn" if p >= 95 else "bad")
    return f'<span class="b {cls}">{p:.2f}%</span>'


def build_quality_report(store: Store, cfg: Config, out_path: Path, esios_client=None) -> Path:
    summary = timeline.series_summary(store)
    checks = quality.run_checks(store, cfg)
    ver = verify.run_verifications(store, cfg, esios_client)
    avail = timeline.gate_availability(store, cfg)
    months = quality.monthly_stats(store, cfg)
    afrr_periods = quality.by_period(store, cfg, ["afrr_band_price", "afrr_band_price_up", "afrr_band_price_down",
                                                 "afrr_energy_price_up", "afrr_energy_price_down"])
    sched = timeline.gate_schedule(cfg, date.today())

    b = []
    b.append(f"<h1>{esc(cfg['project']['name'])} &middot; Phase 1 data quality</h1>")
    b.append(f'<p class="muted">Generated {pd.Timestamp.now(tz=cfg.tz):%Y-%m-%d %H:%M %Z} &middot; '
             f"store <code>{esc(store.root)}</code> &middot; requested window {cfg.data_start()} to {cfg.data_end()}</p>")
    n_series = len(summary)
    srcs = ", ".join(sorted(summary["source"].unique())) if n_series else "none"
    med = checks["completeness_pct"].median() if not checks.empty else float("nan")
    vc = ver["status"].value_counts().to_dict()
    tiles = [("Series", f"{n_series}"), ("Sources", srcs), ("Median completeness", "" if math.isnan(med) else f"{med:.2f}%"),
             ("Verified (data/docs)", f"{vc.get('data', 0) + vc.get('docs', 0)}"),
             ("Mismatches", f"{vc.get('mismatch', 0)}"), ("Open items", f"{vc.get('unverified', 0) + vc.get('secondary', 0) + vc.get('no data', 0)}")]
    b.append('<div class="tiles">' + "".join(f'<div class="tile"><div class="v">{esc(v)}</div><div class="l">{esc(l)}</div></div>' for l, v in tiles) + "</div>")
    if n_series == 0:
        b.append('<div class="card"><p><b>The store is empty.</b> Run <code>bess download all</code> first. '
                 "Verification rows below that need data show <i>no data</i>.</p></div>")

    b.append("<h2>1. Phase-1 verifications</h2>")
    b.append('<p class="muted">Market facts to verify. <i>data</i> = established from the downloaded files; '
             "<i>docs</i> = official document; <i>secondary</i> = trade source only; <i>mismatch</i> = config must be corrected.</p>")
    v = ver.copy()
    v["status"] = v["status"].map(badge)
    b.append(table(v[["item", "question", "finding", "status", "evidence"]], raw_cols={"status"}))

    b.append("<h2>2. Decision timeline</h2>")
    b.append('<p class="muted">Gate and results times per delivery day D (local time, from config). Results times are '
             "used as publication times and are set conservatively late.</p>")
    s = sched.copy()
    s["gate"] = [f"D{cfg['markets'][m]['gate']['day_offset']:+d} {cfg['markets'][m]['gate']['time']}" for m in s["market"]]
    s["results"] = [f"D{cfg['markets'][m]['results']['day_offset']:+d} {cfg['markets'][m]['results']['time']}" for m in s["market"]]
    b.append(table(s[["market", "label", "gate", "results", "covers"]]))
    b.append("<h3>What is known at each gate</h3>")
    b.append('<p class="muted">Median reach of published data, in hours from D 00:00 local (+24 = the whole of D is known; '
             "negative = only data up to that point before D). Share = fraction of sampled days where all of D is known. "
             "Derived from every record's <code>published_at</code>; this is the no-lookahead contract for features.</p>")
    if not avail.empty:
        piv = avail.pivot(index="series", columns="gate", values="horizon_h_median")
        share = avail.pivot(index="series", columns="gate", values="full_day_share")
        cols = [g for g in timeline.GATES if g in piv.columns]
        out = pd.DataFrame(index=piv.index)
        for g in cols:
            out[g] = [f"{h:+.1f} h ({sh:.0%})" if pd.notna(h) else "" for h, sh in zip(piv[g], share[g])]
        b.append(table(out.reset_index()))
    else:
        b.append('<p class="muted">No data.</p>')
    assumed = summary[summary["pub_sources"].str.contains("assumed", na=False)]["series"].tolist() if n_series else []
    if assumed:
        b.append(f'<p><span class="b warn">assumption</span> Publication times of {len(assumed)} series come from '
                 f"undocumented rules (<code>rule:assumed</code>): {esc(', '.join(assumed[:30]))}{' ...' if len(assumed) > 30 else ''}. "
                 "Confirm before relying on intraday-timing features from them.</p>")

    b.append("<h2>3. Series quality</h2>")
    b.append('<p class="muted">Completeness counts expected periods inside each market\'s coverage window '
             "(IDA3 = 12:00-24:00) at that day's resolution, DST-aware. Nothing extreme is dropped: limits, outliers "
             "(|robust z| &gt; 8 within the month) and flat runs are flags for review.</p>")
    if not checks.empty:
        c = checks.copy()
        c["completeness"] = c["completeness_pct"].map(_completeness_badge)
        cols = ["series", "unit", "first_day", "last_day", "completeness", "missing_days", "missing_day_ranges",
                "partial_days", "resolution_changes", "revised_periods", "out_of_limits", "at_limit", "outliers_z8",
                "longest_flat_h", "min", "mean", "max"]
        for ds, g in c.groupby("dataset"):
            b.append(f"<h3>{esc(ds)}</h3>")
            b.append(table(g[cols], raw_cols={"completeness"}))
    else:
        b.append('<p class="muted">No data.</p>')

    b.append("<h2>4. Prices by month</h2>")
    if not months.empty:
        mm = months.pivot(index="month", columns="series", values="mean").reset_index()
        b.append("<h3>Energy markets: monthly mean price (EUR/MWh)</h3>")
        b.append(line_chart(mm, "month", ["da_price_es", "ida1_price_es", "ida2_price_es", "ida3_price_es"], "EUR/MWh"))
        b.append("<h3>aFRR capacity: monthly mean price (EUR/MW per hour)</h3>")
        b.append(line_chart(mm, "month", ["afrr_band_price", "afrr_band_price_up", "afrr_band_price_down"], "EUR/MW/h"))
        b.append(table(months, fmt={"n": "{:,}"}))
    else:
        b.append('<p class="muted">No data.</p>')
    b.append("<h3>aFRR prices by period and regime</h3>")
    b.append('<p class="muted">aFRR capacity prices fell sharply (~26 to ~13 to ~9 EUR/MW/h), so every result is reported by period.</p>')
    b.append(table(afrr_periods.reset_index() if not afrr_periods.empty else afrr_periods))

    b.append("<h2>5. Source inventory</h2>")
    b.append(table(summary, fmt={"median_pub_lag_h": "{:+.1f}", "min_pub_lag_h": "{:+.1f}", "max_pub_lag_h": "{:+.1f}"}))
    b.append('<p class="muted">pub lag = published_at minus delivery_start (negative = published before delivery).</p>')

    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(page(f"{cfg['project']['name']} data quality", "\n".join(b)), encoding="utf-8")
    checks.to_csv(out_path.with_name("series_checks.csv"), index=False)
    ver.to_csv(out_path.with_name("verifications.csv"), index=False)
    return out_path
