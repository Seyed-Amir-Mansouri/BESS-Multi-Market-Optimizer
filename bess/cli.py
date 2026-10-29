"""Command-line interface."""
from __future__ import annotations

import argparse
import logging
import sys
from datetime import date

import pandas as pd

from .config import load_config, secret

log = logging.getLogger("bess")

SOURCES = ["omie", "esios", "entsoe", "weather"]
PROBES = {
    "omie": "https://www.omie.es/es/file-download?parents=marginalpdbc&filename=marginalpdbc_20250101.1",
    "esios": "https://api.esios.ree.es/indicators",
    "entsoe": "https://web-api.tp.entsoe.eu/api",
    "weather": "https://previous-runs-api.open-meteo.com/v1/forecast?latitude=40.4&longitude=-3.7&hourly=temperature_2m_previous_day1&past_days=1&forecast_days=1",
}


def _store(cfg):
    from .data.store import Store
    return Store(cfg.data_dir)


def _dates(cfg, args):
    start = date.fromisoformat(args.start) if getattr(args, "start", None) else cfg.data_start()
    end = date.fromisoformat(args.end) if getattr(args, "end", None) else cfg.data_end()
    return start, end


def cmd_check_network(cfg, args):
    from .data.http import check_reachable
    ok_all = True
    for name, url in PROBES.items():
        ok, msg = check_reachable(url)
        ok_all &= ok
        print(f"{name:8s} {'OK ' if ok else 'FAIL'}  {msg}")
    for src, var in (("esios", "ESIOS_API_KEY"), ("entsoe", "ENTSOE_API_KEY")):
        print(f"{var:15s} {'set' if secret(src) else 'NOT SET'}")
    return 0 if ok_all else 1


def cmd_download(cfg, args):
    store = _store(cfg)
    start, end = _dates(cfg, args)
    sources = SOURCES if args.source == "all" else [args.source]
    rc = 0
    for src in sources:
        print(f"== {src}: {start} .. {end}")
        try:
            if src == "omie":
                from .data.omie import OmieDownloader
                counts = OmieDownloader(cfg, store).download(start, end, force=args.force)
            elif src == "esios":
                from .data.esios import EsiosClient
                client = EsiosClient(cfg, store)
                client.catalog()
                counts = client.download(start, end, force=args.force)
            elif src == "entsoe":
                from .data.entsoe import EntsoeDownloader
                counts = EntsoeDownloader(cfg, store).download(start, end, force=args.force)
            else:
                from .data.weather import WeatherDownloader
                counts = WeatherDownloader(cfg, store).download(start, end, force=args.force)
            for k, v in counts.items():
                print(f"   {k:32s} {v:>10,} rows in store")
        except Exception as e:
            log.error("%s download failed: %s", src, e)
            rc = 1
    return rc


def cmd_parse(cfg, args):
    store = _store(cfg)
    start, end = _dates(cfg, args)
    sources = SOURCES if args.source == "all" else [args.source]
    for src in sources:
        if src == "omie":
            from .data.omie import OmieDownloader
            counts = OmieDownloader(cfg, store).parse(start, end)
        elif src == "esios":
            from .data.esios import EsiosClient
            counts = EsiosClient(cfg, store, offline=True).parse(start, end)
        elif src == "entsoe":
            from .data.entsoe import EntsoeDownloader
            counts = EntsoeDownloader(cfg, store, offline=True).parse()
        else:
            from .data.weather import WeatherDownloader
            counts = WeatherDownloader(cfg, store).parse()
        for k, v in counts.items():
            print(f"{src:8s} {k:32s} {v:>10,}")
    return 0


def cmd_esios_catalog(cfg, args):
    from .data.esios import EsiosClient, strip_accents
    store = _store(cfg)
    cat = store.read_events("esios_catalog")
    if cat.empty or args.refresh:
        cat = EsiosClient(cfg, store).catalog()
    if args.search:
        words = [strip_accents(w) for w in args.search.split()]
        names = cat["name"].astype(str).map(strip_accents)
        cat = cat[names.map(lambda n: all(w in n for w in words))]
    with pd.option_context("display.max_rows", 500, "display.max_colwidth", 110, "display.width", 200):
        print(cat[["id", "name"]].sort_values("id").to_string(index=False))
    return 0


def cmd_calendar(cfg, args):
    from .data.calendar_es import build_calendar
    store = _store(cfg)
    start, end = _dates(cfg, args)
    cal = build_calendar(cfg, start, end)
    cal["day"] = cal["day"].astype(str)
    n = store.write_events("calendar", cal, key=["day"])
    print(f"calendar: {n} days stored; holidays in range:")
    print(cal[cal["is_holiday"]][["day", "holiday"]].to_string(index=False))
    return 0


def cmd_timeline(cfg, args):
    from .data import timeline
    store = _store(cfg)
    print(timeline.gate_schedule(cfg, date.today()).to_string(index=False))
    avail = timeline.gate_availability(store, cfg)
    if avail.empty:
        print("\nstore is empty - nothing to place on the timeline yet")
        return 0
    store.write_events("gate_availability", avail)
    piv = avail.pivot(index="series", columns="gate", values="horizon_h_median")
    print("\nmedian data reach at each gate (hours from D 00:00 local; +24 = all of D known):")
    print(piv[[g for g in timeline.GATES if g in piv.columns]].round(1).to_string())
    return 0


def _esios_client_or_none(cfg, store):
    if not secret("esios"):
        return None
    try:
        from .data.esios import EsiosClient
        return EsiosClient(cfg, store)
    except Exception:
        return None


def cmd_verify(cfg, args):
    from .data.verify import run_verifications
    store = _store(cfg)
    client = None if args.offline else _esios_client_or_none(cfg, store)
    ver = run_verifications(store, cfg, client)
    with pd.option_context("display.max_rows", 500, "display.max_colwidth", 120, "display.width", 250):
        print(ver[["item", "status", "finding"]].to_string(index=False))
    return 0


def cmd_quality(cfg, args):
    from .report import build_quality_report
    store = _store(cfg)
    out = args.out or (cfg.report_dir / "data_quality.html")
    client = None if args.offline else _esios_client_or_none(cfg, store)
    path = build_quality_report(store, cfg, out, client)
    print(f"report written: {path}")
    return 0


def cmd_status(cfg, args):
    from .data.timeline import series_summary
    store = _store(cfg)
    print(f"store: {store.root}")
    s = series_summary(store)
    if s.empty:
        print("empty")
        return 0
    with pd.option_context("display.max_rows", 500, "display.width", 250):
        print(s[["source", "series", "unit", "first_delivery", "last_delivery", "n_records", "resolutions_min"]].to_string(index=False))
    return 0


def cmd_export_duckdb(cfg, args):
    store = _store(cfg)
    path = store.export_duckdb(args.path or (cfg.data_dir / "bess.duckdb"))
    print(f"DuckDB file with views `records` and `events_*`: {path}")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="bess", description="Spanish multi-market BESS optimization tool")
    p.add_argument("--config", help="YAML file merged over the default config")
    p.add_argument("-v", "--verbose", action="store_true")
    sub = p.add_subparsers(dest="cmd", required=True)

    sub.add_parser("check-network", help="probe the data hosts").set_defaults(fn=cmd_check_network)

    d = sub.add_parser("download", help="download and clean data")
    d.add_argument("source", nargs="?", default="all", choices=SOURCES + ["all"])
    d.add_argument("--start"); d.add_argument("--end")
    d.add_argument("--force", action="store_true", help="re-download files already cached")
    d.set_defaults(fn=cmd_download)

    pa = sub.add_parser("parse", help="rebuild clean records from cached raw files")
    pa.add_argument("source", nargs="?", default="all", choices=SOURCES + ["all"])
    pa.add_argument("--start"); pa.add_argument("--end")
    pa.set_defaults(fn=cmd_parse)

    c = sub.add_parser("esios-catalog", help="list/search eSIOS indicators")
    c.add_argument("--search"); c.add_argument("--refresh", action="store_true")
    c.set_defaults(fn=cmd_esios_catalog)

    ca = sub.add_parser("calendar", help="build the calendar table")
    ca.add_argument("--start"); ca.add_argument("--end")
    ca.set_defaults(fn=cmd_calendar)

    sub.add_parser("timeline", help="publication-time timeline per gate").set_defaults(fn=cmd_timeline)

    v = sub.add_parser("verify", help="Phase-1 verifications")
    v.add_argument("--offline", action="store_true", help="do not query eSIOS metadata")
    v.set_defaults(fn=cmd_verify)

    q = sub.add_parser("quality", help="write the HTML data quality report")
    q.add_argument("--out"); q.add_argument("--offline", action="store_true")
    q.set_defaults(fn=cmd_quality)

    sub.add_parser("status", help="store contents").set_defaults(fn=cmd_status)

    e = sub.add_parser("export-duckdb", help="write a persistent DuckDB file with views")
    e.add_argument("path", nargs="?")
    e.set_defaults(fn=cmd_export_duckdb)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    logging.basicConfig(level=logging.DEBUG if args.verbose else logging.INFO,
                        format="%(asctime)s %(levelname)-7s %(name)s: %(message)s", datefmt="%H:%M:%S")
    cfg = load_config(args.config)
    return args.fn(cfg, args) or 0


if __name__ == "__main__":
    sys.exit(main())
