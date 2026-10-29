"""Configuration loading."""
from __future__ import annotations

import copy
import os
from dataclasses import dataclass
from datetime import date
from pathlib import Path
from typing import Any

import yaml

DEFAULT_CONFIG = Path(__file__).parent / "config" / "default.yaml"

SECRET_ENV = {"esios": "ESIOS_API_KEY", "entsoe": "ENTSOE_API_KEY"}


def _deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for key, value in (override or {}).items():
        if isinstance(value, dict) and isinstance(out.get(key), dict):
            out[key] = _deep_merge(out[key], value)
        else:
            out[key] = copy.deepcopy(value)
    return out


@dataclass
class Config:
    raw: dict
    base_dir: Path

    def __getitem__(self, key: str) -> Any:
        return self.raw[key]

    def get(self, key: str, default: Any = None) -> Any:
        return self.raw.get(key, default)

    @property
    def tz(self) -> str:
        return self.raw["project"]["timezone"]

    def path(self, key: str) -> Path:
        p = Path(self.raw["paths"][key])
        return p if p.is_absolute() else (self.base_dir / p)

    @property
    def data_dir(self) -> Path:
        return self.path("data_dir")

    @property
    def report_dir(self) -> Path:
        return self.path("report_dir")

    def data_start(self) -> date:
        return date.fromisoformat(str(self.raw["data"]["start"]))

    def data_end(self) -> date:
        end = self.raw["data"].get("end")
        return date.fromisoformat(str(end)) if end else date.today()


def load_config(path: str | os.PathLike | None = None, base_dir: str | os.PathLike | None = None,
                overrides: dict | None = None) -> Config:
    with open(DEFAULT_CONFIG, encoding="utf-8") as f:
        raw = yaml.safe_load(f)
    if path:
        with open(path, encoding="utf-8") as f:
            raw = _deep_merge(raw, yaml.safe_load(f) or {})
    if overrides:
        raw = _deep_merge(raw, overrides)
    return Config(raw=raw, base_dir=Path(base_dir) if base_dir else Path.cwd())


def secret(source: str) -> str | None:
    """API key for `source` from the environment."""
    value = os.environ.get(SECRET_ENV[source], "").strip()
    return value or None
