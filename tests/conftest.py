"""Shared fixtures."""
from __future__ import annotations

import io
import zipfile
from datetime import date

import pytest

from bess.config import load_config
from bess.data.store import Store
from bess.timeutils import periods_in_day

TZ = "Europe/Madrid"


@pytest.fixture
def cfg(tmp_path):
    return load_config(base_dir=tmp_path, overrides={"paths": {"data_dir": str(tmp_path / "store"),
                                                             "report_dir": str(tmp_path / "reports")}})


@pytest.fixture
def store(cfg):
    return Store(cfg.data_dir)


def marginal_text(day: date, header: str = "MARGINALPDBC", resolution: int | None = 60,
                  periods: range | None = None, price=lambda p: 50.0 + p) -> str:
    """OMIE marginal price file content (year;month;day;period;PT;ES;)."""
    if periods is None:
        periods = range(1, periods_in_day(day, resolution, TZ) + 1)
    lines = [f"{header};"]
    for p in periods:
        v = price(p)
        lines.append(f"{day.year};{day.month:02d};{day.day:02d};{p};{v:.2f};{v:.2f};")
    lines.append("*")
    return "\r\n".join(lines) + "\r\n"


class FakeResponse:
    def __init__(self, status_code=200, content=b"", text=None):
        self.status_code = status_code
        self.content = content
        self.text = text if text is not None else content.decode("latin-1")

    def json(self):
        import json
        return json.loads(self.text)


class FakeHttp:
    """Maps URL substrings to responses; records requested URLs."""

    def __init__(self, routes: dict[str, FakeResponse] | None = None):
        self.routes = routes or {}
        self.calls: list[str] = []

    def get(self, url, params=None, **kw):
        self.calls.append(url)
        for key, resp in self.routes.items():
            if key in url:
                return resp
        return FakeResponse(404, b"<html>not found</html>")


def zip_bytes(files: dict[str, str]) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w") as z:
        for name, content in files.items():
            z.writestr(name, content.encode("latin-1"))
    return buf.getvalue()
