"""Shared HTTP session: retries with exponential backoff, timeouts, polite pacing."""
from __future__ import annotations

import logging
import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from ..config import Config

log = logging.getLogger(__name__)


class Http:
    def __init__(self, cfg: Config, headers: dict | None = None):
        h = cfg["data"]["http"]
        self.timeout = h["timeout_s"]
        self.delay = h["polite_delay_s"]
        self.session = requests.Session()
        retry = Retry(total=h["retries"], backoff_factor=h["backoff_s"],
                      status_forcelist=(429, 500, 502, 503, 504), allowed_methods=("GET",),
                      respect_retry_after_header=True)
        self.session.mount("https://", HTTPAdapter(max_retries=retry))
        self.session.headers["User-Agent"] = h["user_agent"]
        if headers:
            self.session.headers.update(headers)
        self._last = 0.0

    def get(self, url: str, params: dict | list | None = None, **kw) -> requests.Response:
        wait = self.delay - (time.monotonic() - self._last)
        if wait > 0:
            time.sleep(wait)
        try:
            return self.session.get(url, params=params, timeout=self.timeout, **kw)
        finally:
            self._last = time.monotonic()


def check_reachable(url: str, timeout: float = 20) -> tuple[bool, str]:
    """Network probe used by `bess check-network`. Any HTTP answer counts as reachable."""
    try:
        r = requests.get(url, timeout=timeout)
        return True, f"HTTP {r.status_code}"
    except requests.exceptions.ProxyError as e:
        return False, f"blocked by proxy ({e.__class__.__name__}): host not in the allowed network list?"
    except requests.RequestException as e:
        return False, f"{e.__class__.__name__}: {e}"
