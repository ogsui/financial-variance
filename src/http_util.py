"""HTTP helpers for SEC requests."""

from __future__ import annotations

import time

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

from config import SEC_USER_AGENT


def sec_session(user_agent: str | None = None) -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        backoff_factor=0.5,
        status_forcelist=(403, 429, 500, 502, 503, 504),
        allowed_methods=("GET",),
    )
    session.mount("https://", HTTPAdapter(max_retries=retry))
    session.headers.update({"User-Agent": user_agent or SEC_USER_AGENT, "Accept-Encoding": "gzip, deflate"})
    return session


def get_json(session: requests.Session, url: str, timeout: int = 20, pause_s: float = 0.15) -> dict:
    response = session.get(url, timeout=timeout)
    response.raise_for_status()
    time.sleep(pause_s)
    return response.json()
