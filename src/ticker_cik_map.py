"""Ticker to CIK Mapping Module.

Fetches the official SEC company tickers directory and provides lookup utilities
between stock tickers and 10-digit zero-padded SEC Central Index Key (CIK) identifiers.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional
import requests

SEC_TICKERS_URL = "https://www.sec.gov/files/company_tickers.json"
DEFAULT_USER_AGENT = "Ojas Gupta ojasgupta298@gmail.com"


class TickerCIKMapper:
    """Utility class to resolve stock tickers to 10-digit SEC CIK numbers."""

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        cache_file: Optional[Path | str] = None,
    ):
        self.user_agent = user_agent
        self.cache_file = Path(cache_file) if cache_file else None
        self._ticker_to_cik: Dict[str, str] = {}
        self._ticker_to_title: Dict[str, str] = {}
        self._loaded = False

    def load(self, force_refresh: bool = False) -> TickerCIKMapper:
        """Loads ticker-to-CIK mapping from local cache or SEC endpoint."""
        if self._loaded and not force_refresh:
            return self

        if self.cache_file and self.cache_file.exists() and not force_refresh:
            try:
                with open(self.cache_file, "r", encoding="utf-8") as f:
                    cached_data = json.load(f)
                    self._ticker_to_cik = cached_data.get("ticker_to_cik", {})
                    self._ticker_to_title = cached_data.get("ticker_to_title", {})
                    self._loaded = True
                    return self
            except Exception:
                pass

        headers = {"User-Agent": self.user_agent}
        response = requests.get(SEC_TICKERS_URL, headers=headers, timeout=15)
        response.raise_for_status()
        raw_data = response.json()

        # SEC returns dictionary of entries: {"0": {"cik_str": 320193, "ticker": "AAPL", "title": "Apple Inc."}, ...}
        self._ticker_to_cik = {}
        self._ticker_to_title = {}
        for item in raw_data.values():
            ticker = str(item.get("ticker", "")).strip().upper()
            cik = str(item.get("cik_str", "")).strip().zfill(10)
            title = str(item.get("title", "")).strip()
            if ticker and cik:
                self._ticker_to_cik[ticker] = cik
                self._ticker_to_title[ticker] = title

        if self.cache_file:
            self.cache_file.parent.mkdir(parents=True, exist_ok=True)
            with open(self.cache_file, "w", encoding="utf-8") as f:
                json.dump(
                    {
                        "ticker_to_cik": self._ticker_to_cik,
                        "ticker_to_title": self._ticker_to_title,
                    },
                    f,
                    indent=2,
                )

        self._loaded = True
        return self

    def get_cik(self, ticker: str) -> Optional[str]:
        """Returns 10-digit CIK for given ticker symbol."""
        if not self._loaded:
            self.load()
        return self._ticker_to_cik.get(ticker.strip().upper())

    def get_title(self, ticker: str) -> Optional[str]:
        """Returns corporate entity title for given ticker symbol."""
        if not self._loaded:
            self.load()
        return self._ticker_to_title.get(ticker.strip().upper())

    def get_all_mappings(self) -> Dict[str, str]:
        """Returns full ticker -> CIK dictionary."""
        if not self._loaded:
            self.load()
        return dict(self._ticker_to_cik)


def build_ticker_to_cik_map(
    tickers: Optional[list[str]] = None,
    cache_path: Optional[str | Path] = None,
    user_agent: str = DEFAULT_USER_AGENT,
) -> Dict[str, str]:
    """Helper function to get mapping for a list of tickers."""
    mapper = TickerCIKMapper(user_agent=user_agent, cache_file=cache_path).load()
    if tickers is None:
        return mapper.get_all_mappings()
    return {t.upper(): mapper.get_cik(t) for t in tickers if mapper.get_cik(t)}


if __name__ == "__main__":
    sample_tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "WMT"]
    print("Testing SEC Ticker-to-CIK Mapper...")
    mapper = TickerCIKMapper()
    mapper.load()
    for t in sample_tickers:
        cik = mapper.get_cik(t)
        title = mapper.get_title(t)
        print(f"{t}: CIK={cik} ({title})")
