"""Yahoo Finance daily market data ingestion."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd
import yfinance as yf

from config import PRICE_START

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PriceIngester:
    def __init__(
        self,
        start_date: str = PRICE_START,
        end_date: str | None = None,
    ):
        self.start_date = start_date
        self.end_date = end_date

    def fetch_daily_prices(self, ticker: str) -> pd.DataFrame:
        logger.info("Downloading daily prices for %s (%s → %s)...", ticker, self.start_date, self.end_date or "today")
        kwargs = {
            "start": self.start_date,
            "interval": "1d",
            "progress": False,
            "auto_adjust": False,
        }
        if self.end_date:
            kwargs["end"] = self.end_date
        df = yf.download(ticker, **kwargs)

        if df.empty:
            raise ValueError(f"No price data returned for ticker {ticker}")

        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df = df.reset_index()
        df.rename(columns={"Date": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["ticker"] = ticker.upper()
        logger.info("Downloaded %s trading days for %s.", len(df), ticker)
        return df

    def ingest_tickers(
        self,
        tickers: List[str],
        output_csv: Path | str = "data/raw/prices_raw.csv",
    ) -> pd.DataFrame:
        dfs = []
        for ticker in tickers:
            try:
                dfs.append(self.fetch_daily_prices(ticker))
            except Exception as exc:
                logger.error("Failed downloading prices for %s: %s", ticker, exc)

        if not dfs:
            raise RuntimeError("No market prices could be fetched.")

        combined = pd.concat(dfs, ignore_index=True)
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)
        logger.info("Saved %s daily price records to %s", len(combined), out_path.resolve())
        return combined
