"""Yahoo Finance Market Data Ingestion Module.

Fetches historical daily market prices (Close, Adj Close, Volume, High, Low, Open)
via yfinance, resamples to quarter-end values, and saves to data/raw/prices_raw.csv.
Also retains daily granularity for nearest-trading-day tolerance reconciliation.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List
import pandas as pd
import yfinance as yf

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class PriceIngester:
    """Ingests historical market price data using Yahoo Finance."""

    def __init__(
        self,
        start_date: str = "2019-01-01",
        end_date: str = "2026-01-01",
    ):
        self.start_date = start_date
        self.end_date = end_date

    def fetch_daily_prices(self, ticker: str) -> pd.DataFrame:
        """Downloads daily OHLCV data for a ticker."""
        logger.info(f"Downloading daily market prices for {ticker} ({self.start_date} to {self.end_date})...")
        df = yf.download(
            ticker,
            start=self.start_date,
            end=self.end_date,
            interval="1d",
            progress=False,
            auto_adjust=False,
        )

        if df.empty:
            raise ValueError(f"No price data returned for ticker {ticker}")

        # Flatten multi-level columns if returned by yfinance
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = [c[0] for c in df.columns]

        df = df.reset_index()
        df.rename(columns={"Date": "date"}, inplace=True)
        df["date"] = pd.to_datetime(df["date"]).dt.tz_localize(None)
        df["ticker"] = ticker.upper()

        # Resample to quarter-ends for standard reference
        df_sorted = df.set_index("date").sort_index()
        try:
            qe_series = df_sorted["Close"].resample("QE").last()
        except ValueError:
            qe_series = df_sorted["Close"].resample("Q").last()

        df_sorted["quarter_end_resampled_close"] = qe_series.reindex(df_sorted.index, method="bfill")
        df_clean = df_sorted.reset_index()

        logger.info(f"Downloaded {len(df_clean)} trading days for {ticker}.")
        return df_clean

    def ingest_tickers(
        self,
        tickers: List[str],
        output_csv: Path | str = "data/raw/prices_raw.csv",
    ) -> pd.DataFrame:
        """Downloads daily prices for all tickers and writes combined raw prices CSV."""
        dfs = []
        for ticker in tickers:
            try:
                t_df = self.fetch_daily_prices(ticker)
                dfs.append(t_df)
            except Exception as e:
                logger.error(f"Failed downloading prices for {ticker}: {e}")

        if not dfs:
            raise RuntimeError("No market prices could be fetched.")

        combined = pd.concat(dfs, ignore_index=True)
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined.to_csv(out_path, index=False)
        logger.info(f"Saved {len(combined)} daily price records to {out_path.resolve()}")
        return combined


if __name__ == "__main__":
    target_tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "WMT"]
    ingester = PriceIngester()
    prices_df = ingester.ingest_tickers(target_tickers)
    print("\nPrices Raw Sample:")
    print(prices_df[["ticker", "date", "Close", "Adj Close", "Volume"]].head(10))
