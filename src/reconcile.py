"""Multi-Source Financial & Market Data Reconciliation Module.

Reconciles regulatory SEC EDGAR 10-Q/10-K financial statement periods with Yahoo Finance
market equity data. Implements:
1. Normalization to true period-end dates (avoiding calendar-label distortion).
2. Explicit detection and classification of fiscal calendar offsets vs standard calendar years.
3. Nearest-trading-day fuzzy tolerance matching (within +/- 4 calendar days).
4. Data quality and reconciliation audit logging (match rate, average offset, unmatched rows).
5. Generation of data/processed/consolidated.csv.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple
import numpy as np
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Expected fiscal year end month for target companies
KNOWN_FISCAL_YEAR_ENDS = {
    "AAPL": 9,   # September
    "MSFT": 6,   # June
    "AMZN": 12,  # December (Calendar Year)
    "NVDA": 1,   # January
    "WMT": 1,    # January
}


class MultiSourceReconciler:
    """Reconciles disparate financial reporting periods with continuous market trading prices."""

    def __init__(
        self,
        max_tolerance_days: int = 5,
        financials_path: Path | str = "data/raw/financials_raw.csv",
        prices_path: Path | str = "data/raw/prices_raw.csv",
        output_dir: Path | str = "data/processed",
    ):
        self.max_tolerance_days = max_tolerance_days
        self.financials_path = Path(financials_path)
        self.prices_path = Path(prices_path)
        self.output_dir = Path(output_dir)
        self.reconciliation_metrics: Dict[str, dict] = {}

    def _determine_fiscal_calendar_type(self, ticker: str, df_fin: pd.DataFrame) -> Tuple[str, int]:
        """Detects whether company uses non-standard fiscal calendar and identifies year-end month."""
        # Check known definitions first
        if ticker in KNOWN_FISCAL_YEAR_ENDS:
            month = KNOWN_FISCAL_YEAR_ENDS[ticker]
            is_cal = (month == 12)
            desc = "Calendar-Year Aligned (Dec)" if is_cal else f"Non-Calendar Fiscal Year (Month {month})"
            return desc, month

        # Fallback to empirical detection from period_end_date and fp
        q4_rows = df_fin[df_fin["fp"].isin(["Q4", "FY"])]
        if not q4_rows.empty:
            mode_month = int(q4_rows["period_end_date"].dt.month.mode()[0])
            is_cal = (mode_month == 12)
            desc = "Calendar-Year Aligned (Dec)" if is_cal else f"Non-Calendar Fiscal Year (Month {mode_month})"
            return desc, mode_month

        return "Unknown / Unclassified", 12

    def reconcile_company(
        self,
        ticker: str,
        df_fin: pd.DataFrame,
        df_prc: pd.DataFrame,
    ) -> pd.DataFrame:
        """Performs nearest-trading-day fuzzy merge for a single company."""
        df_fin_t = df_fin[df_fin["ticker"] == ticker].copy().sort_values("period_end_date")
        df_prc_t = df_prc[df_prc["ticker"] == ticker].copy().sort_values("date")

        if df_fin_t.empty:
            logger.warning(f"No financials found for {ticker}")
            return pd.DataFrame()
        if df_prc_t.empty:
            logger.warning(f"No price records found for {ticker}")
            return pd.DataFrame()

        fiscal_desc, fy_end_month = self._determine_fiscal_calendar_type(ticker, df_fin_t)

        reconciled_rows = []
        matched_count = 0
        unmatched_count = 0
        date_offsets = []

        price_dates = df_prc_t["date"].values
        price_records = df_prc_t.set_index("date")

        for _, frow in df_fin_t.iterrows():
            target_date = frow["period_end_date"]
            
            # Find closest market price date
            diffs = np.abs((price_dates - np.datetime64(target_date)).astype("timedelta64[D]").astype(int))
            min_idx = np.argmin(diffs)
            min_diff = int(diffs[min_idx])
            closest_pdate = pd.to_datetime(price_dates[min_idx])

            if min_diff <= self.max_tolerance_days:
                matched_count += 1
                date_offsets.append(min_diff)
                matched_price_row = price_records.loc[closest_pdate]

                reconciled_rows.append({
                    "ticker": ticker,
                    "cik": frow["cik"],
                    "period_end_date": target_date.strftime("%Y-%m-%d"),
                    "fiscal_year": int(frow["fy"]) if pd.notna(frow["fy"]) else None,
                    "fiscal_period": frow["fp"],
                    "calendar_frame": frow["calendar_frame"],
                    "fiscal_calendar_type": fiscal_desc,
                    "fiscal_year_end_month": fy_end_month,
                    "revenue": float(frow["revenue"]) if pd.notna(frow["revenue"]) else np.nan,
                    "net_income": float(frow["net_income"]) if pd.notna(frow["net_income"]) else np.nan,
                    "opex": float(frow["opex"]) if pd.notna(frow["opex"]) else np.nan,
                    "price_date": closest_pdate.strftime("%Y-%m-%d"),
                    "stock_price": float(matched_price_row["Close"]),
                    "adj_close": float(matched_price_row["Adj Close"]),
                    "volume": float(matched_price_row["Volume"]),
                    "date_offset_days": min_diff,
                    "reconciliation_status": "MATCHED_EXACT" if min_diff == 0 else "MATCHED_TOLERANCE",
                })
            else:
                unmatched_count += 1
                logger.warning(
                    f"[{ticker}] Period end {target_date.strftime('%Y-%m-%d')} failed to match market price "
                    f"within {self.max_tolerance_days} days (min diff: {min_diff} days)."
                )
                reconciled_rows.append({
                    "ticker": ticker,
                    "cik": frow["cik"],
                    "period_end_date": target_date.strftime("%Y-%m-%d"),
                    "fiscal_year": int(frow["fy"]) if pd.notna(frow["fy"]) else None,
                    "fiscal_period": frow["fp"],
                    "calendar_frame": frow["calendar_frame"],
                    "fiscal_calendar_type": fiscal_desc,
                    "fiscal_year_end_month": fy_end_month,
                    "revenue": float(frow["revenue"]) if pd.notna(frow["revenue"]) else np.nan,
                    "net_income": float(frow["net_income"]) if pd.notna(frow["net_income"]) else np.nan,
                    "opex": float(frow["opex"]) if pd.notna(frow["opex"]) else np.nan,
                    "price_date": None,
                    "stock_price": np.nan,
                    "adj_close": np.nan,
                    "volume": np.nan,
                    "date_offset_days": min_diff,
                    "reconciliation_status": "UNMATCHED",
                })

        total_rows = matched_count + unmatched_count
        match_rate = (matched_count / total_rows * 100.0) if total_rows > 0 else 0.0
        avg_offset = float(np.mean(date_offsets)) if date_offsets else 0.0

        self.reconciliation_metrics[ticker] = {
            "total_financial_periods": total_rows,
            "matched_periods": matched_count,
            "unmatched_periods": unmatched_count,
            "match_rate_pct": round(match_rate, 2),
            "avg_offset_calendar_days": round(avg_offset, 2),
            "fiscal_calendar_type": fiscal_desc,
        }

        logger.info(
            f"[{ticker}] Reconciled {matched_count}/{total_rows} periods "
            f"({match_rate:.1f}% clean match, avg tolerance offset: {avg_offset:.2f} days)."
        )

        return pd.DataFrame(reconciled_rows)

    def reconcile_all(self) -> pd.DataFrame:
        """Reads raw datasets, executes reconciliation, generates quality reports, and outputs consolidated CSV."""
        if not self.financials_path.exists():
            raise FileNotFoundError(f"Missing financials file: {self.financials_path}")
        if not self.prices_path.exists():
            raise FileNotFoundError(f"Missing prices file: {self.prices_path}")

        df_fin = pd.read_csv(self.financials_path)
        df_prc = pd.read_csv(self.prices_path)

        df_fin["period_end_date"] = pd.to_datetime(df_fin["period_end_date"])
        df_prc["date"] = pd.to_datetime(df_prc["date"])

        tickers = df_fin["ticker"].unique()
        all_reconciled = []

        for ticker in tickers:
            rec_df = self.reconcile_company(ticker, df_fin, df_prc)
            if not rec_df.empty:
                all_reconciled.append(rec_df)

        if not all_reconciled:
            raise RuntimeError("Reconciliation produced no valid rows.")

        consolidated = pd.concat(all_reconciled, ignore_index=True)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        consolidated_path = self.output_dir / "consolidated.csv"
        consolidated.to_csv(consolidated_path, index=False)
        logger.info(f"Saved consolidated dataset ({len(consolidated)} rows) to {consolidated_path.resolve()}")

        # Overall summary metric
        total_recs = len(consolidated)
        total_matched = len(consolidated[consolidated["reconciliation_status"] != "UNMATCHED"])
        overall_match_rate = round(total_matched / total_recs * 100.0, 2)

        report = {
            "overall_summary": {
                "total_rows": total_recs,
                "total_matched": total_matched,
                "overall_match_rate_pct": overall_match_rate,
                "max_tolerance_window_days": self.max_tolerance_days,
            },
            "by_company": self.reconciliation_metrics,
        }

        report_path = self.output_dir / "reconciliation_report.json"
        with open(report_path, "w", encoding="utf-8") as f:
            json.dump(report, f, indent=2)
        logger.info(f"Saved reconciliation report to {report_path.resolve()}")

        return consolidated


if __name__ == "__main__":
    reconciler = MultiSourceReconciler()
    res = reconciler.reconcile_all()
    print("\nReconciliation Summary:")
    with open("data/processed/reconciliation_report.json", "r") as f:
        print(f.read())
