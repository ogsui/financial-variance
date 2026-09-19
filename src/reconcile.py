"""Nearest-trading-day join of filings and prices."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np
import pandas as pd

from config import MAX_TOLERANCE_DAYS

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

KNOWN_FISCAL_YEAR_ENDS = {
    "AAPL": 9,
    "MSFT": 6,
    "AMZN": 12,
    "NVDA": 1,
    "WMT": 1,
}


def nearest_price_row(
    price_dates: np.ndarray,
    price_records: pd.DataFrame,
    target_date: pd.Timestamp,
    max_days: int,
) -> Tuple[Optional[pd.Timestamp], Optional[pd.Series], int]:
    diffs = np.abs((price_dates - np.datetime64(target_date)).astype("timedelta64[D]").astype(int))
    min_idx = int(np.argmin(diffs))
    min_diff = int(diffs[min_idx])
    closest = pd.to_datetime(price_dates[min_idx])
    if min_diff > max_days:
        return None, None, min_diff
    return closest, price_records.loc[closest], min_diff


class MultiSourceReconciler:
    def __init__(
        self,
        max_tolerance_days: int = MAX_TOLERANCE_DAYS,
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
        if ticker in KNOWN_FISCAL_YEAR_ENDS:
            month = KNOWN_FISCAL_YEAR_ENDS[ticker]
            desc = "Calendar year (Dec)" if month == 12 else f"Fiscal year end month {month}"
            return desc, month

        q4_rows = df_fin[df_fin["fp"].isin(["Q4", "FY"])]
        if not q4_rows.empty:
            mode_month = int(q4_rows["period_end_date"].dt.month.mode()[0])
            desc = "Calendar year (Dec)" if mode_month == 12 else f"Fiscal year end month {mode_month}"
            return desc, mode_month
        return "Unknown", 12

    def reconcile_company(
        self,
        ticker: str,
        df_fin: pd.DataFrame,
        df_prc: pd.DataFrame,
    ) -> pd.DataFrame:
        df_fin_t = df_fin[df_fin["ticker"] == ticker].copy().sort_values("period_end_date")
        df_prc_t = df_prc[df_prc["ticker"] == ticker].copy().sort_values("date")

        if df_fin_t.empty or df_prc_t.empty:
            logger.warning("Missing financials or prices for %s", ticker)
            return pd.DataFrame()

        fiscal_desc, fy_end_month = self._determine_fiscal_calendar_type(ticker, df_fin_t)
        price_dates = df_prc_t["date"].values
        price_records = df_prc_t.set_index("date")

        reconciled_rows = []
        matched_count = 0
        unmatched_count = 0
        dropped_invalid = 0
        date_offsets = []

        for _, frow in df_fin_t.iterrows():
            revenue = frow.get("revenue")
            if pd.isna(revenue) or float(revenue) <= 0:
                dropped_invalid += 1
                logger.warning(
                    "[%s] Dropping period %s: non-positive revenue (%s)",
                    ticker,
                    frow["period_end_date"],
                    revenue,
                )
                continue

            target_date = frow["period_end_date"]
            closest_pdate, matched_price_row, min_diff = nearest_price_row(
                price_dates, price_records, target_date, self.max_tolerance_days
            )

            filed_raw = frow.get("filed_date")
            filed_ts = pd.to_datetime(filed_raw) if pd.notna(filed_raw) else None
            filing_price_date, filing_price_row, filing_off = (None, None, None)
            if filed_ts is not None:
                filing_price_date, filing_price_row, filing_off = nearest_price_row(
                    price_dates, price_records, filed_ts, self.max_tolerance_days
                )

            status = "UNMATCHED"
            if closest_pdate is not None and matched_price_row is not None:
                matched_count += 1
                date_offsets.append(min_diff)
                status = "MATCHED_EXACT" if min_diff == 0 else "MATCHED_TOLERANCE"
            else:
                unmatched_count += 1
                logger.warning(
                    "[%s] Period end %s unmatched within %s days (min diff %s).",
                    ticker,
                    target_date.strftime("%Y-%m-%d"),
                    self.max_tolerance_days,
                    min_diff,
                )

            def _px(row, col):
                if row is None:
                    return np.nan
                val = row[col]
                if isinstance(val, pd.Series):
                    val = val.iloc[0]
                return float(val)

            reconciled_rows.append({
                "ticker": ticker,
                "cik": frow["cik"],
                "period_end_date": target_date.strftime("%Y-%m-%d"),
                "filed_date": filed_ts.strftime("%Y-%m-%d") if filed_ts is not None else None,
                "fiscal_year": int(frow["fy"]) if pd.notna(frow["fy"]) else None,
                "fiscal_period": frow["fp"],
                "calendar_frame": frow["calendar_frame"],
                "fiscal_calendar_type": fiscal_desc,
                "fiscal_year_end_month": fy_end_month,
                "revenue": float(revenue),
                "net_income": float(frow["net_income"]) if pd.notna(frow["net_income"]) else np.nan,
                "opex": float(frow["opex"]) if pd.notna(frow["opex"]) else np.nan,
                "price_date": closest_pdate.strftime("%Y-%m-%d") if closest_pdate is not None else None,
                "stock_price": _px(matched_price_row, "Close"),
                "adj_close": _px(matched_price_row, "Adj Close"),
                "volume": _px(matched_price_row, "Volume"),
                "filing_price_date": filing_price_date.strftime("%Y-%m-%d") if filing_price_date is not None else None,
                "stock_price_at_filing": _px(filing_price_row, "Close"),
                "date_offset_days": min_diff,
                "filing_offset_days": filing_off,
                "reconciliation_status": status,
            })

        total_rows = matched_count + unmatched_count
        match_rate = (matched_count / total_rows * 100.0) if total_rows > 0 else 0.0
        avg_offset = float(np.mean(date_offsets)) if date_offsets else 0.0
        self.reconciliation_metrics[ticker] = {
            "total_financial_periods": total_rows,
            "matched_periods": matched_count,
            "unmatched_periods": unmatched_count,
            "dropped_invalid_revenue": dropped_invalid,
            "match_rate_pct": round(match_rate, 2),
            "avg_offset_calendar_days": round(avg_offset, 2),
            "fiscal_calendar_type": fiscal_desc,
        }
        logger.info(
            "[%s] Reconciled %s/%s periods (%.1f%% match, avg offset %.2f days, dropped %s invalid).",
            ticker, matched_count, total_rows, match_rate, avg_offset, dropped_invalid,
        )
        return pd.DataFrame(reconciled_rows)

    def reconcile_all(self) -> pd.DataFrame:
        if not self.financials_path.exists():
            raise FileNotFoundError(f"Missing financials file: {self.financials_path}")
        if not self.prices_path.exists():
            raise FileNotFoundError(f"Missing prices file: {self.prices_path}")

        df_fin = pd.read_csv(self.financials_path)
        df_prc = pd.read_csv(self.prices_path)
        df_fin["period_end_date"] = pd.to_datetime(df_fin["period_end_date"])
        df_prc["date"] = pd.to_datetime(df_prc["date"])

        all_reconciled = []
        for ticker in df_fin["ticker"].unique():
            rec_df = self.reconcile_company(ticker, df_fin, df_prc)
            if not rec_df.empty:
                all_reconciled.append(rec_df)

        if not all_reconciled:
            raise RuntimeError("Reconciliation produced no valid rows.")

        consolidated = pd.concat(all_reconciled, ignore_index=True)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        consolidated_path = self.output_dir / "consolidated.csv"
        consolidated.to_csv(consolidated_path, index=False)

        total_recs = len(consolidated)
        total_matched = int((consolidated["reconciliation_status"] != "UNMATCHED").sum())
        overall_match_rate = round(total_matched / total_recs * 100.0, 2) if total_recs else 0.0
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
        logger.info(
            "Saved %s consolidated rows (%.2f%% matched) to %s",
            total_recs, overall_match_rate, consolidated_path.resolve(),
        )
        return consolidated
