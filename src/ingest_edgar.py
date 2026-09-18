"""SEC EDGAR Financials Ingestion Module.

Fetches quarterly financial statements (Revenue, Net Income, Operating Expenses)
from the SEC EDGAR companyfacts XBRL JSON API. Handles tag variation,
filters quarterly durations, reconstructs missing 10-K Q4 values where necessary,
and outputs data/raw/financials_raw.csv.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional
import pandas as pd
import requests

from ticker_cik_map import DEFAULT_USER_AGENT, TickerCIKMapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Primary and fallback XBRL tags across US-GAAP filings
TAG_CANDIDATES = {
    "revenue": [
        "RevenueFromContractWithCustomerExcludingAssessedTax",
        "Revenues",
        "SalesRevenueNet",
        "RevenueFromContractWithCustomerIncludingAssessedTax",
    ],
    "net_income": [
        "NetIncomeLoss",
        "ProfitLoss",
    ],
    "opex": [
        "OperatingExpenses",
        "CostsAndExpenses",
        "CostsAndExpensesOperating",
        "OperatingCostsAndExpenses",
        "SellingGeneralAndAdministrativeExpense",
        "OperatingIncomeLoss",  # Fallback for margin tracking
    ],
}


class EDGARIngester:
    """Ingests quarterly financial facts from SEC EDGAR API."""

    def __init__(
        self,
        user_agent: str = DEFAULT_USER_AGENT,
        start_date: str = "2019-01-01",
        end_date: str = "2026-01-01",
    ):
        self.user_agent = user_agent
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date)
        self.headers = {"User-Agent": self.user_agent}

    def fetch_company_facts(self, cik: str) -> dict:
        """Fetches raw JSON facts for a given 10-digit CIK."""
        cik_clean = str(cik).strip().zfill(10)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_clean}.json"
        response = requests.get(url, headers=self.headers, timeout=20)
        response.raise_for_status()
        return response.json()

    def _extract_metric_series(
        self,
        facts: dict,
        metric_name: str,
        tag_list: List[str],
    ) -> pd.DataFrame:
        """Extracts quarterly observations for a financial metric using candidate tags, choosing the tag with highest coverage."""
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        candidate_dfs = []

        for tag in tag_list:
            if tag in us_gaap and "units" in us_gaap[tag]:
                units_dict = us_gaap[tag]["units"]
                if "USD" in units_dict:
                    raw_df = pd.DataFrame(units_dict["USD"])
                    if raw_df.empty:
                        continue

                    raw_df["end"] = pd.to_datetime(raw_df["end"])
                    if "start" in raw_df.columns:
                        raw_df["start"] = pd.to_datetime(raw_df["start"])
                        raw_df["duration_days"] = (raw_df["end"] - raw_df["start"]).dt.days
                    else:
                        raw_df["start"] = pd.NaT
                        raw_df["duration_days"] = None

                    # Filter by overall date window
                    filtered_df = raw_df[
                        (raw_df["end"] >= self.start_date) & (raw_df["end"] <= self.end_date)
                    ].copy()

                    # Count standard quarterly observations
                    q_df = filtered_df[
                        (filtered_df["duration_days"] >= 70) & (filtered_df["duration_days"] <= 105)
                    ].copy()
                    
                    candidate_dfs.append((len(q_df.drop_duplicates("end")), tag, filtered_df))

        if not candidate_dfs:
            logger.warning(f"No matching tag found for metric '{metric_name}' in tags {tag_list}")
            return pd.DataFrame(columns=["end", metric_name, "form", "fp", "fy", "filed", "frame"])

        # Sort by number of quarterly observations descending
        candidate_dfs.sort(key=lambda x: x[0], reverse=True)
        best_count, matched_tag, units_df = candidate_dfs[0]

        # Isolate quarterly figures: duration between 70 and 105 days
        quarterly_df = units_df[
            (units_df["duration_days"] >= 70) & (units_df["duration_days"] <= 105)
        ].copy()

        # If some quarters (e.g. Q4 in 10-K) are reported only as FY (350-375 days) and 9M (260-285 days), derive Q4
        all_fy = units_df[(units_df["duration_days"] >= 350) & (units_df["duration_days"] <= 375)].copy()
        all_9m = units_df[(units_df["duration_days"] >= 260) & (units_df["duration_days"] <= 285)].copy()

        derived_q4_rows = []
        for fy_val in all_fy["fy"].dropna().unique():
            # Check if this fiscal year already has a Q4 in quarterly_df
            has_q4 = not quarterly_df[(quarterly_df["fy"] == fy_val) & (quarterly_df["fp"] == "Q4")].empty
            if not has_q4:
                fy_row = all_fy[all_fy["fy"] == fy_val].sort_values("filed").drop_duplicates("end", keep="last")
                m9_row = all_9m[all_9m["fy"] == fy_val].sort_values("filed").drop_duplicates("end", keep="last")
                if not fy_row.empty and not m9_row.empty:
                    q4_val = fy_row["val"].values[-1] - m9_row["val"].values[-1]
                    derived_q4_rows.append({
                        "end": fy_row["end"].values[-1],
                        "val": q4_val,
                        "fy": fy_val,
                        "fp": "Q4",
                        "form": fy_row["form"].values[-1],
                        "filed": fy_row["filed"].values[-1],
                        "frame": fy_row.get("frame", pd.Series([None])).values[-1],
                        "start": m9_row["end"].values[-1],
                        "duration_days": 91,
                    })

        if derived_q4_rows:
            q4_df = pd.DataFrame(derived_q4_rows)
            quarterly_df = pd.concat([quarterly_df, q4_df], ignore_index=True)

        # Sort by filing date and keep the latest restatement / report for each period end
        quarterly_df = quarterly_df.sort_values("filed").drop_duplicates(subset=["end"], keep="last")
        quarterly_df = quarterly_df.rename(columns={"val": metric_name})
        quarterly_df["tag_used_" + metric_name] = matched_tag

        return quarterly_df[["end", metric_name, "form", "fp", "fy", "filed", "frame", "tag_used_" + metric_name]]

    def extract_company_quarterlies(self, ticker: str, cik: str) -> pd.DataFrame:
        """Extracts and merges quarterly revenue, net income, and opex for a single company."""
        logger.info(f"Extracting EDGAR facts for {ticker} (CIK: {cik})...")
        raw_json = self.fetch_company_facts(cik)

        rev_df = self._extract_metric_series(raw_json, "revenue", TAG_CANDIDATES["revenue"])
        ni_df = self._extract_metric_series(raw_json, "net_income", TAG_CANDIDATES["net_income"])
        opex_df = self._extract_metric_series(raw_json, "opex", TAG_CANDIDATES["opex"])

        # Base merge on period end date
        merged = rev_df.copy()
        if not ni_df.empty:
            merged = pd.merge(
                merged,
                ni_df[["end", "net_income", "tag_used_net_income"]],
                on="end",
                how="outer",
            )
        else:
            merged["net_income"] = None
            merged["tag_used_net_income"] = None

        if not opex_df.empty:
            merged = pd.merge(
                merged,
                opex_df[["end", "opex", "tag_used_opex"]],
                on="end",
                how="outer",
            )
        else:
            merged["opex"] = None
            merged["tag_used_opex"] = None

        merged["ticker"] = ticker.upper()
        merged["cik"] = str(cik).zfill(10)
        merged = merged.sort_values("end").reset_index(drop=True)
        merged = merged.rename(columns={"end": "period_end_date", "filed": "filed_date", "frame": "calendar_frame"})

        logger.info(f"Successfully extracted {len(merged)} quarterly records for {ticker}.")
        return merged

    def ingest_tickers(
        self,
        tickers: List[str],
        output_csv: Path | str = "data/raw/financials_raw.csv",
    ) -> pd.DataFrame:
        """Ingests financials for all specified tickers and saves to raw CSV."""
        mapper = TickerCIKMapper(user_agent=self.user_agent).load()
        dfs = []
        for ticker in tickers:
            cik = mapper.get_cik(ticker)
            if not cik:
                logger.error(f"Could not resolve CIK for ticker {ticker}. Skipping.")
                continue
            try:
                company_df = self.extract_company_quarterlies(ticker, cik)
                dfs.append(company_df)
            except Exception as e:
                logger.error(f"Failed to extract EDGAR facts for {ticker}: {e}")

        if not dfs:
            raise RuntimeError("No company financials could be extracted.")

        combined_df = pd.concat(dfs, ignore_index=True)
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(out_path, index=False)
        logger.info(f"Saved {len(combined_df)} raw financial rows to {out_path.resolve()}")
        return combined_df


if __name__ == "__main__":
    target_tickers = ["AAPL", "MSFT", "AMZN", "NVDA", "WMT"]
    ingester = EDGARIngester()
    df = ingester.ingest_tickers(target_tickers)
    print("\nFinancials Raw Sample:")
    print(df[["ticker", "period_end_date", "revenue", "net_income", "opex", "fp", "fy"]].head(10))
