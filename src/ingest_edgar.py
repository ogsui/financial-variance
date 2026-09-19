"""SEC EDGAR quarterly financials ingestion."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import List

import pandas as pd

from config import PRICE_START, SEC_USER_AGENT
from http_util import get_json, sec_session
from ticker_cik_map import TickerCIKMapper

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

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
    ],
}


def is_plausible_amount(value: float, metric_name: str) -> bool:
    if pd.isna(value):
        return False
    if metric_name == "revenue" and value <= 0:
        return False
    return True


def derive_q4_rows(units_df: pd.DataFrame, quarterly_df: pd.DataFrame, metric_name: str) -> pd.DataFrame:
    """Derive Q4 as FY minus 9M when a quarterly Q4 is missing. Reject non-positive results."""
    all_fy = units_df[(units_df["duration_days"] >= 350) & (units_df["duration_days"] <= 375)].copy()
    all_9m = units_df[(units_df["duration_days"] >= 260) & (units_df["duration_days"] <= 285)].copy()
    derived = []

    for fy_val in all_fy["fy"].dropna().unique():
        has_q4 = not quarterly_df[(quarterly_df["fy"] == fy_val) & (quarterly_df["fp"] == "Q4")].empty
        if has_q4:
            continue
        fy_row = all_fy[all_fy["fy"] == fy_val].sort_values("filed")
        m9_row = all_9m[all_9m["fy"] == fy_val].sort_values("filed")
        if fy_row.empty or m9_row.empty:
            continue
        fy_end = pd.to_datetime(fy_row["end"].iloc[-1])
        m9_end = pd.to_datetime(m9_row["end"].iloc[-1])
        if fy_end <= m9_end:
            continue
        q4_val = float(fy_row["val"].iloc[-1] - m9_row["val"].iloc[-1])
        if not is_plausible_amount(q4_val, metric_name):
            logger.warning(
                "Skipping derived Q4 for fy=%s metric=%s (value=%s)",
                fy_val,
                metric_name,
                q4_val,
            )
            continue
        derived.append({
            "end": fy_end,
            "val": q4_val,
            "fy": fy_val,
            "fp": "Q4",
            "form": fy_row["form"].iloc[-1],
            "filed": fy_row["filed"].iloc[-1],
            "frame": fy_row["frame"].iloc[-1] if "frame" in fy_row.columns else None,
            "start": m9_end,
            "duration_days": int((fy_end - m9_end).days),
        })
    return pd.DataFrame(derived) if derived else pd.DataFrame()


class EDGARIngester:
    """Ingests quarterly financial facts from the SEC EDGAR API."""

    def __init__(
        self,
        user_agent: str = SEC_USER_AGENT,
        start_date: str = PRICE_START,
        end_date: str | None = None,
    ):
        self.user_agent = user_agent
        self.start_date = pd.to_datetime(start_date)
        self.end_date = pd.to_datetime(end_date) if end_date else pd.Timestamp.today().normalize()
        self.session = sec_session(self.user_agent)

    def fetch_company_facts(self, cik: str) -> dict:
        cik_clean = str(cik).strip().zfill(10)
        url = f"https://data.sec.gov/api/xbrl/companyfacts/CIK{cik_clean}.json"
        return get_json(self.session, url)

    def _extract_metric_series(
        self,
        facts: dict,
        metric_name: str,
        tag_list: List[str],
    ) -> pd.DataFrame:
        us_gaap = facts.get("facts", {}).get("us-gaap", {})
        candidate_dfs = []

        for tag in tag_list:
            if tag not in us_gaap or "units" not in us_gaap[tag]:
                continue
            units_dict = us_gaap[tag]["units"]
            if "USD" not in units_dict:
                continue
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
            filtered_df = raw_df[
                (raw_df["end"] >= self.start_date) & (raw_df["end"] <= self.end_date)
            ].copy()
            q_df = filtered_df[
                (filtered_df["duration_days"] >= 70) & (filtered_df["duration_days"] <= 105)
            ]
            candidate_dfs.append((len(q_df.drop_duplicates("end")), tag, filtered_df))

        if not candidate_dfs:
            logger.warning("No matching tag for metric '%s' in %s", metric_name, tag_list)
            return pd.DataFrame(columns=["end", metric_name, "form", "fp", "fy", "filed", "frame"])

        candidate_dfs.sort(key=lambda x: x[0], reverse=True)
        _, matched_tag, units_df = candidate_dfs[0]

        quarterly_df = units_df[
            (units_df["duration_days"] >= 70) & (units_df["duration_days"] <= 105)
        ].copy()

        q4_df = derive_q4_rows(units_df, quarterly_df, metric_name)
        if not q4_df.empty:
            quarterly_df = pd.concat([quarterly_df, q4_df], ignore_index=True)

        # As originally filed: first report for each period end, not later restatements.
        quarterly_df = quarterly_df.sort_values("filed").drop_duplicates(subset=["end"], keep="first")
        quarterly_df = quarterly_df.rename(columns={"val": metric_name})
        quarterly_df["tag_used_" + metric_name] = matched_tag
        quarterly_df = quarterly_df[quarterly_df[metric_name].map(lambda v: is_plausible_amount(v, metric_name) if metric_name == "revenue" else pd.notna(v))]

        cols = ["end", metric_name, "form", "fp", "fy", "filed", "frame", "tag_used_" + metric_name]
        return quarterly_df[cols]

    def extract_company_quarterlies(self, ticker: str, cik: str) -> pd.DataFrame:
        logger.info("Extracting EDGAR facts for %s (CIK: %s)...", ticker, cik)
        raw_json = self.fetch_company_facts(cik)

        rev_df = self._extract_metric_series(raw_json, "revenue", TAG_CANDIDATES["revenue"])
        ni_df = self._extract_metric_series(raw_json, "net_income", TAG_CANDIDATES["net_income"])
        opex_df = self._extract_metric_series(raw_json, "opex", TAG_CANDIDATES["opex"])

        merged = rev_df.copy()
        if not ni_df.empty:
            merged = pd.merge(
                merged,
                ni_df[["end", "net_income", "tag_used_net_income"]],
                on="end",
                how="left",
            )
        else:
            merged["net_income"] = None
            merged["tag_used_net_income"] = None

        if not opex_df.empty:
            merged = pd.merge(
                merged,
                opex_df[["end", "opex", "tag_used_opex"]],
                on="end",
                how="left",
            )
        else:
            merged["opex"] = None
            merged["tag_used_opex"] = None

        merged["ticker"] = ticker.upper()
        merged["cik"] = str(cik).zfill(10)
        merged = merged.sort_values("end").reset_index(drop=True)
        merged = merged.rename(columns={"end": "period_end_date", "filed": "filed_date", "frame": "calendar_frame"})
        logger.info("Extracted %s quarterly records for %s.", len(merged), ticker)
        return merged

    def ingest_tickers(
        self,
        tickers: List[str],
        output_csv: Path | str = "data/raw/financials_raw.csv",
    ) -> pd.DataFrame:
        mapper = TickerCIKMapper(user_agent=self.user_agent).load()
        dfs = []
        for ticker in tickers:
            cik = mapper.get_cik(ticker)
            if not cik:
                logger.error("Could not resolve CIK for ticker %s. Skipping.", ticker)
                continue
            try:
                dfs.append(self.extract_company_quarterlies(ticker, cik))
            except Exception as exc:
                logger.error("Failed to extract EDGAR facts for %s: %s", ticker, exc)

        if not dfs:
            raise RuntimeError("No company financials could be extracted.")

        combined_df = pd.concat(dfs, ignore_index=True)
        out_path = Path(output_csv)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        combined_df.to_csv(out_path, index=False)
        logger.info("Saved %s raw financial rows to %s", len(combined_df), out_path.resolve())
        return combined_df
