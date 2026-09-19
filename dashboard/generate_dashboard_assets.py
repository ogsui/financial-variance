"""Dashboard Assets Generator Module.

Aggregates backtest results, forecasts, anomalies, and reconciliation metrics
into unified, analysis-ready JSON and CSV formats for Power BI, Tableau, and
the interactive portfolio web dashboard.
"""

from __future__ import annotations

import json
import logging
import math
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def sanitize_json(obj):
    """Recursively convert NaN/Infinity floats to None for valid JSON output."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return None
        return obj
    if isinstance(obj, dict):
        return {k: sanitize_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [sanitize_json(v) for v in obj]
    return obj


def generate_assets(
    processed_dir: Path | str = "data/processed",
    dashboard_dir: Path | str = "dashboard",
):
    """Combines processed outputs into consolidated dashboard datasets."""
    p_dir = Path(processed_dir)
    d_dir = Path(dashboard_dir)
    d_dir.mkdir(parents=True, exist_ok=True)

    cons_path = p_dir / "consolidated.csv"
    backtest_path = p_dir / "backtest_results.csv"
    forecasts_path = p_dir / "forecasts.csv"
    anomalies_path = p_dir / "anomalies.csv"
    reconcile_path = p_dir / "reconciliation_report.json"

    df_anom = pd.read_csv(anomalies_path)
    df_backtest = pd.read_csv(backtest_path)
    df_cons = pd.read_csv(cons_path)

    with open(reconcile_path, "r", encoding="utf-8") as f:
        reconcile_data = json.load(f)

    # Clean and merge for tableau / power bi unified flat file
    unified = df_anom.copy()
    # Add net income, opex from consolidated
    cons_subset = df_cons[["ticker", "period_end_date", "net_income", "opex", "volume"]].drop_duplicates()
    unified = pd.merge(unified, cons_subset, on=["ticker", "period_end_date"], how="left")

    unified_csv_path = d_dir / "dashboard_unified_feed.csv"
    unified.to_csv(unified_csv_path, index=False, na_rep="")
    logger.info(f"Saved unified dashboard dataset to {unified_csv_path.resolve()}")

    # Prepare JSON structure for interactive standalone web dashboard
    companies_data = {}
    for ticker in df_anom["ticker"].unique():
        t_anom = unified[unified["ticker"] == ticker].sort_values("period_end_date").to_dict(orient="records")
        t_bt = df_backtest[df_backtest["ticker"] == ticker].to_dict(orient="records")
        t_rec = reconcile_data["by_company"].get(ticker, {})

        best_rows = df_backtest[(df_backtest["ticker"] == ticker) & (df_backtest["is_best_model"])]
        if best_rows.empty:
            logger.warning(f"No best-model row found for {ticker}; skipping best_model/best_mape fields.")
            best_model = None
            best_mape = None
        else:
            if len(best_rows) > 1:
                logger.warning(
                    f"Multiple best-model rows found for {ticker} ({len(best_rows)}); using the first."
                )
            best_model = best_rows["model"].values[0]
            best_mape = float(best_rows["mape_pct"].values[0])

        companies_data[ticker] = {
            "series": t_anom,
            "backtest": t_bt,
            "reconciliation": t_rec,
            "best_model": best_model,
            "best_mape": best_mape,
        }

    full_payload = {
        "corporate_benchmarks": companies_data,
    }

    dashboard_json_path = d_dir / "dashboard_data.json"
    with open(dashboard_json_path, "w", encoding="utf-8") as f:
        json.dump(sanitize_json(full_payload), f, indent=2)
    logger.info(f"Saved dashboard JSON data feed to {dashboard_json_path.resolve()}")


if __name__ == "__main__":
    generate_assets()