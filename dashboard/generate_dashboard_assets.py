"""Dashboard Assets Generator Module.

Aggregates backtest results, forecasts, anomalies, and reconciliation metrics
into unified, analysis-ready JSON and CSV formats for Power BI, Tableau, and
the interactive portfolio web dashboard.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
import pandas as pd

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


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
    unified.to_csv(unified_csv_path, index=False)
    logger.info(f"Saved unified dashboard dataset to {unified_csv_path.resolve()}")

    # Prepare JSON structure for interactive standalone web dashboard
    companies_data = {}
    for ticker in df_anom["ticker"].unique():
        t_anom = unified[unified["ticker"] == ticker].sort_values("period_end_date").to_dict(orient="records")
        t_bt = df_backtest[df_backtest["ticker"] == ticker].to_dict(orient="records")
        t_rec = reconcile_data["by_company"].get(ticker, {})

        companies_data[ticker] = {
            "series": t_anom,
            "backtest": t_bt,
            "reconciliation": t_rec,
            "best_model": df_backtest[(df_backtest["ticker"] == ticker) & (df_backtest["is_best_model"])]["model"].values[0],
            "best_mape": float(df_backtest[(df_backtest["ticker"] == ticker) & (df_backtest["is_best_model"])]["mape_pct"].values[0]),
        }

    full_payload = {
        "corporate_benchmarks": companies_data,
    }

    dashboard_json_path = d_dir / "dashboard_data.json"
    with open(dashboard_json_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)
    logger.info(f"Saved dashboard JSON data feed to {dashboard_json_path.resolve()}")


if __name__ == "__main__":
    generate_assets()
