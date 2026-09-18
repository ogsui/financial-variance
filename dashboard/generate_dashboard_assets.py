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

    # Generate BCG X Delivery Capability Hubs Operational Dataset
    delivery_hubs = {
        "X_BUILD_LATAM": {
            "name": "LATAM Near-Shore X Build Hub",
            "location": "Sao Paulo / Mexico City",
            "headcount": 420,
            "cycles": [
                {"period": "2024-Q1", "plan_rev_m": 12.5, "actual_rev_m": 12.8, "burn_rate_pct": 96.2, "active_projects": 28, "z": 0.4, "tier": "NORMAL"},
                {"period": "2024-Q2", "plan_rev_m": 14.2, "actual_rev_m": 14.0, "burn_rate_pct": 94.5, "active_projects": 32, "z": -0.3, "tier": "NORMAL"},
                {"period": "2024-Q3", "plan_rev_m": 16.0, "actual_rev_m": 19.8, "burn_rate_pct": 104.2, "active_projects": 39, "z": 2.6, "tier": "HIGH_CONFIDENCE_ANOMALY", "note": "Rapid scale of near-shore engineering hub following 3 Fortune 50 enterprise digital transformation project wins."},
                {"period": "2024-Q4", "plan_rev_m": 18.5, "actual_rev_m": 21.4, "burn_rate_pct": 102.8, "active_projects": 44, "z": 2.1, "tier": "HIGH_CONFIDENCE_ANOMALY", "note": "Sustained high utilization; contractor timesheet correction resolved in Q4."},
                {"period": "2025-Q1", "plan_rev_m": 22.0, "actual_rev_m": 22.3, "burn_rate_pct": 97.4, "active_projects": 48, "z": 0.2, "tier": "NORMAL"},
                {"period": "2025-Q2", "plan_rev_m": 24.5, "actual_rev_m": 25.1, "burn_rate_pct": 98.1, "active_projects": 52, "z": 0.5, "tier": "NORMAL"},
                {"period": "2025-Q3", "plan_rev_m": 27.0, "actual_rev_m": 24.2, "burn_rate_pct": 88.6, "active_projects": 49, "z": -2.2, "tier": "WORTH_A_LOOK", "note": "Project completion gap before Q4 ramp-up; bench utilization dipped by 9%."},
                {"period": "2025-Q4", "plan_rev_m": 29.5, "actual_rev_m": 30.8, "burn_rate_pct": 99.4, "active_projects": 58, "z": 0.8, "tier": "NORMAL"}
            ]
        },
        "X_BUILD_EMEA": {
            "name": "EMEA Capability & Innovation Hub",
            "location": "London / Berlin / Paris",
            "headcount": 580,
            "cycles": [
                {"period": "2024-Q1", "plan_rev_m": 28.0, "actual_rev_m": 27.6, "burn_rate_pct": 93.0, "active_projects": 45, "z": -0.4, "tier": "NORMAL"},
                {"period": "2024-Q2", "plan_rev_m": 30.5, "actual_rev_m": 31.2, "burn_rate_pct": 95.5, "active_projects": 50, "z": 0.6, "tier": "NORMAL"},
                {"period": "2024-Q3", "plan_rev_m": 32.0, "actual_rev_m": 31.5, "burn_rate_pct": 92.1, "active_projects": 51, "z": -0.3, "tier": "NORMAL"},
                {"period": "2024-Q4", "plan_rev_m": 35.0, "actual_rev_m": 38.9, "burn_rate_pct": 101.5, "active_projects": 58, "z": 2.3, "tier": "HIGH_CONFIDENCE_ANOMALY", "note": "Accelerated AI platform deployments for European banking consortium."},
                {"period": "2025-Q1", "plan_rev_m": 37.0, "actual_rev_m": 36.8, "burn_rate_pct": 94.2, "active_projects": 60, "z": -0.1, "tier": "NORMAL"},
                {"period": "2025-Q2", "plan_rev_m": 39.5, "actual_rev_m": 40.2, "burn_rate_pct": 96.0, "active_projects": 63, "z": 0.4, "tier": "NORMAL"},
                {"period": "2025-Q3", "plan_rev_m": 42.0, "actual_rev_m": 41.6, "burn_rate_pct": 93.8, "active_projects": 65, "z": -0.3, "tier": "NORMAL"},
                {"period": "2025-Q4", "plan_rev_m": 45.0, "actual_rev_m": 47.4, "burn_rate_pct": 99.2, "active_projects": 70, "z": 1.4, "tier": "WORTH_A_LOOK", "note": "Late project extensions and year-end backlog milestone recognition."}
            ]
        },
        "X_BUILD_NA": {
            "name": "North America Digital Build Hub",
            "location": "New York / Boston / Silicon Valley",
            "headcount": 820,
            "cycles": [
                {"period": "2024-Q1", "plan_rev_m": 52.0, "actual_rev_m": 53.4, "burn_rate_pct": 96.8, "active_projects": 78, "z": 0.7, "tier": "NORMAL"},
                {"period": "2024-Q2", "plan_rev_m": 56.0, "actual_rev_m": 55.2, "burn_rate_pct": 94.1, "active_projects": 82, "z": -0.4, "tier": "NORMAL"},
                {"period": "2024-Q3", "plan_rev_m": 58.5, "actual_rev_m": 61.8, "burn_rate_pct": 99.5, "active_projects": 88, "z": 1.7, "tier": "WORTH_A_LOOK", "note": "Healthcare tech platform build milestone pulled forward into Q3."},
                {"period": "2024-Q4", "plan_rev_m": 63.0, "actual_rev_m": 64.1, "burn_rate_pct": 97.0, "active_projects": 92, "z": 0.4, "tier": "NORMAL"},
                {"period": "2025-Q1", "plan_rev_m": 66.0, "actual_rev_m": 65.5, "burn_rate_pct": 95.0, "active_projects": 95, "z": -0.2, "tier": "NORMAL"},
                {"period": "2025-Q2", "plan_rev_m": 70.0, "actual_rev_m": 72.8, "burn_rate_pct": 98.6, "active_projects": 102, "z": 1.1, "tier": "NORMAL"},
                {"period": "2025-Q3", "plan_rev_m": 74.0, "actual_rev_m": 79.4, "burn_rate_pct": 102.4, "active_projects": 110, "z": 2.2, "tier": "HIGH_CONFIDENCE_ANOMALY", "note": "Generative AI enterprise rollout across 4 concurrent client accounts."},
                {"period": "2025-Q4", "plan_rev_m": 80.0, "actual_rev_m": 81.2, "burn_rate_pct": 97.5, "active_projects": 114, "z": 0.4, "tier": "NORMAL"}
            ]
        }
    }

    full_payload = {
        "corporate_benchmarks": companies_data,
        "delivery_hubs": delivery_hubs
    }

    dashboard_json_path = d_dir / "dashboard_data.json"
    with open(dashboard_json_path, "w", encoding="utf-8") as f:
        json.dump(full_payload, f, indent=2)
    logger.info(f"Saved dashboard JSON data feed with Delivery Hubs to {dashboard_json_path.resolve()}")


if __name__ == "__main__":
    generate_assets()
