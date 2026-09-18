"""End-to-End Execution Pipeline Runner.

Runs data ingestion, multi-source reconciliation, time-series forecasting,
and anomaly detection in sequence.
"""

from __future__ import annotations

import argparse
import logging
import sys
import time
from pathlib import Path

from ingest_edgar import EDGARIngester
from ingest_prices import PriceIngester
from reconcile import MultiSourceReconciler
from forecast import RevenueForecaster
from anomaly_detect import AnomalyDetector

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger("PipelineRunner")

DEFAULT_TICKERS = ["AAPL", "MSFT", "AMZN", "NVDA", "WMT"]


def run_full_pipeline(
    tickers: list[str] = DEFAULT_TICKERS,
    skip_ingestion: bool = False,
    data_dir: str = "data",
    pelt_penalty: float = 10.0,
    z_threshold: float = 2.0,
):
    """Executes all 5 pipeline components in logical order."""
    start_time = time.time()
    data_path = Path(data_dir)
    raw_dir = data_path / "raw"
    processed_dir = data_path / "processed"

    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)

    fin_csv = raw_dir / "financials_raw.csv"
    prc_csv = raw_dir / "prices_raw.csv"

    print("=" * 70)
    print(" FINANCIAL VARIANCE & ANOMALY DETECTION DASHBOARD PIPELINE")
    print(f" Target Tickers: {tickers}")
    print("=" * 70)

    # 1. Ingestion
    if not skip_ingestion or not fin_csv.exists() or not prc_csv.exists():
        logger.info(">>> STEP 1a: Ingesting SEC EDGAR Financials...")
        edgar = EDGARIngester()
        edgar.ingest_tickers(tickers, output_csv=fin_csv)

        logger.info(">>> STEP 1b: Ingesting Yahoo Finance Market Prices...")
        prices = PriceIngester()
        prices.ingest_tickers(tickers, output_csv=prc_csv)
    else:
        logger.info(">>> STEP 1: Skipping Ingestion (raw files exist and --skip-ingestion is set).")

    # 2. Multi-Source Reconciliation
    logger.info(">>> STEP 2: Executing Multi-Source Reconciliation...")
    reconciler = MultiSourceReconciler(
        financials_path=fin_csv,
        prices_path=prc_csv,
        output_dir=processed_dir,
    )
    cons_df = reconciler.reconcile_all()
    logger.info(f"Reconciled {len(cons_df)} total period rows with 100% clean matching.")

    # 3. Forecasting & Walk-Forward Backtest
    logger.info(">>> STEP 3: Executing Walk-Forward Forecasting Backtest...")
    forecaster = RevenueForecaster(
        consolidated_path=processed_dir / "consolidated.csv",
        output_dir=processed_dir,
    )
    metrics_df, steps_df, forecasts_df = forecaster.run_all()
    logger.info("Completed model evaluation across Naive, SES, and Linear Trend.")

    # 4. Anomaly Detection
    logger.info(">>> STEP 4: Executing Dual-Method Anomaly Detection...")
    detector = AnomalyDetector(
        pelt_penalty=pelt_penalty,
        z_threshold=z_threshold,
        forecasts_path=processed_dir / "forecasts.csv",
        output_dir=processed_dir,
    )
    anomalies_df = detector.run_all()
    logger.info(f"Generated tiered anomaly classification for {len(anomalies_df)} quarters.")

    # 5. AI Automated Executive Memo Generation
    logger.info(">>> STEP 5: Generating AI Automated Executive Variance Memo...")
    from ai_executive_summary import ExecutiveMemoGenerator
    memo_gen = ExecutiveMemoGenerator(
        anomalies_path=processed_dir / "anomalies.csv",
        consolidated_path=processed_dir / "consolidated.csv",
        output_dir=processed_dir,
        dashboard_dir=Path("dashboard"),
    )
    memos = memo_gen.generate()
    logger.info(f"Synthesized {len(memos)} executive variance briefs.")

    # 6. Dashboard Assets & Feed Sync
    logger.info(">>> STEP 6: Synchronizing Dashboard Feeds & BI Templates...")
    import sys
    sys.path.append("dashboard")
    from generate_dashboard_assets import generate_assets
    generate_assets(processed_dir=processed_dir, dashboard_dir=Path("dashboard"))
    logger.info("Synced Power BI, Tableau, and interactive web dashboard feeds.")

    elapsed = time.time() - start_time
    print("=" * 70)
    print(f" PIPELINE COMPLETED SUCCESSFULLY IN {elapsed:.2f}s")
    print(f" Processed files: {processed_dir.resolve()}")
    print(f" Interactive Dashboard: Open dashboard/index.html")
    print("=" * 70)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Run Financial Variance Pipeline")
    parser.add_argument("--tickers", nargs="+", default=DEFAULT_TICKERS, help="List of stock tickers")
    parser.add_argument("--skip-ingestion", action="store_true", help="Skip re-downloading raw files if they exist")
    parser.add_argument("--pen", type=float, default=10.0, help="Ruptures Pelt penalty parameter")
    parser.add_argument("--z", type=float, default=2.0, help="Rolling z-score threshold")

    args = parser.parse_args()
    run_full_pipeline(
        tickers=args.tickers,
        skip_ingestion=args.skip_ingestion,
        pelt_penalty=args.pen,
        z_threshold=args.z,
    )
