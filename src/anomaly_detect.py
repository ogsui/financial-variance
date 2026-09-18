"""Financial Anomaly Detection Module.

Implements dual-method anomaly detection and cross-referenced tiering:
1. Method A — Structural Changepoint Detection (Ruptures PELT with RBF kernel):
   Applies normalized PELT algorithm to identify structural breaks / regime shifts
   in quarterly revenue trajectory.
2. Method B — Rolling Z-Score on Relative Forecast Variance:
   Calculates out-of-sample forecast variance and evaluates z-score relative to
   the trailing 4-quarter historical baseline:
   variance = (actual - forecast) / forecast
   z = (variance - variance.shift(1).rolling(4).mean()) / variance.shift(1).rolling(4).std()
   flag = abs(z) > 2.0 (or absolute variance > 15% during early window)
3. Cross-Referenced Tiering:
   - 'HIGH_CONFIDENCE_ANOMALY': Flagged by BOTH structural regime change and statistical forecast variance.
   - 'WORTH_A_LOOK': Flagged by ONLY ONE of the two methods.
   - 'NORMAL': Not flagged by either method.

Outputs:
data/processed/anomalies.csv
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import numpy as np
import pandas as pd
import ruptures as rpt

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


class AnomalyDetector:
    """Detects structural regime changes and statistical forecast variance anomalies."""

    def __init__(
        self,
        pelt_penalty: float = 3.0,
        z_threshold: float = 2.0,
        rolling_window: int = 4,
        forecasts_path: Path | str = "data/processed/forecasts.csv",
        output_dir: Path | str = "data/processed",
    ):
        self.pelt_penalty = pelt_penalty
        self.z_threshold = z_threshold
        self.rolling_window = rolling_window
        self.forecasts_path = Path(forecasts_path)
        self.output_dir = Path(output_dir)

    def detect_changepoints(self, series: np.ndarray) -> np.ndarray:
        """Applies Ruptures PELT changepoint detection on z-normalized series."""
        clean = np.nan_to_num(series, nan=float(np.nanmedian(series)))
        if len(clean) < 4:
            return np.zeros(len(series), dtype=bool)

        std = float(np.std(clean))
        if std == 0:
            return np.zeros(len(series), dtype=bool)

        norm_series = (clean - np.mean(clean)) / std
        algo = rpt.Pelt(model="rbf").fit(norm_series)
        bkps = algo.predict(pen=self.pelt_penalty)

        flags = np.zeros(len(series), dtype=bool)
        for bkp in bkps:
            idx = bkp - 1
            if 0 <= idx < len(series) and bkp != len(series):
                flags[idx] = True

        return flags

    def detect_rolling_zscore(
        self,
        actual: np.ndarray,
        forecast: np.ndarray,
        forecast_types: np.ndarray,
    ) -> Tuple[np.ndarray, np.ndarray]:
        """Computes rolling z-score of forecast variance relative to historical rolling window."""
        with np.errstate(divide="ignore", invalid="ignore"):
            rel_variance = np.where(forecast != 0, (actual - forecast) / forecast, 0.0)

        var_series = pd.Series(rel_variance)

        # Compute trailing baseline (excluding current observation)
        trailing_mean = var_series.shift(1).rolling(window=self.rolling_window, min_periods=2).mean()
        trailing_std = var_series.shift(1).rolling(window=self.rolling_window, min_periods=2).std()

        # Fallback to inclusive rolling window
        inc_mean = var_series.rolling(window=self.rolling_window, min_periods=2).mean()
        inc_std = var_series.rolling(window=self.rolling_window, min_periods=2).std()

        # Primary z-score against prior quarters
        z_trailing = (var_series - trailing_mean) / trailing_std.replace(0, np.nan)
        z_inc = (var_series - inc_mean) / inc_std.replace(0, np.nan)

        final_z = z_trailing.fillna(z_inc).fillna(0.0).values

        # Only evaluate out-of-sample backtest quarters for forecast anomaly flagging
        flags = np.zeros(len(actual), dtype=bool)
        for i in range(len(actual)):
            if forecast_types[i] == "WalkForward_OutSample":
                if abs(final_z[i]) >= self.z_threshold or (abs(rel_variance[i]) > 0.18 and abs(final_z[i]) > 1.4):
                    flags[i] = True

        return np.round(final_z, 2), flags

    def analyze_company(self, ticker: str, df_c: pd.DataFrame) -> pd.DataFrame:
        """Runs dual-method anomaly detection and cross-referencing on a single company."""
        df_hist = df_c[df_c["forecast_type"] != "Future_Projection"].copy()
        df_hist = df_hist.sort_values("period_end_date").reset_index(drop=True)

        actual_rev = df_hist["actual_revenue"].values
        forecast_rev = df_hist["forecast_revenue"].values
        f_types = df_hist["forecast_type"].values

        # Method A: Structural Changepoints
        cp_flags = self.detect_changepoints(actual_rev)

        # Method B: Statistical Variance Z-score
        z_scores, z_flags = self.detect_rolling_zscore(actual_rev, forecast_rev, f_types)

        df_hist["changepoint_flag"] = cp_flags
        df_hist["variance_zscore"] = z_scores
        df_hist["zscore_flag"] = z_flags

        # Cross-reference with +/- 1 quarter window matching for regime shifts
        tiers = []
        n = len(df_hist)
        for i in range(n):
            is_cp = cp_flags[i]
            is_z = z_flags[i]
            # Check adjacent quarters for changepoint alignment
            adjacent_cp = any(cp_flags[max(0, i - 1): min(n, i + 2)])

            if is_z and adjacent_cp:
                tiers.append("HIGH_CONFIDENCE_ANOMALY")
            elif is_cp or is_z:
                tiers.append("WORTH_A_LOOK")
            else:
                tiers.append("NORMAL")

        df_hist["anomaly_tier"] = tiers
        df_hist["pelt_penalty"] = self.pelt_penalty
        df_hist["z_threshold"] = self.z_threshold

        n_high = (df_hist["anomaly_tier"] == "HIGH_CONFIDENCE_ANOMALY").sum()
        n_look = (df_hist["anomaly_tier"] == "WORTH_A_LOOK").sum()
        logger.info(
            f"[{ticker}] Detection completed: {n_high} High-Confidence anomalies, "
            f"{n_look} Worth-a-Look flags."
        )

        return df_hist

    def run_all(self) -> pd.DataFrame:
        """Executes anomaly detection across all companies and saves data/processed/anomalies.csv."""
        if not self.forecasts_path.exists():
            raise FileNotFoundError(f"Missing forecasts file: {self.forecasts_path}")

        df_all = pd.read_csv(self.forecasts_path)
        tickers = df_all["ticker"].unique()

        processed_dfs = []
        for ticker in tickers:
            sub = df_all[df_all["ticker"] == ticker]
            analyzed = self.analyze_company(ticker, sub)
            processed_dfs.append(analyzed)

        combined_anomalies = pd.concat(processed_dfs, ignore_index=True)

        self.output_dir.mkdir(parents=True, exist_ok=True)
        out_path = self.output_dir / "anomalies.csv"
        combined_anomalies.to_csv(out_path, index=False)
        logger.info(f"Saved anomalies analysis ({len(combined_anomalies)} rows) to {out_path.resolve()}")

        return combined_anomalies


if __name__ == "__main__":
    detector = AnomalyDetector(pelt_penalty=3.0, z_threshold=2.0)
    anomalies_df = detector.run_all()
    print("\nFlagged Anomalies Summary:")
    flagged = anomalies_df[anomalies_df["anomaly_tier"] != "NORMAL"]
    print(flagged[["ticker", "period_end_date", "calendar_frame", "actual_revenue", "forecast_revenue", "variance_zscore", "changepoint_flag", "zscore_flag", "anomaly_tier"]].to_string(index=False))
