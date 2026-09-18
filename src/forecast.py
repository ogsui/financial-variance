"""Forecasting and Walk-Forward Backtesting Module.

Implements three time-series forecasting models for quarterly corporate revenue:
1. Naive Baseline: forecast(t) = actual(t-1)
2. Simple Exponential Smoothing (SES): statsmodels.tsa.holtwinters.SimpleExpSmoothing
3. Linear Trend Regression: sklearn.linear_model.LinearRegression on a time index

Backtest Procedure:
- Walk-forward validation: from the 8th quarter onward, fits each model on all historical
  quarters 0..t-1, predicts quarter t, and logs the out-of-sample forecast and absolute error.
- Computes out-of-sample MAPE (Mean Absolute Percentage Error) and RMSE per method per company.
- Determines the empirical 'best_method' per company based on lowest MAPE.
- Generates out-of-sample forecasts and future 4-quarter baseline projections.
- Outputs data/processed/backtest_results.csv and data/processed/forecasts.csv.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List, Tuple
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error
import statsmodels.tsa.holtwinters as hw

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)


def compute_mape(actual: np.ndarray, pred: np.ndarray) -> float:
    """Calculates Mean Absolute Percentage Error (in percentage: 0 to 100)."""
    mask = (actual != 0) & ~np.isnan(actual) & ~np.isnan(pred)
    if not np.any(mask):
        return np.nan
    return float(np.mean(np.abs((actual[mask] - pred[mask]) / actual[mask])) * 100.0)


def compute_rmse(actual: np.ndarray, pred: np.ndarray) -> float:
    """Calculates Root Mean Squared Error."""
    mask = ~np.isnan(actual) & ~np.isnan(pred)
    if not np.any(mask):
        return np.nan
    return float(np.sqrt(mean_squared_error(actual[mask], pred[mask])))


class RevenueForecaster:
    """Executes walk-forward backtesting and forecasting across corporate revenue series."""

    def __init__(
        self,
        min_train_periods: int = 8,
        consolidated_path: Path | str = "data/processed/consolidated.csv",
        output_dir: Path | str = "data/processed",
    ):
        self.min_train_periods = min_train_periods
        self.consolidated_path = Path(consolidated_path)
        self.output_dir = Path(output_dir)

    def _forecast_step(
        self,
        history: np.ndarray,
        step_ahead: int = 1,
    ) -> Dict[str, float]:
        """Generates 1-step ahead forecasts for Naive, SES, and Linear Trend on historical values."""
        # 1. Naive forecast: last observed value
        naive_val = float(history[-1])

        # 2. Simple Exponential Smoothing
        try:
            ses_model = hw.SimpleExpSmoothing(
                history,
                initialization_method="estimated"
            ).fit(optimized=True)
            ses_val = float(ses_model.forecast(step_ahead)[-1])
        except Exception:
            ses_val = naive_val

        # 3. Linear Trend Regression
        try:
            X_train = np.arange(len(history)).reshape(-1, 1)
            y_train = history
            lr = LinearRegression().fit(X_train, y_train)
            target_idx = np.array([[len(history) + step_ahead - 1]])
            lr_val = float(lr.predict(target_idx)[0])
        except Exception:
            lr_val = naive_val

        return {
            "Naive": max(0.0, naive_val),
            "SimpleExpSmoothing": max(0.0, ses_val),
            "LinearTrend": max(0.0, lr_val),
        }

    def backtest_company(
        self,
        ticker: str,
        df_company: pd.DataFrame,
    ) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """Performs expanding-window walk-forward validation for a single company."""
        df_sorted = df_company.sort_values("period_end_date").reset_index(drop=True)
        revenues = df_sorted["revenue"].values
        n_periods = len(revenues)

        if n_periods <= self.min_train_periods:
            raise ValueError(
                f"Insufficient historical quarters ({n_periods}) for {ticker}. "
                f"Requires at least {self.min_train_periods + 1}."
            )

        step_records = []

        # Walk-forward loop from min_train_periods to n_periods - 1
        for t in range(self.min_train_periods, n_periods):
            train_hist = revenues[:t]
            actual_t = revenues[t]
            target_date = df_sorted.loc[t, "period_end_date"]
            cal_frame = df_sorted.loc[t, "calendar_frame"]
            fp = df_sorted.loc[t, "fiscal_period"]
            fy = df_sorted.loc[t, "fiscal_year"]
            stock_price = df_sorted.loc[t, "stock_price"]

            preds = self._forecast_step(train_hist, step_ahead=1)

            for method_name, pred_val in preds.items():
                abs_err = abs(actual_t - pred_val)
                pct_err = (abs_err / actual_t) * 100.0 if actual_t != 0 else np.nan

                step_records.append({
                    "ticker": ticker,
                    "period_end_date": target_date,
                    "calendar_frame": cal_frame,
                    "fiscal_year": fy,
                    "fiscal_period": fp,
                    "stock_price": stock_price,
                    "actual_revenue": actual_t,
                    "model": method_name,
                    "forecast_revenue": pred_val,
                    "absolute_error": abs_err,
                    "percentage_error_pct": pct_err,
                    "train_size_quarters": t,
                })

        df_steps = pd.DataFrame(step_records)

        # Compute summary metrics per method
        metric_rows = []
        for method in ["Naive", "SimpleExpSmoothing", "LinearTrend"]:
            sub = df_steps[df_steps["model"] == method]
            mape = compute_mape(sub["actual_revenue"].values, sub["forecast_revenue"].values)
            rmse = compute_rmse(sub["actual_revenue"].values, sub["forecast_revenue"].values)
            metric_rows.append({
                "ticker": ticker,
                "model": method,
                "mape_pct": round(mape, 2),
                "rmse": round(rmse, 2),
                "evaluation_quarters": len(sub),
            })

        df_metrics = pd.DataFrame(metric_rows)
        best_model = df_metrics.sort_values("mape_pct").iloc[0]["model"]
        df_metrics["is_best_model"] = (df_metrics["model"] == best_model)

        logger.info(
            f"[{ticker}] Backtest completed: Best model is '{best_model}' "
            f"(MAPE: {df_metrics[df_metrics['model'] == best_model]['mape_pct'].values[0]:.2f}%)."
        )

        return df_steps, df_metrics

    def run_all(self) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """Runs backtesting across all companies, stores results, and builds final forecast series."""
        df_cons = pd.read_csv(self.consolidated_path)
        tickers = df_cons["ticker"].unique()

        all_steps = []
        all_metrics = []

        for ticker in tickers:
            sub = df_cons[df_cons["ticker"] == ticker]
            steps_df, metrics_df = self.backtest_company(ticker, sub)
            all_steps.append(steps_df)
            all_metrics.append(metrics_df)

        df_all_steps = pd.concat(all_steps, ignore_index=True)
        df_all_metrics = pd.concat(all_metrics, ignore_index=True)

        self.output_dir.mkdir(parents=True, exist_ok=True)

        # Save backtest_results.csv
        backtest_path = self.output_dir / "backtest_results.csv"
        df_all_metrics.to_csv(backtest_path, index=False)
        logger.info(f"Saved model backtest comparison to {backtest_path.resolve()}")

        # Build final consolidated forecasts table (using each company's best method as primary baseline)
        best_map = (
            df_all_metrics[df_all_metrics["is_best_model"]]
            .set_index("ticker")["model"]
            .to_dict()
        )

        forecast_rows = []
        for ticker in tickers:
            best_model = best_map[ticker]
            company_cons = df_cons[df_cons["ticker"] == ticker].sort_values("period_end_date").reset_index(drop=True)
            company_steps = df_all_steps[
                (df_all_steps["ticker"] == ticker) & (df_all_steps["model"] == best_model)
            ].set_index("period_end_date")

            for _, crow in company_cons.iterrows():
                pdate = crow["period_end_date"]
                actual_rev = crow["revenue"]
                stock_price = crow["stock_price"]

                if pdate in company_steps.index:
                    step_row = company_steps.loc[pdate]
                    f_val = step_row["forecast_revenue"]
                    var_val = actual_rev - f_val
                    var_pct = (var_val / f_val * 100.0) if f_val != 0 else np.nan
                    f_type = "WalkForward_OutSample"
                else:
                    # Warm-up period prior to min_train_periods (periods 0..7)
                    f_val = actual_rev
                    var_val = 0.0
                    var_pct = 0.0
                    f_type = "Warmup_Baseline"

                forecast_rows.append({
                    "ticker": ticker,
                    "period_end_date": pdate,
                    "calendar_frame": crow["calendar_frame"],
                    "fiscal_year": crow["fiscal_year"],
                    "fiscal_period": crow["fiscal_period"],
                    "actual_revenue": actual_rev,
                    "forecast_revenue": f_val,
                    "variance": var_val,
                    "variance_pct": var_pct,
                    "stock_price": stock_price,
                    "best_method": best_model,
                    "forecast_type": f_type,
                })

            # Append 4 future quarters projection using full historical data with best model
            full_rev = company_cons["revenue"].values
            last_date = pd.to_datetime(company_cons["period_end_date"].iloc[-1])
            for q_ahead in range(1, 5):
                proj_date = (last_date + pd.DateOffset(months=3 * q_ahead)).strftime("%Y-%m-%d")
                future_preds = self._forecast_step(full_rev, step_ahead=q_ahead)
                proj_val = future_preds[best_model]
                forecast_rows.append({
                    "ticker": ticker,
                    "period_end_date": proj_date,
                    "calendar_frame": f"PROJ+{q_ahead}Q",
                    "fiscal_year": None,
                    "fiscal_period": f"+{q_ahead}Q",
                    "actual_revenue": np.nan,
                    "forecast_revenue": proj_val,
                    "variance": np.nan,
                    "variance_pct": np.nan,
                    "stock_price": np.nan,
                    "best_method": best_model,
                    "forecast_type": "Future_Projection",
                })

        df_final_forecasts = pd.DataFrame(forecast_rows)
        forecasts_path = self.output_dir / "forecasts.csv"
        df_final_forecasts.to_csv(forecasts_path, index=False)
        logger.info(f"Saved baseline forecasts series to {forecasts_path.resolve()}")

        return df_all_metrics, df_all_steps, df_final_forecasts


if __name__ == "__main__":
    forecaster = RevenueForecaster()
    metrics, steps, forecasts = forecaster.run_all()
    print("\nBacktest Results Table (MAPE by Method):")
    print(metrics.to_string(index=False))
