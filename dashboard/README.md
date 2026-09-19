# Dashboard Guide: Power BI Desktop & Tableau Public

This directory contains analysis-ready assets to visualize corporate variance and anomalies in **Power BI Desktop**, **Tableau Public**, or via the included **standalone web dashboard**.

---

## 1. Ready-to-Use Data Feeds

| File Name | Purpose | Records |
|---|---|---|
| `dashboard_unified_feed.csv` | Single consolidated flat file containing financials, forecasts, errors, changepoint flags, and stock prices. | 140 quarterly rows |
| `dashboard_data.json` | Hierarchical JSON payload powering `index.html`. | 5 companies |
| `../data/processed/backtest_results.csv` | Evaluation metrics (MAPE & RMSE) across Naive, SES, and Linear Trend. | 15 model rows |

---

## 2. Power BI Desktop Implementation Guide

1. **Import Data**:
   - Open Power BI Desktop $\to$ **Get Data** $\to$ **Text/CSV** $\to$ select `dashboard_unified_feed.csv`.
   - Ensure `period_end_date` and `price_date` are formatted as `Date`, and financial metrics are formatted as `Decimal Number`.

2. **Create Recommended DAX Measures**:
   ```dax
   // Revenue Variance %
   Revenue_Variance_Pct = 
   DIVIDE(
       SUM(dashboard_unified_feed[actual_revenue]) - SUM(dashboard_unified_feed[forecast_revenue]),
       SUM(dashboard_unified_feed[forecast_revenue]),
       0
   )

   // Anomaly Color Marker
   Anomaly_Color = 
   SWITCH(
       SELECTEDVALUE(dashboard_unified_feed[anomaly_tier]),
       "HIGH_CONFIDENCE_ANOMALY", "#EF4444",
       "WORTH_A_LOOK", "#F59E0B",
       "#3B82F6"
   )
   ```

3. **Visual Layout**:
   - **Slicer**: `ticker` (Dropdown or Tile list).
   - **Line Chart (Top)**:
     - X-axis: `calendar_frame` or `period_end_date`
     - Y-axis: `actual_revenue` and `forecast_revenue`
     - Conditional Formatting on markers: Set marker color by `Anomaly_Color`.
   - **Clustered Column Chart (Middle Right)**:
     - Source: `backtest_results.csv`
     - X-axis: `model`
     - Y-axis: `mape_pct`
     - Tooltip: `rmse`
   - **Line Chart (Bottom, Time-Aligned)**:
     - X-axis: `calendar_frame` or `period_end_date`
     - Y-axis: `stock_price`
     - Secondary Y-axis: `volume` (optional bar chart)

---

## 3. Tableau Public Implementation Guide

1. **Connect to Data**:
   - Open Tableau Public $\to$ **To a File** $\to$ **Text file** $\to$ select `dashboard_unified_feed.csv`.
2. **Calculated Fields**:
   - **Variance ($B)**: `([actual_revenue] - [forecast_revenue]) / 1000000000`
   - **Anomaly Shape**:
     ```tableau
     IF [anomaly_tier] = "HIGH_CONFIDENCE_ANOMALY" THEN "Cross"
     ELSEIF [anomaly_tier] = "WORTH_A_LOOK" THEN "Triangle"
     ELSE "Circle"
     END
     ```
3. **Dual-Axis Synchronization**:
   - Plot `actual_revenue` and `forecast_revenue` on Columns (`period_end_date`).
   - Create a secondary synchronized axis with shape mark mapped to `Anomaly Shape`.
4. **Publish**:
   - Save to Tableau Public to generate an interactive web link for portfolio inclusion.

---

## 4. Standalone Web Dashboard

An interactive dashboard is pre-built in [`index.html`](index.html).
- Start a simple HTTP server to view:
  ```bash
  python -m http.server 8000 --directory dashboard
  ```
- Open `http://localhost:8000` in any browser.
 