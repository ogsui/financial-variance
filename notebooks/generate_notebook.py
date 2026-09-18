"""Helper script to construct exploration.ipynb."""

import json

cells = [
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "# Financial Variance & Anomaly Detection Exploration\n",
            "\n",
            "This notebook explores the end-to-end analytical workflow:\n",
            "1. **Multi-Source Data Ingestion & Quality Audit** (SEC EDGAR XBRL vs Yahoo Finance)\n",
            "2. **Fiscal Calendar Normalization & Tolerance Window Matching**\n",
            "3. **Walk-Forward Forecasting Backtesting** (Naive vs SimpleExpSmoothing vs LinearTrend)\n",
            "4. **Dual-Method Anomaly Detection** (Structural PELT Changepoints + Rolling Z-Score Variance)\n",
            "5. **Market Reaction Analysis** (Time-aligning financial surprises with stock price movements)\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "import json\n",
            "from pathlib import Path\n",
            "import matplotlib.pyplot as plt\n",
            "import numpy as np\n",
            "import pandas as pd\n",
            "import ruptures as rpt\n",
            "\n",
            "plt.style.use('seaborn-v0_8-whitegrid' if 'seaborn-v0_8-whitegrid' in plt.style.available else 'default')\n",
            "plt.rcParams['figure.figsize'] = (12, 6)\n",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 1. Inspecting Reconciled Data & Quality Metrics\n",
            "Reviewing the output of multi-source reconciliation across differing fiscal calendars.",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df_cons = pd.read_csv('../data/processed/consolidated.csv')\n",
            "with open('../data/processed/reconciliation_report.json', 'r') as f:\n",
            "    recon_report = json.load(f)\n",
            "\n",
            "print(f\"Overall Match Rate: {recon_report['overall_summary']['overall_match_rate_pct']}%\")\n",
            "print(f\"Tolerance Window: +/- {recon_report['overall_summary']['max_tolerance_window_days']} calendar days\")\n",
            "df_cons.head(8)\n",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 2. Walk-Forward Backtest Results Across Forecasting Models\n",
            "Evaluating out-of-sample MAPE across Naive baseline, Simple Exponential Smoothing, and Linear Trend regression.",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df_backtest = pd.read_csv('../data/processed/backtest_results.csv')\n",
            "display_table = df_backtest.pivot(index='ticker', columns='model', values='mape_pct')\n",
            "print('Out-of-Sample MAPE (%) Comparison:')\n",
            "print(display_table)\n",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "fig, ax = plt.subplots(figsize=(10, 5))\n",
            "display_table.plot(kind='bar', ax=ax, colormap='viridis', width=0.8)\n",
            "ax.set_title('Out-of-Sample Mean Absolute Percentage Error (MAPE %) by Model', fontsize=13, fontweight='bold')\n",
            "ax.set_ylabel('MAPE (%)')\n",
            "ax.set_xlabel('Company Ticker')\n",
            "ax.legend(title='Model')\n",
            "plt.xticks(rotation=0)\n",
            "plt.tight_layout()\n",
            "plt.savefig('../dashboard/backtest_comparison.png', dpi=150)\n",
            "plt.show()\n",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 3. Dual-Method Anomaly Detection: Ruptures PELT & Rolling Z-Scores\n",
            "Detecting structural regime shifts vs. transient forecast variance surprises.",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "df_anom = pd.read_csv('../data/processed/anomalies.csv')\n",
            "flagged = df_anom[df_anom['anomaly_tier'] != 'NORMAL']\n",
            "print(f'Total Anomaly Events Detected: {len(flagged)}')\n",
            "flagged[['ticker', 'period_end_date', 'calendar_frame', 'actual_revenue', 'forecast_revenue', 'variance_zscore', 'anomaly_tier']]\n",
        ],
    },
    {
        "cell_type": "markdown",
        "metadata": {},
        "source": [
            "## 4. Visualizing Anomalies & Market Equity Reaction (e.g. NVDA & MSFT)\n",
            "Time-aligning financial variance changepoints with equity price moves.",
        ],
    },
    {
        "cell_type": "code",
        "execution_count": None,
        "metadata": {},
        "outputs": [],
        "source": [
            "def plot_company_deepdive(ticker):\n",
            "    sub = df_anom[df_anom['ticker'] == ticker].sort_values('period_end_date').reset_index(drop=True)\n",
            "    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True, gridspec_kw={'height_ratios': [2, 1]})\n",
            "    x = range(len(sub))\n",
            "    labels = sub['calendar_frame'].fillna(sub['period_end_date']).values\n",
            "    \n",
            "    ax1.plot(x, sub['actual_revenue'] / 1e9, label='Actual Revenue ($B)', color='#2563eb', linewidth=2.5, marker='o')\n",
            "    ax1.plot(x, sub['forecast_revenue'] / 1e9, label='Forecast Baseline ($B)', color='#64748b', linestyle='--', linewidth=2)\n",
            "    \n",
            "    high_idx = sub[sub['anomaly_tier'] == 'HIGH_CONFIDENCE_ANOMALY'].index\n",
            "    look_idx = sub[sub['anomaly_tier'] == 'WORTH_A_LOOK'].index\n",
            "    \n",
            "    ax1.scatter(high_idx, sub.loc[high_idx, 'actual_revenue'] / 1e9, color='#dc2626', s=130, zorder=5, label='High Confidence Anomaly')\n",
            "    ax1.scatter(look_idx, sub.loc[look_idx, 'actual_revenue'] / 1e9, color='#d97706', s=90, zorder=5, label='Worth a Look')\n",
            "    \n",
            "    ax1.set_title(f'{ticker}: Quarterly Revenue Variance and Anomaly Detection', fontsize=14, fontweight='bold')\n",
            "    ax1.set_ylabel('Quarterly Revenue ($B)')\n",
            "    ax1.legend(loc='upper left')\n",
            "    ax1.grid(True, alpha=0.3)\n",
            "    \n",
            "    ax2.plot(x, sub['stock_price'], label='Quarter-End Stock Price ($)', color='#7c3aed', linewidth=2)\n",
            "    ax2.scatter(high_idx, sub.loc[high_idx, 'stock_price'], color='#dc2626', s=110, zorder=5)\n",
            "    ax2.scatter(look_idx, sub.loc[look_idx, 'stock_price'], color='#d97706', s=70, zorder=5)\n",
            "    \n",
            "    ax2.set_ylabel('Stock Price ($)')\n",
            "    ax2.set_xlabel('Filing Period / Frame')\n",
            "    ax2.set_xticks(list(x))\n",
            "    ax2.set_xticklabels(labels, rotation=45, ha='right', fontsize=9)\n",
            "    ax2.legend(loc='upper left')\n",
            "    ax2.grid(True, alpha=0.3)\n",
            "    \n",
            "    plt.tight_layout()\n",
            "    plt.savefig(f'../dashboard/{ticker}_deepdive.png', dpi=150)\n",
            "    plt.show()\n",
            "\n",
            "plot_company_deepdive('NVDA')\n",
            "plot_company_deepdive('MSFT')\n",
        ],
    },
]

notebook = {
    "cells": cells,
    "metadata": {
        "language_info": {"name": "python"},
        "kernelspec": {"name": "python3", "display_name": "Python 3"},
    },
    "nbformat": 4,
    "nbformat_minor": 5,
}

with open("notebooks/exploration.ipynb", "w", encoding="utf-8") as f:
    json.dump(notebook, f, indent=2)

print("Generated notebooks/exploration.ipynb successfully.")
