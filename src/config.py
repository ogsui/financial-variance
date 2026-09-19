"""Shared pipeline defaults."""

from __future__ import annotations

import os

PELT_PENALTY = 3.0
Z_THRESHOLD = 2.0
MAX_TOLERANCE_DAYS = 5
MIN_TRAIN_PERIODS = 8
ROLLING_Z_WINDOW = 4
PRICE_START = "2019-01-01"

SEC_USER_AGENT = os.environ.get(
    "SEC_USER_AGENT",
    "financial-variance-dashboard research@localhost",
)
