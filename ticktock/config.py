"""Project-wide settings. Paths are relative to the repository root."""

from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
RAW_DIR = ROOT / "data" / "raw"
PRICES_DIR = RAW_DIR / "prices"
RISK_FACTORS_XLSX = RAW_DIR / "risk_factors.xlsx"
EVENTS_DIR = RAW_DIR / "events"
PROCESSED_DIR = ROOT / "data" / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"

TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "NVDA", "META", "TSLA", "BRK.B", "LLY", "V",
    "AVGO", "JPM", "UNH", "WMT", "XOM", "MA", "JNJ", "PG", "HD", "CVX",
]

# Wait horizon in trading days. An order not filled within HORIZON days is
# right-censored ("not within the horizon").
HORIZON = 40

# Hypothetical limit distances from the reference (close) price, in percent.
DISTANCES_PCT = [0.5, 1.0, 2.0, 3.0, 5.0, 8.0]

# Conservative fill rule: price must trade one tick through the limit.
# The 20-year history is split-adjusted, so early prices can be tiny (NVDA
# trades at $0.15 adjusted in 2006). One real cent is then far less than one
# adjusted cent, so the tick is capped at 2 basis points of the price.
TICK = 0.01
TICK_BPS_CAP = 2e-4


def tick_for(price):
    """Tick used by the fill rule at this (split-adjusted) price level."""
    import numpy as np
    return np.minimum(TICK, TICK_BPS_CAP * np.asarray(price, dtype=float))


# Warm-up so rolling features exist.
WARMUP_DAYS = 21

# Chronological splits on the order's start date (20 years of daily bars,
# Oct 2006 - Oct 2026). Training starts are purged so their outcome window
# ends before the next split begins (no leakage).
SPLITS = {
    "train": ("2006-01-01", "2019-12-31"),
    "tune": ("2020-01-01", "2021-12-31"),
    "test": ("2022-01-01", "2026-12-31"),
}
SPLIT_LABELS = {
    "train": "Nov 2006 - Dec 2019",
    "tune": "2020 - 2021 (incl. the COVID crash)",
    "test": "Jan 2022 - Oct 2026",
}

# The risk-factor sheet only has 2024 and 2025 text. Earlier years use the
# 2024 text as a static company profile; 2026 uses the 2025 text.
RISK_YEARS = (2024, 2025)

SEED = 7
