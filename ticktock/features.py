"""Features known at the close of the start day (no look-ahead).

Two layers:
  * daily_features(): one row per (ticker, date), using only bars up to and
    including that date (rolling windows are trailing, baselines are shifted).
  * order_features(): adds the order-specific columns (side, distance,
    distance in volatility units, trend toward the target...).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import load_events, market_index, third_fridays
from .risk_factors import MACRO_THEMES, build_risk_features, risk_feature_columns

LN2x4 = 4 * np.log(2)


def _per_ticker(g: pd.DataFrame) -> pd.DataFrame:
    g = g.sort_values("date").copy()
    lc = np.log(g["close"])
    g["ret"] = lc.diff()
    hl2 = np.log(g["high"] / g["low"]) ** 2 / LN2x4
    g["vol5"] = g["ret"].rolling(5, min_periods=5).std()
    g["vol20"] = g["ret"].rolling(20, min_periods=15).std()
    g["park20"] = np.sqrt(hl2.rolling(20, min_periods=15).mean())
    g["park5"] = np.sqrt(hl2.rolling(5, min_periods=5).mean())
    # Blend of close-to-close and high-low volatility: fills depend on the
    # intraday extremes, which the Parkinson estimator measures directly.
    g["sigma"] = np.sqrt(0.5 * g["vol20"] ** 2 + 0.5 * g["park20"] ** 2)
    g["vol_ratio"] = np.log(np.sqrt(0.5 * g["vol5"] ** 2 + 0.5 * g["park5"] ** 2) / g["sigma"])
    # Traffic: recent volume relative to this stock's own trailing baseline
    # (baseline excludes the recent window, so it is a fair "usual" level).
    base = g["volume"].shift(5).rolling(50, min_periods=15).mean()
    g["traffic5"] = np.log(g["volume"].rolling(5).mean() / base)
    g["traffic1"] = np.log(g["volume"] / base)
    g["ret5"] = lc.diff(5)
    g["ret20"] = lc.diff(20)
    g["gap"] = np.log(g["open"] / g["close"].shift(1))
    g["range_today"] = np.log(g["high"] / g["low"]) / g["park20"]
    g["low20"] = g["low"].rolling(20, min_periods=15).min()
    g["high20"] = g["high"].rolling(20, min_periods=15).max()
    g["bar_index"] = np.arange(len(g))
    return g


def daily_features(prices: pd.DataFrame) -> pd.DataFrame:
    df = pd.concat([_per_ticker(g) for _, g in prices.groupby("ticker")], ignore_index=True)

    mkt = market_index(prices).sort_values("date")
    mkt["mkt_ret5"] = mkt["mkt_ret"].rolling(5).sum()
    mkt["mkt_ret20"] = mkt["mkt_ret"].rolling(20).sum()
    mkt["mkt_vol20"] = mkt["mkt_ret"].rolling(20, min_periods=15).std()
    mkt["mkt_vol5"] = mkt["mkt_ret"].rolling(5).std()
    mkt["mkt_disp20"] = mkt["mkt_disp"].rolling(20, min_periods=15).mean()
    # Market stress: today's market volatility vs its trailing year (z-score,
    # baseline shifted by a day so it only uses the past).
    lv = np.log(mkt["mkt_vol20"])
    mkt["mkt_stress"] = (lv - lv.shift(1).expanding(40).mean()) / lv.shift(1).expanding(40).std()
    mkt["mkt_stress"] = mkt["mkt_stress"].fillna(0.0).clip(-3, 4)
    mkt["mkt_vol_ratio"] = np.log(mkt["mkt_vol5"] / mkt["mkt_vol20"])
    df = df.merge(mkt[["date", "mkt_ret5", "mkt_ret20", "mkt_vol20", "mkt_disp20", "mkt_stress",
                       "mkt_vol_ratio"]], on="date", how="left")

    # Calendar. FOMC dates are scheduled a year ahead, so counting the ones
    # inside the coming window is not look-ahead.
    dates = pd.Index(sorted(df["date"].unique()))
    fomc = set(load_events()["date"])
    is_fomc = np.array([d in fomc for d in dates], dtype=float)
    ahead = pd.Series(is_fomc[::-1]).rolling(10, min_periods=1).sum().values[::-1]
    cal = pd.DataFrame({"date": dates, "fomc_next10": np.r_[ahead[1:], 0.0], "fomc_today": is_fomc})
    df = df.merge(cal, on="date", how="left")
    df["opex"] = third_fridays(df["date"]).astype(float)
    df["dow"] = df["date"].dt.weekday
    df["year"] = df["date"].dt.year
    # Risk text exists for 2024 and 2025 only; see config.RISK_YEARS.
    df["risk_year"] = df["year"].clip(*config.RISK_YEARS)

    risk = build_risk_features()
    rcols = risk_feature_columns(risk)
    risk = risk.rename(columns={"year": "risk_year"})
    df = df.merge(risk[["ticker", "risk_year", "sector", "risk_text"] + rcols],
                  on=["ticker", "risk_year"], how="left")
    # Risk exposure matters most when the whole market is stressed.
    df["riskx_macro_stress"] = df["risk_macro_exposure"] * df["mkt_stress"]
    df["riskx_trade_stress"] = df["risk_trade_geopolitics"] * df["mkt_stress"]
    df["riskx_count_stress"] = df["risk_count"] * df["mkt_stress"]
    return df


def order_features(daily: pd.DataFrame, side: np.ndarray, limit: np.ndarray) -> pd.DataFrame:
    """Order-level features. `daily` rows align one-to-one with side/limit.

    side: array of "buy"/"sell"; limit: limit prices.
    """
    d = daily.reset_index(drop=True).copy()
    side = np.asarray(side)
    limit = np.asarray(limit, dtype=float)
    buy = side == "buy"
    sgn = np.where(buy, -1.0, 1.0)  # +1 if the target is above the price
    ref = d["close"].to_numpy()
    tick = config.tick_for(limit)
    fill_level = np.where(buy, limit - tick, limit + tick)
    log_d = np.where(buy, np.log(ref / fill_level), np.log(fill_level / ref))
    d["side"] = side
    d["limit"] = limit
    d["side_sell"] = (~buy).astype(float)
    d["log_d"] = log_d
    d["dist_pct"] = np.abs(limit / ref - 1) * 100
    sig = d["sigma"].to_numpy()
    d["z"] = log_d / sig
    d["log_z"] = np.log(np.clip(d["z"], 0.05, None))
    d["trend5_toward"] = sgn * d["ret5"] / sig
    d["trend20_toward"] = sgn * d["ret20"] / sig
    d["mkt_trend20_toward"] = sgn * d["mkt_ret20"] / d["mkt_vol20"]
    d["gap_toward"] = sgn * d["gap"] / sig
    # How far the 20-day extreme is, and whether the limit sits beyond it
    # (a buy below the recent low needs a fresh low to fill).
    ext = np.where(buy, np.log(ref / d["low20"]), np.log(d["high20"] / ref))
    d["extreme_z"] = ext / sig
    d["beyond_extreme_z"] = (log_d - ext) / sig
    d["already_through"] = (log_d <= 0).astype(float)
    return d


BASE_FEATURES = [
    "log_z", "z", "log_d", "side_sell", "vol_ratio", "traffic5", "traffic1", "range_today",
    "trend5_toward", "trend20_toward", "gap_toward", "extreme_z", "beyond_extreme_z",
    "mkt_trend20_toward", "mkt_stress", "mkt_vol_ratio", "mkt_disp20", "fomc_next10", "opex",
]
RISK_FEATURES = [f"risk_{t}" for t in [
    "regulatory", "trade_geopolitics", "supply_chain", "ai_tech_shift", "rates_macro",
    "consumer_demand", "litigation", "competition", "mna_integration", "commodity_energy",
    "operational"]] + ["risk_count", "risk_new_count", "risk_macro_exposure",
                       "riskx_macro_stress", "riskx_trade_stress", "riskx_count_stress"]
SECTOR_FEATURES = ["sector_communication", "sector_consumer_cyclical", "sector_consumer_defensive",
                   "sector_energy", "sector_financials", "sector_healthcare", "sector_technology"]
ALL_FEATURES = BASE_FEATURES + RISK_FEATURES + SECTOR_FEATURES
# Used by the inverse-Gaussian + traffic volatility regression (no distance terms).
VOL_FEATURES = ["vol_ratio", "traffic5", "traffic1", "range_today", "mkt_stress", "mkt_vol_ratio",
                "mkt_disp20", "fomc_next10", "opex"] + RISK_FEATURES + SECTOR_FEATURES

assert set(MACRO_THEMES) <= {t.replace("risk_", "") for t in RISK_FEATURES}
