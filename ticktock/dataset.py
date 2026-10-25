"""Build the waiting-time dataset: one row per hypothetical limit order.

For each stock and trading day t (the "start", order placed at the close):
  * create buy orders below the close and sell orders above it at each
    distance in config.DISTANCES_PCT;
  * conservative fill rule: a buy at limit L fills on the first later day
    whose low <= L - tick; a sell when high >= L + tick (tick = $0.01,
    capped at 2 bp of the price for tiny split-adjusted prices);
  * wait = trading days from t to the fill day (1 = next session);
  * if no fill within config.HORIZON days (or before the data ends) the
    order is right-censored at the number of days observed.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .data import load_prices
from .features import daily_features, order_features


def first_passage(low: np.ndarray, high: np.ndarray, start: int, fill_level: float,
                  side: str, horizon: int) -> tuple[int, bool]:
    """Wait and fill flag for one order starting after bar `start`."""
    end = min(start + horizon, len(low) - 1)
    for k, j in enumerate(range(start + 1, end + 1), 1):
        if (side == "buy" and low[j] <= fill_level) or (side == "sell" and high[j] >= fill_level):
            return k, True
    return end - start, False


def _labels_for_ticker(g: pd.DataFrame, horizon: int) -> pd.DataFrame:
    g = g.sort_values("date").reset_index(drop=True)
    n = len(g)
    low, high, close = (g[c].to_numpy() for c in ("low", "high", "close"))
    idx = np.arange(n)[:, None] + np.arange(1, horizon + 1)[None, :]
    valid = idx < n
    idxc = np.clip(idx, 0, n - 1)
    wlow = np.where(valid, low[idxc], np.inf)
    whigh = np.where(valid, high[idxc], -np.inf)
    observed = valid.sum(1)
    rows = []
    for side in ("buy", "sell"):
        for dist in config.DISTANCES_PCT:
            # 4 decimals: early split-adjusted prices are often below $5.
            if side == "buy":
                limit = np.round(close * (1 - dist / 100), 4)
                hit = wlow <= (limit - config.tick_for(limit))[:, None]
            else:
                limit = np.round(close * (1 + dist / 100), 4)
                hit = whigh >= (limit + config.tick_for(limit))[:, None]
            filled = hit.any(1)
            wait = np.where(filled, hit.argmax(1) + 1, observed)
            rows.append(pd.DataFrame({"row": np.arange(n), "side": side, "dist_target": dist,
                                      "limit": limit, "wait": wait, "filled": filled,
                                      "observed": observed}))
    lab = pd.concat(rows, ignore_index=True)
    lab["date"] = g["date"].to_numpy()[lab["row"]]
    lab["ticker"] = g["ticker"].iloc[0]
    return lab.drop(columns="row")


def assign_split(dates: pd.Series, horizon: int = config.HORIZON) -> pd.Series:
    """Chronological split with a purge: training starts whose outcome window
    would run into the tune period are dropped (labelled "purged")."""
    d = pd.to_datetime(dates)
    out = pd.Series("none", index=d.index)
    for name, (a, b) in config.SPLITS.items():
        out[(d >= a) & (d <= b)] = name
    all_days = pd.Index(sorted(d.unique()))
    tune_start = all_days[all_days >= config.SPLITS["tune"][0]][0]
    cutoff = all_days[all_days.get_loc(tune_start) - horizon]
    out[(out == "train") & (d > cutoff)] = "purged"
    return out


def build_dataset(horizon: int = config.HORIZON) -> pd.DataFrame:
    prices = load_prices()
    daily = daily_features(prices)
    labels = pd.concat([_labels_for_ticker(g, horizon) for _, g in prices.groupby("ticker")],
                       ignore_index=True)
    df = labels.merge(daily, on=["ticker", "date"], how="left")
    df = df[df["bar_index"] >= config.WARMUP_DAYS]
    df = df[df["observed"] > 0].reset_index(drop=True)
    feats = order_features(df, df["side"].to_numpy(), df["limit"].to_numpy())
    feats["split"] = assign_split(feats["date"], horizon)
    feats = feats.dropna(subset=["sigma", "traffic5", "mkt_vol20"]).reset_index(drop=True)
    return feats


def save_dataset(df: pd.DataFrame) -> str:
    config.PROCESSED_DIR.mkdir(parents=True, exist_ok=True)
    path = config.PROCESSED_DIR / "orders.parquet"
    df.drop(columns=["risk_text"]).to_parquet(path, index=False)
    return str(path)


def load_dataset() -> pd.DataFrame:
    return pd.read_parquet(config.PROCESSED_DIR / "orders.parquet")
