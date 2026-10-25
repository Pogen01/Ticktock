"""Loading raw daily price bars and the event calendar."""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config


def load_prices(tickers: list[str] | None = None) -> pd.DataFrame:
    """Return a long frame: date, ticker, open, high, low, close, volume.

    The CSVs are split-adjusted daily bars (Yahoo-style). We use the raw
    (split-adjusted, not dividend-adjusted) OHLC so that close, high and
    low are on the same scale as a real limit price.
    """
    tickers = tickers or config.TICKERS
    frames = []
    for t in tickers:
        df = pd.read_csv(config.PRICES_DIR / f"{t}.csv", parse_dates=["Date"])
        df = df.rename(columns=str.lower).rename(columns={"adj close": "adj_close"})
        df["ticker"] = t
        frames.append(df[["date", "ticker", "open", "high", "low", "close", "volume"]])
    out = pd.concat(frames, ignore_index=True)
    out = out.dropna(subset=["open", "high", "low", "close"])
    out = out[(out["high"] >= out["low"]) & (out["close"] > 0)]
    return out.sort_values(["ticker", "date"]).reset_index(drop=True)


def market_index(prices: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight index of the 20 stocks: our stand-in for SPY.

    SPY/VIX were not available offline, so market mood is measured from the
    cross-section itself (average return, average range, dispersion).
    """
    p = prices.sort_values(["ticker", "date"]).copy()
    p["ret"] = p.groupby("ticker")["close"].transform(lambda s: np.log(s).diff())
    p["range"] = np.log(p["high"] / p["low"])
    m = p.groupby("date").agg(mkt_ret=("ret", "mean"), mkt_range=("range", "mean"),
                              mkt_disp=("ret", "std"))
    return m.reset_index()


def load_events() -> pd.DataFrame:
    path = config.EVENTS_DIR / "fomc.csv"
    return pd.read_csv(path, parse_dates=["date"])


def third_fridays(dates: pd.Series) -> pd.Series:
    """Monthly options-expiry flag (third Friday of the month)."""
    d = pd.to_datetime(dates)
    return (d.dt.weekday == 4) & (d.dt.day >= 15) & (d.dt.day <= 21)
