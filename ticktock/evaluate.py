"""The scorecard, identical for every model.

1. Calibration: share of real waits inside the predicted 50% and 80% ranges.
   Uses the randomised PIT for discrete, censored data: a fill on day k has
   PIT uniform on [F(k-1), F(k)]; an order unfilled after C days has PIT
   uniform on [F(C), 1]. Coverage of [a, b] = expected share inside.
2. Brier score for "filled within 1 / 5 / 10 / 20 days" (orders whose answer
   is unknown because the data ended are skipped).
3. Mean censored log-likelihood (higher is better; a proper scoring rule).
4. Speed: milliseconds per prediction.
5. Accuracy (the plain-language number on the website): for the yes/no
   question "will the order fill within t trading days?", the model answers
   yes when its probability is at least 50%. Accuracy is the share of test
   orders it answers correctly. The baseline always gives the most common
   answer for that question, so a model is only useful if it beats it.
"""

from __future__ import annotations

import time

import numpy as np
import pandas as pd

from .models.base import interval_loglik

BRIER_DAYS = [1, 5, 10, 20]
RANGES = {"50%": (0.25, 0.75), "80%": (0.10, 0.90)}


def coverage(F: np.ndarray, wait: np.ndarray, filled: np.ndarray, lo: float, hi: float) -> float:
    r = np.arange(len(wait))
    w = wait.astype(int)
    a = np.where(filled, F[r, np.clip(w - 1, 0, None)], F[r, w])
    b = np.where(filled, F[r, w], 1.0)
    width = np.maximum(b - a, 1e-12)
    inside = np.clip(np.minimum(b, hi) - np.maximum(a, lo), 0, None) / width
    # Zero-width PIT interval (degenerate CDF): inside iff the point is.
    point = (a >= lo) & (a <= hi)
    return float(np.mean(np.where(b - a > 1e-12, inside, point)))


def brier(F: np.ndarray, wait: np.ndarray, filled: np.ndarray, observed: np.ndarray, t: int):
    y = filled & (wait <= t)
    known = y | (observed >= t)
    p = F[known, t]
    return float(np.mean((p - y[known]) ** 2)), int(known.sum())


def accuracy(F: np.ndarray, wait, filled, observed, t: int) -> tuple[float, float]:
    """(model accuracy, always-the-common-answer baseline) for "filled within t days"."""
    y = filled & (wait <= t)
    known = y | (observed >= t)
    yk = y[known]
    acc = float(np.mean((F[known, t] >= 0.5) == yk))
    base = float(max(yk.mean(), 1 - yk.mean()))
    return acc, base


def reliability(F: np.ndarray, wait, filled, observed, t: int, bins: int = 10) -> list[dict]:
    y = filled & (wait <= t)
    known = y | (observed >= t)
    p, yk = F[known, t], y[known]
    edges = np.linspace(0, 1, bins + 1)
    idx = np.clip(np.digitize(p, edges) - 1, 0, bins - 1)
    out = []
    for b in range(bins):
        m = idx == b
        if m.sum() >= 20:
            out.append({"bin": b, "pred": float(p[m].mean()), "obs": float(yk[m].mean()),
                        "n": int(m.sum())})
    return out


def scorecard(model, df: pd.DataFrame) -> dict:
    t0 = time.perf_counter()
    F = model.predict_cdf(df)
    ms = (time.perf_counter() - t0) * 1000 / max(len(df), 1)
    w, f, o = df["wait"].to_numpy(), df["filled"].to_numpy(), df["observed"].to_numpy()
    row = {"model": model.name, "n": len(df)}
    for k, (lo, hi) in RANGES.items():
        row[f"cov{k}"] = coverage(F, w, f, lo, hi)
    for t in BRIER_DAYS:
        row[f"brier@{t}d"] = brier(F, w, f, o, t)[0]
    for t in BRIER_DAYS:
        row[f"acc@{t}d"], row[f"acc_base@{t}d"] = accuracy(F, w, f, o, t)
    row["accuracy"] = float(np.mean([row[f"acc@{t}d"] for t in BRIER_DAYS]))
    row["accuracy_base"] = float(np.mean([row[f"acc_base@{t}d"] for t in BRIER_DAYS]))
    row["censored_ll"] = float(interval_loglik(F, w, f).mean())
    row["ms_per_pred"] = ms
    return row


def slices(df: pd.DataFrame, traffic_edges) -> dict[str, pd.Series]:
    tb = np.digitize(df["traffic5"], traffic_edges)
    d = df["date"]
    years = {f"year {y}": d.dt.year == y for y in sorted(d.dt.year.unique())}
    return {
        "all": pd.Series(True, index=df.index),
        **years,
        "traffic: quiet": pd.Series(tb == 0, index=df.index),
        "traffic: normal": pd.Series(tb == 1, index=df.index),
        "traffic: busy": pd.Series(tb == 2, index=df.index),
        "COVID crash (Feb-Apr 2020)": (d >= "2020-02-15") & (d <= "2020-04-30"),
        "2022 bear market": (d >= "2022-01-01") & (d <= "2022-10-31"),
        "April 2025 tariff shock": (d >= "2025-04-01") & (d <= "2025-04-30"),
        "FOMC within 10 days": df["fomc_next10"] > 0,
        "market stressed (z>1)": df["mkt_stress"] > 1,
        "buy orders": df["side"] == "buy",
        "sell orders": df["side"] == "sell",
    }


def full_report(models, df: pd.DataFrame, traffic_edges) -> pd.DataFrame:
    rows = []
    for name, mask in slices(df, traffic_edges).items():
        sub = df[mask.to_numpy()]
        if len(sub) < 200:
            continue
        for m in models:
            r = scorecard(m, sub)
            r["slice"] = name
            rows.append(r)
    return pd.DataFrame(rows)
