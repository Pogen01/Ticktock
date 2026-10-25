"""Models vs reality: each model's predicted curves against what really
happened on the held-out test period.

Two comparisons per (stock, side, distance) group of test orders:

  * fill curve: the average predicted P(filled by day t), t = 0..horizon,
    against the real fill curve (Kaplan-Meier of the actual waits, so orders
    cut off by the end of the data are handled correctly);
  * timeline: month by month, the average predicted chance of filling within
    10 trading days against the share of that month's orders that really did.

"How far off" is the mean absolute gap between the two curves, in
percentage points.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from . import config
from .models.kaplan_meier import km_cdf

TIMELINE_DAYS = 10
ALL = "ALL"


def _groups(df: pd.DataFrame):
    """Yield (ticker, side, dist, mask) for every combination incl. 'ALL'."""
    tick = df["ticker"].to_numpy()
    side = df["side"].to_numpy()
    dist = df["dist_target"].to_numpy()
    for t in [ALL] + config.TICKERS:
        mt = np.ones(len(df), bool) if t == ALL else tick == t
        for s in (ALL, "buy", "sell"):
            ms = mt if s == ALL else mt & (side == s)
            for d in [ALL] + config.DISTANCES_PCT:
                m = ms if d == ALL else ms & (dist == d)
                if m.any():
                    yield t, s, (ALL if d == ALL else f"{d:g}"), m


def build_comparison(models, test: pd.DataFrame) -> dict:
    test = test.reset_index(drop=True)
    H = config.HORIZON
    T = TIMELINE_DAYS
    w, f, o = test["wait"].to_numpy(), test["filled"].to_numpy(), test["observed"].to_numpy()
    y10 = f & (w <= T)
    # Only orders whose whole 10-day window was observed: near the end of the
    # data, counting quick fills but not the still-open orders would bias it up.
    known10 = o >= T
    month = test["date"].dt.to_period("M").astype(str).to_numpy()
    months = sorted(set(month))
    m_idx = np.searchsorted(months, month)
    preds = {m.name: m.predict_cdf(test).astype(np.float32) for m in models}

    # Month-end close per ticker, for context under the timeline.
    px = (test.drop_duplicates(["ticker", "date"]).sort_values("date")
          .groupby(["ticker", test["date"].dt.to_period("M").astype(str)])["close"].last())

    curves, timeline = {}, {}
    for t, s, d, m in _groups(test):
        actual = km_cdf(w[m], f[m], H)
        entry = {"n": int(m.sum()), "actual": np.round(actual, 4).tolist(), "models": {}}
        k = m & known10
        cnt = np.bincount(m_idx[k], minlength=len(months))
        hits = np.bincount(m_idx[k], weights=y10[k], minlength=len(months))
        with np.errstate(invalid="ignore", divide="ignore"):
            act_m = np.where(cnt >= 5, hits / np.maximum(cnt, 1), np.nan)
        tl = {"actual": act_m, "n": cnt, "models": {}}
        for name, F in preds.items():
            curve = F[m].mean(0).astype(float)
            gap = np.abs(curve[1:] - actual[1:])
            entry["models"][name] = {"curve": np.round(curve, 4).tolist(),
                                     "mean_gap": round(float(gap.mean()), 4),
                                     "max_gap": round(float(gap.max()), 4),
                                     "max_gap_day": int(gap.argmax()) + 1}
            pm = np.bincount(m_idx[k], weights=F[k, T], minlength=len(months))
            with np.errstate(invalid="ignore", divide="ignore"):
                pred_m = np.where(cnt >= 5, pm / np.maximum(cnt, 1), np.nan)
            ok = ~np.isnan(act_m)
            tl["models"][name] = {"series": pred_m,
                                  "mean_gap": round(float(np.abs(pred_m[ok] - act_m[ok]).mean()), 4)
                                  if ok.any() else None}
        curves.setdefault(t, {}).setdefault(s, {})[d] = entry
        timeline.setdefault(t, {}).setdefault(s, {})[d] = tl

    def clean(a):
        return [None if not np.isfinite(x) else round(float(x), 4) for x in a]

    for t in timeline:
        for s in timeline[t]:
            for d, tl in timeline[t][s].items():
                tl["actual"] = clean(tl["actual"])
                tl["n"] = [int(x) for x in tl["n"]]
                for v in tl["models"].values():
                    v["series"] = clean(v["series"])

    prices = {t: [None if (t, mo) not in px.index else round(float(px[(t, mo)]), 2)
                  for mo in months] for t in config.TICKERS}
    return {"period": [str(test["date"].min().date()), str(test["date"].max().date())],
            "horizon": H, "timeline_days": T, "months": months, "models": list(preds),
            "distances": [f"{d:g}" for d in config.DISTANCES_PCT],
            "curves": curves, "timeline": timeline, "prices": prices}
