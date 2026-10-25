"""Step 1: history. Kaplan-Meier curves by distance (in volatility units)
and traffic level. A lookup table of past waits, no assumptions."""

from __future__ import annotations

import numpy as np
import pandas as pd

from .base import WaitModel


def km_cdf(wait: np.ndarray, filled: np.ndarray, horizon: int) -> np.ndarray:
    """Discrete-time Kaplan-Meier CDF on t = 0..horizon."""
    wait = wait.astype(int)
    events = np.bincount(wait[filled], minlength=horizon + 2)[: horizon + 1]
    leaving = np.bincount(wait, minlength=horizon + 2)[: horizon + 1]
    at_risk = len(wait) - np.r_[0, np.cumsum(leaving)[:-1]]
    hazard = np.where(at_risk > 0, events / np.maximum(at_risk, 1), 0.0)
    hazard[0] = 0.0
    return 1.0 - np.cumprod(1.0 - hazard)


class KaplanMeier(WaitModel):
    name = "kaplan_meier"

    def __init__(self, z_bins: int = 12, by_traffic: bool = True, min_rows: int = 80):
        self.z_bins = z_bins
        self.by_traffic = by_traffic
        self.min_rows = min_rows

    def _keys(self, df: pd.DataFrame) -> pd.DataFrame:
        zb = np.clip(np.searchsorted(self.z_edges_, df["z"].to_numpy(), side="right") - 1,
                     0, self.z_bins - 1)
        tb = (np.searchsorted(self.t_edges_, df["traffic5"].to_numpy(), side="right")
              if self.by_traffic else np.zeros(len(df), int))
        return pd.DataFrame({"zb": zb, "tb": tb, "side": df["side"].to_numpy()})

    def fit(self, df: pd.DataFrame) -> "KaplanMeier":
        self.z_edges_ = np.quantile(df["z"], np.linspace(0, 1, self.z_bins + 1))
        self.t_edges_ = np.quantile(df["traffic5"], [1 / 3, 2 / 3])
        keys = self._keys(df)
        w, f = df["wait"].to_numpy(), df["filled"].to_numpy()
        self.curves_ = {}
        for (zb, side), idx in keys.groupby(["zb", "side"]).groups.items():
            self.curves_[(zb, -1, side)] = km_cdf(w[idx], f[idx], self.horizon)
        for (zb, tb, side), idx in keys.groupby(["zb", "tb", "side"]).groups.items():
            if len(idx) >= self.min_rows:
                self.curves_[(zb, tb, side)] = km_cdf(w[idx], f[idx], self.horizon)
        return self

    def predict_cdf(self, df: pd.DataFrame) -> np.ndarray:
        keys = self._keys(df)
        # One lookup per distinct cell instead of per row (test sets are large).
        uniq, inv = np.unique(keys.to_numpy(dtype=object).astype(str), axis=0, return_inverse=True)
        table = np.stack([self.curves_.get((int(zb), int(tb), side), self.curves_[(int(zb), -1, side)])
                          for zb, tb, side in uniq])
        return table[inv.reshape(-1)]
