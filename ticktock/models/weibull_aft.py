"""Step 3b: Weibull accelerated-failure-time model with all features.

    S(t) = exp(-(t / lambda)^k),  log lambda = X @ beta,  log k = W @ alpha

A fitted, theory-free curve. The scale depends on every feature (distance,
volatility, traffic, market stress, calendar, risk factors, sector); the
shape depends on a few, so the curve can be steeper for near orders and
flatter for far ones. Fitted by maximising the interval-censored likelihood.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from ..features import ALL_FEATURES
from .base import EPS, Standardizer, WaitModel, fit_linear_index

SHAPE_FEATURES = ["log_z", "side_sell", "mkt_stress"]


def _cumhaz(t, eta, logk):
    return np.exp(np.exp(logk) * (np.log(np.maximum(t, 1e-12)) - eta))


def weibull_interval_ll(eta, logk, wait, filled):
    w = wait.astype(float)
    Hk = _cumhaz(w, eta, logk)
    Hkm1 = np.where(w > 1, _cumhaz(w - 1, eta, logk), 0.0)
    # log(S(k-1) - S(k)) = -H(k-1) + log(1 - exp(-(H(k) - H(k-1))))
    ev = -Hkm1 + np.log(np.clip(-np.expm1(-(Hk - Hkm1)), EPS, None))
    return np.where(filled, ev, -Hk)


class WeibullAFT(WaitModel):
    name = "weibull_aft"

    def __init__(self, features: list[str] | None = None, shape_features=None, l2: float = 1e-3,
                 name: str | None = None):
        self.features = list(features or ALL_FEATURES)
        self.shape_features = list(shape_features or SHAPE_FEATURES)
        self.l2 = l2
        if name:
            self.name = name

    def fit(self, df: pd.DataFrame) -> "WeibullAFT":
        self.scaler_ = Standardizer(self.features).fit(df)
        self.shape_scaler_ = Standardizer(self.shape_features).fit(df)
        X, W = self.scaler_.transform(df), self.shape_scaler_.transform(df)
        w, f = df["wait"].to_numpy(), df["filled"].to_numpy()
        b0 = np.zeros(X.shape[1]); b0[0] = np.log(np.median(w) + 1)
        a0 = np.zeros(W.shape[1])
        (self.beta_, self.alpha_), self.opt_ = fit_linear_index(
            lambda eta, lk: weibull_interval_ll(eta, lk, w, f), [X, W], [b0, a0], l2=self.l2)
        return self

    def linear_predictors(self, df: pd.DataFrame):
        return self.scaler_.transform(df) @ self.beta_, self.shape_scaler_.transform(df) @ self.alpha_

    def predict_cdf(self, df: pd.DataFrame) -> np.ndarray:
        eta, logk = self.linear_predictors(df)
        t = np.arange(self.horizon + 1, dtype=float)
        F = 1.0 - np.exp(-_cumhaz(t[None, :], eta[:, None], logk[:, None]))
        F[:, 0] = 0.0
        return F

    def coefficients(self) -> pd.Series:
        """Scale coefficients per 1 std of each feature. exp(beta) is the
        multiplicative change in typical wait."""
        return pd.Series(self.beta_, index=["intercept"] + self.features)

    def contributions(self, df: pd.DataFrame) -> pd.DataFrame:
        """Per-row contribution of each feature to log typical wait."""
        X = self.scaler_.transform(df)[:, 1:]
        return pd.DataFrame(X * self.beta_[1:], columns=self.features, index=df.index)
