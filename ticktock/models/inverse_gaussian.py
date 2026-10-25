"""Steps 2 and 3: first-passage ("inverse Gaussian") models.

Work in log prices. With d = log-distance to the fill level and sigma the
daily volatility, a driftless random walk first reaches the level by day t
with probability (reflection principle)

    F(t) = 2 * (1 - Phi(d / (sigma * sqrt(t)))).

Because fills are judged on the daily high/low, the path is effectively
watched continuously, which is what the formula assumes. Probability mass
beyond the horizon is "not within the horizon".

Step 2 (theory) plugs in trailing volatility. Step 3 (traffic) lets the
features rescale the volatility: sigma_eff = sigma * exp(X @ beta), with beta
fitted by maximising the censored waiting-time likelihood directly.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.special import log_ndtr

from ..features import VOL_FEATURES
from .base import EPS, Standardizer, WaitModel, fit_linear_index


def ig_cdf_grid(d: np.ndarray, sigma: np.ndarray, horizon: int) -> np.ndarray:
    t = np.arange(horizon + 1, dtype=float)
    d = np.clip(d, 0.0, None)[:, None]
    s = sigma[:, None] * np.sqrt(np.maximum(t, 1e-12))[None, :]
    F = 2.0 * np.exp(log_ndtr(-d / s))
    F = np.clip(F, 0.0, 1.0)
    F[:, 0] = 0.0
    F[(d[:, 0] <= 0), 1:] = 1.0
    return F


def _log_cdf(d, sigma, t):
    return np.log(2.0) + log_ndtr(-d / (sigma * np.sqrt(t)))


def ig_interval_ll(d, sigma, wait, filled):
    """Per-row interval-censored log-likelihood (stable in logs)."""
    w = wait.astype(float)
    lF_k = _log_cdf(d, sigma, np.maximum(w, 1e-9))
    lF_km1 = np.where(w > 1, _log_cdf(d, sigma, np.maximum(w - 1, 1e-9)), -np.inf)
    # log(F(k) - F(k-1)) = lF_k + log1p(-exp(lF_km1 - lF_k))
    diff = np.clip(lF_km1 - lF_k, None, -1e-12)
    ev = lF_k + np.log(-np.expm1(diff))
    cens = np.log(np.clip(-np.expm1(np.minimum(lF_k, -1e-15)), EPS, None))
    return np.where(filled, np.maximum(ev, np.log(EPS)), cens)


class InverseGaussianTheory(WaitModel):
    name = "ig_theory"

    def predict_cdf(self, df: pd.DataFrame) -> np.ndarray:
        return ig_cdf_grid(df["log_d"].to_numpy(), df["sigma"].to_numpy(), self.horizon)


class InverseGaussianTraffic(WaitModel):
    name = "ig_traffic"

    def __init__(self, features: list[str] | None = None, l2: float = 1e-3):
        self.features = features or VOL_FEATURES
        self.l2 = l2

    def fit(self, df: pd.DataFrame) -> "InverseGaussianTraffic":
        self.scaler_ = Standardizer(self.features).fit(df)
        X = self.scaler_.transform(df)
        d = np.clip(df["log_d"].to_numpy(), 1e-6, None)
        sig = df["sigma"].to_numpy()
        w, f = df["wait"].to_numpy(), df["filled"].to_numpy()
        (self.beta_,), self.opt_ = fit_linear_index(
            lambda eta: ig_interval_ll(d, sig * np.exp(eta), w, f), [X],
            [np.zeros(X.shape[1])], l2=self.l2)
        return self

    def sigma_multiplier(self, df: pd.DataFrame) -> np.ndarray:
        return np.exp(self.scaler_.transform(df) @ self.beta_)

    def predict_cdf(self, df: pd.DataFrame) -> np.ndarray:
        sig = df["sigma"].to_numpy() * self.sigma_multiplier(df)
        return ig_cdf_grid(df["log_d"].to_numpy(), sig, self.horizon)

    def coefficients(self) -> pd.Series:
        return pd.Series(self.beta_, index=["intercept"] + self.features)
