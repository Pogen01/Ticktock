"""Shared model interface, interval-censored likelihood and a fitting helper."""

from __future__ import annotations

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import minimize

from .. import config

EPS = 1e-12


class WaitModel:
    name = "base"
    horizon = config.HORIZON

    def fit(self, df: pd.DataFrame) -> "WaitModel":
        return self

    def predict_cdf(self, df: pd.DataFrame) -> np.ndarray:
        """(n, horizon + 1) array, F[:, 0] == 0, non-decreasing in t."""
        raise NotImplementedError


def interval_loglik(F: np.ndarray, wait: np.ndarray, filled: np.ndarray) -> np.ndarray:
    """Per-order log-likelihood from a CDF grid.

    A fill on day k means the passage happened in (k-1, k]: P = F(k) - F(k-1).
    An unfilled order observed for C days contributes P = 1 - F(C).
    """
    n = len(wait)
    r = np.arange(n)
    w = np.clip(wait.astype(int), 0, F.shape[1] - 1)
    p_event = F[r, w] - F[r, np.clip(w - 1, 0, None)]
    p_cens = 1.0 - F[r, w]
    return np.log(np.clip(np.where(filled, p_event, p_cens), EPS, None))


def fit_linear_index(per_row_ll, blocks: list[np.ndarray], init: list[np.ndarray],
                     l2: float = 1e-3, maxiter: int = 400):
    """Maximise sum(per_row_ll(eta_1, ..., eta_m)) where eta_j = X_j @ beta_j.

    per_row_ll returns per-row log-likelihoods; gradients w.r.t. each eta_j
    are taken by per-row central differences (cheap, vectorised) and chained
    to the coefficients. Intercepts (first column) are not penalised.
    """
    sizes = [b.shape[1] for b in blocks]
    cuts = np.cumsum([0] + sizes)
    n = blocks[0].shape[0]
    h = 1e-5

    def unpack(theta):
        return [theta[cuts[j]:cuts[j + 1]] for j in range(len(blocks))]

    def objective(theta):
        betas = unpack(theta)
        etas = [X @ b for X, b in zip(blocks, betas)]
        ll = per_row_ll(*etas)
        grad = []
        for j in range(len(etas)):
            up = list(etas); up[j] = etas[j] + h
            dn = list(etas); dn[j] = etas[j] - h
            d = (per_row_ll(*up) - per_row_ll(*dn)) / (2 * h)
            grad.append(blocks[j].T @ d)
        pen = sum(l2 * n * np.sum(b[1:] ** 2) for b in betas)
        gpen = [np.r_[0.0, 2 * l2 * n * b[1:]] for b in betas]
        f = -(ll.sum() - pen) / n
        g = -(np.concatenate(grad) - np.concatenate(gpen)) / n
        return f, g

    res = minimize(objective, np.concatenate(init), jac=True, method="L-BFGS-B",
                   options={"maxiter": maxiter})
    return unpack(res.x), res


class Standardizer:
    """Column-wise z-scoring with training means/stds; adds an intercept."""

    def __init__(self, columns: list[str]):
        self.columns = list(columns)

    def fit(self, df: pd.DataFrame) -> "Standardizer":
        X = df[self.columns].astype(float)
        self.mean_ = X.mean().to_numpy()
        sd = X.std().to_numpy()
        self.std_ = np.where(sd > 1e-9, sd, 1.0)
        return self

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        X = (df[self.columns].astype(float).to_numpy() - self.mean_) / self.std_
        X = np.nan_to_num(X, nan=0.0)
        return np.column_stack([np.ones(len(X)), X])


def cdf_quantile(F: np.ndarray, q: float) -> np.ndarray:
    """Smallest t (linearly interpolated between grid days) with F(t) >= q.
    Returns np.inf where the horizon is reached first."""
    n, m = F.shape
    out = np.full(n, np.inf)
    above = F >= q
    has = above.any(1)
    k = above.argmax(1)
    r = np.arange(n)[has]
    kk = k[has]
    lo = F[r, np.clip(kk - 1, 0, None)]
    hi = F[r, kk]
    frac = np.where(hi > lo, (q - lo) / np.where(hi > lo, hi - lo, 1), 1.0)
    out[has] = np.where(kk == 0, 0.0, kk - 1 + frac)
    return out


def save_model(model: WaitModel, path: Path | None = None) -> Path:
    path = path or config.MODELS_DIR / f"{model.name}.pkl"
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    return path


def load_model(name: str) -> WaitModel:
    with open(config.MODELS_DIR / f"{name}.pkl", "rb") as f:
        return pickle.load(f)
