"""Waiting-time models. Every model returns the CDF of the wait on the grid
t = 0, 1, ..., horizon trading days: F[i, t] = P(order i filled within t days).
"""

from .base import WaitModel, load_model, save_model
from .inverse_gaussian import InverseGaussianTheory, InverseGaussianTraffic
from .kaplan_meier import KaplanMeier
from .weibull_aft import WeibullAFT

REGISTRY = {
    "kaplan_meier": KaplanMeier,
    "ig_theory": InverseGaussianTheory,
    "ig_traffic": InverseGaussianTraffic,
    "weibull_aft": WeibullAFT,
}

__all__ = ["WaitModel", "KaplanMeier", "InverseGaussianTheory", "InverseGaussianTraffic",
           "WeibullAFT", "REGISTRY", "load_model", "save_model"]
