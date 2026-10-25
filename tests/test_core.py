"""Fast checks on labels, likelihoods and models (run: pytest -q)."""

import numpy as np
import pandas as pd
import pytest

from ticktock import config
from ticktock.dataset import assign_split, first_passage
from ticktock.evaluate import coverage
from ticktock.models.base import cdf_quantile, interval_loglik
from ticktock.models.inverse_gaussian import ig_cdf_grid, ig_interval_ll
from ticktock.models.kaplan_meier import km_cdf
from ticktock.models.weibull_aft import weibull_interval_ll
from ticktock.risk_factors import build_risk_features, tag_text


def test_fill_rule_needs_one_tick_through():
    low = np.array([100, 99.5, 98.99, 97.0])
    high = low + 1
    # Touching 99.00 is not a fill; 98.99 is.
    assert first_passage(low, high, 0, 99.00 - config.TICK, "buy", 10) == (2, True)
    assert first_passage(low, high, 0, 90.0, "buy", 10) == (3, False)
    assert first_passage(low, high, 0, 100.5, "sell", 10) == (1, True)


def test_split_purges_training_overlap():
    dates = pd.Series(pd.bdate_range("2006-10-25", "2026-10-23"))
    s = assign_split(dates)
    last_train = dates[s == "train"].max()
    first_tune = dates[s == "tune"].min()
    gap = ((dates > last_train) & (dates < first_tune)).sum()
    assert gap >= config.HORIZON - 1
    assert (s[dates.dt.year >= 2022] == "test").all()
    assert (s[dates.dt.year.between(2020, 2021)] == "tune").all()


def test_tick_is_capped_for_tiny_adjusted_prices():
    assert config.tick_for(200.0) == pytest.approx(0.01)
    assert config.tick_for(0.5) == pytest.approx(0.0001)


def test_km_matches_simple_case():
    wait = np.array([1, 2, 2, 3])
    filled = np.array([True, True, False, True])
    F = km_cdf(wait, filled, 4)
    assert F[0] == 0
    assert F[1] == pytest.approx(0.25)
    assert F[2] == pytest.approx(1 - 0.75 * (2 / 3))
    assert F[3] == pytest.approx(1.0)


def test_ig_cdf_and_likelihood_agree():
    d, s = np.array([0.03, 0.05]), np.array([0.02, 0.015])
    F = ig_cdf_grid(d, s, 40)
    assert np.all(np.diff(F, axis=1) >= -1e-12)
    wait, filled = np.array([3, 40]), np.array([True, False])
    ref = interval_loglik(F, wait, filled)
    assert np.allclose(ig_interval_ll(d, s, wait, filled), ref, atol=1e-6)


def test_weibull_likelihood_agrees_with_grid():
    eta, logk = np.array([1.0, 2.5]), np.array([-0.3, 0.2])
    t = np.arange(41.0)
    F = 1 - np.exp(-np.exp(np.exp(logk)[:, None] * (np.log(np.maximum(t, 1e-12)) - eta[:, None])))
    F[:, 0] = 0
    wait, filled = np.array([2, 15]), np.array([True, False])
    assert np.allclose(weibull_interval_ll(eta, logk, wait, filled), interval_loglik(F, wait, filled))


def test_perfectly_calibrated_model_covers_nominal():
    rng = np.random.default_rng(0)
    n, H = 20000, 40
    lam = 8.0
    t = np.arange(H + 1)
    F = np.tile(1 - np.exp(-t / lam), (n, 1))
    T = np.ceil(rng.exponential(lam, n)).astype(int)
    filled = T <= H
    wait = np.where(filled, T, H)
    assert coverage(F, wait, filled, 0.25, 0.75) == pytest.approx(0.5, abs=0.02)
    assert coverage(F, wait, filled, 0.10, 0.90) == pytest.approx(0.8, abs=0.02)


def test_quantiles_from_cdf():
    F = np.array([[0, 0.2, 0.6, 0.9, 0.95]])
    assert cdf_quantile(F, 0.4)[0] == pytest.approx(1.5)
    assert np.isinf(cdf_quantile(F, 0.99)[0])


def test_accuracy_counts_correct_calls():
    from ticktock.evaluate import accuracy
    F = np.array([[0, 0.9, 0.95], [0, 0.2, 0.3], [0, 0.6, 0.7]])
    wait, filled, observed = np.array([1, 2, 2]), np.array([True, False, False]), np.array([2, 2, 2])
    acc, base = accuracy(F, wait, filled, observed, 1)
    assert acc == pytest.approx(2 / 3) and base == pytest.approx(2 / 3)


def test_risk_factor_tagging():
    hits = tag_text("Export controls to China, supply chain constraints at TSMC.")
    assert {"trade_geopolitics", "supply_chain"} <= set(hits)
    rf = build_risk_features()
    assert len(rf) == 40 and set(rf["year"]) == {2024, 2025}
    assert rf["risk_count"].between(1, 11).all()
