"""End-to-end check of the trained service (needs `make train` first)."""

import pytest

from ticktock import config

pytestmark = pytest.mark.skipif(not (config.MODELS_DIR / "meta.json").exists(),
                                reason="models not trained")


@pytest.fixture(scope="module")
def svc():
    from ticktock.service import TickTockService
    return TickTockService.load()


def test_further_target_means_longer_wait(svc):
    near = svc.predict("AAPL", "buy", 220.0, "2025-04-02")
    far = svc.predict("AAPL", "buy", 205.0, "2025-04-02")
    assert near["p_within"]["10"] > far["p_within"]["10"]
    assert near["reference_price"] == far["reference_price"]


def test_marketable_and_replay_outcome(svc):
    r = svc.predict("NVDA", "sell", 1.0, "2025-04-02")
    assert r["marketable"]
    r = svc.predict("NVDA", "buy", 108.0, "2025-04-02")
    assert r["actual"]["filled"] and r["actual"]["days"] >= 1
    assert len(r["cdf"]) == config.HORIZON + 1


def test_comparison_endpoint():
    if not (config.REPORTS_DIR / "comparison.json").exists():
        pytest.skip("comparison not built")
    try:
        from fastapi.testclient import TestClient
    except Exception:
        pytest.skip("test client unavailable")
    from app.server import app
    r = TestClient(app).get("/api/comparison", params={"ticker": "NVDA", "side": "buy", "dist": "2"})
    assert r.status_code == 200
    d = r.json()
    assert len(d["curve"]["actual"]) == config.HORIZON + 1
    assert set(d["curve"]["models"]) == {"kaplan_meier", "ig_theory", "ig_traffic",
                                         "weibull_no_risk", "weibull_aft"}
    assert len(d["timeline"]["actual"]) == len(d["months"])
