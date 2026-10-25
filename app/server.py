"""TickTock demo server.

    uvicorn app.server:app --reload        # then open http://127.0.0.1:8000
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import pandas as pd
from fastapi import FastAPI, HTTPException, Query, Request
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from ticktock import config  # noqa: E402
from ticktock.service import TickTockService  # noqa: E402

app = FastAPI(title="TickTock", version="0.1.0")
svc = TickTockService.load()
STATIC = Path(__file__).parent / "static"


@app.middleware("http")
async def no_stale_frontend(request: Request, call_next):
    # Make browsers revalidate the page and its JS/CSS, so an updated app.js is never hidden by cache
    response = await call_next(request)
    if request.url.path == "/" or request.url.path.startswith("/static/"):
        response.headers["Cache-Control"] = "no-cache"
    return response


@app.get("/api/info")
def info():
    return svc.info()


@app.get("/api/predict")
def predict(ticker: str, side: str, target: float, as_of: str, model: str | None = None):
    if model and model not in svc.models:
        raise HTTPException(400, f"Unknown model {model}")
    try:
        return svc.predict(ticker, side, target, as_of, model)
    except ValueError as e:
        raise HTTPException(400, str(e)) from e


@app.get("/api/history")
def history(ticker: str, as_of: str, before: int = Query(60, le=250),
            after: int = Query(config.HORIZON, le=120)):
    g = svc.prices[svc.prices["ticker"] == ticker].reset_index(drop=True)
    if g.empty:
        raise HTTPException(400, f"Unknown ticker {ticker}")
    ts = pd.Timestamp(as_of)
    past = g[g["date"] <= ts].tail(before)
    future = g[g["date"] > ts].head(after)

    def rows(d):
        return [{"date": r.date.strftime("%Y-%m-%d"), "open": round(r.open, 2),
                 "high": round(r.high, 2), "low": round(r.low, 2), "close": round(r.close, 2)}
                for r in d.itertuples()]

    return {"ticker": ticker, "as_of": past["date"].iloc[-1].strftime("%Y-%m-%d"),
            "past": rows(past), "future": rows(future)}


@app.get("/api/dates")
def dates():
    return svc.dates


@app.get("/api/scorecard")
def scorecard():
    out = {}
    for name in ("test_scorecard", "tune_scorecard"):
        p = config.REPORTS_DIR / f"{name}.csv"
        if p.exists():
            out[name] = pd.read_csv(p).round(4).to_dict(orient="records")
    return out


_comparison = None


@app.get("/api/comparison")
def comparison(ticker: str = "ALL", side: str = "ALL", dist: str = "ALL"):
    """Models vs reality on the held-out test period for one group of orders."""
    global _comparison
    if _comparison is None:
        path = config.REPORTS_DIR / "comparison.json"
        if not path.exists():
            raise HTTPException(404, "Run scripts/build_comparison.py first")
        _comparison = json.loads(path.read_text())
    c = _comparison
    try:
        curve = c["curves"][ticker][side][dist]
        timeline = c["timeline"][ticker][side][dist]
    except KeyError as e:
        raise HTTPException(400, f"No test orders for {ticker} / {side} / {dist}") from e
    return {"ticker": ticker, "side": side, "dist": dist, "period": c["period"],
            "horizon": c["horizon"], "timeline_days": c["timeline_days"], "months": c["months"],
            "distances": c["distances"], "curve": curve, "timeline": timeline,
            "price": c["prices"].get(ticker)}


@app.get("/")
def index():
    return FileResponse(STATIC / "index.html")


app.mount("/static", StaticFiles(directory=STATIC), name="static")
