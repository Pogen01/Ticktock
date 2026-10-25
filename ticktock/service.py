"""Prediction service used by the web demo (and handy from a notebook).

    svc = TickTockService.load()
    svc.predict("AAPL", "buy", 240.0, "2025-04-02")
"""

from __future__ import annotations

import json
from functools import cached_property

import numpy as np
import pandas as pd

from . import config
from .data import load_prices
from .features import RISK_FEATURES, SECTOR_FEATURES, daily_features, order_features
from .models import load_model
from .models.base import cdf_quantile
from .risk_factors import THEMES, tag_text

QUANTILES = {"q10": 0.10, "q25": 0.25, "q50": 0.50, "q75": 0.75, "q90": 0.90}
WITHIN_DAYS = [1, 5, 10, 20]
MODEL_LABELS = {
    "weibull_aft": "Weibull survival + risk factors",
    "weibull_no_risk": "Weibull survival (no risk factors)",
    "ig_traffic": "Random-walk theory + traffic",
    "ig_theory": "Random-walk theory",
    "kaplan_meier": "History (Kaplan-Meier)",
}
FEATURE_LABELS = {
    "log_z": "distance to target", "z": "distance to target", "log_d": "distance to target",
    "side_sell": "buy vs. sell", "vol_ratio": "volatility rising/falling", "traffic5": "5-day traffic",
    "traffic1": "today's traffic", "range_today": "today's range", "trend5_toward":
    "5-day trend toward target", "trend20_toward": "20-day trend toward target",
    "gap_toward": "opening gap", "extreme_z": "room to 20-day extreme",
    "beyond_extreme_z": "beyond 20-day high/low", "mkt_trend20_toward": "market trend",
    "mkt_stress": "market stress", "mkt_vol_ratio": "market volatility change",
    "mkt_disp20": "market dispersion", "fomc_next10": "Fed meeting ahead", "opex": "options expiry",
}


def _clean(x):
    if isinstance(x, (float, np.floating)):
        return None if not np.isfinite(x) else round(float(x), 4)
    return x


class TickTockService:
    def __init__(self, models: dict, meta: dict, prices: pd.DataFrame, daily: pd.DataFrame):
        self.models = models
        self.meta = meta
        self.prices = prices
        self.daily = daily.set_index(["ticker", "date"]).sort_index()
        self.edges = np.array(meta["traffic_edges"])

    @classmethod
    def load(cls) -> "TickTockService":
        meta = json.loads((config.MODELS_DIR / "meta.json").read_text())
        models = {n: load_model(n) for n in meta["models"]}
        prices = load_prices()
        return cls(models, meta, prices, daily_features(prices))

    # ---- metadata -------------------------------------------------------
    @cached_property
    def dates(self) -> list[str]:
        d = self.daily.dropna(subset=["sigma", "traffic5", "mkt_vol20"])
        d = d[d["bar_index"] >= config.WARMUP_DAYS]
        return sorted({x.strftime("%Y-%m-%d") for x in d.index.get_level_values("date")})

    @cached_property
    def accuracy(self) -> dict:
        """Held-out test accuracy per model, from reports/test_scorecard.csv."""
        path = config.REPORTS_DIR / "test_scorecard.csv"
        if not path.exists():
            return {}
        out = {}
        for r in pd.read_csv(path).to_dict(orient="records"):
            out[r["model"]] = {
                "accuracy": _clean(r["accuracy"]), "baseline": _clean(r["accuracy_base"]),
                "by_day": {str(t): _clean(r[f"acc@{t}d"]) for t in WITHIN_DAYS},
                "range80_hit": _clean(r["cov80%"]), "range50_hit": _clean(r["cov50%"]),
                "n": int(r["n"]),
            }
        return out

    def info(self) -> dict:
        companies = (self.daily.reset_index()[["ticker", "sector"]].drop_duplicates("ticker")
                     .set_index("ticker")["sector"].to_dict())
        return {
            "tickers": [{"ticker": t, "sector": companies.get(t)} for t in config.TICKERS],
            "dates": {"first": self.dates[0], "last": self.dates[-1],
                      "default": "2025-04-02" if "2025-04-02" in self.dates else self.dates[-1]},
            "models": [{"id": n, "label": MODEL_LABELS.get(n, n), "test": self.accuracy.get(n)}
                       for n in self.meta["models"]],
            "splits": self.meta.get("split_labels", config.SPLIT_LABELS),
            "test_period": [self.meta.get("first_test_date"), self.meta.get("last_test_date")],
            "default_model": self.meta["default_model"],
            "horizon": config.HORIZON,
            "trained_through": self.meta["trained_through"],
        }

    def _row(self, ticker: str, as_of: str) -> pd.Series:
        ts = pd.Timestamp(as_of)
        g = self.daily.loc[ticker]
        valid = g.index[g.index <= ts]
        if len(valid) == 0:
            raise ValueError(f"No data for {ticker} on or before {as_of}")
        row = g.loc[valid[-1]].copy()
        row["ticker"], row["date"] = ticker, valid[-1]
        return row

    def _orders(self, row: pd.Series, side: str, limits) -> pd.DataFrame:
        limits = np.atleast_1d(np.asarray(limits, dtype=float))
        d = pd.DataFrame([row] * len(limits)).reset_index(drop=True)
        return order_features(d, np.array([side] * len(limits)), limits)

    @staticmethod
    def _summary(F: np.ndarray) -> dict:
        out = {k: _clean(cdf_quantile(F[None, :], q)[0]) for k, q in QUANTILES.items()}
        out["p_within"] = {str(t): _clean(F[t]) for t in WITHIN_DAYS}
        out["p_beyond_horizon"] = _clean(1 - F[-1])
        return out

    # ---- main entry point ----------------------------------------------
    def predict(self, ticker: str, side: str, target: float, as_of: str,
                model: str | None = None) -> dict:
        model = model or self.meta["default_model"]
        if ticker not in config.TICKERS:
            raise ValueError(f"Unknown ticker {ticker}")
        if side not in ("buy", "sell"):
            raise ValueError("side must be buy or sell")
        row = self._row(ticker, as_of)
        ref = float(row["close"])
        marketable = (side == "buy" and target >= ref) or (side == "sell" and target <= ref)
        o = self._orders(row, side, [target])
        F = self.models[model].predict_cdf(o)[0]
        res = {
            "ticker": ticker, "side": side, "target": target, "as_of": row["date"].strftime("%Y-%m-%d"),
            "model": model, "model_test": self.accuracy.get(model),
            "in_sample": bool(row["date"] <= pd.Timestamp(self.meta["trained_through"])),
            "reference_price": round(ref, 2),
            "distance_pct": round(float((target / ref - 1) * 100), 3),
            "distance_sigmas": _clean(float(o["z"].iloc[0])),
            "daily_volatility_pct": round(float(row["sigma"] * 100), 3),
            "marketable": bool(marketable),
            "cdf": [_clean(x) for x in F],
            **self._summary(F),
            "traffic": self._traffic(ticker, row),
            "market": self._market(row),
            "comparison": self._comparison(o),
            "tradeoff": self._tradeoff(row, side, model),
            "risk": self._risk(ticker, row, o),
            "drivers": self._drivers(o),
            "actual": self._actual(ticker, row["date"], side, target),
        }
        res["calendar"] = self._calendar(row["date"], res)
        res["briefing"] = self._briefing(res)
        return res

    # ---- pieces ---------------------------------------------------------
    def _traffic(self, ticker: str, row: pd.Series) -> dict:
        hist = self.daily.loc[ticker]["traffic5"]
        hist = hist[hist.index <= row["date"]].dropna()
        v = float(row["traffic5"])
        level = ["quiet", "normal", "busy"][int(np.digitize(v, self.edges))]
        return {"value": round(v, 3), "ratio": round(float(np.exp(v)), 2), "level": level,
                "percentile": round(float((hist < v).mean() * 100), 1)}

    def _market(self, row: pd.Series) -> dict:
        s = float(row["mkt_stress"])
        level = "calm" if s < -0.5 else "normal" if s < 1 else "stressed"
        return {"stress": round(s, 2), "level": level, "fomc_next10": int(row["fomc_next10"])}

    def _comparison(self, o: pd.DataFrame) -> list[dict]:
        out = []
        for name, m in self.models.items():
            F = m.predict_cdf(o)[0]
            out.append({"model": name, "label": MODEL_LABELS.get(name, name),
                        "test": self.accuracy.get(name), **self._summary(F)})
        return out

    def _tradeoff(self, row: pd.Series, side: str, model: str) -> list[dict]:
        ref = float(row["close"])
        dists = np.round(np.r_[np.arange(0.25, 2, 0.25), np.arange(2, 10.5, 0.5)], 2)
        sgn = -1 if side == "buy" else 1
        limits = np.round(ref * (1 + sgn * dists / 100), 2)
        F = self.models[model].predict_cdf(self._orders(row, side, limits))
        q = {k: cdf_quantile(F, v) for k, v in QUANTILES.items()}
        return [{"distance_pct": float(d), "price": float(lim),
                 "q10": _clean(q["q10"][i]), "q50": _clean(q["q50"][i]), "q90": _clean(q["q90"][i]),
                 "p10": _clean(F[i, 10]), "p_horizon": _clean(F[i, -1])}
                for i, (d, lim) in enumerate(zip(dists, limits))]

    def _risk(self, ticker: str, row: pd.Series, o: pd.DataFrame | None = None) -> dict:
        text = row["risk_text"]
        hits = tag_text(text)
        ry = int(row["risk_year"])
        out = {"year": ry, "order_year": int(row["year"]), "sector": row["sector"], "text": text,
               "themes": [{"theme": t, "phrases": hits[t]} for t in THEMES if t in hits],
               "macro_exposure": int(row["risk_macro_exposure"])}
        other = 2025 if ry == 2024 else 2024
        g = self.daily.loc[ticker]
        texts = g.loc[g["risk_year"] == other, "risk_text"]
        if len(texts):
            out["other_year"] = {"year": other, "text": texts.iloc[0]}
        if o is not None:
            m = self.models["weibull_aft"]
            c = m.contributions(o).iloc[0]
            risk_part = float(c[[f for f in RISK_FEATURES + SECTOR_FEATURES if f in c.index]].sum())
            out["wait_multiplier"] = round(float(np.exp(risk_part)), 3)
        return out

    def _drivers(self, o: pd.DataFrame) -> list[dict]:
        c = self.models["weibull_aft"].contributions(o).iloc[0]
        base = c[[k for k in FEATURE_LABELS if k in c.index]]
        merged = {}
        for k, v in base.items():
            lab = FEATURE_LABELS[k]
            merged[lab] = merged.get(lab, 0.0) + float(v)
        risk = float(c[[f for f in RISK_FEATURES + SECTOR_FEATURES if f in c.index]].sum())
        merged["company risk factors"] = risk
        items = sorted(merged.items(), key=lambda kv: -abs(kv[1]))[:7]
        return [{"driver": k, "log_effect": round(v, 3), "multiplier": round(float(np.exp(v)), 3)}
                for k, v in items]

    def _actual(self, ticker: str, as_of: pd.Timestamp, side: str, target: float) -> dict | None:
        g = self.prices[self.prices["ticker"] == ticker].reset_index(drop=True)
        after = g[g["date"] > as_of].head(config.HORIZON)
        if after.empty:
            return None
        if side == "buy":
            hit = after["low"] <= target - config.tick_for(target)
        else:
            hit = after["high"] >= target + config.tick_for(target)
        if hit.any():
            k = int(np.argmax(hit.to_numpy())) + 1
            return {"filled": True, "days": k,
                    "date": after["date"].iloc[k - 1].strftime("%Y-%m-%d"),
                    "observed_days": len(after)}
        return {"filled": False, "days": len(after), "observed_days": len(after),
                "date": None}

    def _calendar(self, as_of: pd.Timestamp, res: dict) -> dict:
        """Map trading-day quantiles to calendar dates (NYSE sessions where we
        have them, business days beyond the data)."""
        sessions = pd.DatetimeIndex(sorted(self.prices["date"].unique()))
        future = sessions[sessions > as_of]
        extra = pd.bdate_range(sessions[-1] + pd.Timedelta(days=1), periods=config.HORIZON + 5)
        future = future.append(extra)

        def to_date(t):
            if t is None:
                return None
            k = int(np.ceil(max(t, 1e-9)))
            return future[k - 1].strftime("%Y-%m-%d") if 0 < k <= len(future) else None

        return {k: to_date(res[k]) for k in QUANTILES}

    @staticmethod
    def _briefing(r: dict) -> str:
        if r["marketable"]:
            return (f"{r['ticker']} last closed at ${r['reference_price']:.2f}, so a {r['side']} "
                    f"limit at ${r['target']:.2f} is already at or through the market and would "
                    f"fill straight away.")

        def days(x):
            if x is None:
                return f"more than {config.HORIZON} trading days"
            if x < 1:
                return "within the next session"
            return f"{x:.0f} trading day{'s' if round(x) != 1 else ''}"

        lo, hi = r["q10"], r["q90"]
        rng = (f"between {days(lo)} and {days(hi)}" if hi is not None
               else f"at least {days(lo)}, and it may not fill within {config.HORIZON} days")
        p10 = r["p_within"]["10"]
        traffic = {"busy": "Trading is busier than usual for this stock, which tends to shorten waits.",
                   "quiet": "Trading is quieter than usual for this stock, so waits run longer.",
                   "normal": "Trading traffic is about normal for this stock."}[r["traffic"]["level"]]
        return (f"You want {r['ticker']} at ${r['target']:.2f}; it last closed at "
                f"${r['reference_price']:.2f} ({r['distance_pct']:+.2f}%). The typical wait is "
                f"{days(r['q50'])}; 80% of the time it is {rng}. Chance of filling within 10 "
                f"trading days: {p10 * 100:.0f}%. {traffic}")
