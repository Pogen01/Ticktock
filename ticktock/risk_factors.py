"""Turn the qualitative risk-factor sheet into numeric model features.

The sheet has one row per company with free-text "2024 Risk Factors" and
"2025 Risk Factors". We map each text onto a fixed taxonomy of risk themes
with transparent keyword rules, so every flag can be traced back to a phrase.

Features produced per (ticker, year):
  risk_<theme>          1 if the theme is mentioned that year
  risk_count            number of themes mentioned
  risk_new_count        themes mentioned this year but not the year before
  risk_macro_exposure   themes that move with the whole market (rates,
                        trade/geopolitics, consumer demand, commodities)
  sector_<name>         one-hot sector

The year's factors apply to orders placed in that calendar year. They are
annual disclosures, so they are treated as known at the start of the year.
The sheet only covers 2024 and 2025: orders before 2024 use the 2024 text
as a static company profile and 2026 orders use the 2025 text. For the
pre-2024 years this is a mild look-ahead (the text was written later), so
the risk-factor results should be read as "company profile" effects.
"""

from __future__ import annotations

import re

import pandas as pd

from . import config

THEMES: dict[str, list[str]] = {
    "regulatory": [r"antitrust", r"\bDOJ\b", r"\bFTC\b", r"regulat", r"\bDMA\b", r"legislat",
                   r"compliance", r"scrutiny", r"Basel", r"approval", r"\bcaps?\b", r"mandates?",
                   r"Inflation Reduction Act", r"Medicare drug pricing", r"windfall tax", r"privacy",
                   r"restriction"],
    "trade_geopolitics": [r"China", r"Chinese", r"export", r"tariff", r"geopolit", r"Middle East",
                          r"OPEC", r"foreign", r"emerging markets", r"global", r"localization",
                          r"cross-border"],
    "supply_chain": [r"supply chain", r"manufacturing", r"capacity", r"bottleneck", r"TSMC",
                     r"production delays", r"fulfillment", r"inventory"],
    "ai_tech_shift": [r"\bAI\b", r"generative", r"LLM", r"GenAI", r"technology shifts",
                      r"hyperscaler", r"silicon", r"cloud", r"short-form video", r"Reality Labs"],
    "rates_macro": [r"interest rate", r"mortgage", r"macroeconomic", r"inflation", r"net interest",
                    r"dollar", r"credit", r"delinquen", r"real estate", r"banking crisis",
                    r"spending slowdown", r"cyclical"],
    "consumer_demand": [r"consumer", r"demand", r"upgrade cycles", r"trade-down", r"discretionary",
                        r"engagement", r"spending", r"housing turnover", r"EV adoption",
                        r"travel volume"],
    "litigation": [r"lawsuit", r"litigation", r"\btort\b", r"copyright", r"liabilit", r"trials?"],
    "competition": [r"competition", r"competitor", r"price wars", r"disruption", r"market share",
                    r"private label", r"alternative payment", r"fintech", r"biosimilar", r"generic",
                    r"\bBYD\b", r"Temu", r"patent cliff", r"loss of exclusivity"],
    "mna_integration": [r"acquisition", r"integration", r"Activision", r"VMware", r"Pioneer",
                        r"\bHess\b", r"succession"],
    "commodity_energy": [r"crude", r"\boil\b", r"commodity", r"lumber", r"refining",
                         r"energy transition", r"weather", r"lower-carbon", r"drilling", r"OPEC"],
    "operational": [r"cyber", r"breach", r"safety", r"operational", r"labor", r"wage", r"shrink",
                    r"union", r"underwriting", r"misinformation", r"deepfake", r"costs?\b",
                    r"CapEx", r"margins?"],
}

# Themes whose shocks hit many stocks at once: these are the ones expected
# to interact with market-wide stress (e.g. tariffs in April 2025).
MACRO_THEMES = ["trade_geopolitics", "rates_macro", "consumer_demand", "commodity_energy"]

_COMPILED = {k: [re.compile(p, re.IGNORECASE) for p in v] for k, v in THEMES.items()}


def tag_text(text: str) -> dict[str, list[str]]:
    """Return {theme: [matched phrases]} for one risk-factor text."""
    hits: dict[str, list[str]] = {}
    text = text or ""
    for theme, pats in _COMPILED.items():
        found = sorted({m.group(0) for p in pats for m in p.finditer(text)})
        if found:
            hits[theme] = found
    return hits


def load_sheet(path=None) -> pd.DataFrame:
    df = pd.read_excel(path or config.RISK_FACTORS_XLSX)
    df.columns = [c.strip() for c in df.columns]
    df["Ticker"] = df["Ticker"].str.strip()
    return df


def sector_slug(s: str) -> str:
    return re.sub(r"[^a-z]+", "_", s.lower()).strip("_")


def build_risk_features(path=None) -> pd.DataFrame:
    """One row per (ticker, year) with the numeric risk features."""
    sheet = load_sheet(path)
    sectors = sorted(sheet["Sector"].map(sector_slug).unique())
    rows = []
    for _, r in sheet.iterrows():
        prev: set[str] = set()
        for year in (2024, 2025):
            text = r[f"{year} Risk Factors"]
            hits = tag_text(text)
            themes = set(hits)
            row = {"ticker": r["Ticker"], "year": year, "sector": sector_slug(r["Sector"]),
                   "risk_text": text}
            for th in THEMES:
                row[f"risk_{th}"] = int(th in themes)
            row["risk_count"] = len(themes)
            row["risk_new_count"] = len(themes - prev) if year == 2025 else 0
            row["risk_macro_exposure"] = sum(th in themes for th in MACRO_THEMES)
            for s in sectors:
                row[f"sector_{s}"] = int(row["sector"] == s)
            rows.append(row)
            prev = themes
    out = pd.DataFrame(rows)
    missing = set(config.TICKERS) - set(out["ticker"])
    if missing:
        raise ValueError(f"Risk sheet is missing tickers: {sorted(missing)}")
    return out


def risk_feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns
            if (c.startswith("risk_") and c != "risk_text") or c.startswith("sector_")]
