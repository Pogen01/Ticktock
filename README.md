# TickTock

**How long until your limit order fills?**

When you want a better price than the current one, you place a limit order and wait. Most apps just say "pending". TickTock forecasts the wait: a range, the chance of filling within 1 / 5 / 10 / 20 trading days, and the chance it doesn't fill at all within 40 trading days.

> You want NVDA at $108.21; it last closed at $110.42 (−2.00%). The typical wait is 4 trading days. Chance of filling within 10 trading days: 62%. Trading is quieter than usual for this stock, so waits run longer.

TickTock does **not** predict price direction. It predicts the distribution of the waiting time.

## Quick start here

```bash
python3 -m venv .venv && source .venv/bin/activate
make install       # pip install -r requirements.txt
make train         # build dataset, fit and score every model (about 2-4 min, ~4 GB RAM)
make test          # unit and end-to-end tests
make serve         # demo at http://127.0.0.1:8000
```

On Windows (no `make`), use the PowerShell helper instead; it creates the virtual environment on first run:

```powershell
powershell -ExecutionPolicy Bypass -File run.ps1          # install (first time) and start the demo
powershell -ExecutionPolicy Bypass -File run.ps1 train    # retrain
powershell -ExecutionPolicy Bypass -File run.ps1 test     # tests
```

Python 3.10 or newer. The trained models in `models/` ship with the repo, so `make serve` (or `run.ps1`) works without retraining.

## The demo

- **Inputs:** stock, buy or sell, desired price (type it in or drag the distance slider), a replay date between Nov 2006 and Oct 2026 (the order is placed at that day's close), and the model. The model menu shows each model's test accuracy.
- **Outputs:** the 80% and 50% wait ranges, chance of filling within 1, 5, 10 and 20 days, chance of not filling within 40 days, a traffic gauge (this stock's volume against its usual level) and a market stress gauge.
- **Charts:** fill probability day by day, the price/wait trade-off curve (click it to pick a price), and the price around the replay date. Tick **Reveal** to see what actually happened and whether the real fill landed inside the forecast range.
- **Replay:** ▶ steps forward one trading day at a time while keeping your limit price fixed, so you can watch the forecast change as the price and traffic move. Try NVDA, buy −2%, 2025‑04‑02 (the tariff sell-off).
- **Explanations:** the main drivers of the forecast and the company's 2024/2025 risk factors with the themes they map to.
- **Accuracy:** the forecast card shows the selected model's accuracy on the held-out test period, and the comparison table shows it for every model. Replay dates before Nov 2021 are inside the training data, and the page says so.
- **Models vs reality tab:** pick a stock (or all 20), side and distance. For every held-out order from Jan 2022 to Oct 2026 it plots each model's average predicted fill curve (chance filled by day 1…40) against the real fill curve, and month by month the predicted vs real chance of filling within 10 days, with the stock's price underneath. A table shows how far off each model is in percentage points. Built by `scripts/build_comparison.py` from the saved models (no retraining).
- **Scorecard tab:** one accuracy card per model, then the full held-out results (Jan 2022 – Oct 2026) and the tune results.

## Data

| What | Where | Notes |
|---|---|---|
| Daily OHLCV bars, Oct 2006 – Oct 2026, 20 tickers | `data/raw/prices/*.csv` | Split-adjusted. The 20 largest US companies at the start of 2024. V starts Mar 2008, AVGO Aug 2009, TSLA Jun 2010, META May 2012 |
| Risk factors per company, 2024 and 2025 | `data/raw/risk_factors.xlsx` | Free-text, one row per company |
| FOMC meeting dates, 2006 – 2026 | `data/raw/events/fomc.csv` | Scheduled meetings only (announced a year ahead, so safe as forward-looking features). 2006–2023 and 2026 were entered by hand without network access: spot-check against federalreserve.gov |

**Important difference from the original plan** (`docs/TICKTOCK_PROJECT.md`): the plan calls for 1-minute bars and waits in minutes. The price data in this project is daily, so waits are measured in **trading days** with a 40-day horizon and the distances are 0.5–8% instead of 0.05–0.5%. Every part of the pipeline (fill rule, censoring, features, models, scorecard) carries over unchanged to minute bars; only the horizon, the distances and the rolling windows in `ticktock/config.py` and `ticktock/features.py` would change. SPY and VIX were not available, so market mood is measured from the 20 stocks themselves (equal-weight index, volatility, cross-sectional dispersion).

## How the dataset is built

`ticktock/dataset.py` makes one row per **hypothetical limit order**:

1. For every stock and trading day, place buy orders below the close and sell orders above it at 0.5, 1, 2, 3, 5 and 8%.
2. **Fill rule (conservative):** a buy at limit L fills on the first later session whose low is ≤ L − $0.01; a sell when the high is ≥ L + $0.01. Just touching the price is not enough, because other orders were queued there first.
3. **Wait** is the number of trading days to the fill. Orders not filled within 40 days, or before the data ends, are **right-censored** (we know only that the wait was at least that long).
4. Features use only information available at the close of the start day.

This gives 1,161,432 orders. **Splits** are by start date: train Nov 2006 – Dec 2019 (741k orders), tune 2020 – 2021 (121k, includes the COVID crash), test Jan 2022 – Oct 2026 (289k). Training orders whose 40-day outcome window would run into the next period are purged, so labels never leak across splits. The final models are refit on everything whose outcome was known before 2022 and scored once on the test period.

**Split-adjusted prices.** Early prices in the 20-year files are split-adjusted and can be tiny (NVDA is $0.15 in 2006). One real cent was far smaller than one adjusted cent then, so the one-tick fill rule uses $0.01 capped at 2 basis points of the price, and hypothetical limits are rounded to 4 decimals.

## Features

| Group | Features |
|---|---|
| Distance | log-distance to the fill level, **distance in volatility units** (the strongest predictor) |
| Volatility | blend of close-to-close and high-low (Parkinson) volatility over 20 days; 5-day vs 20-day ratio; today's range |
| Traffic | 5-day and 1-day volume relative to the stock's own trailing 50-day baseline |
| Trend | 5- and 20-day returns signed *toward* the target (in volatility units), opening gap, room to the 20-day high/low |
| Market | 20-day market trend, market stress (market volatility z-score against its own past), volatility change, dispersion |
| Calendar | Fed meeting within the next 10 sessions, monthly options expiry |
| **Risk factors** | see below |
| Sector | one-hot, from the risk sheet |

### Risk factors as variables

`ticktock/risk_factors.py` maps each company's risk text onto 11 themes with transparent keyword rules, so every flag can be traced to a phrase: regulatory, trade & geopolitics, supply chain, AI/tech shift, rates & macro, consumer demand, litigation, competition, M&A integration, commodities & energy, operational. From these it builds:

- a 0/1 flag per theme;
- `risk_count` (how many themes), `risk_new_count` (themes new in 2025);
- `risk_macro_exposure` (themes that move with the whole market: trade, rates, consumer, commodities);
- interactions with market stress (`riskx_macro_stress`, `riskx_trade_stress`, `riskx_count_stress`), because company risk should matter most when the whole market is stressed.

The 2024 text applies to 2024 orders and the 2025 text to 2025 and 2026 orders. The sheet has no earlier text, so orders from 2006–2023 use the 2024 text as a fixed company profile. That is a mild look-ahead for those years (the text was written later), so read the risk effect as a company-profile effect.

## Models

All models return the full waiting-time CDF on days 0–40, so they share one scorecard (`ticktock/models/`).

| # | Model | Idea |
|---|---|---|
| 1 | **History** (Kaplan–Meier) | Past waits for orders of similar distance (in volatility units), traffic level and side |
| 2 | **Random-walk theory** (inverse Gaussian) | First passage of a driftless random walk: P(filled by t) = 2·(1 − Φ(d / σ√t)). Uses only distance and volatility |
| 3 | **Theory + traffic** | Same formula, with volatility rescaled by traffic, market stress, calendar and risk features; fitted on the censored waiting-time likelihood |
| 3b | **Weibull survival (AFT)** | A fitted, theory-free waiting-time curve. The scale depends on every feature; the shape depends on distance, side and market stress |
| 3b′ | Weibull without risk factors | Ablation, to measure what the risk factors add |

Models are fitted by maximising the interval-censored likelihood (a fill on day k means the price crossed during (k−1, k]). There is no dependency on `lifelines`; the fitting code is in `ticktock/models/base.py`.

## Results (Jan 2022 – Oct 2026, held out)

**Accuracy** (the number shown on the website): for every test order we ask "will it fill within 1, 5, 10 and 20 trading days?". The model answers yes when its probability is at least 50%, and accuracy is the share of correct answers, averaged over the four questions. The **baseline** always gives the most common answer and scores **63.3%**.

| Model | Accuracy | Acc. ≤10d | 50% range hit | 80% range hit | Brier ≤ 5d | Brier ≤ 20d | Censored LL | ms / prediction |
|---|---|---|---|---|---|---|---|---|
| History (Kaplan–Meier) | **77.0%** | 74.9% | 49.7% | 79.8% | 0.1647 | 0.1667 | -2.349 | 0.0019 |
| Random-walk theory | **77.0%** | 74.9% | 47.9% | 77.6% | 0.1650 | 0.1678 | -2.376 | 0.0013 |
| Theory + traffic | **77.1%** | 75.0% | 48.8% | 78.6% | 0.1640 | 0.1668 | -2.363 | 0.0019 |
| Weibull (no risk factors) | **77.0%** | 74.8% | 50.4% | 80.8% | 0.1645 | 0.1667 | -2.374 | 0.0012 |
| Weibull + risk factors | **77.1%** | 74.9% | 50.4% | 80.5% | 0.1643 | 0.1663 | -2.375 | 0.0025 |

Range-hit targets are 50% and 80% (closer is better). The full log is in `EXPERIMENTS.md`, slices in `reports/test_slices.csv` (by year, 2022 bear market, April 2025, traffic, events) and `reports/tune_slices.csv` (COVID crash).

What we found:

- **Every model is right about 77% of the time, about 14 points above the baseline.** The yes/no accuracies are within 0.2 points of each other, because the dominant signal (distance measured in volatility units) is shared by all of them.
- **The models differ in calibration, not in accuracy.** Both Weibull models hit their 50% and 80% ranges almost exactly; the pure random-walk theory is too narrow, as the research literature predicts. The history model has the best censored log-likelihood.
- **Calibration holds year by year.** The Weibull models' 80% range hit stays between 79.6% and 81.3% in every test year from 2022 to 2026.
- **Risk factors add almost nothing** (Weibull with vs. without: within 0.1 points), consistent with the earlier one-year build. With only 2024/2025 text they act as company identities.
- **Stress tests:** accuracy drops to 75–76% in the 2022 bear market and 73–74% in the April 2025 tariff shock, where the 80% range hit falls to 71–77% (Weibull models degrade least).
- The demo's default model is chosen on the tune set (most accurate there): Weibull + risk factors.

## Project layout

```
ticktock/
  config.py          tickers, horizon, distances, splits
  data.py            price loading, equal-weight market index, events
  risk_factors.py    risk text -> themes -> numeric features
  features.py        daily and order-level features (no look-ahead)
  dataset.py         hypothetical orders, fill rule, censoring, splits
  models/            Kaplan-Meier, inverse Gaussian (theory, + traffic), Weibull AFT
  evaluate.py        scorecard: coverage, Brier, censored LL, speed, slices
  compare.py         models vs reality: predicted vs real fill curves and monthly timelines
  service.py         prediction service used by the web demo
scripts/run_pipeline.py   dataset -> fit -> tune/test scorecards -> saved models
app/server.py        FastAPI backend (/api/info, /api/predict, /api/history, /api/scorecard, /api/comparison)
app/static/          single-page front end (plain JS, SVG charts, light and dark)
models/              trained models (pickles) + meta.json
reports/             scorecards, slices, reliability data, coefficients
EXPERIMENTS.md       experiment log
docs/TICKTOCK_PROJECT.md   original project brief
```

## API

`GET /api/predict?ticker=AAPL&side=buy&target=215&as_of=2025-04-02&model=weibull_aft` returns the CDF, quantiles (`q10`…`q90`, `null` = beyond 40 days), `p_within`, `p_beyond_horizon`, traffic and market gauges, the trade-off curve, every model's summary, the main drivers, the risk-factor panel, the actual outcome (for replay dates) and a one-paragraph briefing.

## Limitations and next steps

- Daily bars only: swap in 1-minute bars (Alpaca or Databento) to forecast intraday waits as in the original brief.
- Add SPY, VIX (FRED `VIXCLS`), earnings dates and CPI release dates as features.
- A price trading through your limit doesn't guarantee a fill; Nasdaq ITCH queue data could measure that gap.
- Planned extras from the brief: Chronos forecaster comparison, Tiger Data storage, Gemini for natural-language requests, ElevenLabs spoken briefing.

Research prototype, not financial advice.
