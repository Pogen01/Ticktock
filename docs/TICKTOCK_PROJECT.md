# TickTock — Project Instructions

Use this file as (a) the instructions for the team's Claude project and (b) the build plan for the coding agent in Delta. Read it fully before answering or writing code.

---

## 1. What TickTock is

When a trader wants a better price than the current one, they place a **limit order** and wait. Today their app just says "pending". TickTock tells them **how long they're likely to wait**:

> "You want AAPL at $228.00; it's $228.40 now. Expected wait: **4 to 15 minutes** (80% range). **65%** chance of filling within 30 minutes, **20%** chance it doesn't fill today. The market is busy right now, so waits are shorter than usual."

The user can drag their desired price and see the trade-off: **better price = longer, less certain wait.**

The core idea: **market traffic** (how busy trading is) changes how much prices move, which changes the waiting-time curve. We model that and show it.

**Pitch line:** "Big trading firms have had fill-time models for 20 years, including research here in Oxford. Retail traders still get 'pending'. TickTock gives everyone a forecast of how long they'll wait for their price."

**We do not predict price direction.** We predict the *waiting time distribution* (a range plus probabilities). Say this clearly whenever asked.

---

## 2. Context and constraints

- 4-person team at **OxHack 2026** (Oxford), theme **"Time"**, demos on **Sunday 25 October 2026**. Time is the main constraint: build the smallest complete version first.
- The team is new to trading. Explain terms in plain language the first time they appear.
- **HRT challenge** (main target): "best use of data for predictions and decision making", **built and collaborated on in Delta**. Judged: speed & performance 30%, technical implementation 25%, innovation 25%, polish & usability 20%. Do the work in Delta and save/share the Delta threads.
- Other prizes: **Tiger Data** (store bars and results as time series), **Vultr** (compute; $100 credits), **Gemini API** (must power the product, not just chat), **ElevenLabs** (spoken briefing), Hackiest Hack, MLH DEV post, demo sticker.

---

## 3. Data

### Stocks
The **20 largest US companies by market value at the start of 2024** (not today's ranking, to avoid hindsight). Provisional list, **verify ranks 18–20 against a market-cap snapshot from 29 Dec 2023**:

AAPL, MSFT, GOOGL, AMZN, NVDA, META, TSLA, BRK.B, LLY, V, AVGO, JPM, UNH, WMT, XOM, MA, JNJ, PG, HD, plus one of COST / MRK / ORCL.

Note: BRK.B is written differently by different APIs (`BRK.B`, `BRK-B`, `BRK/B`).

### Sources
| Data | Source | Notes |
|---|---|---|
| 1-minute bars, Jan 2024 – Dec 2025 | **Alpaca** free plan (history to 2016, 200 requests/min) | Free plan = **IEX feed only** (~2.5% of US volume). Prices are fine; volume is a partial traffic measure. Upgrade option: **Databento** 1-minute bars with the $125 free credits (full market) |
| Market-wide traffic | SPY 1-minute bars, same source | |
| Fear index | **VIX** daily close from FRED (series `VIXCLS`) | Use the *previous* day's value |
| Earnings dates | Any public earnings calendar | Hand-check a few |
| Fed decisions and CPI releases | federalreserve.gov and bls.gov schedules | ~20 dates/year; hand-enter |
| Queue check (optional) | Nasdaq ITCH free archive, 8–12 Dec 2025 | For measuring "price touched" vs "really filled" |

### Time handling (important, and on-theme)
- Keep only regular hours **09:30–16:00 America/New_York**. Store timestamps in UTC, convert with a proper timezone library so US daylight-saving changes are handled correctly.
- Flag **half days** (13:00 close).
- Drop or mark missing minutes; IEX bars have gaps in quiet periods.

---

## 4. Building the waiting-time dataset (everything depends on this)

For each stock and trading day:

1. Sample random start minutes (e.g. 30 per stock per day, avoiding the last 15 minutes).
2. At each start minute, create hypothetical **buy** orders below the current price and **sell** orders above it, at distances of **0.05%, 0.1%, 0.25%, 0.5%**.
3. **Fill rule (conservative):** a buy at limit *L* counts as filled in the first later minute whose **low ≤ L − $0.01**; a sell when **high ≥ L + $0.01**. ("Touching" isn't enough, because others were queued at that price first.)
4. **Wait** = minutes from start to fill. If not filled by 16:00 → **censored** ("unfilled today", waited at least this long). No overnight carry.
5. Record features **known at the start minute only** (never use later information):

| Feature | How |
|---|---|
| Traffic | Volume and trade count over last 5 and 15 min, ÷ that stock's typical value for that time of day (computed from training data only) |
| Recent volatility | Std. dev. of 1-min log returns over last 30 min |
| Distance in volatility units | log-distance ÷ (volatility × √minutes) — likely the strongest predictor |
| Time of day | Minutes since 09:30; flags for first and last 30 min; minutes left until close |
| Overnight gap | Today's open vs yesterday's close |
| Market mood | Same traffic and volatility features for SPY |
| VIX | Previous day's close |
| Recent trend | Return over last 30–60 min (weak; direction) |
| Event flags | Earnings day/day after, Fed day, CPI day, third-Friday options expiry, half day |
| Stock, side, distance | Categorical / numeric |

Save as Parquet: one row per hypothetical order.

### Splits
- **Train:** Jan–Sep 2024 · **Tune:** Oct–Dec 2024 · **Test:** all of 2025 (use once, at the end).
- 2025 includes the **April 2025 tariff sell-off**: report it as a stress test.

---

## 5. Models (keep to this list)

| Step | Model | Question it answers | Tool |
|---|---|---|---|
| 1 | **History** (Kaplan–Meier curves by distance, optionally by traffic level) | What did past waits look like? Baseline | `lifelines` |
| 2 | **Inverse Gaussian from theory** | Does textbook random-walk maths explain waits? Uses only distance and recent volatility | `scipy` |
| 3 | **Inverse Gaussian + traffic** | Does traffic change the curve? Volatility (and drift) predicted from the features, then plugged into the formula | `scipy` + a small regression |
| 3b | **Weibull survival model with the same features** | Does a fitted, theory-free curve beat the theory? Research says pure first-passage models underpredict real fill times, so this is the likely winner | `lifelines` `WeibullAFTFitter` |
| 4 | **Chronos-2 / Chronos-Bolt** (LLM-style forecaster) | Can a big pretrained model beat our small one? Sample future price paths, record when each first reaches the fill level | `chronos-forecasting` |
| Stretch | Bayesian hierarchical version of step 3 | How much does traffic matter for each stock? | PyMC |

### Inverse Gaussian notes
- Work in log-prices. Let *d* = log-distance to the fill level (including the 1-cent rule), *σ* = volatility per minute, *μ* = drift per minute toward the target.
- If *μ* > 0: time to first reach the level ~ Inverse Gaussian with mean *d/μ* and shape *d²/σ²*.
- If *μ* ≈ 0: P(filled by *t*) = 2 · (1 − Φ(*d* / (σ√*t*))).
- P(unfilled today) = 1 − P(filled by minutes-until-close). Quantiles of the waiting time come from inverting the CDF; probability mass beyond the close is "not today".

### Chronos notes
- Run on a fixed sample (e.g. 500 test orders) and record time per prediction.
- Normalise prices; don't pass ticker names or dates.

---

## 6. Scorecard (same for every model, on 2025)

1. **Calibration (headline):** share of real waits falling inside the predicted 50% and 80% ranges. Target ≈ 50% and 80%.
2. **Fill-within-time accuracy:** Brier score for "filled within 5 / 30 / 60 minutes", plus a reliability plot.
3. **Censored log-likelihood** (a proper scoring rule for survival models).
4. **Speed:** milliseconds per prediction.
5. **Slices:** quiet vs busy traffic; April 2025; event days.

### Experiment log
Keep `EXPERIMENTS.md`, one row per run:

| ID | Model | Features | Trained on | Settings | 50% cov. | 80% cov. | Brier@30m | Censored LL | ms/pred | Notes | Delta thread |
|---|---|---|---|---|---|---|---|---|---|---|---|

Fix random seeds. Never tune on 2025.

---

## 7. Storage (Tiger Data)

- `bars_1m` hypertable (time, symbol, OHLCV, trade count), segment by symbol.
- Continuous aggregate `traffic_5m` (volume, trades, volatility per 5 minutes per symbol).
- `orders_dataset` and `predictions` tables for the dashboard.
- Connection string in `.env` (git-ignored). Provide `.env.example`.

---

## 8. The app / demo

- **Inputs:** stock, side (buy/sell), desired price (or slider as % from current), a replay date and time from 2025 (markets are closed on demo day).
- **Outputs:** predicted wait range (50% and 80%), chance of filling within 5 / 30 / 60 min, chance of not filling today, and a live **traffic gauge** (quiet / normal / busy vs that stock's usual).
- **Trade-off chart:** desired price vs expected wait.
- **Replay mode:** step through a real 2025 day and watch the forecast change as traffic changes (e.g. a Fed announcement).
- **Gemini:** turns requests like "I want Tesla a bit cheaper but can't wait more than 10 minutes" into a suggested limit price using the model's outputs. Gemini never invents numbers; it only chooses from model outputs.
- **ElevenLabs:** reads a short briefing ("NVDA is busy right now. A limit 0.2% below should fill in about 6 minutes.").
- **Speed matters** (30% of HRT score): precompute where possible; the inverse Gaussian is a closed-form formula, so predictions should be instant.
- Show "Research prototype, not financial advice."

---

## 9. Build order and roles

| Milestone | Owner | Done when |
|---|---|---|
| M1 Data download (20 stocks + SPY, 2024–2025) into Parquet and Tiger | Person 1 | Row counts per symbol per day look right |
| M2 Waiting-time dataset builder | Person 1 → 2 | Parquet with orders, waits, censoring and features; sanity plots |
| M3 Steps 1 and 2 + scorecard | Person 2 | First calibration chart on the tune set |
| M4 Steps 3 and 3b (traffic) | Person 3 | Traffic effect shown; scorecard rows logged |
| M5 Dashboard skeleton in Delta using step 2 | Person 4 | End-to-end demo with any model |
| M6 Swap in best model; replay mode; Gemini; ElevenLabs | Person 4 | Demo flows smoothly |
| M7 Chronos comparison | Person 3 | 500-order comparison row in the log |
| M8 Final test on 2025, slides, rehearsal ×2, submit early | All | Results slide from `EXPERIMENTS.md` |

If time runs out after M5, the project is still complete. Everything after is improvement.

---

## 10. Known weaknesses and prepared answers

| Judge question | Answer |
|---|---|
| "Can you predict direction?" | No, and we don't try. We predict waiting time, which is what matters to someone placing a limit order |
| "Price touching your level isn't a fill" | Correct; we require the price to go 1 cent past, and (if done) measured the real gap with Nasdaq ITCH queue data |
| "First-passage models are known to be poor" (Lo, MacKinlay & Zhang 2002) | Yes; that's why the theory model is our baseline and we show how much traffic-aware and fitted survival models improve on it |
| "Your volume is only IEX" | Prices are reliable; volume is a consistent relative traffic measure. Full-market data (Databento) is the upgrade |
| "Isn't this already done?" | In research and institutional tools, yes (Lo et al. 2002; Columbia 2021; Oxford-Man Institute 2023). Not for retail users as a waiting-time range. That's the gap |

### Related work to cite
- Lo, MacKinlay & Zhang (2002), *Econometric Models of Limit-Order Executions*, Journal of Financial Economics.
- Maglaras, Moallemi & Wang, deep learning estimation of fill probabilities (Columbia).
- Arroyo, Cartea, Moreno-Pino & Zohren (2023), *Deep Attentive Survival Analysis in Limit Order Books*, Oxford-Man Institute.
- US Patent 8,645,259 (filed 2008): limit order execution probability for users (random-walk model).

---

## 11. How Claude should work on this project

- Short, practical answers; lead with the answer.
- Smallest working version first, then improve.
- Look up anything about current APIs, libraries or data availability instead of relying on memory.
- Never use information from after the prediction moment in features (no look-ahead).
- Flag judge-facing weak points with a suggested answer.
- Remind the team if work drifts outside Delta.
