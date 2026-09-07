# Clarity v12 evidence-engine research notes

Clarity v12 is a **research and paper-planning system**, not a guarantee of profit and not an automated trader. The v12 change is mainly methodological: it makes the historical test look more like the alert behavior a user actually experiences and makes the “Ready” gate much harder to pass.

## Why these feature families remain

Clarity keeps a deliberately small set of interpretable feature families rather than adding dozens of technical indicators. The goal is to reduce data-mining risk.

- **Intermediate momentum**: Jegadeesh & Titman (1993), *Returns to Buying Winners and Selling Losers*, documents persistence in past winners over longer horizons. This supports including momentum context, but it does **not** prove that a 15-minute move will continue.
- **Moving-average / trading-range context**: Brock, Lakonishok & LeBaron (1992) historically tested moving-average and trading-range-break rules. Clarity treats these as candidate information, not as laws.
- **Trading frictions and overtrading**: Barber & Odean (2000) documents the performance penalty associated with very active individual-investor trading. v12 therefore includes execution-drag scenarios and deliberately suppresses repeat alerts.
- **Backtest overfitting**: Bailey et al., *The Probability of Backtest Overfitting*, motivates keeping model/threshold selection separate from a later untouched period and limiting how aggressively Clarity searches thresholds.

Reference pages:
- https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1993.tb04702.x
- https://onlinelibrary.wiley.com/doi/10.1111/j.1540-6261.1992.tb04681.x
- https://faculty.haas.berkeley.edu/odean/papers%20current%20versions/individual_investor_performance_final.pdf
- https://papers.ssrn.com/sol3/papers.cfm?abstract_id=2326253

## What changed from v11

### 1. Whole-day time splits
v11 sorted individual five-minute rows and split 60/20/20. That can put highly correlated observations from the same trading day into adjacent model-selection sets.

v12 splits by **complete trading dates**. A market day belongs entirely to training, threshold-selection, or untouched testing.

### 2. Session-aware intraday features
v11 could let an early-morning rolling window reach backward into the previous trading session. v12 calculates 15/60-minute momentum from the **current session only** and normalizes recent volume against roughly the **same clock-time slot on prior sessions** when enough history exists. This avoids confusing the overnight gap or the market's normal opening-volume surge with a fresh intraday signal.

### 3. The model target is less sparse
v11 trained the classifier to predict whether the full 2R target was hit first. The first real run produced only a 4% target-first rate, making that target too sparse for a small recent-history model.

v12 predicts whether the planned setup ends with **positive net R after the base execution-drag scenario**. Full target-hit rate remains a diagnostic, but is not the classifier target.

### 4. Alert events, not every qualifying bar
v11’s historical “alert count” was really the number of qualifying five-minute snapshots. Consecutive bars could therefore be counted as separate observations even though a live user would experience them as one setup.

v12 reconstructs the live alert rule:
- probability crosses the chosen line, or makes a large surge above it;
- short-term direction guardrail must pass;
- same ticker enters a 60-minute cooldown;
- if several tickers cross on the same scan timestamp, only the highest-evidence name is counted.

These events are still not perfectly statistically independent because different stocks can move together. v12 therefore also reports the number of distinct event days and a day-clustered return check.

### 5. Friction + stress scenarios
The default configuration uses:
- **5 bps round-trip** as the base execution-drag scenario;
- **15 bps round-trip** as a stress scenario.

These are scenario assumptions for spread/slippage/execution drag. They are **not claims about Robinhood’s fees or the user’s actual fills**. If measured personal execution data becomes available, these should be replaced with empirically observed assumptions.

### 6. A genuinely stricter Ready gate
The untouched test must pass **all** configured gates, including:
- enough separated alerts;
- enough distinct trading days;
- average R above a material minimum after base drag;
- positive average R under the stress-drag scenario;
- a positive day-clustered lower-bound check;
- a higher positive-outcome rate than comparable positive-momentum historical snapshots;
- model AUC above a small minimum.

A merely positive +0.01R result can no longer turn the app green.

### 7. Human-facing percentile instead of raw classifier probability
Classifier probabilities can have unintuitive numeric levels. v12 stores the probability distribution from the untouched historical period and converts the live model output to an **evidence percentile**.

For example, “92nd percentile” means the model score is higher than roughly 92% of recent historical watchlist snapshots in that reference period. It is **not** a 92% chance of profit.

## Important remaining limitations

- yfinance is convenient research data and may be delayed, incomplete, or revised. Its documentation says intraday data cannot extend beyond the last 60 days, so Clarity uses nearly that full recent window rather than pretending to have years of five-minute history.
- A recent 56-calendar-day model is regime-sensitive and is not a substitute for a multi-year institutional-grade database.
- Five-minute OHLC bars cannot identify target-versus-stop ordering when both prices occur within the same bar; those ambiguous samples are removed.
- Same-day and cross-stock correlations remain. The “day-clustered” check helps, but does not make the test perfectly independent.
- Threshold selection still uses one historical validation segment. The final untouched segment helps protect against selection bias but cannot eliminate backtest overfitting.
- Live execution, latency, spreads, and human response time can make real outcomes worse than a historical simulation.

The intended behavior is conservative: when the evidence is weak, **Clarity should say no and stay quiet**.

- yfinance download documentation (including the 60-day intraday limit): https://ranaroussi.github.io/yfinance/reference/api/yfinance.download.html
