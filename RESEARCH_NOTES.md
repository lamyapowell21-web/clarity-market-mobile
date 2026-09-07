# Clarity evidence engine — research basis

Clarity v11 deliberately separates **feature selection** from **historical validation**.
The academic literature tells Clarity what kinds of signals are reasonable to test; the daily backtest determines whether those signals actually had useful out-of-sample behavior in the recent universe Clarity watches.

## What the model tests

- **Intermediate momentum:** 20- and 60-trading-day returns and moving-average position. Jegadeesh & Titman (1993) documented intermediate-horizon momentum, but that paper does not prove a 5-minute trade will continue, so Clarity treats daily momentum only as context.
- **Trading volume:** daily and intraday relative volume. Lee & Swaminathan (2000) found that volume contains information about momentum persistence and reversal.
- **Trading-range / breakout information:** distance from a recent short-term high. Brock, Lakonishok & LeBaron (1992) found historical information in moving-average and trading-range-break rules.
- **Systematic technical evidence rather than visual guessing:** Lo, Mamaysky & Wang (2000) showed that technical patterns can be tested algorithmically and can carry incremental information.
- **Intraday context and timing:** market return, volatility/range and minutes from the open. Gao, Han, Li & Zhou (2018) documented intraday market momentum, while Heston, Korajczyk & Sadka (2010) documented strong intraday return patterns and short-horizon reversals linked to liquidity effects.

## How Clarity avoids pretending the backtest is proof

- Samples are ordered chronologically.
- The earliest 60% fits the logistic model.
- The next 20% chooses the alert probability threshold.
- The final 20% is untouched until evaluation.
- If the chosen threshold does not produce positive average R in the untouched test with enough alerts, `ready_for_alerts` is false and the evidence-backed scanner does not send live alerts.
- Five-minute bars where both the stop and target are touched are discarded because OHLC data cannot tell which happened first.
- The default backtest transaction-cost assumption is 0 bps because no cost/slippage assumption was supplied. This is shown explicitly in `cloud_config.json` and can be changed.

This design is intentionally conservative because financial backtests are vulnerable to overfitting. Bailey et al. (2015) specifically discuss the problem of selecting strategies using historical simulations and the risk that apparent performance degrades out of sample.

## Sources

- Jegadeesh, N. & Titman, S. (1993), *Returns to Buying Winners and Selling Losers: Implications for Stock Market Efficiency*, Journal of Finance. https://doi.org/10.1111/j.1540-6261.1993.tb04702.x
- Lee, C.M.C. & Swaminathan, B. (2000), *Price Momentum and Trading Volume*, Journal of Finance. https://doi.org/10.1111/0022-1082.00280
- Brock, W., Lakonishok, J. & LeBaron, B. (1992), *Simple Technical Trading Rules and the Stochastic Properties of Stock Returns*, Journal of Finance. https://doi.org/10.1111/j.1540-6261.1992.tb04681.x
- Lo, A.W., Mamaysky, H. & Wang, J. (2000), *Foundations of Technical Analysis: Computational Algorithms, Statistical Inference, and Empirical Implementation*, Journal of Finance. https://doi.org/10.1111/0022-1082.00265
- Gao, L., Han, Y., Li, S.Z. & Zhou, G. (2018), *Market Intraday Momentum*, Journal of Financial Economics. https://doi.org/10.1016/j.jfineco.2018.05.009
- Heston, S.L., Korajczyk, R.A. & Sadka, R. (2010), *Intraday Patterns in the Cross-section of Stock Returns*, Journal of Finance. https://doi.org/10.1111/j.1540-6261.2010.01573.x
- Bailey, D.H., Borwein, J., López de Prado, M. & Zhu, Q.J. (2015), *The Probability of Backtest Overfitting*, Journal of Computational Finance / SSRN. https://ssrn.com/abstract=2326253
- Barber, B.M. & Odean, T. (2000), *Trading Is Hazardous to Your Wealth*, Journal of Finance. https://doi.org/10.1111/0022-1082.00226
