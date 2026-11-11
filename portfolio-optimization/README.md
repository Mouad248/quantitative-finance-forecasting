# Portfolio optimization and walk-forward backtesting

This project asks how to divide a portfolio across assets while balancing expected return, volatility and loss risk. It estimates returns and covariance, chooses asset weights, then simulates how a strategy behaves as prices change and trading costs are paid.

The main implementation is **`quantopt/`**. The `src/portopt/` and `portfolio_optimization/` folders contain an older, separate implementation. They remain for reference; the verified example uses `quantopt`.

## Run the offline example

From this project folder, on the existing local Python 3.13 installation:

```powershell
py -3.13 demo_test.py
```

The example creates 504 business-day price observations for four anonymous assets, using seed 42. It estimates weights on an initial training window, runs minimum-variance, risk-parity and CVaR optimization, then performs a quarterly minimum-variance backtest with a rolling 126-day estimation window. It writes aggregate results to `reports/offline_demo.json`. It fetches no data and needs no credentials.

All demo prices are **synthetic**. Its gains or losses illustrate program behavior and are not evidence of market performance. A fresh environment needs NumPy, pandas, SciPy and scikit-learn for this example; plotting additionally uses Matplotlib and Seaborn. Full dependencies are listed in `requirements.txt`.

## How the code fits together

1. `quantopt/returns/`: convert prices to returns and estimate expected returns using historical averages, CAPM or Black-Litterman inputs.
2. `quantopt/risk/`: estimate covariance using sample, exponentially weighted, shrinkage or factor methods; calculate volatility and risk contributions.
3. `quantopt/optimization/`: choose portfolio weights. Mean-variance methods seek minimum volatility, maximum Sharpe ratio or a return/risk target. Risk parity balances risk contributions rather than capital weights. CVaR optimization reduces average loss in the worst part of the scenario distribution.
4. `quantopt/backtest/engine.py`: repeatedly estimate a strategy on a rolling window, hold and drift the weights between trades, limit turnover if configured, and deduct trading costs.
5. `quantopt/analytics/` and `quantopt/plotting/`: summarize realized returns and draw charts.

Expected return and covariance inputs to mean-variance optimization must use consistent annualization. The backtest supplies **log-return estimation windows**; the example explicitly converts these to simple returns before fitting its estimators. Portfolio accounting and performance analytics use **simple returns**.

## Backtest timing and accounting

Each evaluation day first earns its simple return using opening portfolio weights. Weights then drift with asset prices. On a rebalance date, the estimator observes a window through that day's close and trades at that close; the new weights earn the following day's return. This assumes execution at the observed close, without an execution delay or slippage beyond the configured cost model.

Monthly and quarterly schedules use the last observed trading day of each resampled period, including the final partial period. Costs are fractions of wealth after that day's market move: `net = (1 + gross) * (1 - cost) - 1`. The backtest starts from equal weights unless specified, after the lookback period; initial funding costs are not charged. It supports fully invested, long-only target weights. Failed rebalances log a warning and retain drifted holdings. `weights_history` records rebalance targets; `realized_weights` records weights after the market move and before trading.

The recovery corrected return accounting, weight drift, calendar scheduling, duplicate fitting of an already fitted strategy, and the risk-parity analytical gradient. The legacy downloader now raises on failure; synthetic prices require an explicit mock option. The previously malformed `demo_test.py` was replaced with the runnable offline example.

## Verification

To run the tests using this workspace's existing pytest installation:

```powershell
$env:PYTHONPATH = (Resolve-Path '..\..\_workspace\runtime\python314\core').Path + ';' + (Get-Location).Path
py -3.13 -m pytest -q tests
```

**52 tests pass**. SciPy emitted 56 warnings about clipping trial solver steps to weight bounds; these remain visible. See `reports/verification.json` for the test results. Regression tests use known price paths to check opening weights, drift, cost timing, close-to-next-day trading and weekend month ends.

CVaR uses a nonsmooth sample objective with SLSQP. Optimized weights are sensitive to estimation errors, constraints and sample selection.

Personal author/profile labels were removed from active metadata and documentation. Original files and stale generated package metadata remain in ignored `.private/` storage. No CV or interview preparation files were added.
