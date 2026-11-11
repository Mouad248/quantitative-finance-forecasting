"""Offline example: seeded synthetic prices, no market downloads."""
import json
from pathlib import Path
import numpy as np
import pandas as pd
import quantopt as qo


def main():
    np.random.seed(42)
    rng = np.random.default_rng(42)
    dates = pd.bdate_range("2020-01-01", periods=504)
    columns = list("ABCD")
    daily = rng.normal(0.0003, [0.008, 0.012, 0.016, 0.02], (504, 4))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(daily, axis=0)), index=dates, columns=columns)
    training = qo.prices_to_returns(prices, method="simple").iloc[:126]
    mu = qo.MeanHistoricalReturn().fit(training).expected_returns()
    cov = qo.LedoitWolfCovariance().fit(training).covariance()
    minimum = qo.EfficientFrontier(mu, cov).min_volatility()
    parity = qo.RiskParity(cov)
    parity_weights = parity.optimize()
    cvar = qo.CVaROptimizer(training)
    cvar_weights = cvar.optimize()

    def strategy(window):
        simple = np.expm1(window)  # Engine supplies log returns.
        expected = qo.MeanHistoricalReturn().fit(simple).expected_returns()
        covariance = qo.LedoitWolfCovariance().fit(simple).covariance()
        optimizer = qo.EfficientFrontier(expected, covariance)
        optimizer.min_volatility()
        return optimizer

    result = qo.WalkForwardBacktester(
        prices, strategy, qo.BacktestConfig(lookback_days=126, rebalance_freq="QE")
    ).run()
    record = {
        "data_kind": "seeded synthetic prices; not market performance",
        "seed": 42, "price_rows": len(prices), "assets": len(columns),
        "minimum_volatility_weights": minimum.to_dict(),
        "risk_parity_weights": parity_weights.to_dict(),
        "risk_parity_contributions": parity.risk_contributions().to_dict(),
        "cvar_weights": cvar_weights.to_dict(), "training_cvar": float(cvar.cvar_),
        "backtest": {
            "evaluation_days": len(result.portfolio_returns),
            "rebalance_dates": [str(d.date()) for d in result.rebalance_dates],
            "total_net_return": float(result.cumulative_returns.iloc[-1]),
            "sum_cost_fractions": float(result.transaction_costs.sum()),
            "assumption": "observe current close, trade at that close, new weights earn next day return",
        },
    }
    output = Path(__file__).parent / "reports" / "offline_demo.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(record["backtest"], indent=2))
    print("Synthetic results saved to reports/offline_demo.json")


if __name__ == "__main__":
    main()
