import numpy as np
import pandas as pd
import pytest

from quantopt.backtest.engine import (
    WalkForwardBacktester,
    BacktestConfig,
    TransactionCostModel,
)
from quantopt.optimization.factory import OptimizerFactory

@pytest.fixture
def price_data():
    np.random.seed(42)
    dates = pd.date_range("2020-01-01", periods=1000, freq="B")
    rets = np.random.normal(0.0002, 0.01, (1000, 3))
    prices = pd.DataFrame(100 * np.exp(np.cumsum(rets, axis=0)), index=dates, columns=["A", "B", "C"])
    return prices

def test_transaction_cost_model():
    tc = TransactionCostModel(proportional=0.001, fixed=0.0, market_impact=0.0)
    
    # 50% turnover
    delta = np.array([0.5, -0.5])
    cost = tc.compute(delta)
    
    # total volume = 1.0. cost = 1.0 * 0.001 = 0.001
    assert cost == pytest.approx(0.001)

def test_walk_forward_backtest(price_data):
    config = BacktestConfig(
        lookback_days=100,
        rebalance_freq="ME", # Month end
        transaction_cost=TransactionCostModel(proportional=0.001),
        max_turnover=0.5
    )
    
    factory = OptimizerFactory()
    
    # Simple risk parity strategy
    def strategy(window_returns):
        from quantopt.risk.covariance import SampleCovariance
        S = SampleCovariance(frequency=252).fit(window_returns).covariance()
        return factory.build("risk_parity", Sigma=S)
        
    engine = WalkForwardBacktester(
        prices=price_data,
        optimizer_factory=strategy,
        config=config,
    )
    
    res = engine.run()
    
    # Check returns are reasonable
    assert len(res.portfolio_returns) == len(price_data) - config.lookback_days - 1
    assert len(res.weights_history) > 0 # At least some months
    assert res.transaction_costs.sum() > 0 # Paid some fees
    
    # Max turnover checked
    assert res.turnover_history.max() <= 0.5 + 1e-4
    
def test_walk_forward_comparison(price_data):
    config = BacktestConfig(
        lookback_days=100,
        rebalance_freq="ME",
        transaction_cost=TransactionCostModel(proportional=0.000)
    )
    
    factory = OptimizerFactory()
    
    def strategy_ew(window_returns):
        # We can implement EW as Risk Parity with Identity cov
        n = window_returns.shape[1]
        I = pd.DataFrame(np.eye(n), index=window_returns.columns, columns=window_returns.columns)
        return factory.build("risk_parity", Sigma=I)
        
    strategies = {"Equal_Weight": strategy_ew}
    
    engine = WalkForwardBacktester(
        prices=price_data,
        optimizer_factory=None, # used per strategies dict
        config=config,
    )
    
    results = engine.run_comparison(strategies, config)
    
    assert "Equal_Weight" in results
    res = results["Equal_Weight"]
    
    assert len(res.summary.columns) == 1
    assert res.summary.columns[0] == "Equal_Weight"
