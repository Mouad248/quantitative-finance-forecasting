"""Known-path accounting regressions independent of optimizer convergence."""
import numpy as np
import pandas as pd
import pytest

from quantopt.backtest.engine import WalkForwardBacktester, BacktestConfig, TransactionCostModel
from quantopt.optimization.base import BaseOptimizer


class FixedOptimizer(BaseOptimizer):
    def __init__(self, assets, weights):
        super().__init__(assets)
        self.weights_ = pd.Series(weights, index=assets)

    def optimize(self, **kwargs):
        raise AssertionError("A fitted optimizer must not be optimized again")


def make_prices():
    return pd.DataFrame({"A": [100., 100., 200., 100., 100.], "B": [100.] * 5},
                        index=pd.bdate_range("2024-01-25", periods=5))


def test_buy_and_hold_uses_opening_weights_and_carries_drift():
    prices = make_prices()
    factory = lambda window: FixedOptimizer(prices.columns, [0.5, 0.5])
    result = WalkForwardBacktester(prices, factory, BacktestConfig(lookback_days=1, rebalance_freq="YE")).run()
    np.testing.assert_allclose(result.gross_returns, [0.5, -1/3, 0], atol=1e-12)
    assert result.cumulative_returns.iloc[-1] == pytest.approx(0)
    assert list(result.weights_history.index) == result.rebalance_dates


def test_weekend_month_end_rebalances_on_last_observed_day():
    dates = pd.bdate_range("2024-03-25", periods=6)
    prices = pd.DataFrame(100.0, index=dates, columns=["A", "B"])
    seen = []
    def factory(window):
        seen.append(window.index[-1])
        return FixedOptimizer(prices.columns, [1.0, 0.0])
    result = WalkForwardBacktester(prices, factory, BacktestConfig(lookback_days=1, rebalance_freq="ME")).run()
    assert pd.Timestamp("2024-03-29") in result.rebalance_dates
    assert seen == result.rebalance_dates


def test_close_trades_affect_next_day_and_charge_cost_after_market_move():
    prices = make_prices()
    def factory(window):
        return FixedOptimizer(prices.columns, [1.0, 0.0])
    result = WalkForwardBacktester(prices, factory, BacktestConfig(
        lookback_days=1, rebalance_freq="D", transaction_cost=TransactionCostModel(proportional=0.01)
    )).run()
    assert result.gross_returns.iloc[0] == pytest.approx(0.5)
    assert result.portfolio_returns.iloc[0] == pytest.approx(1.5 * (1 - 0.01 * 2/3) - 1)
    assert result.gross_returns.iloc[1] == pytest.approx(-0.5)


@pytest.mark.parametrize("bad", [np.nan, np.inf, 0, -1])
def test_invalid_prices_fail(bad):
    prices = make_prices()
    prices.iloc[2, 0] = bad
    with pytest.raises(ValueError):
        WalkForwardBacktester(prices, None, BacktestConfig(lookback_days=1))


def test_insufficient_history_fails_clearly():
    with pytest.raises(ValueError, match="lookback_days"):
        WalkForwardBacktester(make_prices(), None, BacktestConfig(lookback_days=100))


@pytest.mark.parametrize("relative", ["src/portopt/data.py", "portfolio_optimization/src/portopt/data.py"])
def test_legacy_download_failure_never_substitutes_mock_data(monkeypatch, relative):
    import importlib.util
    import sys
    from pathlib import Path
    from types import SimpleNamespace
    path = Path(__file__).resolve().parents[1] / relative
    spec = importlib.util.spec_from_file_location("legacy_data", path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    monkeypatch.setitem(sys.modules, "yfinance", SimpleNamespace(download=lambda *a, **k: pd.DataFrame()))
    with pytest.raises(RuntimeError, match="Market-data download failed"):
        module.download_prices(["A"], "2020-01-01", "2020-02-01")
