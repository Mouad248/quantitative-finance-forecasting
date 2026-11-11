"""
QuantOpt

Institutional MPT portfolio optimizer: Black-Litterman returns, factor-model
risk, CVaR optimization, risk parity, and walk-forward backtesting.
"""

__version__ = "1.0.0"
__author__ = "Anonymous"

# Public API imports
# We use direct imports here for simplicity and because the library is relatively
# lightweight and does not have heavy binary startup costs outside of NumPy/Pandas.
from quantopt.returns.preprocessing import (
    prices_to_returns,
    returns_to_prices,
    annualization_factor,
)
from quantopt.returns.estimators import (
    MeanHistoricalReturn,
    CAPMReturn,
    BlackLittermanReturn,
)
from quantopt.risk.covariance import (
    SampleCovariance,
    EWMCovariance,
    LedoitWolfCovariance,
    FactorModelCovariance,
)
from quantopt.risk.metrics import (
    marginal_risk_contribution,
    component_risk_contribution,
    percent_risk_contribution,
    diversification_ratio,
    concentration_hhi,
    effective_n,
    risk_report,
)
from quantopt.optimization.efficient_frontier import (
    EfficientFrontier,
    OptimizationError,
    InfeasibleError,
)
from quantopt.optimization.risk_parity import RiskParity
from quantopt.optimization.cvar_optimizer import CVaROptimizer
from quantopt.optimization.factory import OptimizerFactory
from quantopt.analytics.performance import (
    performance_summary,
    sharpe_ratio,
    max_drawdown,
)
from quantopt.backtest.engine import (
    WalkForwardBacktester,
    BacktestConfig,
    TransactionCostModel,
)

__all__ = [
    # version
    "__version__",
    # returns
    "prices_to_returns",
    "returns_to_prices",
    "annualization_factor",
    "MeanHistoricalReturn",
    "CAPMReturn",
    "BlackLittermanReturn",
    # risk
    "SampleCovariance",
    "EWMCovariance",
    "LedoitWolfCovariance",
    "FactorModelCovariance",
    "marginal_risk_contribution",
    "component_risk_contribution",
    "percent_risk_contribution",
    "diversification_ratio",
    "concentration_hhi",
    "effective_n",
    "risk_report",
    # optimization
    "EfficientFrontier",
    "RiskParity",
    "CVaROptimizer",
    "OptimizerFactory",
    "OptimizationError",
    "InfeasibleError",
    # analytics
    "performance_summary",
    "sharpe_ratio",
    "max_drawdown",
    # backtest
    "WalkForwardBacktester",
    "BacktestConfig",
    "TransactionCostModel",
]
