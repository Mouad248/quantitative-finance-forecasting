import pandas as pd
from typing import Optional, Literal

from quantopt.optimization.base import BaseOptimizer
from quantopt.optimization.efficient_frontier import EfficientFrontier
from quantopt.optimization.risk_parity import RiskParity
from quantopt.optimization.cvar_optimizer import CVaROptimizer


class OptimizerFactory:
    """
    Fluent factory for constructing and injecting QuantOpt optimizers.
    Primarily utilized by WalkForwardBacktester to dispatch optimizations
    programmatically.
    """

    @staticmethod
    def efficient_frontier(
        mu: pd.Series, Sigma: pd.DataFrame, **kwargs
    ) -> EfficientFrontier:
        """Return a ready-to-run EfficientFrontier instance."""
        return EfficientFrontier(mu, Sigma, **kwargs)

    @staticmethod
    def risk_parity(Sigma: pd.DataFrame, **kwargs) -> RiskParity:
        """Return a ready-to-run RiskParity instance."""
        return RiskParity(Sigma, **kwargs)

    @staticmethod
    def cvar(returns: pd.DataFrame, **kwargs) -> CVaROptimizer:
        """Return a ready-to-run CVaROptimizer instance."""
        return CVaROptimizer(returns, **kwargs)

    def build(
        self,
        strategy: Literal["max_sharpe", "min_vol", "risk_parity", "cvar"],
        mu: Optional[pd.Series] = None,
        Sigma: Optional[pd.DataFrame] = None,
        returns: Optional[pd.DataFrame] = None,
        rf: float = 0.0,
        **kwargs,
    ) -> BaseOptimizer:
        """
        Construct and fit an optimizer dynamically from a strategy name.

        Parameters
        ----------
        strategy : {"max_sharpe", "min_vol", "risk_parity", "cvar"}
            The algorithm to run.
        mu : pd.Series, optional
            Required for max_sharpe and min_vol.
        Sigma : pd.DataFrame, optional
            Required for max_sharpe, min_vol, and risk_parity.
        returns : pd.DataFrame, optional
            Required for cvar.
        rf : float, default 0.0
            Risk free rate used for max_sharpe.
        kwargs :
            Passed directly to the optimizer constructor.

        Returns
        -------
        BaseOptimizer
            A fully fitted optimizer instance (weights_ are populated).
        
        Raises
        ------
        ValueError
            If required inputs for the selected strategy are missing, or strategy string is unknown.
        """
        
        if strategy == "max_sharpe":
            if mu is None or Sigma is None:
                raise ValueError("max_sharpe requires both `mu` and `Sigma` parameters.")
            opt = self.efficient_frontier(mu, Sigma, **kwargs)
            opt.max_sharpe(risk_free_rate=rf)
            return opt
            
        elif strategy == "min_vol":
            if mu is None or Sigma is None:
                raise ValueError("min_vol requires both `mu` and `Sigma` parameters.")
            opt = self.efficient_frontier(mu, Sigma, **kwargs)
            opt.min_volatility()
            return opt
            
        elif strategy == "risk_parity":
            if Sigma is None:
                raise ValueError("risk_parity requires the `Sigma` parameter.")
            opt = self.risk_parity(Sigma, **kwargs)
            opt.optimize()
            return opt
            
        elif strategy == "cvar":
            if returns is None:
                raise ValueError("cvar requires the `returns` parameter (T x N DataFrame).")
            opt = self.cvar(returns, **kwargs)
            opt.optimize()
            return opt
            
        else:
            valid = ["max_sharpe", "min_vol", "risk_parity", "cvar"]
            raise ValueError(f"Unknown strategy '{strategy}'. Valid options are: {valid}")
