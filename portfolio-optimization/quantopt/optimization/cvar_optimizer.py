import numpy as np
import pandas as pd
import scipy.optimize as sco
from typing import Optional, Tuple

from quantopt.optimization.base import BaseOptimizer
from quantopt.optimization.efficient_frontier import OptimizationError


class CVaROptimizer(BaseOptimizer):
    """
    Minimizes Conditional Value-at-Risk (Expected Shortfall) using the
    Rockafellar-Uryasev (2000) piecewise-linear sample objective with SLSQP.

    Parameters
    ----------
    returns : pd.DataFrame
        Historical or Monte Carlo scenario returns (T rows, N columns).
    beta : float, default 0.95
        CVaR confidence level.
    weight_bounds : tuple, default (0.0, 1.0)
        Min/max bounds for all asset weights.
    mean_return_target : float, optional
        Minimum portfolio mean return constraint.
    """

    def __init__(
        self,
        returns: pd.DataFrame,
        beta: float = 0.95,
        weight_bounds: Tuple[float, float] = (0.0, 1.0),
        mean_return_target: Optional[float] = None,
    ) -> None:
        super().__init__(assets=returns.columns)
        self.returns_df = returns
        self.returns = returns.values
        if not 0 < beta < 1:
            raise ValueError("beta must be between 0 and 1.")
        self.beta = beta
        self.weight_bounds = weight_bounds
        self.mean_return_target = mean_return_target
        
        self.T, self.N = self.returns.shape
        self.mu = self.returns.mean(axis=0)

        self.var_: float = 0.0
        self.cvar_: float = 0.0
        self.scenario_losses_: Optional[np.ndarray] = None
        self.tail_scenarios_: Optional[np.ndarray] = None

    def optimize(self, **kwargs) -> pd.Series:
        """
        Solve the CVaR minimization problem.
        The decision variable x is length N+1: [w_1, ..., w_N, alpha]
        where w are portfolio weights and alpha is the Value at Risk.
        """
        def objective(x: np.ndarray) -> float:
            w = x[:-1]
            alpha = x[-1]
            
            # losses = -R * w
            losses = - (self.returns @ w)
            exceedances = np.maximum(losses - alpha, 0.0)
            
            return float(alpha + np.mean(exceedances) / (1.0 - self.beta))

        def gradient(x: np.ndarray) -> np.ndarray:
            w = x[:-1]
            alpha = x[-1]
            
            losses = - (self.returns @ w)
            # Boolean mask of tail scenarios
            tail_mask = (losses > alpha)
            
            # gradient w.r.t w: -(1 / ((1-beta)*T)) * R^T * 1_{loss > alpha}
            grad_w = - (self.returns[tail_mask].sum(axis=0)) / (self.T * (1.0 - self.beta))
            
            # gradient w.r.t alpha: 1 - (count_tail) / ((1-beta)*T)
            grad_alpha = 1.0 - np.sum(tail_mask) / (self.T * (1.0 - self.beta))
            
            return np.concatenate([grad_w, [grad_alpha]])
            
        # Basic sum to 1
        def sum_to_one(x: np.ndarray) -> float:
            return float(np.sum(x[:-1]) - 1.0)
            
        def sum_to_one_jac(x: np.ndarray) -> np.ndarray:
            jac = np.ones_like(x)
            jac[-1] = 0.0
            return jac
            
        constraints = [{"type": "eq", "fun": sum_to_one, "jac": sum_to_one_jac}]
        
        # Mean return target constraint
        if self.mean_return_target is not None:
            def ret_cons(x: np.ndarray) -> float:
                return float(x[:-1] @ self.mu - self.mean_return_target)
            def ret_jac(x: np.ndarray) -> np.ndarray:
                jac = np.zeros_like(x)
                jac[:-1] = self.mu
                return jac
            constraints.append({"type": "ineq", "fun": ret_cons, "jac": ret_jac})

        # Set up bounds for [w_1...w_N, alpha]
        bounds = [self.weight_bounds] * self.N + [(None, None)]
        
        # Initial guess: equal weight, alpha = historical VaR of equal weight
        w_guess = np.ones(self.N) / self.N
        port_returns_guess = self.returns @ w_guess
        alpha_guess = -np.quantile(port_returns_guess, 1.0 - self.beta)
        x0 = np.concatenate([w_guess, [alpha_guess]])
        
        res = sco.minimize(
            objective,
            x0,
            method="SLSQP",
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options={"maxiter": 3000, "ftol": 1e-10},
        )
        
        if not res.success:
            raise OptimizationError(f"CVaR optimization failed: {res.message}")
            
        w_opt = self._validate_weights(res.x[:-1])
        self.var_ = float(res.x[-1])
        
        # Compute exact final losses
        final_losses = - (self.returns @ w_opt)
        self.scenario_losses_ = final_losses
        
        tail_mask = final_losses > self.var_
        self.tail_scenarios_ = np.flatnonzero(tail_mask)
        
        excs = np.maximum(final_losses - self.var_, 0.0)
        self.cvar_ = self.var_ + np.mean(excs) / (1.0 - self.beta)
        
        self.weights_ = pd.Series(w_opt, index=self.assets)
        return self.weights_

    def cvar_decomposition(self) -> pd.Series:
        """
        CVaR contribution by asset. The asset-level expected loss conditional
        on the portfolio being in the tail.
        
        CVaR_i = w_i * E[-r_i | port_loss > VaR]
        """
        self._check_fitted()
        w = self.weights_.values # type: ignore
        
        if len(self.tail_scenarios_) == 0: # type: ignore
            return pd.Series(0.0, index=self.assets)
            
        tail_returns = self.returns[self.tail_scenarios_]
        expected_tail_loss_per_asset = -np.mean(tail_returns, axis=0)
        cvar_contrib = w * expected_tail_loss_per_asset
        
        return pd.Series(cvar_contrib, index=self.assets)

    def var_at_confidence(self, beta: Optional[float] = None) -> float:
        """
        Return the historical empirical VaR at a particular confidence level.
        Defaults to the fitted CVaR beta if None.
        """
        self._check_fitted()
        if beta is None:
            beta = self.beta
            
        return float(-np.quantile(-self.scenario_losses_, 1.0 - beta)) # type: ignore
