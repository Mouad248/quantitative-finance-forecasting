import warnings
import numpy as np
import pandas as pd
import scipy.optimize as sco
from typing import Optional, Dict, Any

from quantopt.optimization.base import BaseOptimizer
from quantopt.optimization.constraints import ConstraintSet


class QuantOptError(Exception):
    """Base exception for QuantOpt errors."""
    pass


class OptimizationError(QuantOptError):
    """Raised when an optimization solver fails to converge."""
    pass


class InfeasibleError(QuantOptError):
    """Raised when a requested optimization target is mathematically infeasible."""
    pass


class EfficientFrontier(BaseOptimizer):
    """
    Mean-Variance portfolio optimization.

    Parameters
    ----------
    mu : pd.Series
        Annualized expected returns.
    Sigma : pd.DataFrame
        Annualized covariance matrix.
    constraint_set : ConstraintSet, optional
        Custom bounds and constraints. If None, defaults to long-only + sum-to-one.
    l2_gamma : float, default 0.0
        L2 regularization parameter to penalize extreme weight concentration.
    solver_opts : dict, optional
        Options passed directly to scipy.optimize.minimize (e.g. {'maxiter': 3000, 'ftol': 1e-10}).
    """

    def __init__(
        self,
        mu: pd.Series,
        Sigma: pd.DataFrame,
        constraint_set: Optional[ConstraintSet] = None,
        l2_gamma: float = 0.0,
        solver_opts: Optional[Dict[str, Any]] = None,
    ) -> None:
        if not mu.index.equals(Sigma.index) or not mu.index.equals(Sigma.columns):
            raise ValueError("mu and Sigma must have identical indices.")
            
        super().__init__(assets=mu.index)
        self.mu = mu.values
        self.Sigma = Sigma.values
        self.l2_gamma = l2_gamma
        self.n_assets = len(mu)
        
        if constraint_set is None:
            cs = ConstraintSet(self.n_assets).long_only().sum_to_one()
            self.bounds = cs.bounds()
            self.constraints = cs.constraints()
            self.cs = cs
        else:
            self.bounds = constraint_set.bounds()
            self.constraints = constraint_set.constraints()
            self.cs = constraint_set
            
        self.solver_opts = solver_opts or {"maxiter": 3000, "ftol": 1e-10}

    def _solve(self, objective, gradient, initial_guess, bounds, constraints) -> np.ndarray:
        """Internal SLSQP solver wrapper with error handling."""
        res = sco.minimize(
            objective,
            initial_guess,
            method="SLSQP",
            jac=gradient,
            bounds=bounds,
            constraints=constraints,
            options=self.solver_opts,
        )
        if not res.success:
            raise OptimizationError(f"Optimization failed: {res.message}")
            
        return self._validate_weights(res.x)

    def optimize(self, **kwargs) -> pd.Series:
        """Alias for max_sharpe."""
        return self.max_sharpe(**kwargs)

    def max_sharpe(self, risk_free_rate: float = 0.0) -> pd.Series:
        """
        Maximize the Sharpe ratio: (w^T mu - rf) / sqrt(w^T Sigma w)
        """
        def objective(w: np.ndarray) -> float:
            ret = w @ self.mu - risk_free_rate
            vol = np.sqrt(w @ self.Sigma @ w)
            if vol == 0:
                return float('inf')
            # Negate because we minimize
            return float(-(ret / vol) + self.l2_gamma * np.sum(w**2))

        def gradient(w: np.ndarray) -> np.ndarray:
            ret = w @ self.mu - risk_free_rate
            S_w = self.Sigma @ w
            vol2 = w @ S_w
            vol = np.sqrt(vol2)
            
            if vol == 0:
                # Discontinuous at origin, return zeros
                return np.zeros_like(w)
                
            # Analytical gradient of Sharpe Ratio w.r.t w
            # d(SR)/dw = (mu * vol - ret * (Sigma * w) / vol) / vol^2
            # We minimize -SR, so we return negative of that
            grad_sr = (self.mu * vol - ret * (S_w / vol)) / vol2
            return -grad_sr + 2.0 * self.l2_gamma * w

        # 5 Random restarts
        best_w = None
        best_obj = float('inf')
        last_error = None
        
        for _ in range(5):
            guess = np.random.dirichlet(np.ones(self.n_assets))
            try:
                w_opt = self._solve(objective, gradient, guess, self.bounds, self.constraints)
                obj_val = objective(w_opt)
                if obj_val < best_obj:
                    best_obj = obj_val
                    best_w = w_opt
            except OptimizationError as e:
                last_error = e

        if best_w is None:
            raise OptimizationError(
                f"Max Sharpe failed in all 5 random restarts. Last error: {last_error}"
            )
            
        self.weights_ = pd.Series(best_w, index=self.assets)
        return self.weights_

    def min_volatility(self) -> pd.Series:
        """Minimize portfolio variance w^T Sigma w."""
        def objective(w: np.ndarray) -> float:
            return float(w @ self.Sigma @ w + self.l2_gamma * np.sum(w**2))

        def gradient(w: np.ndarray) -> np.ndarray:
            return 2.0 * self.Sigma @ w + 2.0 * self.l2_gamma * w

        guess = np.ones(self.n_assets) / self.n_assets
        w_opt = self._solve(objective, gradient, guess, self.bounds, self.constraints)
        
        self.weights_ = pd.Series(w_opt, index=self.assets)
        return self.weights_

    def efficient_return(self, target_return: float) -> pd.Series:
        """Minimize volatility subject to a target return."""
        if target_return > np.max(self.mu):
            raise InfeasibleError(f"Target return {target_return} is above max asset return.")
        if target_return < np.min(self.mu):
            raise InfeasibleError(f"Target return {target_return} is below min asset return.")

        def objective(w: np.ndarray) -> float:
            return float(w @ self.Sigma @ w + self.l2_gamma * np.sum(w**2))

        def gradient(w: np.ndarray) -> np.ndarray:
            return 2.0 * self.Sigma @ w + 2.0 * self.l2_gamma * w

        def ret_constraint(w: np.ndarray) -> float:
            return float(w @ self.mu - target_return)
            
        def ret_jac(w: np.ndarray) -> np.ndarray:
            return self.mu

        constraints = self.constraints.copy()
        constraints.append({"type": "ineq", "fun": ret_constraint, "jac": ret_jac})

        guess = np.ones(self.n_assets) / self.n_assets
        w_opt = self._solve(objective, gradient, guess, self.bounds, constraints)
        
        self.weights_ = pd.Series(w_opt, index=self.assets)
        return self.weights_

    def efficient_risk(self, target_volatility: float) -> pd.Series:
        """Maximize return subject to a target volatility."""
        
        # First check the absolute minimum volatility possible
        min_vol_weights = self.min_volatility()
        min_vol = np.sqrt(min_vol_weights.values @ self.Sigma @ min_vol_weights.values)
        
        # Reset internal fitted state because we just ran an optimization
        self.weights_ = None 
        
        if target_volatility < min_vol - 1e-4:
            raise InfeasibleError(
                f"Target volatility {target_volatility:.4f} is below the minimum "
                f"achievable volatility {min_vol:.4f}."
            )

        def objective(w: np.ndarray) -> float:
            # Minimize negative return
            return float(-(w @ self.mu) + self.l2_gamma * np.sum(w**2))

        def gradient(w: np.ndarray) -> np.ndarray:
            return -self.mu + 2.0 * self.l2_gamma * w

        def risk_constraint(w: np.ndarray) -> float:
            # target_volatility^2 - variance >= 0
            return float(target_volatility**2 - w @ self.Sigma @ w)
            
        def risk_jac(w: np.ndarray) -> np.ndarray:
            return -2.0 * self.Sigma @ w

        constraints = self.constraints.copy()
        constraints.append({"type": "ineq", "fun": risk_constraint, "jac": risk_jac})

        guess = np.ones(self.n_assets) / self.n_assets
        w_opt = self._solve(objective, gradient, guess, self.bounds, constraints)
        
        self.weights_ = pd.Series(w_opt, index=self.assets)
        return self.weights_

    def efficient_frontier_points(
        self,
        n_points: int = 50,
        risk_free_rate: float = 0.0,
    ) -> pd.DataFrame:
        """
        Trace the efficient frontier across return targets.
        """
        # Determine bounds
        min_ret_vol_weights = self.min_volatility()
        min_ret = float(min_ret_vol_weights.values @ self.mu)
        max_ret = float(np.max(self.mu))

        # Sweep returns
        targets = np.linspace(min_ret * 1.01, max_ret * 0.99, n_points)
        
        results = []
        for target in targets:
            try:
                w = self.efficient_return(target)
                r = float(w.values @ self.mu)
                v = np.sqrt(w.values @ self.Sigma @ w.values)
                sr = (r - risk_free_rate) / v
                
                row = {
                    "return": r,
                    "volatility": v,
                    "sharpe": sr,
                }
                # Add weights
                for asset, weight in w.items():
                    row[asset] = weight # type: ignore
                    
                results.append(row)
            except (OptimizationError, InfeasibleError):
                continue
                
        # Revert internal state
        self.weights_ = None
        
        return pd.DataFrame(results)
