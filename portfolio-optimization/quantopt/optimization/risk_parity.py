import numpy as np
import pandas as pd
import scipy.optimize as sco
from typing import Optional

from quantopt.optimization.base import BaseOptimizer
from quantopt.optimization.efficient_frontier import OptimizationError


class RiskParity(BaseOptimizer):
    """
    Equal Risk Contribution (ERC) and generalized Risk Parity optimization.

    Parameters
    ----------
    Sigma : pd.DataFrame
        Annualized covariance matrix.
    risk_budgets : pd.Series, optional
        Target percentage risk contribution for each asset.
        If None, equal risk budgets (1/N) are used.
    long_only : bool, default True
        If True, restricts weights to be non-negative.
    """

    def __init__(
        self,
        Sigma: pd.DataFrame,
        risk_budgets: Optional[pd.Series] = None,
        long_only: bool = True,
    ) -> None:
        super().__init__(assets=Sigma.index)
        self.Sigma = Sigma.values
        self.n_assets = len(Sigma)
        self.long_only = long_only
        
        if risk_budgets is None:
            self.b_target = np.ones(self.n_assets) / self.n_assets
        else:
            budgets = risk_budgets.reindex(Sigma.index).fillna(0.0).values
            if np.any(budgets < 0):
                raise ValueError("Risk budgets cannot be negative.")
            b_sum = budgets.sum()
            if b_sum == 0:
                raise ValueError("Risk budgets sum to zero.")
            self.b_target = budgets / b_sum

    def optimize(self, **kwargs) -> pd.Series:
        """
        Solve for the weights that minimize the squared deviations of
        risk contributions from their target budgets.
        """
        def objective(w: np.ndarray) -> float:
            vol = np.sqrt(w @ self.Sigma @ w)
            if vol == 0:
                return float('inf')
            
            mrc = (self.Sigma @ w) / vol
            rc = w * mrc
            
            # Minimize sum_i (RC_i - b_i * vol)^2
            residuals = rc - self.b_target * vol
            return float(np.sum(residuals**2))

        def gradient(w: np.ndarray) -> np.ndarray:
            vol = np.sqrt(w @ self.Sigma @ w)
            if vol == 0:
                return np.zeros_like(w)
                
            S_w = self.Sigma @ w
            mrc = S_w / vol
            rc = w * mrc
            
            residuals = rc - self.b_target * vol
            
            # J[i,j] = d(w_i * (Sigma @ w)_i / vol) / dw_j.
            J = np.diag(mrc) + np.diag(w) @ self.Sigma / vol - np.outer(rc, S_w) / (vol**2)
            
            # Derivative of objective: 2 * sum_i (residual_i * d_residual_i/dw)
            # d_residual_i / dw = dRC_i/dw - b_i * mrc
            d_res = J - np.outer(self.b_target, mrc)
            
            grad = 2.0 * residuals @ d_res
            return grad

        # Constraints
        constraints = [{"type": "eq", "fun": lambda x: float(np.sum(x) - 1.0)}]
        
        # Bounds
        if self.long_only:
            bounds = [(1e-6, 1.0) for _ in range(self.n_assets)]
        else:
            bounds = [(-1.0, 1.0) for _ in range(self.n_assets)]

        best_w = None
        best_obj = float('inf')
        last_error = None
        
        # 15 random restarts with Dirichlet distribution
        for _ in range(15):
            guess = np.random.dirichlet(np.ones(self.n_assets))
            try:
                res = sco.minimize(
                    objective,
                    guess,
                    method="SLSQP",
                    jac=gradient,
                    bounds=bounds,
                    constraints=constraints,
                    options={"maxiter": 3000, "ftol": 1e-12},
                )
                if res.success and res.fun < best_obj:
                    best_obj = res.fun
                    best_w = res.x
            except Exception as e:
                last_error = e

        if best_w is None:
            raise OptimizationError(f"Risk Parity convergence failed. Last error: {last_error}")
            
        w_opt = self._validate_weights(best_w)
        self.weights_ = pd.Series(w_opt, index=self.assets)
        return self.weights_

    def risk_contributions(self) -> pd.Series:
        """Return the percentage risk contribution of each asset."""
        self._check_fitted()
        w = self.weights_.values # type: ignore
        mrc = (self.Sigma @ w) / np.sqrt(w @ self.Sigma @ w)
        rc = w * mrc
        prc = rc / rc.sum()
        return pd.Series(prc, index=self.assets)

    def concentration_check(self, tol: float = 0.01) -> bool:
        """
        Check if the final risk contributions match the target budgets.

        Parameters
        ----------
        tol : float, default 0.01
            Tolerance for deviation from target.

        Returns
        -------
        bool
        """
        self._check_fitted()
        actual_prc = self.risk_contributions().values
        max_dev = np.max(np.abs(actual_prc - self.b_target))
        return bool(max_dev <= tol)
