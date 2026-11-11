import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Tuple


class BaseOptimizer(ABC):
    """
    Abstract base class for all portfolio optimizers.
    
    Parameters
    ----------
    assets : pd.Index
        Asset names/identifiers corresponding to the covariance matrix.
    """

    def __init__(self, assets: pd.Index) -> None:
        self.assets = assets
        self.weights_: Optional[pd.Series] = None

    @abstractmethod
    def optimize(self, **kwargs) -> pd.Series:
        """
        Return optimal weights as a pd.Series indexed by asset name.

        Returns
        -------
        pd.Series
        """
        pass

    def portfolio_performance(
        self,
        mu: pd.Series,
        Sigma: pd.DataFrame,
        risk_free_rate: float = 0.0,
    ) -> Tuple[float, float, float]:
        """
        Calculate expected return, volatility, and Sharpe ratio of the fitted portfolio.

        Parameters
        ----------
        mu : pd.Series
            Expected annualized returns.
        Sigma : pd.DataFrame
            Expected annualized covariance matrix.
        risk_free_rate : float, default 0.0
            Annualized risk-free rate.

        Returns
        -------
        Tuple[float, float, float]
            (annualized_return, annualized_volatility, sharpe_ratio)
        """
        self._check_fitted()
        
        from quantopt.risk.metrics import portfolio_volatility
        w = self.weights_.reindex(self.assets).fillna(0.0) # type: ignore
        mu_aligned = mu.reindex(self.assets).fillna(0.0)
        
        ret = float(w @ mu_aligned)
        vol = portfolio_volatility(w, Sigma)
        
        if vol == 0:
            sr = 0.0
        else:
            sr = (ret - risk_free_rate) / vol
            
        return ret, vol, sr

    def clean_weights(
        self,
        threshold: float = 1e-4,
        rounding: Optional[int] = None,
    ) -> pd.Series:
        """
        Zero out weights below `threshold`, then renormalize.
        If `rounding` is not None, round each weight to that many decimal places
        and renormalize again.

        Parameters
        ----------
        threshold : float, default 1e-4
            Absolute cutoff below which weights are set to 0.
        rounding : int, optional
            Number of decimal places to round to.

        Returns
        -------
        pd.Series
            Cleaned and normalized weights.
        """
        self._check_fitted()
        w = self.weights_.copy() # type: ignore
        
        # Zero threshold
        w[np.abs(w) < threshold] = 0.0
        
        # Renormalize
        if w.sum() == 0:
            raise ValueError("All weights were cleaned to zero. Threshold may be too high.")
        w /= w.sum()
        
        # Rounding
        if rounding is not None:
            w = np.round(w, rounding)
            if w.sum() == 0:
                raise ValueError("All weights were rounded to zero.")
            w /= w.sum()
            
        return w

    def _validate_weights(self, w: np.ndarray) -> np.ndarray:
        """
        Post-processing after scipy.optimize.minimize.
        1. Clip near-zero values
        2. Renormalize to sum to 1 if applicable
        """
        # 1. Clip numerical noise close to 0
        w = np.where(np.abs(w) < 1e-9, 0.0, w)
        
        # 2. Renormalize only if it is a fully invested portfolio (sum ~ 1)
        w_sum = np.sum(w)
        if abs(w_sum - 1.0) < 1e-3:
            return w / w_sum
            
        return w

    def _check_fitted(self) -> None:
        """Raise RuntimeError if self.weights_ is None."""
        if self.weights_ is None:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted yet. "
                "Call 'optimize()' before using this method."
            )

    def __repr__(self) -> str:
        status = "fitted" if self.weights_ is not None else "unfitted"
        return f"{self.__class__.__name__}(assets={len(self.assets)}, status={status})"
