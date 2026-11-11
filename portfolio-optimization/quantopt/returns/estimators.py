"""
Return estimators for portfolio optimization.
"""
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional


class BaseReturnsEstimator(ABC):
    """
    Abstract base class for all expected return estimators.
    """

    def __init__(self) -> None:
        self.assets: Optional[pd.Index] = None
        self._mu: Optional[pd.Series] = None

    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> "BaseReturnsEstimator":
        """
        Fit the estimator to historical log or simple returns.

        Parameters
        ----------
        returns : pd.DataFrame
            Asset returns matrix.

        Returns
        -------
        self
        """
        pass

    def expected_returns(self) -> pd.Series:
        """
        Return the estimated expected returns.

        Returns
        -------
        pd.Series
            Annualized expected returns indexed by asset.
        """
        self._check_fitted()
        return self._mu  # type: ignore

    def _check_fitted(self) -> None:
        if self._mu is None:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted yet. "
                "Call 'fit' with appropriate arguments before using this method."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._mu is not None else "unfitted"
        n_assets = len(self.assets) if self.assets is not None else "unknown"
        return f"{self.__class__.__name__}(assets={n_assets}, status={status})"


class MeanHistoricalReturn(BaseReturnsEstimator):
    """
    Historical mean returns, annualized.

    Parameters
    ----------
    frequency : int, default 252
        Annualization factor (252 for daily, 12 for monthly).
    exponential_weighting : bool, default False
        If True, weight observations exponentially such that recent
        returns are weighted more heavily.
    span : int, default 60
        Span for exponential weighting (if enabled). Weight for time t
        is proportional to (1 - 2/(span+1))^(T-1-t).
    compounding : bool, default True
        If True, use geometric compounding `(1+r)^freq - 1`.
        If False, use arithmetic annualization `r * freq`.
    """

    def __init__(
        self,
        frequency: int = 252,
        exponential_weighting: bool = False,
        span: int = 60,
        compounding: bool = True,
    ) -> None:
        super().__init__()
        self.frequency = frequency
        self.exponential_weighting = exponential_weighting
        self.span = span
        self.compounding = compounding
        self.mean_daily_returns_: Optional[pd.Series] = None

    def fit(self, returns: pd.DataFrame) -> "MeanHistoricalReturn":
        from quantopt.utils.validation import validate_returns
        validate_returns(returns)
        self.assets = returns.columns

        T = len(returns)

        if self.exponential_weighting:
            # Span -> decay lambda
            decay = 1.0 - 2.0 / (self.span + 1)
            # weights[t] = decay^(T-1-t)
            time_indices = np.arange(T)
            powers = (T - 1) - time_indices
            weights = np.power(decay, powers)
            weights /= weights.sum()

            # Weighted mean (broadcasting weights across assets)
            daily_mean = returns.values.T @ weights
            mean_series = pd.Series(daily_mean, index=self.assets)
        else:
            mean_series = returns.mean(axis=0)

        self.mean_daily_returns_ = mean_series

        if self.compounding:
            self._mu = (1.0 + mean_series) ** self.frequency - 1.0
        else:
            self._mu = mean_series * self.frequency

        return self


class CAPMReturn(BaseReturnsEstimator):
    """
    Capital Asset Pricing Model (CAPM) return estimator.
    Computes expected returns via historical Betas against a market portfolio.

    Parameters
    ----------
    market_returns : pd.Series
        Returns of the benchmark/market portfolio.
    risk_free_rate : float, default 0.0
        Annualized risk-free rate.
    frequency : int, default 252
        Annualization factor.
    """

    def __init__(
        self,
        market_returns: pd.Series,
        risk_free_rate: float = 0.0,
        frequency: int = 252,
    ) -> None:
        super().__init__()
        self.market_returns = market_returns
        self.risk_free_rate = risk_free_rate
        self.frequency = frequency
        self.betas_: Optional[pd.Series] = None
        self.market_premium_: Optional[float] = None

    def fit(self, returns: pd.DataFrame) -> "CAPMReturn":
        from quantopt.utils.validation import validate_returns
        if not isinstance(self.market_returns, pd.Series):
            raise ValueError("market_returns must be a pd.Series.")

        # Align series to common dates, drop NaNs
        aligned = pd.concat([returns, self.market_returns.rename("MKT")], axis=1).dropna()
        if len(aligned) < 2:
            raise ValueError("Not enough overlapping valid data points to compute Beta.")

        aligned_rets = aligned[returns.columns]
        aligned_mkt = aligned["MKT"]

        validate_returns(aligned_rets)
        self.assets = aligned_rets.columns

        # Daily risk-free rate
        rf_daily = self.risk_free_rate / self.frequency

        # Excess returns
        excess_assets = aligned_rets - rf_daily
        excess_mkt = aligned_mkt - rf_daily

        # Compute Betas via OLS matrix logic: Cov(R_i, R_m) / Var(R_m)
        mkt_var = np.var(excess_mkt, ddof=1)
        if mkt_var == 0:
            betas = np.zeros(len(self.assets))
        else:
            # np.cov rowvar=False -> rows are observations.
            # cov matrix size: (N+1) x (N+1) if we concat mkt.
            # Simpler: dot product of demeaned variables
            mkt_demeaned = excess_mkt - excess_mkt.mean()
            asset_demeaned = excess_assets.subtract(excess_assets.mean(axis=0))
            covariances = (asset_demeaned.T @ mkt_demeaned) / (len(aligned) - 1)
            betas = covariances / mkt_var

        self.betas_ = pd.Series(betas.values, index=self.assets)

        # Annualize market return via geometric compounding
        mkt_mean_daily = aligned_mkt.mean()
        annualized_mkt_return = (1.0 + mkt_mean_daily) ** self.frequency - 1.0
        self.market_premium_ = annualized_mkt_return - self.risk_free_rate

        # E[R_i] = R_f + Beta_i * Market_Premium
        self._mu = self.risk_free_rate + self.betas_ * self.market_premium_

        return self


class BlackLittermanReturn(BaseReturnsEstimator):
    """
    Black-Litterman Expected Return estimator.
    Blends market equilibrium implied returns with absolute/relative investor views
    updating prior via a full Bayesian posterior.

    Parameters
    ----------
    market_caps : pd.Series
        Market capitalization of each asset. Internally normalized to sum to 1.
    risk_aversion : float, default 2.5
        Market risk aversion coefficient (delta).
    tau : float, default 0.05
        Scalar indicating uncertainty of the prior. Typically 0.01 - 0.10.
    P : Optional[np.ndarray], default None
        (K x N) Pick matrix mapping views to assets.
    Q : Optional[np.ndarray], default None
        (K,) vector of views (expected returns over frequency period).
    omega : Optional[np.ndarray], default None
        (K x K) covariance matrix of views. If None, uses proportional-to-variance.
    frequency : int, default 252
        Annualization factor for equilibrium variance.
    """

    def __init__(
        self,
        market_caps: pd.Series,
        risk_aversion: float = 2.5,
        tau: float = 0.05,
        P: Optional[np.ndarray] = None,
        Q: Optional[np.ndarray] = None,
        omega: Optional[np.ndarray] = None,
        frequency: int = 252,
    ) -> None:
        super().__init__()
        self.market_caps = market_caps
        self.risk_aversion = risk_aversion
        self.tau = tau
        self.P = P
        self.Q = Q
        self.omega = omega
        self.frequency = frequency

        self.pi_: Optional[pd.Series] = None
        self.posterior_cov_: Optional[np.ndarray] = None
        self._omega_diag_: Optional[np.ndarray] = None

    def implied_confidence(self) -> Optional[np.ndarray]:
        """
        Return the per-view confidence (diagonal of Omega matrix).
        Lower values mean higher confidence in the view.
        """
        if self._omega_diag_ is None:
            return None
        return self._omega_diag_

    def fit(self, returns: pd.DataFrame) -> "BlackLittermanReturn":
        from quantopt.utils.validation import validate_returns
        validate_returns(returns)
        self.assets = returns.columns
        N = len(self.assets)

        # 1. Base Equillibrium Prior
        # Sigma is annualized covariance
        sigma = returns.cov().values * self.frequency

        w_mkt = self.market_caps.reindex(self.assets).fillna(0.0).values
        w_sum = w_mkt.sum()
        if w_sum == 0:
            raise ValueError("market_caps sum to zero for the given assets.")
        w_mkt /= w_sum

        # Implied Returns Pi = delta * Sigma * w_mkt
        pi = self.risk_aversion * sigma @ w_mkt
        self.pi_ = pd.Series(pi, index=self.assets)

        # 2. Check no-views condition
        if self.P is None or self.Q is None:
            if self.P is not None or self.Q is not None:
                raise ValueError("Both P and Q must be provided to express views.")
            # Posterior = Prior
            self._mu = self.pi_.copy()
            self.posterior_cov_ = (1 + self.tau) * sigma
            return self

        # 3. Process views
        P = np.asarray(self.P, dtype=float)
        Q = np.asarray(self.Q, dtype=float)
        
        # Squeeze 1D vectors
        if Q.ndim == 2 and Q.shape[1] == 1:
            Q = Q.ravel()

        if P.shape[1] > N:
            raise ValueError(
                f"Pick matrix P has {P.shape[1]} columns, but there are only {N} assets."
            )
        elif P.shape[1] < N:
             # Zero-pad remaining columns if necessary to match asset dimensions
             P = np.pad(P, ((0, 0), (0, N - P.shape[1])), mode="constant")
             
             
        K = P.shape[0]
        if Q.shape[0] != K:
            raise ValueError(f"Pick matrix P has {K} rows, but views vector Q has {Q.shape[0]} elements.")

        tau_sigma = self.tau * sigma

        # 4. Omega construction (Idzorek's method if None)
        if self.omega is None:
            # Omega = diag(P * (tau * Sigma) * P^T)
            # Ensures variance of views is proportional to uncertainty in prior
            omega = np.diag(np.diag(P @ tau_sigma @ P.T))
        else:
            omega = np.asarray(self.omega, dtype=float)
            if omega.shape != (K, K):
                raise ValueError(f"Omega must be shape ({K}, {K}) based on {K} views.")

        self._omega_diag_ = np.diag(omega)

        # 5. Master Formula Implementation (M^-1 and mu_BL)
        # Invert (tau * Sigma) 
        try:
            tau_sigma_inv = np.linalg.inv(tau_sigma)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            tau_sigma_inv = np.linalg.pinv(tau_sigma)
            
        try:
            omega_inv = np.linalg.inv(omega)
        except np.linalg.LinAlgError:
            omega_inv = np.linalg.pinv(omega)

        # Posterior covariance
        M_inv = tau_sigma_inv + P.T @ omega_inv @ P
        
        try:
            M = np.linalg.inv(M_inv)
        except np.linalg.LinAlgError:
            M = np.linalg.pinv(M_inv)
            
        self.posterior_cov_ = M + sigma

        # Posterior expected returns
        mu_bl = M @ (tau_sigma_inv @ pi + P.T @ omega_inv @ Q)
        self._mu = pd.Series(mu_bl, index=self.assets)

        return self
