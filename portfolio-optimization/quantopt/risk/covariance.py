import warnings
import numpy as np
import pandas as pd
from abc import ABC, abstractmethod
from typing import Optional, Dict

from quantopt.utils.validation import validate_returns, project_psd


class BaseCovarianceEstimator(ABC):
    """
    Abstract base class for covariance estimators.
    """

    def __init__(self) -> None:
        self.assets: Optional[pd.Index] = None
        self._S: Optional[pd.DataFrame] = None

    @abstractmethod
    def fit(self, returns: pd.DataFrame) -> "BaseCovarianceEstimator":
        """
        Fit the covariance estimator.

        Parameters
        ----------
        returns : pd.DataFrame
            Matrix of historical returns.

        Returns
        -------
        self
        """
        pass

    def covariance(self) -> pd.DataFrame:
        """
        Return the annualized covariance matrix.

        Returns
        -------
        pd.DataFrame
            N x N covariance matrix.
        """
        self._check_fitted()
        return self._S  # type: ignore

    def correlation(self) -> pd.DataFrame:
        """
        Derive the correlation matrix from the estimated covariance.

        Returns
        -------
        pd.DataFrame
            N x N correlation matrix.
        """
        cov = self.covariance().values
        vols = np.sqrt(np.diag(cov))
        
        # Handle zero variance seamlessly
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_vols = 1.0 / vols
            inv_vols[np.isinf(inv_vols)] = 0.0
            
        corr = cov * np.outer(inv_vols, inv_vols)
        # Ensure diagonals are exactly 1
        np.fill_diagonal(corr, 1.0)
        
        # Fix NaNs resulting from 0/0
        corr[np.isnan(corr)] = 0.0
        
        return pd.DataFrame(corr, index=self.assets, columns=self.assets)

    def std(self) -> pd.Series:
        """
        Return the annualized volatility per asset (sqrt of diagonal).

        Returns
        -------
        pd.Series
            Volatilities indexed by asset.
        """
        cov = self.covariance()
        return pd.Series(np.sqrt(np.diag(cov)), index=self.assets)

    def _enforce_psd(self, S: np.ndarray) -> np.ndarray:
        """
        Project matrix to be Positive Semi-Definite.
        """
        return project_psd(S)

    def _check_fitted(self) -> None:
        if self._S is None:
            raise RuntimeError(
                f"{self.__class__.__name__} is not fitted yet. "
                "Call 'fit' before using this method."
            )

    def __repr__(self) -> str:
        status = "fitted" if self._S is not None else "unfitted"
        n_assets = len(self.assets) if self.assets is not None else "unknown"
        return f"{self.__class__.__name__}(assets={n_assets}, status={status})"


class SampleCovariance(BaseCovarianceEstimator):
    """
    Sample covariance estimator.

    Parameters
    ----------
    frequency : int, default 252
        Annualization factor.
    fix_psd : bool, default True
        If True, ensures the resulting covariance matrix is PSD via projection.
    """

    def __init__(self, frequency: int = 252, fix_psd: bool = True) -> None:
        super().__init__()
        self.frequency = frequency
        self.fix_psd = fix_psd
        self.n_obs_: int = 0
        self.condition_number_: float = 0.0

    def fit(self, returns: pd.DataFrame) -> "SampleCovariance":
        validate_returns(returns)
        self.assets = returns.columns
        self.n_obs_ = len(returns)

        cov_matrix = np.cov(returns.values, rowvar=False) * self.frequency

        if self.fix_psd:
            cov_matrix = self._enforce_psd(cov_matrix)
            
        try:
            eigenvals = np.linalg.eigvalsh(cov_matrix)
            eigenvals = np.abs(eigenvals)  # handle numerical negatives
            min_eig = eigenvals.min()
            max_eig = eigenvals.max()
            
            if min_eig == 0:
                self.condition_number_ = np.inf
            else:
                self.condition_number_ = float(max_eig / min_eig)
                
            if self.condition_number_ > 1000:
                warnings.warn(
                    f"Ill-conditioned sample covariance matrix "
                    f"(condition number: {self.condition_number_:.1f}). "
                    "Consider using Ledoit-Wolf or FactorModel covariance instead."
                )
        except np.linalg.LinAlgError:
            self.condition_number_ = np.inf

        self._S = pd.DataFrame(cov_matrix, index=self.assets, columns=self.assets)
        return self


class EWMCovariance(BaseCovarianceEstimator):
    """
    Exponentially Weighted Moving (EWM) Covariance.
    Gives more weight to recent observations, capturing volatility clustering.

    Parameters
    ----------
    span : int, default 60
        Decay span. Weight assigned to observation at time t is `(1 - 2/(span+1))^(T-t)`.
    frequency : int, default 252
        Annualization factor.
    """

    def __init__(self, span: int = 60, frequency: int = 252) -> None:
        super().__init__()
        self.span = span
        self.frequency = frequency
        self.effective_observations_: float = 0.0

    def fit(self, returns: pd.DataFrame) -> "EWMCovariance":
        validate_returns(returns)
        self.assets = returns.columns
        
        T, N = returns.shape
        decay = 1.0 - 2.0 / (self.span + 1.0)
        
        # Build weight vector weights[t] = decay^(T-1-t)
        time_idx = np.arange(T)
        powers = (T - 1) - time_idx
        weights = np.power(decay, powers)
        
        # Normalize weights
        w_sum = weights.sum()
        weights /= w_sum
        
        self.effective_observations_ = float(1.0 / np.sum(weights**2))
        
        R = returns.values
        # Weighted mean
        mu_w = R.T @ weights
        
        # Weighted covariance
        # S = Sum(w_t * (r_t - mu_w) @ (r_t - mu_w)^T)
        # We can vectorize: S = (R - mu).T @ diag(w) @ (R - mu)
        R_centered = R - mu_w
        S = (R_centered.T * weights) @ R_centered
        
        S *= self.frequency
        S_psd = self._enforce_psd(S)
        
        self._S = pd.DataFrame(S_psd, index=self.assets, columns=self.assets)
        return self


class LedoitWolfCovariance(BaseCovarianceEstimator):
    """
    Ledoit-Wolf shrinkage estimator (using Oracle Approximating Shrinkage).
    Blends the sample covariance with a highly structured target matrix (mean variance identity)
    to minimize Mean Squared Error, yielding a well-conditioned invertible matrix.

    Parameters
    ----------
    frequency : int, default 252
        Annualization factor.
    """

    def __init__(self, frequency: int = 252) -> None:
        super().__init__()
        self.frequency = frequency
        self.shrinkage_: float = 0.0

    def fit(self, returns: pd.DataFrame) -> "LedoitWolfCovariance":
        from sklearn.covariance import OAS
        validate_returns(returns)
        self.assets = returns.columns

        # OAS computes optimal shrinkage intrinsically
        oas = OAS(store_precision=False, assume_centered=False)
        oas.fit(returns.values)
        
        # Extract covariance and rescale to annual
        cov_matrix = oas.covariance_ * self.frequency
        self.shrinkage_ = float(oas.shrinkage_)
        
        cov_psd = self._enforce_psd(cov_matrix)
        self._S = pd.DataFrame(cov_psd, index=self.assets, columns=self.assets)
        return self

    def shrinkage_target(self) -> pd.DataFrame:
        """
        Return the structured target matrix toward which the sample covariance was shrunk.
        The default target is the mean of eigenvalues (mean variance) times Identity.

        Returns
        -------
        pd.DataFrame
            N x N scalar variance target matrix.
        """
        self._check_fitted()
        cov = self._S.values  # type: ignore
        mean_var = np.trace(cov) / len(self.assets)
        target = np.eye(len(self.assets)) * mean_var
        return pd.DataFrame(target, index=self.assets, columns=self.assets)

    def blend_report(self) -> Dict[str, float]:
        """
        Return the blend weights used to form the final covariance matrix.

        Returns
        -------
        dict
            Dictionary containing shrinkage factor and component weights.
        """
        self._check_fitted()
        return {
            "shrinkage": self.shrinkage_,
            "sample_weight": 1.0 - self.shrinkage_,
            "target_weight": self.shrinkage_,
        }


class FactorModelCovariance(BaseCovarianceEstimator):
    """
    Statistical Factor Model Covariance based on Principal Component Analysis (Barra-style).
    Decomposes covariance into systematic risk (driven by K latent factors) and
    idiosyncratic specific risk.

    Parameters
    ----------
    n_factors : int, optional
        Fixed number of factors to use. If None, determined via variance_threshold.
    variance_threshold : float, default 0.80
        Target cumulative explained variance fraction to determine K (if n_factors=None).
    frequency : int, default 252
        Annualization factor.
    min_factors : int, default 1
        Minimum number of factors.
    max_factors : int, default 20
        Maximum number of factors.
    """

    def __init__(
        self,
        n_factors: Optional[int] = None,
        variance_threshold: float = 0.80,
        frequency: int = 252,
        min_factors: int = 1,
        max_factors: int = 20,
    ) -> None:
        super().__init__()
        self.n_factors = n_factors
        self.variance_threshold = variance_threshold
        self.frequency = frequency
        self.min_factors = min_factors
        self.max_factors = max_factors

        self.factor_loadings_: Optional[pd.DataFrame] = None
        self.factor_covariance_: Optional[pd.DataFrame] = None
        self.idiosyncratic_var_: Optional[pd.Series] = None
        self.n_factors_used_: int = 0
        self.explained_variance_ratio_: Optional[np.ndarray] = None

    def fit(self, returns: pd.DataFrame) -> "FactorModelCovariance":
        from sklearn.decomposition import PCA
        validate_returns(returns)
        self.assets = returns.columns
        T, N = returns.shape

        R_dm = returns.values - returns.values.mean(axis=0)

        # Determine K if not provided (run full PCA first to examine variance ratios)
        if self.n_factors is None:
            K_max = min(N, T)
            pca_full = PCA(n_components=K_max)
            pca_full.fit(R_dm)
            cumulative_var = np.cumsum(pca_full.explained_variance_ratio_)
            
            # Find index where cumulative variance >= threshold
            meets_thresh = np.where(cumulative_var >= self.variance_threshold)[0]
            if len(meets_thresh) > 0:
                K = meets_thresh[0] + 1
            else:
                K = K_max
                
            # Clamp to min/max
            K = min(max(K, self.min_factors), self.max_factors)
            K = min(K, K_max)
        else:
            K = min(self.n_factors, min(N, T))

        self.n_factors_used_ = K

        # Step 1: PCA
        pca = PCA(n_components=K)
        
        # Step 2: Extract factor scores (F) and loadings (B)
        F = pca.fit_transform(R_dm)       # T x K factor realizations
        B = pca.components_.T             # N x K factor loadings
        
        self.explained_variance_ratio_ = pca.explained_variance_ratio_

        # Step 3: Factor covariance (annualized)
        if K == 1:
            F_cov = np.array([[np.var(F[:, 0], ddof=1) * self.frequency]])
        else:
            F_cov = np.cov(F.T) * self.frequency
            
        # Step 4: Idiosyncratic variance
        R_hat = F @ B.T                 # T x N systematic component
        residuals = R_dm - R_hat
        D = np.var(residuals, axis=0, ddof=1) * self.frequency  # N-vector
        
        # Step 5: Reconstruct and export
        S = B @ F_cov @ B.T + np.diag(D)
        S_psd = self._enforce_psd(S)

        self._S = pd.DataFrame(S_psd, index=self.assets, columns=self.assets)
        
        factor_names = [f"Factor_{i+1}" for i in range(K)]
        self.factor_loadings_ = pd.DataFrame(B, index=self.assets, columns=factor_names)
        self.factor_covariance_ = pd.DataFrame(F_cov, index=factor_names, columns=factor_names)
        self.idiosyncratic_var_ = pd.Series(D, index=self.assets)

        return self

    def systematic_variance_fraction(self) -> float:
        """
        Fraction of total variance explained by the latent factors (trace ratio).
        
        Returns
        -------
        float
        """
        self._check_fitted()
        B = self.factor_loadings_.values      # type: ignore
        F_cov = self.factor_covariance_.values # type: ignore
        D = self.idiosyncratic_var_.values    # type: ignore
        
        systematic_diag = np.diag(B @ F_cov @ B.T)
        total_trace = systematic_diag.sum() + D.sum()
        
        if total_trace == 0:
            return 0.0
            
        return float(systematic_diag.sum() / total_trace)

    def factor_correlation(self) -> pd.DataFrame:
        """
        Correlation matrix of the K latent factors.

        Returns
        -------
        pd.DataFrame
            K x K matrix.
        """
        self._check_fitted()
        cov = self.factor_covariance_.values # type: ignore
        vols = np.sqrt(np.diag(cov))
        
        with np.errstate(divide='ignore', invalid='ignore'):
            inv_vols = 1.0 / vols
            inv_vols[np.isinf(inv_vols)] = 0.0
            
        corr = cov * np.outer(inv_vols, inv_vols)
        np.fill_diagonal(corr, 1.0)
        
        return pd.DataFrame(
            corr, 
            index=self.factor_covariance_.index, # type: ignore
            columns=self.factor_covariance_.columns # type: ignore
        )

    def factor_vif(self) -> pd.Series:
        """
        Variance Inflation Factor (VIF) per factor to detect multicollinearity.
        Since these are PCA factors, they should ideally be orthogonal and VIF ~ 1.

        Returns
        -------
        pd.Series
            VIF scores per factor.
        """
        self._check_fitted()
        corr = self.factor_correlation().values
        
        try:
            inv_corr = np.linalg.inv(corr)
            vifs = np.diag(inv_corr)
        except np.linalg.LinAlgError:
            vifs = np.full(corr.shape[0], np.inf)
            
        return pd.Series(vifs, index=self.factor_covariance_.index) # type: ignore
