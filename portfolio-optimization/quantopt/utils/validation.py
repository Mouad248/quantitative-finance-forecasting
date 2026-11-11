import warnings
import numpy as np
import pandas as pd


def validate_returns(returns: pd.DataFrame) -> None:
    """
    Ensure the provided returns matrix is valid.

    Parameters
    ----------
    returns : pd.DataFrame
        Matrix of asset returns.

    Raises
    ------
    ValueError
        If returns is not a DataFrame, or lacks a DatetimeIndex,
        or contains entirely NaN columns.
    """
    if not isinstance(returns, pd.DataFrame):
        raise ValueError("Returns must be a pandas DataFrame.")

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Returns must have a DatetimeIndex.")

    if returns.empty:
        raise ValueError("Returns DataFrame is empty.")

    nan_counts = returns.isna().sum()
    empty_cols = nan_counts[nan_counts == len(returns)].index.tolist()
    if empty_cols:
        raise ValueError(f"The following columns are entirely NaN: {empty_cols}")

    nan_pcts = returns.isna().mean()
    high_nan_cols = nan_pcts[nan_pcts > 0.05].index.tolist()
    if high_nan_cols:
        warnings.warn(
            f"The following columns have >5% NaN values after cleaning: {high_nan_cols}"
        )


def validate_prices(prices: pd.DataFrame) -> None:
    """
    Ensure the provided prices matrix is valid.

    Parameters
    ----------
    prices : pd.DataFrame
        Matrix of asset prices.

    Raises
    ------
    ValueError
        If prices is not a DataFrame, lacks a DatetimeIndex,
        has <2 rows, or contains non-positive prices.
    """
    if not isinstance(prices, pd.DataFrame):
        raise ValueError("Prices must be a pandas DataFrame.")

    if not isinstance(prices.index, pd.DatetimeIndex):
        raise ValueError("Prices must have a DatetimeIndex.")

    if len(prices) < 2:
        raise ValueError("Prices DataFrame must have at least 2 rows to compute returns.")

    if (prices <= 0).any().any():
        raise ValueError("Prices must be strictly positive (<=0 corrupts log returns).")


def validate_weights(weights: pd.Series, tol: float = 1e-4) -> None:
    """
    Ensure the provided weights sum to 1.

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights.
    tol : float, default 1e-4
        Tolerance for sum-to-one constraint.

    Raises
    ------
    ValueError
        If weights is not a Series, or if the sum deviates from 1 by > tol.
    """
    if not isinstance(weights, pd.Series):
        raise ValueError("Weights must be a pandas Series.")

    weight_sum = weights.sum()
    if np.abs(weight_sum - 1.0) > tol:
        raise ValueError(
            f"Weights must sum to 1. Current sum is {weight_sum:.6f} "
            f"(deviation {np.abs(weight_sum - 1.0):.6f} > tolerance {tol})."
        )

    high_concentration = weights[weights > 0.50].index.tolist()
    if high_concentration:
        warnings.warn(
            f"High concentration warning: weights > 50% for assets: {high_concentration}"
        )


def check_psd(matrix: np.ndarray, name: str = "matrix") -> bool:
    """
    Check if a matrix is positive semi-definite (PSD).

    Parameters
    ----------
    matrix : np.ndarray
        Square matrix to check.
    name : str, default "matrix"
        Name of the matrix for logging.

    Returns
    -------
    bool
        True if PSD, False otherwise.
    """
    try:
        eigenvals = np.linalg.eigvalsh(matrix)
        return bool(np.all(eigenvals >= -1e-8))
    except np.linalg.LinAlgError:
        return False


def project_psd(matrix: np.ndarray, floor: float = 1e-8) -> np.ndarray:
    """
    Project a square symmetric matrix onto the cone of positive semi-definite matrices.
    Uses eigendecomposition to clip negative eigenvalues, then reconstitutes.

    Parameters
    ----------
    matrix : np.ndarray
        Square input matrix.
    floor : float, default 1e-8
        Minimum allowed eigenvalue.

    Returns
    -------
    np.ndarray
        Valid PSD symmetric matrix.
    """
    # Ensure symmetry
    sym_matrix = (matrix + matrix.T) / 2.0

    # Eigendecomposition
    eigenvals, eigenvects = np.linalg.eigh(sym_matrix)

    # Clip eigenvalues to floor
    clipped_vals = np.maximum(eigenvals, floor)

    # Reconstruct
    psd_matrix = eigenvects @ np.diag(clipped_vals) @ eigenvects.T

    # Ensure symmetric again due to numerical noise
    return (psd_matrix + psd_matrix.T) / 2.0
