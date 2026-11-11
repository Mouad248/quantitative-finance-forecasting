import logging
import numpy as np
import pandas as pd
from typing import Literal, Optional

from quantopt.utils.validation import validate_prices, validate_returns

logger = logging.getLogger(__name__)


def prices_to_returns(
    prices: pd.DataFrame,
    method: Literal["log", "simple"] = "log",
    fill_method: Literal["ffill", "drop", "none"] = "ffill",
    min_obs_pct: float = 0.95,
) -> pd.DataFrame:
    """
    Convert a price matrix to returns.

    Parameters
    ----------
    prices : pd.DataFrame
        Historical price levels of assets. Must have a DatetimeIndex.
    method : {"log", "simple"}, default "log"
        "log" computes np.log(prices).diff()
        "simple" computes prices.pct_change()
    fill_method : {"ffill", "drop", "none"}, default "ffill"
        How to handle missing prices before computing returns.
        "ffill" forward-fills prices.
        "drop" drops any row with missing prices.
        "none" leaves NaNs as-is.
    min_obs_pct : float, default 0.95
        Drop any column (asset) where the proportion of non-NaN returns
        is less than this threshold.

    Returns
    -------
    pd.DataFrame
        Asset returns (log or simple), starting from the second date.

    Raises
    ------
    ValueError
        If prices are invalid, contain non-positive values, or are improperly formatted.

    Examples
    --------
    >>> prices = pd.DataFrame({"A": [100, 105, 102], "B": [50, 52, 51]})
    >>> returns = prices_to_returns(prices, method="simple")
    """
    validate_prices(prices)

    if fill_method == "ffill":
        prices = prices.ffill()
    elif fill_method == "drop":
        prices = prices.dropna()
    elif fill_method == "none":
        pass
    else:
        raise ValueError(f"Unknown fill_method: {fill_method}")

    if method == "log":
        returns = np.log(prices).diff().iloc[1:]
    elif method == "simple":
        returns = prices.pct_change().iloc[1:]
    else:
        raise ValueError(f"Unknown method '{method}'. Must be 'log' or 'simple'.")

    # Drop columns with insufficient data depth
    n_rows = len(returns)
    valid_counts = returns.count()
    valid_pcts = valid_counts / n_rows
    to_drop = valid_pcts[valid_pcts < min_obs_pct].index

    if len(to_drop) > 0:
        logger.warning(
            f"Dropping {len(to_drop)} assets due to < {min_obs_pct:.1%} valid observations."
        )
        returns = returns.drop(columns=to_drop)

    return returns


def returns_to_prices(
    returns: pd.DataFrame,
    initial_price: float = 100.0,
    method: Literal["log", "simple"] = "log",
) -> pd.DataFrame:
    """
    Reconstruct a price index from a return series.

    Parameters
    ----------
    returns : pd.DataFrame
        Asset or portfolio returns.
    initial_price : float, default 100.0
        Starting price level at the beginning of the series.
    method : {"log", "simple"}, default "log"
        Must match the method used to compute the returns.

    Returns
    -------
    pd.DataFrame
        Reconstructed price paths, starting at initial_price. Note that the index
        will cover the same dates as the returns; initial_price represents the
        price at the end of the first period, scaled from a pre-series state.
        (Usually prepending a date is unwieldy).

    Raises
    ------
    ValueError
        If method is unrecognized.
    """
    validate_returns(returns)

    # Creating a starting point: we prepend a row for the starting date.
    # We infer the gap before the first index to place the T_0 date.
    if len(returns) > 1:
        gap = returns.index[1] - returns.index[0]
        t0 = returns.index[0] - gap
    else:
        t0 = returns.index[0] - pd.Timedelta(days=1)

    returns_with_t0 = pd.concat(
        [
            pd.DataFrame(0.0, index=[t0], columns=returns.columns),
            returns,
        ]
    )

    if method == "log":
        # log returns: prices = P_0 * exp(cumsum(r))
        cumulative = returns_with_t0.cumsum()
        prices = np.exp(cumulative) * initial_price
    elif method == "simple":
        # simple returns: prices = P_0 * cumprod(1+r)
        cumulative = (1.0 + returns_with_t0).cumprod()
        prices = cumulative * initial_price
    else:
        raise ValueError(f"Unknown method '{method}'. Must be 'log' or 'simple'.")

    return prices


def annualization_factor(
    returns: pd.DataFrame, override: Optional[int] = None
) -> int:
    """
    Infer the annualization factor from the return series frequency.

    Parameters
    ----------
    returns : pd.DataFrame
        Return series with a DatetimeIndex.
    override : int, optional
        If provided, return this factor directly, ignoring index inference.

    Returns
    -------
    int
        The estimated annualization factor (e.g., 252 for daily, 52 for weekly).

    Raises
    ------
    ValueError
        If returns does not possess a pd.DatetimeIndex and override is None.
    """
    if override is not None:
        return int(override)

    if not isinstance(returns.index, pd.DatetimeIndex):
        raise ValueError("Annualization factor inference requires a DatetimeIndex.")

    if len(returns) < 2:
        return 252  # Fallback guess for tiny sequences if override not set

    # Median difference between timestamps in days
    deltas = returns.index[1:] - returns.index[:-1]
    median_gap = deltas.to_series().median().days

    if median_gap <= 1:
        return 252  # Daily
    elif median_gap <= 7:
        return 52   # Weekly
    elif median_gap <= 31:
        return 12   # Monthly
    else:
        return 1    # Annual or lower frequency


def demean_cross_sectional(returns: pd.DataFrame) -> pd.DataFrame:
    """
    Subtract the cross-sectional mean across assets at each timestamp.
    Useful for creating market-neutral (dollar-neutral) return panels.

    Parameters
    ----------
    returns : pd.DataFrame
        Return series to demean.

    Returns
    -------
    pd.DataFrame
        Returns matrix where the mean across columns at each row is zero.
    """
    validate_returns(returns)
    cross_mean = returns.mean(axis=1)
    return returns.subtract(cross_mean, axis=0)


def winsorize_returns(
    returns: pd.DataFrame,
    lower_pct: float = 0.01,
    upper_pct: float = 0.99,
) -> pd.DataFrame:
    """
    Clip extreme outliers in the return distributions.

    Parameters
    ----------
    returns : pd.DataFrame
        Returns to clean.
    lower_pct : float, default 0.01
        The quantile below which returns are capped at the lower bound.
    upper_pct : float, default 0.99
        The quantile above which returns are capped at the upper bound.

    Returns
    -------
    pd.DataFrame
        New DataFrame with clipped returns.
    """
    validate_returns(returns)

    # Compute bounds column-wise
    lower_bounds = returns.quantile(lower_pct)
    upper_bounds = returns.quantile(upper_pct)

    # Count how many would be clipped to log it
    clipped_mask = (returns < lower_bounds) | (returns > upper_bounds)
    clipped_counts = clipped_mask.sum()
    total_clipped = clipped_counts.sum()

    if total_clipped > 0:
        logger.warning(
            f"Winsorizing {total_clipped} observations across {len(returns.columns)} assets."
        )

    # Clip column-wise. DataFrame.clip natively accepts Series bounds sharing the column index.
    return returns.clip(lower=lower_bounds, upper=upper_bounds, axis=1)
