"""
Performance analytics and full tear sheet capabilities.
"""
import numpy as np
import scipy.stats as stats
import pandas as pd
from typing import Optional, Literal


def annualized_return(
    returns: pd.Series,
    frequency: int = 252,
    method: Literal["geometric", "arithmetic"] = "geometric",
) -> float:
    """
    Method:
    - geometric: (1 + prod(1+r))^(frequency/T) - 1
    - arithmetic: r.mean() * frequency
    """
    if len(returns) == 0:
        return 0.0

    if method == "geometric":
        compounded = (1.0 + returns).prod()
        # Handle zero-return series natively
        if compounded == 1.0 and returns.sum() == 0:
            return 0.0
        return float(compounded ** (frequency / len(returns)) - 1.0)
    elif method == "arithmetic":
        return float(returns.mean() * frequency)
    else:
        raise ValueError("Method must be 'geometric' or 'arithmetic'.")


def annualized_volatility(returns: pd.Series, frequency: int = 252) -> float:
    if len(returns) < 2:
        return np.nan
    return float(returns.std(ddof=1) * np.sqrt(frequency))


def sharpe_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    frequency: int = 252,
) -> float:
    ann_ret = annualized_return(returns, frequency=frequency, method="geometric")
    ann_vol = annualized_volatility(returns, frequency=frequency)
    if not np.isfinite(ann_vol) or ann_vol == 0:
        return 0.0
    return (ann_ret - risk_free_rate) / ann_vol


def sortino_ratio(
    returns: pd.Series,
    risk_free_rate: float = 0.0,
    frequency: int = 252,
    mar: float = 0.0,
) -> float:
    """
    Downside deviation = sqrt(mean(min(r - mar, 0)^2)) * sqrt(frequency)
    """
    ann_ret = annualized_return(returns, frequency=frequency)
    
    downside = np.minimum(returns - mar, 0.0)
    downside_var = np.mean(downside**2)
    downside_dev = np.sqrt(downside_var) * np.sqrt(frequency)
    
    if downside_dev == 0:
        return np.inf
        
    return float((ann_ret - risk_free_rate) / downside_dev)


def drawdown_series(returns: pd.Series) -> pd.Series:
    """Return the time series of % drawdowns from peak."""
    cumulative = (1.0 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    return drawdown


def max_drawdown(returns: pd.Series) -> float:
    """Compute peak-to-trough max drawdown. Returns negative float."""
    if len(returns) == 0:
        return 0.0
    dd = drawdown_series(returns)
    return float(dd.min())


def max_drawdown_duration(returns: pd.Series) -> Optional[int]:
    """
    Number of periods from the start of the max drawdown trough to recovery.
    Returns None if not recovered.
    """
    if len(returns) == 0:
        return 0
        
    cumulative = (1.0 + returns).cumprod()
    peak = cumulative.cummax()
    drawdown = (cumulative - peak) / peak
    
    if drawdown.min() == 0:
        return 0  # No drawdown
        
    trough_idx = drawdown.idxmin()
    trough_val = cumulative.loc[trough_idx]
    peak_val = peak.loc[trough_idx]
    
    # Slice after trough
    post_trough = cumulative.loc[trough_idx:]
    recovered = post_trough[post_trough >= peak_val]
    
    if len(recovered) == 0:
        return None  # Has not recovered
        
    recovery_idx = recovered.index[0]
    
    # Calculate duration (number of periods)
    loc_start = cumulative.index.get_loc(trough_idx)
    loc_end = cumulative.index.get_loc(recovery_idx)
    return int(loc_end - loc_start)


def calmar_ratio(returns: pd.Series, frequency: int = 252) -> float:
    ann_ret = annualized_return(returns, frequency=frequency)
    mdd = max_drawdown(returns)
    
    if mdd == 0:
        return np.nan
        
    return float(ann_ret / abs(mdd))


def omega_ratio(
    returns: pd.Series,
    threshold: float = 0.0,
    frequency: int = 252,
) -> float:
    """
    Omega = E[max(r - L, 0)] / E[max(L - r, 0)]
    where L is the per-period threshold.
    """
    L = threshold / frequency
    
    upside = np.maximum(returns - L, 0.0)
    downside = np.maximum(L - returns, 0.0)
    
    up_mean = upside.mean()
    down_mean = downside.mean()
    
    if down_mean == 0:
        return np.inf
        
    return float(up_mean / down_mean)


def value_at_risk_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical VaR. Returns positive loss float."""
    return float(-returns.quantile(1.0 - confidence))


def cvar_historical(returns: pd.Series, confidence: float = 0.95) -> float:
    """Historical CVaR. Returns positive conditional loss float."""
    var = value_at_risk_historical(returns, confidence)
    tail = returns[returns <= -var]
    if len(tail) == 0:
        return var
    return float(-tail.mean())


def factor_attribution(
    portfolio_returns: pd.Series,
    factor_returns: pd.DataFrame,
    frequency: int = 252,
) -> pd.DataFrame:
    """
    OLS regression: portfolio_returns = alpha + Beta * factor_returns.
    """
    # Align
    aligned = pd.concat([portfolio_returns.rename("Port"), factor_returns], axis=1).dropna()
    y = aligned["Port"].values
    X = aligned.drop(columns=["Port"]).values
    factor_names = factor_returns.columns.tolist()
    
    T, K = X.shape
    
    # Add intercept column at index 0
    X_with_alpha = np.column_stack([np.ones(T), X])
    
    # OLS: B = (X^T X)^-1 X^T y
    coefs, residuals, rank, s = np.linalg.lstsq(X_with_alpha, y, rcond=None)
    
    # T-stats and P-values
    y_hat = X_with_alpha @ coefs
    resids = y - y_hat
    resid_var = np.sum(resids**2) / (T - K - 1)
    
    try:
        cov_b = resid_var * np.linalg.inv(X_with_alpha.T @ X_with_alpha)
        std_errs = np.sqrt(np.diag(cov_b))
    except np.linalg.LinAlgError:
        std_errs = np.full(coefs.shape, np.nan)
        
    t_stats = coefs / std_errs
    # Two-sided p-value
    p_values = stats.t.sf(np.abs(t_stats), T - K - 1) * 2
    
    # Annualize alpha
    coefs[0] = coefs[0] * frequency
    
    results = pd.DataFrame(
        {
            "coefficient": coefs,
            "t_stat": t_stats,
            "p_value": p_values,
        },
        index=["alpha"] + factor_names
    )
    return results


def rolling_metrics(
    returns: pd.Series,
    window: int = 252,
    frequency: int = 252,
) -> pd.DataFrame:
    """Compute rolling Sharpe, Volatility, and Max Drawdown."""
    
    # It is faster to use pandas rolling combinations than nested apply
    roll = returns.rolling(window)
    
    roll_mean = roll.mean() * frequency
    roll_std = roll.std() * np.sqrt(frequency)
    
    # Sharpe
    roll_sharpe = roll_mean / roll_std
    
    # Max Drawdown
    # Mdd = min over window of (cum_r / cummax - 1)
    # This is trickier to vectorize perfectly in pandas, fallback to apply for MDD
    def mdd(s):
        return max_drawdown(s)
        
    roll_mdd = roll.apply(mdd, raw=False)
    
    return pd.DataFrame({
        "sharpe": roll_sharpe,
        "volatility": roll_std,
        "max_drawdown": roll_mdd
    })


def performance_summary(
    returns: pd.Series,
    benchmark: Optional[pd.Series] = None,
    risk_free_rate: float = 0.0,
    frequency: int = 252,
    name: str = "Portfolio",
) -> pd.DataFrame:
    """Full performance tearsheet."""
    
    ann_ret = annualized_return(returns, frequency)
    ann_vol = annualized_volatility(returns, frequency)
    sr = sharpe_ratio(returns, risk_free_rate, frequency)
    sortino = sortino_ratio(returns, risk_free_rate, frequency)
    calmar = calmar_ratio(returns, frequency)
    omega = omega_ratio(returns, 0.0, frequency)
    mdd = max_drawdown(returns)
    mdd_dur = max_drawdown_duration(returns)
    var = value_at_risk_historical(returns)
    cvar = cvar_historical(returns)
    
    skew = float(returns.skew())
    kurt = float(returns.kurt()) # excess kurtosis
    
    best_day = float(returns.max())
    worst_day = float(returns.min())
    
    metrics = {
        "Annualized Return": f"{ann_ret:.2%}",
        "Annualized Volatility": f"{ann_vol:.2%}",
        "Sharpe Ratio": f"{sr:.2f}",
        "Sortino Ratio": f"{sortino:.2f}",
        "Calmar Ratio": f"{calmar:.2f}",
        "Omega Ratio": f"{omega:.2f}",
        "Max Drawdown": f"{mdd:.2%}",
        "Max Drawdown Duration": f"{mdd_dur} periods" if mdd_dur is not None else "Not Recovered",
        "VaR (95%)": f"{var:.2%}",
        "CVaR (95%)": f"{cvar:.2%}",
        "Skewness": f"{skew:.2f}",
        "Excess Kurtosis": f"{kurt:.2f}",
        "Best Day": f"{best_day:.2%}",
        "Worst Day": f"{worst_day:.2%}",
    }
    
    if benchmark is not None:
        aligned = pd.concat([returns, benchmark.rename("BM")], axis=1).dropna()
        prorated_r = aligned.iloc[:, 0]
        prorated_b = aligned["BM"]
        
        b_ann_ret = annualized_return(prorated_b, frequency)
        b_ann_vol = annualized_volatility(prorated_b, frequency)
        
        # OLS Beta and Alpha
        cov_matrix = np.cov(prorated_r, prorated_b)
        beta = cov_matrix[0, 1] / cov_matrix[1, 1]
        daily_alpha = prorated_r.mean() - beta * prorated_b.mean()
        ann_alpha = daily_alpha * frequency
        
        # Tracking Error and Info Ratio
        diff = prorated_r - prorated_b
        te = diff.std() * np.sqrt(frequency)
        ir = (ann_ret - b_ann_ret) / te if te > 0 else 0.0
        
        corr = cov_matrix[0, 1] / (prorated_r.std() * prorated_b.std())
        
        metrics.update({
            "Benchmark Return": f"{b_ann_ret:.2%}",
            "Benchmark Volatility": f"{b_ann_vol:.2%}",
            "Beta": f"{beta:.2f}",
            "Alpha (Ann.)": f"{ann_alpha:.2%}",
            "Tracking Error (Ann.)": f"{te:.2%}",
            "Information Ratio": f"{ir:.2f}",
            "Correlation": f"{corr:.2f}",
        })
        
    return pd.DataFrame.from_dict(metrics, orient="index", columns=[name])
