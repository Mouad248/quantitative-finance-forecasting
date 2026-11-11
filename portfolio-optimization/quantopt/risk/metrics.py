"""
Portfolio-level risk metrics.
"""
import numpy as np
import pandas as pd
import scipy.stats as stats
from typing import Optional


def portfolio_volatility(weights: pd.Series, Sigma: pd.DataFrame) -> float:
    """
    Return annualized portfolio volatility sqrt(w^T * Sigma * w).
    
    Parameters
    ----------
    weights : pd.Series
        Portfolio weights.
    Sigma : pd.DataFrame
        Annualized covariance matrix.

    Returns
    -------
    float
    """
    w = weights.reindex(Sigma.index).fillna(0.0).values
    S = Sigma.values
    return float(np.sqrt(w @ S @ w))


def marginal_risk_contribution(weights: pd.Series, Sigma: pd.DataFrame) -> pd.Series:
    """
    Sensitivity of portfolio volatility to a marginal increase in current weights.
    MRC_i = (Sigma * w)_i / sigma_p

    Parameters
    ----------
    weights : pd.Series
        Portfolio weights.
    Sigma : pd.DataFrame
        Annualized covariance matrix.

    Returns
    -------
    pd.Series
    """
    w = weights.reindex(Sigma.index).fillna(0.0).values
    S = Sigma.values
    vol = np.sqrt(w @ S @ w)
    
    if vol == 0:
        return pd.Series(0.0, index=Sigma.index)
        
    mrc = (S @ w) / vol
    return pd.Series(mrc, index=Sigma.index)


def component_risk_contribution(weights: pd.Series, Sigma: pd.DataFrame) -> pd.Series:
    """
    Absolute contribution of each asset to the total portfolio volatility.
    CRC_i = w_i * MRC_i

    Parameters
    ----------
    weights : pd.Series
    Sigma : pd.DataFrame

    Returns
    -------
    pd.Series
    """
    w = weights.reindex(Sigma.index).fillna(0.0)
    mrc = marginal_risk_contribution(weights, Sigma)
    return w * mrc


def percent_risk_contribution(weights: pd.Series, Sigma: pd.DataFrame) -> pd.Series:
    """
    Percentage of total portfolio volatility contributed by each asset.
    PRC_i = CRC_i / sigma_p

    Parameters
    ----------
    weights : pd.Series
    Sigma : pd.DataFrame

    Returns
    -------
    pd.Series
    """
    crc = component_risk_contribution(weights, Sigma)
    vol = portfolio_volatility(weights, Sigma)
    
    if vol == 0:
        return pd.Series(1.0 / len(weights), index=Sigma.index)
        
    return crc / vol


def diversification_ratio(weights: pd.Series, Sigma: pd.DataFrame) -> float:
    """
    Ratio of the weighted average of individual volatilities to the portfolio volatility.
    DR = (sum w_i * sigma_i) / sigma_p
    Higher is better. 1.0 means no diversification benefit.

    Parameters
    ----------
    weights : pd.Series
    Sigma : pd.DataFrame

    Returns
    -------
    float
    """
    w = weights.reindex(Sigma.index).fillna(0.0).values
    vols = np.sqrt(np.diag(Sigma.values))
    
    weighted_vol = np.sum(w * vols)
    port_vol = portfolio_volatility(weights, Sigma)
    
    if port_vol == 0:
        return 1.0
        
    return float(weighted_vol / port_vol)


def concentration_hhi(weights: pd.Series) -> float:
    """
    Herfindahl-Hirschman Index (HHI) for weight concentration.
    HHI = sum w_i^2

    Range [1/N, 1]. Lower means less concentrated.

    Parameters
    ----------
    weights : pd.Series

    Returns
    -------
    float
    """
    w = weights.values
    return float(np.sum(w**2))


def effective_n(weights: pd.Series) -> float:
    """
    Effective number of independent positions (inverse of HHI).

    Parameters
    ----------
    weights : pd.Series

    Returns
    -------
    float
    """
    hhi = concentration_hhi(weights)
    return 1.0 / hhi if hhi > 0 else 0.0


def portfolio_beta(
    weights: pd.Series,
    Sigma: pd.DataFrame,
    market_weights: pd.Series,
) -> float:
    """
    Beta of the portfolio relative to a market portfolio, computed via the covariance matrix.
    beta_p = (w^T * Sigma * w_mkt) / (w_mkt^T * Sigma * w_mkt)

    Parameters
    ----------
    weights : pd.Series
    Sigma : pd.DataFrame
    market_weights : pd.Series

    Returns
    -------
    float
    """
    w = weights.reindex(Sigma.index).fillna(0.0).values
    w_mkt = market_weights.reindex(Sigma.index).fillna(0.0).values
    S = Sigma.values
    
    mkt_var = w_mkt @ S @ w_mkt
    if mkt_var == 0:
        return 0.0
        
    covar = w @ S @ w_mkt
    return float(covar / mkt_var)


def tracking_error(
    weights: pd.Series,
    benchmark_weights: pd.Series,
    Sigma: pd.DataFrame,
) -> float:
    """
    Ex-ante tracking error relative to a benchmark.
    TE = sqrt((w - w_bm)^T * Sigma * (w - w_bm))

    Parameters
    ----------
    weights : pd.Series
    benchmark_weights : pd.Series
    Sigma : pd.DataFrame

    Returns
    -------
    float
    """
    w = weights.reindex(Sigma.index).fillna(0.0).values
    w_bm = benchmark_weights.reindex(Sigma.index).fillna(0.0).values
    S = Sigma.values
    
    diff = w - w_bm
    return float(np.sqrt(np.maximum(diff @ S @ diff, 0.0)))


def factor_exposure(weights: pd.Series, factor_loadings: pd.DataFrame) -> pd.Series:
    """
    Portfolio exposure to underlying factors.
    B^T * w

    Parameters
    ----------
    weights : pd.Series
        N-vector of weights.
    factor_loadings : pd.DataFrame
        N x K matrix of factor loadings.

    Returns
    -------
    pd.Series
        K-vector of portfolio exposures.
    """
    w = weights.reindex(factor_loadings.index).fillna(0.0).values
    B = factor_loadings.values
    
    exposures = B.T @ w
    return pd.Series(exposures, index=factor_loadings.columns)


def var_parametric(
    weights: pd.Series,
    Sigma: pd.DataFrame,
    confidence: float = 0.95,
    frequency: int = 252,
) -> float:
    """
    Gaussian Value-at-Risk (VaR).
    VaR = z_alpha * sigma_p / sqrt(frequency)

    Parameters
    ----------
    weights : pd.Series
    Sigma : pd.DataFrame
    confidence : float, default 0.95
    frequency : int, default 252

    Returns
    -------
    float
        Positive value representing loss.
    """
    vol = portfolio_volatility(weights, Sigma)
    daily_vol = vol / np.sqrt(frequency)
    z_score = stats.norm.ppf(confidence)
    
    return float(z_score * daily_vol)


def cvar_parametric(
    weights: pd.Series,
    Sigma: pd.DataFrame,
    confidence: float = 0.95,
    frequency: int = 252,
) -> float:
    """
    Gaussian Conditional Value-at-Risk (CVaR).
    Expected loss given return < -VaR.

    CVaR = phi(z_alpha) / (1 - confidence) * sigma_p / sqrt(frequency)
    where phi is the standard normal PDF.

    Parameters
    ----------
    weights : pd.Series
    Sigma : pd.DataFrame
    confidence : float, default 0.95
    frequency : int, default 252

    Returns
    -------
    float
        Positive value representing conditional loss.
    """
    vol = portfolio_volatility(weights, Sigma)
    daily_vol = vol / np.sqrt(frequency)
    
    z_score = stats.norm.ppf(confidence)
    pdf_val = stats.norm.pdf(z_score)
    
    return float((pdf_val / (1.0 - confidence)) * daily_vol)


def stress_test(
    weights: pd.Series,
    scenario_returns: pd.DataFrame,
) -> pd.Series:
    """
    Evaluate portfolio P&L under a set of historical or synthetic scenarios.

    Parameters
    ----------
    weights : pd.Series
    scenario_returns : pd.DataFrame
        Rows are scenarios, columns are assets.

    Returns
    -------
    pd.Series
        Portfolio return per scenario.
    """
    w = weights.reindex(scenario_returns.columns).fillna(0.0).values
    R = scenario_returns.values
    
    port_returns = R @ w
    return pd.Series(port_returns, index=scenario_returns.index)


def risk_report(
    weights: pd.Series,
    Sigma: pd.DataFrame,
    benchmark_weights: Optional[pd.Series] = None,
    confidence: float = 0.95,
) -> pd.DataFrame:
    """
    Comprehensive single-column tear sheet of portfolio risk metrics.

    Parameters
    ----------
    weights : pd.Series
    Sigma : pd.DataFrame
    benchmark_weights : pd.Series, optional
    confidence : float, default 0.95

    Returns
    -------
    pd.DataFrame
    """
    vol = portfolio_volatility(weights, Sigma)
    dr = diversification_ratio(weights, Sigma)
    hhi = concentration_hhi(weights)
    eff_n = effective_n(weights)
    pvar = var_parametric(weights, Sigma, confidence)
    pcvar = cvar_parametric(weights, Sigma, confidence)
    
    # Top 3 assets by Marginal Risk Contribution
    mrc = marginal_risk_contribution(weights, Sigma)
    top_3_mrc = mrc.nlargest(3)
    mrc_str = ", ".join([f"{idx} ({val:.2%})" for idx, val in top_3_mrc.items()])
    
    # Top 3 assets by Percent Risk Contribution
    prc = percent_risk_contribution(weights, Sigma)
    top_3_prc = prc.nlargest(3)
    prc_str = ", ".join([f"{idx} ({val:.1%})" for idx, val in top_3_prc.items()])
    
    metrics = {
        "Volatility (Ann.)": f"{vol:.2%}",
        "Diversification Ratio": f"{dr:.2f}",
        "Concentration HHI": f"{hhi:.4f}",
        "Effective N": f"{eff_n:.1f}",
        f"Parametric VaR ({confidence:.0%})": f"{pvar:.2%}",
        f"Parametric CVaR ({confidence:.0%})": f"{pcvar:.2%}",
        "Top 3 MRC": mrc_str,
        "Top 3 PRC": prc_str,
    }
    
    if benchmark_weights is not None:
        te = tracking_error(weights, benchmark_weights, Sigma)
        beta = portfolio_beta(weights, Sigma, benchmark_weights)
        metrics["Tracking Error (Ann.)"] = f"{te:.2%}"
        metrics["Portfolio Beta"] = f"{beta:.3f}"
        
    return pd.DataFrame.from_dict(metrics, orient="index", columns=["Value"])
