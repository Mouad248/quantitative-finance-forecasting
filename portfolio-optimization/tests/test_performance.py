import numpy as np
import pandas as pd
import pytest

from quantopt.analytics.performance import (
    annualized_return,
    annualized_volatility,
    sharpe_ratio,
    sortino_ratio,
    calmar_ratio,
    omega_ratio,
    max_drawdown,
    max_drawdown_duration,
    value_at_risk_historical,
    cvar_historical,
    factor_attribution,
    rolling_metrics,
    performance_summary,
)

@pytest.fixture
def return_series():
    np.random.seed(42)
    # 2 years of daily data
    idx = pd.date_range("2020-01-01", periods=504, freq="B")
    return pd.Series(np.random.normal(0.0005, 0.015, 504), index=idx)

@pytest.fixture
def factor_data():
    np.random.seed(42)
    idx = pd.date_range("2020-01-01", periods=504, freq="B")
    port = pd.Series(np.random.normal(0.0005, 0.015, 504), index=idx)
    factors = pd.DataFrame(np.random.normal(0, 0.01, (504, 3)), index=idx, columns=["F1", "F2", "F3"])
    return port, factors

def test_annualized_metrics(return_series):
    ret = annualized_return(return_series, method="geometric")
    vol = annualized_volatility(return_series)
    sr = sharpe_ratio(return_series, risk_free_rate=0.0)
    
    assert ret > -1.0 # arbitrary sensible bound
    assert vol > 0.0
    
    # Approx SR logic
    assert np.allclose(sr, ret / vol, rtol=1e-2)

def test_drawdown(return_series):
    mdd = max_drawdown(return_series)
    dur = max_drawdown_duration(return_series)
    
    assert mdd <= 0.0
    
    if dur is not None:
        assert dur >= 0
        
def test_ratios(return_series):
    sortino = sortino_ratio(return_series)
    calmar = calmar_ratio(return_series)
    omega = omega_ratio(return_series)
    
    # Types and bounds
    assert isinstance(sortino, float)
    assert isinstance(calmar, float) or np.isnan(calmar)
    assert isinstance(omega, float)
    assert omega > 0

def test_var_cvar(return_series):
    var = value_at_risk_historical(return_series, 0.95)
    cvar = cvar_historical(return_series, 0.95)
    
    assert var > 0
    assert cvar >= var

def test_factor_attribution(factor_data):
    port, factors = factor_data
    df = factor_attribution(port, factors)
    
    assert len(df) == 4 # Alpha + 3 factors
    assert "coefficient" in df.columns
    assert "p_value" in df.columns

def test_rolling_metrics(return_series):
    roll = rolling_metrics(return_series, window=63)
    
    assert len(roll) == len(return_series)
    assert roll["sharpe"].isna().sum() == 62 # first 62 missing
    assert "max_drawdown" in roll.columns

def test_performance_summary(factor_data):
    port, _ = factor_data
    # Use one of the factors as BM
    bm = factor_data[1]["F1"]
    
    rep = performance_summary(port, benchmark=bm)
    assert len(rep) > 10
    
    assert "Sharpe Ratio" in rep.index
    assert "Beta" in rep.index
    assert "Tracking Error (Ann.)" in rep.index
