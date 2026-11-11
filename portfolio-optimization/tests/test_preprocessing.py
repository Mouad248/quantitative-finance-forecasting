import numpy as np
import pandas as pd
import pytest

from quantopt.returns.preprocessing import (
    prices_to_returns,
    returns_to_prices,
    annualization_factor,
    demean_cross_sectional,
    winsorize_returns,
)
from quantopt.utils.validation import check_psd, project_psd

def test_prices_to_returns():
    idx = pd.date_range("2020-01-01", periods=3)
    pr = pd.DataFrame({"A": [100, 105, 102], "B": [50, 50, 52]}, index=idx)
    ret = prices_to_returns(pr, method="simple")
    
    assert len(ret) == 2
    assert ret.index[0] == pd.Timestamp("2020-01-02") # integer index drops first row
    np.testing.assert_allclose(ret["A"].values, [0.05, -0.02857143], rtol=1e-5)
    np.testing.assert_allclose(ret["B"].values, [0.0, 0.04], rtol=1e-5)

def test_returns_to_prices():
    idx = pd.date_range("2020-01-01", periods=2)
    ret = pd.DataFrame({"A": [0.05, -0.02857143], "B": [0.0, 0.04]}, index=idx)
    pr = returns_to_prices(ret, method="simple", initial_price=1.0)
    
    assert len(pr) == 3
    np.testing.assert_allclose(pr.iloc[0].values, [1.0, 1.0])
    np.testing.assert_allclose(pr.iloc[-1].values, [1.02, 1.04], rtol=1e-5)

def test_annualization_factor():
    idx_daily = pd.date_range("2020-01-01", periods=10, freq="B")
    ret_daily = pd.Series(np.random.randn(10), index=idx_daily)
    assert annualization_factor(ret_daily) == 252

    idx_monthly = pd.date_range("2020-01-01", periods=10, freq="ME")
    ret_monthly = pd.Series(np.random.randn(10), index=idx_monthly)
    assert annualization_factor(ret_monthly) == 12

def test_demean_cross_sectional():
    idx = pd.date_range("2020-01-01", periods=2)
    ret = pd.DataFrame({"A": [0.05, 0.01], "B": [0.03, -0.01]}, index=idx)
    demeaned = demean_cross_sectional(ret)
    
    np.testing.assert_allclose(demeaned.sum(axis=1).values, [0.0, 0.0], atol=1e-10)

def test_winsorize_returns():
    idx = pd.date_range("2020-01-01", periods=5)
    ret = pd.DataFrame({"X": [-0.5, -0.01, 0.0, 0.02, 0.6]}, index=idx)
    win = winsorize_returns(ret, lower_pct=0.20, upper_pct=0.80)
    
    assert win.iloc[1, 0] == -0.01
    assert win.iloc[3, 0] == 0.02
    
def test_psd_check_project():
    # Not PSD
    A = np.array([[1.0, 0.9, 0.9], [0.9, 1.0, 0.9], [0.9, 0.9, 0.1]])
    
    assert not check_psd(A)
    
    # Project
    A_psd = project_psd(A)
    assert check_psd(A_psd)
    
    # Needs to be close to symmetric
    np.testing.assert_allclose(A_psd, A_psd.T)
