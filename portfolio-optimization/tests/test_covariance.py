import numpy as np
import pandas as pd
import pytest

from quantopt.risk.covariance import (
    SampleCovariance,
    EWMCovariance,
    LedoitWolfCovariance,
    FactorModelCovariance,
)

@pytest.fixture
def returns_data():
    np.random.seed(42)
    idx = pd.date_range("2020-01-01", periods=300, freq="B")
    return pd.DataFrame(np.random.normal(0.0005, 0.01, (300, 4)), index=idx, columns=list("ABCD"))

def test_sample_covariance(returns_data):
    est = SampleCovariance(frequency=252)
    est.fit(returns_data)
    S = est.covariance()
    
    # check PSD
    eigvals = np.linalg.eigvalsh(S.values)
    assert np.all(eigvals >= -1e-10)
    
    # manual check
    manual_S = returns_data.cov() * 252
    np.testing.assert_allclose(S.values, manual_S.values)
    
def test_ewm_covariance(returns_data):
    base_S = returns_data.cov() * 252
    
    est = EWMCovariance(span=180, frequency=252)
    est.fit(returns_data)
    S = est.covariance()
    
    assert list(S.index) == list("ABCD")
    assert np.all(np.linalg.eigvalsh(S.values) >= -1e-10)
    
    # Should not equal equal-weighted sample cov exactly
    assert not np.allclose(S.values, base_S.values)

def test_ledoit_wolf_covariance(returns_data):
    est = LedoitWolfCovariance()
    est.fit(returns_data)
    S = est.covariance()
    
    assert est.shrinkage_ is not None
    assert 0 <= est.shrinkage_ <= 1.0
    assert np.all(np.linalg.eigvalsh(S.values) >= -1e-10)

def test_factor_model_covariance(returns_data):
    # Just 2 components
    est = FactorModelCovariance(n_factors=2, frequency=252)
    est.fit(returns_data)
    S = est.covariance()
    
    assert est.factor_loadings_ is not None
    assert est.factor_loadings_.shape == (4, 2)
    assert est.idiosyncratic_var_ is not None
    
    # Check PSD
    assert np.all(np.linalg.eigvalsh(S.values) >= -1e-10)

def test_correlation_volatility_extraction(returns_data):
    est = SampleCovariance(frequency=252)
    est.fit(returns_data)
    
    corr = est.correlation()
    vol = est.std()
    
    assert np.allclose(np.diag(corr), 1.0)
    assert vol["A"] > 0
    
    # check correlation reconstruction
    D = np.diag(vol.values)
    reconst_S = D @ corr.values @ D
    np.testing.assert_allclose(reconst_S, est.covariance().values, rtol=1e-5)
