import numpy as np
import pandas as pd
import pytest

from quantopt.returns.estimators import (
    MeanHistoricalReturn,
    CAPMReturn,
    BlackLittermanReturn,
)

@pytest.fixture
def sample_returns():
    np.random.seed(42)
    idx = pd.date_range("2020-01-01", periods=252, freq="B")
    return pd.DataFrame(np.random.normal(0.0005, 0.01, (252, 3)), index=idx, columns=["A", "B", "C"])

def test_mean_historical(sample_returns):
    est = MeanHistoricalReturn(frequency=252)
    est.fit(sample_returns)
    mu = est.expected_returns()
    
    assert len(mu) == 3
    assert list(mu.index) == ["A", "B", "C"]
    
    # Manual check
    manual_mu = (1 + sample_returns.mean()) ** 252 - 1
    np.testing.assert_allclose(mu.values, manual_mu.values, rtol=1e-5)

def test_capm_return(sample_returns):
    # Mock benchmark
    bm = sample_returns.mean(axis=1) + np.random.normal(0, 0.001, 252)
    
    est = CAPMReturn(market_returns=bm, risk_free_rate=0.02)
    est.fit(sample_returns)
    mu = est.expected_returns()
    
    assert len(mu) == 3
    assert np.all(mu > -1.0) # Reasonable bounds
    
    # Check betas
    np.testing.assert_allclose(est.betas_.shape, (3,))

def test_black_litterman(sample_returns):
    market_caps = pd.Series([100.0, 200.0, 300.0], index=["A", "B", "C"])
    
    # Views: A will beat B by 2%, C absolute return is 5%
    P = pd.DataFrame([
        [1.0, -1.0, 0.0],
        [0.0, 0.0, 1.0]
    ], columns=["A", "B", "C"])
    Q = pd.Series([0.02, 0.05])
    
    bl = BlackLittermanReturn(
        market_caps=market_caps,
        P=P,
        Q=Q,
        tau=0.05,
    )
    bl.fit(sample_returns)
    
    post_mu = bl.expected_returns()
    
    assert len(post_mu) == 3
    assert bl.posterior_cov_ is not None
    assert bl.posterior_cov_.shape == (3, 3)
    
    # The view that A beats B should pull A up and B down relative to prior
    assert post_mu["A"] - post_mu["B"] > bl.pi_["A"] - bl.pi_["B"]
