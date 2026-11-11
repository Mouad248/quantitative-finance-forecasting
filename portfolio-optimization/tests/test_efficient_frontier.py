import numpy as np
import pandas as pd
import pytest

from quantopt.optimization.efficient_frontier import EfficientFrontier, OptimizationError, InfeasibleError
from quantopt.optimization.constraints import ConstraintSet

@pytest.fixture
def opt_inputs():
    np.random.seed(42)
    n = 4
    mu = pd.Series(np.random.uniform(0.05, 0.15, n), index=list("ABCD"))
    
    # Generate PSD cov
    A = np.random.normal(0, 1, (n, n))
    cov = pd.DataFrame(A @ A.T, index=mu.index, columns=mu.index)
    
    return mu, cov

def test_max_sharpe(opt_inputs):
    mu, cov = opt_inputs
    ef = EfficientFrontier(mu, cov)
    w = ef.max_sharpe(risk_free_rate=0.0)
    
    # Should sum to 1 and be positive
    np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-5)
    assert np.all(w >= -1e-5)

def test_min_volatility(opt_inputs):
    mu, cov = opt_inputs
    ef = EfficientFrontier(mu, cov)
    w = ef.min_volatility()
    
    np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-5)
    
    ret, vol, _ = ef.portfolio_performance(mu, cov)
    assert vol < np.sqrt(np.diag(cov)).min() # Vol must be less than min asset vol (diversification)

def test_efficient_risk_return(opt_inputs):
    mu, cov = opt_inputs
    ef = EfficientFrontier(mu, cov)
    
    # Target return 
    target_ret = float(mu.mean())
    w = ef.efficient_return(target_ret)
    ret, _, _ = ef.portfolio_performance(mu, cov)
    np.testing.assert_allclose(ret, target_ret, rtol=0.10)
    
    # Target risk
    w_min = ef.min_volatility()
    min_vol = np.sqrt(w_min.values @ cov.values @ w_min.values)
    
    target_vol = min_vol * 1.5
    w2 = ef.efficient_risk(target_vol)
    _, vol2, _ = ef.portfolio_performance(mu, cov)
    np.testing.assert_allclose(vol2, target_vol, rtol=0.10)

def test_custom_constraints(opt_inputs):
    mu, cov = opt_inputs
    
    # 1. Long/Short max pos
    cs = ConstraintSet(len(mu)).long_short(gross_exposure=1.5, net_exposure=0.0).max_position(0.8)
    
    ef = EfficientFrontier(mu, cov, constraint_set=cs)
    w = ef.min_volatility()
    
    # 2. Check net exposure 0
    np.testing.assert_allclose(w.sum(), 0.0, atol=1e-4)
    
    # 3. Check max position
    assert np.max(w) <= 0.8 + 1e-4

def test_infeasible_err(opt_inputs):
    mu, cov = opt_inputs
    ef = EfficientFrontier(mu, cov)
    
    with pytest.raises(InfeasibleError):
        ef.efficient_return(2.0) # way out of bounds
        
    w_min = ef.min_volatility()
    min_vol = np.sqrt(w_min.values @ cov.values @ w_min.values)
    
    with pytest.raises(InfeasibleError):
        ef.efficient_risk(min_vol / 2) # below global minimum
