import numpy as np
import pandas as pd
import pytest

from quantopt.optimization.cvar_optimizer import CVaROptimizer

@pytest.fixture
def returns_data():
    np.random.seed(42)
    # Fat tails for C
    A = np.random.normal(0.001, 0.01, 500)
    B = np.random.normal(0.002, 0.02, 500)
    C = np.random.standard_t(df=3, size=500) * 0.01 + 0.003
    return pd.DataFrame({"A": A, "B": B, "C": C})

def test_cvar_optimization(returns_data):
    opt = CVaROptimizer(returns_data, beta=0.95)
    w = opt.optimize()
    
    np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-5)
    
    assert opt.var_ > 0
    assert opt.cvar_ > opt.var_
    
    # Check decomposition
    decomp = opt.cvar_decomposition()
    np.testing.assert_allclose(decomp.sum(), opt.cvar_, rtol=0.10)
    
def test_cvar_mean_return_target(returns_data):
    # Equal weight mean return
    ew_ret = returns_data.mean(axis=1).mean()
    
    # Target above ew
    target = ew_ret * 1.1
    
    opt = CVaROptimizer(returns_data, beta=0.95, mean_return_target=target)
    w = opt.optimize()
    
    port_mu = w.values @ returns_data.mean().values
    assert port_mu >= target - 1e-5
    
def test_cvar_bounds(returns_data):
    opt = CVaROptimizer(returns_data, beta=0.95, weight_bounds=(0.1, 0.5))
    w = opt.optimize()
    
    assert np.all(w >= 0.1 - 1e-4)
    assert np.all(w <= 0.5 + 1e-4)
    np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-5)
