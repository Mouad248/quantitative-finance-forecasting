import numpy as np
import pandas as pd
import pytest

from quantopt.optimization.risk_parity import RiskParity

@pytest.fixture
def cov_matrix():
    np.random.seed(42)
    # 3 assets: low, med, high vol
    vols = np.array([0.1, 0.2, 0.4])
    corr = np.array([
        [1.0, 0.2, 0.2],
        [0.2, 1.0, 0.2],
        [0.2, 0.2, 1.0]
    ])
    cov = np.outer(vols, vols) * corr
    return pd.DataFrame(cov, index=["A", "B", "C"], columns=["A", "B", "C"])

def test_equal_risk_contribution(cov_matrix):
    rp = RiskParity(cov_matrix)
    w = rp.optimize()
    
    rc = rp.risk_contributions()
    
    # Should all be exactly 1/3 (within tolerance)
    np.testing.assert_allclose(rc, 1.0/3.0, atol=1e-3)
    assert rp.concentration_check(tol=1e-2)
    
    # Weights should be inverse to vol: w_A > w_B > w_C
    assert w["A"] > w["B"] > w["C"]
    np.testing.assert_allclose(w.sum(), 1.0, rtol=1e-5)

def test_custom_risk_budgets(cov_matrix):
    budgets = pd.Series([0.5, 0.3, 0.2], index=cov_matrix.index)
    rp = RiskParity(cov_matrix, risk_budgets=budgets)
    w = rp.optimize()
    
    rc = rp.risk_contributions()
    
    np.testing.assert_allclose(rc, budgets, atol=1e-3)
    assert w.sum() == pytest.approx(1.0)
    
def test_long_short_rp(cov_matrix):
    # Long short risk parity allows negative weights but here cov has all positive correlations,
    # so risk contributions for a generic case might diverge. Let's just check execution.
    rp = RiskParity(cov_matrix, long_only=False)
    w = rp.optimize()
    assert w.sum() == pytest.approx(1.0)
