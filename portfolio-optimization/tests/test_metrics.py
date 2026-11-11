import numpy as np
import pandas as pd
import pytest

from quantopt.risk.metrics import (
    portfolio_volatility,
    marginal_risk_contribution,
    component_risk_contribution,
    percent_risk_contribution,
    diversification_ratio,
    concentration_hhi,
    portfolio_beta,
    tracking_error,
    var_parametric,
    cvar_parametric,
    risk_report
)

@pytest.fixture
def risk_inputs():
    np.random.seed(42)
    w = pd.Series([0.4, 0.4, 0.2], index=["A", "B", "C"])
    cov = pd.DataFrame([
        [0.04, 0.01, 0.01],
        [0.01, 0.05, 0.02],
        [0.01, 0.02, 0.06]
    ], index=w.index, columns=w.index)
    return w, cov

def test_volatility(risk_inputs):
    w, cov = risk_inputs
    vol = portfolio_volatility(w, cov)
    manual = np.sqrt(w.values @ cov.values @ w.values)
    assert vol == pytest.approx(manual)

def test_risk_contributions(risk_inputs):
    w, cov = risk_inputs
    
    mrc = marginal_risk_contribution(w, cov)
    crc = component_risk_contribution(w, cov)
    prc = percent_risk_contribution(w, cov)
    
    vol = portfolio_volatility(w, cov)
    
    # Identities
    np.testing.assert_allclose(crc.sum(), vol, rtol=1e-5)
    np.testing.assert_allclose(prc.sum(), 1.0, rtol=1e-5)
    np.testing.assert_allclose(crc, w * mrc, rtol=1e-5)

def test_diversification_ratio(risk_inputs):
    w, cov = risk_inputs
    dr = diversification_ratio(w, cov)
    
    # Weighted avg vol
    wav = np.sum(w * np.sqrt(np.diag(cov)))
    vol = portfolio_volatility(w, cov)
    
    assert dr == pytest.approx(wav / vol)
    assert dr >= 1.0 # diversification benefit

def test_concentration(risk_inputs):
    w, _ = risk_inputs
    hhi = concentration_hhi(w)
    
    assert hhi == pytest.approx(0.4**2 + 0.4**2 + 0.2**2)

def test_beta_and_te(risk_inputs):
    w, cov = risk_inputs
    bm_w = pd.Series([0.333, 0.333, 0.334], index=w.index)
    
    beta = portfolio_beta(w, cov, bm_w)
    te = tracking_error(w, bm_w, cov)
    
    assert te > 0
    assert beta > 0

def test_var_cvar(risk_inputs):
    w, cov = risk_inputs
    var = var_parametric(w, cov, confidence=0.95)
    cvar = cvar_parametric(w, cov, confidence=0.95)
    
    assert cvar > var > 0

def test_risk_report(risk_inputs):
    w, cov = risk_inputs
    bm_w = pd.Series([0.333, 0.333, 0.334], index=w.index)
    
    rep = risk_report(w, cov, benchmark_weights=bm_w)
    assert len(rep) > 5
    assert "Volatility (Ann.)" in rep.index
    assert "Expected Loss" not in rep.index # Just making sure it's valid
