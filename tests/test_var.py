import numpy as np
import pandas as pd
import pytest
from scipy.stats import norm, t
from src.var_models import historical_var_cvar, normal_var_cvar, student_t_var_cvar, student_t_var_cvar_fitted
from src.ru_optimization import ru_cvar_single, ru_portfolio_optimization
from src.numerical_var import numerical_var_pinball, numerical_var_root_finding_normal, numerical_var_root_finding_t

np.random.seed(42)

@pytest.fixture
def mock_returns():
    # Simulate some t-distributed returns
    return t.rvs(df=4, loc=0.0005, scale=0.015, size=2000)

@pytest.fixture
def mock_losses(mock_returns):
    return -mock_returns

def test_cvar_ge_var(mock_returns, mock_losses):
    alpha = 0.95
    
    var_h, cvar_h = historical_var_cvar(mock_losses, alpha)
    assert cvar_h >= var_h
    
    var_n, cvar_n = normal_var_cvar(mock_returns, alpha)
    assert cvar_n >= var_n
    
    var_t, cvar_t, _, _, _ = student_t_var_cvar(mock_returns, alpha)
    assert cvar_t >= var_t

def test_ru_single_equals_historical(mock_losses):
    alpha = 0.95
    var_h, cvar_h = historical_var_cvar(mock_losses, alpha)
    var_ru, cvar_ru = ru_cvar_single(mock_losses, alpha)
    
    np.testing.assert_almost_equal(cvar_h, cvar_ru, decimal=4)
    # var might have small numerical differences, but should be close
    np.testing.assert_almost_equal(var_h, var_ru, decimal=4)

def test_numerical_var_agrees_with_quantile(mock_losses):
    alpha = 0.95
    var_quantile = np.quantile(mock_losses, alpha)
    var_pinball = numerical_var_pinball(mock_losses, alpha)
    np.testing.assert_almost_equal(var_quantile, var_pinball, decimal=3)

def test_normal_numerical_root(mock_returns):
    alpha = 0.99
    var_ana, _ = normal_var_cvar(mock_returns, alpha)
    var_root = numerical_var_root_finding_normal(mock_returns, alpha)
    np.testing.assert_almost_equal(var_ana, var_root, decimal=5)

def test_t_cvar_converges_to_normal():
    # Simulate with large nu
    returns = norm.rvs(loc=0, scale=0.02, size=5000)
    var_n, cvar_n = normal_var_cvar(returns, alpha=0.99)
    # Force evaluate t with nu=1000
    var_t, cvar_t = student_t_var_cvar_fitted(nu=1000, mu=np.mean(returns), s=np.std(returns, ddof=1), alpha=0.99)
    
    np.testing.assert_almost_equal(var_n, var_t, decimal=3)
    np.testing.assert_almost_equal(cvar_n, cvar_t, decimal=3)

def test_closed_form_t_cvar_matches_mc():
    alpha = 0.99
    nu, mu, s = 4.0, 0.0, 0.02
    var_ana, cvar_ana = student_t_var_cvar_fitted(nu, mu, s, alpha)
    
    # Monte carlo
    np.random.seed(42)
    sim_returns = t.rvs(df=nu, loc=mu, scale=s, size=1_000_000)
    sim_losses = -sim_returns
    var_mc = np.quantile(sim_losses, alpha)
    cvar_mc = sim_losses[sim_losses >= var_mc].mean()
    
    # Check within 2% relative error
    assert np.abs(cvar_ana - cvar_mc) / cvar_ana < 0.02
