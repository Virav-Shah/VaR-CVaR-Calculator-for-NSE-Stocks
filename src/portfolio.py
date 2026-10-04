import numpy as np
import pandas as pd
from src.ru_optimization import ru_portfolio_optimization
from src.var_models import normal_var_cvar
import scipy.optimize as opt

def equal_weight_portfolio(returns: pd.DataFrame):
    """
    Returns the series of equal-weighted portfolio returns.
    """
    weights = np.ones(returns.shape[1]) / returns.shape[1]
    return returns.dot(weights)

def min_variance_portfolio(returns: pd.DataFrame, w_max: float = 0.30):
    """
    Min-variance portfolio weights.
    """
    cov = returns.cov().values
    M = returns.shape[1]
    
    def objective(w):
        return w.T @ cov @ w
        
    cons = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1.0})
    bounds = [(0, w_max) for _ in range(M)]
    w0 = np.ones(M) / M
    
    res = opt.minimize(objective, w0, method='SLSQP', bounds=bounds, constraints=cons)
    if res.success:
        return res.x
    return w0

def check_normal_portfolio_var(returns: pd.DataFrame, weights: np.ndarray, alpha: float = 0.99):
    """
    Cross-check Normal VaR against analytical formula.
    """
    port_returns = returns.dot(weights)
    var_emp, cvar_emp = normal_var_cvar(port_returns.values, alpha)
    
    mu_p = np.mean(returns.values, axis=0).dot(weights)
    cov = returns.cov().values
    sigma_p = np.sqrt(weights.T @ cov @ weights)
    
    from scipy.stats import norm
    z = norm.ppf(alpha)
    var_ana = -mu_p + sigma_p * z
    
    return var_emp, var_ana

def out_of_sample_test(returns: pd.DataFrame, train_end="2015-12-31", alpha=0.99):
    """
    Train weights on pre-2016 data, evaluate realized CVaR on post-2015 data.
    """
    train = returns[:train_end]
    test = returns[train_end:]
    
    if len(train) == 0 or len(test) == 0:
        return None
        
    # Equal weight
    w_ew = np.ones(returns.shape[1]) / returns.shape[1]
    
    # Min variance
    w_mv = min_variance_portfolio(train, w_max=0.30)
    
    # Min CVaR
    w_cvar, _, _ = ru_portfolio_optimization(train, alpha=alpha, w_max=0.30)
    
    results = {}
    for name, w in zip(["Equal Weight", "Min Variance", "Min CVaR"], [w_ew, w_mv, w_cvar]):
        if np.isnan(w).any():
            continue
        test_returns = test.dot(w)
        test_losses = -test_returns.values
        var = np.quantile(test_losses, alpha)
        cvar = test_losses[test_losses >= var].mean()
        results[name] = {"VaR": var, "CVaR": cvar}
        
    return pd.DataFrame(results).T
