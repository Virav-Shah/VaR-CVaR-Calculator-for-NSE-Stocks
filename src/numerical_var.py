import numpy as np
import scipy.optimize as opt
from scipy.stats import norm, t

def pinball_loss(u, alpha):
    """
    rho_alpha(u) = u * (alpha - 1{u < 0})
    """
    return u * (alpha - (u < 0))

def numerical_var_pinball(losses: np.ndarray, alpha: float = 0.99):
    """
    Numerical VaR by minimizing the pinball (quantile) loss.
    VaR = argmin_q sum(rho_alpha(L_i - q))
    """
    if len(losses) == 0:
        return np.nan
        
    def objective(q):
        return np.sum(pinball_loss(losses - q, alpha))
    
    res = opt.minimize_scalar(objective, bounds=(losses.min(), losses.max()), method='bounded')
    if res.success:
        return res.x
    return np.nan

def numerical_var_root_finding_normal(returns: np.ndarray, alpha: float = 0.99):
    """
    Root finding for Normal VaR: solve F(x) = alpha
    We solve for VaR in the loss distribution directly.
    Loss distribution has mean -mu, std sigma.
    F(q) = norm.cdf(q, loc=-mu, scale=sigma) = alpha
    So we solve F(q) - alpha = 0
    """
    mu = np.mean(returns)
    sigma = np.std(returns, ddof=1)
    
    def objective(q):
        return norm.cdf(q, loc=-mu, scale=sigma) - alpha
        
    # Bounds for VaR
    lower = -mu - 5 * sigma
    upper = -mu + 5 * sigma
    
    try:
        var = opt.brentq(objective, lower, upper)
        return var
    except ValueError:
        return np.nan

def numerical_var_root_finding_t(returns: np.ndarray, alpha: float = 0.99):
    """
    Root finding for t-distribution VaR: solve F(q) - alpha = 0
    Loss distribution: shifted and scaled t.
    """
    nu, mu, s = t.fit(returns)
    # Loss distribution has location -mu, scale s
    def objective(q):
        return t.cdf(q, df=nu, loc=-mu, scale=s) - alpha
        
    lower = -mu - 10 * s
    upper = -mu + 10 * s
    
    try:
        var = opt.brentq(objective, lower, upper)
        return var
    except ValueError:
        return np.nan

def fit_t_mle_optimizer(returns: np.ndarray):
    """
    Fit t-distribution using scipy.optimize.minimize (L-BFGS-B)
    to compare with t.fit().
    """
    # Negative log-likelihood
    def nll(params):
        nu, mu, s = params
        if nu <= 0 or s <= 0:
            return np.inf
        return -np.sum(t.logpdf(returns, df=nu, loc=mu, scale=s))
    
    # Initial guess
    mu0 = np.mean(returns)
    s0 = np.std(returns, ddof=1)
    nu0 = 5.0
    
    bounds = [(2.01, 100), (None, None), (1e-6, None)]
    
    res = opt.minimize(nll, x0=[nu0, mu0, s0], method='L-BFGS-B', bounds=bounds)
    if res.success:
        return res.x # nu, mu, s
    return np.array([np.nan, np.nan, np.nan])
