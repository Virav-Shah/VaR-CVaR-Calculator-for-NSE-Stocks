import numpy as np
from scipy.stats import norm, t
import warnings

def historical_var_cvar(losses: np.ndarray, alpha: float = 0.99):
    """
    Historical Simulation for VaR and CVaR.
    losses: array of losses (positive values mean losing money)
    alpha: confidence level
    """
    if len(losses) == 0:
        return np.nan, np.nan
    var = np.quantile(losses, alpha)
    tail_losses = losses[losses >= var]
    cvar = tail_losses.mean() if len(tail_losses) > 0 else np.nan
    return var, cvar

def normal_var_cvar(returns: np.ndarray, alpha: float = 0.99, assume_zero_mean: bool = False):
    """
    Parametric Normal VaR and CVaR.
    returns: array of returns
    alpha: confidence level
    """
    if len(returns) == 0:
        return np.nan, np.nan
    
    mu = 0 if assume_zero_mean else np.mean(returns)
    sigma = np.std(returns, ddof=1)
    
    z_alpha = norm.ppf(alpha)
    var = -mu + sigma * z_alpha
    cvar = -mu + sigma * norm.pdf(z_alpha) / (1 - alpha)
    
    return var, cvar

def student_t_var_cvar(returns: np.ndarray, alpha: float = 0.99):
    """
    Parametric Student's t VaR and CVaR.
    returns: array of returns
    """
    if len(returns) == 0:
        return np.nan, np.nan
    
    # Fit t-distribution, constrain nu > 2
    # scipy.stats.t.fit takes data, and optionally floc, fscale to fix them.
    # It doesn't natively support bounds on nu, but we can use numerical optimization
    # if it fails, or just trust the fit and cap it.
    # A common approach is just fitting, usually nu > 2 for equity returns.
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        # To enforce nu > 2.1 we can use a custom objective, but t.fit often works
        # Let's try t.fit first.
        nu, mu, s = t.fit(returns)
    
    # Enforce nu > 2 strictly for CVaR to exist, although t.fit might yield something else
    if nu <= 2:
        nu = 2.1 # Cap to prevent undefined variance/CVaR formulas blowing up
        
    t_alpha = t.ppf(alpha, nu)
    var = -mu + s * t_alpha
    
    # CVaR = -mu + s * [ f_nu(t_alpha) / (1-alpha) ] * [ (nu + t_alpha^2) / (nu - 1) ]
    f_nu = t.pdf(t_alpha, nu)
    cvar = -mu + s * (f_nu / (1 - alpha)) * ((nu + t_alpha**2) / (nu - 1))
    
    return var, cvar, nu, mu, s

def student_t_var_cvar_fitted(nu: float, mu: float, s: float, alpha: float = 0.99):
    """
    Calculate t-VaR/CVaR from pre-fitted parameters (useful for rolling windows to save compute).
    """
    t_alpha = t.ppf(alpha, nu)
    var = -mu + s * t_alpha
    f_nu = t.pdf(t_alpha, nu)
    cvar = -mu + s * (f_nu / (1 - alpha)) * ((nu + t_alpha**2) / (nu - 1))
    return var, cvar
