import numpy as np
import pandas as pd
from scipy.stats import chi2
from src.var_models import historical_var_cvar, normal_var_cvar, student_t_var_cvar_fitted
from scipy.stats import t
import warnings

def kupiec_pof_test(exceptions: np.ndarray, alpha: float):
    """
    Kupiec Proportion of Failures (POF) test.
    alpha is the confidence level (e.g. 0.99). The expected exception rate is p = 1 - alpha.
    """
    N = len(exceptions)
    x = np.sum(exceptions)
    p_exp = 1 - alpha
    p_obs = x / N
    
    if x == 0 or x == N:
        return np.nan
        
    lr = -2 * np.log(((1 - p_exp)**(N - x) * p_exp**x) / ((1 - p_obs)**(N - x) * p_obs**x))
    p_value = 1 - chi2.cdf(lr, df=1)
    return p_value

def christoffersen_test(exceptions: np.ndarray):
    """
    Christoffersen independence test.
    """
    T00, T01, T10, T11 = 0, 0, 0, 0
    for i in range(1, len(exceptions)):
        if exceptions[i-1] == 0 and exceptions[i] == 0:
            T00 += 1
        elif exceptions[i-1] == 0 and exceptions[i] == 1:
            T01 += 1
        elif exceptions[i-1] == 1 and exceptions[i] == 0:
            T10 += 1
        elif exceptions[i-1] == 1 and exceptions[i] == 1:
            T11 += 1
            
    if T00 + T01 == 0 or T10 + T11 == 0:
        return np.nan
        
    pi_01 = T01 / (T00 + T01)
    pi_11 = T11 / (T10 + T11)
    pi = (T01 + T11) / (T00 + T01 + T10 + T11)
    
    # Handle 0 probabilities to avoid log(0)
    def log_term(p, k):
        return k * np.log(p) if p > 0 and k > 0 else 0
        
    term1 = log_term(1 - pi, T00 + T10) + log_term(pi, T01 + T11)
    term2 = log_term(1 - pi_01, T00) + log_term(pi_01, T01) + log_term(1 - pi_11, T10) + log_term(pi_11, T11)
    
    lr = -2 * (term1 - term2)
    p_value = 1 - chi2.cdf(lr, df=1)
    return p_value

def basel_traffic_light(exceptions: int):
    """
    Basel traffic light for 250-day window at 99%.
    """
    if exceptions <= 4:
        return "Green"
    elif exceptions <= 9:
        return "Yellow"
    else:
        return "Red"

def backtest_model(returns: pd.Series, window: int = 250, alpha: float = 0.99, method="historical"):
    """
    Rolling backtest engine.
    method: "historical", "normal", "student_t"
    """
    losses = -returns
    N = len(returns)
    
    var_preds = np.full(N, np.nan)
    cvar_preds = np.full(N, np.nan)
    
    # State for t-distribution to refit every 5 days
    t_params = None
    
    for i in range(window, N):
        window_returns = returns.iloc[i-window:i].values
        window_losses = losses.iloc[i-window:i].values
        
        if method == "historical":
            var, cvar = historical_var_cvar(window_losses, alpha)
        elif method == "normal":
            var, cvar = normal_var_cvar(window_returns, alpha)
        elif method == "student_t":
            if i % 5 == 0 or t_params is None:
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore")
                    nu, mu, s = t.fit(window_returns)
                    if nu <= 2: nu = 2.1
                    t_params = (nu, mu, s)
            var, cvar = student_t_var_cvar_fitted(*t_params, alpha)
        else:
            raise ValueError("Unknown method")
            
        var_preds[i] = var
        cvar_preds[i] = cvar
        
    realized_losses = losses.values
    
    # Filter only days where predictions exist
    valid_idx = ~np.isnan(var_preds)
    var_preds_valid = var_preds[valid_idx]
    cvar_preds_valid = cvar_preds[valid_idx]
    realized_losses_valid = realized_losses[valid_idx]
    
    exceptions = (realized_losses_valid > var_preds_valid).astype(int)
    
    # Kupiec & Christoffersen
    kupiec_p = kupiec_pof_test(exceptions, alpha)
    christoffersen_p = christoffersen_test(exceptions)
    
    # ES Check (average realized loss on exception days vs avg predicted CVaR on those days)
    exc_idx = exceptions == 1
    if np.sum(exc_idx) > 0:
        es_ratio = realized_losses_valid[exc_idx].mean() / cvar_preds_valid[exc_idx].mean()
    else:
        es_ratio = np.nan
        
    # Basel (if window is exactly 250, typically applied to 1 year back. 
    # Here we just apply it to rolling 250 days over the whole series and report avg exceptions per 250 days)
    total_exceptions = np.sum(exceptions)
    years = len(var_preds_valid) / 250
    exceptions_per_year = total_exceptions / years if years > 0 else 0
    basel = basel_traffic_light(exceptions_per_year)
    
    results = {
        "Exceptions": total_exceptions,
        "Exception Rate (%)": total_exceptions / len(var_preds_valid) * 100,
        "Kupiec p-value": kupiec_p,
        "Christoffersen p-value": christoffersen_p,
        "ES Ratio": es_ratio,
        "Basel (Avg/yr)": basel,
        "Valid Days": len(var_preds_valid)
    }
    
    return results, pd.Series(var_preds, index=returns.index), pd.Series(cvar_preds, index=returns.index)
