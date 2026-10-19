import numpy as np
import pandas as pd
from scipy.stats import chi2
from src.var_models import historical_var_cvar, normal_var_cvar, student_t_var_cvar_fitted
from scipy.stats import t
import warnings

def coverage_statistics(exceptions, alpha):
    from scipy.special import xlogy
    e = np.asarray(exceptions)
    if e.ndim != 1 or len(e) == 0 or not 0 < alpha < 1 or not np.isin(e, [0, 1]).all():
        raise ValueError("Nonempty binary exceptions and alpha in (0,1) required")
    e = e.astype(int)
    n, x = len(e), e.sum()
    def ll(success, total, prob):
        return xlogy(success, prob) + xlogy(total - success, 1 - prob)
    lr_pof = max(0., float(2 * (ll(x, n, x/n) - ll(x, n, 1-alpha))))
    lr_ind = np.nan
    if n > 1:
        counts = np.zeros((2, 2), dtype=int)
        np.add.at(counts, (e[:-1], e[1:]), 1)
        totals = counts.sum(axis=1)
        # No transitions from one state: independence is not identifiable.
        if (totals > 0).all():
            probs = counts[:, 1] / totals
            pooled = counts[:, 1].sum() / totals.sum()
            lr_ind = max(0., float(2 * (sum(ll(counts[i, 1], totals[i], probs[i]) for i in (0, 1))
                                      - ll(counts[:, 1].sum(), totals.sum(), pooled))))
    lr_cc = lr_pof + lr_ind
    return {"LR POF": lr_pof, "Kupiec p-value": chi2.sf(lr_pof, 1),
            "LR Independence": lr_ind, "Christoffersen p-value": chi2.sf(lr_ind, 1),
            "LR Conditional Coverage": lr_cc, "Conditional Coverage p-value": chi2.sf(lr_cc, 2)}


def kupiec_pof_test(exceptions, alpha):
    return coverage_statistics(exceptions, alpha)["Kupiec p-value"]


def christoffersen_test(exceptions):
    return coverage_statistics(exceptions, .99)["Christoffersen p-value"]


def evaluate_forecasts(losses, var, cvar=None, alpha=.99):
    """Common evaluator for baseline and external date-indexed ML forecasts."""
    from src.numerical_var import pinball_loss
    for series in (losses, var):
        if not series.index.is_unique or not series.index.is_monotonic_increasing:
            raise ValueError("Forecast evaluation requires unique chronological dates")
    frame = pd.concat({"loss": losses, "var": var}, axis=1)
    if cvar is not None:
        frame["cvar"] = cvar
    frame = frame.replace([np.inf, -np.inf], np.nan).dropna()
    if frame.empty:
        raise ValueError("No aligned finite forecasts to evaluate")
    e = (frame.loss > frame['var']).astype(int)
    out = coverage_statistics(e, alpha)
    out.update({"Exceptions": int(e.sum()), "Exception Rate (%)": 100 * e.mean(),
                "Valid Days": len(e), "Pinball Loss": float(pinball_loss(frame.loss - frame['var'], alpha).mean()),
                "Calibration Error": abs(e.mean() - (1-alpha)),
                "Mean VaR": frame['var'].mean(),
                "Breach Severity": (frame.loss - frame['var'])[e.astype(bool)].mean(),
                "Basel (Last 250)": basel_traffic_light(int(e.iloc[-250:].sum())) if alpha == .99 and len(e) >= 250 else "N/A",
                "ES Ratio": np.nan})
    if cvar is not None and e.sum():
        denom = frame.loc[e.astype(bool), 'cvar'].mean()
        out['ES Ratio'] = frame.loc[e.astype(bool), 'loss'].mean() / denom if denom else np.nan
    return out

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
    returns = returns.dropna().sort_index()
    if window < 2 or len(returns) <= window:
        raise ValueError("Need more observations than the rolling window")
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
        
    var_series = pd.Series(var_preds, index=returns.index)
    cvar_series = pd.Series(cvar_preds, index=returns.index)
    results = evaluate_forecasts(losses, var_series, cvar_series, alpha)
    return results, var_series, cvar_series
