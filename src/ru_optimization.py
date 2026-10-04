import numpy as np
import scipy.optimize as opt
import pandas as pd

def ru_cvar_single(losses: np.ndarray, alpha: float = 0.99):
    """
    Rockafellar-Uryasev CVaR for a single asset.
    F_alpha(zeta) = zeta + (1 / ((1-alpha)N)) * sum(max(L_i - zeta, 0))
    """
    N = len(losses)
    if N == 0:
        return np.nan, np.nan
        
    def objective(zeta):
        return zeta + (1.0 / ((1.0 - alpha) * N)) * np.sum(np.maximum(losses - zeta, 0))
    
    # We know zeta* = VaR_alpha, which is near the alpha-quantile
    guess_zeta = np.quantile(losses, alpha)
    
    # Find minimum
    res = opt.minimize_scalar(objective, bounds=(losses.min(), losses.max()), method='bounded')
    if res.success:
        cvar = res.fun
        var = res.x
        return var, cvar
    else:
        return np.nan, np.nan

def ru_portfolio_optimization(returns: pd.DataFrame, alpha: float = 0.99, w_max: float = 0.30, target_return: float = None):
    """
    Rockafellar-Uryasev portfolio optimization using Linear Programming.
    returns: DataFrame of asset returns (shape: N dates x M assets)
    """
    N, M = returns.shape
    r = returns.values
    
    # Decision variables: x = [w_1, ..., w_M, zeta, u_1, ..., u_N]^T
    # Size = M + 1 + N
    
    # Objective: min zeta + (1/((1-alpha)N)) * sum(u_i)
    c = np.zeros(M + 1 + N)
    c[M] = 1.0  # coefficient for zeta
    c[M+1:] = 1.0 / ((1.0 - alpha) * N)  # coefficients for u_i
    
    # Constraints:
    # 1. u_i >= -r_i * w - zeta  =>  -r_i * w - zeta - u_i <= 0
    # 2. sum w = 1               =>  A_eq * x = 1
    # 3. optional target return  =>  mu^T w >= target_return  => -mu^T w <= -target_return
    
    # Inequality constraints (A_ub * x <= b_ub)
    A_ub = np.zeros((N, M + 1 + N))
    A_ub[:, :M] = -r         # -r_i * w
    A_ub[:, M] = -1.0        # - zeta
    np.fill_diagonal(A_ub[:, M+1:], -1.0) # - u_i
    b_ub = np.zeros(N)
    
    if target_return is not None:
        mu = r.mean(axis=0)
        A_ret = np.zeros((1, M + 1 + N))
        A_ret[0, :M] = -mu
        A_ub = np.vstack([A_ub, A_ret])
        b_ub = np.append(b_ub, -target_return)
        
    # Equality constraints
    A_eq = np.zeros((1, M + 1 + N))
    A_eq[0, :M] = 1.0
    b_eq = np.array([1.0])
    
    # Bounds
    bounds = []
    for _ in range(M):
        bounds.append((0, w_max))  # 0 <= w_j <= w_max
    bounds.append((None, None))    # zeta is unbounded
    for _ in range(N):
        bounds.append((0, None))   # u_i >= 0
        
    res = opt.linprog(c, A_ub=A_ub, b_ub=b_ub, A_eq=A_eq, b_eq=b_eq, bounds=bounds, method="highs")
    
    if res.success:
        weights = res.x[:M]
        var = res.x[M]
        cvar = res.fun
        return weights, var, cvar
    else:
        return np.nan * np.ones(M), np.nan, np.nan
