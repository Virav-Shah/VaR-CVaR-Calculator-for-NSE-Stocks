import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from scipy.stats import norm, t, probplot
from statsmodels.graphics.tsaplots import plot_acf

def plot_histogram_with_overlays(returns: pd.Series, ticker: str):
    """
    Return histogram with Normal and t overlays.
    """
    plt.figure(figsize=(10, 6))
    sns.histplot(returns, bins=100, stat='density', alpha=0.6, label='Empirical')
    
    # Normal fit
    mu, sigma = np.mean(returns), np.std(returns, ddof=1)
    x = np.linspace(returns.min(), returns.max(), 1000)
    plt.plot(x, norm.pdf(x, mu, sigma), 'r-', lw=2, label='Normal Fit')
    
    # t fit
    nu, mu_t, s_t = t.fit(returns)
    plt.plot(x, t.pdf(x, nu, mu_t, s_t), 'g-', lw=2, label=f't Fit (df={nu:.2f})')
    
    plt.title(f'Return Distribution for {ticker}')
    plt.legend()
    return plt.gcf()

def plot_qq(returns: pd.Series, ticker: str):
    """
    QQ plots against Normal and t.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Normal QQ
    probplot(returns, dist="norm", plot=axes[0])
    axes[0].set_title(f"Normal Q-Q Plot ({ticker})")
    
    # t QQ
    nu, _, _ = t.fit(returns)
    probplot(returns, dist=t(nu), plot=axes[1])
    axes[1].set_title(f"Student-t Q-Q Plot ({ticker}, df={nu:.2f})")
    
    return fig

def plot_rolling_var(returns: pd.Series, var_dict: dict, alpha: float, ticker: str):
    """
    Rolling VaR against returns.
    var_dict: {"Historical": series, "Normal": series, ...}
    """
    plt.figure(figsize=(12, 6))
    # Plot losses instead of returns so VaR is on the same scale
    losses = -returns
    plt.plot(losses.index, losses, color='gray', alpha=0.5, label='Realized Loss')
    
    colors = ['r', 'b', 'g']
    for (name, var_series), color in zip(var_dict.items(), colors):
        plt.plot(var_series.index, var_series, color=color, label=f'{name} VaR')
        
        # Mark exceptions
        exceptions = losses[losses > var_series]
        if not exceptions.empty:
            plt.scatter(exceptions.index, exceptions, color=color, marker='x', s=10)
            
    plt.title(f'{alpha*100}% Rolling VaR for {ticker}')
    plt.legend()
    return plt.gcf()

def plot_ru_objective(losses: np.ndarray, alpha: float = 0.99):
    """
    RU objective F_alpha(zeta) vs zeta.
    """
    N = len(losses)
    def obj(zeta):
        return zeta + (1.0 / ((1.0 - alpha) * N)) * np.sum(np.maximum(losses - zeta, 0))
    
    zetas = np.linspace(np.quantile(losses, 0.90), np.quantile(losses, 0.999), 100)
    fs = [obj(z) for z in zetas]
    
    var = np.quantile(losses, alpha)
    cvar = obj(var)
    
    plt.figure(figsize=(8, 5))
    plt.plot(zetas, fs, 'b-', label=r'$F_\alpha(\zeta)$')
    plt.axvline(var, color='r', linestyle='--', label=f'VaR ({var:.4f})')
    plt.axhline(cvar, color='g', linestyle='--', label=f'CVaR ({cvar:.4f})')
    plt.title(f'Rockafellar-Uryasev Objective Function (alpha={alpha})')
    plt.xlabel(r'$\zeta$')
    plt.ylabel(r'$F_\alpha(\zeta)$')
    plt.legend()
    return plt.gcf()

def plot_acf_returns_vs_squared(returns: pd.Series, ticker: str):
    """
    ACF of returns vs squared returns to show volatility clustering.
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    plot_acf(returns.dropna(), ax=axes[0], title=f"ACF of Returns ({ticker})", lags=40)
    plot_acf(returns.dropna()**2, ax=axes[1], title=f"ACF of Squared Returns ({ticker})", lags=40)
    return fig
