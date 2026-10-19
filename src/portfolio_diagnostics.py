"""Comparable realized portfolio diagnostics, all on the same OOS dates."""
import numpy as np
import pandas as pd


def empirical_es(losses, alpha):
    """Empirical quantile integral, including fractional boundary mass."""
    losses = np.asarray(losses, dtype=float)
    if not len(losses) or not np.isfinite(losses).all() or not 0 < alpha < 1:
        raise ValueError('Finite nonempty losses and confidence in (0,1) required')
    var = np.quantile(losses, alpha, method='inverted_cdf')
    return float(var + np.maximum(losses - var, 0).mean() / (1 - alpha))


def portfolio_diagnostics(log_returns, weights, alpha=.99):
    if log_returns.empty or not log_returns.index.is_monotonic_increasing or not log_returns.index.is_unique:
        raise ValueError('Nonempty chronological returns required')
    if set(weights.index) != set(log_returns.columns):
        raise ValueError('Weights must cover precisely the return universe')
    weights = weights.reindex(log_returns.columns)
    if not np.isfinite(weights.to_numpy()).all() or not np.isfinite(log_returns.to_numpy()).all():
        raise ValueError('Finite returns and weights required')
    if (weights < 0).any().any() or not np.allclose(weights.sum(), 1):
        raise ValueError('Long-only weights must sum to one')
    daily = np.expm1(log_returns).dot(weights)
    wealth = (1 + daily).cumprod()
    # Include initial wealth=1: a loss on day one must appear in drawdown.
    drawdown = wealth / wealth.cummax().clip(lower=1) - 1
    metrics = pd.DataFrame(index=weights.columns)
    metrics['Annualized return'] = wealth.iloc[-1] ** (252 / len(daily)) - 1
    metrics['Annualized volatility'] = daily.std(ddof=1) * np.sqrt(252)
    metrics['Daily tail loss (ES)'] = [empirical_es(-daily[c], alpha) for c in daily]
    metrics['Maximum drawdown'] = drawdown.min()
    metrics['Worst day'] = daily.min()
    metrics['Effective holdings'] = 1 / weights.pow(2).sum()
    metrics.index.name = 'Portfolio'
    return daily, wealth, drawdown, metrics


def common_stress_days(daily, benchmark='Equal Weight', count=10):
    """Same dates for every strategy, chosen by benchmark losses only."""
    worst = daily[benchmark].nsmallest(count).index
    return daily.loc[worst].rename_axis('Date')
