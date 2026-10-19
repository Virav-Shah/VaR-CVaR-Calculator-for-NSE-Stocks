"""Causal close-of-day features. Targets remain untrimmed next-session losses."""
import numpy as np
import pandas as pd


def build_features(prices, factors, ticker):
    if not prices.index.is_monotonic_increasing or not prices.index.is_unique:
        raise ValueError('Prices must have a unique chronological index')
    factors = factors.reindex(prices.index)
    r_all = np.log(prices / prices.shift(1))
    r = r_all[ticker]
    f = pd.DataFrame(index=prices.index)
    for n in (5, 21, 63):
        f[f'vol_{n}'] = r.rolling(n).std()
    f['vol_ratio'] = f.vol_5 / f.vol_63.replace(0, np.nan)
    high, low = (factors[f'{ticker}_{x}'].where(lambda x: x > 0) for x in ('high', 'low'))
    f['parkinson_21'] = np.sqrt((np.log(high / low)**2).rolling(21).mean() / (4 * np.log(2)))
    f['downside_21'] = r.clip(upper=0).pow(2).rolling(21).mean()
    f['skew_63'] = r.rolling(63).skew()
    f['kurt_63'] = r.rolling(63).kurt()
    f['abs_return'] = r.abs()
    f['down_return'] = r.clip(upper=0)
    f['dispersion'] = r_all.std(axis=1, skipna=False)
    market = np.log(factors.nifty / factors.nifty.shift(1))
    f['beta_21'] = r.rolling(21).cov(market) / market.rolling(21).var().replace(0, np.nan)
    f['correlation_21'] = r.rolling(21).corr(market)
    vix = np.log(factors.vix.where(factors.vix > 0))
    f['vix_z_63'] = (vix - vix.rolling(63).mean()) / vix.rolling(63).std().replace(0, np.nan)
    f['vix_change_5'] = vix.diff(5)
    volume = factors[f'{ticker}_volume']
    f['volume_z_21'] = (volume - volume.rolling(21).mean()) / volume.rolling(21).std().replace(0, np.nan)
    f['fx_return'] = np.log(factors.fx / factors.fx.shift(1))
    f['momentum_5'] = r.rolling(5).sum()
    return f.replace([np.inf, -np.inf], np.nan)


def supervised_data(prices, factors, ticker):
    """Index is forecast origin t; target_date is the actual next equity session."""
    X = build_features(prices, factors, ticker)
    y = -np.log(prices[ticker] / prices[ticker].shift(1)).shift(-1)
    dates = pd.Series(prices.index, index=prices.index).shift(-1)
    valid = X.notna().all(axis=1) & y.notna() & dates.notna()
    return X.loc[valid], y.loc[valid], dates.loc[valid]


def population_stability_index(reference, recent, bins=10):
    """Training-only quantile edges, with infinite tails and smoothed counts."""
    a, b = np.asarray(reference, float), np.asarray(recent, float)
    a, b = a[np.isfinite(a)], b[np.isfinite(b)]
    if len(a) == 0 or len(b) == 0:
        return np.nan
    internal = np.unique(np.quantile(a, np.linspace(0, 1, bins + 1)))[1:-1]
    if len(internal) == 0:
        # Preserve sensitivity when a formerly constant feature changes.
        eps = max(abs(a[0]), 1) * 1e-6
        internal = np.array([a[0] - eps, a[0] + eps])
    edges = np.r_[-np.inf, internal, np.inf]
    p = np.histogram(a, edges)[0] + .5
    q = np.histogram(b, edges)[0] + .5
    p, q = p / p.sum(), q / q.sum()
    return float(np.sum((q - p) * np.log(q / p)))


def drift_table(train, recent):
    values = {c: population_stability_index(train[c], recent[c]) for c in train}
    table = pd.DataFrame({'PSI': values})
    table['Status'] = table.PSI.map(lambda x: 'Stable' if x < .1 else 'Moderate drift' if x < .25 else 'Review / retrain')
    return table.rename_axis('Feature')
