"""Pooled residual quantile models and strictly past-only calibration."""
import warnings
import numpy as np
import pandas as pd
from scipy.stats import t
from src.var_models import student_t_var_cvar_fitted

LEVELS = (.95, .99)


def student_context(returns, window=250, refit_every=5):
    """At origin t, use returns through t to forecast t+1.

    Match the existing Student-t benchmark's fitting schedule and df floor so
    the hybrid's starting point is directly comparable to the V1 benchmark.
    """
    returns = returns.dropna().sort_index()
    volatility = returns.rolling(20).std()
    records, params = [], None
    for i in range(window, len(returns)):
        if params is None or i % refit_every == 0:
            with warnings.catch_warnings():
                warnings.simplefilter('ignore')
                nu, mu, scale = t.fit(returns.iloc[i-window:i].to_numpy())
            params = (max(nu, 2.1) if nu <= 2 else nu, mu, scale)
        row = dict(Origin=returns.index[i-1], TargetDate=returns.index[i],
                   Scale=volatility.iloc[i-1], df=params[0], LossLocation=-params[1],
                   StudentScale=params[2])
        for a in LEVELS:
            row[f't_var_{a:g}'], row[f't_es_{a:g}'] = student_t_var_cvar_fitted(*params, a)
        records.append(row)
    return pd.DataFrame(records).set_index('Origin')


def rolling_error_calibration(loss, raw_var, scale, alpha, window=250):
    """Rolling order statistic of PREVIOUS normalized forecast errors.

    Not an IID conformal guarantee for dependent, changing financial data.
    First window observations warm up the calibrator and are not scored.
    Calibration may raise or lower a forecast; no clipping of target losses.
    """
    if not loss.index.equals(raw_var.index) or not loss.index.equals(scale.index):
        raise ValueError('Calibration inputs must have identical indexes')
    if not loss.index.is_unique or not loss.index.is_monotonic_increasing:
        raise ValueError('Calibration requires unique chronological dates')
    if window < 2 or not 0 < alpha < 1 or (scale <= 0).any():
        raise ValueError('Positive scales, valid confidence and window required')
    if not np.isfinite(np.column_stack([loss, raw_var, scale])).all():
        raise ValueError('Calibration inputs must be finite')
    rank = int(np.ceil((window + 1) * alpha))
    if rank > window:
        raise ValueError('Window too small for finite conformal order statistic')
    errors = (loss - raw_var) / scale
    # shift(1) is essential: today's realized loss cannot affect today's buffer.
    buffer_z = errors.shift(1).rolling(window, min_periods=window).apply(
        lambda x: np.partition(x, rank-1)[rank-1], raw=True)
    return raw_var + scale * buffer_z, scale * buffer_z


def student_tail_es(q95, q99, nu):
    """One location-scale Student-t tail matching BOTH calibrated quantiles.

    Nu is estimated from past returns. Location and scale are inferred from
    q95 and q99. This extrapolates beyond .995 without a flat cap, but remains
    an explicit parametric tail assumption, not a distribution-free ES claim.
    Equal quantiles imply a degenerate zero-scale limiting distribution.
    """
    q95, q99, nu = np.broadcast_arrays(np.asarray(q95, float), np.asarray(q99, float), np.asarray(nu, float))
    if not np.isfinite(np.stack([q95,q99,nu])).all() or (nu <= 1).any() or (q99 < q95).any():
        raise ValueError('Ordered finite quantiles and finite df > 1 required')
    z95, z99 = t.ppf(.95, nu), t.ppf(.99, nu)
    scale = (q99-q95)/(z99-z95)
    location = q95-scale*z95
    es = {}
    for a, z in zip(LEVELS, (z95,z99)):
        standard_es = t.pdf(z, nu)*(nu+z*z)/((1-a)*(nu-1))
        es[a] = location + scale*standard_es
    return es[.95], es[.99], location, scale


def calibrate_frame(frame, window=250):
    """Preserve raw forecasts and expose calibration/reordering separately."""
    out = frame.copy()
    for a in LEVELS:
        q, buffer = rolling_error_calibration(out.Loss, out[f'q_{a:g}'], out.Scale, a, window)
        out[f'raw_q_{a:g}'] = out[f'q_{a:g}']
        out[f'buffer_{a:g}'] = buffer
        out[f'q_{a:g}'] = q
    valid = out[[f'q_{a:g}' for a in LEVELS]].notna().all(axis=1)
    ordered = np.sort(out.loc[valid, ['q_0.95','q_0.99']].to_numpy(), axis=1)
    for j,a in enumerate(LEVELS):
        out.loc[valid,f'order_adjustment_{a:g}'] = ordered[:,j]-out.loc[valid,f'q_{a:g}']
        out.loc[valid,f'q_{a:g}'] = ordered[:,j]
    es95,es99,loc,scale = student_tail_es(ordered[:,0],ordered[:,1],out.loc[valid,'df'])
    out.loc[valid,'cvar_0.95'],out.loc[valid,'cvar_0.99'] = es95,es99
    out.loc[valid,'TailLocation'],out.loc[valid,'TailScale'] = loc,scale
    return out


def build_panel(prices, v1_root, contexts):
    """Combine stocks by date; all normalization uses origin-date volatility."""
    frames, feature_columns = [], None
    for ticker in prices.columns:
        history = pd.read_csv(v1_root / ticker.replace('.NS','') / 'features.csv',index_col=0,parse_dates=True)
        context = contexts[ticker].reindex(history.index)
        if context.isna().any().any() or (context.Scale <= 0).any():
            raise ValueError(f'Missing/nonpositive context for {ticker}')
        X = history.copy()
        for name in ('vol_5','vol_21','vol_63','parkinson_21','abs_return','down_return','dispersion','momentum_5'):
            X[name] = X[name]/context.Scale
        X['downside_21'] = X.downside_21/context.Scale.pow(2)
        # Put the statistical anchor first and keep target metadata out of X.
        anchors = pd.DataFrame({f'student_t_var_{a:g}_norm':context[f't_var_{a:g}']/context.Scale for a in LEVELS},index=X.index)
        X = pd.concat([anchors,X],axis=1)
        X['inverse_student_df'] = 1/context['df']
        for name in prices.columns:
            X['stock_'+name.replace('.NS','')] = float(name == ticker)
        feature_columns = list(X.columns)
        X['Scale'] = context.Scale
        X['df'] = context['df']
        X['TargetDate'] = context.TargetDate
        X['Loss'] = -np.log(prices[ticker]/prices[ticker].shift(1)).shift(-1).reindex(X.index)
        for a in LEVELS:
            X[f't_var_{a:g}'] = context[f't_var_{a:g}']
        X['Ticker'] = ticker
        frames.append(X.rename_axis('Origin').reset_index().set_index(['Origin','Ticker']))
    panel = pd.concat(frames).sort_index()
    counts = panel.groupby(level='Origin').size()
    dates = counts.index[counts == len(prices.columns)]
    panel = panel.loc[dates]
    if not np.isfinite(panel[feature_columns+['Scale','Loss']].to_numpy()).all():
        raise ValueError('Nonfinite panel data')
    return panel, feature_columns, dates


def panel_slice(panel, dates):
    """Whole dates are assigned together; never split individual stock rows."""
    return panel.loc[panel.index.get_level_values('Origin').isin(dates)]


class PooledHybrid:
    def __init__(self, params=None, seed=42):
        self.params, self.seed = dict(params or {}), seed

    def fit(self, panel, features, validation=None):
        from lightgbm import LGBMRegressor, early_stopping
        self.features, self.models = list(features), {}
        params = dict(n_estimators=150, learning_rate=.03, max_depth=3,num_leaves=7,
                      min_child_samples=300, subsample=.8,subsample_freq=1,
                      colsample_bytree=.9,verbosity=-1,n_jobs=2,device_type='cpu')
        params.update(self.params)
        for a in LEVELS:
            # Conditional quantile of a shifted target = conditional loss quantile minus anchor.
            target = (panel.Loss-panel[f't_var_{a:g}'])/panel.Scale
            model = LGBMRegressor(objective='quantile',alpha=a,random_state=self.seed,**params)
            kwargs = {}
            if validation is not None:
                vy = (validation.Loss-validation[f't_var_{a:g}'])/validation.Scale
                kwargs = dict(eval_set=[(validation[self.features],vy)],callbacks=[early_stopping(20,verbose=False)])
            model.fit(panel[self.features],target,**kwargs)
            self.models[a] = model
        return self

    def predict_native(self, panel):
        return np.column_stack([panel[f't_var_{a:g}'].to_numpy()+panel.Scale.to_numpy()*self.models[a].predict(panel[self.features]) for a in LEVELS])

    def predict(self, panel):
        return np.sort(self.predict_native(panel),axis=1)

    def explain(self, panel, alpha=.99):
        import shap
        explanation = shap.TreeExplainer(self.models[alpha])(panel[self.features])
        values = explanation.values*panel.Scale.to_numpy()[:,None]
        bases = np.asarray(explanation.base_values).reshape(-1)*panel.Scale.to_numpy()
        native = panel[f't_var_{alpha:g}'].to_numpy()+bases+values.sum(axis=1)
        expected = self.predict_native(panel)[:,LEVELS.index(alpha)]
        np.testing.assert_allclose(native,expected,atol=1e-7)
        return values,bases
