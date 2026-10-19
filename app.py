"""Six-tab risk workbench. Model training is performed offline with run_ml.py."""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
from src.data_loader import RAW_DATA_PATH, MARKET_DATA_PATH
from src.returns import calculate_log_returns
from src.backtest import backtest_model
from src.plots import plot_histogram_with_overlays, plot_qq, plot_acf_returns_vs_squared

st.set_page_config(page_title='NSE ML Risk & Governance', layout='wide')
st.title('NSE ML Risk & Governance Workbench')
st.caption('One-session loss forecasts • chronological validation • model provenance')
if not Path(RAW_DATA_PATH).exists():
    st.info('Run python generate_results.py to download closing prices first.')
    st.stop()
prices = pd.read_csv(RAW_DATA_PATH, index_col=0, parse_dates=True)
returns = calculate_log_returns(prices)
asset = st.sidebar.selectbox('Asset', list(prices.columns) + ['Equal-Weight Portfolio'])
alpha = st.sidebar.selectbox('Confidence', [.95, .99], index=1)
is_portfolio = asset == 'Equal-Weight Portfolio'
series = returns.dropna().mean(axis=1) if is_portfolio else returns[asset].dropna()
folder = Path('results/ml') / asset.replace('.NS', '')
ready = (folder / 'manifest.json').exists()
v2_folder = Path('results/dashboard_v2') / asset.replace('.NS', '')
v2_ready = (Path('results/dashboard_v2/headline_summary.csv').exists()
            and (v2_folder / 'hybrid_calibrated.csv.gz').exists())
st.sidebar.caption(f'Price history: {prices.index.min().date()} – {prices.index.max().date()}')
if ready:
    manifest = json.loads((folder / 'manifest.json').read_text())
    st.sidebar.success(f'OOS: {manifest["oos_start"][:10]} – {manifest["oos_end"][:10]}')
else:
    st.sidebar.info('Pooled V2 forecast ready for this asset.' if v2_ready else 'ML artifacts not yet generated for this asset.')


def read(name):
    return pd.read_csv(folder / name, index_col=0, parse_dates=True)


def show(fig):
    st.pyplot(fig)
    plt.close(fig)


@st.cache_data
def baselines(s, a):
    return {m: backtest_model(s, 250, a, m) for m in ('historical', 'normal', 'student_t')}


def missing():
    st.info('Generate ML artifacts with: python run_ml.py --trials 10. Portfolio ML is not fitted; portfolio baselines remain available.')


tabs = st.tabs(['EDA & Regimes', 'VaR / CVaR', 'Regulatory Backtesting', 'ML Lifecycle', 'XAI & Governance', 'Portfolio Optimization'])
with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        show(plot_histogram_with_overlays(series, asset))
    with c2:
        st.dataframe(pd.DataFrame({'Value': {'Mean': series.mean(), 'Std': series.std(), 'Skew': series.skew(), 'Excess kurtosis': series.kurt(), 'Minimum': series.min()}}))
    show(plot_qq(series, asset))
    show(plot_acf_returns_vs_squared(series, asset))
    if ready:
        features = read('features.csv')
        fig, ax = plt.subplots(figsize=(11, 7))
        import seaborn as sns
        sns.heatmap(features.corr(), cmap='RdBu_r', center=0, ax=ax)
        show(fig)
        st.line_chart(features[['vol_5', 'vol_21', 'vol_63']])
        st.caption('VIX is standardized for the model; the overlay below shows observed VIX on a separate axis.')
        factors = pd.read_csv(MARKET_DATA_PATH, index_col=0, parse_dates=True)
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(features.index, features.vol_21, label='21-session volatility')
        ax.set_ylabel('Realized volatility')
        twin = ax.twinx()
        twin.plot(factors.index, factors.vix, color='orange', alpha=.6)
        twin.set_ylabel('India VIX')
        show(fig)
with tabs[1]:
    if v2_ready:
        from src.hybrid_dashboard import render_comparison
        render_comparison(asset, alpha)
    if ready:
        st.subheader('Original V1 results (full evaluation period)')
        comparison = pd.read_csv(folder / 'var_cvar_comparison.csv')
        selected = comparison[(comparison.Ticker == asset) & (comparison.Alpha == alpha)]
        st.dataframe(selected, hide_index=True)
        import altair as alt
        order = selected.sort_values('VaR').Method.tolist()
        intervals = alt.Chart(selected).mark_rule(strokeWidth=3, color='#8899aa').encode(y=alt.Y('Method:N',sort=order,title=None), x=alt.X('VaR:Q',axis=alt.Axis(format='.1%'),title='Mean daily loss threshold'), x2='CVaR:Q')
        endpoints = selected.melt(id_vars=['Method'],value_vars=['VaR','CVaR'],var_name='Risk measure',value_name='Threshold')
        points = alt.Chart(endpoints).mark_point(filled=True,size=100).encode(y=alt.Y('Method:N',sort=order,title=None),x='Threshold:Q',color='Risk measure:N',shape='Risk measure:N',tooltip=['Method:N','Risk measure:N',alt.Tooltip('Threshold:Q',format='.2%')])
        st.altair_chart((intervals + points).properties(height=260),width='stretch')
        st.caption('Dots distinguish the VaR boundary from average tail loss. The connecting interval shows tail severity; the two values are not additive. Wider limits alone do not establish better calibration.')
        st.caption('All rows are means over identical OOS dates. ML CVaR is a grid estimate with a flat tail above 99.5%, which can understate extreme tail losses.')
        model = read('lightgbm_predictions.csv')
        static = pd.Series(np.quantile(-series.loc[series.index < model.index.min()].tail(250), alpha), index=model.index)
        st.line_chart(pd.DataFrame({'Static pre-OOS historical VaR': static, 'Dynamic ML VaR': model[f'q_{alpha:g}'], 'Dynamic ML CVaR estimate': model[f'cvar_{alpha:g}']}))
    else:
        if not v2_ready:
            missing()
        result = baselines(series, alpha)
        st.dataframe(pd.DataFrame({m: {'Latest VaR': v[1].dropna().iloc[-1], 'Latest CVaR': v[2].dropna().iloc[-1]} for m, v in result.items()}).T)
with tabs[2]:
    if ready:
        scores = pd.read_csv(folder / 'backtest_results.csv')
        st.dataframe(scores[(scores.Ticker == asset) & (scores.Alpha == alpha)], hide_index=True)
        forecast = read('lightgbm_predictions.csv')
        historical = read(f'historical_{alpha:g}_predictions.csv')
        chart = pd.DataFrame({'Loss': forecast.Loss, 'Rolling historical VaR': historical.VaR, 'ML VaR': forecast[f'q_{alpha:g}']})
        import altair as alt
        long = chart.rename_axis('Date').reset_index().melt('Date', var_name='Series', value_name='Value')
        lines = alt.Chart(long).mark_line().encode(x='Date:T', y='Value:Q', color='Series:N', tooltip=['Date:T','Series:N','Value:Q'])
        breaches = chart[chart.Loss > chart['ML VaR']].reset_index()
        dots = alt.Chart(breaches).mark_point(color='red', size=35).encode(x='Date:T', y='Loss:Q', tooltip=['Date:T','Loss:Q'])
        st.altair_chart((lines + dots).interactive(), width='stretch')
    else:
        result = baselines(series, alpha)
        st.dataframe(pd.DataFrame({m: v[0] for m, v in result.items()}).T)
        st.line_chart(pd.DataFrame({'Loss': -series, **{m: v[1] for m, v in result.items()}}))
    st.caption('Basel colors apply only at 99% to the last 250 valid observations. These statistical diagnostics do not establish regulatory approval. Undefined independence tests are shown as missing, never as a pass.')
with tabs[3]:
    if not ready:
        missing()
    else:
        import altair as alt
        folds = pd.concat([pd.read_csv(folder / 'validation_folds.csv'), pd.read_csv(folder / 'lightgbm_folds.csv')], ignore_index=True)
        bars = []
        for i, row in folds.iterrows():
            for name, start, end in [('Train','TrainStart','TrainEnd'), ('Gap','TrainEnd','TestStart'), (row.Stage,'TestStart','TestEnd')]:
                bars.append({'Fold': f'{i:03} {row.Stage}', 'Segment': name, 'Start': row[start], 'End': row[end]})
        st.altair_chart(alt.Chart(pd.DataFrame(bars)).mark_bar().encode(x='Start:T', x2='End:T', y='Fold:N', color='Segment:N', tooltip=['Fold','Segment','Start','End']), width='stretch')
        kind = st.selectbox('Tuning model', ['lightgbm', 'linear'])
        trials = pd.read_csv(folder / f'{kind}_trials.csv')
        st.line_chart(trials.set_index('number')[['value']])
        st.dataframe(trials, hide_index=True)
        pred = read(f'{kind}_predictions.csv')
        calibration = pd.DataFrame({'Nominal coverage': [.95,.96,.97,.98,.99,.995], 'Empirical coverage': [(pred.Loss <= pred[f'q_{q:g}']).mean() for q in [.95,.96,.97,.98,.99,.995]]})
        calibration['Coverage gap (pp)'] = 100*(calibration['Empirical coverage']-calibration['Nominal coverage'])
        ideal = alt.Chart(pd.DataFrame({'x':[.9,1.], 'y':[.9,1.]})).mark_line(strokeDash=[5,5],color='#999999').encode(x='x:Q',y='y:Q')
        observed = alt.Chart(calibration).mark_line(point=True).encode(x=alt.X('Nominal coverage:Q',scale=alt.Scale(domain=[.9,1]),axis=alt.Axis(format='.0%')),y=alt.Y('Empirical coverage:Q',scale=alt.Scale(domain=[.9,1]),axis=alt.Axis(format='.0%')),tooltip=[alt.Tooltip('Nominal coverage:Q',format='.1%'),alt.Tooltip('Empirical coverage:Q',format='.2%'),alt.Tooltip('Coverage gap (pp):Q',format='+.2f')])
        st.altair_chart((ideal + observed).properties(height=300),width='stretch')
        st.caption('The dashed diagonal is perfect coverage. Below it means too many breaches; above it means fewer breaches than targeted. Hover for the gap in percentage points.')
        st.caption('Calibration is shown only for fitted quantiles. Hyperparameters and boosting rounds are selected before the OOS holdout; subsequent refits use only past data.')
        st.json(manifest)
with tabs[4]:
    if not (ready or v2_ready):
        missing()
    else:
        explanation_options = (['Pooled hybrid V2 (calibrated)'] if v2_ready else []) + (['LightGBM V1'] if ready else [])
        explanation_version = st.selectbox('Explanation model', explanation_options)
        if explanation_version.startswith('Pooled'):
            from src.hybrid_dashboard import render_explanation
            render_explanation(asset)
        else:
            import shap
            sv, meta, xf = read('shap_values.csv'), read('shap_metadata.csv'), read('shap_features.csv')
            st.caption('Tree SHAP explains the source model selected by quantile sorting on each date. The base value is that model’s expected output, not an unconditional historical VaR.')
            import altair as alt
            importance = sv.abs().mean().rename('Mean absolute contribution').rename_axis('Feature').reset_index()
            st.altair_chart(alt.Chart(importance).mark_bar().encode(y=alt.Y('Feature:N',sort='-x'),x=alt.X('Mean absolute contribution:Q',axis=alt.Axis(format='.2%'),title='Mean absolute contribution to daily VaR'),tooltip=['Feature:N',alt.Tooltip('Mean absolute contribution:Q',format='.3%')]).properties(height=420),width='stretch')
            sample = np.linspace(0, len(sv)-1, min(400, len(sv)), dtype=int)
            explanation = shap.Explanation(values=sv.iloc[sample].to_numpy(), base_values=meta.Base.iloc[sample].to_numpy(), data=xf.iloc[sample].to_numpy(), feature_names=list(sv.columns))
            shap.plots.beeswarm(explanation, show=False)
            show(plt.gcf())
            pred = read('lightgbm_predictions.csv')
            preferred = pred.index[pred.Loss > pred['q_0.99']].union(pred.nlargest(min(10,len(pred)), 'q_0.99').index)
            options = list(preferred) + [d for d in pred.index if d not in preferred]
            day = st.selectbox('Inspect a breach / high-risk date (99% VaR)', options, format_func=lambda d: d.strftime('%Y-%m-%d'))
            loc = sv.index.get_loc(day)
            st.write(f'Forecast model block: {meta.iloc[loc].Block:g}; source quantile: {meta.iloc[loc].SourceQuantile:g}')
            shap.plots.waterfall(shap.Explanation(values=sv.iloc[loc].to_numpy(), base_values=float(meta.iloc[loc].Base), data=xf.iloc[loc].to_numpy(), feature_names=list(sv.columns)), show=False)
            show(plt.gcf())
            st.subheader('Risk-driver contributions over time')
            top = sv.abs().mean().nlargest(6).index
            temporal = sv[top].copy()
            temporal['Other features'] = sv.drop(columns=top).sum(axis=1)
            temporal = temporal.rename_axis('Date').reset_index().melt('Date', var_name='Feature', value_name='Contribution')
            import altair as alt
            st.altair_chart(alt.Chart(temporal).mark_area().encode(x='Date:T', y=alt.Y('Contribution:Q', stack='zero'), color='Feature:N', tooltip=['Date:T', 'Feature:N', 'Contribution:Q']).interactive(), width='stretch')
            st.subheader('Latest 60-session feature drift')
            from src.features import drift_table
            recent = pred.tail(60)
            anchor_block = int(recent.Block.iloc[0])
            fold = pd.read_csv(folder / 'lightgbm_folds.csv').set_index('Block').loc[anchor_block]
            feature_history = read('features.csv')
            reference = feature_history.loc[fold.TrainStart:fold.TrainEnd]
            recent_features = feature_history.loc[pd.DatetimeIndex(pd.to_datetime(recent.Origin))]
            drift = drift_table(reference, recent_features).reset_index()
            def color(value):
                return 'background-color: ' + ('#c6efce' if value == 'Stable' else '#ffeb9c' if value == 'Moderate drift' else '#ffc7ce')
            st.dataframe(drift.style.map(color, subset=['Status']), hide_index=True)
            st.caption(f'PSI compares the latest {len(recent)} forecast sessions with the training distribution of block {anchor_block}, active at the start of this window. The reference excludes all monitored observations. Thresholds 0.1 / 0.25 are review heuristics.')
with tabs[5]:
    from src.portfolio_dashboard import render_portfolio
    render_portfolio(returns, alpha)
