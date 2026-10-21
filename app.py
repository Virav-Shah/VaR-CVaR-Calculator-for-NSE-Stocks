"""Risk analytics dashboard using forecasts and explanations trained offline."""
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

st.set_page_config(page_title='NSE Market Risk Analytics', layout='wide')
st.title('NSE Market Risk Analytics')
st.caption('Daily risk forecasts • explainable machine learning • portfolio analysis')
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


tabs = st.tabs(['EDA & Regimes', 'VaR / CVaR', 'Regulatory Backtesting', 'Explainable AI', 'Portfolio Optimization'])
with tabs[0]:
    c1, c2 = st.columns(2)
    with c1:
        show(plot_histogram_with_overlays(series, asset))
    with c2:
        st.dataframe(pd.DataFrame({'Value': {'Mean': series.mean(), 'Std': series.std(), 'Skew': series.skew(), 'Excess kurtosis': series.kurt(), 'Minimum': series.min()}}))
    show(plot_qq(series, asset))
    st.caption('Q–Q plots compare observed returns with each distribution. Points close to the red fitted line indicate a closer fit; departures in the tails reveal extreme moves that the distribution does not capture. Horizontal quantile scales differ between the two distributions.')
    show(plot_acf_returns_vs_squared(series, asset))
    if ready:
        features = read('features.csv')
        fig, ax = plt.subplots(figsize=(11, 7))
        import seaborn as sns
        sns.heatmap(features.corr(), cmap='RdBu_r', center=0, ax=ax)
        show(fig)
        st.line_chart(features[['vol_5', 'vol_21', 'vol_63']])
        st.caption('India VIX (Volatility Index) reflects expected NIFTY 50 volatility over the next 30 calendar days, based on option prices. Higher readings mean larger expected price swings, without predicting direction. The chart shows the observed index as an annualised percentage; the model uses its relative level and five-session change. [NSE definition](https://www.nseindia.com/static/products-services/indices-indiavix-index).')
        factors = pd.read_csv(MARKET_DATA_PATH, index_col=0, parse_dates=True)
        fig, ax = plt.subplots(figsize=(12, 3))
        ax.plot(features.index, features.vol_21, label='21-session volatility')
        ax.set_ylabel('Realized volatility')
        twin = ax.twinx()
        twin.plot(factors.index, factors.vix, color='orange', alpha=.6)
        twin.set_ylabel('India VIX (annualised %)')
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
    from src.hybrid_dashboard import render_explanation, render_v1_explanation
    v1_explanation_ready = all((v2_folder/name).exists() for name in
                               ('v1_shap_values.csv.gz', 'v1_shap_metadata.csv.gz', 'v1_forecasts.csv.gz'))
    explanation_options = (['Pooled hybrid V2 (calibrated)'] if v2_ready else []) + (['LightGBM V1'] if v1_explanation_ready else [])
    if not explanation_options:
        st.info('Stock-level explanations are available for the ten individual stocks. Select a stock in the sidebar.')
    else:
        explanation_version = st.selectbox('Explanation model', explanation_options, key='explanation_model')
        if explanation_version.startswith('Pooled'):
            render_explanation(asset)
        else:
            render_v1_explanation(asset)
with tabs[4]:
    from src.portfolio_dashboard import render_portfolio
    render_portfolio(returns, alpha)
