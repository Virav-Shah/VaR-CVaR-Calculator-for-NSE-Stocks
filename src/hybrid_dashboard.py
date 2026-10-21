"""Read-only views of the frozen V1/V2 experiment."""
from pathlib import Path
import pandas as pd
import altair as alt
import streamlit as st

ROOT = Path('results/dashboard_v2')
FEATURE_LABELS = {
    'student_t_var_0.95_norm':'Student-t 95% limit',
    'student_t_var_0.99_norm':'Student-t 99% limit',
    'vix_z_63':'India VIX (relative)', 'vix_change_5':'India VIX 5-day change',
    'volume_z_21':'Trading volume', 'vol_ratio':'Short/long volatility',
    'vol_5':'5-day volatility', 'vol_21':'21-day volatility', 'vol_63':'63-day volatility',
    'parkinson_21':'Intraday price range', 'downside_21':'Recent downside risk',
    'skew_63':'Return asymmetry', 'kurt_63':'Extreme-return frequency',
    'abs_return':'Latest move size', 'down_return':'Latest negative return',
    'dispersion':'Differences across stocks', 'beta_21':'Market sensitivity',
    'correlation_21':'Market co-movement', 'fx_return':'Rupee-dollar move',
    'momentum_5':'Recent return trend', 'inverse_student_df':'Tail heaviness',
}


def feature_contributions(row):
    """Keep the five largest signed effects and aggregate all remaining effects."""
    top = row.abs().nlargest(5).index
    contributions = row.loc[top].copy()
    if len(row) > len(top):
        contributions.loc['other_features'] = row.drop(top).sum()
    return contributions


def render_prediction_breakdown(components, forecast):
    """Display the full additive identity in the same units as the bars."""
    st.markdown('**How the final risk limit is calculated**')
    breakdown = (100 * components).rename('Log-loss percentage points').to_frame()
    breakdown.loc['Final risk limit'] = 100 * forecast
    st.dataframe(breakdown.style.format('{:+.3f}'), width='stretch')
    st.caption(f'The entries above the final row add to {100 * components.sum():.3f} percentage points. The saved forecast is {forecast:.3%} daily log loss. These are changes to a risk limit, not shares of a total.')


def render_top_features(row):
    """Plot saved feature contributions without loading a training library."""
    top = row.abs().nlargest(5).index
    chart = feature_contributions(row).mul(100).rename('ContributionPP').rename_axis('Feature').reset_index()
    chart['Driver'] = chart.Feature.map(lambda f:FEATURE_LABELS.get(f, 'Other features (combined)' if f == 'other_features' else f.replace('stock_','Stock: ')))
    chart['Effect'] = chart.ContributionPP.map(lambda x:'Raises risk limit' if x>=0 else 'Lowers risk limit')
    st.altair_chart(alt.Chart(chart).mark_bar().encode(y=alt.Y('Driver:N',sort=chart.Driver.tolist(),title=None,axis=alt.Axis(labelLimit=220)),x=alt.X('ContributionPP:Q',axis=alt.Axis(format='+.2f'),title='Change in risk limit (log-loss percentage points)'),color=alt.Color('Effect:N',scale=alt.Scale(domain=['Raises risk limit','Lowers risk limit'],range=['#c62828','#1976d2'])),tooltip=['Driver:N','Feature:N',alt.Tooltip('ContributionPP:Q',title='Change (percentage points)',format='+.3f')]).properties(height=280).configure_legend(orient='bottom',columns=1),width='stretch')
    st.caption('SHAP values are signed changes to the forecast, not percentages that should total 100%. For example, +0.30 percentage points raises a 4.00% risk limit to 4.30%. The five largest feature effects are shown individually; all remaining features are combined in the last bar.')
    if any(feature.startswith('vix_') for feature in top):
        st.caption('India VIX (Volatility Index) reflects expected NIFTY 50 price swings over the next 30 calendar days, based on option prices and quoted as an annualised percentage. Higher readings mean larger expected swings, without predicting direction. The relative feature compares log VIX with its 63-session history; the change feature measures its five-session log change. [NSE definition](https://www.nseindia.com/static/products-services/indices-indiavix-index).')
    return top


def load_v1_explanation(asset, root=ROOT):
    folder = root/asset.replace('.NS','')
    pred = pd.read_csv(folder/'v1_forecasts.csv.gz',index_col=0,parse_dates=True)
    values = pd.read_csv(folder/'v1_shap_values.csv.gz',index_col=0,parse_dates=True)
    meta = pd.read_csv(folder/'v1_shap_metadata.csv.gz',index_col=0,parse_dates=True)
    if not (pred.index.equals(values.index) and pred.index.equals(meta.index)):
        raise ValueError('V1 forecasts and explanations must cover identical dates')
    return pred, values, meta


def render_v1_explanation(asset):
    pred, values, meta = load_v1_explanation(asset)
    preferred = list(pred.nlargest(10,'Loss').index)
    preferred += [d for d in pred.index[pred.Loss > pred['q_0.99']] if d not in preferred]
    options = preferred + [d for d in pred.index if d not in preferred]
    day = st.selectbox('Inspect a crash / breach date (99% VaR)',options,format_func=lambda d:d.strftime('%Y-%m-%d'),key='v1_crash_date')
    row, m = values.loc[day], meta.loc[day]
    top = render_top_features(row)
    st.caption('Top five contributions to V1’s 99% risk limit. Positive values raise the forecast; negative values lower it. These describe model behavior, not the causes of a market crash.')
    c1,c2 = st.columns(2)
    c1.metric('V1 risk limit',f'{pred.loc[day,"q_0.99"]:.2%}')
    c2.metric('Actual loss',f'{pred.loc[day,"Loss"]:.2%}')
    components = pd.Series({'Model starting value':m.Base,'Top five feature contributions':row.loc[top].sum(),'Other feature contributions':row.drop(top).sum()})
    render_prediction_breakdown(components, m.Prediction)
    st.caption(f'Model block {int(m.Block)}. Quantile sorting selected the {m.SourceQuantile:.1%} source model on this date.')


def render_comparison(asset, alpha):
    if not (ROOT/'headline_summary.csv').exists():
        return
    st.subheader('Model performance comparison')
    summary = pd.read_csv(ROOT/'headline_summary.csv')
    summary = summary[summary.Alpha == alpha].copy()
    for column in ['CoveragePasses','IndependencePasses','BothPasses']:
        summary[column] = summary[column].astype(str) + '/' + summary.Assets.astype(str)
    st.dataframe(summary[['Version','CoveragePasses','IndependencePasses','BothPasses','MeanBreachRate','MeanPinball','MeanVaR','Governance']].rename(columns={
        'CoveragePasses':'Coverage passes','IndependencePasses':'Independence passes','BothPasses':'Both pass',
        'MeanBreachRate':'Breaches (%)','MeanPinball':'Forecast loss ↓','MeanVaR':'Mean VaR','Governance':'Model assessment'}), hide_index=True)
    st.caption(f'Expected breaches: {100*(1-alpha):g}%. Lower forecast loss is better. Pass means p > 0.05, not proof of safety. Every version uses the same dates after a 250-session calibration warm-up.')
    st.info('Retrospective research comparison: V1 results influenced the V2 design. Independent future testing and separate expected-shortfall validation are still needed.')
    scores = pd.read_csv(ROOT/'version_comparison.csv')
    selected = scores[(scores.Ticker == asset) & (scores.Alpha == alpha)]
    with st.expander('This stock: detailed test results'):
        st.dataframe(selected, hide_index=True)
    folder = ROOT/asset.replace('.NS','')
    if not (folder/'hybrid_calibrated.csv.gz').exists():
        st.caption('Forecast history is unavailable for this asset.')
        return
    pred = pd.read_csv(folder/'hybrid_calibrated.csv.gz',index_col=0,parse_dates=True).dropna(subset=[f'q_{alpha:g}'])
    window = st.selectbox('Forecast comparison window',[250,750,len(pred)],format_func=lambda n:f'Last {n:,} sessions',key='hybrid_window')
    dates = pred.tail(window).index
    v1_path = Path('results/ml')/asset.replace('.NS','')/'lightgbm_predictions.csv'
    baseline_path = Path('results/ml')/asset.replace('.NS','')/f'student_t_{alpha:g}_predictions.csv'
    if v1_path.exists() and baseline_path.exists():
        v1 = pd.read_csv(v1_path,index_col=0,parse_dates=True)
        baseline = pd.read_csv(baseline_path,index_col=0,parse_dates=True)
        v1_var, baseline_var = v1.loc[dates,f'q_{alpha:g}'], baseline.loc[dates,'VaR']
    else:
        v1 = pd.read_csv(folder/'v1_forecasts.csv.gz',index_col=0,parse_dates=True)
        baseline = pd.read_csv(folder/'student_t_forecasts.csv.gz',index_col=0,parse_dates=True)
        v1_var, baseline_var = v1.loc[dates,f'q_{alpha:g}'], baseline.loc[dates,f'VaR_{alpha:g}']
    lines = pd.DataFrame({'Student-t':baseline_var,'V1':v1_var,'V2 pooled':pred.loc[dates,f'raw_q_{alpha:g}'],'V2 calibrated':pred.loc[dates,f'q_{alpha:g}']}).rename_axis('Date').reset_index().melt('Date',var_name='Model',value_name='VaR')
    actual = pred.loc[dates,['Loss',f'q_{alpha:g}']].rename_axis('Date').reset_index()
    actual['Breach'] = actual.Loss > actual[f'q_{alpha:g}']
    line = alt.Chart(lines).mark_line().encode(x='Date:T',y=alt.Y('VaR:Q',axis=alt.Axis(format='.0%'),title='Daily loss / risk limit'),color='Model:N',tooltip=['Date:T','Model:N',alt.Tooltip('VaR:Q',format='.2%')])
    dots = alt.Chart(actual).mark_point(size=18,opacity=.55).encode(x='Date:T',y='Loss:Q',color=alt.Color('Breach:N',scale=alt.Scale(domain=[False,True],range=['#999999','#c62828']),legend=alt.Legend(title='V2 breach')),tooltip=['Date:T',alt.Tooltip('Loss:Q',format='.2%'),'Breach:N'])
    st.altair_chart((line+dots).resolve_scale(color='independent').properties(height=340).configure_legend(orient='bottom',columns=2).interactive(),width='stretch')
    st.caption('Red dots identify losses that exceeded calibrated V2. A higher risk limit is useful only if it improves calibration without becoming unnecessarily conservative.')
    with st.expander('How much did calibration change the risk limit?'):
        st.line_chart(pred.loc[dates,[f'raw_q_{alpha:g}',f'q_{alpha:g}']].rename(columns={f'raw_q_{alpha:g}':'Raw V2',f'q_{alpha:g}':'Calibrated V2'}))
        st.caption('The signed adjustment uses only the previous 250 realized forecast errors, normalized by volatility. It can raise or lower the limit.')


def render_explanation(asset):
    folder = ROOT/asset.replace('.NS','')
    pred = pd.read_csv(folder/'hybrid_calibrated.csv.gz',index_col=0,parse_dates=True).dropna(subset=['q_0.99'])
    values = pd.read_csv(folder/'shap_values.csv.gz',index_col=0,parse_dates=True)
    meta = pd.read_csv(folder/'shap_metadata.csv.gz',index_col=0,parse_dates=True)
    preferred = list(pred.nlargest(10,'Loss').index)
    preferred += [d for d in pred.index[pred.Loss > pred['q_0.99']] if d not in preferred]
    options = preferred + [d for d in pred.index if d not in preferred]
    day = st.selectbox('Inspect a crash / breach date (99% VaR)',options,format_func=lambda d:d.strftime('%Y-%m-%d'),key='v2_crash_date')
    row = values.loc[day]
    top = render_top_features(row)
    st.caption('Top five contributions to the 99% residual model. These explain the model output, not the causes of a market crash. Calibration and quantile ordering are shown separately below.')
    m = meta.loc[day]
    components = pd.Series({'Student-t starting limit':m.StudentBase,'Average AI adjustment':m.TreeBase,'Top five feature contributions':row.loc[top].sum(),'Other feature contributions':row.drop(top).sum(),'Raw quantile ordering':m.RawOrderAdjustment,'Calibration buffer':m.CalibrationBuffer,'Final quantile ordering':m.FinalOrderAdjustment})
    c1,c2,c3 = st.columns(3)
    c1.metric('Final V2 risk limit',f'{m.FinalVaR99:.2%}')
    c2.metric('Actual loss',f'{pred.loc[day,"Loss"]:.2%}')
    c3.metric('Calibration change (pp)',f'{100 * m.CalibrationBuffer:+.2f}')
    render_prediction_breakdown(components, m.FinalVaR99)
    st.caption(f'Model block {int(m.Block)}. Features were available at the preceding forecast origin.')
    st.caption('CVaR uses a Student-t tail fitted through the 95% and 99% risk limits. This is an explicit tail assumption, not an independently validated AI estimate of extreme losses.')
