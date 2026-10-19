"""Decision-focused portfolio comparison charts for the Streamlit workbench."""
import altair as alt
import numpy as np
import pandas as pd
import streamlit as st
from src.portfolio import min_variance_portfolio
from src.ru_optimization import ru_portfolio_optimization
from src.portfolio_diagnostics import portfolio_diagnostics, common_stress_days

NAMES = ['Equal Weight', 'Min Variance', 'Min CVaR']
COLORS = ['#9aa4b2', '#4c9aff', '#f5a623']


@st.cache_data
def fitted_weights(train, alpha):
    optimized, _, _ = ru_portfolio_optimization(train, alpha=alpha)
    return pd.DataFrame({'Equal Weight': np.full(train.shape[1], 1/train.shape[1]),
                         'Min Variance': min_variance_portfolio(train), 'Min CVaR': optimized}, index=train.columns)


def render_portfolio(returns, alpha):
    clean = returns.dropna()
    train = clean.loc[clean.index <= '2015-12-31']
    test = clean.loc[clean.index > '2015-12-31']
    if train.empty or test.empty:
        st.info('Both training history through 2015 and subsequent OOS observations are required.')
        return
    weights = fitted_weights(train, alpha)
    if not np.isfinite(weights.to_numpy()).all():
        st.error('Portfolio optimization failed; no comparison is displayed.')
        return
    daily, wealth, drawdown, metrics = portfolio_diagnostics(test, weights, alpha)
    color = alt.Color('Portfolio:N', scale=alt.Scale(domain=NAMES, range=COLORS), legend=alt.Legend(orient='bottom', title=None, columns=1))
    st.subheader('Does lower tail risk justify the return trade-off?')
    st.caption(f'All strategies use the same {len(test):,} unseen sessions: {test.index[0]:%d %b %Y}–{test.index[-1]:%d %b %Y}. Weights were fitted through 31 Dec 2015 and are held at daily rebalancing targets. This comparison covers the full stock universe, independent of the asset selector.')
    baseline = metrics.loc['Equal Weight']
    cols = st.columns(3)
    for col, name in zip(cols, NAMES):
        es = metrics.loc[name, 'Daily tail loss (ES)']
        delta = None if name == 'Equal Weight' else f"{(es / baseline['Daily tail loss (ES)'] - 1)*100:+.1f}% vs equal weight"
        col.metric(f'{name} · {alpha:.0%} daily ES', f'{es:.2%}', delta, delta_color='inverse')
    st.caption(f'ES is the average loss in the worst {1-alpha:.0%} of daily outcomes, estimated with fractional boundary weighting. Lower is better; it is not annualized.')
    data = metrics.reset_index()
    scatter = alt.Chart(data).mark_circle(size=230).encode(
        x=alt.X('Daily tail loss (ES):Q', axis=alt.Axis(format='.1%'), title=f'{alpha:.0%} daily tail loss (lower is better)', scale=alt.Scale(zero=False)),
        y=alt.Y('Annualized return:Q', axis=alt.Axis(format='.1%'), title='Annualized return (higher is better)', scale=alt.Scale(zero=False)),
        color=color, tooltip=['Portfolio:N', alt.Tooltip('Daily tail loss (ES):Q', format='.2%'), alt.Tooltip('Annualized return:Q', format='.2%'), alt.Tooltip('Maximum drawdown:Q', format='.2%')])
    labels = scatter.mark_text(align='left', dx=12, dy=-12, fontSize=12).encode(text='Portfolio:N')
    st.altair_chart((scatter + labels).properties(height=320).interactive(), width='stretch')
    cv = metrics.loc['Min CVaR']
    st.write(f"Against equal weight, Min CVaR changed realized daily ES by {(cv['Daily tail loss (ES)']-baseline['Daily tail loss (ES)'])*100:+.2f} percentage points and annualized return by {(cv['Annualized return']-baseline['Annualized return'])*100:+.2f} percentage points. Optimization on training data does not guarantee an OOS improvement.")
    st.subheader('How deep were the losses, and how long did recovery take?')
    underwater = drawdown.rename_axis('Date').reset_index().melt('Date', var_name='Portfolio', value_name='Drawdown')
    st.altair_chart(alt.Chart(underwater).mark_line().encode(x='Date:T', y=alt.Y('Drawdown:Q', axis=alt.Axis(format='.0%'), title='Decline from prior portfolio peak'), color=color, tooltip=['Date:T','Portfolio:N',alt.Tooltip('Drawdown:Q',format='.2%')]).properties(height=300).interactive(), width='stretch')
    st.caption('Zero means the portfolio has recovered its previous high. Deep troughs show capital loss; long stretches below zero show time spent waiting for recovery. Initial capital is included in the peak calculation.')
    st.subheader('Did the portfolios protect capital on the same stress days?')
    stress = common_stress_days(daily)
    order = [d.strftime('%Y-%m-%d') for d in stress.index]
    cells = stress.reset_index().melt('Date', var_name='Portfolio', value_name='Daily return')
    cells['Date'] = cells.Date.dt.strftime('%Y-%m-%d')
    scale = max(abs(cells['Daily return'].min()), abs(cells['Daily return'].max()))
    heat = alt.Chart(cells).mark_rect().encode(x=alt.X('Portfolio:N', sort=NAMES, title=None, axis=alt.Axis(labelAngle=0)), y=alt.Y('Date:N', sort=order, title=None), color=alt.Color('Daily return:Q', scale=alt.Scale(domain=[-scale,0,scale], range=['#b91c1c','#f8fafc','#15803d']), legend=alt.Legend(format='.1%')), tooltip=['Date:N','Portfolio:N',alt.Tooltip('Daily return:Q',format='.2%')])
    text = heat.mark_text().encode(text=alt.Text('Daily return:Q',format='.2%'), color=alt.condition('abs(datum["Daily return"]) > '+str(scale*.55),alt.value('white'),alt.value('#111827')))
    st.altair_chart((heat + text).properties(height=330), width='stretch')
    st.caption('The ten worst equal-weight days define one common stress set, ordered from worst to less severe. Each cell is that strategy’s return on that exact date; no strategy gets to choose its own comparison days.')
    st.subheader('Which holdings changed relative to equal weight?')
    active = weights[['Min Variance','Min CVaR']].subtract(weights['Equal Weight'], axis=0)
    active.index = active.index.str.replace('.NS','',regex=False)
    active = active.rename_axis('Stock').reset_index().melt('Stock',var_name='Portfolio',value_name='Active weight')
    bars = alt.Chart(active).mark_bar().encode(y=alt.Y('Stock:N', sort=weights.index.str.replace('.NS','',regex=False).tolist()), x=alt.X('Active weight:Q', axis=alt.Axis(format='+.0%'), title='Allocation change vs equal weight'), color=color, tooltip=['Stock:N','Portfolio:N',alt.Tooltip('Active weight:Q',format='+.2%')])
    st.altair_chart(bars.encode(yOffset=alt.YOffset('Portfolio:N', sort=NAMES[1:])).properties(height=440), width='stretch')
    st.caption('Right of zero means overweight; left means underweight. Equal weight is the zero reference, so identical baseline bars are omitted. Each asset has a 30% weight cap.')
    with st.expander('Full performance scorecard and allocation weights'):
        percent_cols = [c for c in metrics if c != 'Effective holdings']
        st.dataframe(metrics.style.format({**{c:'{:.2%}' for c in percent_cols}, 'Effective holdings':'{:.1f}'}))
        st.caption('Effective holdings = 1 / sum(weight²): concentration expressed as an equivalent number of equally weighted positions. Return and volatility use 252 sessions/year.')
        st.dataframe(weights.style.format('{:.1%}'))
    with st.expander('Compounded growth of 100 starting units'):
        growth = (wealth*100).rename_axis('Date').reset_index().melt('Date',var_name='Portfolio',value_name='Wealth')
        st.altair_chart(alt.Chart(growth).mark_line().encode(x='Date:T', y=alt.Y('Wealth:Q',title='Value of 100 initial units'), color=color, tooltip=['Date:T','Portfolio:N',alt.Tooltip('Wealth:Q',format='.1f')]).interactive(), width='stretch')
    st.caption('Returns aggregate constituent simple returns before compounding. No transaction costs, taxes or market impact. The existing optimization fits log-return scenarios, a small-return approximation; reported performance uses realized simple returns. All displayed results are retrospective.')
