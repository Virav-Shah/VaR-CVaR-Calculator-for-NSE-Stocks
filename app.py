import streamlit as st
import pandas as pd
import numpy as np
import os
from src.data_loader import RAW_DATA_PATH, download_and_clean_data
from src.returns import calculate_log_returns
from src.backtest import backtest_model
from src.plots import plot_rolling_var, plot_histogram_with_overlays

st.set_page_config(page_title="NSE VaR & CVaR Calculator", layout="wide")

st.title("VaR & CVaR Calculator for NSE Stocks")

@st.cache_data
def load_data():
    if not os.path.exists(RAW_DATA_PATH):
        download_and_clean_data()
    df = pd.read_csv(RAW_DATA_PATH, index_col=0, parse_dates=True)
    returns = calculate_log_returns(df)
    return df, returns

with st.spinner("Loading data..."):
    prices, returns_df = load_data()

st.sidebar.header("Configuration")

tickers = list(prices.columns)
selected_asset = st.sidebar.selectbox("Select Asset or Portfolio", tickers + ["Equal-Weight Portfolio"])

alpha = st.sidebar.selectbox("Confidence Level (Alpha)", [0.95, 0.99], index=1)
window = st.sidebar.selectbox("Rolling Window (Days)", [250, 500, 1000], index=0)

if selected_asset == "Equal-Weight Portfolio":
    w = np.ones(returns_df.shape[1]) / returns_df.shape[1]
    series_to_analyze = returns_df.dot(w)
    series_to_analyze.name = "Portfolio"
else:
    series_to_analyze = returns_df[selected_asset]

series_to_analyze = series_to_analyze.dropna()

st.subheader(f"Exploratory Data Analysis: {selected_asset}")
import scipy.stats as stats
col1, col2 = st.columns(2)
with col1:
    fig_hist = plot_histogram_with_overlays(series_to_analyze, selected_asset)
    st.pyplot(fig_hist)
with col2:
    mean_val = series_to_analyze.mean()
    std_val = series_to_analyze.std()
    skew_val = series_to_analyze.skew()
    kurt_val = series_to_analyze.kurt() # Excess kurtosis
    min_val = series_to_analyze.min()
    max_val = series_to_analyze.max()
    jb_stat, jb_p = stats.jarque_bera(series_to_analyze)
    
    stats_df = pd.DataFrame({
        "Metric": ["Mean", "Std Dev", "Skew", "Excess Kurtosis", "Min", "Max", "Jarque-Bera p-value"],
        "Value": [f"{mean_val:.4f}", f"{std_val:.4f}", f"{skew_val:.4f}", f"{kurt_val:.4f}", f"{min_val:.4f}", f"{max_val:.4f}", f"{jb_p:.4e}"]
    })
    st.dataframe(stats_df, hide_index=True)

st.subheader("Backtesting Models (Historical, Normal, Student-t)")

@st.cache_data
def run_all_backtests(series, w, a):
    results = {}
    preds = {}
    for method in ["historical", "normal", "student_t"]:
        res, var_s, cvar_s = backtest_model(series, window=w, alpha=a, method=method)
        results[method] = res
        preds[method] = var_s
    return results, preds

with st.spinner("Running backtests... (t-distribution fitting may take a minute)"):
    results_dict, preds_dict = run_all_backtests(series_to_analyze, window, alpha)

# Display results table
res_df = pd.DataFrame(results_dict).T
st.dataframe(res_df.style.format({
    "Exceptions": "{:.0f}",
    "Exception Rate (%)": "{:.2f}%",
    "Kupiec p-value": "{:.4f}",
    "Christoffersen p-value": "{:.4f}",
    "ES Ratio": "{:.2f}",
    "Valid Days": "{:.0f}"
}))

st.subheader("Rolling VaR vs Realized Losses")
fig_var = plot_rolling_var(series_to_analyze, preds_dict, alpha, selected_asset)
st.pyplot(fig_var)

st.markdown("""
### Methodology & Limitations
- **Historical**: Empirical quantile of past losses. Slow to react to regime shifts.
- **Normal**: Assumes returns are normally distributed. Often underestimates tail risk (high exception rate at 99%).
- **Student-t**: Accounts for fat tails. Typically performs best for equity returns.
- **Limitations**: All methods assume returns are independent and identically distributed within the window. Volatility clustering (evident in ACF of squared returns) violates this, often causing the Christoffersen independence test to fail.
""")
