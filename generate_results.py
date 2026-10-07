"""
Generate all results (figures + tables) for the VaR & CVaR project.
Run: python generate_results.py
"""
import os
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving files
import matplotlib.pyplot as plt
import scipy.stats as stats

from src.data_loader import RAW_DATA_PATH, download_and_clean_data
from src.returns import calculate_log_returns
from src.backtest import backtest_model
from src.plots import (
    plot_histogram_with_overlays,
    plot_qq,
    plot_rolling_var,
    plot_ru_objective,
    plot_acf_returns_vs_squared,
)
from src.portfolio import out_of_sample_test, min_variance_portfolio
from src.ru_optimization import ru_cvar_single, ru_portfolio_optimization
from src.var_models import historical_var_cvar, normal_var_cvar, student_t_var_cvar

FIGURES_DIR = "results/figures"
TABLES_DIR = "results/tables"
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(TABLES_DIR, exist_ok=True)

# ── 1. Load data ──────────────────────────────────────────────────────────────
print("=" * 60)
print("STEP 1: Loading data...")
if not os.path.exists(RAW_DATA_PATH):
    download_and_clean_data()
prices = pd.read_csv(RAW_DATA_PATH, index_col=0, parse_dates=True)
returns = calculate_log_returns(prices)
tickers = list(returns.columns)
print(f"  Loaded {len(tickers)} tickers, {returns.shape[0]} trading days")
print(f"  Date range: {returns.index[0].date()} to {returns.index[-1].date()}")

# ── 2. Exploratory Data Analysis table ────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 2: Generating EDA statistics table...")
eda_rows = []
for ticker in tickers:
    s = returns[ticker].dropna()
    jb_stat, jb_p = stats.jarque_bera(s)
    eda_rows.append({
        "Ticker": ticker.replace(".NS", ""),
        "Observations": len(s),
        "Mean (%)": f"{s.mean()*100:.4f}",
        "Std Dev (%)": f"{s.std()*100:.4f}",
        "Skewness": f"{s.skew():.4f}",
        "Excess Kurtosis": f"{s.kurt():.4f}",
        "Min (%)": f"{s.min()*100:.4f}",
        "Max (%)": f"{s.max()*100:.4f}",
        "JB p-value": f"{jb_p:.2e}",
    })
eda_df = pd.DataFrame(eda_rows)
eda_df.to_csv(os.path.join(TABLES_DIR, "eda_statistics.csv"), index=False)
print("  Saved: tables/eda_statistics.csv")
print(eda_df.to_string(index=False))

# ── 3. Histogram + distribution overlays for each ticker ─────────────────────
print("\n" + "=" * 60)
print("STEP 3: Generating histogram plots...")
for ticker in tickers:
    s = returns[ticker].dropna()
    fig = plot_histogram_with_overlays(s, ticker.replace(".NS", ""))
    fname = f"histogram_{ticker.replace('.NS', '')}.png"
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: figures/{fname}")

# ── 4. QQ plots for each ticker ──────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 4: Generating QQ plots...")
for ticker in tickers:
    s = returns[ticker].dropna()
    fig = plot_qq(s, ticker.replace(".NS", ""))
    fname = f"qq_{ticker.replace('.NS', '')}.png"
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: figures/{fname}")

# ── 5. ACF plots for each ticker ─────────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 5: Generating ACF plots...")
for ticker in tickers:
    s = returns[ticker].dropna()
    fig = plot_acf_returns_vs_squared(s, ticker.replace(".NS", ""))
    fname = f"acf_{ticker.replace('.NS', '')}.png"
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: figures/{fname}")

# ── 6. Backtest all tickers × all methods ────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 6: Running backtests (this may take a few minutes)...")
ALPHA = 0.99
WINDOW = 250
all_backtest_rows = []

for ticker in tickers:
    s = returns[ticker].dropna()
    preds_dict = {}
    print(f"\n  Backtesting {ticker}...")

    for method in ["historical", "normal", "student_t"]:
        print(f"    Method: {method}...", end=" ", flush=True)
        res, var_s, cvar_s = backtest_model(s, window=WINDOW, alpha=ALPHA, method=method)
        preds_dict[method] = var_s
        row = {"Ticker": ticker.replace(".NS", ""), "Method": method}
        row.update(res)
        all_backtest_rows.append(row)
        print(f"Exceptions={res['Exceptions']:.0f}, Rate={res['Exception Rate (%)']:.2f}%")

    # Rolling VaR plot for this ticker
    fig = plot_rolling_var(s, preds_dict, ALPHA, ticker.replace(".NS", ""))
    fname = f"rolling_var_{ticker.replace('.NS', '')}.png"
    fig.savefig(os.path.join(FIGURES_DIR, fname), dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"    Saved: figures/{fname}")

backtest_df = pd.DataFrame(all_backtest_rows)
backtest_df.to_csv(os.path.join(TABLES_DIR, "backtest_results.csv"), index=False)
print(f"\n  Saved: tables/backtest_results.csv")

# ── 7. Backtest summary (aggregated across all tickers) ──────────────────────
print("\n" + "=" * 60)
print("STEP 7: Generating backtest summary...")
summary_rows = []
for method in ["historical", "normal", "student_t"]:
    subset = backtest_df[backtest_df["Method"] == method]
    summary_rows.append({
        "Method": method,
        "Avg Exception Rate (%)": f"{subset['Exception Rate (%)'].mean():.2f}",
        "Avg Kupiec p-value": f"{subset['Kupiec p-value'].mean():.4f}",
        "Avg Christoffersen p-value": f"{subset['Christoffersen p-value'].mean():.4f}",
        "Avg ES Ratio": f"{subset['ES Ratio'].mean():.2f}",
    })
summary_df = pd.DataFrame(summary_rows)
summary_df.to_csv(os.path.join(TABLES_DIR, "backtest_summary.csv"), index=False)
print("  Saved: tables/backtest_summary.csv")
print(summary_df.to_string(index=False))

# ── 8. Single-asset VaR/CVaR comparison table ────────────────────────────────
print("\n" + "=" * 60)
print("STEP 8: Generating VaR/CVaR comparison table...")
var_compare_rows = []
for ticker in tickers:
    s = returns[ticker].dropna()
    losses = -s.values

    var_h, cvar_h = historical_var_cvar(losses, ALPHA)
    var_n, cvar_n = normal_var_cvar(s.values, ALPHA)
    result = student_t_var_cvar(s.values, ALPHA)
    var_t, cvar_t, nu = result[0], result[1], result[2]

    var_compare_rows.append({
        "Ticker": ticker.replace(".NS", ""),
        "Historical VaR (%)": f"{var_h*100:.2f}",
        "Historical CVaR (%)": f"{cvar_h*100:.2f}",
        "Normal VaR (%)": f"{var_n*100:.2f}",
        "Normal CVaR (%)": f"{cvar_n*100:.2f}",
        "Student-t VaR (%)": f"{var_t*100:.2f}",
        "Student-t CVaR (%)": f"{cvar_t*100:.2f}",
        "Fitted nu (df)": f"{nu:.2f}",
    })
var_compare_df = pd.DataFrame(var_compare_rows)
var_compare_df.to_csv(os.path.join(TABLES_DIR, "var_cvar_comparison.csv"), index=False)
print("  Saved: tables/var_cvar_comparison.csv")
print(var_compare_df.to_string(index=False))

# ── 9. RU Objective Function plot (using RELIANCE as example) ────────────────
print("\n" + "=" * 60)
print("STEP 9: Generating RU objective function plot...")
rel_losses = -returns["RELIANCE.NS"].dropna().values
fig = plot_ru_objective(rel_losses, alpha=ALPHA)
fig.savefig(os.path.join(FIGURES_DIR, "ru_objective_RELIANCE.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: figures/ru_objective_RELIANCE.png")

# Also verify RU CVaR matches historical
var_ru, cvar_ru = ru_cvar_single(rel_losses, ALPHA)
var_h, cvar_h = historical_var_cvar(rel_losses, ALPHA)
print(f"  RU CVaR = {cvar_ru:.6f}, Historical CVaR = {cvar_h:.6f}, Diff = {abs(cvar_ru - cvar_h):.6f}")

# ── 10. Portfolio out-of-sample test ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 10: Running portfolio out-of-sample test...")
returns_clean = returns.dropna()
oos_result = out_of_sample_test(returns_clean, train_end="2015-12-31", alpha=ALPHA)
if oos_result is not None:
    oos_result.to_csv(os.path.join(TABLES_DIR, "portfolio_oos_results.csv"))
    print("  Saved: tables/portfolio_oos_results.csv")
    print(oos_result.to_string())
else:
    print("  WARNING: Out-of-sample test returned None (insufficient data)")

# ── 11. Portfolio weights comparison ─────────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 11: Generating portfolio weights comparison...")
train_returns = returns_clean[:"2015-12-31"]

w_ew = np.ones(len(tickers)) / len(tickers)
w_mv = min_variance_portfolio(train_returns, w_max=0.30)
w_cvar, _, _ = ru_portfolio_optimization(train_returns, alpha=ALPHA, w_max=0.30)

weights_df = pd.DataFrame({
    "Ticker": [t.replace(".NS", "") for t in tickers],
    "Equal Weight (%)": [f"{w*100:.1f}" for w in w_ew],
    "Min Variance (%)": [f"{w*100:.1f}" for w in w_mv],
    "Min CVaR (%)": [f"{w*100:.1f}" for w in w_cvar] if not np.isnan(w_cvar).any() else ["N/A"] * len(tickers),
})
weights_df.to_csv(os.path.join(TABLES_DIR, "portfolio_weights.csv"), index=False)
print("  Saved: tables/portfolio_weights.csv")
print(weights_df.to_string(index=False))

# ── 12. Equal-weight portfolio backtest ──────────────────────────────────────
print("\n" + "=" * 60)
print("STEP 12: Running equal-weight portfolio backtest...")
w = np.ones(returns_clean.shape[1]) / returns_clean.shape[1]
port_returns = returns_clean.dot(w)
port_returns.name = "Portfolio"
port_preds = {}

for method in ["historical", "normal", "student_t"]:
    print(f"  Method: {method}...", end=" ", flush=True)
    res, var_s, _ = backtest_model(port_returns, window=WINDOW, alpha=ALPHA, method=method)
    port_preds[method] = var_s
    print(f"Exceptions={res['Exceptions']:.0f}, Rate={res['Exception Rate (%)']:.2f}%")

fig = plot_rolling_var(port_returns, port_preds, ALPHA, "Equal-Weight Portfolio")
fig.savefig(os.path.join(FIGURES_DIR, "rolling_var_portfolio.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: figures/rolling_var_portfolio.png")

fig = plot_histogram_with_overlays(port_returns, "Equal-Weight Portfolio")
fig.savefig(os.path.join(FIGURES_DIR, "histogram_portfolio.png"), dpi=150, bbox_inches="tight")
plt.close(fig)
print("  Saved: figures/histogram_portfolio.png")

# ── Done ─────────────────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print("ALL RESULTS GENERATED SUCCESSFULLY!")
print(f"  Figures: {FIGURES_DIR}/")
print(f"  Tables:  {TABLES_DIR}/")
print("=" * 60)
