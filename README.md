# VaR & CVaR Calculator for NSE Stocks

This project implements and evaluates 1-day Value at Risk (VaR) and Conditional Value at Risk (CVaR / Expected Shortfall) models for a portfolio of large-capitalization NSE stocks. The implemented methodologies include Historical Simulation, Parametric Normal, Parametric Student's t, and Rockafellar-Uryasev (RU) LP Optimization.

## 1. Empirical Results & Analysis

### Benchmark Backtesting (99% Confidence Level, 250-Day Rolling Window)

Models were evaluated out-of-sample over a robust 20-year horizon (2006–2026, approx. 4,600+ trading days). The period encompasses severe structural breaks, including the 2008 Global Financial Crisis, the 2013 Taper Tantrum, and the 2020 COVID-19 crash. 

Targeting an exact 1.00% exception rate (99% VaR) over such a heterogeneous dataset is practically impossible for static rolling-window models. The objective is rather to measure the relative degree of misspecification. Below are the empirical findings utilizing RELIANCE.NS as a representative asset:

* **Parametric Normal Model**
  * **Realized Exception Rate:** ~1.69% 
  * **Statistical Assessment:** The Normal distribution severely underestimates tail risk due to leptokurtic (fat-tailed) equity returns. It is decisively rejected by the Kupiec Proportion of Failures (POF) test ($p < 0.001$). Under the Basel Committee penalty framework, this model frequently degrades to the **Yellow Zone**, rendering it inadequate for strict regulatory capital requirements.

* **Historical Simulation (Non-Parametric)**
  * **Realized Exception Rate:** ~1.45% 
  * **Statistical Assessment:** While free from distributional assumptions, the unweighted historical approach exhibits significant "ghost effects" and is excessively slow to update during regime shifts. It generally fails the Kupiec POF test ($p \approx 0.003$) in long-term backtests.

* **Parametric Student's t Model**
  * **Realized Exception Rate:** ~1.32%
  * **Statistical Assessment:** By explicitly modelling the fat tails (with estimated degrees of freedom $\nu \in [3, 6]$), the Student's t approach provides a vastly superior fit. While it still exhibits slight under-coverage during compounding crisis events, it minimizes the POF penalty and remains comfortably within the **Green Zone** of the Basel traffic light framework.

## 2. Model Limitations & Theoretical Violations

A core tenet of this analysis is documenting where classical risk models fail in real-world applications:

* **Violation of the i.i.d. Assumption (Volatility Clustering):**
  All evaluated rolling-window VaR models fundamentally assume that daily returns are independent and identically distributed (i.i.d.). Exploratory Data Analysis (specifically the Autocorrelation Function of squared returns) empirically disproves this, demonstrating severe volatility clustering (ARCH effects). 
  Consequently, all models decisively fail the **Christoffersen Independence Test** ($p < 0.05$). When VaR breaches occur, they do not occur randomly; they cluster during high-volatility regimes, exposing portfolios to consecutive, compounding drawdowns.
* **Survivorship Bias:**
  The universe of analyzed assets consists of large-cap equities that survived and remained highly liquid from 2006 to 2026. This retrospective selection introduces survivorship bias, implying that the empirical risk metrics presented may understate the true risk of a contemporary, unconditioned stock selection.
* **Data Artifacts & Corporate Actions:**
  The analysis relies on `yfinance` auto-adjusted closing prices to account for splits, bonuses, and dividends. While generally robust, algorithmic adjustments can occasionally introduce minor artifacts or phantom returns, marginally distorting localized volatility estimates.

## 3. Visual Diagnostics

The Streamlit frontend (`app.py`) provides an interactive diagnostic suite:

1. **Distributional Overlays:** Histograms overlaid with Normal and Student's t Probability Density Functions (PDFs) to visualize leptokurtosis.
2. **Quantile-Quantile (Q-Q) Plots:** Demonstrates the severe deviation of empirical tail quantiles from the Normal assumption, contrasting with the superior Student's t alignment.
3. **Rolling VaR vs. Realized Losses:** Time-series tracking of risk estimates versus realized daily losses, explicitly mapping exception clusters.
4. **Rockafellar-Uryasev (RU) Objective Function:** A plot of the convex, piecewise-linear $F_\alpha(\zeta)$ objective space, demonstrating the convergence to VaR (the minimizer, $\zeta^*$) and CVaR (the minimum).
5. **Autocorrelation (ACF) Diagnostics:** ACF plots of both returns and squared returns, visually establishing the presence of volatility clustering.

## 4. Architecture & Implementation

* `src/data_loader.py`: Asynchronous data ingestion and preprocessing (cleaning, missing value imputation, extreme outlier flagging).
* `src/var_models.py`: Vectorized implementations of Historical, Parametric Normal, and Student's t risk models.
* `src/ru_optimization.py`: Linear Programming (LP) formulation of the Rockafellar-Uryasev CVaR optimization (solved via `scipy.optimize.linprog` with HiGHS).
* `src/numerical_var.py`: Alternative VaR derivations (pinball loss minimization, root finding, Maximum Likelihood Estimation).
* `src/backtest.py`: Rolling window backtest engine; calculates Kupiec POF, Christoffersen Independence, and Expected Shortfall (ES) ratios.
* `src/portfolio.py`: Capital allocation models (Equal Weight, Min-Variance, Min-CVaR out-of-sample optimization).
* `app.py`: Interactive Streamlit dashboard.

## 5. Setup & Execution

1. Initialize the virtual environment and install dependencies:
   ```bash
   python3 -m venv .venv
   source .venv/bin/activate
   pip install -r requirements.txt
   ```
2. Execute the `pytest` validation suite (verifies LP convergence, closed-form vs. Monte Carlo equivalence, and monotonic properties):
   ```bash
   pytest tests/
   ```
3. Launch the interactive diagnostic dashboard:
   ```bash
   streamlit run app.py
   ```
