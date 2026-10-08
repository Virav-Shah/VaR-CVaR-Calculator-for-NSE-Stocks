# VaR & CVaR Calculator for NSE Stocks

> A complete risk measurement system that answers one question every bank, hedge fund, and portfolio manager asks every single day: **"What is the worst-case loss I could face tomorrow?"**

[![Streamlit App](https://static.streamlit.io/badges/streamlit_badge_black_white.svg)](https://mkxk2evmghkv5rbpugw99x.streamlit.app/)
🔗 **Live Web Application:** [https://mkxk2evmghkv5rbpugw99x.streamlit.app/](https://mkxk2evmghkv5rbpugw99x.streamlit.app/)

---

## Table of Contents

1. [What This Project Does — In Plain Financial English](#1-what-this-project-does--in-plain-financial-english)
2. [The Financial Problem Being Solved](#2-the-financial-problem-being-solved)
3. [The Stocks We Analyse](#3-the-stocks-we-analyse)
4. [Step-by-Step: How We Get From Raw Prices to Risk Numbers](#4-step-by-step-how-we-get-from-raw-prices-to-risk-numbers)
5. [The Three VaR Models — What They Mean Financially](#5-the-three-var-models--what-they-mean-financially)
6. [CVaR / Expected Shortfall — Going Beyond VaR](#6-cvar--expected-shortfall--going-beyond-var)
7. [The Rockafellar-Uryasev Optimisation — Smarter Portfolio Allocation](#7-the-rockafellar-uryasev-optimisation--smarter-portfolio-allocation)
8. [Backtesting — Did Our Risk Predictions Actually Work?](#8-backtesting--did-our-risk-predictions-actually-work)
9. [Results and What They Mean](#9-results-and-what-they-mean)
10. [Where These Models Break Down — Honest Limitations](#10-where-these-models-break-down--honest-limitations)
11. [Visual Diagnostics — What Each Chart Shows](#11-visual-diagnostics--what-each-chart-shows)
12. [Setup & Execution](#12-setup--execution)

---

## 1. What This Project Does — In Plain Financial English

Imagine you are a risk manager at a bank. You hold stocks worth ₹10 crore. Your CEO asks: *"If things go badly tomorrow, how much money could we lose?"*

This project answers that question using **three different methods**, tests whether those answers were historically accurate, and even tells you how to **allocate money across stocks to minimise your worst-case losses**.

Specifically, this project:

- **Downloads** 20 years of daily closing prices for 10 major Indian (NSE) stocks
- **Calculates daily returns** (how much each stock went up or down every day)
- **Fits statistical models** to understand the pattern of those returns
- **Computes VaR** (Value at Risk): *"What is the maximum I could lose on 99 out of 100 days?"*
- **Computes CVaR** (Conditional VaR / Expected Shortfall): *"On that 1 worst day out of 100, how bad could it actually get?"*
- **Backtests** these predictions against 20 years of actual market data to see which model was most accurate
- **Optimises portfolios** to find the allocation of money across stocks that minimises CVaR

---

## 2. The Financial Problem Being Solved

### Why Risk Measurement Matters

Every financial institution — banks, insurance companies, mutual funds, hedge funds — is **required by regulators** (RBI in India, Basel Committee globally) to set aside capital to cover potential losses. If they underestimate risk, they might go bankrupt (like Lehman Brothers in 2008). If they overestimate it, they lock up too much capital and miss profitable opportunities.

### VaR — The Industry Standard

**Value at Risk (VaR)** is the single most widely used risk metric in the financial industry. It was popularised by JP Morgan in the 1990s (through their RiskMetrics system) and is now **mandated by the Basel Accords** for bank capital requirements.

**VaR answers:** *"With X% confidence, my portfolio will NOT lose more than ₹Y in the next trading day."*

**Example:** If 99% VaR = 3.5%, it means: *On 99 out of 100 trading days, the portfolio will not lose more than 3.5% of its value. On 1 out of 100 days, it could lose more.*

### CVaR — What VaR Doesn't Tell You

VaR has a major weakness: it tells you the *boundary* of the bad zone, but not *how bad things get inside it*. If VaR says "you won't lose more than 3.5% on 99 days out of 100", it says nothing about that 1 remaining day — you could lose 4%, or 40%.

**CVaR (Conditional VaR, also called Expected Shortfall)** answers: *"On those worst 1% of days, what is my AVERAGE loss?"*

CVaR is always >= VaR, and it captures "tail risk" — the extreme, rare events that can bankrupt institutions.

> **Key insight:** After the 2008 financial crisis, the Basel Committee (Basel III / FRTB) moved from VaR to Expected Shortfall as the primary risk metric for market risk capital. This project implements both.

---

## 3. The Stocks We Analyse

We analyse **10 large-cap NSE stocks** spanning key sectors of the Indian economy:

| Stock | Sector | Why It Matters |
|-------|--------|----------------|
| RELIANCE | Energy / Conglomerate | Largest company by market cap |
| TCS | IT Services | Represents tech sector; dollar-earning |
| HDFC BANK | Banking (Private) | Largest private bank |
| INFOSYS | IT Services | Second-largest IT company |
| ICICI BANK | Banking (Private) | Major private bank |
| SBI | Banking (PSU) | Largest public-sector bank |
| ITC | FMCG / Consumer | Defensive; low volatility |
| HUL | FMCG / Consumer | Defensive; low volatility |
| L&T | Infrastructure | Cyclical; high beta |
| BHARTI AIRTEL | Telecom | Sector leader |

**Time Period:** January 2006 to January 2026 (~20 years, ~4,800 trading days)

This 20-year window is chosen deliberately because it includes:
- **2008 Global Financial Crisis** — markets fell 50%+ in months
- **2013 Taper Tantrum** — sudden capital outflows from emerging markets
- **2018 IL&FS Crisis** — Indian credit crisis
- **2020 COVID Crash** — fastest bear market in history (30% drop in 1 month)
- **2022 Rate Hike Cycle** — sustained volatility from global tightening

These events are *exactly* the scenarios VaR is supposed to protect against.

---

## 4. Step-by-Step: How We Get From Raw Prices to Risk Numbers

Here is the complete pipeline, explained from a financial perspective:

### Step 1: Download Adjusted Closing Prices

We download the **adjusted closing price** for each stock. "Adjusted" means the price has been corrected for:
- **Stock splits** (e.g., if a Rs.2000 stock splits 1:10 to Rs.200, old prices are divided by 10)
- **Bonus shares** (free shares given to existing holders)
- **Dividends** (the ex-dividend price drop is added back)

Without adjustment, a stock split would look like a 90% crash in raw data, which would completely corrupt our risk numbers.

### Step 2: Data Quality Checks

Before any analysis, we clean the data:
- **Remove duplicate trading days** (data errors)
- **Flag returns > 20% in a single day** — these could be genuine (circuit limits on NSE are 20%) or data errors. We flag them for manual review.
- **Flag stale prices** — if a stock shows zero return for 5+ consecutive days, it may have been suspended or the data may be stale. Stale prices understate true volatility.

### Step 3: Calculate Log Returns

We convert prices to **logarithmic returns** (also called continuously compounded returns):

```
r_t = ln(P_t / P_{t-1})
```

**Why log returns instead of simple percentage returns?**

| Property | Simple Returns | Log Returns |
|----------|---------------|-------------|
| Multi-day returns | Multiplicative (messy) | Additive (just sum them up) |
| Symmetry | +50% gain does not equal -50% loss | Symmetric around zero |
| Statistical modelling | Harder to model | Easier — can use Normal/t distributions |
| Industry standard | Used for single-period | Used for risk modelling |

**Example:**
- Stock goes from Rs.100 to Rs.110: log return = ln(110/100) = +9.53%
- Stock goes from Rs.110 to Rs.100: log return = ln(100/110) = -9.53%
- The gains and losses are symmetric, which is critical for statistical modelling.

### Step 4: Convert Returns to Losses

We **flip the sign**: Loss = -Return.

If a stock returned -3% (fell 3%), the loss is +3%. This is a convention so that VaR and CVaR are reported as **positive numbers** (a VaR of 3% means you could lose 3%, not that you gain 3%).

### Step 5: Compute VaR and CVaR Using Three Methods

Using a **rolling window** (default: 250 trading days = 1 year), we compute VaR and CVaR using three different models (described in the next section).

For each day `t`, we look at the previous 250 days of returns, estimate the risk, and then check what actually happened on day `t`. This simulates what a risk manager would do every morning.

### Step 6: Backtest the Predictions

We compare our VaR predictions against actual losses over the entire 20-year period to see:
- How often did losses exceed our VaR prediction? (These are called **exceptions** or **breaches**)
- Did exceptions occur randomly, or did they cluster together?
- When losses exceeded VaR, was our CVaR estimate accurate?

### Step 7: Portfolio Optimisation

Beyond individual stocks, we optimise how to allocate money across all 10 stocks to **minimise the portfolio's CVaR** using the Rockafellar-Uryasev Linear Programming method.

---

## 5. The Three VaR Models — What They Mean Financially

### Model 1: Historical Simulation (Non-Parametric)

**Financial Intuition:** *"The future will look like the past."*

**How it works:**
1. Take the last 250 days of actual losses
2. Sort them from smallest to largest
3. VaR (99%) = the loss value at the 99th percentile

**Example:** If we have 250 days of losses and sort them, VaR(99%) is roughly the 2nd or 3rd worst day.

**Advantages:**
- No assumptions about the shape of returns — uses actual historical data
- Captures fat tails if they occurred in the window

**Disadvantages:**
- **Ghost effects:** If a crash happened 249 days ago, it's in the window. On day 251, it drops out, and VaR suddenly falls — even though nothing changed in the market.
- **Slow to react:** If volatility suddenly spikes (like March 2020), it takes many days for the rolling window to reflect the new reality.

### Model 2: Parametric Normal (Variance-Covariance)

**Financial Intuition:** *"Returns follow a bell curve."*

**How it works:**
1. Estimate the mean (mu) and standard deviation (sigma) of returns from the 250-day window
2. Assume returns follow a Normal (Gaussian) distribution
3. Use the formula: **VaR = -mu + sigma x z_alpha** where z_alpha = 2.326 for 99% confidence

**Financial reality check — why this is a problem:**

The Normal distribution assumes extreme events are virtually impossible. Under a Normal distribution:
- A 5-sigma event (like a -10% daily move) should happen once every 14,000 years
- In reality, the Indian market has seen 5-sigma events multiple times in just 20 years

This is because stock returns have **fat tails** (also called leptokurtosis) — extreme events are far more common than the Normal bell curve predicts.

**Result:** The Normal model consistently **underestimates** risk, producing the highest exception rate of all three models.

### Model 3: Parametric Student's t-Distribution

**Financial Intuition:** *"Returns follow a bell curve, but with fatter tails."*

**How it works:**
1. Fit a Student's t-distribution to the 250-day window, estimating three parameters:
   - **nu (degrees of freedom):** Controls how fat the tails are. Lower nu = fatter tails. For equity returns, nu is typically 3-6.
   - **mu:** Location (similar to the mean)
   - **s (sigma):** Scale (similar to standard deviation)
2. Use the t-distribution's inverse CDF to compute VaR
3. Use the closed-form CVaR formula specific to the t-distribution

**Why this works better:**

With nu around 4 (typical for Indian equities), the t-distribution assigns much more probability to extreme events. A 5-sigma event that is "impossible" under Normal is "rare but plausible" under Student's t.

**Mathematical connection:** As nu approaches infinity, the Student's t distribution converges to the Normal distribution. So the Normal model is actually a special case of the t model — just with unrealistically thin tails.

**Result:** The Student's t model produces the closest exception rate to the theoretical 1% target, making it the best-performing model.

---

## 6. CVaR / Expected Shortfall — Going Beyond VaR

### Why CVaR is Financially Superior to VaR

| Property | VaR | CVaR |
|----------|-----|------|
| Answers | "What's the boundary of the worst 1%?" | "What's the average loss in the worst 1%?" |
| Sub-additive? | **No** — diversifying can increase VaR (nonsensical) | **Yes** — diversifying always reduces or maintains CVaR |
| Coherent risk measure? | **No** | **Yes** (satisfies all 4 axioms) |
| Regulatory status | Basel II standard | Basel III / FRTB standard (replacing VaR) |
| Tail information | None (single threshold) | Full (average of tail losses) |

**Sub-additivity failure of VaR — a concrete example:**

Suppose you have two bonds, each with a 3% chance of defaulting. Individually, their 95% VaR = 0 (no loss 97% of the time). But combined, the portfolio has a 6% chance of at least one defaulting, so the 95% VaR > 0. The portfolio VaR is **greater** than the sum of individual VaRs — meaning "diversification increased risk," which is financially absurd. CVaR never has this problem.

### The Rockafellar-Uryasev Insight

CVaR cannot be directly optimised using simple calculus because it depends on VaR, which itself is a quantile (not smooth). In 2000, Rockafellar and Uryasev proved that CVaR can be reformulated as a **convex optimisation problem** using an auxiliary variable zeta:

```
CVaR_alpha = min_zeta { zeta + 1/((1-alpha)*N) * SUM max(Loss_i - zeta, 0) }
```

**Financial meaning of this formula:**
- zeta is a "candidate VaR threshold" we're searching over
- max(Loss_i - zeta, 0) captures how much each loss exceeds the threshold (shortfall)
- We average those exceedances and add zeta
- The zeta that minimises this expression turns out to equal VaR, and the minimum value equals CVaR

This is brilliant because it transforms CVaR from an intractable quantile-based problem into a simple **Linear Program** that standard optimisers can solve in seconds.

---

## 7. The Rockafellar-Uryasev Optimisation — Smarter Portfolio Allocation

### What This Module Actually Does

Given 10 stocks, there are infinitely many ways to allocate money across them. The project finds **three different "optimal" allocations** and compares them:

| Strategy | Objective | Financial Logic |
|----------|-----------|-----------------|
| **Equal Weight** | 10% in each stock | Baseline; no optimisation |
| **Minimum Variance** | Minimise portfolio variance | Classic Markowitz approach; focuses on average dispersion |
| **Minimum CVaR (RU)** | Minimise worst-case tail losses | Focuses specifically on extreme downside |

### How the LP Formulation Works (Financially)

The optimiser decides how much money to put in each stock (w1, w2, ..., w10) such that:
- All weights sum to 100% (you're fully invested)
- No single stock gets more than 30% (diversification constraint)
- No short selling (all weights >= 0)
- The portfolio's CVaR is minimised

The **out-of-sample test** trains on pre-2016 data and evaluates on post-2016 data. This prevents "overfitting" — we can't just find weights that worked historically and assume they'll work in the future.

---

## 8. Backtesting — Did Our Risk Predictions Actually Work?

Backtesting is the process of checking whether your risk model's predictions were accurate when compared to what actually happened. This project uses four backtesting frameworks:

### 8.1 Exception Counting

An **exception** (or **breach**) occurs when the actual loss exceeds the predicted VaR.

For a 99% VaR model, we expect exceptions 1% of the time. Over 4,600 trading days, that means ~46 exceptions expected.

- **Too many exceptions** -> model underestimates risk (dangerous)
- **Too few exceptions** -> model overestimates risk (capital-inefficient)

### 8.2 Kupiec Proportion of Failures (POF) Test

**What it tests:** *"Is the observed exception rate statistically different from the expected 1%?"*

- Uses a **Likelihood Ratio test** with chi-squared distribution
- **p-value > 0.05** -> cannot reject the model -> model is acceptable
- **p-value < 0.05** -> model is statistically rejected -> exception rate is significantly different from 1%

**Analogy:** If you flip a "fair coin" (expected 50% heads) and get 70% heads, the Kupiec test tells you whether that deviation is due to random chance or because the coin is actually biased.

### 8.3 Christoffersen Independence Test

**What it tests:** *"Do exceptions occur randomly, or do they cluster together?"*

Even if the overall exception rate is exactly 1%, if all 46 exceptions happen in one month (during a crisis), the model is still dangerous because:
- A bank might face 10 consecutive days of VaR breaches
- Capital buffers could be exhausted before recovery
- This violates the i.i.d. assumption

The test uses a **Markov chain transition matrix** to check whether the probability of an exception tomorrow depends on whether there was an exception today.

**Result in this project:** All models fail this test because volatility clusters in reality (a bad day is usually followed by more bad days).

### 8.4 Basel Traffic Light System

The Basel Committee classifies VaR models using a "traffic light" based on the number of exceptions in a 250-day (1-year) window at 99% confidence:

| Zone | Exceptions per Year | Consequence |
|------|-------------------|-------------|
| Green | 0-4 | Model accepted; no penalty |
| Yellow | 5-9 | Model questioned; capital surcharge applied |
| Red | 10+ | Model rejected; mandatory switch to standardised approach |

### 8.5 Expected Shortfall (ES) Ratio

**What it tests:** *"On exception days, was the actual loss close to our CVaR prediction?"*

**ES Ratio = Average Realised Loss on Exception Days / Average Predicted CVaR on Exception Days**

- ES Ratio close to 1.0 -> CVaR was well-calibrated
- ES Ratio > 1.0 -> Actual tail losses exceeded CVaR predictions (model underestimates tail severity)
- ES Ratio < 1.0 -> CVaR was conservative (over-estimated tail severity)

---

## 9. Results and What They Mean

### Backtest Results (RELIANCE.NS, 99% Confidence, 250-Day Window)

| Metric | Historical | Normal | Student's t |
|--------|-----------|--------|-------------|
| Exception Rate (target: 1.00%) | ~1.45% | ~1.69% | ~1.32% |
| Kupiec POF p-value | ~0.003 (rejected) | ~0.000 (rejected) | ~0.05 (borderline) |
| Christoffersen p-value | < 0.05 (fails) | < 0.05 (fails) | < 0.05 (fails) |
| Basel Traffic Light | Green | Yellow | Green |

### What This Tells Us Financially

1. **Normal model is the worst:** It underestimates risk because it ignores fat tails. If a bank used this model alone, it would hold insufficient capital and face regulatory penalties (Yellow zone).

2. **Historical model is mediocre:** Better than Normal, but still too many exceptions. The ghost effect and slow adaptation hurt it.

3. **Student's t model is the best:** By modelling fat tails explicitly (nu around 3-6 for Indian equities), it comes closest to the 1% target. It stays comfortably in the Basel Green zone.

4. **ALL models fail the Christoffersen test:** This is not a bug — it's a fundamental limitation. Volatility in the real market clusters (bad days follow bad days), but all three models assume each day is independent. Fixing this requires GARCH-type models (which dynamically update volatility estimates), which is beyond the scope of this project but would be a natural extension.

### Exploratory Data Analysis Findings

The dashboard shows key statistics for each stock:

- **Mean daily return:** Typically 0.03-0.08% (annualised ~8-20%)
- **Standard deviation:** Typically 1.5-2.5% daily (annualised ~24-40%)
- **Skewness:** Slightly negative (losses are more extreme than gains — "crashes are faster than rallies")
- **Excess Kurtosis:** Typically 5-15 (Normal distribution = 0). Confirms **fat tails**.
- **Jarque-Bera p-value:** Essentially 0 for all stocks. Statistically **proves** that returns are NOT normally distributed.

---

## 10. Where These Models Break Down — Honest Limitations

### 10.1 Volatility Clustering (ARCH/GARCH Effects)

All three models use a **fixed rolling window** and assume returns are i.i.d. (independent and identically distributed). In reality:
- After a large loss, volatility tends to stay high for days/weeks
- After calm periods, volatility tends to stay low
- This is called **volatility clustering** and is visible in the ACF of squared returns

**Consequence:** VaR predictions are too high during calm periods (wasting capital) and too low during crises (when you need protection most).

**Fix (not implemented):** GARCH(1,1) or EWMA models that give more weight to recent observations.

### 10.2 Survivorship Bias

We analyse stocks that are large-cap *today*. But 20 years ago, some of these may have been smaller, and other large-caps that existed then may have gone bankrupt or been delisted. By only looking at "winners," we understate the true risk.

### 10.3 Liquidity Risk

VaR assumes you can sell your position at the market price. During a crisis, **bid-ask spreads widen**, order books thin out, and the actual selling price may be far worse than the last traded price. This project does not adjust for liquidity.

### 10.4 Correlation Breakdown

During crises, stocks that normally have low correlation suddenly move together (everyone panics simultaneously). Our portfolio optimisation uses historical correlations, which understate crisis-time co-movement.

### 10.5 Data Limitations

- `yfinance` adjusted prices may have minor errors around corporate actions
- NSE circuit limits (20% upper/lower) cap daily returns, which slightly distorts the true tail distribution
- We use daily frequency; intraday risk is not captured

---

## 11. Visual Diagnostics — What Each Chart Shows

### Histogram with Normal and t-Distribution Overlays
- **What it shows:** The actual distribution of daily returns (grey bars) overlaid with what the Normal model predicts (red line) and what the Student's t model predicts (green line)
- **What to look for:** The grey bars extend further into the tails than the red Normal curve — these are the fat tails. The green t-curve fits much better.

### Rolling VaR vs. Realised Losses
- **What it shows:** A time series of daily losses (grey), with VaR predictions from each model overlaid (coloured lines). Red X marks = exceptions (days when losses exceeded VaR).
- **What to look for:** Exception clusters during crises (2008, 2020) and the Normal model's VaR line being consistently lower than others.

### Rockafellar-Uryasev Objective Function
- **What it shows:** The convex objective function F_alpha(zeta) plotted against zeta. The minimum of this curve gives CVaR, and the zeta at the minimum gives VaR.
- **Why it matters:** Proves that CVaR optimisation is a convex problem (unique minimum), making it computationally tractable.

### ACF of Returns vs. Squared Returns
- **What it shows:** Autocorrelation of raw returns (left) vs. squared returns (right)
- **What to look for:** Raw returns show almost no autocorrelation (prices are hard to predict — consistent with market efficiency). Squared returns show **significant** autocorrelation at many lags — proving that volatility persists (ARCH effects / volatility clustering).

---

## 12. Setup & Execution

### Prerequisites
- Python 3.9+
- Internet connection (for initial stock data download)

### Installation
```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

### Run Tests (Validates Mathematical Correctness)
```bash
pytest tests/
```
The tests verify:
- CVaR >= VaR always holds (monotonicity)
- Rockafellar-Uryasev CVaR matches Historical CVaR (optimisation convergence)
- Numerical VaR (pinball loss) matches quantile VaR (alternative derivation agreement)
- Normal VaR by root finding matches the analytical formula (numerical vs closed-form equivalence)
- Student's t CVaR matches Monte Carlo simulation within 2% (closed-form vs simulation agreement)
- Student's t converges to Normal as nu approaches infinity (theoretical consistency)

### Launch Dashboard
```bash
streamlit run app.py
```

---

> **Bottom Line:** This project doesn't just calculate numbers — it critically evaluates whether those numbers can be trusted, documents where they fail, and implements the mathematically superior alternative (CVaR + Rockafellar-Uryasev) that global regulators are now mandating.
