# Pooled hybrid V2: methodology and audit trail

## Prediction timing and data

At the close of session t, forecast the log loss `-log(P[t+1]/P[t])`. Inputs and the 20-session sample standard deviation of returns end at t. Divide the target and return-sized features by that **origin-date** volatility, never by volatility measured on the target day. The sign convention reports positive numbers for losses. These are log-loss units, approximately percentage losses for small moves.

Use the same ten stocks, feature history, origin dates and V1 temporal design. Each fit uses 750 complete dates and all ten stocks (7,500 rows). Stock indicators permit stock-specific effects; normalized volatility, return and macro features permit shared patterns. This increases observations and diversity, but does not multiply the number of independent market crises by ten. The selected present-day stocks also retain survivorship bias.

The panel has 4,316 origins per stock. Tuning uses the original three purged validation blocks before the first forecast block, ten Optuna trials and normalized pinball loss. The five-session gap separates training labels from validation/test origins. All stocks from a date are assigned together. Hyperparameters and the median early-stopping iteration count are frozen before the 54 rolling out-of-sample refits, spaced 60 dates apart. Each refit again uses the most recent 750 training dates behind the gap. Models are deterministic with seed 42 and two CPU workers.

Selected configuration: depth 2, four leaves, minimum 340 observations per leaf, learning rate 0.047208, 34 trees per quantile. The 31 inputs are 18 existing features (with scale adjustments), two normalized Student-t quantiles, inverse degrees of freedom and ten stock indicators.

## Statistical anchor and residual target

Reproduce the V1 Student-t baseline: maximum-likelihood fit to 250 historical returns, refreshed every five sessions. The inherited degrees-of-freedom rule sets fitted values at or below 2 to 2.1. Fits are rolling and use only known data. Verify every forecast anchor against the saved V1 baseline.

For confidence `a`, train quantile regression on `(next_loss - Student_t_VaR[a]) / origin_volatility`. Reconstruct `raw_VaR[a] = Student_t_VaR[a] + origin_volatility * predicted_residual[a]`. The normalized statistical quantiles also appear as features. This learns a correction around the existing model, rather than fitting a tail limit without a starting point. Predict both 95% and 99% and sort to prevent crossing. Record this ordering adjustment separately.

## Rolling residual calibration

For each stock and confidence, define the realized standardized raw forecast error `e[s] = (loss[s] - raw_VaR[s]) / scale[s]`.

Before predicting date d, take **only the 250 errors preceding d**. Select the order statistic at rank `ceil(251*a)`: rank 239 for 95%, rank 249 for 99%. Add that error quantile multiplied by the current origin volatility to the raw forecast. The implementation shifts errors by one before rolling. Negative buffers are allowed: a consistently excessive forecast can be reduced. Always use raw forecast errors, not errors from the already corrected series. Sort the two adjusted quantiles and record any final ordering change.

This is a rolling, conformal-style residual correction. Financial observations are dependent and the models change over time; an IID conformal guarantee does not apply. It is not the adaptive algorithm or theorem in [Gibbs and Candès, Adaptive Conformal Inference Under Distribution Shift](https://proceedings.neurips.cc/paper_files/paper/2021/hash/0d441de75945e5acbc865406fc9a2559-Abstract.html). A 250-day window is particularly noisy at 99% and can react excessively to a few extreme errors. Passing Kupiec is neither guaranteed nor the optimization objective.

The first 250 raw predictions warm up the calibration. All five comparison versions are evaluated on the remaining identical 2,931 days, rather than comparing a shortened calibrated series to the full V1 history. Calibration does not retrain V1 or V2.

## Expected shortfall beyond the old .995 cap

A fixed multiplier such as `ES = 1.2 * VaR` is not generally correct: the ratio depends on confidence, degrees of freedom and location. Instead, use one Student-t distribution that matches the ordered 95% and 99% quantiles while retaining the degrees of freedom estimated from historical returns.

For standardized Student-t quantiles `z95, z99`, infer:

```
scale = (q99 - q95) / (z99 - z95)
location = q95 - scale * z95
ES[a] = location + scale * pdf_t(z[a]) * (df + z[a]^2) / ((1-a)*(df-1))
```

This handles nonzero location, extrapolates the tail beyond .995 and keeps ES above its corresponding VaR. Equal quantiles imply a zero-scale degenerate limit; this is allowed mathematically and should be monitored. The [SciPy Student-t location/scale documentation](https://docs.scipy.org/doc/scipy-0.19.0/reference/generated/scipy.stats.t.html) specifies the distribution convention.

This is a **parametric ES proxy**, not proof of accurate extreme-loss prediction. Matching two quantiles does not establish the shape of losses farther into the tail, and the symmetric Student-t assumption may miss crash asymmetry. The empirical ES ratio is a diagnostic only; a separate formal ES validation remains outstanding. V1's original capped ES remains unchanged in its raw comparison row.

## Explanations and governance

SHAP explains the native 99% residual tree, converted back into loss units. The dashboard's top five bars can raise or lower the risk limit. Reconcile the entire final prediction as:

```
Student-t anchor + expected residual output + sum(all SHAP contributions)
+ raw ordering adjustment + calibration buffer + final ordering adjustment
```

The buffer is not silently attributed to the AI. SHAP describes model contributions, not causal explanations for crashes. Explanations refer to forecasts made before the displayed realized-loss date.

Coverage and independence passes use p > .05, count undefined tests as not passing and are shown alongside breach rates and pinball loss. A pass is a failure to reject a particular null, not model approval. Ten related stocks, two levels and multiple versions introduce multiple-comparison and dependence concerns. Global tests can also hide poor performance within crisis periods.

V1 test results motivated the V2 design, so this is a retrospective benchmark even though each individual forecast is chronologically valid. No tuning is performed on the reported comparison period, but design selection has already seen its outcomes. Freeze the design before genuinely new future evaluation; validate crisis behavior, ES, data operations and monitoring before any deployment. No model receives a 'production ready' label from these tests alone.

## Reproduction and artifacts

Run `python run_hybrid.py --calibration-only` to produce V1 calibration without refitting V1. Run `python run_hybrid.py --trials 10` for the full experiment. Both use the existing `src/backtest.py` evaluator. `generate_results.py --hybrid-only` dispatches to the same CLI.

`results/ml_v2/` holds cached causal Student-t contexts, pooled panel, tuning trials, fold timing, raw/calibrated forecasts, SHAP values, comparisons and a manifest. `results/models_v2/` holds 54 model bundles, each containing two quantile regressors. The manifest includes frozen parameters, source/input hashes and package versions. A snapshot hashes V1 feature, forecast, backtest and manifest inputs; the pipeline verifies they are unchanged. Small comparison tables are exported to `results/tables/v1_v2_*.csv`, and compressed per-stock comparison/SHAP data to `results/dashboard_v2/` for review and Streamlit deployment. Original V1 outputs stay in `results/ml/` and `results/models/`.

`python -m scripts.validate_hybrid` independently replays saved models, checks prices/volatility, fold boundaries, calibration, ordering, ES bounds, SHAP reconciliation and V1 hashes. Tests include changing future losses and confirming earlier calibration remains unchanged.
