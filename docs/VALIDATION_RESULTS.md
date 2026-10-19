# Completed ML validation run

Ten stocks; 3,181 matched OOS sessions per stock, 2013-02-07 through 2025-12-31. Both models used ten Optuna trials, 750-row rolling training, a 375-row validation period, five-row purges and 60-row refits. Hyperparameters were frozen before OOS evaluation.

## What the evidence supports

LightGBM improves independence-test pass counts over historical simulation, but its unconditional coverage is worse. The linear model undercovers substantially. These runs do not justify promoting either ML model to production or claiming regulatory compliance. At 99%, Student-t is better calibrated in this dataset. No hyperparameters were changed after inspecting these OOS results.

## 95% confidence

| Model | Mean pinball loss | Mean breach rate | Mean VaR | Kupiec p > .05 | Independence p > .05 |
|---|---:|---:|---:|---:|---:|
| historical | 0.001867 | 5.33% | 2.41% | 10/10 | 4/10 |
| lightgbm | 0.001862 | 5.94% | 2.29% | 5/10 | 7/10 |
| linear | 0.002173 | 8.10% | 2.27% | 0/10 | 0/10 |
| normal | 0.001873 | 4.07% | 2.66% | 2/10 | 4/10 |
| student_t | 0.001861 | 4.73% | 2.50% | 10/10 | 2/10 |

Metrics are equal-weight averages across stocks; pass counts are per-stock tests, not pooled regulatory approvals. Correlated stocks and multiple tests limit interpretation.

## 99% confidence

| Model | Mean pinball loss | Mean breach rate | Mean VaR | Kupiec p > .05 | Independence p > .05 |
|---|---:|---:|---:|---:|---:|
| historical | 0.000625 | 1.39% | 4.01% | 5/10 | 7/10 |
| lightgbm | 0.000644 | 1.68% | 3.74% | 3/10 | 8/10 |
| linear | 0.000671 | 2.27% | 3.81% | 0/10 | 5/10 |
| normal | 0.000622 | 1.47% | 3.79% | 2/10 | 7/10 |
| student_t | 0.000622 | 1.03% | 4.32% | 10/10 | 6/10 |

Metrics are equal-weight averages across stocks; pass counts are per-stock tests, not pooled regulatory approvals. Correlated stocks and multiple tests limit interpretation.

## Verification

- 23 tests passed in the isolated Mac environment, including portfolio diagnostics that verify fractional-tail ES, initial-day drawdowns and common stress-date selection.
- All 63,620 ML forecast rows and 1,080 serialized model sets passed artifact validation.
- Every saved model reproduces its corresponding forecast block after reloading; SHAP additivity, target alignment and matched baseline dates pass.
- PyTorch 2.14.1 completed a real MPS forward/backward/optimizer step with direct hardware access. LightGBM and linear regression run on CPU.
- Training began using the existing runtime plus temporary ML dependencies while the isolated environment downloaded. Exact training package versions are recorded per asset; the completed Mac environment subsequently passed tests and all artifact reload checks.

## Portfolio visuals

The portfolio tab now compares tail loss against annualized return, underwater drawdowns, performance on the same ten benchmark stress days, and active weights relative to equal weight. Compounded wealth and full tables remain available in expanders. Portfolio allocations use only pre-2016 data; OOS evaluation uses weighted simple returns. ES is an exact empirical tail average with fractional boundary mass, not a mean over an arbitrarily rounded number of days.

## Files

Canonical matched-date tables: `results/tables/backtest_results.csv` and `results/tables/var_cvar_comparison.csv`. Prior full-history tables are preserved with `_legacy_full_history` suffixes. Models and detailed per-stock forecasts, tuning trials, folds and SHAP are under `results/models/` and `results/ml/`. These bulky local artifacts are Git-ignored.

## Remaining limitations

ML CVaR uses a flat tail beyond .995 and can materially understate ES. The universe is survivorship-selected and vendor adjustments are retrospective. A 750-row fit contains little information about 99% tails. Five-session purging does not make overlapping market histories independent. Transaction costs and market impact are absent. PSI is a review heuristic. Model explanations are associations, not causal attributions.
