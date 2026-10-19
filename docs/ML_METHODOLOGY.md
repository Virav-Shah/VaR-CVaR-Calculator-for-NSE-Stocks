# Conditional risk upgrade: audit and methodology

## Repository audit

The original repository has ten stocks (not eight), 4,933 daily price rows from
2006-01-02 through 2025-12-31, nine source modules, one test module, a result
generator, a Streamlit page, a scratch runner, and precomputed tables/figures.
The notebook directory is a placeholder. Existing cached closing prices are
preserved. Generated artifacts and virtual environments are not source code.

The original app was a single page, rather than the four-tab implementation
assumed by the proposal. It now has five tabs, including Explainable AI and portfolio analysis. Training details are documented here rather than in a separate lifecycle tab. Static baseline generation remains
available with the existing command.

Corrections to the proposal and original implementation:

- An arithmetic mean on an uneven quantile grid is not ES. We integrate the
  piecewise-linear quantile function, and explicitly assume a flat terminal tail
  from .995 to 1. This finite-grid CVaR estimate can understate true ES. At 99%,
  half the tail probability lies above .995. This uncertainty is material.
- Noncrossing quantiles imply non-strict inequalities; equal quantiles and
  ES = VaR can be mathematically valid. Sorting is applied row-wise.
- StandardScaler is fitted on training data only. sklearn QuantileRegressor
  provides L1 regularization, not L2. Both linear and LightGBM hyperparameters
  are tuned chronologically using the existing pinball-loss implementation.
- Rolling history shared by adjacent features is not automatically leakage.
  The five-observation purge separates train labels and future forecasts;
  it does not remove dependence across all 63-session lookbacks.
- USD/INR daily bars finish after NSE close, so one available FX observation is
  lagged. VIX/NIFTY values are used only after their NSE close. A real deployment
  must additionally verify vendor publication latency.
- VIX is standardized on a trailing window; raw price and volume levels are not
  features. Targets and extreme losses are never winsorized.
- Coverage likelihoods are computed in log space, including zero/all breaches.
  Independence is undefined when one transition state is unobserved. Conditional
  coverage is correspondingly undefined rather than reported as a pass.
- Basel traffic light uses actual exceptions over the latest 250 valid forecasts,
  only at 99%. It is a diagnostic, not regulatory certification.
- Minimum-variance SLSQP uses a tighter convergence tolerance for small daily
  variance objectives; a regression test checks that it improves on equal weights.
- Portfolio evaluation excludes its training cutoff date from test data.
  Wealth curves aggregate constituent simple returns, then compound; LP risk
  scenarios still use the existing log-return approximation.

## Feature and target contract

`src/features.py` produces 18 features at close t: volatility at 5/21/63 sessions,
5/63 volatility ratio, Parkinson 21-session volatility, downside second moment,
63-session skew/kurtosis, absolute/downside return, cross-sectional dispersion,
21-session NIFTY beta/correlation, VIX 63-session log z-score, five-session VIX
log change, volume z-score, lagged FX log return, and five-session momentum.
The target is -log(P[t+1]/P[t]); forecast artifacts are indexed by realization
date, with the original feature date retained separately. Complete-case warmup
and missing features are dropped, never filled from the future. Macro gaps are
forward-filled at most two equity sessions; equities are not forward-filled.

## Validation and frozen holdout

Defaults: 750 usable training rows, five-row purge, 375 validation rows in
125-row walk-forward folds, another five-row purge, then the untouched OOS
period. Both models use identical dates. Rolling refits use the latest 750
eligible rows, a five-row gap, and forecast 60 rows. Missing-data exclusions can
make these rows span more than the nominal number of exchange sessions.

Optuna minimizes validation pinball loss averaged across the six fitted
quantiles. LightGBM early stopping uses validation folds only; the median best
iteration count from the winning trial is frozen for OOS refits. Hyperparameters
are not retuned on the OOS period. Advancing refits may use earlier OOS outcomes
once those outcomes are historical, as in an operational forecast service.

The baseline historical/normal/Student-t models use trailing 250 observations,
then are evaluated only on the ML forecast dates. The model menu should be
judged by all metrics, including average VaR and breach severity; no improvement
or independence-test pass is promised. A 750-row fit has only about 7.5 expected
99% exceedances. Thirty observations per leaf does not resolve extreme-tail
sampling uncertainty. Confidence intervals and alternative tail assumptions
remain important before any deployment.

## Governance

Each asset has a manifest with input hashes, package versions, frozen parameters,
feature schema, window configuration and limitations. Each refit is serialized
under `results/models/`; only load joblib files from trusted local runs. Every
OOS row has a block ID. SHAP is computed using that row's contemporaneous model.
After sorting, the source quantile can differ by row: we explain the selected
source model and save its identity. Additivity is verified numerically. This is
conditional attribution of the selected branch, not a globally smooth explanation
of the sorting operation. SHAP base values are expected model outputs, not
historical unconditional VaR. PSI uses training-derived bins with infinite tails. The former dashboard drift view compared the
latest 60 forecast observations against the training distribution of the model
active at the start of that window, avoiding overlap and unstable one-day drift
reports when the final forecast block was short. That view was removed when the interface was simplified; per-block PSI remains archived.

The dataset is a selected surviving equity universe and retrospectively adjusted
vendor history, not a point-in-time vintage. Cached prices may have a different
adjustment vintage from freshly downloaded OHLCV. These limitations preclude a
claim of production readiness or causal explanation.

## Mac environment

Run `sh scripts/setup_mps.sh`, then `source .venv-mps/bin/activate`.
This creates a separate local environment. LightGBM and sklearn run on CPU;
PyTorch MPS is a distinct accelerator backend and does not accelerate these
estimators. `python scripts/check_mps.py` performs a real quantile-loss forward,
backward, and optimizer step on MPS; it exits unsuccessfully if unavailable.
The smoke test is not an additional trained production neural model.

## Commands

```sh
python -m pip install -r requirements-ml.txt
python -m pytest -q
python run_ml.py --trials 10
# One asset for a faster development run:
python generate_results.py --ml-only --tickers RELIANCE.NS --trials 3
# Full legacy figure generation plus ML and canonical comparison tables:
python generate_results.py --with-ml
streamlit run app.py
```

ML outputs live in `results/ml/` (CSV data, folds, trials, SHAP and manifests),
with model objects in `results/models/`. These bulky local outputs are ignored
by Git. Do not compare legacy full-history metrics with the matched OOS table.
The dashboard supports baselines without ML artifacts and does not train models
on a widget change. Portfolio ML is not fitted; its existing optimization and
baseline analysis remain available.

## Primary references

- [PyTorch MPS](https://docs.pytorch.org/docs/stable/notes/mps.html)
- [LightGBM parameters](https://lightgbm.readthedocs.io/en/stable/Parameters.html)
- [LightGBM early stopping](https://lightgbm.readthedocs.io/en/stable/pythonapi/lightgbm.early_stopping.html)
- [sklearn QuantileRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.QuantileRegressor.html)
- [Basel market risk framework](https://www.bis.org/committees/bcbs/basel-framework/standard/mar)

## Local verification

The Apple Silicon environment was created with Python 3.14 and PyTorch 2.14.1.
A real MPS quantile-loss forward/backward/optimizer test passed with direct
hardware access. A sandboxed process reports MPS unavailable, so repeat the
hardware test in a normal local terminal if needed. The package set is recorded
in `requirements-mps.lock.txt`; that lock is specific to this tested Mac runtime.

The full suite passed 23 tests in this environment, including the six-tab
Streamlit application at both confidence levels. Upstream seaborn and LightGBM
emit deprecation warnings; they do not indicate failed assertions. Once all
assets finish, run `python -m scripts.validate_artifacts` to verify every saved
model against its original forecasts and check SHAP reconstruction, matched
baseline dates, loss targets and chronological fold boundaries.

## Interpreting the revised charts

Portfolio charts use identical OOS dates. The risk/return plot puts daily ES on
the horizontal axis and annualized compound return on the vertical axis. The
underwater plot includes starting capital when calculating prior peaks. The
stress heatmap uses the ten worst equal-weight dates for all portfolios, so
comparisons are date-matched. Active-weight bars use equal weight as zero and
show percentage-point allocation differences. Wealth curves remain optional.

VaR and CVaR are shown as separate endpoints of an interval, never stacked.
Calibration includes a perfect-coverage diagonal and identifies undercoverage.
SHAP importance is sorted horizontally with contributions shown in return units.
