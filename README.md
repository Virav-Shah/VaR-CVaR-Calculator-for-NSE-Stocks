# NSE Market Risk Analytics

Explainable machine learning for stock risk forecasting and portfolio analysis.

[GitHub repository](https://github.com/Virav-Shah/nse-market-risk-analytics) · [Streamlit application](https://mkxk2evmghkv5rbpugw99x.streamlit.app/)

A reproducible study of next-day downside risk for ten Indian stocks. It compares Student-t with separate-stock LightGBM (V1), then tests a pooled statistical/ML hybrid (V2), rolling calibration and explicit tail extrapolation.

**Main finding:** pooling and a Student-t starting point improve the AI forecasts; calibration improves coverage but can make risk limits unnecessarily wide. The project reports that trade-off instead of claiming production readiness.

## Measured V1 → V2 results

99% VaR; **2,931 identical sessions per stock, 2014-02-11 to 2025-12-31**. All versions exclude the same 250-session calibration warm-up. Counts are stocks with p > 0.05, out of ten.

| Model | Coverage passes | Independence passes | Both pass | Breaches (target 1%) | Mean VaR | Forecast loss ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Student-t baseline | 10 | 6 | 6 | 0.986% | 4.31% | 0.000614 |
| LightGBM V1 | 3 | 10 | 3 | 1.614% | 3.71% | 0.000637 |
| V1 + calibration | 10 | 9 | 9 | 0.805% | 4.95% | 0.000627 |
| Pooled hybrid V2 | 9 | 10 | 9 | 1.088% | 4.05% | 0.000580 |
| V2 + calibration | 10 | 9 | 9 | 0.925% | 4.96% | 0.000631 |

Uncalibrated pooled V2 reduces average quantile forecast loss by **5.5% versus Student-t** and **8.9% versus V1** at 99%. This is a descriptive historical improvement, not a claim of statistically significant superiority. It still fails the coverage test for INFY. Calibrated V2 passes coverage for all stocks but fails independence for ITC and has higher forecast loss than Student-t. At 95%, raw pooled V2 passes coverage for all ten and independence for seven.

V1 results informed this redesign, so V2 is a **retrospective research benchmark**, not an untouched final test. Neither version is approved for production. Read [complete results and limitations](docs/V2_RESULTS.md); compact, versionable evidence is in [the comparison table](results/tables/v1_v2_comparison.csv) and [summary](results/tables/v1_v2_summary.csv). Original V1 results remain in [their validation report](docs/VALIDATION_RESULTS.md).

## What changed

- **Pooled training:** 750 trading dates × 10 stocks, normalized by volatility known at the forecast origin. Stocks share market crashes, so 7,500 rows are not 7,500 independent observations.
- **Hybrid forecasting:** Student-t supplies the starting VaR; two LightGBM quantile models learn normalized adjustments at 95% and 99%. Whole dates stay together in purged chronological splits.
- **Calibration experiment:** a signed adjustment from the preceding 250 normalized forecast errors. Raw and calibrated predictions are both retained. There is no guaranteed pass rate.
- **Expected shortfall:** a Student-t tail matching both risk limits replaces V1’s flat tail above 99.5%. Its location, scale and degrees of freedom are explicit; it remains a model assumption requiring separate validation.
- **Explainability:** top-five signed SHAP contributions on crash/breach dates, with the statistical starting point, calibration and quantile ordering reconciled separately.

See [V2 methodology](docs/V2_METHODOLOGY.md) and [V1 methodology](docs/ML_METHODOLOGY.md).

## Run locally on a Mac

```bash
sh scripts/setup_mps.sh
source .venv-mps/bin/activate
python scripts/check_mps.py
# If V1 artifacts do not yet exist:
python run_ml.py --trials 10
# Fast calibration experiment using saved V1 forecasts:
python run_hybrid.py --calibration-only
# Complete pooled V2 experiment (10 validation trials, 54 walk-forward refits):
python run_hybrid.py --trials 10
streamlit run app.py
```

`python generate_results.py --hybrid-only --trials 10` is an equivalent entry point. The isolated environment supports PyTorch MPS; LightGBM uses CPU because this implementation does not run on Apple MPS. No neural network is added merely to use the GPU.

Training/model artifacts stay in ignored `results/ml/`, `results/models/`, `results/ml_v2/` and `results/models_v2/`. A compact, compressed dashboard bundle with forecast history and SHAP explanations is versioned under `results/dashboard_v2/`; it lets Streamlit work from a clean clone without shipping trained model binaries. Retraining from a clean checkout needs V1 generation first; package versions for the Mac training environment are pinned in `requirements-mps.lock.txt`.

## Dashboard

Five tabs cover market regimes, risk forecasts, backtesting, Explainable AI and portfolio analysis. The VaR/CVaR tab leads with the matched V1/V2 comparison, actual losses and breaches. Explainable AI shows the top five feature contributions for both V1 and V2, using saved SHAP values without requiring the SHAP training library on the web server. Portfolio charts show the return/tail-loss trade-off, drawdowns, losses on common stress dates and allocation changes from equal weights.

For Streamlit Community Cloud, select repository `Virav-Shah/nse-market-risk-analytics`, branch `main`, and entry point `app.py`. Runtime dependencies are in `requirements.txt`; the bundled dashboard runs without offline training or the Mac MPS environment. If the existing deployment retains the former repository name, reconnect it to the renamed repository in Streamlit settings.

To run only the dashboard locally:

```bash
pip install -r requirements.txt
streamlit run app.py
```

## Validation

```bash
python -m pytest tests -q
python -m scripts.validate_artifacts
python -m scripts.validate_hybrid
```

Tests cover temporal leakage safeguards, rolling calibration, tail formulas, quantile ordering, SHAP reconciliation, portfolio diagnostics and dashboard interactions. The V2 artifact audit replays all 54 saved pooled models, checks 31,810 forecast rows, verifies normalization against prices, and confirms V1 input hashes are unchanged.

The [archived original overview](docs/LEGACY_PROJECT_OVERVIEW.md) preserves the earlier baseline narrative. Its historical numbers are not the current matched experiment.
