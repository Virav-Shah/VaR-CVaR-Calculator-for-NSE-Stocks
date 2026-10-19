# V1 → V2: measured results

The original research run completed locally on 2026-10-19 using cached data ending 2025-12-31. Results and a compact dashboard bundle were subsequently published to GitHub. This report preserves the research checkpoint; the README documents the current presentation and additional development-score evidence.

**Plain-language interpretation:** the pooled AI makes more accurate risk forecasts than the original AI; its safety buffer catches underestimation but often raises the risk limit too much.

## Fair comparison

Ten stocks, 2,931 identical evaluated sessions per stock from **2014-02-11 through 2025-12-31**. Forecasts start on 2013-02-07, but every comparison row excludes the first 250 sessions used to initialize calibration. Therefore the V1 and Student-t counts here differ from the original full-period report. Target breach rates are 5% and 1%. Coverage and independence pass counts use p > .05. Both-pass counts require both tests for the same stock.

### 99% VaR

| Version | Coverage | Independence | Both | Mean breach rate | Mean VaR | Mean pinball loss ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Student-t | 10/10 | 6/10 | 6/10 | 0.986% | 4.308% | 0.00061415 |
| LightGBM V1 | 3/10 | 10/10 | 3/10 | 1.614% | 3.713% | 0.00063715 |
| V1 + calibration | 10/10 | 9/10 | 9/10 | 0.805% | 4.945% | 0.00062719 |
| Pooled hybrid V2 | 9/10 | 10/10 | 9/10 | 1.088% | 4.055% | 0.00058019 |
| V2 + calibration | 10/10 | 9/10 | 9/10 | 0.925% | 4.961% | 0.00063098 |

The pooled hybrid reduces mean pinball loss by 5.53% versus Student-t and 8.94% versus V1. Mean VaR is about 5.88% lower than Student-t, so the improvement does not simply come from inflating the risk limit. These are point estimates, not a significance test of model superiority. INFY still fails coverage: 1.399% breaches, Kupiec p = 0.0407.

Calibrated V2 raises mean VaR about 22.3% relative to raw V2 and increases pinball loss about 8.8%. It passes coverage for all ten stocks, but ITC fails independence (p = 0.0147). It does not dominate raw V2, and its forecast loss is about 2.7% higher than Student-t. The simpler V1-plus-calibration experiment also achieves 9/10 both-pass stocks. Adding model complexity cannot be justified by pass counts alone.

### 95% VaR

| Version | Coverage | Independence | Both | Mean breach rate | Mean VaR | Mean pinball loss ↓ |
|---|---:|---:|---:|---:|---:|---:|
| Student-t | 9/10 | 2/10 | 2/10 | 4.558% | 2.482% | 0.00183337 |
| LightGBM V1 | 7/10 | 6/10 | 4/10 | 5.841% | 2.258% | 0.00183609 |
| V1 + calibration | 10/10 | 8/10 | 8/10 | 4.828% | 2.506% | 0.00181720 |
| Pooled hybrid V2 | 10/10 | 7/10 | 7/10 | 5.049% | 2.386% | 0.00178607 |
| V2 + calibration | 10/10 | 6/10 | 6/10 | 4.933% | 2.532% | 0.00181724 |

Raw V2 has the lowest mean pinball loss here as well, improving 2.58% on Student-t. Calibration reduces its independence passes from seven to six and worsens pinball loss. Keep calibration as a visible experiment rather than making an unconditional claim that the extra layer helps.

## Expected shortfall

The original V1 estimate used a flat tail above .995. V2 extrapolates using a Student-t tail fitted through its 95% and 99% VaR. The mean per-stock realized/predicted ES ratio at 99% changes from 1.299 for V1 to 1.038 for raw V2 (Student-t: 1.053; calibrated V2: 1.045). This is an encouraging diagnostic, **not a formal ES backtest**. Models select different breach days, so these ratios are not a common-event causal comparison. Formula correctness does not establish true tail accuracy.

## What belongs in the portfolio story

The strongest story is an evidence-based model-development cycle: identify V1 undercoverage; pool and normalize data; incorporate a strong statistical starting point; test calibration as an ablation; preserve all versions; explain predictions; and report the residual failures. Do not label any version production ready. Pooled V2 is a promising research challenger, with raw and calibrated variants showing different trade-offs.

V1 outcomes influenced V2 design, so these dates are no longer an untouched final holdout. No reported-period tuning was used, but design selection has still seen the outcomes. Next validation should freeze the design and use genuinely unseen future data, report crisis/subperiod behavior and uncertainty accounting for shared market dates, and test ES separately. The present stock universe and adjusted historical data also do not eliminate survivorship or data-revision bias.

## Implementation and verification

- 54 pooled model bundles (two quantile models each), trained on 750 complete dates / 7,500 rows per refit.
- 31,810 raw forecasts; 29,310 calibrated forecast rows after warm-up.
- Artifact replay passed for every saved pooled model. Actual losses and 20-day volatility match prices; training labels precede forecast origins; calibration and SHAP decompositions reproduce saved outputs.
- V1 input hashes match the pre-experiment snapshot. Original forecasts, models and V1 results remain available.
- At that development checkpoint, 29 automated tests passed, covering calibration causality, Student-t tail recovery, non-crossing outputs, SHAP, backtesting, portfolio diagnostics and the then-six-tab dashboard. The current interface has five tabs; subsequent deployed V1 explanation checks are documented in the README.
- The isolated Mac `.venv-mps` environment is reused. LightGBM trains on CPU; PyTorch MPS support remains available for workloads that actually require it.

See [methodology](V2_METHODOLOGY.md), [per-stock comparison CSV](../results/tables/v1_v2_comparison.csv), [headline CSV](../results/tables/v1_v2_summary.csv) and the local manifest at `results/ml_v2/manifest.json`.
