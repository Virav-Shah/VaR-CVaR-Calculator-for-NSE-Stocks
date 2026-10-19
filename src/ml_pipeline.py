"""Artifact-producing ML run; dashboard reads outputs without retraining."""
from pathlib import Path
from dataclasses import asdict
import hashlib
import json
import platform
import importlib.metadata
import joblib
import numpy as np
import pandas as pd
from src.data_loader import download_market_factors
from src.features import supervised_data, drift_table
from src.ml_var import WalkForwardConfig, QUANTILES, tune_model, walk_forward, validation_splits
from src.backtest import backtest_model, evaluate_forecasts


def run_ml_pipeline(prices, tickers=None, config=None, trials=10, output='results/ml', refresh=False):
    config = config or WalkForwardConfig()
    root = Path(output)
    root.mkdir(parents=True, exist_ok=True)
    factors = download_market_factors(prices, refresh=refresh)
    scores, comparisons = [], []
    for ticker in tickers or list(prices.columns):
        print(f'ML: {ticker}', flush=True)
        folder = root / ticker.replace('.NS', '')
        folder.mkdir(exist_ok=True)
        (folder / 'manifest.json').unlink(missing_ok=True)
        asset_score_start, asset_compare_start = len(scores), len(comparisons)
        X, y, dates = supervised_data(prices, factors, ticker)
        X.to_csv(folder / 'features.csv')
        validation = []
        for block, (tr, va) in enumerate(validation_splits(len(X), config)):
            validation.append(dict(Block=block, Stage='Validation', TrainStart=X.index[tr[0]],
                                   TrainEnd=X.index[tr[-1]], LastLabel=dates.iloc[tr[-1]],
                                   TestStart=X.index[va[0]], TestEnd=X.index[va[-1]], Gap=config.gap))
        pd.DataFrame(validation).to_csv(folder / 'validation_folds.csv', index=False)
        frozen = {}
        for kind in ('linear', 'lightgbm'):
            params, trial_table = tune_model(X, y, kind, config, trials)
            frozen[kind] = params
            trial_table.to_csv(folder / f'{kind}_trials.csv', index=False)
            destination = folder / kind
            destination.mkdir(exist_ok=True)
            model_dir = root.parent / 'models' / ticker.replace('.NS', '') / kind
            model_dir.mkdir(parents=True, exist_ok=True)
            attribution, base_rows = [], []
            def save_block(block, model, train, test, pred):
                joblib.dump(model, model_dir / f'block_{block:03}.joblib')
                drift = drift_table(train, test)
                drift.to_csv(destination / f'drift_{block:03}.csv')
                if kind == 'lightgbm':
                    values, bases, sources = model.explain(test)
                    attribution.append(pd.DataFrame(values, index=pred.index, columns=test.columns))
                    base_rows.append(pd.DataFrame({'Base': bases, 'SourceQuantile': sources,
                                                   'Block': block, 'Prediction': pred['q_0.99'].to_numpy()}, index=pred.index))
            pred, folds = walk_forward(X, y, dates, kind, params, config, save_block)
            pred.to_csv(folder / f'{kind}_predictions.csv')
            folds.to_csv(folder / f'{kind}_folds.csv', index=False)
            if attribution:
                pd.concat(attribution).to_csv(folder / 'shap_values.csv')
                pd.concat(base_rows).to_csv(folder / 'shap_metadata.csv')
                aligned = X.loc[pd.DatetimeIndex(pd.to_datetime(pred.Origin))].copy()
                aligned.index = pred.index
                aligned.to_csv(folder / 'shap_features.csv')
            for alpha in (.95, .99):
                score = evaluate_forecasts(pred.Loss, pred[f'q_{alpha:g}'], pred[f'cvar_{alpha:g}'], alpha)
                scores.append(dict(Ticker=ticker, Method=kind, Alpha=alpha, **score))
                comparisons.append(dict(Ticker=ticker, Method=kind, Alpha=alpha,
                                        VaR=pred[f'q_{alpha:g}'].mean(), CVaR=pred[f'cvar_{alpha:g}'].mean(),
                                        Period='Matched OOS mean; ML ES uses flat terminal tail'))
            # This index is common across both ML models.
            oos_dates = pred.index
        series = np.log(prices[ticker] / prices[ticker].shift(1)).dropna()
        for alpha in (.95, .99):
            for kind in ('historical', 'normal', 'student_t'):
                _, var, es = backtest_model(series, 250, alpha, kind)
                var, es = var.reindex(oos_dates), es.reindex(oos_dates)
                if var.isna().any() or es.isna().any():
                    raise ValueError('Baseline missing a matched OOS date')
                pd.DataFrame({'Loss': -series.reindex(oos_dates), 'VaR': var, 'CVaR': es}).to_csv(folder / f'{kind}_{alpha:g}_predictions.csv')
                scores.append(dict(Ticker=ticker, Method=kind, Alpha=alpha,
                                   **evaluate_forecasts(-series.reindex(oos_dates), var, es, alpha)))
                comparisons.append(dict(Ticker=ticker, Method=kind, Alpha=alpha, VaR=var.mean(), CVaR=es.mean(), Period='Matched OOS mean'))
        manifest = dict(ticker=ticker, config=asdict(config), frozen_parameters=frozen,
                        quantiles=QUANTILES.tolist(), feature_columns=list(X.columns),
                        observations=len(X), first_origin=str(X.index[0]), last_origin=str(X.index[-1]),
                        oos_start=str(oos_dates[0]), oos_end=str(oos_dates[-1]),
                        prices_sha256=hashlib.sha256(prices.to_csv().encode()).hexdigest(),
                        factors_sha256=hashlib.sha256(factors.to_csv().encode()).hexdigest(),
                        python=platform.python_version(), packages={p: importlib.metadata.version(p) for p in ['numpy','pandas','scikit-learn','lightgbm','shap','optuna']},
                        es_assumption='Linear quantile interpolation; constant tail above 0.995. Not exact ES.',
                        forecast_timing='At NSE close t predict loss on next equity session; FX lagged one available observation.',
                        limitations=['Retrospectively adjusted prices; no point-in-time vendor vintage.',
                                     'Survivorship-selected universe.', '750 observations contain only ~7.5 expected 99% exceptions.',
                                     'Five-session purge separates labels, not all 63-day feature history.',
                                     'PSI is a diagnostic heuristic, not an automatic trading decision.'])
        pd.DataFrame(scores[asset_score_start:]).to_csv(folder / 'backtest_results.csv', index=False)
        pd.DataFrame(comparisons[asset_compare_start:]).to_csv(folder / 'var_cvar_comparison.csv', index=False)
        (folder / 'manifest.json').write_text(json.dumps(manifest, indent=2))
        # Save after each asset so completed artifacts survive a later failure.
        for name in ('backtest_results.csv', 'var_cvar_comparison.csv'):
            completed = [pd.read_csv(p.parent / name) for p in root.glob('*/manifest.json')]
            pd.concat(completed, ignore_index=True).to_csv(root / name, index=False)
    return pd.DataFrame(scores)
