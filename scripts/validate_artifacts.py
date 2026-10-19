"""Verify saved forecasts, SHAP additivity, fold timing and serialized models.

Run from the repository: python -m scripts.validate_artifacts
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd


def validate(root=Path('results/ml')):
    prices = pd.read_csv('data/raw/prices.csv', index_col=0, parse_dates=True)
    manifests = list(root.glob('*/manifest.json'))
    expected = {t.replace('.NS', '') for t in prices.columns}
    if {p.parent.name for p in manifests} != expected:
        raise AssertionError('Complete artifacts required for every price-series asset')
    total_rows, model_count = 0, 0
    for path in manifests:
        folder = path.parent
        manifest = json.loads(path.read_text())
        X = pd.read_csv(folder / 'features.csv', index_col=0, parse_dates=True)
        ticker = manifest['ticker']
        actual = -np.log(prices[ticker] / prices[ticker].shift(1))
        common = None
        for kind in ('linear', 'lightgbm'):
            pred = pd.read_csv(folder / f'{kind}_predictions.csv', index_col=0, parse_dates=True)
            assert pred.index.is_unique and pred.index.is_monotonic_increasing
            if common is None:
                common = pred.index
            else:
                assert pred.index.equals(common)
            qcols = [f'q_{q:g}' for q in manifest['quantiles']]
            values = pred[qcols].to_numpy()
            assert np.isfinite(values).all() and (np.diff(values, axis=1) >= 0).all()
            np.testing.assert_allclose(actual.reindex(pred.index), pred.Loss, atol=1e-12)
            for a in (.95, .99):
                assert (pred[f'cvar_{a:g}'] + 1e-12 >= pred[f'q_{a:g}']).all()
            folds = pd.read_csv(folder / f'{kind}_folds.csv')
            assert (pd.to_datetime(folds.LastLabel) < pd.to_datetime(folds.TestStart)).all()
            for block, rows in pred.groupby('Block'):
                model = joblib.load(root.parent / 'models' / folder.name / kind / f'block_{block:03}.joblib')
                data = X.loc[pd.DatetimeIndex(pd.to_datetime(rows.Origin))]
                np.testing.assert_allclose(model.predict(data), rows[qcols], atol=1e-8, rtol=1e-7)
                model_count += 1
            total_rows += len(pred)
        shap = pd.read_csv(folder / 'shap_values.csv', index_col=0, parse_dates=True)
        meta = pd.read_csv(folder / 'shap_metadata.csv', index_col=0, parse_dates=True)
        assert shap.index.equals(common) and meta.index.equals(common)
        np.testing.assert_allclose(meta.Base + shap.sum(axis=1), meta.Prediction, atol=1e-7)
        np.testing.assert_allclose(meta.Prediction, pred['q_0.99'], atol=1e-12)
        score = pd.read_csv(folder / 'backtest_results.csv')
        assert len(score) == 10 and score['Valid Days'].eq(len(common)).all()
        for a in (.95, .99):
            for kind in ('normal', 'historical', 'student_t'):
                baseline = pd.read_csv(folder / f'{kind}_{a:g}_predictions.csv', index_col=0, parse_dates=True)
                assert baseline.index.equals(common)
        print(f'{ticker}: {len(common)} matched OOS dates, both models and SHAP verified', flush=True)
    print(json.dumps({'assets': len(manifests), 'ml_forecast_rows': total_rows,
                      'serialized_model_sets_verified': model_count}, indent=2))


if __name__ == '__main__':
    validate()
