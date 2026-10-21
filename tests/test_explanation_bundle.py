"""Audit deployed V1 explanations using CSVs, without opening the website."""
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
from src.hybrid_dashboard import load_v1_explanation


TICKERS = list(pd.read_csv('data/raw/prices.csv', nrows=0).columns[1:])


@pytest.mark.parametrize('ticker', TICKERS)
def test_packaged_v1_shap_reconstructs_published_forecasts(ticker):
    pred, values, meta = load_v1_explanation(ticker, Path('results/dashboard_v2'))
    assert len(pred) > 0 and pred.index.is_unique and pred.index.is_monotonic_increasing
    assert len(values.columns) == 18
    assert np.isfinite(values.to_numpy()).all()
    np.testing.assert_allclose(meta.Base + values.sum(axis=1), meta.Prediction, atol=1e-7)
    np.testing.assert_allclose(meta.Prediction, pred['q_0.99'], atol=1e-10)


@pytest.mark.parametrize('ticker', TICKERS)
def test_displayed_v2_contributions_reconstruct_final_forecasts(ticker):
    from src.hybrid_dashboard import feature_contributions

    folder = Path('results/dashboard_v2') / ticker.replace('.NS', '')
    values = pd.read_csv(folder / 'shap_values.csv.gz', index_col=0, parse_dates=True)
    meta = pd.read_csv(folder / 'shap_metadata.csv.gz', index_col=0, parse_dates=True)
    pred = pd.read_csv(folder / 'hybrid_calibrated.csv.gz', index_col=0, parse_dates=True)
    total = (meta.StudentBase + meta.TreeBase + values.sum(axis=1)
             + meta.RawOrderAdjustment + meta.CalibrationBuffer + meta.FinalOrderAdjustment)
    available = meta.FinalVaR99.notna()
    np.testing.assert_allclose(total[available], meta.FinalVaR99[available], atol=1e-7)
    np.testing.assert_allclose(meta.FinalVaR99[available], pred.loc[meta.index[available], 'q_0.99'], atol=1e-10)
    # The displayed aggregation must retain negative effects as well as positive ones.
    for day in pred.loc[meta.index[available]].nlargest(10, 'Loss').index:
        shown = feature_contributions(values.loc[day])
        assert len(shown) == 6
        np.testing.assert_allclose(shown.sum(), values.loc[day].sum(), atol=1e-12)
