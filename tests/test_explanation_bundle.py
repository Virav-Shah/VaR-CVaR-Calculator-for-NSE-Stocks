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
