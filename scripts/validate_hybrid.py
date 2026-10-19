"""Replay saved V2 models and audit normalization, timing, calibration and SHAP.

Run from repository root: python -m scripts.validate_hybrid
"""
from pathlib import Path
import json
import joblib
import numpy as np
import pandas as pd
from src.hybrid_pipeline import digest, summarize
from src.hybrid_var import calibrate_frame, panel_slice


def validate(root=Path('results/ml_v2')):
    manifest = json.loads((root/'manifest.json').read_text())
    assert digest('data/raw/prices.csv') == manifest['prices_sha256']
    assert digest(root/'panel.csv') == manifest['panel_sha256']
    assert all(digest(p)==h for p,h in manifest['source_hashes'].items())
    assert all(digest(p)==h for p,h in json.loads((root/'v1_snapshot.json').read_text()).items())
    panel = pd.read_csv(root/'panel.csv',parse_dates=['Origin','TargetDate']).set_index(['Origin','Ticker'])
    prices = pd.read_csv('data/raw/prices.csv',index_col=0,parse_dates=True)
    dates = panel.index.get_level_values('Origin').unique()
    assert panel.groupby(level='Origin').size().eq(len(prices.columns)).all()
    folds = pd.read_csv(root/'pooled_folds.csv')
    assert folds.TrainDates.eq(750).all() and folds.TrainRows.eq(7500).all()
    assert (pd.to_datetime(folds.LastLabel)<pd.to_datetime(folds.TestStart)).all()
    predictions = []
    for fold in folds.itertuples():
        test = panel_slice(panel,dates[(dates>=pd.Timestamp(fold.TestStart)) & (dates<=pd.Timestamp(fold.TestEnd))])
        model = joblib.load(root.parent/'models_v2'/f'block_{fold.Block:03}.joblib')
        assert model.features == manifest['features']
        replay = pd.DataFrame(model.predict(test),index=test.index,columns=['q_0.95','q_0.99'])
        replay['TargetDate'] = test.TargetDate
        predictions.append(replay)
    replay = pd.concat(predictions)
    rows = 0
    for ticker in prices:
        folder = root/ticker.replace('.NS','')
        raw = pd.read_csv(folder/'hybrid_raw.csv',index_col=0,parse_dates=True)
        calibrated = pd.read_csv(folder/'hybrid_calibrated.csv',index_col=0,parse_dates=True)
        actual = -np.log(prices[ticker]/prices[ticker].shift(1))
        np.testing.assert_allclose(raw.Loss,actual.loc[raw.index],atol=1e-12)
        origin = pd.DatetimeIndex(pd.to_datetime(raw.Origin))
        scale = actual.rolling(20).std().loc[origin]
        np.testing.assert_allclose(raw.Scale,scale,atol=1e-12)
        regenerated = replay.xs(ticker,level='Ticker').set_index('TargetDate')
        assert regenerated.index.equals(raw.index)
        np.testing.assert_allclose(regenerated,raw[['q_0.95','q_0.99']],atol=1e-8)
        expected = calibrate_frame(raw)
        for a in (.95,.99):
            for prefix in ['q','cvar','buffer','order_adjustment']:
                np.testing.assert_allclose(expected[f'{prefix}_{a:g}'],calibrated[f'{prefix}_{a:g}'],atol=1e-12,equal_nan=True)
            assert calibrated[f'q_{a:g}'].iloc[:250].isna().all()
            valid = calibrated.iloc[250:]
            assert np.isfinite(valid[[f'q_{a:g}',f'cvar_{a:g}']]).all().all()
            assert (valid[f'cvar_{a:g}'] >= valid[f'q_{a:g}']-1e-12).all()
        assert (valid['q_0.99']>=valid['q_0.95']).all()
        assert (valid['cvar_0.99']>=valid['cvar_0.95']).all()
        sv = pd.read_csv(folder/'shap_values.csv',index_col=0,parse_dates=True)
        meta = pd.read_csv(folder/'shap_metadata.csv',index_col=0,parse_dates=True)
        assert sv.index.equals(raw.index) and meta.index.equals(raw.index)
        native = meta.StudentBase+meta.TreeBase+sv.sum(axis=1)
        np.testing.assert_allclose(native,raw['native_q_0.99'],atol=1e-7)
        complete = native+meta.RawOrderAdjustment+meta.CalibrationBuffer+meta.FinalOrderAdjustment
        np.testing.assert_allclose(complete,calibrated['q_0.99'],atol=1e-7,equal_nan=True)
        rows += len(raw)
        print(f'{ticker}: {len(raw)} forecasts; {len(valid)} calibrated dates verified',flush=True)
    scores = pd.read_csv(root/'version_comparison.csv')
    assert len(scores)==100 and scores['Valid Days'].nunique()==1
    assert scores.groupby(['Ticker','Alpha']).Version.nunique().eq(5).all()
    pd.testing.assert_frame_equal(summarize(scores),pd.read_csv(root/'headline_summary.csv'),check_exact=False)
    print(json.dumps({'assets':len(prices.columns),'raw_forecast_rows':rows,'pooled_model_sets':len(folds),'status':'passed'},indent=2))


if __name__=='__main__':
    validate()
