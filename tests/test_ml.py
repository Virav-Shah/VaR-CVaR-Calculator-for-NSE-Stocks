import numpy as np
import pandas as pd
import pytest
from src.features import build_features, supervised_data, population_stability_index
from src.ml_var import WalkForwardConfig, validation_splits, walk_forward, quantile_cvar, QuantileMLModel
from src.backtest import coverage_statistics, evaluate_forecasts


@pytest.fixture
def market():
    rng = np.random.default_rng(91)
    index = pd.bdate_range('2015-01-01', periods=300)
    prices = pd.DataFrame(np.exp(np.cumsum(rng.normal(0,.02,(300,2)), axis=0))*100, index=index, columns=['A','B'])
    factors = pd.DataFrame(index=index)
    for c in prices:
        factors[f'{c}_high'] = prices[c]*1.01
        factors[f'{c}_low'] = prices[c]*.99
        factors[f'{c}_volume'] = rng.integers(100,1000,len(index))
    factors['vix'] = np.exp(rng.normal(3,.1,len(index)))
    factors['nifty'] = prices.mean(axis=1)
    factors['fx'] = np.exp(np.cumsum(rng.normal(0,.001,len(index))))*80
    return prices, factors


def test_future_changes_do_not_change_features(market):
    prices, factors = market
    old = build_features(prices, factors, 'A')
    p, f = prices.copy(), factors.copy()
    p.iloc[180:] *= 5
    f.iloc[180:] *= 7
    pd.testing.assert_frame_equal(old.iloc[:180], build_features(p, f, 'A').iloc[:180])


def test_targets_are_next_session_and_unclipped(market):
    p, f = market
    p.loc[p.index[200], 'A'] *= .1
    X, y, dates = supervised_data(p,f,'A')
    origin = p.index[199]
    assert dates.loc[origin] == p.index[200]
    assert y.loc[origin] > 2
    assert len(X.columns) == 18


def test_walk_forward_purge_and_holdout(market):
    p,f = market
    X,y,dates = supervised_data(p,f,'A')
    config = WalkForwardConfig(80,40,5,30,20)
    for tr, va in validation_splits(len(X),config):
        assert va[0] - tr[-1] == 6
        assert dates.iloc[tr[-1]] < X.index[va[0]]
        assert va[-1] < config.test_start
    pred, folds = walk_forward(X,y,dates,'linear',{'penalty':.0001},config)
    assert pred.index.equals(pd.DatetimeIndex(dates.iloc[config.test_start:],name='Date'))
    assert (folds.LastLabel < folds.TestStart).all()
    assert np.all(np.diff(pred.iloc[:,:6],axis=1) >= 0)


def test_quantile_integral_accounts_for_unequal_spacing():
    qs = np.array([.95,.96,.97,.98,.99,.995])
    expected = ((.995**2-.95**2)/2 + .005*.995)/.05
    assert quantile_cvar(qs[None,:],.95)[0] == pytest.approx(expected)
    assert quantile_cvar(np.ones((2,6)),.99).tolist() == pytest.approx([1,1])


def test_coverage_extreme_samples_are_stable():
    from scipy.stats import chi2
    zero = coverage_statistics(np.zeros(250),.99)
    assert zero['Kupiec p-value'] == pytest.approx(chi2.sf(-2*250*np.log(.99),1))
    assert np.isnan(zero['Christoffersen p-value'])
    assert np.isfinite(coverage_statistics(np.ones(5000),.99)['LR POF'])


def test_basel_uses_last_actual_window_and_only_99():
    dates = pd.bdate_range('2020-01-01',periods=500)
    losses = pd.Series(np.r_[np.ones(250)*2,np.zeros(250)],index=dates)
    var = pd.Series(1.,index=dates)
    assert evaluate_forecasts(losses,var,alpha=.99)['Basel (Last 250)'] == 'Green'
    assert evaluate_forecasts(losses,var,alpha=.95)['Basel (Last 250)'] == 'N/A'


def test_psi_identical_and_shifted():
    a = np.linspace(0,1,1000)
    assert population_stability_index(a,a) == 0
    assert population_stability_index(a,a+10) > .25
    assert population_stability_index(np.ones(750),np.ones(60)*3) > .25


def test_shap_reconstructs_sorted_quantile(market):
    pytest.importorskip('lightgbm')
    pytest.importorskip('shap')
    p,f = market
    X,y,_ = supervised_data(p,f,'A')
    model = QuantileMLModel('lightgbm',{'n_estimators':8,'min_child_samples':30}).fit(X.iloc[:150],y.iloc[:150])
    values,bases,sources = model.explain(X.iloc[150:160])
    np.testing.assert_allclose(values.sum(axis=1)+bases,model.predict(X.iloc[150:160])[:,4],atol=1e-7)
    assert len(sources) == 10


def test_macro_alignment_and_cache_preserves_prices(market, monkeypatch, tmp_path):
    from src.data_loader import download_market_factors
    p, _ = market
    raw = pd.DataFrame(index=p.index)
    columns = {}
    for t in p.columns:
        for field in ('Open','High','Low','Close','Volume'):
            columns[(field,t)] = np.ones(len(p))*100
    for t in ('^INDIAVIX','^NSEI','INR=X'):
        columns[('Close',t)] = np.arange(len(p),dtype=float)+1
    raw = pd.DataFrame(columns,index=p.index)
    raw.loc[p.index[10:14],('Close','^INDIAVIX')] = np.nan
    monkeypatch.setattr('src.data_loader.yf.download',lambda *a,**kw:raw)
    before = p.copy()
    data = download_market_factors(p,tmp_path/'market.csv')
    assert data.vix.iloc[11] == 10
    assert np.isnan(data.vix.iloc[12])
    assert data.fx.iloc[5] == 5
    pd.testing.assert_frame_equal(p,before)


def test_pipeline_roundtrip_and_matched_dates(market, monkeypatch, tmp_path):
    from src.ml_pipeline import run_ml_pipeline
    import joblib
    p, f = market
    monkeypatch.setattr('src.ml_pipeline.download_market_factors',lambda *a,**kw:f)
    output = tmp_path / 'ml'
    scores = run_ml_pipeline(p,['A'],WalkForwardConfig(150,40,5,60,20),1,str(output))
    assert len(scores) == 10
    assert scores['Valid Days'].nunique() == 1
    folder = output / 'A'
    pred = pd.read_csv(folder/'lightgbm_predictions.csv',index_col=0,parse_dates=True)
    xf = pd.read_csv(folder/'shap_features.csv',index_col=0,parse_dates=True)
    model = joblib.load(tmp_path/'models'/'A'/'lightgbm'/'block_000.joblib')
    np.testing.assert_allclose(model.predict(xf),pred.iloc[:,:6],atol=1e-8)
    assert (folder/'manifest.json').exists()
    assert (folder/'lightgbm'/'drift_000.csv').exists()


def test_minimum_variance_does_not_stop_at_equal_weights():
    from src.portfolio import min_variance_portfolio
    rng = np.random.default_rng(8)
    r = pd.DataFrame(rng.normal(size=(1000,4))*[.005,.01,.02,.04])
    w = min_variance_portfolio(r,w_max=.7)
    equal = np.full(4,.25)
    assert w @ r.cov().to_numpy() @ w < .6*(equal @ r.cov().to_numpy() @ equal)
    assert w.sum() == pytest.approx(1)


def test_coverage_rejects_nonbinary_values():
    with pytest.raises(ValueError):
        coverage_statistics([0,.5,1],.99)


def test_tuning_never_reads_holdout_labels(market):
    from src.ml_var import tune_model
    p, f = market
    X, y, _ = supervised_data(p, f, 'A')
    config = WalkForwardConfig(80,40,5,30,20)
    changed = y.copy()
    changed.iloc[config.test_start:] = 1000
    params_a, trials_a = tune_model(X,y,'linear',config,1)
    params_b, trials_b = tune_model(X,changed,'linear',config,1)
    assert params_a == params_b
    assert trials_a.value.iloc[0] == pytest.approx(trials_b.value.iloc[0])
