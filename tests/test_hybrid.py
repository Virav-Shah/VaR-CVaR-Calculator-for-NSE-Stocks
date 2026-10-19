import numpy as np
import pandas as pd
import pytest
from scipy.stats import t
from src.hybrid_var import (rolling_error_calibration,student_tail_es,calibrate_frame,
                            PooledHybrid,panel_slice)
from src.hybrid_pipeline import summarize


def test_calibration_is_shifted_and_scale_equivariant():
    index = pd.bdate_range('2020-01-01',periods=300)
    loss = pd.Series(np.arange(300)/10000,index=index)
    raw = pd.Series(.01,index=index)
    scale = pd.Series(.02,index=index)
    forecast,buffer = rolling_error_calibration(loss,raw,scale,.99)
    assert forecast.iloc[:250].isna().all()
    # rank 249 of the previous 250 observations: index 248.
    assert forecast.iloc[250] == pytest.approx(loss.iloc[248])
    altered = loss.copy()
    altered.iloc[250:] = 100
    changed,_ = rolling_error_calibration(altered,raw,scale,.99)
    pd.testing.assert_series_equal(forecast.iloc[:251],changed.iloc[:251])
    scaled,_ = rolling_error_calibration(loss*7,raw*7,scale*7,.99)
    np.testing.assert_allclose(scaled,forecast*7,equal_nan=True)


def test_calibration_uses_raw_error_not_previous_corrected_forecasts():
    index = pd.bdate_range('2020-01-01',periods=300)
    raw = pd.Series(.01,index=index)
    loss = pd.Series(.03,index=index)
    value,_ = rolling_error_calibration(loss,raw,pd.Series(.02,index=index),.99)
    np.testing.assert_allclose(value.iloc[250:],.03)


def test_student_tail_recovers_nonzero_location_and_heavy_tail():
    nu,location,scale = np.array([2.1,4,10,50]), .012,.03
    q95,q99 = location+scale*t.ppf(.95,nu),location+scale*t.ppf(.99,nu)
    e95,e99,loc,s = student_tail_es(q95,q99,nu)
    np.testing.assert_allclose(loc,location)
    np.testing.assert_allclose(s,scale)
    z=t.ppf(.99,nu)
    expected=location+scale*t.pdf(z,nu)*(nu+z*z)/(.01*(nu-1))
    np.testing.assert_allclose(e99,expected)
    assert (e99 >= e95).all() and (e95 >= q95).all() and (e99 >= q99).all()
    a,b,_,s = student_tail_es([.02],[.02],[4])
    assert a[0] == b[0] == .02 and s[0] == 0


def test_non_crossing_calibrated_forecasts_and_es():
    rng=np.random.default_rng(1)
    index=pd.bdate_range('2020-01-01',periods=350)
    f=pd.DataFrame({'Loss':rng.standard_t(4,350)*.02,'q_0.95':.03,'q_0.99':.04,'Scale':.02,'df':4},index=index)
    out=calibrate_frame(f).dropna()
    assert (out['q_0.99']>=out['q_0.95']).all()
    assert (out['cvar_0.99']>=out['q_0.99']).all()
    assert (out['cvar_0.99']>=out['cvar_0.95']).all()


def test_whole_dates_stay_together_and_residual_shap_adds_up():
    rng=np.random.default_rng(8)
    dates=pd.bdate_range('2020-01-01',periods=120)
    idx=pd.MultiIndex.from_product([dates,['A','B']],names=['Origin','Ticker'])
    panel=pd.DataFrame({'feature':rng.normal(size=len(idx)),'Scale':.02,'t_var_0.95':.03,'t_var_0.99':.05,
                        'Loss':rng.standard_t(5,len(idx))*.02},index=idx)
    train=panel_slice(panel,dates[:100]);test=panel_slice(panel,dates[105:])
    assert len(train)==200 and len(test)==30
    assert not set(train.index.get_level_values(0)) & set(test.index.get_level_values(0))
    model=PooledHybrid({'n_estimators':8,'min_child_samples':30}).fit(train,['feature'])
    sv,base=model.explain(test)
    np.testing.assert_allclose(test['t_var_0.99']+base+sv.sum(axis=1),model.predict_native(test)[:,1],atol=1e-7)


def test_governance_never_equates_test_passes_with_production_readiness():
    scores=pd.DataFrame({'Alpha':[.99], 'Version':['V2'], 'Kupiec p-value':[.5], 'Christoffersen p-value':[.5],
                         'Exception Rate (%)':[1.], 'Pinball Loss':[.001],'Mean VaR':[.04],'Mean ES':[.05]})
    result=summarize(scores)
    assert 'independent validation pending' in result.Governance.iloc[0]
    assert result.BothPasses.iloc[0]==1
