import numpy as np
import pandas as pd
import pytest
from src.portfolio_diagnostics import empirical_es, portfolio_diagnostics, common_stress_days


def test_es_includes_fractional_boundary_mass():
    assert empirical_es([0.,1.,2.,3.],.625) == pytest.approx((3+.5*2)/1.5)
    assert empirical_es([0.,1.,2.,3.],.99) == pytest.approx(3.)


def test_simple_returns_are_aggregated_and_initial_drawdown_counted():
    dates = pd.bdate_range('2020-01-01',periods=3)
    simple = pd.DataFrame({'A':[-.2,.1,.1],'B':[-.1,.05,.05]},index=dates)
    weights = pd.DataFrame({'Equal Weight':[.5,.5],'Other':[1.,0.]},index=['A','B'])
    daily,wealth,dd,metrics = portfolio_diagnostics(np.log1p(simple),weights,.95)
    assert daily.iloc[0,0] == pytest.approx(-.15)
    assert dd.iloc[0,0] == pytest.approx(-.15)
    assert wealth.iloc[-1,0] == pytest.approx(.85*1.075**2)
    assert metrics.loc['Equal Weight','Effective holdings'] == pytest.approx(2)


def test_stress_dates_are_selected_once_from_benchmark():
    daily = pd.DataFrame({'Equal Weight':[-.1,-.2,.1],'Other':[-.8,.4,-.9]},index=pd.bdate_range('2020-01-01',periods=3))
    stress = common_stress_days(daily,count=2)
    assert stress.index.tolist() == daily.index[[1,0]].tolist()
    assert stress.Other.tolist() == [.4,-.8]
