"""Optional local dashboard checks using the versioned forecast bundle."""
from pathlib import Path
import pytest


@pytest.mark.skipif(not Path('results/dashboard_v2/BHARTIARTL/v1_shap_values.csv.gz').exists(), reason='Dashboard bundle not generated')
def test_dashboard_all_five_tabs_render():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path('app.py').resolve()), default_timeout=180).run()
    assert not app.exception
    assert [tab.label for tab in app.tabs] == ['EDA & Regimes', 'VaR / CVaR', 'Regulatory Backtesting', 'Explainable AI', 'Portfolio Optimization']
    app.sidebar.selectbox[1].set_value(.95).run()
    assert not app.exception
    if Path('results/dashboard_v2/BHARTIARTL/hybrid_calibrated.csv.gz').exists():
        version = next(s for s in app.selectbox if s.label == 'Explanation model')
        assert version.value == 'Pooled hybrid V2 (calibrated)'
        day = next(s for s in app.selectbox if s.key == 'v2_crash_date')
        day.select_index(1).run()
        assert not app.exception
        next(s for s in app.selectbox if s.label == 'Explanation model').set_value('LightGBM V1').run()
        assert not app.exception
        day = next(s for s in app.selectbox if s.key == 'v1_crash_date')
        day.select_index(1).run()
        assert not app.exception
