"""Opt-in dashboard integration test; run after generating real ML artifacts."""
from pathlib import Path
import pytest


@pytest.mark.skipif(not Path('results/ml/BHARTIARTL/manifest.json').exists(), reason='Real ML artifacts not generated')
def test_dashboard_all_six_tabs_render():
    from streamlit.testing.v1 import AppTest
    app = AppTest.from_file(str(Path('app.py').resolve()), default_timeout=180).run()
    assert not app.exception
    assert len(app.tabs) == 6
    app.sidebar.selectbox[1].set_value(.95).run()
    assert not app.exception
    if Path('results/ml_v2/manifest.json').exists():
        version = next(s for s in app.selectbox if s.label == 'Explanation model')
        assert version.value == 'Pooled hybrid V2 (calibrated)'
        day = next(s for s in app.selectbox if s.key == 'v2_crash_date')
        day.select_index(1).run()
        assert not app.exception
        next(s for s in app.selectbox if s.label == 'Explanation model').set_value('LightGBM V1').run()
        assert not app.exception
