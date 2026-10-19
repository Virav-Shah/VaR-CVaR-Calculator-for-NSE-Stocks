"""Package only forecast and explanation outputs needed by Streamlit.

No training models or tuning data are published in this compact bundle.
"""
from pathlib import Path
import gzip
import shutil
import pandas as pd


def gzip_copy(source, destination):
    with open(source, 'rb') as src, gzip.open(destination, 'wb', compresslevel=9) as dst:
        shutil.copyfileobj(src, dst)


def package(source=Path('results/ml_v2'), destination=Path('results/dashboard_v2')):
    destination.mkdir(parents=True, exist_ok=True)
    for filename in ('version_comparison.csv', 'headline_summary.csv'):
        original = source / filename
        if not original.exists():
            raise FileNotFoundError(f'Run the V2 pipeline first: {original}')
        shutil.copyfile(original, destination / filename)
    v1_root = Path('results/ml')
    for folder in sorted(p for p in source.iterdir() if p.is_dir()):
        output = destination / folder.name
        output.mkdir(exist_ok=True)
        for filename in ('hybrid_calibrated.csv', 'shap_values.csv', 'shap_metadata.csv'):
            original = folder / filename
            if not original.exists():
                raise FileNotFoundError(f'Missing dashboard forecast artifact: {original}')
            gzip_copy(original, output / f'{filename}.gz')
        v1 = pd.read_csv(v1_root/folder.name/'lightgbm_predictions.csv', index_col=0, parse_dates=True)
        v1[['Loss','q_0.95','q_0.99']].to_csv(output/'v1_forecasts.csv')
        gzip_copy(output/'v1_forecasts.csv', output/'v1_forecasts.csv.gz')
        (output/'v1_forecasts.csv').unlink()
        baseline = pd.DataFrame(index=v1.index, data={'Loss':v1.Loss})
        for alpha in (.95,.99):
            b = pd.read_csv(v1_root/folder.name/f'student_t_{alpha:g}_predictions.csv', index_col=0, parse_dates=True)
            baseline[f'VaR_{alpha:g}'] = b.VaR.reindex(v1.index)
        baseline.to_csv(output/'student_t_forecasts.csv')
        gzip_copy(output/'student_t_forecasts.csv', output/'student_t_forecasts.csv.gz')
        (output/'student_t_forecasts.csv').unlink()
    print(f'Packaged dashboard artifacts for {len(list(destination.glob("*/hybrid_calibrated.csv.gz")))} assets in {destination}', flush=True)


if __name__ == '__main__':
    package()
