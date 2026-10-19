"""V1-preserving improvements: python run_hybrid.py [--calibration-only]."""
import argparse
from pathlib import Path
import shutil
from src.hybrid_pipeline import run_hybrid


def export_tables():
    """Small reviewable results can be versioned; model bundles stay local."""
    destination = Path('results/tables')
    destination.mkdir(parents=True,exist_ok=True)
    for source,target in [('version_comparison.csv','v1_v2_comparison.csv'),
                          ('headline_summary.csv','v1_v2_summary.csv')]:
        shutil.copyfile(Path('results/ml_v2')/source,destination/target)
    # A calibration-only first run has no completed V2 forecast/SHAP bundle yet.
    if Path('results/ml_v2/manifest.json').exists():
        from scripts.package_dashboard import package
        package()
    else:
        print('Comparison tables exported. Run the full V2 pipeline to refresh the dashboard bundle.', flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--trials',type=int,default=10)
    parser.add_argument('--calibration-only',action='store_true')
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('trials must be positive')
    run_hybrid(args.trials,args.calibration_only)
    export_tables()


if __name__ == '__main__':
    main()
