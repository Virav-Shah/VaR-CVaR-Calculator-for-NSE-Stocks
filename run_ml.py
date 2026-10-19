"""Run reproducible ML results: python run_ml.py --trials 10."""
import argparse
from src.data_loader import download_and_clean_data
from src.ml_pipeline import run_ml_pipeline
from src.ml_var import WalkForwardConfig


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--tickers', nargs='+')
    parser.add_argument('--trials', type=int, default=10)
    parser.add_argument('--refresh', action='store_true')
    parser.add_argument('--train-size', type=int, default=750)
    parser.add_argument('--validation-size', type=int, default=375)
    parser.add_argument('--refit-every', type=int, default=60)
    args = parser.parse_args()
    if args.trials < 1:
        parser.error('--trials must be positive')
    prices = download_and_clean_data()
    if args.tickers and not set(args.tickers).issubset(prices.columns):
        parser.error('Unknown ticker')
    run_ml_pipeline(prices, args.tickers, WalkForwardConfig(train_size=args.train_size,
                    validation_size=args.validation_size, refit_every=args.refit_every), args.trials, refresh=args.refresh)


if __name__ == '__main__':
    main()
