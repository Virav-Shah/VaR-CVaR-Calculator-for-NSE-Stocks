import os
import pandas as pd
import yfinance as yf
import logging

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

TICKERS = [
    "RELIANCE.NS", "TCS.NS", "HDFCBANK.NS", "INFY.NS", "ICICIBANK.NS",
    "SBIN.NS", "ITC.NS", "HINDUNILVR.NS", "LT.NS", "BHARTIARTL.NS"
]
START_DATE = "2006-01-01"
END_DATE = "2026-01-01"

RAW_DATA_PATH = "data/raw/prices.csv"

def download_and_clean_data(tickers=TICKERS, start=START_DATE, end=END_DATE, save_path=RAW_DATA_PATH):
    if os.path.exists(save_path):
        logging.info(f"Data already exists at {save_path}. Skipping download.")
        return pd.read_csv(save_path, index_col=0, parse_dates=True)

    logging.info(f"Downloading data for {tickers} from {start} to {end}")
    
    # Download data
    df = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
    
    if df.empty:
        logging.error("Downloaded dataframe is empty!")
        return df

    # Print first valid date for each ticker
    logging.info("First valid date for each ticker:")
    for ticker in tickers:
        first_valid = df[ticker].first_valid_index()
        logging.info(f"{ticker}: {first_valid.strftime('%Y-%m-%d') if first_valid else 'No data'}")
        if first_valid and first_valid > pd.to_datetime('2006-02-01'):
            logging.warning(f"{ticker} starts after Jan 2006. Consider replacing.")

    # 1. Drop duplicate dates and all-NaN rows
    df = df[~df.index.duplicated(keep='first')]
    df = df.dropna(how='all')
    
    # Calculate returns for flagging
    returns = df.pct_change()
    
    # 3. Flag any daily |return| > 20%
    for ticker in tickers:
        large_returns = returns[ticker][returns[ticker].abs() > 0.20]
        if not large_returns.empty:
            logging.warning(f"Large daily return (>20%) flagged for {ticker}:")
            for date, val in large_returns.items():
                logging.warning(f"  {date.strftime('%Y-%m-%d')}: {val:.2%}")

    # 4. Flag long runs of zero returns (stale prices)
    for ticker in tickers:
        zero_runs = (returns[ticker] == 0).astype(int).groupby((returns[ticker] != 0).cumsum()).cumsum()
        max_run = zero_runs.max()
        if max_run > 5:
            logging.warning(f"Long run of zero returns (max {max_run} days) flagged for {ticker}.")

    # Save to CSV
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    df.to_csv(save_path)
    logging.info(f"Data saved to {save_path}")
    logging.info(f"Shape: {df.shape} (~{df.shape[0]} trading days expected)")
    
    return df

if __name__ == "__main__":
    download_and_clean_data()

MARKET_DATA_PATH = 'data/raw/market_factors.csv'
MACRO_TICKERS = {'vix': '^INDIAVIX', 'nifty': '^NSEI', 'fx': 'INR=X'}


def download_market_factors(prices, save_path=MARKET_DATA_PATH, refresh=False):
    """Adjusted OHLCV on the existing equity calendar; never overwrite prices.csv.

    FX daily bars close after NSE: shift one observation before alignment.
    Macro missing observations are filled for at most two equity sessions.
    """
    from pathlib import Path
    if Path(save_path).exists() and not refresh:
        cached = pd.read_csv(save_path, index_col=0, parse_dates=True)
        return cached.reindex(prices.index)
    start = (prices.index.min() - pd.Timedelta(days=10)).strftime('%Y-%m-%d')
    end = (prices.index.max() + pd.Timedelta(days=1)).strftime('%Y-%m-%d')
    symbols = list(prices.columns) + list(MACRO_TICKERS.values())
    raw = yf.download(symbols, start=start, end=end, auto_adjust=True,
                      progress=False, threads=False)
    if raw.empty:
        raise ValueError('Market download returned no observations; cached closes remain intact.')
    result = pd.DataFrame(index=prices.index)
    for ticker in prices.columns:
        for field in ['Open', 'High', 'Low', 'Close', 'Volume']:
            result[f'{ticker}_{field.lower()}'] = raw[(field, ticker)].reindex(prices.index)
    for name, ticker in MACRO_TICKERS.items():
        s = raw[('Close', ticker)].dropna()
        if s.empty:
            raise ValueError(f'Missing required macro series: {ticker}')
        if name == 'fx':
            s = s.shift(1)
        # Align to the equity calendar, then apply the two-session fill limit.
        result[name] = s.reindex(prices.index).ffill(limit=2)
    required = [f'{t}_{f}' for t in prices.columns for f in ['open', 'high', 'low', 'close', 'volume']]
    missing = [c for c in required if result[c].notna().sum() == 0]
    if missing:
        raise ValueError(f'Missing OHLCV columns: {missing}')
    Path(save_path).parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(save_path)
    return result
