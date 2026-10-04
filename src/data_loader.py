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
