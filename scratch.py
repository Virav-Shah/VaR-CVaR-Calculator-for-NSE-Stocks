import pandas as pd
from src.data_loader import RAW_DATA_PATH, download_and_clean_data
from src.returns import calculate_log_returns
from src.backtest import backtest_model
import os

if not os.path.exists(RAW_DATA_PATH):
    download_and_clean_data()
df = pd.read_csv(RAW_DATA_PATH, index_col=0, parse_dates=True)
returns = calculate_log_returns(df)

tickers = list(returns.columns)
print("Evaluating RELIANCE.NS as an example...")
series = returns["RELIANCE.NS"].dropna()

for method in ["historical", "normal", "student_t"]:
    res, _, _ = backtest_model(series, window=250, alpha=0.99, method=method)
    print(f"--- {method.upper()} ---")
    for k, v in res.items():
        if isinstance(v, float):
            print(f"{k}: {v:.4f}")
        else:
            print(f"{k}: {v}")
