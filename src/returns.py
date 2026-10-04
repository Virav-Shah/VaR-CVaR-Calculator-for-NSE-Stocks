import pandas as pd
import numpy as np

def calculate_log_returns(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate log returns from prices: r_t = ln(P_t / P_{t-1})
    """
    log_returns = np.log(prices / prices.shift(1))
    return log_returns.dropna(how='all')

def calculate_losses(log_returns: pd.DataFrame) -> pd.DataFrame:
    """
    Calculate losses: loss = -return
    VaR and CVaR are reported as positive numbers.
    """
    return -log_returns

def prepare_data(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Returns the losses dataframe for the assets.
    """
    log_returns = calculate_log_returns(prices)
    losses = calculate_losses(log_returns)
    return losses
