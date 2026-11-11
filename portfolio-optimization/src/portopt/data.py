"""Data downloading and preprocessing module."""
import numpy as np
import pandas as pd


def download_prices(tickers, start_date, end_date):
    """Download adjusted close prices for given tickers."""
    try:
        import yfinance as yf
        data = yf.download(tickers, start=start_date, end=end_date, progress=False)
        if data.empty:
            raise RuntimeError("No market data downloaded.")
        prices = data['Adj Close']
        if isinstance(prices, pd.Series):
            prices = prices.to_frame(name=tickers[0])
        prices = prices.dropna()
        if prices.empty:
            raise RuntimeError("No valid adjusted prices after cleaning.")
        return prices
    except Exception as exc:
        raise RuntimeError("Market-data download failed; no synthetic prices were substituted.") from exc



def compute_returns(prices):
    """Compute daily log returns from prices."""
    return np.log(prices / prices.shift(1)).dropna()


def annualize_params(returns, trading_days=252):
    """Compute annualized mean returns and covariance matrix."""
    daily_returns = returns.mean()
    daily_cov = returns.cov()

    mu = daily_returns * trading_days
    sigma = daily_cov * trading_days

    return mu.values, sigma.values

