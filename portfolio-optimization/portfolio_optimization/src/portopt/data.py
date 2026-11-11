"""Data downloading and preprocessing module."""
import numpy as np
import pandas as pd
from datetime import datetime, timedelta


def generate_mock_data(tickers, start_date, end_date):
    """Generate mock price data for demonstration."""
    start = pd.to_datetime(start_date)
    end = pd.to_datetime(end_date)
    dates = pd.date_range(start=start, end=end, freq='D')
    dates = dates[dates.weekday < 5]  # Remove weekends

    np.random.seed(42)  # For reproducible demo data
    prices = {}

    # Base prices and volatilities for different assets
    base_prices = {'AAPL': 150, 'MSFT': 300, 'AMZN': 3000, 'GOOG': 2500,
                   'META': 200, 'NVDA': 500, 'TSLA': 800, 'JPM': 150}
    volatilities = {'AAPL': 0.25, 'MSFT': 0.22, 'AMZN': 0.30, 'GOOG': 0.25,
                    'META': 0.35, 'NVDA': 0.45, 'TSLA': 0.50, 'JPM': 0.20}

    for ticker in tickers:
        base_price = base_prices.get(ticker, 100)
        vol = volatilities.get(ticker, 0.25)

        # Generate correlated random walk
        n_days = len(dates)
        # ~20% annual return, vol annual
        returns = np.random.normal(0.0008, vol/np.sqrt(252), n_days)
        price_series = [base_price]

        for ret in returns[1:]:
            price_series.append(price_series[-1] * (1 + ret))

        prices[ticker] = price_series

    return pd.DataFrame(prices, index=dates)


def download_prices(tickers, start_date, end_date, use_mock=False):
    """Download adjusted close prices for given tickers."""
    if use_mock:
        print("Using mock data for demonstration...")
        return generate_mock_data(tickers, start_date, end_date)

    try:
        import yfinance as yf
        # Try to download real data
        data = yf.download(tickers, start=start_date,
                           end=end_date, progress=False)

        if data.empty:
            raise RuntimeError("No market data downloaded.")

        if len(tickers) == 1:
            if 'Adj Close' in data.columns:
                prices = data['Adj Close'].to_frame()
                prices.columns = tickers
            else:
                prices = data['Close'].to_frame()
                prices.columns = tickers
        else:
            if 'Adj Close' in data.columns:
                prices = data['Adj Close']
            else:
                prices = data['Close']

        prices = prices.dropna()
        if prices.empty or len(prices) == 0:
            raise RuntimeError("No valid market prices after cleaning.")

        print(f"Successfully downloaded real data for {len([t for t in tickers if t in prices.columns])} tickers")
        return prices

    except Exception as e:
        raise RuntimeError("Market-data download failed; use_mock=True is required for synthetic data.") from e


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
