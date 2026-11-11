import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))
from portopt.data import download_prices, compute_returns, annualize_params


# Test with mock data
tickers = ["AAPL", "MSFT", "NVDA"]
prices = download_prices(tickers, "2020-01-01", "2024-01-01", use_mock=True)
print(f"Downloaded data shape: {prices.shape}")
print("Success!")

