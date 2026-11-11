> Legacy implementation. Start with the root README and offline demo. Market download failures raise an error; request mock data explicitly.

# Portfolio Optimization

A Python project for portfolio optimization using Modern Portfolio Theory. Downloads stock price data, builds the efficient frontier, and identifies optimal portfolios including maximum Sharpe ratio, maximum weighted Sharpe ratio, and minimum volatility portfolios.

## Features

- Download stock price data using yfinance
- Calculate efficient frontier
- Find maximum Sharpe ratio portfolio
- Find maximum weighted Sharpe ratio portfolio (loads on high individual-asset Sharpes)
- Find minimum volatility portfolio
- Generate random portfolio scatter for visualization
- Create professional matplotlib visualization

## Installation

1. Create a virtual environment:

```bash
python -m venv venv
```

2. Activate the virtual environment:

```bash
# Windows
venv\Scripts\activate

# macOS/Linux
source venv/bin/activate
```

3. Install dependencies:

```bash
pip install -r requirements.txt
```

## Usage

Run with default parameters (8 tech stocks, 2018-2025, 2% risk-free rate):

```bash
python main.py
```

Use mock data for demonstration (useful if network issues occur):

```bash
python main.py --mock
```

Customize parameters:

```bash
python main.py --tickers AAPL,MSFT,AMZN,GOOG --start 2020-01-01 --end 2024-01-01 --rf 0.03
```

### Command Line Arguments

- `--tickers`: Comma-separated list of stock tickers (default: AAPL,MSFT,AMZN,GOOG,META,NVDA,TSLA,JPM)
- `--start`: Start date in YYYY-MM-DD format (default: 2018-01-01)
- `--end`: End date in YYYY-MM-DD format (default: 2025-01-01)
- `--rf`: Annual risk-free rate (default: 0.02)
- `--no-short`: Disable short selling, long-only portfolios (default: True)
- `--seed`: Random seed for reproducible results (default: 42)
- `--mock`: Use mock data instead of downloading from yfinance (useful for testing or when network issues occur)

## Output

The script will:

1. Download and process price data
2. Calculate optimal portfolio weights
3. Print a summary table with returns, volatility, and Sharpe ratios
4. Generate and display a plot showing:
   - Scatter of 25,000 random portfolios colored by Sharpe ratio
   - Efficient frontier curve
   - Three optimal portfolios marked with colored triangles
5. Save the plot as `portfolio_frontier.png`

## Project Structure

```
portfolio_optimization/
├── src/portopt/
│   ├── __init__.py
│   ├── data.py          # Data downloading and preprocessing
│   ├── opt.py           # Portfolio optimization functions
│   └── plot.py          # Visualization functions
├── main.py              # Main script with CLI
├── requirements.txt     # Dependencies
└── README.md           # This file
```
