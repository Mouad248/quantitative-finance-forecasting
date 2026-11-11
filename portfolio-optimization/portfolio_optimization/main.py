"""Main script for portfolio optimization."""
import argparse
import sys
from pathlib import Path
import numpy as np

# Add src to path
sys.path.insert(0, str(Path(__file__).parent / "src"))


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Portfolio Optimization")
    parser.add_argument("--tickers", type=str,
                        default="AAPL,MSFT,AMZN,GOOG,META,NVDA,TSLA,JPM",
                        help="Comma-separated list of tickers")
    parser.add_argument("--start", type=str, default="2018-01-01",
                        help="Start date (YYYY-MM-DD)")
    parser.add_argument("--end", type=str, default="2025-01-01",
                        help="End date (YYYY-MM-DD)")
    parser.add_argument("--rf", type=float, default=0.02,
                        help="Risk-free rate (annual)")
    parser.add_argument("--no-short", action="store_true", default=True,
                        help="Disable short selling")
    parser.add_argument("--seed", type=int, default=42,
                        help="Random seed")
    parser.add_argument("--mock", action="store_true", default=False,
                        help="Use mock data instead of downloading from yfinance")
    return parser.parse_args()


def main():
    """Main execution function."""
    from portopt.opt import (max_sharpe, max_weighted_sharpe, min_vol, efficient_frontier,
                             generate_random_portfolios, perf)
    from portopt.plot import plot_efficient_frontier
    from portopt.data import download_prices, compute_returns, annualize_params

    args = parse_args()

    # Parse tickers
    tickers = [t.strip() for t in args.tickers.split(",")]
    print(f"Optimizing portfolio for: {tickers}")
    print(f"Period: {args.start} to {args.end}")
    print(f"Risk-free rate: {args.rf:.2%}")

    # Download and process data
    print("\nDownloading price data...")
    prices = download_prices(tickers, args.start, args.end, use_mock=args.mock)
    
    # Check if we have valid data
    if prices.empty or len(prices) == 0:
        print("No valid price data available. Falling back to mock data...")
        prices = download_prices(tickers, args.start, args.end, use_mock=True)
    
    returns = compute_returns(prices)
    mu, sigma = annualize_params(returns)

    print(f"Data shape: {prices.shape}")
    if len(prices) > 0:
        print(f"Date range: {prices.index[0].date()} to {prices.index[-1].date()}")
    else:
        print("No valid date range available")

    # Set up constraints
    n = len(tickers)
    if args.no_short:
        bounds = [(0, 1) for _ in range(n)]
    else:
        bounds = [(-1, 1) for _ in range(n)]

    constraints = [{'type': 'eq', 'fun': lambda w: np.sum(w) - 1}]

    # Find optimal portfolios
    print("\nOptimizing portfolios...")
    max_sharpe_weights = max_sharpe(mu, sigma, args.rf, bounds, constraints)
    max_weighted_sharpe_weights = max_weighted_sharpe(
        mu, sigma, args.rf, bounds, constraints)
    min_vol_weights = min_vol(mu, sigma, bounds, constraints)

    # Calculate performance
    max_sharpe_perf = perf(max_sharpe_weights, mu, sigma, args.rf)
    max_weighted_sharpe_perf = perf(
        max_weighted_sharpe_weights, mu, sigma, args.rf)
    min_vol_perf = perf(min_vol_weights, mu, sigma, args.rf)

    # Generate efficient frontier and random portfolios
    print("Generating efficient frontier...")
    frontier_points = efficient_frontier(mu, sigma, bounds, constraints, n=100)

    print("Generating random portfolios...")
    random_portfolios = generate_random_portfolios(mu, sigma, args.rf, bounds, constraints,
                                                   n_portfolios=25000, seed=args.seed)

    # Print results
    print("\n" + "="*80)
    print("PORTFOLIO OPTIMIZATION RESULTS")
    print("="*80)

    print(
        f"\n{'Portfolio':<25} {'Return':<10} {'Vol':<10} {'Sharpe':<10} {'Weights'}")
    print("-" * 80)

    print(f"{'Maximum Sharpe':<25} {max_sharpe_perf[0]:<10.3f} {max_sharpe_perf[1]:<10.3f} "
          f"{max_sharpe_perf[2]:<10.3f} {dict(zip(tickers, max_sharpe_weights))}")

    print(f"{'Max Weighted Sharpe':<25} {max_weighted_sharpe_perf[0]:<10.3f} "
          f"{max_weighted_sharpe_perf[1]:<10.3f} {max_weighted_sharpe_perf[2]:<10.3f} "
          f"{dict(zip(tickers, max_weighted_sharpe_weights))}")

    print(f"{'Minimum Volatility':<25} {min_vol_perf[0]:<10.3f} {min_vol_perf[1]:<10.3f} "
          f"{min_vol_perf[2]:<10.3f} {dict(zip(tickers, min_vol_weights))}")

    # Create plot
    print("\nGenerating plot...")
    plot_efficient_frontier(random_portfolios, frontier_points,
                            max_sharpe_perf, max_weighted_sharpe_perf, min_vol_perf)

    print("Plot saved as 'portfolio_frontier.png'")


if __name__ == "__main__":
    import numpy as np
    main()
