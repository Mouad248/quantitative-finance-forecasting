"""Plotting functions for portfolio optimization."""
import matplotlib.pyplot as plt
import numpy as np


def plot_efficient_frontier(random_portfolios, frontier_points, max_sharpe_point,
                            max_weighted_sharpe_point, min_vol_point, save_path="portfolio_frontier.png"):
    """Create the portfolio optimization plot."""
    fig, ax = plt.subplots(figsize=(14, 9))

    # Extract data for random portfolios
    if random_portfolios:
        vols, rets, sharpes = zip(*random_portfolios)
        scatter = ax.scatter(vols, rets, c=sharpes,
                             alpha=0.5, s=1, cmap='viridis')
        cbar = plt.colorbar(scatter, ax=ax)
        cbar.set_label('Sharpe', rotation=270, labelpad=20)

    # Plot efficient frontier
    if frontier_points:
        f_vols, f_rets, _ = zip(*frontier_points)
        ax.plot(f_vols, f_rets, 'k--', linewidth=2, marker='o', markersize=3,
                label='Efficient frontier')

    # Plot special portfolios
    if max_sharpe_point:
        ret, vol, sharpe = max_sharpe_point
        ax.scatter(vol, ret, marker='^', color='red', s=200,
                   label=f'Maximum Sharpe ratio (SR={sharpe:.3f})', zorder=5)

    if max_weighted_sharpe_point:
        ret, vol, sharpe = max_weighted_sharpe_point
        ax.scatter(vol, ret, marker='^', color='magenta', s=200,
                   label=f'Maximum weighted Sharpe ratio (SR={sharpe:.3f})', zorder=5)

    if min_vol_point:
        ret, vol, sharpe = min_vol_point
        ax.scatter(vol, ret, marker='^', color='green', s=200,
                   label=f'Minimum volatility (Vol={vol:.3f})', zorder=5)

    # Formatting
    ax.set_xlabel('Annualised volatility')
    ax.set_ylabel('Annualised returns')
    ax.set_title('Portfolio Optimization in Python')
    ax.legend(loc='upper left')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    plt.show()

