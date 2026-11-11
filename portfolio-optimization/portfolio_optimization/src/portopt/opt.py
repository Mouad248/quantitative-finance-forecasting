"""Portfolio optimization functions."""
import numpy as np
from scipy.optimize import minimize


def perf(weights, mu, sigma, rf):
    """Calculate portfolio performance metrics."""
    ann_ret = np.dot(weights, mu)
    ann_vol = np.sqrt(np.dot(weights, np.dot(sigma, weights)))
    sharpe = (ann_ret - rf) / ann_vol if ann_vol > 0 else 0
    return ann_ret, ann_vol, sharpe


def max_sharpe(mu, sigma, rf, bounds, constraints):
    """Find maximum Sharpe ratio portfolio."""
    n = len(mu)

    def neg_sharpe(weights):
        ret, vol, sharpe = perf(weights, mu, sigma, rf)
        return -sharpe if vol > 0 else 1e10

    x0 = np.ones(n) / n
    result = minimize(neg_sharpe, x0, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    if result.success:
        return result.x
    else:
        return x0


def max_weighted_sharpe(mu, sigma, rf, bounds, constraints):
    """Find maximum weighted Sharpe ratio portfolio."""
    # Individual asset Sharpe ratios: s_i = (mu_i - rf) / sqrt(sigma_ii)
    s = (mu - rf) / np.sqrt(np.diag(sigma))
    s = np.nan_to_num(s, nan=0.0, posinf=0.0, neginf=0.0)

    def neg_weighted_sharpe(weights):
        return -np.dot(weights, s)

    n = len(mu)
    x0 = np.ones(n) / n
    result = minimize(neg_weighted_sharpe, x0, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    if result.success:
        return result.x
    else:
        return x0


def min_vol(mu, sigma, bounds, constraints):
    """Find minimum volatility portfolio."""
    n = len(mu)

    def portfolio_vol(weights):
        return np.sqrt(np.dot(weights, np.dot(sigma, weights)))

    x0 = np.ones(n) / n
    result = minimize(portfolio_vol, x0, method='SLSQP',
                      bounds=bounds, constraints=constraints)

    if result.success:
        return result.x
    else:
        return x0


def efficient_frontier(mu, sigma, bounds, constraints, n=100):
    """Generate efficient frontier points."""
    # Find min and max return portfolios
    min_vol_weights = min_vol(mu, sigma, bounds, constraints)
    min_ret = np.dot(min_vol_weights, mu)

    # Find maximum return (corner case)
    n_assets = len(mu)
    max_ret_weights = np.zeros(n_assets)
    max_ret_weights[np.argmax(mu)] = 1.0
    max_ret = np.max(mu)

    # Generate target returns
    target_returns = np.linspace(min_ret, max_ret, n)
    frontier_points = []

    for target_ret in target_returns:
        # Add constraint for target return
        ret_constraint = {'type': 'eq', 'fun': lambda w,
                          tr=target_ret: np.dot(w, mu) - tr}
        all_constraints = constraints + [ret_constraint]

        def portfolio_vol(weights):
            return np.dot(weights, np.dot(sigma, weights))

        x0 = np.ones(n_assets) / n_assets
        result = minimize(portfolio_vol, x0, method='SLSQP',
                          bounds=bounds, constraints=all_constraints)

        if result.success:
            weights = result.x
            # Use rf=0 for frontier
            ret, vol, sharpe = perf(weights, mu, sigma, 0)
            frontier_points.append((vol, ret, sharpe))

    # Sort by volatility
    frontier_points.sort(key=lambda x: x[0])
    return frontier_points


def generate_random_portfolios(mu, sigma, rf, bounds, constraints, n_portfolios=25000, seed=42):
    """Generate random feasible portfolios."""
    rng = np.random.default_rng(seed)
    n_assets = len(mu)
    portfolios = []

    attempts = 0
    max_attempts = n_portfolios * 10

    while len(portfolios) < n_portfolios and attempts < max_attempts:
        attempts += 1

        # Generate random weights
        if bounds[0][0] >= 0:  # Long-only case
            weights = rng.random(n_assets)
            weights /= weights.sum()
        else:  # Allow short selling
            weights = rng.uniform(-1, 1, n_assets)
            weights /= np.abs(weights).sum()

        # Check constraints
        valid = True
        for constraint in constraints:
            if constraint['type'] == 'eq':
                if abs(constraint['fun'](weights)) > 1e-6:
                    valid = False
                    break

        if valid:
            ret, vol, sharpe = perf(weights, mu, sigma, rf)
            portfolios.append((vol, ret, sharpe))

    return portfolios

