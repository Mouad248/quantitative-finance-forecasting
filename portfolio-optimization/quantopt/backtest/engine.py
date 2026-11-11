import logging
import numpy as np
import pandas as pd
from dataclasses import dataclass, field
from typing import Optional, Callable, Dict

from quantopt.analytics.performance import performance_summary
from quantopt.optimization.base import BaseOptimizer

logger = logging.getLogger(__name__)


@dataclass
class TransactionCostModel:
    """
    Transaction cost model incorporating proportional spreads, fixed fees,
    and square-root market impact.
    """
    proportional: float = 0.0010
    fixed: float = 0.0
    market_impact: float = 0.0

    def compute(self, delta_weights: np.ndarray) -> float:
        """
        Compute total transaction costs as a fraction of portfolio AUM.

        Parameters
        ----------
        delta_weights : np.ndarray
            Change in portfolio weights for each asset.

        Returns
        -------
        float
        """
        abs_delta = np.abs(delta_weights)
        trade_volume = np.sum(abs_delta)
        
        if trade_volume == 0:
            return 0.0
            
        proportional_cost = self.proportional * trade_volume
        impact_cost = self.market_impact * np.sum(np.sqrt(abs_delta))
        
        return float(proportional_cost + impact_cost + self.fixed)


@dataclass
class BacktestConfig:
    lookback_days: int = 252
    rebalance_freq: str = "M"
    transaction_cost: TransactionCostModel = field(default_factory=TransactionCostModel)
    max_turnover: Optional[float] = None
    min_weight_threshold: float = 1e-4
    initial_capital: float = 1_000_000.0


@dataclass
class BacktestResult:
    portfolio_returns: pd.Series
    gross_returns: pd.Series
    weights_history: pd.DataFrame
    realized_weights: pd.DataFrame
    turnover_history: pd.Series
    transaction_costs: pd.Series
    rebalance_dates: list[pd.Timestamp]
    summary: pd.DataFrame
    cumulative_returns: pd.Series
    dollar_value: pd.Series


class WalkForwardBacktester:
    """
    Walk-forward active portfolio backtester with mark-to-market drift.
    """

    def __init__(
        self,
        prices: pd.DataFrame,
        optimizer_factory: Callable[[pd.DataFrame], BaseOptimizer],
        config: BacktestConfig,
        benchmark_prices: Optional[pd.Series] = None,
        initial_weights: Optional[pd.Series] = None,
    ) -> None:
        if (not isinstance(prices.index, pd.DatetimeIndex)
                or not prices.index.is_monotonic_increasing or not prices.index.is_unique
                or not prices.columns.is_unique or prices.shape[1] == 0
                or not np.isfinite(prices.to_numpy()).all() or (prices <= 0).any().any()):
            raise ValueError("Prices require unique assets, sorted unique dates and finite positive values.")
        if not 1 <= config.lookback_days < len(prices) - 1:
            raise ValueError("lookback_days must leave at least one evaluation return.")
        if config.max_turnover is not None and not 0 <= config.max_turnover <= 1:
            raise ValueError("max_turnover must be between 0 and 1.")
        if not 0 <= config.min_weight_threshold < 1 or config.initial_capital <= 0:
            raise ValueError("Invalid weight threshold or initial capital.")
        self.prices = prices
        self.optimizer_factory = optimizer_factory
        self.config = config
        self.benchmark_prices = benchmark_prices
        
        # Determine assets and shape
        self.assets = prices.columns
        self.N = len(self.assets)
        self.dates = prices.index
        self.T = len(prices)
        
        if initial_weights is None:
            self.initial_weights = pd.Series(1.0 / self.N, index=self.assets)
        else:
            self.initial_weights = initial_weights.reindex(self.assets).fillna(0.0)
            if (not np.isfinite(self.initial_weights).all()
                    or (self.initial_weights < 0).any() or self.initial_weights.sum() <= 0):
                raise ValueError("Initial weights must be finite, nonnegative and have a positive sum.")
            self.initial_weights /= self.initial_weights.sum()

    def _apply_turnover_constraint(
        self,
        w_old: pd.Series,
        w_new: pd.Series,
        max_turnover: float,
    ) -> pd.Series:
        """Apply proportional shrinkage to meet max turnover constraint."""
        one_way = np.sum(np.abs(w_new - w_old)) / 2.0
        
        if one_way <= max_turnover + 1e-6:
            return w_new
            
        shrinkage = max_turnover / one_way
        w_constrained = w_old + shrinkage * (w_new - w_old)
        return w_constrained / w_constrained.sum()

    def run(self) -> BacktestResult:
        # 1. Compute returns
        log_returns = np.log(self.prices).diff().iloc[1:]
        all_dates = log_returns.index
        
        # 2. Determine rebalance dates
        # Use pandas resample to get business month ends or similar frequencies
        freq = self.config.rebalance_freq
        
        # Pandas 2.1+ requires 'ME' for MonthEnd instead of 'M'
        if freq == "M":
            freq = "ME"
        elif freq == "Q":
            freq = "QE"
        elif freq == "A" or freq == "Y":
            freq = "YE"
            
        # Last observed trading date in each period, including weekend month ends.
        observed_dates = pd.Series(all_dates, index=all_dates)
        valid_rebalance_dates = pd.DatetimeIndex(observed_dates.resample(freq).last().dropna())
        
        # Only start after lookback_days
        start_date = all_dates[self.config.lookback_days]
        rebalance_dates = [d for d in valid_rebalance_dates if d >= start_date]

        # 3. Initialize state tracking
        current_weights = self.initial_weights.copy()
        
        port_ret = pd.Series(np.nan, index=all_dates)
        gross_ret = pd.Series(np.nan, index=all_dates)
        realized_w = pd.DataFrame(np.nan, index=all_dates, columns=self.assets)
        target_w_history = pd.DataFrame(np.nan, index=rebalance_dates, columns=self.assets)
        turnover = pd.Series(0.0, index=rebalance_dates)
        tc_series = pd.Series(0.0, index=rebalance_dates)

        # 4. Main Loop
        # Start iterating from lookback_days forward
        loop_dates = all_dates[self.config.lookback_days:]
        
        # We need historical windows, the index slicing works but positional is safer
        loc_map = {d: i + 1 for i, d in enumerate(all_dates)} 

        for t_date in loop_dates:
            # a. DRIFT WEIGHTS (mark-to-market)
            r_t = log_returns.loc[t_date]
            factor = np.exp(r_t)
            # Earn the day's simple return with opening holdings, before trading.
            gross = float(current_weights @ (factor - 1.0))
            w_drifted = current_weights * factor
            w_drifted /= w_drifted.sum()  # Renormalize
            
            realized_w.loc[t_date] = w_drifted

            # b. REBALANCE
            TC = 0.0
            day_turnover = 0.0
            
            if t_date in rebalance_dates:
                # Observe through today's close; new weights earn returns tomorrow.
                end_loc = loc_map[t_date]
                start_loc = end_loc - self.config.lookback_days
                
                # Slicing the log_returns Series
                window_returns = log_returns.iloc[start_loc:end_loc]
                
                try:
                    # ii. Factory build
                    opt = self.optimizer_factory(window_returns)
                    # iii. Optimize
                    if opt.weights_ is None:
                        opt.optimize()
                    
                    # iv. Threshold cleanup
                    target_weights = opt.clean_weights(threshold=self.config.min_weight_threshold)
                    
                    target_weights = target_weights.reindex(self.assets)
                    if (not np.isfinite(target_weights).all() or (target_weights < 0).any()
                            or not np.isclose(target_weights.sum(), 1.0)):
                        raise ValueError("Backtester requires aligned, fully invested long-only weights.")
                    # v. Turnover constraint
                    if self.config.max_turnover is not None:
                        target_weights = self._apply_turnover_constraint(
                            w_drifted, target_weights, self.config.max_turnover
                        )
                        
                    # vi. & vii. Transaction Costs
                    delta = target_weights - w_drifted
                    TC = self.config.transaction_cost.compute(delta.values)
                    if not 0 <= TC < 1:
                        raise ValueError("Trading cost must be a fraction in [0, 1).")
                    day_turnover = np.sum(np.abs(delta)) / 2.0
                    
                    # viii. Apply
                    current_weights = target_weights
                    
                except Exception as e:
                    logger.warning(f"Rebalance failed on {t_date.date()}: {str(e)}. Holding position.")
                    # Keep drifted weights as current weights
                    current_weights = w_drifted
                    TC = 0.0
                    day_turnover = 0.0
                    
                target_w_history.loc[t_date] = current_weights
                turnover.loc[t_date] = day_turnover
                tc_series.loc[t_date] = TC
            else:
                current_weights = w_drifted

            # Costs are a fraction of wealth after today's market move.
            net = (1.0 + gross) * (1.0 - TC) - 1.0

            gross_ret.loc[t_date] = gross
            port_ret.loc[t_date] = net

        # Cleanup resulting NaNs (from lookback period)
        port_ret = port_ret.dropna()
        gross_ret = gross_ret.dropna()
        realized_w = realized_w.dropna(how='all')
        target_w_history = target_w_history.dropna(how='all')
        
        cum_ret = (1.0 + port_ret).cumprod() - 1.0
        dollar_value = (1.0 + cum_ret) * self.config.initial_capital
        
        # Benchmark logic for summary
        bm = None
        if self.benchmark_prices is not None:
            # Match sizes
            bm = self.benchmark_prices.pct_change(fill_method=None).reindex(port_ret.index)

        # Performance summary
        # Assuming annualization correctly defaults to 252 (daily)
        summary = performance_summary(
            port_ret,
            benchmark=bm,
        )

        return BacktestResult(
            portfolio_returns=port_ret,
            gross_returns=gross_ret,
            weights_history=target_w_history,
            realized_weights=realized_w,
            turnover_history=turnover,
            transaction_costs=tc_series,
            rebalance_dates=list(rebalance_dates),
            summary=summary,
            cumulative_returns=cum_ret,
            dollar_value=dollar_value,
        )

    def run_comparison(
        self,
        strategies: Dict[str, Callable[[pd.DataFrame], BaseOptimizer]],
        config: BacktestConfig,
    ) -> Dict[str, BacktestResult]:
        """Run multiple strategies using the same dataset and configuration."""
        results = {}
        for name, factory in strategies.items():
            tester = WalkForwardBacktester(
                prices=self.prices,
                optimizer_factory=factory,
                config=config,
                benchmark_prices=self.benchmark_prices,
                initial_weights=self.initial_weights,
            )
            results[name] = tester.run()
            # Overwrite summary name
            results[name].summary.columns = [name]
            
        return results
