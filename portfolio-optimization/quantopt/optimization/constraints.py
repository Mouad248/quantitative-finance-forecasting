import numpy as np
import pandas as pd
from typing import List, Tuple, Dict, Any, Optional


class ConstraintSet:
    """
    Fluent API for building scipy.optimize bounds and constraints matrices.

    Parameters
    ----------
    n_assets : int
        Number of assets in the optimization universe.

    Usage
    -----
    >>> cs = (ConstraintSet(n_assets)
    >>>       .long_only()
    >>>       .sum_to_one()
    >>>       .max_position(0.20))
    >>> bounds = cs.bounds()
    >>> constraints = cs.constraints()
    """

    def __init__(self, n_assets: int) -> None:
        self.n_assets = n_assets
        # Default infinite bounds
        self._bounds_list = [(None, None) for _ in range(n_assets)]
        self._constraints_list: List[Dict[str, Any]] = []

    def long_only(self) -> "ConstraintSet":
        """
        Set all lower bounds to 0, upper bounds to 1.
        """
        for i in range(self.n_assets):
            lb = self._bounds_list[i][0]
            ub = self._bounds_list[i][1]
            new_lb = 0.0 if lb is None else max(0.0, lb)
            new_ub = 1.0 if ub is None else min(1.0, ub)
            self._bounds_list[i] = (new_lb, new_ub)
        return self

    def long_short(self, gross_exposure: float = 1.0, net_exposure: float = 0.0) -> "ConstraintSet":
        """
        Allow negative weights constraint for long/short portfolios.
        
        NOTE: |w_i| is non-differentiable. A strict implementation requires
        adding N auxiliary variables (t_1...t_N) to linearize the absolute value:
        w_i <= t_i
        -w_i <= t_i
        sum(t_i) <= gross_exposure
        
        Because scipy.optimize.minimize requires the objective function to handle
        the expanded variable vector, we implement this here as a simplified constraint 
        that evaluates the non-differentiable form. This works reasonably well with SLSQP
        for small to medium universes, but true linearization would require expanding
        the solver state in EfficientFrontier. We document this approximation.
        """
        # Ensure unbounded individual weights
        for i in range(self.n_assets):
            self._bounds_list[i] = (-1.0, 1.0)
            
        def gross_exp_constraint(w: np.ndarray) -> float:
            # sum(|w|) <= gross_exposure -> gross_exposure - sum(|w|) >= 0
            return float(gross_exposure - np.sum(np.abs(w)))

        def net_exp_constraint(w: np.ndarray) -> float:
            # sum(w) = net_exposure
            return float(np.sum(w) - net_exposure)

        self._constraints_list.append(
            {"type": "ineq", "fun": gross_exp_constraint}
        )
        self._constraints_list.append(
            {"type": "eq", "fun": net_exp_constraint}
        )
        return self

    def max_position(self, max_weight: float) -> "ConstraintSet":
        """
        Set upper bound for each asset to max_weight.
        """
        for i in range(self.n_assets):
            lb = self._bounds_list[i][0]
            ub = self._bounds_list[i][1]
            new_ub = max_weight if ub is None else min(max_weight, ub)
            self._bounds_list[i] = (lb, new_ub)
        return self

    def min_position(self, min_weight: float, active_only: bool = True) -> "ConstraintSet":
        """
        Set soft lower bounds. If active_only=True, this behaves as a condition:
        if w_i > 0, then w_i >= min_weight.
        Note: SLSQP deals poorly with discontinuous conditionals. The robust way is
        mixed-integer programming, which we omit. Thus we implement this via bound
        if active_only=False, and as a soft penalty otherwise (omitted here for pure bounds).
        """
        if not active_only:
            for i in range(self.n_assets):
                lb = self._bounds_list[i][0]
                ub = self._bounds_list[i][1]
                new_lb = min_weight if lb is None else max(min_weight, lb)
                self._bounds_list[i] = (new_lb, ub)
        # If active_only=True, requires MINLP logic.
        return self

    def sum_to_one(self, tol: float = 1e-8) -> "ConstraintSet":
        """
        Add equality constraint: sum w_i = 1.
        """
        # Provide gradient for the constraint: jacobian of sum(w) is all 1s
        def sum_eq(w: np.ndarray) -> float:
            return float(np.sum(w) - 1.0)
            
        def sum_jac(w: np.ndarray) -> np.ndarray:
            return np.ones_like(w)
            
        self._constraints_list.append(
            {"type": "eq", "fun": sum_eq, "jac": sum_jac}
        )
        return self

    def sector_neutral(
        self,
        sector_map: Dict[str, List[int]],
        max_deviation: float = 0.05,
        benchmark_weights: Optional[np.ndarray] = None,
    ) -> "ConstraintSet":
        """
        Add sector neutrality constraints.
        
        Parameters
        ----------
        sector_map : Dict[str, List[int]]
            Map of sector name to list of integer column indices in the returns matrix.
        max_deviation : float
            Maximum allowable deviation from benchmark.
        benchmark_weights : Optional[np.ndarray]
            Benchmark weight vector.
        """
        n_sectors = len(sector_map)
        
        for s_name, indices in sector_map.items():
            if benchmark_weights is not None:
                b_s = float(np.sum(benchmark_weights[indices]))
            else:
                b_s = 1.0 / n_sectors
                
            idx_array = np.array(indices, dtype=int)
            
            # Constraint 1: sum(w_s) <= b_s + dev -> (b_s + dev) - sum(w_s) >= 0
            def sector_ub(w: np.ndarray, s_idx=idx_array, target=b_s) -> float:
                return float((target + max_deviation) - np.sum(w[s_idx]))
                
            # Constraint 2: sum(w_s) >= b_s - dev -> sum(w_s) - (b_s - dev) >= 0
            def sector_lb(w: np.ndarray, s_idx=idx_array, target=b_s) -> float:
                return float(np.sum(w[s_idx]) - (target - max_deviation))
                
            self._constraints_list.append({"type": "ineq", "fun": sector_ub})
            self._constraints_list.append({"type": "ineq", "fun": sector_lb})
            
        return self

    def max_turnover(
        self,
        current_weights: np.ndarray,
        max_one_way_turnover: float,
    ) -> "ConstraintSet":
        """
        Add max turnover constraint relative to current_weights.
        
        NOTE: |w - w_c| is non-differentiable. Evaluated as a soft inequality.
        """
        def turnover_ineq(w: np.ndarray) -> float:
            # sum(|w - w_c|) / 2 <= max_turnover
            # max_turnover - sum(|w - w_c|)/2 >= 0
            turnover = np.sum(np.abs(w - current_weights)) / 2.0
            return float(max_one_way_turnover - turnover)
            
        self._constraints_list.append({"type": "ineq", "fun": turnover_ineq})
        return self

    def factor_exposure(
        self,
        factor_loadings: np.ndarray,
        min_exposure: Optional[np.ndarray] = None,
        max_exposure: Optional[np.ndarray] = None,
    ) -> "ConstraintSet":
        """
        Constrain portfolio factor exposures: min_exp <= B^T * w <= max_exp.
        
        factor_loadings: N x K matrix
        min_exposure: K vector
        max_exposure: K vector
        """
        # B^T * w is K vector
        B_T = factor_loadings.T
        
        if max_exposure is not None:
            # B^T * w <= max_exp -> max_exp - B^T * w >= 0
            def factor_ub(w: np.ndarray) -> np.ndarray:
                return max_exposure - B_T @ w
                
            def factor_ub_jac(w: np.ndarray) -> np.ndarray:
                return -B_T
                
            self._constraints_list.append(
                {"type": "ineq", "fun": factor_ub, "jac": factor_ub_jac}
            )
            
        if min_exposure is not None:
            # B^T * w >= min_exp -> B^T * w - min_exp >= 0
            def factor_lb(w: np.ndarray) -> np.ndarray:
                return (B_T @ w) - min_exposure
                
            def factor_lb_jac(w: np.ndarray) -> np.ndarray:
                return B_T
                
            self._constraints_list.append(
                {"type": "ineq", "fun": factor_lb, "jac": factor_lb_jac}
            )
            
        return self

    def bounds(self) -> List[Tuple[Optional[float], Optional[float]]]:
        """
        Return the scipy bounds tuple list.
        If no bounds set, returns default long_only (0,1).
        """
        # Check if all bounds are still (None, None)
        all_none = all(b[0] is None and b[1] is None for b in self._bounds_list)
        if all_none:
            self.long_only()
            
        return self._bounds_list

    def constraints(self) -> List[Dict[str, Any]]:
        """
        Return the list of scipy constraint dicts.
        """
        return self._constraints_list
