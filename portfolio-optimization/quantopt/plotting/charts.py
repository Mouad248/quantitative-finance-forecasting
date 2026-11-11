import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Optional, Dict, Tuple


def _ensure_ax(ax: Optional[plt.Axes], figsize: Tuple[float, float] = (10, 6)) -> plt.Axes:
    """Helper to ensure an ax object exists."""
    if ax is None:
        sns.set_theme(style="whitegrid")
        _, ax = plt.subplots(figsize=figsize)
    return ax


def plot_efficient_frontier(
    frontier: pd.DataFrame,
    tangency_point: Optional[Tuple[float, float]] = None,
    risk_free_rate: float = 0.0,
    benchmark_points: Optional[Dict[str, Tuple[float, float]]] = None,
    title: str = "Efficient Frontier",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    ax = _ensure_ax(ax, figsize=(10, 6))
    
    sc = ax.scatter(
        frontier["volatility"],
        frontier["return"],
        c=frontier["sharpe"],
        cmap="viridis",
        marker="o",
        s=40,
        zorder=2,
    )
    plt.colorbar(sc, ax=ax, label="Sharpe Ratio")
    
    if tangency_point is not None:
        vt, rt = tangency_point
        ax.scatter(vt, rt, color="gold", marker="*", s=300, edgecolor="black", zorder=3, label="Max Sharpe")
        
        # Draw CML
        cml_x = np.linspace(0, vt * 1.5, 100)
        cml_y = risk_free_rate + (rt - risk_free_rate) / vt * cml_x
        ax.plot(cml_x, cml_y, color="red", linestyle="--", zorder=1, label="Capital Market Line")

    if benchmark_points is not None:
        for name, (v, r) in benchmark_points.items():
            ax.scatter(v, r, marker="X", s=100, label=name, zorder=3)

    ax.set_title(title)
    ax.set_xlabel("Annualized Volatility")
    ax.set_ylabel("Annualized Return")
    if tangency_point is not None or benchmark_points is not None:
        ax.legend()
        
    return ax


def plot_weights(
    weights: pd.Series,
    title: str = "Portfolio Weights",
    sort: bool = True,
    threshold: float = 0.005,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    ax = _ensure_ax(ax, figsize=(8, max(5, len(weights) * 0.3)))
    
    w = weights[weights.abs() >= threshold].copy()
    if sort:
        w = w.sort_values(ascending=True)
        
    colors = ["#2ca02c" if val >= 0 else "#d62728" for val in w]
    w.plot.barh(ax=ax, color=colors, width=0.7)
    
    ax.axvline(0, color="black", linewidth=1.2)
    ax.set_title(title)
    ax.set_xlabel("Weight")
    
    return ax


def plot_weights_history(
    weights_history: pd.DataFrame,
    title: str = "Portfolio Weights Over Time",
    top_n: int = 10,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    ax = _ensure_ax(ax, figsize=(12, 6))
    
    mean_w = weights_history.mean().sort_values(ascending=False)
    
    if len(mean_w) > top_n:
        top_assets = mean_w.index[:top_n].tolist()
        df_plot = weights_history[top_assets].copy()
        df_plot["Other"] = weights_history.drop(columns=top_assets).sum(axis=1)
    else:
        df_plot = weights_history.copy()
        
    df_plot.plot.area(ax=ax, cmap="tab20", alpha=0.8, linewidth=0)
    
    ax.set_title(title)
    ax.set_ylabel("Weight")
    ax.set_xlim(df_plot.index.min(), df_plot.index.max()) # type: ignore
    ax.legend(loc="center left", bbox_to_anchor=(1.0, 0.5))
    
    return ax


def plot_drawdown(
    returns: pd.Series,
    benchmark_returns: Optional[pd.Series] = None,
    title: str = "Drawdown",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    ax = _ensure_ax(ax, figsize=(10, 5))
    
    def calc_dd(ret):
        cum = (1.0 + ret).cumprod()
        peak = cum.cummax()
        return (cum - peak) / peak

    dd = calc_dd(returns)
    ax.fill_between(dd.index, dd, 0, color="red", alpha=0.3, label="Portfolio Drawdown")
    ax.plot(dd.index, dd, color="darkred", linewidth=1.0)
    
    if benchmark_returns is not None:
        b_dd = calc_dd(benchmark_returns)
        ax.plot(b_dd.index, b_dd, color="C0", linestyle="--", linewidth=1.5, label="Benchmark Drawdown")

    # Annotate max drawdown
    mdd_val = dd.min()
    mdd_idx = dd.idxmin()
    ax.scatter(mdd_idx, mdd_val, color="black", s=50, zorder=5)
    ax.annotate(
        f"{mdd_val:.1%}",
        xy=(mdd_idx, mdd_val), # type: ignore
        xytext=(10, 10),
        textcoords="offset points",
        weight="bold"
    )

    ax.set_title(title)
    ax.set_ylabel("Drawdown")
    ax.legend()
    
    return ax


def plot_cumulative_returns(
    returns_dict: Dict[str, pd.Series],
    risk_free_rate: float = 0.0,
    frequency: int = 252,
    title: str = "Cumulative Returns",
    log_scale: bool = False,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    ax = _ensure_ax(ax, figsize=(12, 6))
    
    for name, rets in returns_dict.items():
        cum = (1.0 + rets).cumprod()
        ax.plot(cum.index, cum, label=name)
        
    ax.axhline(1.0, color="black", linestyle="--", linewidth=1)
    
    # Shade crises if dates overlap
    all_idx = pd.concat(returns_dict.values(), axis=1).index
    min_date = all_idx.min()
    max_date = all_idx.max()

    crisis_periods = [
        (pd.Timestamp("2008-09-01"), pd.Timestamp("2009-03-01"), "GFC"),
        (pd.Timestamp("2020-02-19"), pd.Timestamp("2020-03-23"), "COVID"),
    ]

    for start, end, label in crisis_periods:
        if min_date <= end and max_date >= start:
            ax.axvspan(max(min_date, start), min(max_date, end), color="grey", alpha=0.2)
            
    if log_scale:
        ax.set_yscale("log")
        ax.set_ylabel("Cumulative Return (Log Scale)")
    else:
        ax.set_ylabel("Cumulative Return")
        
    ax.set_title(title)
    ax.legend()
    
    return ax


def plot_risk_contributions(
    weights: pd.Series,
    Sigma: pd.DataFrame,
    title: str = "Risk Contributions vs Weights",
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    ax = _ensure_ax(ax, figsize=(10, len(weights) * 0.35))
    
    w = weights.reindex(Sigma.index).fillna(0.0)
    vol = np.sqrt(w.values @ Sigma.values @ w.values)
    
    if vol == 0:
        rc = pd.Series(0.0, index=w.index)
    else:
        mrc = (Sigma.values @ w.values) / vol
        rc = w * mrc
        rc = rc / rc.sum()
        
    df = pd.DataFrame({"Weight": w, "Risk Contribution": rc})
    df = df.sort_values("Risk Contribution", ascending=True)

    y = np.arange(len(df))
    height = 0.35

    ax.barh(y - height/2, df["Weight"], height, label="Weight", color="#B0C4DE")
    ax.barh(y + height/2, df["Risk Contribution"], height, label="Risk Contribution", color="#FA8072")

    ax.set_yticks(y)
    ax.set_yticklabels(df.index)
    ax.legend()
    ax.set_title(title)
    
    return ax


def plot_correlation_matrix(
    cov_estimator,
    title: str = "Correlation Matrix",
    annot: bool = True,
    ax: Optional[plt.Axes] = None,
) -> plt.Axes:
    import scipy.cluster.hierarchy as sch
    
    ax = _ensure_ax(ax, figsize=(10, 8))
    
    corr = cov_estimator.correlation()
    
    # Compute distances and linkage
    dist = 1.0 - corr.values
    linkage = sch.linkage(sch.distance.squareform(dist), method='ward')
    order = sch.leaves_list(linkage)
    
    # Reorder
    corr_reordered = corr.iloc[order, order]
    
    sns.heatmap(
        corr_reordered,
        ax=ax,
        annot=annot,
        cmap="RdBu_r",
        vmin=-1,
        vmax=1,
        center=0,
        fmt=".2f",
        square=True,
        linewidths=0.5,
    )
    
    ax.set_title(title)
    return ax


def plot_rolling_metrics(
    returns: pd.Series,
    window: int = 63,
    frequency: int = 252,
    title: str = "Rolling Performance Metrics",
) -> Tuple[plt.Figure, np.ndarray]:
    """Returns a Figure and array of 3 Axes natively."""
    sns.set_theme(style="whitegrid")
    fig, axes = plt.subplots(3, 1, figsize=(12, 12), sharex=True)
    
    from quantopt.analytics.performance import rolling_metrics as calc_rolling
    
    roll = calc_rolling(returns, window=window, frequency=frequency)
    
    # 1. Sharpe
    ax = axes[0]
    ax.plot(roll.index, roll["sharpe"], color="indigo")
    ax.axhline(0, color="black", linewidth=1.2)
    ax.axhline(1, color="forestgreen", linestyle="--")
    ax.set_title(f"Rolling Sharpe Ratio ({window} days)")
    ax.set_ylabel("Sharpe")

    # 2. Volatility
    ax = axes[1]
    ax.plot(roll.index, roll["volatility"], color="darkorange")
    ax.set_title(f"Rolling Volatility ({window} days)")
    ax.set_ylabel("Volatility")

    # 3. MDD
    ax = axes[2]
    ax.fill_between(roll.index, roll["max_drawdown"], 0, color="red", alpha=0.3)
    ax.plot(roll.index, roll["max_drawdown"], color="darkred")
    ax.set_title(f"Rolling Max Drawdown ({window} days)")
    ax.set_ylabel("Drawdown")
    
    fig.suptitle(title, fontsize=16)
    fig.tight_layout()
    
    return fig, axes # type: ignore
