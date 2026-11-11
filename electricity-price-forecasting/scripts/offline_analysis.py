"""Inspect supplied CSVs and run a cheap chronological persistence/conformal baseline.

No pickles, neural checkpoints, external services or training loops are used.
"""
from pathlib import Path
import json
import sys

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from tools.prediction_quantiles_tools import exec_cp
from tools.stat_tests_tools import kupiec_test


def main():
    inventory = []
    for path in sorted((ROOT / "data" / "datasets").glob("*.csv")):
        frame = pd.read_csv(path)
        targets = [c for c in frame if c.startswith("TARG__")]
        inventory.append({
            "file": path.name, "rows": len(frame), "target_columns": targets,
            "first_date": str(frame["Date"].iloc[0]), "last_date": str(frame["Date"].iloc[-1]),
            "nonfinite_targets": int((~np.isfinite(frame[targets].to_numpy(dtype=float))).sum()),
        })
    source = ROOT / "data" / "datasets" / "DE_market__2015-01-01__2020-12-31.csv"
    frame = pd.read_csv(source).sort_values(["Date", "Hour"])
    groups = frame.groupby("Date", sort=True)
    if not (groups.size() == 24).all() or any(list(g["Hour"]) != list(range(24)) for _, g in groups):
        raise ValueError("Baseline requires 24 distinct ordered hours for every day.")
    dates = pd.to_datetime(list(groups.groups))
    if not ((dates[1:] - dates[:-1]).days == 1).all():
        raise ValueError("Baseline requires consecutive dates.")
    prices = frame["TARG__DE_price"].to_numpy(dtype=float).reshape(-1, 24)
    if not np.isfinite(prices).all() or len(prices) < 243:
        raise ValueError("Insufficient finite prices for the requested windows.")
    actual = prices[1:][-242:]
    predicted = prices[:-1][-242:]
    alpha = 0.2
    intervals = exec_cp(predicted[:182, :, None], actual[:182], predicted[182:, :, None],
                        {"target_alpha": [alpha]}).reshape(60, 24, 3)
    target = actual[182:]
    lower, median, upper = [intervals[:, :, i] for i in range(3)]
    hits = ((target >= lower) & (target <= upper))
    width = upper - lower
    winkler = width + (2 / alpha) * (np.maximum(lower - target, 0) + np.maximum(target - upper, 0))
    coverage_test = kupiec_test(hits.ravel(), alpha=alpha)
    result = {
        "scope": "Fresh CSV-based baseline; not archived neural forecasts or paper reproduction",
        "datasets": inventory,
        "baseline": {
            "market": "DE_price", "forecast": "previous day's same-hour observed price",
            "calibration_days": 182, "test_days": 60,
            "calibration_start": str(dates[-242].date()), "calibration_end": str(dates[-61].date()),
            "test_start": str(dates[-60].date()), "test_end": str(dates[-1].date()),
            "nominal_coverage": 1 - alpha, "test_observations": int(target.size),
            "mae": float(np.mean(np.abs(target - median))),
            "empirical_interval_coverage": float(hits.mean()),
            "mean_interval_width": float(width.mean()), "mean_winkler_score": float(winkler.mean()),
            "kupiec_LR": float(coverage_test["LR_UC"]), "kupiec_p_value": float(coverage_test["p-value"]),
            "caveat": "Time-dependent observations: this baseline does not establish exchangeability or independent-hit test assumptions.",
        },
        "archived_pickle_files_loaded": 0,
        "neural_models_trained": 0,
    }
    output = ROOT / "reports" / "offline_analysis.json"
    output.parent.mkdir(exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(result["baseline"], indent=2))
    print("CSV datasets inspected:", len(inventory))


if __name__ == "__main__":
    main()
