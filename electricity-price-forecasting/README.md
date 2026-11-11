# Probabilistic electricity-price forecasting

This project studies **day-ahead electricity prices**: forecast the next day's 24 hourly prices, together with uncertainty intervals. A point prediction gives one expected value; a probabilistic forecast also describes a range of plausible values. The research question is whether ensemble forecasts and conformal adjustments improve the accuracy and coverage of those ranges as new prices arrive.

This is an archived research implementation with supplied market CSVs, experiment configurations, tuned hyperparameters and saved forecast files. Its public research/software provenance is retained in [NOTICE.md](NOTICE.md). No claim of original authorship or a production deployment is established by this recovery.

## What is verified

**23 tests pass** on the available local Python 3.13 installation. They check exact conformal residual ranks, small calibration samples, prediction shapes, non-mutating conformalized quantiles, coverage-test boundary cases, NumPy compatibility and safe imports of the legacy scripts.

A fresh CSV-only baseline runs without downloads or neural training. Seven supplied datasets were inspected. The German-market example uses the previous day's price at the same hour as a simple forecast, calibrates on 182 consecutive days, and evaluates on the following 60 days.

| Fresh baseline result | Value |
|---|---:|
| Calibration window | 2020-05-04 to 2020-11-01 |
| Test window | 2020-11-02 to 2020-12-31 |
| Test hourly observations | 1,440 |
| Nominal interval coverage | 80% |
| Observed interval coverage | 70.97% |
| Mean absolute point error | 11.69 |
| Mean interval width | 27.73 |

Price errors and widths use the source price units. This simple baseline **misses its coverage target**. Its results are not neural-model results or a reproduction of the paper. Time dependence and distribution shifts can invalidate the exchangeability assumptions used by ordinary split conformal prediction.

Details are in [reports/offline_analysis.json](reports/offline_analysis.json) and [reports/verification.json](reports/verification.json).

## Run the cheap checks

From this project directory, using the existing local Python 3.13 libraries:

```powershell
py -3.13 scripts/offline_analysis.py
```

This reads CSVs, computes the baseline and writes an aggregate JSON report. It never loads pickle files or starts TensorFlow training.

```powershell
$env:PYTHONPATH = (Resolve-Path '..\..\_workspace\runtime\python314\core').Path + ';' + (Get-Location).Path
py -3.13 -m pytest -q tests/test_recovery.py
```

`requirements-core.txt` lists the dependencies for the offline checks. `requirements.txt` records the training environment.

## Understand the original research pipeline

1. **Inputs:** `data/datasets/` contains seven hourly market datasets: Germany (`DE_price`) and six Italian market zones (`MGP_*`). Columns distinguish targets, known future inputs, lagged features, date and hour.
2. **Configurations:** `experiments/tasks/<market>/<model>/<run>/` stores training-window settings, preprocessing, forecast horizon and tuned hyperparameters. Four component runs support ensemble aggregation.
3. **Neural forecasts:** `run_recalibration.py` and `tools/PrTSF_Recalib_tools.py` implement rolling recalibration/retraining. Model families include point forecasts, direct quantiles and Normal, Student-t or Johnson-SU distributional forecasts.
4. **Postprocessing:** `exec_qra_cp.py` and `tools/quant_proc_tools.py` combine ensemble predictions and apply Quantile Regression Averaging (QRA), point-forecast conformal prediction (CP), conformalized quantile regression (CQR), or online interval adjustment.
5. **Analysis:** `results_analysis.py` and supporting tools calculate point/quantile errors, interval coverage and width, Winkler scores, coverage tests and forecast-comparison tests, and produce plots/tables.

The legacy scripts now expose `main()` and run only when invoked directly. Importing them no longer starts lengthy experiments or rewrites results. Running the training or postprocessing commands directly can still write into existing experiment paths: copy the experiment folders and update run IDs first.

## Interpret the uncertainty methods

For miscoverage `alpha = 0.2`, an interval targets 80% marginal coverage. This is not a claim that every hour or every market condition receives 80% coverage.

CP measures absolute errors on a past calibration window, chooses a residual rank, and widens the point forecast by that amount. CQR adjusts predicted lower and upper quantiles using calibration errors. The asymmetric CQR implementation calibrates each tail at `alpha/2`. With too few calibration observations, a finite-sample correction can legitimately become infinite rather than support a narrow interval.

QRA learns quantile predictions from ensemble outputs using quantile regression. Online adjustment uses past interval misses to update corrections over time. Online routines use `np.inf` for NumPy 2 compatibility.

Coverage is the fraction of observations inside an interval. Width measures how broad it is. Winkler score penalizes both wide intervals and misses. Pinball loss evaluates quantile forecasts. The Kupiec test checks an unconditional coverage hypothesis; its usual interpretation also needs assumptions about the observations. A small test p-value cannot establish why coverage failed, and a large one does not prove calibration.

## Repairs and archived results

The point-conformal routine now uses the exact `ceil((n+1)*(1-alpha))`-th residual rather than applying a second, incorrect quantile-index transformation. CQR uses the same finite-sample rank rule for its tails and no longer mutates the caller's forecast array. Point intervals support multiple test days and raise ordinary errors for incompatible shapes.

The Kupiec implementation now handles all-hit and all-miss samples without `0 * log(0)` producing NaN. Its hypothesis label now refers to coverage. Removed `np.infty` references were replaced by `np.inf` for NumPy 2 compatibility.

These corrections affect newly computed intervals and statistics. All 147 saved pickle streams were inspected structurally with `pickletools`, without deserialization or execution.

Original files are backed up under ignored `.private/`. IDE settings containing local context were moved there. Source-author attribution was consolidated into `NOTICE.md`, and license notices remain intact. No CV or interview materials were added; no email or publishing function was invoked.
