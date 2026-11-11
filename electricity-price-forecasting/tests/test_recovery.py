import numpy as np
import pandas as pd
import pytest

from tools.cp_pi import mytan
from tools.prediction_quantiles_tools import (
    build_alpha_quantiles_map, build_target_quantiles, conformal_quantile,
    exec_cp, exec_cqr, fix_quantile_crossing,
)
from tools.stat_tests_tools import kupiec_test


def test_conformal_order_statistic_is_exact_not_interpolated():
    assert conformal_quantile([1, 2, 3, 4], 0.4) == 3
    np.testing.assert_allclose(conformal_quantile([[1, 10], [2, 20], [3, 30], [4, 40]], 0.4), [3, 30])


def test_insufficient_calibration_returns_infinite_correction():
    assert np.isinf(conformal_quantile([1, 2, 3], 0.01))


@pytest.mark.parametrize("scores,alpha", [([], 0.2), ([np.nan], 0.2), ([1], 0), ([1], 1)])
def test_invalid_conformal_inputs(scores, alpha):
    with pytest.raises(ValueError):
        conformal_quantile(scores, alpha)


def test_point_intervals_support_multiple_days_without_exiting():
    cal = np.zeros((4, 1, 1))
    result = exec_cp(cal, np.array([[1], [2], [3], [4]]), np.array([[[10]], [[20]]]),
                     {"target_alpha": [0.4]})
    np.testing.assert_allclose(result, [[7, 10, 13], [17, 20, 23]])


def test_point_interval_shapes_are_validated():
    with pytest.raises(ValueError):
        exec_cp(np.zeros((4, 2)), np.zeros((4, 2)), np.zeros((1, 2, 1)), {"target_alpha": [0.2]})


def test_cqr_does_not_mutate_base_forecasts():
    quantiles = build_target_quantiles([0.4])
    base = np.array([[[0., 5., 10.]]])
    original = base.copy()
    calibration = np.repeat(base, 9, axis=0)
    settings = {"target_alpha": [0.4], "q_alpha_map": build_alpha_quantiles_map([0.4], quantiles)}
    result = exec_cqr(calibration, np.full((9, 1), 5.), base, settings)
    np.testing.assert_array_equal(base, original)
    assert np.all(np.diff(result, axis=1) >= 0)


@pytest.mark.parametrize("hits", [[0] * 20, [1] * 20, [0] * 2 + [1] * 18])
def test_kupiec_is_finite_for_boundary_and_interior_counts(hits):
    result = kupiec_test(hits, alpha=0.1)
    assert np.isfinite(result["LR_UC"])
    assert 0 <= result["p-value"] <= 1


def test_kupiec_all_misses_rejects_instead_of_nan_pass():
    result = kupiec_test([0] * 20, alpha=0.1)
    assert result["LR_UC"] == pytest.approx(-40 * np.log(0.1))
    assert not result["passed"]


def test_kupiec_nominal_coverage_has_zero_likelihood_ratio():
    assert kupiec_test(pd.DataFrame({"hit": [1] * 90 + [0] * 10}), alpha=0.1)["LR_UC"] == pytest.approx(0)


@pytest.mark.parametrize("hits", [[], [0, 2], [1, np.nan], [[1, 0], [0, 1]]])
def test_invalid_coverage_observations_rejected(hits):
    with pytest.raises(ValueError):
        kupiec_test(hits)


def test_quantile_crossing_is_sorted():
    np.testing.assert_array_equal(fix_quantile_crossing([[3, 1, 2]]), [[1, 2, 3]])


def test_numpy_two_saturation_uses_supported_infinity_constant():
    assert mytan(np.pi) == np.inf
    assert mytan(-np.pi) == -np.inf


@pytest.mark.parametrize("module_name", ["run_recalibration", "exec_qra_cp", "results_analysis"])
def test_legacy_entry_points_import_without_training_or_overwriting(module_name):
    import importlib
    module = importlib.import_module(module_name)
    assert callable(module.main)
