from __future__ import annotations

import numpy as np

from ics_symbolic_distill.detection.cusum import fit_cusum_params, run_cusum
from ics_symbolic_distill.detection.metrics import to_intervals


def test_to_intervals_edge_cases() -> None:
    assert to_intervals([]) == []
    assert to_intervals([0, 0, 0]) == []
    assert to_intervals([1, 1, 1]) == [(0, 2)]
    assert to_intervals([0, 1, 1, 0, 1, 0, 1]) == [(1, 2), (4, 4), (6, 6)]


def test_cusum_recurrence_and_cap() -> None:
    calib = np.array([0.0, 0.0, 0.0, 10.0, 10.0])
    params = fit_cusum_params(calib, s=1.0, g=1.0)
    assert np.isclose(params.delta, calib.mean() + calib.std())
    assert params.threshold > 0.0
    assert np.isclose(params.growth_cap, params.threshold + params.delta)
    assert np.isclose(params.threshold, params.max_calib_cusum)

    test = np.array([3.0, 3.0, 3.0])
    values, alarms = run_cusum(test, params)
    expected = []
    cusum = 0.0
    for residual in test:
        cusum = min(max(0.0, cusum + residual - params.delta), params.growth_cap)
        expected.append(cusum)
    np.testing.assert_allclose(values, expected)
    np.testing.assert_array_equal(alarms, values > params.threshold)


def test_cusum_uses_supplied_calibration_residuals() -> None:
    train_like = np.array([0.0, 0.0, 0.0])
    calib = np.array([1.0, 2.0, 3.0])
    params = fit_cusum_params(calib, s=1.42, g=5.98)
    assert not np.isclose(params.delta, train_like.mean() + train_like.std())
    assert np.isclose(params.delta, calib.mean() + calib.std())
