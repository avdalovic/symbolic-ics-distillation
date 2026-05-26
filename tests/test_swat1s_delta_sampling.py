from __future__ import annotations

import numpy as np

from ics_symbolic_distill.detection.swat1s_delta_sampling import (
    coverage_stratified_indices,
    protected_division,
    reconstruct_next_from_delta,
)


def test_coverage_stratified_sampling_is_deterministic_and_unique() -> None:
    n = 200
    x_current = np.zeros((n, 4), dtype=float)
    x_next = x_current.copy()
    x_next[20, 2] = 1.0
    x_next[100, 3] = 1.0
    y_delta = np.linspace(0.0, 10.0, n)

    idx_a, audit_a = coverage_stratified_indices(
        target="LIT101",
        y_delta_fit_pool=y_delta,
        x_current_fit_pool=x_current,
        x_next_fit_pool=x_next,
        actuator_indices=[2, 3],
        sample_size=50,
        seed=1337,
    )
    idx_b, audit_b = coverage_stratified_indices(
        target="LIT101",
        y_delta_fit_pool=y_delta,
        x_current_fit_pool=x_current,
        x_next_fit_pool=x_next,
        actuator_indices=[2, 3],
        sample_size=50,
        seed=1337,
    )

    np.testing.assert_array_equal(idx_a, idx_b)
    assert audit_a.to_dict() == audit_b.to_dict()
    assert len(idx_a) == len(np.unique(idx_a))


def test_coverage_stratified_indices_are_inside_fit_pool() -> None:
    n = 100
    x_current = np.zeros((n, 3), dtype=float)
    x_next = x_current.copy()
    y_delta = np.sin(np.linspace(0.0, 5.0, n))
    idx, audit = coverage_stratified_indices(
        target="FIT201",
        y_delta_fit_pool=y_delta,
        x_current_fit_pool=x_current,
        x_next_fit_pool=x_next,
        actuator_indices=[2],
        sample_size=40,
    )

    assert idx.size == audit.sample_size_actual
    assert idx.min() >= 0
    assert idx.max() < n


def test_sensor_delta_reconstruction() -> None:
    current = np.array([1.0, 2.0, 3.0])
    delta_hat = np.array([0.5, -0.25, 0.0])
    np.testing.assert_allclose(reconstruct_next_from_delta(current, delta_hat), np.array([1.5, 1.75, 3.0]))


def test_protected_division_is_finite() -> None:
    numerator = np.array([1.0, 2.0, -3.0])
    denominator = np.array([0.0, 1e-12, -2.0])
    result = protected_division(numerator, denominator)
    assert np.isfinite(result).all()
