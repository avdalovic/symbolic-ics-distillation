from __future__ import annotations

import numpy as np

from ics_symbolic_distill.detection.symbolic_eval import evaluate_equation


FEATURES = ["FIT101", "LIT101", "MV101"]
X = np.array(
    [
        [1.0, 2.0, 3.0],
        [4.0, 5.0, 6.0],
    ],
    dtype=float,
)


def test_equation_x_index_variables() -> None:
    np.testing.assert_allclose(evaluate_equation("x0 + x1", FEATURES, X), X[:, 0] + X[:, 1])


def test_equation_feature_names() -> None:
    np.testing.assert_allclose(evaluate_equation("FIT101 + 2 * LIT101", FEATURES, X), X[:, 0] + 2 * X[:, 1])


def test_equation_square_function() -> None:
    np.testing.assert_allclose(evaluate_equation("square(x0) + MV101", FEATURES, X), X[:, 0] ** 2 + X[:, 2])


def test_invalid_equation_returns_nans() -> None:
    result = evaluate_equation("FIT101 + UNKNOWN_TAG", FEATURES, X)
    assert result.shape == (2,)
    assert np.isnan(result).all()
