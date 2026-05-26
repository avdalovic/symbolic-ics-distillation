from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np


@dataclass(frozen=True)
class CusumParams:
    """Per-sensor CUSUM parameters fit on benign calibration residuals."""

    delta: float
    threshold: float
    growth_cap: float
    max_calib_cusum: float
    s: float
    g: float

    def to_dict(self) -> dict[str, float]:
        return asdict(self)


def _positive_or_floor(value: float, floor: float = 1e-6) -> float:
    val = float(value)
    if not np.isfinite(val) or val <= 0.0:
        return float(floor)
    return val


def fit_cusum_params(residuals: np.ndarray, *, s: float = 1.42, g: float = 5.98) -> CusumParams:
    """Fit GECO-style CUSUM parameters from benign residuals.

    ``delta`` is ``mean(residuals) + std(residuals)``. The calibration CUSUM is
    run without the test-time growth cap. The threshold is ``s`` times the
    maximum calibration CUSUM, and the growth cap is ``threshold + g * delta``.
    """

    r = np.asarray(residuals, dtype=np.float64).reshape(-1)
    r = r[np.isfinite(r)]
    if r.size == 0:
        delta = 1e-6
    else:
        delta = _positive_or_floor(float(np.mean(r) + np.std(r)))

    cusum = 0.0
    train_max = 0.0
    for value in r:
        cusum = max(0.0, cusum + float(value) - delta)
        train_max = max(train_max, cusum)

    threshold = _positive_or_floor(float(s) * train_max)
    growth_cap = _positive_or_floor(threshold + float(g) * delta)
    return CusumParams(
        delta=float(delta),
        threshold=float(threshold),
        growth_cap=float(growth_cap),
        max_calib_cusum=float(train_max),
        s=float(s),
        g=float(g),
    )


def run_cusum(residuals: np.ndarray, params: CusumParams) -> tuple[np.ndarray, np.ndarray]:
    """Run capped test-time CUSUM and return CUSUM values plus binary alarms."""

    r = np.asarray(residuals, dtype=np.float64).reshape(-1)
    cusum_values = np.zeros(r.shape[0], dtype=np.float64)
    alarms = np.zeros(r.shape[0], dtype=np.int64)
    cusum = 0.0
    for i, value in enumerate(r):
        residual = 0.0 if not np.isfinite(value) else float(value)
        raw = max(0.0, cusum + residual - params.delta)
        cusum = min(raw, params.growth_cap)
        cusum_values[i] = cusum
        alarms[i] = int(cusum > params.threshold)
    return cusum_values, alarms
