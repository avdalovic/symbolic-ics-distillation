"""Detection utilities for symbolic and neural ICS predictors."""

from .cusum import CusumParams, fit_cusum_params, run_cusum
from .metrics import compute_detection_metrics, to_intervals
from .symbolic_eval import evaluate_equation, load_pareto_front
from .swat1s_delta_sampling import (
    coverage_stratified_indices,
    protected_division,
    reconstruct_next_from_delta,
    uniform_grid_indices,
)

__all__ = [
    "CusumParams",
    "compute_detection_metrics",
    "evaluate_equation",
    "fit_cusum_params",
    "load_pareto_front",
    "coverage_stratified_indices",
    "protected_division",
    "reconstruct_next_from_delta",
    "run_cusum",
    "to_intervals",
    "uniform_grid_indices",
]
