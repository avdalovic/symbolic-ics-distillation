from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class SampleAudit:
    target: str
    sample_policy: str
    sample_size_requested: int
    sample_size_actual: int
    num_transition_rows: int
    num_delta_bin_rows: int
    num_uniform_fill_rows: int
    delta_min: float
    delta_median: float
    delta_p90: float
    delta_p99: float
    fraction_of_actuator_transition_windows_covered: float

    def to_dict(self) -> dict[str, float | int | str]:
        return asdict(self)


def protected_division(x: np.ndarray | float, y: np.ndarray | float, eps: float = 1e-6) -> np.ndarray:
    """Numerically finite protected division used by rich diagnostic equations."""

    numerator = np.asarray(x, dtype=np.float64)
    denominator = np.asarray(y, dtype=np.float64)
    safe = np.where(np.abs(denominator) < float(eps), np.sign(denominator) * float(eps), denominator)
    safe = np.where(safe == 0.0, float(eps), safe)
    return numerator / safe


def reconstruct_next_from_delta(current_values: np.ndarray, delta_hat: np.ndarray) -> np.ndarray:
    """Convert a predicted sensor delta into next-value prediction."""

    return np.asarray(current_values, dtype=np.float64) + np.asarray(delta_hat, dtype=np.float64)


def uniform_grid_indices(n_rows: int, sample_size: int) -> np.ndarray:
    """Return a deterministic temporal grid over ``[0, n_rows)``."""

    n = int(n_rows)
    requested = int(sample_size)
    if n <= 0:
        return np.array([], dtype=np.int64)
    if requested <= 0 or requested >= n:
        return np.arange(n, dtype=np.int64)
    return np.unique(np.linspace(0, n - 1, num=requested, dtype=np.int64)).astype(np.int64)


def _grid_from_values(values: Sequence[int] | np.ndarray, budget: int) -> np.ndarray:
    arr = np.unique(np.asarray(values, dtype=np.int64))
    if arr.size == 0 or int(budget) <= 0:
        return np.array([], dtype=np.int64)
    if arr.size <= int(budget):
        return arr
    positions = uniform_grid_indices(arr.size, int(budget))
    return arr[positions]


def actuator_transition_windows(
    x_current: np.ndarray,
    x_next: np.ndarray,
    actuator_indices: Sequence[int],
    *,
    radius: int = 5,
) -> tuple[np.ndarray, list[tuple[int, int]]]:
    """Return rows within ``radius`` seconds of any actuator state transition."""

    current = np.asarray(x_current)
    nxt = np.asarray(x_next)
    if current.shape != nxt.shape:
        raise ValueError(f"x_current/x_next shape mismatch: {current.shape} vs {nxt.shape}")
    n = current.shape[0]
    if not actuator_indices:
        return np.array([], dtype=np.int64), []
    actuator_idx = np.asarray(list(actuator_indices), dtype=np.int64)
    changed = np.any(current[:, actuator_idx] != nxt[:, actuator_idx], axis=1)
    event_rows = np.flatnonzero(changed)
    windows: list[tuple[int, int]] = []
    rows: list[np.ndarray] = []
    for row in event_rows:
        start = max(0, int(row) - int(radius))
        end = min(n - 1, int(row) + int(radius))
        windows.append((start, end))
        rows.append(np.arange(start, end + 1, dtype=np.int64))
    if not rows:
        return np.array([], dtype=np.int64), windows
    return np.unique(np.concatenate(rows)).astype(np.int64), windows


def _delta_bin_indices(abs_delta: np.ndarray, candidate_indices: np.ndarray, budget: int) -> np.ndarray:
    if candidate_indices.size == 0 or int(budget) <= 0:
        return np.array([], dtype=np.int64)
    values = np.asarray(abs_delta, dtype=np.float64)[candidate_indices]
    finite_mask = np.isfinite(values)
    candidate_indices = candidate_indices[finite_mask]
    values = values[finite_mask]
    if candidate_indices.size == 0:
        return np.array([], dtype=np.int64)

    quantiles = np.quantile(values, np.linspace(0.0, 1.0, 11))
    bins: list[np.ndarray] = []
    for i in range(10):
        lo = quantiles[i]
        hi = quantiles[i + 1]
        if i == 9:
            mask = (values >= lo) & (values <= hi)
        else:
            mask = (values >= lo) & (values < hi)
        if not np.any(mask) and lo == hi:
            mask = values == lo
        if np.any(mask):
            bins.append(candidate_indices[mask])
    if not bins:
        return np.array([], dtype=np.int64)

    per_bin = max(1, int(np.ceil(int(budget) / len(bins))))
    chosen = [_grid_from_values(bin_indices, per_bin) for bin_indices in bins]
    merged = np.unique(np.concatenate([arr for arr in chosen if arr.size]))
    if merged.size > int(budget):
        merged = _grid_from_values(merged, int(budget))
    return merged.astype(np.int64)


def coverage_stratified_indices(
    *,
    target: str,
    y_delta_fit_pool: np.ndarray,
    x_current_fit_pool: np.ndarray,
    x_next_fit_pool: np.ndarray,
    actuator_indices: Sequence[int],
    sample_size: int,
    seed: int = 1337,
    transition_radius: int = 5,
    transition_fraction: float = 0.30,
    delta_fraction: float = 0.40,
) -> tuple[np.ndarray, SampleAudit]:
    """Deterministic training-only coverage sampling for 1-second SWaT PySR.

    The seed is intentionally present for provenance/tie-breaking stability, but
    the current policy uses temporal grids rather than stochastic draws.
    """

    del seed
    n = int(np.asarray(y_delta_fit_pool).shape[0])
    requested = min(max(int(sample_size), 0), n)
    if requested == 0:
        audit = SampleAudit(
            target=target,
            sample_policy="coverage_stratified",
            sample_size_requested=int(sample_size),
            sample_size_actual=0,
            num_transition_rows=0,
            num_delta_bin_rows=0,
            num_uniform_fill_rows=0,
            delta_min=float("nan"),
            delta_median=float("nan"),
            delta_p90=float("nan"),
            delta_p99=float("nan"),
            fraction_of_actuator_transition_windows_covered=0.0,
        )
        return np.array([], dtype=np.int64), audit

    abs_delta = np.abs(np.asarray(y_delta_fit_pool, dtype=np.float64).reshape(-1))
    all_indices = np.arange(n, dtype=np.int64)

    transition_rows, transition_windows = actuator_transition_windows(
        x_current_fit_pool,
        x_next_fit_pool,
        actuator_indices,
        radius=int(transition_radius),
    )
    transition_budget = int(round(requested * float(transition_fraction)))
    selected_transition = _grid_from_values(transition_rows, transition_budget)

    delta_budget = int(round(requested * float(delta_fraction)))
    selected_delta = _delta_bin_indices(abs_delta, all_indices, delta_budget)

    selected = np.unique(np.concatenate([selected_transition, selected_delta])).astype(np.int64)
    remaining_budget = requested - selected.size
    selected_uniform = np.array([], dtype=np.int64)
    if remaining_budget > 0:
        remaining = np.setdiff1d(all_indices, selected, assume_unique=False)
        selected_uniform = _grid_from_values(remaining, remaining_budget)
        selected = np.unique(np.concatenate([selected, selected_uniform])).astype(np.int64)

    if selected.size > requested:
        selected = _grid_from_values(selected, requested)

    if transition_windows:
        covered = 0
        selected_set = set(int(x) for x in selected)
        for start, end in transition_windows:
            if any(row in selected_set for row in range(start, end + 1)):
                covered += 1
        transition_fraction_covered = covered / len(transition_windows)
    else:
        transition_fraction_covered = 0.0

    finite_delta = abs_delta[np.isfinite(abs_delta)]
    audit = SampleAudit(
        target=target,
        sample_policy="coverage_stratified",
        sample_size_requested=int(sample_size),
        sample_size_actual=int(selected.size),
        num_transition_rows=int(np.intersect1d(selected, selected_transition).size),
        num_delta_bin_rows=int(np.setdiff1d(np.intersect1d(selected, selected_delta), selected_transition).size),
        num_uniform_fill_rows=int(np.setdiff1d(selected_uniform, np.union1d(selected_transition, selected_delta)).size),
        delta_min=float(np.min(finite_delta)) if finite_delta.size else float("nan"),
        delta_median=float(np.median(finite_delta)) if finite_delta.size else float("nan"),
        delta_p90=float(np.percentile(finite_delta, 90)) if finite_delta.size else float("nan"),
        delta_p99=float(np.percentile(finite_delta, 99)) if finite_delta.size else float("nan"),
        fraction_of_actuator_transition_windows_covered=float(transition_fraction_covered),
    )
    return np.sort(selected).astype(np.int64), audit


def sampling_audit_for_uniform(target: str, y_delta_fit_pool: np.ndarray, sample_size: int, actual_size: int) -> SampleAudit:
    abs_delta = np.abs(np.asarray(y_delta_fit_pool, dtype=np.float64).reshape(-1))
    finite_delta = abs_delta[np.isfinite(abs_delta)]
    return SampleAudit(
        target=target,
        sample_policy="uniform_grid",
        sample_size_requested=int(sample_size),
        sample_size_actual=int(actual_size),
        num_transition_rows=0,
        num_delta_bin_rows=0,
        num_uniform_fill_rows=int(actual_size),
        delta_min=float(np.min(finite_delta)) if finite_delta.size else float("nan"),
        delta_median=float(np.median(finite_delta)) if finite_delta.size else float("nan"),
        delta_p90=float(np.percentile(finite_delta, 90)) if finite_delta.size else float("nan"),
        delta_p99=float(np.percentile(finite_delta, 99)) if finite_delta.size else float("nan"),
        fraction_of_actuator_transition_windows_covered=0.0,
    )
