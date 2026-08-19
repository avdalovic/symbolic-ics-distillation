from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


def _maybe_array(payload: Mapping[str, np.ndarray], key: str) -> np.ndarray | None:
    if key not in payload:
        return None
    arr = np.asarray(payload[key])
    if arr.size == 0:
        return None
    return arr


def _scalar_string(payload: Mapping[str, np.ndarray], key: str, default: str) -> str:
    if key not in payload:
        return default
    arr = np.asarray(payload[key])
    if arr.shape == ():
        return str(arr.item())
    return str(arr.tolist())


def _scalar_float(payload: Mapping[str, np.ndarray], key: str, default: float) -> float:
    if key not in payload:
        return float(default)
    arr = np.asarray(payload[key])
    if arr.shape == ():
        return float(arr.item())
    if arr.size == 1:
        return float(arr.reshape(-1)[0])
    raise ValueError(f"Expected scalar normalization stat for {key}, got shape {arr.shape}")


def load_normalization_stats(path: str | Path) -> dict[str, Any]:
    """Load normalization stats saved by the training pipeline.

    Current training stores z-score ``std`` after applying ``std_floor``:
    ``std_safe = max(train_std, std_floor)``. For inversion, this saved safe
    scale is the scientifically correct scale because it is what training and
    export used to normalize every split.
    """

    stats_path = Path(path).expanduser()
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}")
    payload = np.load(stats_path, allow_pickle=False)
    files = list(payload.files)
    stats: dict[str, Any] = {
        "path": str(stats_path),
        "stat_keys": files,
        "normalization_mode": _scalar_string(payload, "normalization_mode", "none").lower(),
        "std_floor": _scalar_float(payload, "std_floor", 0.0),
        "fit_split": _scalar_string(payload, "fit_split", "train"),
        "feature_columns": [str(x) for x in _maybe_array(payload, "feature_columns").tolist()]
        if _maybe_array(payload, "feature_columns") is not None
        else [],
        "sensor_idx": _maybe_array(payload, "sensor_idx").astype(int).tolist()
        if _maybe_array(payload, "sensor_idx") is not None
        else [],
        "actuator_idx": _maybe_array(payload, "actuator_idx").astype(int).tolist()
        if _maybe_array(payload, "actuator_idx") is not None
        else [],
        "mean": _maybe_array(payload, "mean"),
        "std": _maybe_array(payload, "std"),
        "median": _maybe_array(payload, "median"),
        "iqr": _maybe_array(payload, "iqr"),
        "data_min": _maybe_array(payload, "data_min"),
        "data_max": _maybe_array(payload, "data_max"),
        "minmax_variable_mask": _maybe_array(payload, "minmax_variable_mask"),
    }
    if stats["normalization_mode"] == "standard":
        stats["normalization_mode"] = "zscore"
    return stats


def _mode(stats: Mapping[str, Any]) -> str:
    mode = str(stats.get("normalization_mode", stats.get("mode", "none"))).lower()
    return "zscore" if mode == "standard" else mode


def normalization_formula(stats: Mapping[str, Any]) -> dict[str, str]:
    mode = _mode(stats)
    if mode == "zscore":
        return {
            "normalize": "x_norm = (x_raw - mean) / std_safe",
            "inverse": "x_raw = x_norm * std_safe + mean",
            "scale_note": "std_safe is the saved std vector after applying std_floor during training.",
        }
    if mode == "minmax":
        return {
            "normalize": (
                "x_norm = (x_raw - data_min) / (data_max_safe - data_min) for variable "
                "channels; static channels are left in raw units"
            ),
            "inverse": (
                "x_raw = x_norm * (data_max_safe - data_min) + data_min for variable "
                "channels; static channels are left unchanged"
            ),
            "scale_note": "data_max in saved stats is the safe max used by training.",
        }
    if mode == "robust":
        return {
            "normalize": (
                "sensor channels: x_norm = (x_raw - median) / iqr_safe; actuator "
                "channels are left in raw units"
            ),
            "inverse": (
                "sensor channels: x_raw = x_norm * iqr_safe + median; actuator "
                "channels are left unchanged"
            ),
            "scale_note": "iqr_safe is the sensor-only scale used by training.",
        }
    if mode in {"none", ""}:
        return {
            "normalize": "x_norm = x_raw",
            "inverse": "x_raw = x_norm",
            "scale_note": "No normalization is applied.",
        }
    raise ValueError(f"Unsupported normalization mode: {mode}")


def _scale_vector(stats: Mapping[str, Any]) -> tuple[str, np.ndarray | None]:
    mode = _mode(stats)
    if mode == "zscore":
        std = stats.get("std")
        return "std", None if std is None else np.asarray(std, dtype=np.float32)
    if mode == "minmax":
        data_min = stats.get("data_min")
        data_max = stats.get("data_max")
        if data_min is None or data_max is None:
            return "data_range", None
        return "data_range", np.asarray(data_max, dtype=np.float32) - np.asarray(data_min, dtype=np.float32)
    if mode == "robust":
        iqr = stats.get("iqr")
        return "iqr", None if iqr is None else np.asarray(iqr, dtype=np.float32)
    return "none", None


def describe_normalization_stats(
    stats: Mapping[str, Any],
    feature_columns: Sequence[str] | None = None,
    *,
    near_zero_threshold: float = 1e-12,
) -> dict[str, Any]:
    """Return a JSON-safe summary of saved normalization stats."""

    columns = list(feature_columns or stats.get("feature_columns") or [])
    scale_name, scale = _scale_vector(stats)
    mode = _mode(stats)
    summary: dict[str, Any] = {
        "normalization_mode": mode,
        "fit_split": str(stats.get("fit_split", "train")),
        "std_floor": float(stats.get("std_floor", 0.0) or 0.0),
        "stat_keys": list(stats.get("stat_keys", [])),
        "formula": normalization_formula(stats),
        "scale_key": scale_name,
        "scale_shape": None if scale is None else list(scale.shape),
        "scale_min": None if scale is None else float(np.min(scale)),
        "scale_mean": None if scale is None else float(np.mean(scale)),
        "scale_max": None if scale is None else float(np.max(scale)),
        "zero_scale_count": 0,
        "near_zero_scale_count": 0,
        "zero_scale_channels": [],
        "near_zero_scale_channels": [],
        "floor_scale_channels": [],
        "static_minmax_channels": [],
    }
    if scale is None:
        return summary

    zero_idx = np.where(scale == 0.0)[0].astype(int).tolist()
    near_idx = np.where(np.abs(scale) < float(near_zero_threshold))[0].astype(int).tolist()
    floor_idx: list[int] = []
    if mode == "zscore":
        std_floor = float(stats.get("std_floor", 0.0) or 0.0)
        floor_idx = np.where(scale <= std_floor + 1e-12)[0].astype(int).tolist()
    static_idx: list[int] = []
    if mode == "minmax" and stats.get("minmax_variable_mask") is not None:
        mask = np.asarray(stats["minmax_variable_mask"], dtype=bool)
        static_idx = np.where(~mask)[0].astype(int).tolist()

    def names(indices: Sequence[int]) -> list[str]:
        if columns and len(columns) > max(indices, default=-1):
            return [str(columns[i]) for i in indices]
        return [str(i) for i in indices]

    summary.update(
        {
            "zero_scale_count": len(zero_idx),
            "near_zero_scale_count": len(near_idx),
            "zero_scale_channels": names(zero_idx),
            "near_zero_scale_channels": names(near_idx),
            "floor_scale_channels": names(floor_idx),
            "static_minmax_channels": names(static_idx),
        }
    )
    return summary


def _require_vector(stats: Mapping[str, Any], key: str, num_features: int | None = None) -> np.ndarray:
    value = stats.get(key)
    if value is None:
        raise ValueError(f"Normalization stats for mode {_mode(stats)!r} require {key}")
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim != 1:
        raise ValueError(f"Normalization stat {key} must be 1D, got shape {arr.shape}")
    if num_features is not None and arr.shape[0] != num_features:
        raise ValueError(f"Normalization stat {key} length {arr.shape[0]} != num_features {num_features}")
    return arr


def _num_features(stats: Mapping[str, Any]) -> int:
    for key in ("mean", "std", "median", "iqr", "data_min", "data_max"):
        arr = stats.get(key)
        if arr is not None:
            return int(np.asarray(arr).shape[0])
    columns = stats.get("feature_columns") or []
    if columns:
        return len(columns)
    raise ValueError("Cannot infer number of features from normalization stats")


def _feature_axis(arr: np.ndarray, num_features: int) -> int:
    if arr.ndim == 0:
        raise ValueError("Expected an array with a feature dimension")
    if arr.shape[-1] == num_features:
        return arr.ndim - 1
    if arr.ndim >= 2 and arr.shape[1] == num_features:
        return 1
    raise ValueError(
        f"Cannot infer feature axis for shape {arr.shape}; expected last dim or dim=1 to equal {num_features}"
    )


def _reshape_for_axis(vec: np.ndarray, ndim: int, axis: int) -> np.ndarray:
    shape = [1] * ndim
    shape[axis] = int(vec.shape[0])
    return vec.reshape(shape)


def _check_positive_scale(scale: np.ndarray, key: str) -> None:
    bad = np.where(scale <= 0.0)[0]
    if bad.size:
        raise ValueError(f"Normalization stat {key} contains non-positive scale entries at {bad.tolist()}")


def normalize_raw_features(raw: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    """Normalize raw full-feature arrays exactly like ``OneStepDataset``.

    Supports full feature arrays shaped ``[N, F]``, ``[N, F, H]``, or any shape
    where either the last axis or axis 1 is the feature dimension.
    """

    arr = np.asarray(raw, dtype=np.float32)
    mode = _mode(stats)
    if mode in {"none", ""}:
        return arr.astype(np.float32, copy=True)

    num_features = _num_features(stats)
    axis = _feature_axis(arr, num_features)
    if mode == "zscore":
        mean = _require_vector(stats, "mean", num_features)
        std = _require_vector(stats, "std", num_features)
        _check_positive_scale(std, "std")
        return ((arr - _reshape_for_axis(mean, arr.ndim, axis)) / _reshape_for_axis(std, arr.ndim, axis)).astype(
            np.float32
        )
    if mode == "minmax":
        data_min = _require_vector(stats, "data_min", num_features)
        data_max = _require_vector(stats, "data_max", num_features)
        mask = stats.get("minmax_variable_mask")
        if mask is None:
            raise ValueError("Minmax normalization stats require minmax_variable_mask")
        mask_arr = np.asarray(mask, dtype=bool)
        scale = data_max - data_min
        _check_positive_scale(scale, "data_max - data_min")
        scaled = (arr - _reshape_for_axis(data_min, arr.ndim, axis)) / _reshape_for_axis(scale, arr.ndim, axis)
        mask_view = _reshape_for_axis(mask_arr, arr.ndim, axis)
        return np.where(mask_view, scaled, arr).astype(np.float32)
    if mode == "robust":
        median = _require_vector(stats, "median", num_features)
        iqr = _require_vector(stats, "iqr", num_features)
        _check_positive_scale(iqr, "iqr")
        sensor_idx = [int(i) for i in stats.get("sensor_idx") or []]
        if not sensor_idx:
            raise ValueError("Robust normalization stats require sensor_idx")
        sensor_mask = np.zeros(num_features, dtype=bool)
        sensor_mask[sensor_idx] = True
        scaled = (arr - _reshape_for_axis(median, arr.ndim, axis)) / _reshape_for_axis(iqr, arr.ndim, axis)
        return np.where(_reshape_for_axis(sensor_mask, arr.ndim, axis), scaled, arr).astype(np.float32)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def inverse_normalize_features(norm: np.ndarray, stats: Mapping[str, Any]) -> np.ndarray:
    """Invert normalized full-feature arrays using the training formula.

    For current z-score checkpoints this is:
    ``x_raw = x_norm * saved_std_safe + saved_mean``.
    """

    arr = np.asarray(norm, dtype=np.float32)
    mode = _mode(stats)
    if mode in {"none", ""}:
        return arr.astype(np.float32, copy=True)

    num_features = _num_features(stats)
    axis = _feature_axis(arr, num_features)
    if mode == "zscore":
        mean = _require_vector(stats, "mean", num_features)
        std = _require_vector(stats, "std", num_features)
        _check_positive_scale(std, "std")
        return (arr * _reshape_for_axis(std, arr.ndim, axis) + _reshape_for_axis(mean, arr.ndim, axis)).astype(
            np.float32
        )
    if mode == "minmax":
        data_min = _require_vector(stats, "data_min", num_features)
        data_max = _require_vector(stats, "data_max", num_features)
        mask = stats.get("minmax_variable_mask")
        if mask is None:
            raise ValueError("Minmax normalization stats require minmax_variable_mask")
        mask_arr = np.asarray(mask, dtype=bool)
        scale = data_max - data_min
        _check_positive_scale(scale, "data_max - data_min")
        restored = arr * _reshape_for_axis(scale, arr.ndim, axis) + _reshape_for_axis(data_min, arr.ndim, axis)
        return np.where(_reshape_for_axis(mask_arr, arr.ndim, axis), restored, arr).astype(np.float32)
    if mode == "robust":
        median = _require_vector(stats, "median", num_features)
        iqr = _require_vector(stats, "iqr", num_features)
        _check_positive_scale(iqr, "iqr")
        sensor_idx = [int(i) for i in stats.get("sensor_idx") or []]
        if not sensor_idx:
            raise ValueError("Robust normalization stats require sensor_idx")
        sensor_mask = np.zeros(num_features, dtype=bool)
        sensor_mask[sensor_idx] = True
        restored = arr * _reshape_for_axis(iqr, arr.ndim, axis) + _reshape_for_axis(median, arr.ndim, axis)
        return np.where(_reshape_for_axis(sensor_mask, arr.ndim, axis), restored, arr).astype(np.float32)
    raise ValueError(f"Unsupported normalization mode: {mode}")


def inverse_normalize_targets(
    norm_targets: np.ndarray,
    stats: Mapping[str, Any],
    sensor_idx: Sequence[int],
) -> np.ndarray:
    """Invert normalized sensor-target arrays through ``sensor_idx``.

    Target arrays are expected to have target columns on the last axis, e.g.
    ``[N, 25]`` or ``[N, horizon, 25]``. The scale and offset are selected from
    the full feature stats by ``sensor_idx``; callers must not assume sensors
    occupy ``features[:25]``.
    """

    arr = np.asarray(norm_targets, dtype=np.float32)
    idx = np.asarray([int(i) for i in sensor_idx], dtype=np.int64)
    if arr.shape[-1] != idx.shape[0]:
        raise ValueError(f"Target last dim {arr.shape[-1]} does not match sensor_idx length {idx.shape[0]}")

    mode = _mode(stats)
    if mode in {"none", ""}:
        return arr.astype(np.float32, copy=True)

    num_features = _num_features(stats)
    if np.any(idx < 0) or np.any(idx >= num_features):
        raise ValueError(f"sensor_idx contains out-of-range values for num_features={num_features}")

    if mode == "zscore":
        mean = _require_vector(stats, "mean", num_features)[idx]
        std = _require_vector(stats, "std", num_features)[idx]
        _check_positive_scale(std, "std[sensor_idx]")
        return (arr * _reshape_for_axis(std, arr.ndim, arr.ndim - 1) + _reshape_for_axis(mean, arr.ndim, arr.ndim - 1)).astype(
            np.float32
        )
    if mode == "minmax":
        data_min = _require_vector(stats, "data_min", num_features)[idx]
        data_max = _require_vector(stats, "data_max", num_features)[idx]
        mask = stats.get("minmax_variable_mask")
        if mask is None:
            raise ValueError("Minmax normalization stats require minmax_variable_mask")
        mask_arr = np.asarray(mask, dtype=bool)[idx]
        scale = data_max - data_min
        _check_positive_scale(scale, "data_max - data_min at sensor_idx")
        restored = arr * _reshape_for_axis(scale, arr.ndim, arr.ndim - 1) + _reshape_for_axis(
            data_min, arr.ndim, arr.ndim - 1
        )
        return np.where(_reshape_for_axis(mask_arr, arr.ndim, arr.ndim - 1), restored, arr).astype(np.float32)
    if mode == "robust":
        median = _require_vector(stats, "median", num_features)[idx]
        iqr = _require_vector(stats, "iqr", num_features)[idx]
        _check_positive_scale(iqr, "iqr[sensor_idx]")
        return (arr * _reshape_for_axis(iqr, arr.ndim, arr.ndim - 1) + _reshape_for_axis(median, arr.ndim, arr.ndim - 1)).astype(
            np.float32
        )
    raise ValueError(f"Unsupported normalization mode: {mode}")
