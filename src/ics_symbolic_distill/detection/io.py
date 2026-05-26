from __future__ import annotations

import json
import platform
import sys
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np

from ics_symbolic_distill.data.normalization import (
    inverse_normalize_features,
    inverse_normalize_targets,
    load_normalization_stats,
)
from ics_symbolic_distill.distillation.prepare import squeeze_horizon_one


@dataclass
class DetectionSplit:
    x_current_raw: np.ndarray
    actual_next_raw: np.ndarray
    mlp_pred_next_raw: np.ndarray
    actual_delta_raw: np.ndarray
    mlp_pred_delta_raw: np.ndarray
    labels: np.ndarray | None
    feature_columns: list[str]
    target_columns: list[str]
    sensor_idx: list[int]
    source: str
    split_name: str


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return out


def _jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if np.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [_jsonable(v) for v in value]
    return str(value)


def _first_existing(path: Path, names: list[str]) -> Path:
    for name in names:
        candidate = path / name
        if candidate.exists():
            return candidate
    raise FileNotFoundError(f"None of these files exist under {path}: {names}")


def load_distillation_split(
    root: str | Path,
    *,
    require_labels: bool = False,
    split_name: str | None = None,
) -> DetectionSplit:
    """Load raw-unit distillation arrays from ``prepare_distillation_data.py`` output."""

    path = Path(root)
    feature_columns = [str(x) for x in read_json(path / "distill_feature_columns.json")]
    target_columns = [str(x) for x in read_json(path / "distill_target_columns.json")]
    sensor_idx = [int(x) for x in read_json(path / "distill_sensor_idx.json")]
    suffix = f"_{split_name}" if split_name else ""
    x = np.load(
        _first_existing(path, [f"distill_inputs{suffix}.npy", "distill_inputs_current_raw.npy"])
    ).astype(np.float64)
    actual_next = np.load(
        _first_existing(path, [f"distill_actual_next{suffix}.npy", "distill_actual_next_raw.npy"])
    ).astype(np.float64)
    mlp_next = np.load(
        _first_existing(path, [f"distill_pred_next_mlp{suffix}.npy", "distill_pred_next_raw_mlp.npy"])
    ).astype(np.float64)
    labels_candidates = [f"distill_labels{suffix}.npy", "distill_labels.npy"]
    labels_path = next((path / name for name in labels_candidates if (path / name).exists()), path / labels_candidates[0])
    labels = np.load(labels_path).astype(np.int64).reshape(-1) if labels_path.exists() else None
    if require_labels and labels is None:
        raise FileNotFoundError(f"Missing labels for detection split: {labels_path}")
    current_sensor = x[:, sensor_idx]
    actual_delta_path = next(
        (
            path / name
            for name in [f"distill_actual_delta{suffix}.npy", "distill_actual_delta_raw.npy"]
            if (path / name).exists()
        ),
        None,
    )
    mlp_delta_path = next(
        (
            path / name
            for name in [f"distill_pred_delta_mlp{suffix}.npy", "distill_pred_delta_raw_mlp.npy"]
            if (path / name).exists()
        ),
        None,
    )
    actual_delta = (
        np.load(actual_delta_path).astype(np.float64)
        if actual_delta_path is not None
        else (actual_next - current_sensor).astype(np.float64)
    )
    mlp_delta = (
        np.load(mlp_delta_path).astype(np.float64)
        if mlp_delta_path is not None
        else (mlp_next - current_sensor).astype(np.float64)
    )
    _validate_split_shapes(x, actual_next, mlp_next, labels, feature_columns, target_columns, sensor_idx, source=str(path))
    if actual_delta.shape != actual_next.shape or mlp_delta.shape != actual_next.shape:
        raise ValueError(f"{path}: delta arrays must match actual_next shape")
    return DetectionSplit(
        x,
        actual_next,
        mlp_next,
        actual_delta,
        mlp_delta,
        labels,
        feature_columns,
        target_columns,
        sensor_idx,
        str(path),
        split_name or "distill",
    )


def _split_name_from_export(root: Path) -> str:
    metadata_path = root / "metadata.json"
    if metadata_path.exists():
        metadata_payload = read_json(metadata_path)
        return str(metadata_payload.get("split_name", root.name)).lower()
    return root.name.lower()


def load_model_export_raw_split(root: str | Path, *, require_labels: bool = True) -> DetectionSplit:
    """Load a model export directory and invert normalized arrays to raw units."""

    path = Path(root)
    metadata_path = path / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing model export metadata: {metadata_path}")
    meta = read_json(metadata_path)
    split = _split_name_from_export(path)
    inputs = np.load(path / f"{split}_inputs.npy")
    preds = np.load(path / f"{split}_neural_preds.npy")
    actual = np.load(path / f"{split}_actual_next.npy")
    labels_path = path / f"{split}_labels.npy"
    labels = np.load(labels_path).astype(np.int64).reshape(-1) if labels_path.exists() else None
    if require_labels and labels is None:
        raise FileNotFoundError(
            f"Missing model export labels: {labels_path}. Re-run export_model_predictions.py after this update."
        )

    stats_path = meta.get("normalization_stats_path")
    columns_path = meta.get("columns_json_path")
    if not stats_path:
        raise FileNotFoundError("Export metadata does not include normalization_stats_path")
    if not columns_path:
        raise FileNotFoundError("Export metadata does not include columns_json_path")
    stats = load_normalization_stats(stats_path)
    columns = read_json(columns_path)
    feature_columns = [str(x) for x in columns["feature_columns"]]
    target_columns = [str(x) for x in columns["target_columns"]]
    sensor_idx = [int(x) for x in columns["sensor_idx"]]

    x_norm = np.asarray(inputs, dtype=np.float32)
    if x_norm.ndim != 3:
        raise ValueError(f"Expected export inputs [N, F, H], got {x_norm.shape}")
    x_current_raw = inverse_normalize_features(x_norm[:, :, -1], stats).astype(np.float64)
    pred_next_raw = inverse_normalize_targets(squeeze_horizon_one(preds, "preds"), stats, sensor_idx).astype(np.float64)
    actual_next_raw = inverse_normalize_targets(squeeze_horizon_one(actual, "actual"), stats, sensor_idx).astype(np.float64)
    current_sensor = x_current_raw[:, sensor_idx]
    actual_delta_raw = actual_next_raw - current_sensor
    mlp_delta_raw = pred_next_raw - current_sensor
    _validate_split_shapes(
        x_current_raw,
        actual_next_raw,
        pred_next_raw,
        labels,
        feature_columns,
        target_columns,
        sensor_idx,
        source=str(path),
    )
    return DetectionSplit(
        x_current_raw,
        actual_next_raw,
        pred_next_raw,
        actual_delta_raw,
        mlp_delta_raw,
        labels,
        feature_columns,
        target_columns,
        sensor_idx,
        str(path),
        split,
    )


def _validate_split_shapes(
    x: np.ndarray,
    actual_next: np.ndarray,
    mlp_next: np.ndarray,
    labels: np.ndarray | None,
    feature_columns: list[str],
    target_columns: list[str],
    sensor_idx: list[int],
    *,
    source: str,
) -> None:
    if x.ndim != 2 or x.shape[1] != len(feature_columns):
        raise ValueError(f"{source}: X must be [N, {len(feature_columns)}], got {x.shape}")
    if actual_next.shape != mlp_next.shape:
        raise ValueError(f"{source}: actual/pred shape mismatch {actual_next.shape} vs {mlp_next.shape}")
    if actual_next.ndim != 2 or actual_next.shape[1] != len(target_columns):
        raise ValueError(f"{source}: targets must be [N, {len(target_columns)}], got {actual_next.shape}")
    if x.shape[0] != actual_next.shape[0]:
        raise ValueError(f"{source}: X/target row mismatch {x.shape[0]} vs {actual_next.shape[0]}")
    if labels is not None and labels.shape[0] != x.shape[0]:
        raise ValueError(f"{source}: labels length {labels.shape[0]} != samples {x.shape[0]}")
    for j, feature_idx in enumerate(sensor_idx):
        if target_columns[j] != feature_columns[int(feature_idx)]:
            raise ValueError(
                f"{source}: target_columns[{j}]={target_columns[j]!r} != "
                f"feature_columns[sensor_idx[{j}]]={feature_columns[int(feature_idx)]!r}"
            )


def package_versions() -> dict[str, str]:
    names = ["numpy", "pandas", "sklearn", "sympy", "torch"]
    out: dict[str, str] = {"python": sys.version.split()[0], "platform": platform.platform()}
    for name in names:
        dist_name = "scikit-learn" if name == "sklearn" else name
        try:
            out[name] = metadata.version(dist_name)
        except metadata.PackageNotFoundError:
            out[name] = "not-installed"
    return out
