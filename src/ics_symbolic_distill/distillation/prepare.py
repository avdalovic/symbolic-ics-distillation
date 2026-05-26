from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from ics_symbolic_distill.data.normalization import (
    describe_normalization_stats,
    inverse_normalize_features,
    inverse_normalize_targets,
    load_normalization_stats,
    normalization_formula,
)


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Mapping[str, Any] | Sequence[Any]) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def squeeze_horizon_one(arr: np.ndarray, name: str) -> np.ndarray:
    """Convert ``[N, 1, T]`` horizon-one arrays to ``[N, T]``."""

    value = np.asarray(arr)
    if value.ndim == 2:
        return value
    if value.ndim == 3 and value.shape[1] == 1:
        return value[:, 0, :]
    raise ValueError(f"{name} must have shape [N, T] or [N, 1, T], got {value.shape}")


def align_mlp_to_gru_overlap(mlp_arr: np.ndarray, n_gru: int, name: str) -> np.ndarray:
    """Align current-state MLP samples to GRU anchors by taking the last ``n_gru`` rows."""

    value = np.asarray(mlp_arr)
    if value.shape[0] < int(n_gru):
        raise ValueError(f"{name} has {value.shape[0]} samples but GRU requires {n_gru}")
    return value[-int(n_gru) :]


def _safe_name(prefix: str, name: str) -> str:
    cleaned = re.sub(r"[^0-9A-Za-z_]", "_", str(name))
    if not cleaned or cleaned[0].isdigit():
        cleaned = f"v_{cleaned}"
    return f"{prefix}_{cleaned}"


def build_name_mapping(
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    sensor_idx: Sequence[int],
) -> dict[str, Any]:
    feature_to_index = {str(name): int(i) for i, name in enumerate(feature_columns)}
    target_to_index = {str(name): int(i) for i, name in enumerate(target_columns)}
    target_index_to_feature_index = {str(i): int(sensor_idx[i]) for i in range(len(sensor_idx))}
    delta_names = [f"{name}_delta" for name in target_columns]
    next_names = [f"{name}_next" for name in target_columns]
    return {
        "feature_name_to_index": feature_to_index,
        "target_name_to_index": target_to_index,
        "target_index_to_feature_index": target_index_to_feature_index,
        "target_name_to_feature_index": {
            str(target_columns[i]): int(sensor_idx[i]) for i in range(len(target_columns))
        },
        "current_feature_safe_names": {
            str(name): _safe_name("x", str(name)) for name in feature_columns
        },
        "delta_target_safe_names": {
            str(name): _safe_name("y", str(name)) for name in delta_names
        },
        "next_target_safe_names": {
            str(name): _safe_name("y", str(name)) for name in next_names
        },
        "safe_name_to_original": {
            **{_safe_name("x", str(name)): str(name) for name in feature_columns},
            **{_safe_name("y", str(name)): str(name) for name in delta_names},
            **{_safe_name("y", str(name)): str(name) for name in next_names},
        },
    }


def compute_temporal_summary_features(
    window_raw: np.ndarray,
    feature_columns: Sequence[str],
    *,
    dt_model_step: float = 1.0,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Create explicit temporal/rate summaries from raw GRU windows.

    ``window_raw`` must be shaped ``[N, F, H]`` with the latest timestep at
    ``window_raw[:, :, -1]``.
    """

    window = np.asarray(window_raw, dtype=np.float32)
    if window.ndim != 3:
        raise ValueError(f"window_raw must be [N, F, H], got {window.shape}")
    n_samples, num_features, history_len = window.shape
    if len(feature_columns) != num_features:
        raise ValueError("feature_columns length does not match window feature dimension")
    if history_len < 11:
        raise ValueError("history_len must be at least 11 for delta/rate/std windows up to 10")

    dt = float(dt_model_step)
    if dt <= 0:
        raise ValueError("dt_model_step must be positive")

    current = window[:, :, -1]
    operations: list[tuple[str, np.ndarray]] = [("current", current)]
    for k in (1, 5, 10):
        delta = current - window[:, :, -(k + 1)]
        operations.append((f"delta_{k}", delta))
    for k in (1, 5, 10):
        delta = current - window[:, :, -(k + 1)]
        operations.append((f"rate_{k}", delta / (k * dt)))
    for k in (5, 10):
        operations.append((f"mean_{k}", window[:, :, -k:].mean(axis=2)))
    for k in (5, 10):
        operations.append((f"std_{k}", window[:, :, -k:].std(axis=2)))

    names: list[str] = []
    columns = []
    for feature_idx, feature_name in enumerate(feature_columns):
        for op_name, values in operations:
            names.append(f"{feature_name}_{op_name}")
            columns.append(values[:, feature_idx])
    features = np.stack(columns, axis=1).astype(np.float32)
    mapping = {
        "temporal_feature_name_to_index": {name: i for i, name in enumerate(names)},
        "temporal_feature_safe_names": {name: _safe_name("x", name) for name in names},
        "safe_name_to_original": {_safe_name("x", name): name for name in names},
        "history_len": int(history_len),
        "dt_model_step": dt,
        "operations": [name for name, _ in operations],
    }
    if features.shape != (n_samples, len(names)):
        raise RuntimeError("Internal temporal feature shape mismatch")
    return features, names, mapping


def _subset_temporal_features(
    temporal_features: np.ndarray,
    temporal_names: Sequence[str],
    operations: Sequence[str],
    keep_operations: Sequence[str],
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    keep = set(keep_operations)
    num_ops = len(operations)
    if num_ops <= 0:
        raise ValueError("operations must not be empty")
    if len(temporal_names) % num_ops != 0:
        raise ValueError("temporal_names length must be divisible by operation count")
    if temporal_features.shape[1] != len(temporal_names):
        raise ValueError("temporal_features column count does not match temporal_names")

    keep_indices = [i for i, name in enumerate(temporal_names) if operations[i % num_ops] in keep]
    names = [str(temporal_names[i]) for i in keep_indices]
    features = temporal_features[:, keep_indices].astype(np.float32)
    mapping = {
        "temporal_feature_name_to_index": {name: i for i, name in enumerate(names)},
        "temporal_feature_safe_names": {name: _safe_name("x", name) for name in names},
        "safe_name_to_original": {_safe_name("x", name): name for name in names},
        "source_operations": list(operations),
        "operations": list(keep_operations),
    }
    return features, names, mapping


def compact_temporal_operations(*, dt_model_step: float = 1.0) -> tuple[list[str], str]:
    """Return the conservative compact temporal operation set.

    Current features are excluded because they duplicate
    ``distill_inputs_current_raw.npy``. Rate features are excluded unless a
    future caller records a real physical sampling interval explicitly; the
    current pipeline only records model-step units, so deltas carry the same
    information without redundant linear scaling.
    """

    _ = float(dt_model_step)
    return (
        ["delta_1", "delta_5", "delta_10", "mean_5", "mean_10", "std_5", "std_10"],
        "current is stored separately; rates are excluded because no physical dt_seconds is recorded.",
    )


def compute_compact_temporal_summary_features(
    window_raw: np.ndarray,
    feature_columns: Sequence[str],
    *,
    dt_model_step: float = 1.0,
) -> tuple[np.ndarray, list[str], dict[str, Any]]:
    """Create the compact nonredundant temporal feature matrix."""

    full_features, full_names, full_mapping = compute_temporal_summary_features(
        window_raw,
        feature_columns,
        dt_model_step=dt_model_step,
    )
    keep_operations, reason = compact_temporal_operations(dt_model_step=dt_model_step)
    compact_features, compact_names, compact_mapping = _subset_temporal_features(
        full_features,
        full_names,
        full_mapping["operations"],
        keep_operations,
    )
    compact_mapping.update(
        {
            "history_len": full_mapping["history_len"],
            "dt_model_step": full_mapping["dt_model_step"],
            "compact_policy": reason,
        }
    )
    return compact_features, compact_names, compact_mapping


def build_matrix_name_mapping(names: Sequence[str]) -> dict[str, Any]:
    return {
        "feature_name_to_index": {str(name): i for i, name in enumerate(names)},
        "feature_safe_names": {str(name): _safe_name("x", str(name)) for name in names},
        "safe_name_to_original": {_safe_name("x", str(name)): str(name) for name in names},
    }


def _load_export(export_dir: str | Path) -> dict[str, Any]:
    root = Path(export_dir)
    metadata_path = root / "metadata.json"
    if not metadata_path.exists():
        raise FileNotFoundError(f"Missing export metadata: {metadata_path}")
    metadata = read_json(metadata_path)
    split = str(metadata.get("split_name", "val")).lower()
    payload = {
        "root": root,
        "metadata": metadata,
        "metadata_path": metadata_path,
        "inputs": np.load(root / f"{split}_inputs.npy"),
        "preds": np.load(root / f"{split}_neural_preds.npy"),
        "actual": np.load(root / f"{split}_actual_next.npy"),
    }
    stats_path = metadata.get("normalization_stats_path")
    columns_path = metadata.get("columns_json_path")
    checkpoint_path = metadata.get("checkpoint_path")
    if not stats_path and checkpoint_path:
        stats_path = str(Path(checkpoint_path).parent / "normalization_stats.npz")
    if not columns_path and checkpoint_path:
        columns_path = str(Path(checkpoint_path).parent / "columns.json")
    if not stats_path or not Path(stats_path).exists():
        raise FileNotFoundError(f"Missing normalization stats for export {root}: {stats_path}")
    if not columns_path or not Path(columns_path).exists():
        raise FileNotFoundError(f"Missing columns metadata for export {root}: {columns_path}")
    manifest_path = Path(checkpoint_path).parent / "manifest.json" if checkpoint_path else None
    payload["stats_path"] = Path(stats_path)
    payload["columns_path"] = Path(columns_path)
    payload["manifest_path"] = manifest_path
    payload["stats"] = load_normalization_stats(stats_path)
    payload["columns"] = read_json(columns_path)
    payload["manifest"] = read_json(manifest_path) if manifest_path and manifest_path.exists() else {}
    return payload


def _same_list(name: str, left: Sequence[Any], right: Sequence[Any], errors: list[str]) -> None:
    if list(left) != list(right):
        errors.append(f"{name} differs")


def _validate_exports(gru: Mapping[str, Any], mlp: Mapping[str, Any]) -> None:
    errors: list[str] = []
    gru_meta = gru["metadata"]
    mlp_meta = mlp["metadata"]
    gru_columns = gru["columns"]
    mlp_columns = mlp["columns"]

    _same_list("feature column order", gru_columns.get("feature_columns", []), mlp_columns.get("feature_columns", []), errors)
    _same_list("target column order", gru_columns.get("target_columns", []), mlp_columns.get("target_columns", []), errors)
    _same_list("sensor_idx", gru_columns.get("sensor_idx", []), mlp_columns.get("sensor_idx", []), errors)

    feature_columns = [str(x) for x in gru_columns.get("feature_columns", [])]
    target_columns = [str(x) for x in gru_columns.get("target_columns", [])]
    sensor_idx = [int(i) for i in gru_columns.get("sensor_idx", [])]
    if len(target_columns) != len(sensor_idx):
        errors.append("target column count does not match sensor_idx length")
    for target_i, feature_i in enumerate(sensor_idx[: len(target_columns)]):
        if feature_i >= len(feature_columns):
            errors.append(f"sensor_idx[{target_i}]={feature_i} is out of range")
        elif target_columns[target_i] != feature_columns[feature_i]:
            errors.append(
                f"target_columns[{target_i}]={target_columns[target_i]!r} != "
                f"feature_columns[sensor_idx[{target_i}]]={feature_columns[feature_i]!r}"
            )

    for key in ("normalization_mode", "std_floor", "fit_split"):
        if gru["stats"].get(key) != mlp["stats"].get(key):
            errors.append(f"normalization stat {key} differs")
    for key in ("mean", "std", "median", "iqr", "data_min", "data_max", "minmax_variable_mask"):
        left = gru["stats"].get(key)
        right = mlp["stats"].get(key)
        if left is None and right is None:
            continue
        if (left is None) != (right is None):
            errors.append(f"normalization stat {key} presence differs")
            continue
        if np.asarray(left).shape != np.asarray(right).shape:
            errors.append(f"normalization stat {key} shape differs")
        elif np.asarray(left).size and not np.allclose(np.asarray(left), np.asarray(right), equal_nan=True):
            errors.append(f"normalization stat {key} values differ")

    if str(gru_meta.get("architecture")).lower() != "gru":
        errors.append(f"GRU export architecture is {gru_meta.get('architecture')!r}")
    if str(mlp_meta.get("architecture")).lower() != "mlp":
        errors.append(f"MLP export architecture is {mlp_meta.get('architecture')!r}")
    if int(gru_meta.get("history_len", -1)) != 60:
        errors.append(f"GRU history_len must be 60, got {gru_meta.get('history_len')}")
    if int(mlp_meta.get("history_len", -1)) != 1:
        errors.append(f"MLP history_len must be 1, got {mlp_meta.get('history_len')}")
    if int(gru_meta.get("horizon", -1)) != 1 or int(mlp_meta.get("horizon", -1)) != 1:
        errors.append("Both exports must use horizon=1")
    if len(target_columns) != 25:
        errors.append(f"Expected 25 target sensors, got {len(target_columns)}")

    if errors:
        raise ValueError("Export compatibility check failed:\n- " + "\n- ".join(errors))


def _array_ranges(arr: np.ndarray, names: Sequence[str]) -> dict[str, dict[str, float]]:
    out: dict[str, dict[str, float]] = {}
    values = np.asarray(arr)
    for i, name in enumerate(names):
        col = values[:, i]
        out[str(name)] = {
            "min": float(np.min(col)),
            "mean": float(np.mean(col)),
            "max": float(np.max(col)),
        }
    return out


def _check_finite(named_arrays: Mapping[str, np.ndarray]) -> dict[str, bool]:
    return {name: bool(np.isfinite(value).all()) for name, value in named_arrays.items()}


def lit101_sanity_summary(
    *,
    current_raw: np.ndarray,
    actual_next_raw: np.ndarray,
    actual_delta_raw: np.ndarray,
    pred_next_raw_gru: np.ndarray,
    pred_delta_raw_gru: np.ndarray,
    pred_next_raw_mlp: np.ndarray,
    pred_delta_raw_mlp: np.ndarray,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
) -> dict[str, Any]:
    wanted_features = ["LIT101", "FIT101", "FIT201", "MV101", "P101"]
    feature_idx = {name: int(feature_columns.index(name)) for name in wanted_features if name in feature_columns}
    if "LIT101" not in target_columns:
        raise ValueError("LIT101 is not present in target_columns")
    lit_target_idx = int(target_columns.index("LIT101"))
    lit_feature_idx = int(feature_columns.index("LIT101"))
    sample_idx = np.linspace(0, current_raw.shape[0] - 1, num=min(5, current_raw.shape[0]), dtype=int)

    representative = []
    for idx in sample_idx.tolist():
        row = {
            "sample_index": int(idx),
            "current_raw_LIT101": float(current_raw[idx, lit_feature_idx]),
            "actual_next_raw_LIT101": float(actual_next_raw[idx, lit_target_idx]),
            "actual_delta_raw_LIT101": float(actual_delta_raw[idx, lit_target_idx]),
            "gru_pred_next_raw_LIT101": float(pred_next_raw_gru[idx, lit_target_idx]),
            "gru_pred_delta_raw_LIT101": float(pred_delta_raw_gru[idx, lit_target_idx]),
            "mlp_pred_next_raw_LIT101": float(pred_next_raw_mlp[idx, lit_target_idx]),
            "mlp_pred_delta_raw_LIT101": float(pred_delta_raw_mlp[idx, lit_target_idx]),
        }
        for feature in ["FIT101", "FIT201", "MV101", "P101"]:
            if feature in feature_idx:
                row[f"current_raw_{feature}"] = float(current_raw[idx, feature_idx[feature]])
        representative.append(row)

    def stats(values: np.ndarray) -> dict[str, float]:
        return {
            "min": float(np.min(values)),
            "mean": float(np.mean(values)),
            "max": float(np.max(values)),
        }

    ranges = {
        "current_raw_LIT101": stats(current_raw[:, lit_feature_idx]),
        "actual_delta_raw_LIT101": stats(actual_delta_raw[:, lit_target_idx]),
        "gru_delta_raw_LIT101": stats(pred_delta_raw_gru[:, lit_target_idx]),
        "mlp_delta_raw_LIT101": stats(pred_delta_raw_mlp[:, lit_target_idx]),
    }
    for feature in ["FIT101", "FIT201"]:
        if feature in feature_idx:
            ranges[f"current_raw_{feature}"] = stats(current_raw[:, feature_idx[feature]])

    return {
        "target_index_LIT101": lit_target_idx,
        "feature_indices": feature_idx,
        "representative_samples": representative,
        "min_mean_max": ranges,
    }


def prepare_distillation_data(
    *,
    gru_export: str | Path,
    mlp_export: str | Path,
    out: str | Path,
    dt_model_step: float = 1.0,
) -> dict[str, Any]:
    gru = _load_export(gru_export)
    mlp = _load_export(mlp_export)
    _validate_exports(gru, mlp)

    feature_columns = [str(x) for x in gru["columns"]["feature_columns"]]
    target_columns = [str(x) for x in gru["columns"]["target_columns"]]
    sensor_idx = [int(i) for i in gru["columns"]["sensor_idx"]]
    actuator_idx = [int(i) for i in gru["columns"].get("actuator_idx", [])]
    stats = gru["stats"]

    gru_inputs_norm = np.asarray(gru["inputs"], dtype=np.float32)
    mlp_inputs_norm = np.asarray(mlp["inputs"], dtype=np.float32)
    if gru_inputs_norm.ndim != 3 or gru_inputs_norm.shape[1] != len(feature_columns):
        raise ValueError(f"GRU inputs must be [N, 51, 60]-style, got {gru_inputs_norm.shape}")
    if mlp_inputs_norm.ndim != 3 or mlp_inputs_norm.shape[1] != len(feature_columns) or mlp_inputs_norm.shape[2] != 1:
        raise ValueError(f"MLP inputs must be [N, 51, 1]-style, got {mlp_inputs_norm.shape}")

    gru_pred_norm = squeeze_horizon_one(gru["preds"], "gru predictions")
    gru_actual_norm = squeeze_horizon_one(gru["actual"], "gru actual")
    mlp_pred_norm = align_mlp_to_gru_overlap(squeeze_horizon_one(mlp["preds"], "mlp predictions"), gru_inputs_norm.shape[0], "mlp predictions")
    mlp_actual_norm = align_mlp_to_gru_overlap(squeeze_horizon_one(mlp["actual"], "mlp actual"), gru_inputs_norm.shape[0], "mlp actual")
    mlp_inputs_norm = align_mlp_to_gru_overlap(mlp_inputs_norm, gru_inputs_norm.shape[0], "mlp inputs")

    current_norm_gru = gru_inputs_norm[:, :, -1]
    current_norm_mlp = mlp_inputs_norm[:, :, 0]
    gru_window_raw = inverse_normalize_features(gru_inputs_norm, stats)
    current_raw_gru = inverse_normalize_features(current_norm_gru, stats)
    current_raw_mlp = inverse_normalize_features(current_norm_mlp, stats)

    pred_next_raw_gru = inverse_normalize_targets(gru_pred_norm, stats, sensor_idx)
    actual_next_raw_gru = inverse_normalize_targets(gru_actual_norm, stats, sensor_idx)
    pred_next_raw_mlp = inverse_normalize_targets(mlp_pred_norm, stats, sensor_idx)
    actual_next_raw_mlp = inverse_normalize_targets(mlp_actual_norm, stats, sensor_idx)

    current_sensor_raw = current_raw_gru[:, sensor_idx]
    pred_delta_raw_gru = (pred_next_raw_gru - current_sensor_raw).astype(np.float32)
    pred_delta_raw_mlp = (pred_next_raw_mlp - current_sensor_raw).astype(np.float32)
    actual_delta_raw = (actual_next_raw_gru - current_sensor_raw).astype(np.float32)

    current_alignment_max_abs = float(np.max(np.abs(current_raw_gru - current_raw_mlp)))
    actual_alignment_max_abs = float(np.max(np.abs(actual_next_raw_gru - actual_next_raw_mlp)))

    temporal_features, temporal_names, temporal_mapping = compute_temporal_summary_features(
        gru_window_raw,
        feature_columns,
        dt_model_step=dt_model_step,
    )
    compact_operations, compact_policy = compact_temporal_operations(dt_model_step=dt_model_step)
    compact_temporal_features, compact_temporal_names, compact_temporal_mapping = _subset_temporal_features(
        temporal_features,
        temporal_names,
        temporal_mapping["operations"],
        compact_operations,
    )
    compact_temporal_mapping.update(
        {
            "history_len": temporal_mapping["history_len"],
            "dt_model_step": temporal_mapping["dt_model_step"],
            "compact_policy": compact_policy,
        }
    )
    current_plus_compact_features = np.concatenate(
        [current_raw_gru.astype(np.float32), compact_temporal_features.astype(np.float32)],
        axis=1,
    )
    current_plus_compact_names = feature_columns + compact_temporal_names
    current_plus_compact_mapping = build_matrix_name_mapping(current_plus_compact_names)
    name_mapping = build_name_mapping(feature_columns, target_columns, sensor_idx)
    normalization_summary = describe_normalization_stats(stats, feature_columns)

    arrays = {
        "distill_inputs_current_raw.npy": current_raw_gru.astype(np.float32),
        "distill_pred_next_raw_gru.npy": pred_next_raw_gru.astype(np.float32),
        "distill_pred_delta_raw_gru.npy": pred_delta_raw_gru.astype(np.float32),
        "distill_pred_next_raw_mlp.npy": pred_next_raw_mlp.astype(np.float32),
        "distill_pred_delta_raw_mlp.npy": pred_delta_raw_mlp.astype(np.float32),
        "distill_actual_next_raw.npy": actual_next_raw_gru.astype(np.float32),
        "distill_actual_delta_raw.npy": actual_delta_raw.astype(np.float32),
        "distill_gru_temporal_features_raw.npy": temporal_features.astype(np.float32),
        "distill_gru_temporal_features_compact_raw.npy": compact_temporal_features.astype(np.float32),
        "distill_gru_current_plus_temporal_compact_raw.npy": current_plus_compact_features.astype(np.float32),
    }
    split_name = str(gru["metadata"].get("split_name", "val")).lower()
    if split_name in {"val", "calib", "validation"}:
        arrays.update(
            {
                "distill_inputs_val.npy": current_raw_gru.astype(np.float32),
                "distill_actual_next_val.npy": actual_next_raw_gru.astype(np.float32),
                "distill_pred_next_mlp_val.npy": pred_next_raw_mlp.astype(np.float32),
                "distill_actual_delta_val.npy": actual_delta_raw.astype(np.float32),
                "distill_pred_delta_mlp_val.npy": pred_delta_raw_mlp.astype(np.float32),
            }
        )
    finite_checks = _check_finite(arrays)
    if not all(finite_checks.values()):
        bad = [name for name, ok in finite_checks.items() if not ok]
        raise ValueError(f"Saved arrays contain NaN or inf: {bad}")

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, value in arrays.items():
        np.save(out_dir / name, value)

    feature_names = list(feature_columns)
    delta_target_names = [f"{name}_delta" for name in target_columns]
    next_target_names = [f"{name}_next" for name in target_columns]
    write_json(out_dir / "distill_feature_columns.json", feature_names)
    write_json(out_dir / "distill_target_columns.json", target_columns)
    write_json(out_dir / "distill_sensor_idx.json", sensor_idx)
    write_json(out_dir / "distill_current_feature_names.json", feature_names)
    write_json(out_dir / "distill_delta_target_names.json", delta_target_names)
    write_json(out_dir / "distill_next_target_names.json", next_target_names)
    write_json(out_dir / "distill_name_mapping.json", name_mapping)
    write_json(out_dir / "distill_gru_temporal_feature_names.json", temporal_names)
    write_json(out_dir / "distill_gru_temporal_name_mapping.json", temporal_mapping)
    write_json(out_dir / "distill_gru_temporal_feature_names_compact.json", compact_temporal_names)
    write_json(out_dir / "distill_gru_temporal_name_mapping_compact.json", compact_temporal_mapping)
    write_json(
        out_dir / "distill_gru_current_plus_temporal_compact_feature_names.json",
        current_plus_compact_names,
    )
    write_json(
        out_dir / "distill_gru_current_plus_temporal_compact_name_mapping.json",
        current_plus_compact_mapping,
    )

    lit101 = lit101_sanity_summary(
        current_raw=current_raw_gru,
        actual_next_raw=actual_next_raw_gru,
        actual_delta_raw=actual_delta_raw,
        pred_next_raw_gru=pred_next_raw_gru,
        pred_delta_raw_gru=pred_delta_raw_gru,
        pred_next_raw_mlp=pred_next_raw_mlp,
        pred_delta_raw_mlp=pred_delta_raw_mlp,
        feature_columns=feature_columns,
        target_columns=target_columns,
    )

    floor_channels = set(normalization_summary.get("floor_scale_channels") or [])
    floor_actuator_names = [feature_columns[i] for i in actuator_idx if feature_columns[i] in floor_channels]
    floor_actuator_ranges = _array_ranges(current_raw_gru[:, actuator_idx], [feature_columns[i] for i in actuator_idx])
    if floor_actuator_names:
        floor_actuator_ranges = {name: floor_actuator_ranges[name] for name in floor_actuator_names}

    validation_rule = gru["manifest"].get("validation_split_rule", {})
    metadata = {
        "dataset": gru["metadata"].get("dataset_name"),
        "split": gru["metadata"].get("split_name"),
        "validation_ratio": validation_rule.get("val_ratio"),
        "N_aligned_samples": int(current_raw_gru.shape[0]),
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "sensor_idx": sensor_idx,
        "gru_export_path": str(Path(gru_export)),
        "mlp_export_path": str(Path(mlp_export)),
        "normalization_formula": normalization_formula(stats),
        "normalization_stat_keys": list(stats.get("stat_keys", [])),
        "normalization_summary": normalization_summary,
        "zero_or_near_zero_scale_channels": {
            "zero": normalization_summary.get("zero_scale_channels", []),
            "near_zero": normalization_summary.get("near_zero_scale_channels", []),
            "floor_scale": normalization_summary.get("floor_scale_channels", []),
        },
        "inversion_method_used": "ics_symbolic_distill.data.normalization inverse utilities",
        "alignment_rule": "MLP exports aligned to GRU anchors by taking the last N_gru MLP samples.",
        "dt_model_step": float(dt_model_step),
        "array_shapes": {name: list(value.shape) for name, value in arrays.items()},
        "temporal_feature_counts": {
            "full": int(temporal_features.shape[1]),
            "compact": int(compact_temporal_features.shape[1]),
            "current_plus_compact": int(current_plus_compact_features.shape[1]),
        },
        "temporal_operations": {
            "full": list(temporal_mapping["operations"]),
            "compact": list(compact_operations),
        },
        "temporal_output_notes": [
            "The full temporal output includes current and rate features for completeness.",
            "The compact temporal output excludes current because it duplicates distill_inputs_current_raw.npy.",
            "The compact temporal output excludes rate features because no physical dt_seconds is recorded; with dt_model_step=1, rate_k is a linear rescaling of delta_k.",
            "The current_plus_compact matrix is the recommended default input matrix for GRU temporal-summary surrogate PySR experiments.",
        ],
        "finite_checks": finite_checks,
        "alignment_checks": {
            "current_raw_gru_vs_aligned_mlp_max_abs": current_alignment_max_abs,
            "actual_next_gru_vs_aligned_mlp_max_abs": actual_alignment_max_abs,
            "current_inputs_close": bool(np.allclose(current_raw_gru, current_raw_mlp, atol=1e-5, rtol=1e-5)),
            "actual_next_close": bool(
                np.allclose(actual_next_raw_gru, actual_next_raw_mlp, atol=1e-5, rtol=1e-5)
            ),
        },
        "physical_plausibility_checks": {
            "floor_scale_actuator_ranges": floor_actuator_ranges,
            "all_saved_arrays_finite": bool(all(finite_checks.values())),
        },
        "notes": [
            "MLP arrays are direct current-state distillation targets.",
            "GRU current-state arrays are a GRU current-state projection / temporal-summary surrogate, not full GRU distillation.",
            "Temporal/rate features are provided for more faithful GRU surrogate analysis.",
        ],
        "lit101_sanity": lit101,
    }
    write_json(out_dir / "metadata.json", metadata)

    return {
        "out": str(out_dir),
        "files": sorted([path.name for path in out_dir.iterdir() if path.is_file()]),
        "array_shapes": metadata["array_shapes"],
        "normalization_summary": normalization_summary,
        "alignment_checks": metadata["alignment_checks"],
        "lit101_sanity": lit101,
        "floor_scale_actuator_ranges": floor_actuator_ranges,
    }
