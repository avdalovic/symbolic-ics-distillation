from __future__ import annotations

import csv
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np
import torch

from ics_symbolic_distill.data.normalization import inverse_normalize_features, load_normalization_stats
from ics_symbolic_distill.models import DirectTrajectoryPredictor
from ics_symbolic_distill.training.config import load_resolved_config
from ics_symbolic_distill.utils import get_device


@dataclass(frozen=True)
class MLPAttributionData:
    inputs_norm_aligned: np.ndarray
    inputs_raw_aligned: np.ndarray
    distill_inputs_raw: np.ndarray
    feature_columns: list[str]
    target_columns: list[str]
    sensor_idx: list[int]
    normalization_stats: dict[str, Any]
    alignment_start: int
    alignment_max_abs_raw: float


def read_json(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def write_json(path: str | Path, payload: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def align_last_n(arr: np.ndarray, n: int) -> tuple[np.ndarray, int]:
    """Return the final ``n`` rows and the start offset in the source array."""

    value = np.asarray(arr)
    n = int(n)
    if n <= 0:
        raise ValueError("n must be positive")
    if value.shape[0] < n:
        raise ValueError(f"Cannot align {value.shape[0]} rows to requested n={n}")
    start = int(value.shape[0] - n)
    return value[start:], start


def _columns_from_checkpoint(checkpoint: str | Path) -> dict[str, Any]:
    columns_path = Path(checkpoint).expanduser().resolve().parent / "columns.json"
    if not columns_path.exists():
        raise FileNotFoundError(f"Missing checkpoint columns.json: {columns_path}")
    return read_json(columns_path)


def load_mlp_model(
    *,
    checkpoint: str | Path,
    config: str | Path,
    device: torch.device | None = None,
) -> tuple[DirectTrajectoryPredictor, dict[str, Any], Any]:
    """Load a trained current-state MLP with strict checkpoint matching."""

    cfg, _ = load_resolved_config(config)
    if str(cfg.model.get("architecture", "")).lower() != "mlp":
        raise ValueError(f"Expected MLP config, got architecture={cfg.model.get('architecture')!r}")
    if int(cfg.model.get("history_len", 1)) != 1:
        raise ValueError(f"Expected current-state MLP history_len=1, got {cfg.model.get('history_len')}")
    if int(cfg.model.get("horizon", 1)) != 1:
        raise ValueError(f"Expected horizon=1, got {cfg.model.get('horizon')}")

    columns = _columns_from_checkpoint(checkpoint)
    feature_columns = [str(x) for x in columns["feature_columns"]]
    sensor_idx = [int(i) for i in columns["sensor_idx"]]
    actuator_idx = [int(i) for i in columns.get("actuator_idx", [])]
    target_columns = [str(x) for x in columns["target_columns"]]
    _validate_sensor_mapping(feature_columns, target_columns, sensor_idx)

    model = DirectTrajectoryPredictor(
        sensor_idx=sensor_idx,
        actuator_idx=actuator_idx,
        num_tags=len(feature_columns),
        history_len=int(cfg.model.get("history_len", 1)),
        horizon=int(cfg.model.get("horizon", 1)),
        hidden_dim=int(cfg.model.get("hidden_dim", 128)),
        num_layers=int(cfg.model.get("num_layers", 1)),
        dropout=float(cfg.model.get("dropout", 0.0)),
        architecture=str(cfg.model.get("architecture", "mlp")),
        score_aggregation=str(cfg.model.get("score_aggregation", "mean")),
        horizon_loss_weights=cfg.model.get("horizon_loss_weights"),
        horizon_weighting=str(cfg.model.get("horizon_weighting", "uniform")),
        horizon_gamma=float(cfg.model.get("horizon_gamma", 0.9)),
        prediction_mode=str(cfg.model.get("prediction_mode", "deterministic")),
        logvar_min=float(cfg.model.get("logvar_min", -6.0)),
        logvar_max=float(cfg.model.get("logvar_max", 3.0)),
        transformer_heads=int(cfg.model.get("transformer_heads", 4)),
        transformer_ff_dim=int(cfg.model.get("transformer_ff_dim", 128)),
        ae_hidden_1=int(cfg.model.get("ae_hidden_1", 64)),
        ae_hidden_2=int(cfg.model.get("ae_hidden_2", 16)),
    )
    target_device = device or get_device(cfg.train.get("device", "cpu"))
    model = model.to(target_device)
    state = torch.load(Path(checkpoint).expanduser().resolve(), map_location=target_device, weights_only=False)
    model.load_state_dict(state["model_state"], strict=True)
    model.eval()
    return model, state, cfg


def _validate_sensor_mapping(
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    sensor_idx: Sequence[int],
) -> None:
    if len(target_columns) != len(sensor_idx):
        raise ValueError("target_columns length does not match sensor_idx length")
    errors = []
    for target_i, feature_i in enumerate(sensor_idx):
        if int(feature_i) < 0 or int(feature_i) >= len(feature_columns):
            errors.append(f"sensor_idx[{target_i}]={feature_i} out of range")
        elif str(target_columns[target_i]) != str(feature_columns[int(feature_i)]):
            errors.append(
                f"target_columns[{target_i}]={target_columns[target_i]!r} != "
                f"feature_columns[sensor_idx[{target_i}]]={feature_columns[int(feature_i)]!r}"
            )
    if errors:
        raise ValueError("Invalid sensor target mapping:\n- " + "\n- ".join(errors))


def load_aligned_mlp_inputs(
    *,
    mlp_export: str | Path,
    distill_dir: str | Path,
    checkpoint: str | Path,
) -> MLPAttributionData:
    """Load normalized MLP inputs and align them to the distillation overlap.

    MLP export arrays are longer than the GRU overlap because history_len=1.
    Alignment follows the distillation bridge convention: take the last
    ``N_overlap`` MLP samples.
    """

    mlp_root = Path(mlp_export)
    distill_root = Path(distill_dir)
    mlp_meta = read_json(mlp_root / "metadata.json")
    split = str(mlp_meta.get("split_name", "val")).lower()
    inputs_norm = np.load(mlp_root / f"{split}_inputs.npy").astype(np.float32)
    if inputs_norm.ndim != 3 or inputs_norm.shape[2] != 1:
        raise ValueError(f"Expected MLP inputs [N, F, 1], got {inputs_norm.shape}")

    distill_inputs_raw = np.load(distill_root / "distill_inputs_current_raw.npy").astype(np.float32)
    feature_columns = [str(x) for x in read_json(distill_root / "distill_feature_columns.json")]
    target_columns = [str(x) for x in read_json(distill_root / "distill_target_columns.json")]
    sensor_idx = [int(i) for i in read_json(distill_root / "distill_sensor_idx.json")]
    _validate_sensor_mapping(feature_columns, target_columns, sensor_idx)

    columns = _columns_from_checkpoint(checkpoint)
    if [str(x) for x in columns["feature_columns"]] != feature_columns:
        raise ValueError("MLP checkpoint feature columns do not match distillation feature columns")
    if [str(x) for x in columns["target_columns"]] != target_columns:
        raise ValueError("MLP checkpoint target columns do not match distillation target columns")
    if [int(i) for i in columns["sensor_idx"]] != sensor_idx:
        raise ValueError("MLP checkpoint sensor_idx does not match distillation sensor_idx")

    stats_path = Path(checkpoint).expanduser().resolve().parent / "normalization_stats.npz"
    stats = load_normalization_stats(stats_path)
    if [str(x) for x in stats.get("feature_columns", [])] != feature_columns:
        raise ValueError("Normalization feature columns do not match distillation feature columns")

    aligned_norm, start = align_last_n(inputs_norm, distill_inputs_raw.shape[0])
    aligned_raw = inverse_normalize_features(aligned_norm, stats)[:, :, 0].astype(np.float32)
    max_abs = float(np.max(np.abs(aligned_raw - distill_inputs_raw)))
    if not np.allclose(aligned_raw, distill_inputs_raw, atol=1e-5, rtol=1e-5):
        raise ValueError(
            "Aligned normalized MLP inputs do not invert to distillation raw inputs; "
            f"max_abs_diff={max_abs}"
        )

    return MLPAttributionData(
        inputs_norm_aligned=aligned_norm,
        inputs_raw_aligned=aligned_raw,
        distill_inputs_raw=distill_inputs_raw,
        feature_columns=feature_columns,
        target_columns=target_columns,
        sensor_idx=sensor_idx,
        normalization_stats=stats,
        alignment_start=start,
        alignment_max_abs_raw=max_abs,
    )


def _model_single_output(model: torch.nn.Module, x_single: torch.Tensor) -> torch.Tensor:
    output = model(x_single.unsqueeze(0))
    if output.ndim != 3 or output.shape[0] != 1 or output.shape[1] != 1:
        raise ValueError(f"Expected model output [1, 1, targets], got {tuple(output.shape)}")
    return output[0, 0, :]


def _vmap_jacobian_batch(model: torch.nn.Module, x_batch: torch.Tensor) -> torch.Tensor:
    from torch.func import jacrev, vmap

    jac_fn = jacrev(lambda x: _model_single_output(model, x))
    return vmap(jac_fn)(x_batch)


def _loop_jacobian_batch(model: torch.nn.Module, x_batch: torch.Tensor) -> torch.Tensor:
    jacobians = []
    for i in range(x_batch.shape[0]):
        x_single = x_batch[i].detach().clone().requires_grad_(True)
        y = _model_single_output(model, x_single)
        rows = []
        for target_idx in range(y.shape[0]):
            grad = torch.autograd.grad(y[target_idx], x_single, retain_graph=True)[0]
            rows.append(grad)
        jacobians.append(torch.stack(rows, dim=0))
    return torch.stack(jacobians, dim=0)


def convert_norm_grad_to_raw_sensitivity(
    norm_grad: np.ndarray,
    safe_std: np.ndarray,
    sensor_idx: Sequence[int],
) -> np.ndarray:
    """Apply dy_raw/dx_raw = dy_norm/dx_norm * std_target/std_input."""

    grad = np.asarray(norm_grad, dtype=np.float64)
    std = np.asarray(safe_std, dtype=np.float64)
    idx = np.asarray([int(i) for i in sensor_idx], dtype=np.int64)
    if grad.shape != (idx.shape[0], std.shape[0]):
        raise ValueError(f"Expected gradient shape {(idx.shape[0], std.shape[0])}, got {grad.shape}")
    if np.any(std <= 0.0):
        raise ValueError("safe_std must be strictly positive")
    ratio = std[idx].reshape(-1, 1) / std.reshape(1, -1)
    return (grad * ratio).astype(np.float32)


def delta_sensitivity_from_raw_next(
    raw_next_sensitivity: np.ndarray,
    sensor_idx: Sequence[int],
) -> np.ndarray:
    """Convert next-value sensitivity to delta sensitivity by subtracting self identity."""

    out = np.asarray(raw_next_sensitivity, dtype=np.float32).copy()
    for target_idx, feature_idx in enumerate(sensor_idx):
        out[target_idx, int(feature_idx)] -= 1.0
    return out


def compute_gradient_attribution(
    model: torch.nn.Module,
    inputs_norm: np.ndarray,
    stats: Mapping[str, Any],
    sensor_idx: Sequence[int],
    *,
    sample_size: int | None = None,
    seed: int = 0,
    batch_size: int = 128,
    device: torch.device | None = None,
) -> dict[str, Any]:
    """Compute E[|per-sample input gradient|] for a normalized MLP.

    This intentionally computes per-sample Jacobians and averages absolute
    values. It does not backpropagate through a batch-summed output, which can
    hide dependencies when signs differ across samples.
    """

    model.eval()
    x_all = np.asarray(inputs_norm, dtype=np.float32)
    if x_all.ndim != 3:
        raise ValueError(f"inputs_norm must be [N, F, H], got {x_all.shape}")
    n_total = x_all.shape[0]
    if sample_size is None or int(sample_size) <= 0 or int(sample_size) >= n_total:
        sample_indices = np.arange(n_total, dtype=np.int64)
    else:
        rng = np.random.default_rng(int(seed))
        sample_indices = np.sort(rng.choice(n_total, size=int(sample_size), replace=False)).astype(np.int64)
    x_sample = x_all[sample_indices]

    target_device = device or next(model.parameters()).device
    std = np.asarray(stats["std"], dtype=np.float64)
    idx = np.asarray([int(i) for i in sensor_idx], dtype=np.int64)
    ratio = torch.as_tensor(std[idx].reshape(1, -1, 1) / std.reshape(1, 1, -1), dtype=torch.float32, device=target_device)

    norm_abs_sum = None
    raw_next_abs_sum = None
    raw_delta_abs_sum = None
    n_seen = 0
    used_backend = "torch.func.vmap_jacrev"
    for start in range(0, x_sample.shape[0], int(batch_size)):
        batch_np = x_sample[start : start + int(batch_size)]
        x_batch = torch.as_tensor(batch_np, dtype=torch.float32, device=target_device)
        try:
            jac = _vmap_jacobian_batch(model, x_batch)
        except Exception:
            used_backend = "autograd_sample_loop"
            jac = _loop_jacobian_batch(model, x_batch)
        if jac.ndim != 4 or jac.shape[-1] != 1:
            raise ValueError(f"Expected Jacobian [B, T, F, 1], got {tuple(jac.shape)}")
        jac = jac[..., 0]  # [B, targets, features]
        raw_next = jac * ratio
        raw_delta = raw_next.clone()
        target_arange = torch.arange(len(idx), device=target_device)
        raw_delta[:, target_arange, torch.as_tensor(idx, device=target_device)] -= 1.0

        norm_batch = jac.abs().sum(dim=0).detach().cpu().numpy()
        raw_next_batch = raw_next.abs().sum(dim=0).detach().cpu().numpy()
        raw_delta_batch = raw_delta.abs().sum(dim=0).detach().cpu().numpy()
        if norm_abs_sum is None:
            norm_abs_sum = np.zeros_like(norm_batch, dtype=np.float64)
            raw_next_abs_sum = np.zeros_like(raw_next_batch, dtype=np.float64)
            raw_delta_abs_sum = np.zeros_like(raw_delta_batch, dtype=np.float64)
        norm_abs_sum += norm_batch
        raw_next_abs_sum += raw_next_batch
        raw_delta_abs_sum += raw_delta_batch
        n_seen += int(x_batch.shape[0])

    if n_seen <= 0 or norm_abs_sum is None or raw_next_abs_sum is None or raw_delta_abs_sum is None:
        raise RuntimeError("No samples were processed for gradient attribution")
    return {
        "grad_norm_next": (norm_abs_sum / n_seen).astype(np.float32),
        "sensitivity_raw_next": (raw_next_abs_sum / n_seen).astype(np.float32),
        "sensitivity_raw_delta": (raw_delta_abs_sum / n_seen).astype(np.float32),
        "sample_indices": sample_indices,
        "num_samples": int(n_seen),
        "backend": used_backend,
    }


def rank_top_features(
    matrix: np.ndarray,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    sensor_idx: Sequence[int],
    *,
    top_k: int = 10,
) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    if values.shape != (len(target_columns), len(feature_columns)):
        raise ValueError(f"Expected matrix shape {(len(target_columns), len(feature_columns))}, got {values.shape}")
    targets = []
    for target_idx, target_name in enumerate(target_columns):
        order = np.argsort(-values[target_idx], kind="mergesort")[: int(top_k)]
        top_features = []
        for rank, feature_idx in enumerate(order, start=1):
            top_features.append(
                {
                    "rank": int(rank),
                    "feature": str(feature_columns[int(feature_idx)]),
                    "feature_index": int(feature_idx),
                    "value": float(values[target_idx, int(feature_idx)]),
                    "is_self_feature": int(feature_idx) == int(sensor_idx[target_idx]),
                }
            )
        targets.append(
            {
                "target": str(target_name),
                "target_index": int(target_idx),
                "target_feature_index": int(sensor_idx[target_idx]),
                "top_features": top_features,
            }
        )
    return {"targets": targets}


def detect_floored_channels(
    stats: Mapping[str, Any],
    feature_columns: Sequence[str],
    *,
    tolerance: float = 1e-8,
) -> dict[str, Any]:
    """Identify channels whose saved safe std is at the configured floor."""

    if str(stats.get("normalization_mode", "")).lower() not in {"zscore", "standard"}:
        return {
            "std_floor": stats.get("std_floor"),
            "tolerance": float(tolerance),
            "indices": [],
            "channels": [],
            "reason": "Normalization mode is not zscore; std_floor masking was not applied.",
        }
    if stats.get("std") is None:
        raise ValueError("zscore normalization stats require std")
    std = np.asarray(stats["std"], dtype=np.float64)
    std_floor = float(stats.get("std_floor", 0.0) or 0.0)
    if len(feature_columns) != std.shape[0]:
        raise ValueError("feature_columns length does not match std length")
    indices = np.where(np.abs(std - std_floor) <= float(tolerance))[0].astype(int).tolist()
    return {
        "std_floor": std_floor,
        "tolerance": float(tolerance),
        "indices": indices,
        "channels": [str(feature_columns[i]) for i in indices],
        "std_values": {str(feature_columns[i]): float(std[i]) for i in indices},
        "reason": (
            "Saved z-score std equals std_floor, so raw-unit sensitivities are inflated "
            "by the chain-rule factor std_target / std_input."
        ),
    }


def rank_top_features_excluding(
    matrix: np.ndarray,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    sensor_idx: Sequence[int],
    *,
    exclude_indices: Sequence[int],
    excluded_reason: str,
    top_k: int = 10,
) -> dict[str, Any]:
    values = np.asarray(matrix, dtype=np.float64)
    excluded = {int(i) for i in exclude_indices}
    if values.shape != (len(target_columns), len(feature_columns)):
        raise ValueError(f"Expected matrix shape {(len(target_columns), len(feature_columns))}, got {values.shape}")
    targets = []
    for target_idx, target_name in enumerate(target_columns):
        ordered = [int(i) for i in np.argsort(-values[target_idx], kind="mergesort") if int(i) not in excluded]
        top_features = []
        for rank, feature_idx in enumerate(ordered[: int(top_k)], start=1):
            top_features.append(
                {
                    "rank": int(rank),
                    "feature": str(feature_columns[feature_idx]),
                    "feature_index": int(feature_idx),
                    "value": float(values[target_idx, feature_idx]),
                    "is_self_feature": feature_idx == int(sensor_idx[target_idx]),
                    "excluded_from_nonfloored_ranking": False,
                }
            )
        targets.append(
            {
                "target": str(target_name),
                "target_index": int(target_idx),
                "target_feature_index": int(sensor_idx[target_idx]),
                "top_features": top_features,
            }
        )
    return {
        "targets": targets,
        "excluded_feature_indices": sorted(excluded),
        "excluded_features": [str(feature_columns[i]) for i in sorted(excluded)],
        "excluded_reason": excluded_reason,
    }


def rank_of_feature(
    matrix: np.ndarray,
    target_name: str,
    feature_name: str,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
) -> int | None:
    if target_name not in target_columns or feature_name not in feature_columns:
        return None
    values = np.asarray(matrix)
    target_idx = int(list(target_columns).index(target_name))
    feature_idx = int(list(feature_columns).index(feature_name))
    order = np.argsort(-values[target_idx], kind="mergesort")
    where = np.where(order == feature_idx)[0]
    return None if where.size == 0 else int(where[0] + 1)


def rank_of_feature_excluding(
    matrix: np.ndarray,
    target_name: str,
    feature_name: str,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    *,
    exclude_indices: Sequence[int],
) -> int | None:
    if target_name not in target_columns or feature_name not in feature_columns:
        return None
    feature_idx = int(list(feature_columns).index(feature_name))
    if feature_idx in {int(i) for i in exclude_indices}:
        return None
    values = np.asarray(matrix)
    target_idx = int(list(target_columns).index(target_name))
    order = [int(i) for i in np.argsort(-values[target_idx], kind="mergesort") if int(i) not in set(exclude_indices)]
    for rank, idx in enumerate(order, start=1):
        if idx == feature_idx:
            return int(rank)
    return None


def target_top_features(
    matrix: np.ndarray,
    target_name: str,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    sensor_idx: Sequence[int],
    *,
    top_k: int = 10,
    exclude_indices: Sequence[int] | None = None,
) -> list[dict[str, Any]]:
    if target_name not in target_columns:
        raise ValueError(f"Unknown target: {target_name}")
    values = np.asarray(matrix, dtype=np.float64)
    target_idx = int(list(target_columns).index(target_name))
    excluded = {int(i) for i in (exclude_indices or [])}
    ordered = [int(i) for i in np.argsort(-values[target_idx], kind="mergesort") if int(i) not in excluded]
    return [
        {
            "rank": int(rank),
            "feature": str(feature_columns[feature_idx]),
            "feature_index": int(feature_idx),
            "value": float(values[target_idx, feature_idx]),
            "is_self_feature": feature_idx == int(sensor_idx[target_idx]),
        }
        for rank, feature_idx in enumerate(ordered[: int(top_k)], start=1)
    ]


def summarize_target_attribution(
    *,
    target_name: str,
    matrices: Mapping[str, np.ndarray],
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    sensor_idx: Sequence[int],
    floored: Mapping[str, Any],
    watched_features: Sequence[str],
    top_k: int = 10,
) -> dict[str, Any]:
    exclude_indices = [int(i) for i in floored.get("indices", [])]
    views: dict[str, Any] = {}
    for view_name, matrix in matrices.items():
        include_nonfloored = view_name in {
            "raw_next_sensitivity",
            "raw_delta_sensitivity",
            "corr_mlp_pred_delta",
        }
        view = {
            "top10": target_top_features(
                matrix,
                target_name,
                feature_columns,
                target_columns,
                sensor_idx,
                top_k=top_k,
            ),
            "ranks": {},
        }
        if include_nonfloored:
            view["top10_nonfloored"] = target_top_features(
                matrix,
                target_name,
                feature_columns,
                target_columns,
                sensor_idx,
                top_k=top_k,
                exclude_indices=exclude_indices,
            )
        for feature in watched_features:
            full_rank = rank_of_feature(matrix, target_name, feature, feature_columns, target_columns)
            nonfloored_rank = rank_of_feature_excluding(
                matrix,
                target_name,
                feature,
                feature_columns,
                target_columns,
                exclude_indices=exclude_indices,
            )
            feature_idx = list(feature_columns).index(feature) if feature in feature_columns else None
            view["ranks"][str(feature)] = {
                "rank": full_rank,
                "rank_nonfloored": nonfloored_rank,
                "is_floored_channel": feature_idx in set(exclude_indices) if feature_idx is not None else None,
            }
        views[view_name] = view
    return {
        "target": target_name,
        "target_index": int(list(target_columns).index(target_name)),
        "target_feature_index": int(sensor_idx[list(target_columns).index(target_name)]),
        "floored_channels": floored,
        "views": views,
    }


def build_target_attribution_summary(
    *,
    target_name: str,
    matrices: Mapping[str, np.ndarray],
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    sensor_idx: Sequence[int],
    floored: Mapping[str, Any],
    watched_features: Sequence[str],
    top_k: int = 10,
) -> dict[str, Any]:
    required = {
        "grad_norm_next": "top10_normalized_next_gradient_features",
        "raw_next_sensitivity": "top10_raw_next_sensitivity_features",
        "raw_delta_sensitivity": "top10_raw_delta_sensitivity_features",
        "corr_mlp_pred_delta": "top10_corr_mlp_pred_delta_features",
    }
    missing = [name for name in required if name not in matrices]
    if missing:
        raise ValueError(f"Missing attribution matrices for summary: {missing}")
    exclude_indices = [int(i) for i in floored.get("indices", [])]
    target_idx = int(list(target_columns).index(target_name))
    target_feature_idx = int(sensor_idx[target_idx])

    summary: dict[str, Any] = {
        "target": target_name,
        "target_index": target_idx,
        "target_feature_index": target_feature_idx,
        "target_feature": str(feature_columns[target_feature_idx]),
        "floored_channels": floored,
        "top_k": int(top_k),
    }
    for matrix_name, output_key in required.items():
        summary[output_key] = target_top_features(
            matrices[matrix_name],
            target_name,
            feature_columns,
            target_columns,
            sensor_idx,
            top_k=top_k,
        )
    summary["top10_raw_next_sensitivity_nonfloored_features"] = target_top_features(
        matrices["raw_next_sensitivity"],
        target_name,
        feature_columns,
        target_columns,
        sensor_idx,
        top_k=top_k,
        exclude_indices=exclude_indices,
    )
    summary["top10_raw_delta_sensitivity_nonfloored_features"] = target_top_features(
        matrices["raw_delta_sensitivity"],
        target_name,
        feature_columns,
        target_columns,
        sensor_idx,
        top_k=top_k,
        exclude_indices=exclude_indices,
    )
    summary["top10_corr_mlp_pred_delta_nonfloored_features"] = target_top_features(
        matrices["corr_mlp_pred_delta"],
        target_name,
        feature_columns,
        target_columns,
        sensor_idx,
        top_k=top_k,
        exclude_indices=exclude_indices,
    )

    rank_views = {
        "normalized_next_gradient": "grad_norm_next",
        "raw_next_sensitivity": "raw_next_sensitivity",
        "raw_next_sensitivity_nonfloored": "raw_next_sensitivity",
        "raw_delta_sensitivity": "raw_delta_sensitivity",
        "raw_delta_sensitivity_nonfloored": "raw_delta_sensitivity",
        "corr_mlp_pred_delta": "corr_mlp_pred_delta",
        "corr_mlp_pred_delta_nonfloored": "corr_mlp_pred_delta",
    }
    summary["ranks"] = {}
    for feature in watched_features:
        feature_ranks: dict[str, int | None] = {}
        for view_name, matrix_name in rank_views.items():
            if view_name.endswith("_nonfloored"):
                feature_ranks[view_name] = rank_of_feature_excluding(
                    matrices[matrix_name],
                    target_name,
                    feature,
                    feature_columns,
                    target_columns,
                    exclude_indices=exclude_indices,
                )
            else:
                feature_ranks[view_name] = rank_of_feature(
                    matrices[matrix_name],
                    target_name,
                    feature,
                    feature_columns,
                    target_columns,
                )
        feature_idx = list(feature_columns).index(feature) if feature in feature_columns else None
        feature_ranks["is_floored_channel"] = feature_idx in set(exclude_indices) if feature_idx is not None else None
        summary["ranks"][str(feature)] = feature_ranks

    if exclude_indices:
        interpretation = (
            "Raw-unit sensitivity is mathematically valid but diagnostic only for top-k selection here: "
            "several input channels use std_floor, which inflates raw sensitivity by std_target/std_input. "
            "For first PySR top-k selection, prefer MLP predicted-delta correlation, with normalized "
            "gradient as a model-internal diagnostic."
        )
        recommendation = "Use corr_mlp_pred_delta_nonfloored for attribution-guided PySR top-k selection."
    else:
        interpretation = (
            "No floored channels were detected; raw sensitivity and correlation can both be inspected for top-k selection."
        )
        recommendation = "Inspect normalized gradient, raw delta sensitivity, and predicted-delta correlation together."
    summary["interpretation"] = interpretation
    summary["recommendation"] = recommendation
    return summary


def write_rankings_json(
    path: str | Path,
    matrix: np.ndarray,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    sensor_idx: Sequence[int],
    *,
    attribution_type: str,
    top_k: int = 10,
) -> dict[str, Any]:
    payload = rank_top_features(
        matrix,
        feature_columns,
        target_columns,
        sensor_idx,
        top_k=top_k,
    )
    payload["attribution_type"] = attribution_type
    write_json(path, payload)
    return payload


def write_rankings_json_excluding(
    path: str | Path,
    matrix: np.ndarray,
    feature_columns: Sequence[str],
    target_columns: Sequence[str],
    sensor_idx: Sequence[int],
    *,
    attribution_type: str,
    exclude_indices: Sequence[int],
    excluded_reason: str,
    top_k: int = 10,
) -> dict[str, Any]:
    payload = rank_top_features_excluding(
        matrix,
        feature_columns,
        target_columns,
        sensor_idx,
        exclude_indices=exclude_indices,
        excluded_reason=excluded_reason,
        top_k=top_k,
    )
    payload["attribution_type"] = attribution_type
    write_json(path, payload)
    return payload


def write_topk_csv(
    path: str | Path,
    ranking_payloads: Mapping[str, Mapping[str, Any]],
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "attribution_type",
                "target",
                "rank",
                "feature",
                "feature_index",
                "value",
                "is_self_feature",
            ],
        )
        writer.writeheader()
        for attr_type, payload in ranking_payloads.items():
            for target in payload["targets"]:
                for item in target["top_features"]:
                    writer.writerow(
                        {
                            "attribution_type": attr_type,
                            "target": target["target"],
                            "rank": item["rank"],
                            "feature": item["feature"],
                            "feature_index": item["feature_index"],
                            "value": item["value"],
                            "is_self_feature": item["is_self_feature"],
                        }
                    )
    return out


def absolute_pearson_correlation(features: np.ndarray, targets: np.ndarray) -> np.ndarray:
    """Absolute Pearson correlation with constant columns mapped to zero."""

    x = np.asarray(features, dtype=np.float64)
    y = np.asarray(targets, dtype=np.float64)
    if x.ndim != 2 or y.ndim != 2:
        raise ValueError("features and targets must both be 2D")
    if x.shape[0] != y.shape[0]:
        raise ValueError("features and targets must have the same number of rows")
    x_center = x - x.mean(axis=0, keepdims=True)
    y_center = y - y.mean(axis=0, keepdims=True)
    x_norm = np.sqrt(np.sum(x_center * x_center, axis=0))
    y_norm = np.sqrt(np.sum(y_center * y_center, axis=0))
    numerator = y_center.T @ x_center
    denom = y_norm.reshape(-1, 1) * x_norm.reshape(1, -1)
    out = np.zeros_like(numerator, dtype=np.float64)
    np.divide(numerator, denom, out=out, where=denom > 0.0)
    out = np.abs(out)
    out[~np.isfinite(out)] = 0.0
    return out.astype(np.float32)
