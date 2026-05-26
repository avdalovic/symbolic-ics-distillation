from __future__ import annotations

import json
import warnings
from pathlib import Path
from typing import Optional

import numpy as np
import torch
from omegaconf import DictConfig, OmegaConf
from torch.utils.data import DataLoader

from ics_symbolic_distill.data import TrajectorySplits, make_trajectory_splits
from ics_symbolic_distill.training.config import (
    load_experiment_config,
    load_resolved_config,
)
from ics_symbolic_distill.training.model import (
    build_model,
    evaluate_mse,
    load_model_checkpoint,
    load_normalization_stats,
)
from ics_symbolic_distill.utils import get_device


def _split_dataset(splits: TrajectorySplits, split: str):
    split_name = str(split).lower()
    if split_name == "train":
        return splits.train
    if split_name == "val":
        return splits.val
    if split_name == "test":
        return splits.test
    raise ValueError("split must be one of: train, val, test")


def _loader(dataset, cfg: DictConfig) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=int(cfg.dataset.batch_size),
        shuffle=False,
        num_workers=int(cfg.dataset.num_workers),
        drop_last=False,
    )


def _model_outputs(model, x_window: torch.Tensor, y_future: torch.Tensor):
    if model.architecture in {"autoencoder_sensors", "autoencoder_full"}:
        x_curr = x_window[:, :, -1]
        pred = model.reconstruct_current(x_window)
        if model.architecture == "autoencoder_full":
            pred_sensor = pred.index_select(dim=1, index=model.sensor_idx_tensor)
        else:
            pred_sensor = pred
        target_sensor = x_curr.index_select(dim=1, index=model.sensor_idx_tensor)
        return pred_sensor.unsqueeze(1), target_sensor.unsqueeze(1)

    mu, _ = model.predict_distribution(x_window)
    return mu, y_future


@torch.no_grad()
def _collect(model, loader, device: torch.device):
    inputs_all = []
    preds_all = []
    targets_all = []
    labels_all = []
    model.eval()
    for x_window, y_future, labels in loader:
        x_window = x_window.to(device)
        y_future = y_future.to(device)
        preds, targets = _model_outputs(model, x_window, y_future)
        inputs_all.append(x_window.detach().cpu())
        preds_all.append(preds.detach().cpu())
        targets_all.append(targets.detach().cpu())
        labels_all.append(labels.detach().cpu())

    if not inputs_all:
        raise RuntimeError("No samples available for export")
    return (
        torch.cat(inputs_all, dim=0).numpy(),
        torch.cat(preds_all, dim=0).numpy(),
        torch.cat(targets_all, dim=0).numpy(),
        torch.cat(labels_all, dim=0).numpy(),
    )


def _mse(preds: np.ndarray, targets: np.ndarray) -> float:
    return float(np.mean((np.asarray(preds) - np.asarray(targets)) ** 2))


def _target_definition(cfg: DictConfig, architecture: str) -> str:
    if architecture in {"autoencoder_sensors", "autoencoder_full"}:
        return "current-step sensor reconstruction from the last timestep in the input window"
    horizon = int(cfg.model.get("horizon", cfg.dataset.get("horizon", 1)))
    sample_stride = int(cfg.model.get("sample_stride", 1))
    return (
        "sensor forecast targets at anchor + k * sample_stride for "
        f"k=1..{horizon}, sample_stride={sample_stride}"
    )


def _read_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def _copy_cfg(cfg: DictConfig) -> DictConfig:
    return OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))


def _apply_checkpoint_columns(cfg: DictConfig, columns: dict) -> DictConfig:
    cfg = _copy_cfg(cfg)
    feature_columns = columns.get("feature_columns") or []
    if feature_columns:
        cfg.dataset.tag_columns = list(feature_columns)
    if columns.get("sensor_idx") is not None:
        cfg.dataset.sensor_idx = [int(i) for i in columns["sensor_idx"]]
    if columns.get("actuator_idx") is not None:
        cfg.dataset.actuator_idx = [int(i) for i in columns["actuator_idx"]]
    return cfg


def _effective_model_values(cfg: DictConfig) -> dict:
    return {
        "architecture": str(cfg.model.get("architecture", "gru")).lower(),
        "history_len": int(cfg.model.get("history_len", cfg.dataset.get("window_size", 1))),
        "horizon": int(cfg.model.get("horizon", cfg.dataset.get("horizon", 1))),
        "sample_stride": int(cfg.model.get("sample_stride", 1)),
        "target_sensor_only": bool(cfg.model.get("target_sensor_only", True)),
    }


def _compare_checkpoint_config(state: dict, cfg: DictConfig) -> None:
    ckpt_config = state.get("config")
    if not ckpt_config:
        warnings.warn(
            "Checkpoint has no embedded config; export compatibility checks are limited.",
            RuntimeWarning,
        )
        return
    ckpt_cfg = OmegaConf.create(ckpt_config)
    current = _effective_model_values(cfg)
    saved = _effective_model_values(ckpt_cfg)
    errors = [
        f"{key}: checkpoint={saved[key]!r} export={current[key]!r}"
        for key in ["architecture", "history_len", "horizon", "sample_stride", "target_sensor_only"]
        if saved[key] != current[key]
    ]
    if errors:
        raise ValueError("Checkpoint/export config mismatch:\n- " + "\n- ".join(errors))


def _compare_split_metadata(
    *,
    state: dict,
    splits: TrajectorySplits,
    columns: Optional[dict],
) -> None:
    errors = []
    model_meta = state.get("model_metadata") or {}
    saved_num_tags = model_meta.get("num_tags")
    if saved_num_tags is not None and int(saved_num_tags) != int(splits.num_tags):
        errors.append(f"num_tags: checkpoint={saved_num_tags} export={splits.num_tags}")

    saved_columns = None
    if columns and columns.get("feature_columns"):
        saved_columns = [str(x) for x in columns["feature_columns"]]
    elif model_meta.get("feature_columns"):
        saved_columns = [str(x) for x in model_meta["feature_columns"]]

    if saved_columns is None:
        warnings.warn(
            "Checkpoint directory has no columns.json and checkpoint has no feature column metadata; "
            "column-order compatibility could not be fully checked.",
            RuntimeWarning,
        )
    elif saved_columns != list(splits.feature_columns):
        errors.append("column order differs between checkpoint and export data/config")

    if errors:
        raise ValueError("Checkpoint/export data mismatch:\n- " + "\n- ".join(errors))


def _resolve_from_experiment(experiment: str | Path) -> tuple[DictConfig, Path, Path]:
    cfg, exp_path = load_experiment_config(experiment)
    ckpt_path = Path(str(cfg.train.ckpt_dir)) / "best.pth"
    resolved = ckpt_path.parent / "resolved_config.yaml"
    if resolved.exists():
        resolved_cfg, resolved_path = load_resolved_config(resolved)
        return resolved_cfg, ckpt_path, resolved_path
    return cfg, ckpt_path, exp_path


def export_model_predictions(
    *,
    checkpoint: str | Path | None = None,
    config: str | Path | None = None,
    experiment: str | Path | None = None,
    split: str = "val",
    normal_only: bool = False,
    out: str | Path,
) -> dict:
    if experiment:
        cfg, ckpt_path, config_path = _resolve_from_experiment(experiment)
    else:
        if checkpoint is None or config is None:
            raise ValueError("Provide either --experiment or both --checkpoint and --config")
        cfg, config_path = load_resolved_config(config)
        ckpt_path = Path(checkpoint).expanduser()

    ckpt_path = ckpt_path.resolve()
    config_path = config_path.resolve()
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    state_for_checks = torch.load(ckpt_path, map_location="cpu", weights_only=False)
    _compare_checkpoint_config(state_for_checks, cfg)

    ckpt_dir = ckpt_path.parent
    normalization_stats_path = ckpt_dir / "normalization_stats.npz"
    columns_path = ckpt_dir / "columns.json"
    columns_payload = None
    normalization_override = None
    scaler_source = "recomputed_from_export_data"

    if columns_path.exists():
        columns_payload = _read_json(columns_path)
        cfg = _apply_checkpoint_columns(cfg, columns_payload)
    else:
        warnings.warn(
            f"Missing checkpoint column metadata: {columns_path}; export will use config/raw CSV ordering.",
            RuntimeWarning,
        )

    if normalization_stats_path.exists():
        normalization_override = load_normalization_stats(normalization_stats_path)
        cfg.dataset.normalization.std_floor = float(normalization_override["std_floor"])
        scaler_source = "checkpoint_dir"
    else:
        warnings.warn(
            f"Missing checkpoint normalization stats: {normalization_stats_path}; export will recompute stats.",
            RuntimeWarning,
        )

    device = get_device(cfg.train.get("device", "cpu"))
    splits = make_trajectory_splits(cfg, normalization_override=normalization_override)
    _compare_split_metadata(state=state_for_checks, splits=splits, columns=columns_payload)
    model = build_model(cfg, splits).to(device)
    state = load_model_checkpoint(model, ckpt_path, device=device)

    export_dataset = _split_dataset(splits, split)
    export_loader = _loader(export_dataset, cfg)
    inputs, preds, targets, labels = _collect(model, export_loader, device)

    keep_mask = np.ones(labels.shape[0], dtype=bool)
    if normal_only:
        keep_mask = labels < 0.5
        inputs = inputs[keep_mask]
        preds = preds[keep_mask]
        targets = targets[keep_mask]
        labels = labels[keep_mask]

    out_dir = Path(out)
    out_dir.mkdir(parents=True, exist_ok=True)
    split_name = str(split).lower()
    inputs_path = out_dir / f"{split_name}_inputs.npy"
    preds_path = out_dir / f"{split_name}_neural_preds.npy"
    targets_path = out_dir / f"{split_name}_actual_next.npy"
    labels_path = out_dir / f"{split_name}_labels.npy"
    metadata_path = out_dir / "metadata.json"

    np.save(inputs_path, inputs.astype(np.float32))
    np.save(preds_path, preds.astype(np.float32))
    np.save(targets_path, targets.astype(np.float32))
    np.save(labels_path, labels.astype(np.float32))

    sensor_names = [splits.feature_columns[i] for i in splits.sensor_idx]
    actuator_names = [splits.feature_columns[i] for i in splits.actuator_idx]
    target_names = sensor_names if bool(cfg.model.get("target_sensor_only", True)) else splits.feature_columns
    export_mse = _mse(preds, targets)

    val_mse = export_mse
    if split_name != "val" or normal_only:
        val_loader = _loader(splits.val, cfg)
        val_mse = evaluate_mse(model, val_loader, device)

    metadata = {
        "dataset_name": str(cfg.dataset.name),
        "split_name": split_name,
        "normal_only": bool(normal_only),
        "feature_column_names": splits.feature_columns,
        "sensor_column_names": sensor_names,
        "actuator_column_names": actuator_names,
        "target_column_names": target_names,
        "model_type": "neural_model",
        "architecture": str(cfg.model.get("architecture", "gru")),
        "checkpoint_path": str(ckpt_path),
        "resolved_config_path": str(config_path),
        "normalization_stats_path": str(normalization_stats_path) if normalization_stats_path.exists() else None,
        "columns_json_path": str(columns_path) if columns_path.exists() else None,
        "scaler_source": scaler_source,
        "checkpoint_epoch": int(state.get("epoch", -1)),
        "checkpoint_best_val": float(state.get("best_val", float("nan"))),
        "history_len": int(cfg.model.get("history_len", cfg.dataset.get("window_size", 1))),
        "horizon": int(cfg.model.get("horizon", cfg.dataset.get("horizon", 1))),
        "sample_stride": int(cfg.model.get("sample_stride", 1)),
        "prediction_mode": str(cfg.model.get("prediction_mode", "deterministic")),
        "target_definition": _target_definition(cfg, str(cfg.model.get("architecture", "gru")).lower()),
        "input_shape_convention": "[num_samples, num_tags, history_len]",
        "target_shape_convention": "[num_samples, horizon, num_target_columns]",
        "normalization": splits.normalization,
        "validation_mse": float(val_mse),
        "export_mse": float(export_mse),
        "num_samples_before_filter": int(keep_mask.shape[0]),
        "num_samples_exported": int(inputs.shape[0]),
        "array_shapes": {
            inputs_path.name: list(inputs.shape),
            preds_path.name: list(preds.shape),
            targets_path.name: list(targets.shape),
            labels_path.name: list(labels.shape),
        },
    }
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "inputs": str(inputs_path),
        "predictions": str(preds_path),
        "targets": str(targets_path),
        "labels": str(labels_path),
        "metadata": str(metadata_path),
        "num_samples": int(inputs.shape[0]),
        "validation_mse": float(val_mse),
    }
