from __future__ import annotations

import json
import shutil
import time
from pathlib import Path
from typing import Optional

import numpy as np
import torch
import torch.nn.functional as F
from omegaconf import DictConfig

from ics_symbolic_distill.data import TrajectorySplits, build_dataloaders, make_trajectory_splits
from ics_symbolic_distill.models import DirectTrajectoryPredictor
from ics_symbolic_distill.training.config import save_resolved_config, to_container
from ics_symbolic_distill.utils import get_device, set_seed


def build_model(cfg: DictConfig, splits: TrajectorySplits) -> DirectTrajectoryPredictor:
    return DirectTrajectoryPredictor(
        sensor_idx=splits.sensor_idx,
        actuator_idx=splits.actuator_idx,
        num_tags=splits.num_tags,
        history_len=int(cfg.model.get("history_len", cfg.dataset.get("window_size", 1))),
        horizon=int(cfg.model.get("horizon", cfg.dataset.get("horizon", 1))),
        hidden_dim=int(cfg.model.get("hidden_dim", 128)),
        num_layers=int(cfg.model.get("num_layers", 1)),
        dropout=float(cfg.model.get("dropout", 0.0)),
        architecture=str(cfg.model.get("architecture", "gru")),
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


def model_metadata(cfg: DictConfig, splits: TrajectorySplits) -> dict:
    return {
        "dataset_name": str(cfg.dataset.name),
        "architecture": str(cfg.model.get("architecture", "gru")),
        "num_tags": int(splits.num_tags),
        "feature_columns": list(splits.feature_columns),
        "history_len": int(cfg.model.get("history_len", cfg.dataset.get("window_size", 1))),
        "horizon": int(cfg.model.get("horizon", cfg.dataset.get("horizon", 1))),
        "sample_stride": int(cfg.model.get("sample_stride", 1)),
        "target_sensor_only": bool(cfg.model.get("target_sensor_only", True)),
    }


def _target_columns(cfg: DictConfig, splits: TrajectorySplits) -> list[str]:
    if bool(cfg.model.get("target_sensor_only", True)):
        return [splits.feature_columns[i] for i in splits.sensor_idx]
    return list(splits.feature_columns)


def _array_or_empty(value, dtype=np.float32) -> np.ndarray:
    if value is None:
        return np.asarray([], dtype=dtype)
    return np.asarray(value, dtype=dtype)


def save_normalization_stats(splits: TrajectorySplits, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    normalization = splits.normalization
    np.savez(
        out,
        normalization_mode=np.asarray(str(normalization.get("mode", "none"))),
        std_floor=np.asarray(float(normalization.get("std_floor", 0.0)), dtype=np.float32),
        fit_split=np.asarray(str(normalization.get("fit_split", "train"))),
        feature_columns=np.asarray(splits.feature_columns, dtype=str),
        sensor_idx=np.asarray(splits.sensor_idx, dtype=np.int64),
        actuator_idx=np.asarray(splits.actuator_idx, dtype=np.int64),
        mean=_array_or_empty(normalization.get("mean")),
        std=_array_or_empty(normalization.get("std")),
        median=_array_or_empty(normalization.get("median")),
        iqr=_array_or_empty(normalization.get("iqr")),
        data_min=_array_or_empty(normalization.get("data_min")),
        data_max=_array_or_empty(normalization.get("data_max")),
        minmax_variable_mask=_array_or_empty(
            normalization.get("minmax_variable_mask"),
            dtype=bool,
        ),
    )
    return out


def load_normalization_stats(path: str | Path) -> dict:
    stats_path = Path(path)
    if not stats_path.exists():
        raise FileNotFoundError(f"Normalization stats not found: {stats_path}")
    payload = np.load(stats_path, allow_pickle=False)

    def _maybe_array(name: str):
        arr = payload[name]
        return None if arr.size == 0 else arr

    return {
        "normalization_mode": str(payload["normalization_mode"].item()),
        "std_floor": float(payload["std_floor"].item()),
        "fit_split": str(payload["fit_split"].item()),
        "feature_columns": [str(x) for x in payload["feature_columns"].tolist()],
        "sensor_idx": payload["sensor_idx"].astype(int).tolist(),
        "actuator_idx": payload["actuator_idx"].astype(int).tolist(),
        "mean": _maybe_array("mean"),
        "std": _maybe_array("std"),
        "median": _maybe_array("median"),
        "iqr": _maybe_array("iqr"),
        "data_min": _maybe_array("data_min"),
        "data_max": _maybe_array("data_max"),
        "minmax_variable_mask": _maybe_array("minmax_variable_mask"),
    }


def save_columns_json(cfg: DictConfig, splits: TrajectorySplits, path: str | Path) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    sensor_columns = [splits.feature_columns[i] for i in splits.sensor_idx]
    actuator_columns = [splits.feature_columns[i] for i in splits.actuator_idx]
    payload = {
        "feature_columns": list(splits.feature_columns),
        "sensor_columns": sensor_columns,
        "actuator_columns": actuator_columns,
        "target_columns": _target_columns(cfg, splits),
        "sensor_idx": list(splits.sensor_idx),
        "actuator_idx": list(splits.actuator_idx),
        "label_column": cfg.dataset.get("label_column"),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def save_manifest_json(
    cfg: DictConfig,
    splits: TrajectorySplits,
    path: str | Path,
    *,
    checkpoint_epoch: Optional[int],
    validation_mse: Optional[float],
    best_checkpoint: str | Path,
    resolved_config: str | Path,
    normalization_stats: str | Path,
    columns_json: str | Path,
) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    window = int(cfg.dataset.window_size)
    horizon = int(cfg.dataset.horizon)
    payload = {
        "dataset_name": str(cfg.dataset.name),
        "model_architecture": str(cfg.model.get("architecture", "gru")),
        "num_tags": int(splits.num_tags),
        "history_len": int(cfg.model.get("history_len", cfg.dataset.get("window_size", 1))),
        "horizon": int(cfg.model.get("horizon", cfg.dataset.get("horizon", 1))),
        "sample_stride": int(cfg.model.get("sample_stride", 1)),
        "sampling_stride": int(cfg.dataset.sampling_stride),
        "validation_split_rule": {
            "source": "tail_of_downsampled_train",
            "val_ratio": float(cfg.dataset.val_ratio),
            "minimum_rows": int(window + horizon + 1),
            "maximum_rows_rule": "no more than half of downsampled train rows",
        },
        "train_rows_after_downsampling": int(splits.train_rows),
        "validation_rows_after_downsampling": int(splits.val_rows),
        "test_rows_after_downsampling": int(splits.test_rows),
        "checkpoint_epoch": None if checkpoint_epoch is None else int(checkpoint_epoch),
        "validation_mse": None if validation_mse is None else float(validation_mse),
        "best_checkpoint": str(best_checkpoint),
        "resolved_config": str(resolved_config),
        "normalization_stats": str(normalization_stats),
        "columns_json": str(columns_json),
    }
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return out


def build_optimizer(cfg: DictConfig, model: torch.nn.Module) -> torch.optim.Optimizer:
    name = str(cfg.train.get("optimizer", "adamw")).lower()
    lr = float(cfg.train.lr)
    weight_decay = float(cfg.train.weight_decay)
    if name == "adamw":
        return torch.optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "adam":
        return torch.optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    if name == "sgd":
        return torch.optim.SGD(
            model.parameters(),
            lr=lr,
            weight_decay=weight_decay,
            momentum=float(cfg.train.get("momentum", 0.9)),
        )
    raise ValueError(f"Unsupported optimizer: {name}")


def _move_batch(batch, device: torch.device):
    x_window, y_future, labels = batch
    return x_window.to(device), y_future.to(device), labels.to(device)


def _batch_loss_and_outputs(
    model: DirectTrajectoryPredictor,
    x_window: torch.Tensor,
    y_future: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    if model.architecture in {"autoencoder_sensors", "autoencoder_full"}:
        x_curr = x_window[:, :, -1]
        pred = model.reconstruct_current(x_window)
        if model.architecture == "autoencoder_full":
            pred_sensor = pred.index_select(dim=1, index=model.sensor_idx_tensor)
        else:
            pred_sensor = pred
        target_sensor = x_curr.index_select(dim=1, index=model.sensor_idx_tensor)
        loss = F.mse_loss(pred_sensor, target_sensor)
        return loss, pred_sensor.unsqueeze(1), target_sensor.unsqueeze(1)

    mu, log_var = model.predict_distribution(x_window)
    loss = model.compute_prediction_loss(mu, y_future, log_var=log_var)
    return loss, mu, y_future


def train_epoch(
    model: DirectTrajectoryPredictor,
    loader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    *,
    grad_clip: float,
    max_batches: Optional[int] = None,
) -> float:
    model.train()
    total = 0.0
    batches = 0
    for batch in loader:
        if max_batches is not None and batches >= max_batches:
            break
        x_window, y_future, _ = _move_batch(batch, device)
        loss, _, _ = _batch_loss_and_outputs(model, x_window, y_future)
        optimizer.zero_grad()
        loss.backward()
        if grad_clip > 0.0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()
        total += float(loss.item())
        batches += 1
    return total / max(batches, 1)


@torch.no_grad()
def evaluate_mse(
    model: DirectTrajectoryPredictor,
    loader,
    device: torch.device,
    *,
    max_batches: Optional[int] = None,
) -> float:
    model.eval()
    total = 0.0
    batches = 0
    for batch in loader:
        if max_batches is not None and batches >= max_batches:
            break
        x_window, y_future, _ = _move_batch(batch, device)
        loss, _, _ = _batch_loss_and_outputs(model, x_window, y_future)
        total += float(loss.item())
        batches += 1
    return total / max(batches, 1)


def save_checkpoint(
    *,
    model: DirectTrajectoryPredictor,
    optimizer: Optional[torch.optim.Optimizer],
    epoch: int,
    best_val: float,
    cfg: DictConfig,
    path: Path,
    metadata: Optional[dict] = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "model_state": model.state_dict(),
        "optimizer_state": optimizer.state_dict() if optimizer is not None else None,
        "epoch": int(epoch),
        "best_val": float(best_val),
        "config": to_container(cfg),
        "model_metadata": metadata or {},
    }
    torch.save(payload, path)
    return path


def load_model_checkpoint(
    model: DirectTrajectoryPredictor,
    checkpoint_path: str | Path,
    *,
    device: torch.device,
) -> dict:
    ckpt_path = Path(checkpoint_path)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")
    state = torch.load(ckpt_path, map_location=device, weights_only=False)
    model.load_state_dict(state["model_state"], strict=True)
    return state


def train_model(cfg: DictConfig) -> dict:
    set_seed(cfg.train.get("seed"))
    device = get_device(cfg.train.get("device", "cpu"))
    ckpt_dir = Path(str(cfg.train.ckpt_dir))
    ckpt_dir.mkdir(parents=True, exist_ok=True)

    splits = make_trajectory_splits(cfg)
    resolved_config_path = save_resolved_config(cfg, ckpt_dir / "resolved_config.yaml")
    normalization_stats_path = save_normalization_stats(splits, ckpt_dir / "normalization_stats.npz")
    columns_path = save_columns_json(cfg, splits, ckpt_dir / "columns.json")
    manifest_path = save_manifest_json(
        cfg,
        splits,
        ckpt_dir / "manifest.json",
        checkpoint_epoch=None,
        validation_mse=None,
        best_checkpoint=ckpt_dir / "best.pth",
        resolved_config=resolved_config_path,
        normalization_stats=normalization_stats_path,
        columns_json=columns_path,
    )
    train_loader, val_loader, _ = build_dataloaders(cfg, splits)
    model = build_model(cfg, splits).to(device)
    optimizer = build_optimizer(cfg, model)
    checkpoint_metadata = model_metadata(cfg, splits)

    best_val = float("inf")
    best_epoch = 0
    best_epoch_path: Optional[Path] = None
    grad_clip = float(cfg.train.get("grad_clip", 0.0) or 0.0)
    max_train_batches = cfg.train.get("max_train_batches")
    max_val_batches = cfg.train.get("max_val_batches")
    max_train_batches = None if max_train_batches is None else int(max_train_batches)
    max_val_batches = None if max_val_batches is None else int(max_val_batches)
    log_interval = int(cfg.train.get("log_interval", 10))
    epochs = int(cfg.train.epochs)

    for epoch in range(1, epochs + 1):
        start = time.time()
        train_loss = train_epoch(
            model,
            train_loader,
            optimizer,
            device,
            grad_clip=grad_clip,
            max_batches=max_train_batches,
        )
        val_loss = evaluate_mse(model, val_loader, device, max_batches=max_val_batches)
        elapsed = time.time() - start

        if val_loss < best_val:
            best_val = val_loss
            best_epoch = epoch
            best_epoch_path = save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val=best_val,
                cfg=cfg,
                path=ckpt_dir / f"epoch_{epoch:03d}.pth",
                metadata=checkpoint_metadata,
            )
            shutil.copy2(best_epoch_path, ckpt_dir / "best.pth")

        ckpt_every = int(cfg.train.get("ckpt_every", 0) or 0)
        if ckpt_every > 0 and epoch % ckpt_every == 0:
            save_checkpoint(
                model=model,
                optimizer=optimizer,
                epoch=epoch,
                best_val=best_val,
                cfg=cfg,
                path=ckpt_dir / f"epoch_{epoch:03d}.pth",
                metadata=checkpoint_metadata,
            )

        if epoch == 1 or epoch % log_interval == 0 or epoch == epochs:
            print(
                f"epoch={epoch:03d} train_mse={train_loss:.6f} "
                f"val_mse={val_loss:.6f} elapsed_sec={elapsed:.1f}"
            )

    if best_epoch_path is None:
        save_checkpoint(
            model=model,
            optimizer=optimizer,
            epoch=0,
            best_val=best_val,
            cfg=cfg,
            path=ckpt_dir / "best.pth",
            metadata=checkpoint_metadata,
        )

    save_manifest_json(
        cfg,
        splits,
        manifest_path,
        checkpoint_epoch=best_epoch,
        validation_mse=best_val,
        best_checkpoint=ckpt_dir / "best.pth",
        resolved_config=resolved_config_path,
        normalization_stats=normalization_stats_path,
        columns_json=columns_path,
    )

    return {
        "checkpoint_dir": str(ckpt_dir),
        "best_checkpoint": str(ckpt_dir / "best.pth"),
        "best_epoch": best_epoch,
        "best_val_mse": best_val,
        "resolved_config": str(resolved_config_path),
        "normalization_stats": str(normalization_stats_path),
        "columns": str(columns_path),
        "manifest": str(manifest_path),
    }
