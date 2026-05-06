from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from omegaconf import OmegaConf
from torch.utils.data import DataLoader

import ics_symbolic_distill
from ics_symbolic_distill.data import make_trajectory_splits
from ics_symbolic_distill.inference import export_model_predictions
from ics_symbolic_distill.training.config import load_experiment_config
from ics_symbolic_distill.training.model import (
    build_model,
    load_model_checkpoint,
    train_model,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


def _write_swat_like_csv(path: Path, rows: int) -> None:
    t = np.arange(rows, dtype=np.float32)
    df = pd.DataFrame(
        {
            "Timestamp": [f"2020-01-01 00:{i:02d}:00" for i in range(rows)],
            "FIT101": np.sin(t / 5.0) + 0.01 * t,
            "LIT101": np.cos(t / 7.0) + 0.02 * t,
            "MV101": (t.astype(int) % 2).astype(np.float32),
            "P101": ((t.astype(int) // 5) % 2).astype(np.float32),
            "P102": ((t.astype(int) // 7) % 2).astype(np.float32),
            "Normal/Attack": ["Normal"] * rows,
        }
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def _tiny_swat_gru_cfg(tmp_path: Path):
    train_csv = tmp_path / "data" / "swat_train.csv"
    test_csv = tmp_path / "data" / "swat_test.csv"
    _write_swat_like_csv(train_csv, rows=80)
    _write_swat_like_csv(test_csv, rows=36)

    cfg, _ = load_experiment_config(REPO_ROOT / "configs" / "experiment" / "swat_gru.yaml")
    return OmegaConf.merge(
        cfg,
        {
            "dataset": {
                "train_csv": str(train_csv),
                "test_csv": str(test_csv),
                "sampling_stride": 1,
                "window_size": 6,
                "horizon": 1,
                "val_ratio": 0.25,
                "batch_size": 8,
                "num_workers": 0,
                "shuffle_train": False,
            },
            "model": {
                "history_len": 6,
                "horizon": 1,
                "sample_stride": 1,
                "hidden_dim": 8,
                "num_layers": 1,
                "dropout": 0.0,
            },
            "train": {
                "epochs": 1,
                "device": "cpu",
                "ckpt_dir": str(tmp_path / "checkpoints" / "swat" / "gru_h1"),
                "log_interval": 1,
                "seed": 7,
            },
        },
    )


def test_package_imports_and_configs_load() -> None:
    assert ics_symbolic_distill.__version__ == "0.1.0"

    cfg, path = load_experiment_config(REPO_ROOT / "configs" / "experiment" / "swat_gru.yaml")
    assert path.name == "swat_gru.yaml"
    assert cfg.dataset.name == "swat"
    assert cfg.model.architecture == "gru"

    mlp_cfg, _ = load_experiment_config(
        REPO_ROOT / "configs" / "experiment" / "swat_mlp_current.yaml"
    )
    assert mlp_cfg.model.architecture == "mlp"
    assert int(mlp_cfg.model.history_len) == 1

    for name in ["pysr_lit101_all.yaml", "pysr_lit101_topk.yaml", "pysr_lit101_physical.yaml"]:
        loaded = OmegaConf.load(REPO_ROOT / "configs" / "distill" / name)
        assert loaded.distill.target == "LIT101"


def test_swat_loader_initializes_and_gru_forward_pass(tmp_path: Path) -> None:
    cfg = _tiny_swat_gru_cfg(tmp_path)
    splits = make_trajectory_splits(cfg)
    assert splits.feature_columns == ["FIT101", "LIT101", "MV101", "P101", "P102"]
    assert [splits.feature_columns[i] for i in splits.sensor_idx] == ["FIT101", "LIT101"]
    assert [splits.feature_columns[i] for i in splits.actuator_idx] == ["MV101", "P101", "P102"]

    loader = DataLoader(splits.val, batch_size=4, shuffle=False)
    x_window, y_future, labels = next(iter(loader))
    assert x_window.shape == (4, 5, 6)
    assert y_future.shape == (4, 1, 2)
    assert labels.shape == (4,)

    model = build_model(cfg, splits)
    with torch.no_grad():
        preds = model(x_window)
    assert preds.shape == y_future.shape
    assert torch.isfinite(preds).all()


def test_training_checkpoint_roundtrip_and_export_smoke(tmp_path: Path) -> None:
    cfg = _tiny_swat_gru_cfg(tmp_path)
    result = train_model(cfg)

    checkpoint = Path(result["best_checkpoint"])
    resolved_config = Path(result["resolved_config"])
    normalization_stats = Path(result["normalization_stats"])
    columns_json = Path(result["columns"])
    manifest_json = Path(result["manifest"])
    assert checkpoint.exists()
    assert resolved_config.exists()
    assert normalization_stats.exists()
    assert columns_json.exists()
    assert manifest_json.exists()

    npz = np.load(normalization_stats, allow_pickle=False)
    assert npz["normalization_mode"].item() == "zscore"
    assert npz["mean"].shape == (5,)
    assert npz["std"].shape == (5,)

    columns = json.loads(columns_json.read_text(encoding="utf-8"))
    assert columns["feature_columns"] == ["FIT101", "LIT101", "MV101", "P101", "P102"]
    assert columns["sensor_columns"] == ["FIT101", "LIT101"]
    assert columns["actuator_columns"] == ["MV101", "P101", "P102"]
    assert columns["target_columns"] == ["FIT101", "LIT101"]
    assert columns["label_column"] == "Normal/Attack"

    manifest = json.loads(manifest_json.read_text(encoding="utf-8"))
    assert manifest["dataset_name"] == "swat"
    assert manifest["model_architecture"] == "gru"
    assert manifest["checkpoint_epoch"] == 1
    assert manifest["validation_mse"] is not None

    splits = make_trajectory_splits(cfg)
    model = build_model(cfg, splits)
    state = load_model_checkpoint(model, checkpoint, device=torch.device("cpu"))
    assert int(state["epoch"]) == 1
    assert "model_state" in state

    out_dir = tmp_path / "exports" / "swat" / "gru_h1" / "val"
    export_result = export_model_predictions(
        checkpoint=checkpoint,
        config=resolved_config,
        split="val",
        normal_only=True,
        out=out_dir,
    )

    expected_names = ["val_inputs.npy", "val_neural_preds.npy", "val_actual_next.npy", "metadata.json"]
    for name in expected_names:
        assert (out_dir / name).exists()

    inputs = np.load(out_dir / "val_inputs.npy")
    preds = np.load(out_dir / "val_neural_preds.npy")
    targets = np.load(out_dir / "val_actual_next.npy")
    metadata = json.loads((out_dir / "metadata.json").read_text(encoding="utf-8"))

    assert inputs.shape[0] == preds.shape[0] == targets.shape[0] == export_result["num_samples"]
    assert inputs.shape[1:] == (5, 6)
    assert preds.shape[1:] == (1, 2)
    assert targets.shape[1:] == (1, 2)
    assert metadata["dataset_name"] == "swat"
    assert metadata["split_name"] == "val"
    assert metadata["normal_only"] is True
    assert metadata["architecture"] == "gru"
    assert metadata["resolved_config_path"] == str(resolved_config.resolve())
    assert metadata["normalization_stats_path"] == str(normalization_stats.resolve())
    assert metadata["columns_json_path"] == str(columns_json.resolve())
    assert metadata["scaler_source"] == "checkpoint_dir"
    assert metadata["input_shape_convention"] == "[num_samples, num_tags, history_len]"
    assert metadata["target_shape_convention"] == "[num_samples, horizon, num_target_columns]"
    assert metadata["array_shapes"]["val_inputs.npy"] == list(inputs.shape)
