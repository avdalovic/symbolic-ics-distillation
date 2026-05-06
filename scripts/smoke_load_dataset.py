#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from torch.utils.data import DataLoader

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ics_symbolic_distill.data import load_dataset_arrays, make_trajectory_splits
from ics_symbolic_distill.training import load_experiment_config


def _count_csv_rows(path: Path) -> int:
    with path.open("rb") as handle:
        rows = sum(1 for _ in handle)
    return max(rows - 1, 0)


def _resolve_repo_path(path: str | Path) -> Path:
    p = Path(path).expanduser()
    if p.is_absolute():
        return p
    return REPO_ROOT / p


def main() -> None:
    parser = argparse.ArgumentParser(description="Load a configured dataset and print shape metadata.")
    parser.add_argument("--config", required=True, help="Experiment YAML path.")
    parser.add_argument("--split", default="val", choices=["train", "val", "test"])
    args = parser.parse_args()

    cfg, cfg_path = load_experiment_config(args.config)
    train_csv = _resolve_repo_path(cfg.dataset.train_csv)
    test_csv = _resolve_repo_path(cfg.dataset.test_csv)

    loaded = load_dataset_arrays(cfg)
    splits = make_trajectory_splits(cfg)
    dataset = getattr(splits, args.split)
    batch = next(iter(DataLoader(dataset, batch_size=min(int(cfg.dataset.batch_size), 8), shuffle=False)))
    x_window, y_future, labels = batch

    payload = {
        "config": str(cfg_path),
        "dataset": str(cfg.dataset.name),
        "train_csv": str(train_csv),
        "test_csv": str(test_csv),
        "raw_train_rows": _count_csv_rows(train_csv),
        "raw_test_rows": _count_csv_rows(test_csv),
        "downsampled_train_rows_total": int(loaded.train.shape[0]),
        "downsampled_test_rows_total": int(loaded.test.shape[0]),
        "train_rows_after_validation_split": int(splits.train_rows),
        "validation_rows": int(splits.val_rows),
        "test_rows": int(splits.test_rows),
        "feature_columns": int(len(splits.feature_columns)),
        "sensor_columns": int(len(splits.sensor_idx)),
        "actuator_columns": int(len(splits.actuator_idx)),
        "first_5_feature_columns": list(splits.feature_columns[:5]),
        "history_len": int(cfg.model.get("history_len", cfg.dataset.get("window_size", 1))),
        "horizon": int(cfg.model.get("horizon", cfg.dataset.get("horizon", 1))),
        "sample_stride": int(cfg.model.get("sample_stride", 1)),
        "sampling_stride": int(cfg.dataset.sampling_stride),
        "batch_split": args.split,
        "input_shape_for_one_batch": list(x_window.shape),
        "target_shape_for_one_batch": list(y_future.shape),
        "label_shape_for_one_batch": list(labels.shape),
    }
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
