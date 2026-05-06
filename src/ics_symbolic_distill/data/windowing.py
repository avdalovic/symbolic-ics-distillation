from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd
import torch
from omegaconf import DictConfig
from torch.utils.data import DataLoader, Dataset

from .ics_metadata import build_sensor_actuator_indices, normalize_attack_labels
from .wadi_constants import apply_drop_columns, resolve_wadi_drop_columns


@dataclass(frozen=True)
class LoadedArrays:
    train: np.ndarray
    test: np.ndarray
    train_labels: Optional[np.ndarray]
    test_labels: Optional[np.ndarray]
    feature_columns: list[str]


@dataclass(frozen=True)
class TrajectorySplits:
    train: Dataset
    val: Dataset
    test: Dataset
    num_tags: int
    feature_columns: list[str]
    sensor_idx: list[int]
    actuator_idx: list[int]
    train_rows: int
    val_rows: int
    test_rows: int
    normalization: dict


def _prepare_dataframe(df: pd.DataFrame, cfg: DictConfig) -> tuple[pd.DataFrame, Optional[pd.Series]]:
    time_col = cfg.dataset.get("time_column")
    if time_col and time_col in df.columns:
        df = df.drop(columns=[time_col])

    label_col = cfg.dataset.get("label_column")
    labels = None
    if label_col and label_col in df.columns:
        labels = df[label_col].copy()
        df = df.drop(columns=[label_col])
    return df, labels


def _select_swat_columns(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    cols = cfg.dataset.get("tag_columns")
    if cols:
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in SWaT DataFrame: {sorted(missing)}")
        return df[list(cols)]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    label_col = cfg.dataset.get("label_column")
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    if not numeric_cols:
        raise ValueError("SWaT CSV must contain numeric feature columns")
    return df[numeric_cols]


def _select_wadi_columns(df: pd.DataFrame, cfg: DictConfig) -> pd.DataFrame:
    drop_cols = resolve_wadi_drop_columns(cfg)
    cols = cfg.dataset.get("tag_columns")
    if cols:
        missing = set(cols) - set(df.columns)
        if missing:
            raise ValueError(f"Missing columns in WADI DataFrame: {sorted(missing)}")
        selected_cols = apply_drop_columns([str(c) for c in cols], drop_cols)
        if not selected_cols:
            raise ValueError("No WADI feature columns remain after drop-column filtering")
        return df[selected_cols]

    numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
    label_col = cfg.dataset.get("label_column")
    if label_col in numeric_cols:
        numeric_cols.remove(label_col)
    numeric_cols = apply_drop_columns(numeric_cols, drop_cols)
    if not numeric_cols:
        raise ValueError("WADI CSV must contain numeric feature columns")
    return df[numeric_cols]


def load_swat_arrays(cfg: DictConfig) -> LoadedArrays:
    train_path = Path(str(cfg.dataset.train_csv))
    test_path = Path(str(cfg.dataset.test_csv))
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"SWaT CSVs not found: train={train_path} test={test_path}")

    stride = int(cfg.dataset.sampling_stride)
    train_df = pd.read_csv(train_path).iloc[::stride].reset_index(drop=True)
    test_df = pd.read_csv(test_path).iloc[::stride].reset_index(drop=True)

    train_df, train_labels = _prepare_dataframe(train_df, cfg)
    test_df, test_labels = _prepare_dataframe(test_df, cfg)

    train_sel = _select_swat_columns(train_df, cfg)
    feature_columns = [str(c) for c in train_sel.columns]
    test_sel = test_df[feature_columns]

    return LoadedArrays(
        train=train_sel.to_numpy(dtype=np.float32),
        test=test_sel.to_numpy(dtype=np.float32),
        train_labels=normalize_attack_labels(train_labels),
        test_labels=normalize_attack_labels(test_labels),
        feature_columns=feature_columns,
    )


def load_wadi_arrays(cfg: DictConfig) -> LoadedArrays:
    train_path = Path(str(cfg.dataset.train_csv))
    test_path = Path(str(cfg.dataset.test_csv))
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"WADI CSVs not found: train={train_path} test={test_path}")

    stride = int(cfg.dataset.sampling_stride)
    train_df = pd.read_csv(train_path).iloc[::stride].reset_index(drop=True)
    test_df = pd.read_csv(test_path).iloc[::stride].reset_index(drop=True)

    train_df, train_labels = _prepare_dataframe(train_df, cfg)
    test_df, test_labels = _prepare_dataframe(test_df, cfg)

    train_sel = _select_wadi_columns(train_df, cfg)
    test_sel = _select_wadi_columns(test_df, cfg)

    shared_cols = [c for c in train_sel.columns if c in test_sel.columns]
    if not shared_cols:
        raise ValueError("No shared numeric feature columns between WADI train/test CSVs")
    train_sel = train_sel[shared_cols]
    test_sel = test_sel[shared_cols]

    keep_mask = train_sel.notna().any(axis=0)
    if not bool(np.all(keep_mask.to_numpy())):
        keep_cols = train_sel.columns[keep_mask].tolist()
        train_sel = train_sel[keep_cols]
        test_sel = test_sel[keep_cols]

    if train_sel.isna().any().any() or test_sel.isna().any().any():
        med = train_sel.median(axis=0, numeric_only=True)
        train_sel = train_sel.fillna(med).fillna(0.0)
        test_sel = test_sel.fillna(med).fillna(0.0)

    return LoadedArrays(
        train=train_sel.to_numpy(dtype=np.float32),
        test=test_sel.to_numpy(dtype=np.float32),
        train_labels=normalize_attack_labels(train_labels),
        test_labels=normalize_attack_labels(test_labels),
        feature_columns=[str(c) for c in train_sel.columns],
    )


def load_dataset_arrays(cfg: DictConfig) -> LoadedArrays:
    name = str(cfg.dataset.name).lower()
    if name == "swat":
        return load_swat_arrays(cfg)
    if name == "wadi":
        return load_wadi_arrays(cfg)
    raise ValueError(f"Unsupported dataset: {cfg.dataset.name}")


class OneStepDataset(Dataset):
    """Dataset for one-step dynamics prediction: x_t -> x_{t+1}."""

    def __init__(
        self,
        data: np.ndarray,
        norm_mode: str = "zscore",
        std_floor: float = 1e-2,
        mean: Optional[np.ndarray] = None,
        std: Optional[np.ndarray] = None,
        median: Optional[np.ndarray] = None,
        iqr: Optional[np.ndarray] = None,
        data_min: Optional[np.ndarray] = None,
        data_max: Optional[np.ndarray] = None,
        minmax_variable_mask: Optional[np.ndarray] = None,
        sensor_idx: Optional[list[int]] = None,
        actuator_idx: Optional[list[int]] = None,
        labels: Optional[np.ndarray] = None,
    ) -> None:
        super().__init__()
        if data.ndim != 2:
            raise ValueError("data must be 2D [time, num_tags]")

        self.norm_mode = (norm_mode or "none").lower()
        if self.norm_mode == "standard":
            self.norm_mode = "zscore"
        self.std_floor = max(float(std_floor), 1e-12)
        self._raw = np.asarray(data, dtype=np.float32)
        self.num_tags = int(self._raw.shape[1])
        self.sensor_idx, self.actuator_idx = build_sensor_actuator_indices(
            "",
            [f"x{i}" for i in range(self.num_tags)],
            sensor_idx=sensor_idx,
            actuator_idx=actuator_idx,
        )
        self._sensor_mask = np.zeros(self.num_tags, dtype=bool)
        self._actuator_mask = np.zeros(self.num_tags, dtype=bool)
        self._sensor_mask[self.sensor_idx] = True
        self._actuator_mask[self.actuator_idx] = True

        self._mean = np.zeros(self.num_tags, dtype=np.float32)
        self._std = np.ones(self.num_tags, dtype=np.float32)
        self._median = np.zeros(self.num_tags, dtype=np.float32)
        self._iqr = np.ones(self.num_tags, dtype=np.float32)
        self._data_min = None
        self._data_max = None
        self._minmax_variable_mask = None

        if self.norm_mode == "zscore":
            _mean = mean if mean is not None else self._raw.mean(axis=0)
            _std = std if std is not None else self._raw.std(axis=0)
            self._mean = np.asarray(_mean, dtype=np.float32)
            self._std = np.maximum(np.asarray(_std, dtype=np.float32), self.std_floor)
            self._data = ((self._raw - self._mean) / self._std).astype(np.float32)
        elif self.norm_mode == "minmax":
            _min = data_min if data_min is not None else self._raw.min(axis=0)
            _max = data_max if data_max is not None else self._raw.max(axis=0)
            _min = np.asarray(_min, dtype=np.float32)
            _max = np.asarray(_max, dtype=np.float32)
            data_range = _max - _min
            if minmax_variable_mask is not None:
                self._minmax_variable_mask = np.asarray(minmax_variable_mask, dtype=bool)
            else:
                self._minmax_variable_mask = np.asarray(data_range >= 1e-6, dtype=bool)
            safe_range = np.where(self._minmax_variable_mask, data_range, 1.0).astype(np.float32)
            self._data_min = _min
            self._data_max = (_min + safe_range).astype(np.float32)
            scaled = (self._raw - self._data_min) / (self._data_max - self._data_min)
            self._data = np.where(self._minmax_variable_mask.reshape(1, -1), scaled, self._raw).astype(
                np.float32
            )
            self._mean = self._data_min.copy()
            self._std = (self._data_max - self._data_min).astype(np.float32)
        elif self.norm_mode == "robust":
            _median = median if median is not None else np.median(self._raw, axis=0)
            if iqr is not None:
                _iqr = iqr
            else:
                q75 = np.percentile(self._raw, 75, axis=0)
                q25 = np.percentile(self._raw, 25, axis=0)
                _iqr = q75 - q25
            self._median = np.asarray(_median, dtype=np.float32)
            _iqr = np.asarray(_iqr, dtype=np.float32)
            sensor_iqr = _iqr[self._sensor_mask]
            sensor_scale = np.maximum(sensor_iqr, self.std_floor).astype(np.float32)
            sensor_scale = np.where(sensor_iqr < self.std_floor, 1.0, sensor_scale).astype(np.float32)
            self._iqr = np.ones(self.num_tags, dtype=np.float32)
            self._iqr[self._sensor_mask] = sensor_scale
            centered = self._raw - self._median
            robust_data = centered.astype(np.float32)
            if self.sensor_idx:
                robust_data[:, self._sensor_mask] = centered[:, self._sensor_mask] / self._iqr[
                    self._sensor_mask
                ]
            if self.actuator_idx:
                robust_data[:, self._actuator_mask] = self._raw[:, self._actuator_mask]
            self._data = robust_data.astype(np.float32)
            self._mean = self._median.copy()
            self._std = self._iqr.copy()
        else:
            self._data = self._raw.astype(np.float32)

        self.mean = self._mean
        self.std = self._std
        self.median = self._median
        self.iqr = self._iqr
        self._labels = labels.astype(np.float32) if labels is not None else None
        self._len = int(len(self._data) - 1)
        if self._len <= 0:
            raise ValueError("time series too short for one-step dataset")

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self._len:
            raise IndexError(idx)
        x_t = torch.from_numpy(self._data[idx]).float()
        x_tp1 = torch.from_numpy(self._data[idx + 1]).float()
        label = torch.tensor(0.0, dtype=torch.float32)
        if self._labels is not None:
            label = torch.tensor(float(self._labels[idx + 1]), dtype=torch.float32)
        return x_t, x_tp1, label

    def denormalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if self.norm_mode == "zscore":
            mean = torch.as_tensor(self._mean, dtype=x.dtype, device=x.device)
            std = torch.as_tensor(self._std, dtype=x.dtype, device=x.device)
            return x * std + mean
        if self.norm_mode == "minmax":
            data_min = torch.as_tensor(self._data_min, dtype=x.dtype, device=x.device)
            data_max = torch.as_tensor(self._data_max, dtype=x.dtype, device=x.device)
            restored = x * (data_max - data_min) + data_min
            if self._minmax_variable_mask is None:
                return restored
            mask = torch.as_tensor(self._minmax_variable_mask, dtype=torch.bool, device=x.device)
            mask = mask.view((1,) * (x.ndim - 1) + (mask.numel(),))
            return torch.where(mask, restored, x)
        if self.norm_mode == "robust":
            median = torch.as_tensor(self._median, dtype=x.dtype, device=x.device)
            iqr = torch.as_tensor(self._iqr, dtype=x.dtype, device=x.device)
            restored = x * iqr + median
            sensor_mask = torch.as_tensor(self._sensor_mask, dtype=torch.bool, device=x.device)
            sensor_mask = sensor_mask.view((1,) * (x.ndim - 1) + (sensor_mask.numel(),))
            return torch.where(sensor_mask, restored, x)
        return x


class OneStepHistoryFutureDataset(Dataset):
    """Sliding history-to-future trajectory view over normalized one-step data."""

    def __init__(
        self,
        base: OneStepDataset,
        history_len: int,
        future_horizon: int,
        *,
        sample_stride: int = 1,
        target_sensor_only: bool = True,
        label_mode: str = "horizon_max",
    ) -> None:
        super().__init__()
        self.base = base
        self.history_len = int(history_len)
        self.future_horizon = int(future_horizon)
        self.sample_stride = int(sample_stride)
        self.target_sensor_only = bool(target_sensor_only)
        self.label_mode = str(label_mode or "horizon_max").lower()

        if self.history_len <= 0:
            raise ValueError("history_len must be positive")
        if self.future_horizon <= 0:
            raise ValueError("future_horizon must be positive")
        if self.sample_stride <= 0:
            raise ValueError("sample_stride must be positive")
        if self.label_mode not in {"horizon_max", "t_plus_1"}:
            raise ValueError("label_mode must be one of: horizon_max, t_plus_1")

        self._data = base._data
        self._labels = base._labels
        self.num_tags = int(base.num_tags)
        self.sensor_idx = list(base.sensor_idx)
        self.actuator_idx = list(base.actuator_idx)
        self.norm_mode = str(base.norm_mode)
        self.mean = base.mean
        self.std = base.std
        self.median = base.median
        self.iqr = base.iqr

        self.target_indices = self.sensor_idx if self.target_sensor_only else list(range(self.num_tags))
        self.n_target_tags = int(len(self.target_indices))

        max_offset = int((self.history_len - 1 + self.future_horizon) * self.sample_stride)
        self._len = int(self._data.shape[0] - max_offset)
        if self._len <= 0:
            raise ValueError(
                "time series too short for history/future trajectory dataset "
                f"(len={self._data.shape[0]}, history_len={self.history_len}, "
                f"future_horizon={self.future_horizon}, sample_stride={self.sample_stride})"
            )

    def __len__(self) -> int:
        return self._len

    def __getitem__(self, idx: int):
        if idx < 0 or idx >= self._len:
            raise IndexError(idx)

        anchor = idx + (self.history_len - 1) * self.sample_stride
        hist_idx = idx + np.arange(self.history_len) * self.sample_stride
        future_idx = anchor + (np.arange(self.future_horizon) + 1) * self.sample_stride

        window = self._data[hist_idx]
        future = self._data[future_idx]
        if self.target_sensor_only:
            future = future[:, self.target_indices]

        window_tensor = torch.from_numpy(window.T.copy()).float()
        target_tensor = torch.from_numpy(future.copy()).float()

        if self._labels is not None:
            if self.label_mode == "t_plus_1":
                label_value = float(self._labels[future_idx[0]])
            else:
                label_value = float(np.max(self._labels[future_idx]))
            label = torch.tensor(label_value, dtype=torch.float32)
        else:
            label = torch.tensor(0.0, dtype=torch.float32)
        return window_tensor, target_tensor, label

    def denormalize_tensor(self, x: torch.Tensor) -> torch.Tensor:
        if not self.target_sensor_only:
            return self.base.denormalize_tensor(x)
        if x.shape[-1] != len(self.sensor_idx):
            raise ValueError(
                "For sensor-only targets, denormalize_tensor expects last dim "
                f"{len(self.sensor_idx)} but got {x.shape[-1]}"
            )
        full_shape = list(x.shape[:-1]) + [self.num_tags]
        full = torch.zeros(full_shape, dtype=x.dtype, device=x.device)
        full[..., self.sensor_idx] = x
        denorm_full = self.base.denormalize_tensor(full)
        return denorm_full[..., self.sensor_idx]


def _apply_train_debug_options(
    train_arr: np.ndarray,
    train_labels: Optional[np.ndarray],
    cfg: DictConfig,
) -> tuple[np.ndarray, Optional[np.ndarray]]:
    max_samples = cfg.dataset.get("max_samples")
    if max_samples is not None:
        max_samples = int(max_samples)
        train_arr = train_arr[:max_samples]
        if train_labels is not None:
            train_labels = train_labels[:max_samples]

    debug_first_k = cfg.dataset.get("debug_first_k")
    if debug_first_k:
        debug_first_k = int(debug_first_k)
        train_arr = train_arr[:debug_first_k]
        if train_labels is not None:
            train_labels = train_labels[:debug_first_k]

    window = int(cfg.dataset.window_size)
    horizon = int(cfg.dataset.horizon)
    if cfg.dataset.get("debug_repeat_first"):
        if len(train_arr) == 0:
            raise ValueError("No samples available to repeat for debug mode")
        repeat_len = int(cfg.dataset.get("debug_repeat_length") or (window + horizon + 5))
        train_arr = np.repeat(train_arr[0:1], repeat_len, axis=0)
        if train_labels is not None:
            train_labels = np.repeat(train_labels[0:1], repeat_len, axis=0)
    elif debug_first_k:
        needed = max((window + horizon + 1) * 2, window + horizon + 5)
        if len(train_arr) < needed:
            repeats = int(np.ceil(needed / len(train_arr)))
            train_arr = np.tile(train_arr, (repeats, 1))
            if train_labels is not None:
                train_labels = np.tile(train_labels, repeats)
    return train_arr, train_labels


def _normalization_stats(
    train_core: np.ndarray,
    norm_mode: str,
    std_floor: float,
    sensor_idx: list[int],
    actuator_idx: list[int],
) -> dict:
    if norm_mode == "zscore":
        return {"mean": train_core.mean(axis=0), "std": train_core.std(axis=0)}
    if norm_mode == "minmax":
        return {"data_min": train_core.min(axis=0), "data_max": train_core.max(axis=0)}
    if norm_mode == "robust":
        probe = OneStepDataset(
            train_core,
            norm_mode=norm_mode,
            std_floor=std_floor,
            sensor_idx=sensor_idx,
            actuator_idx=actuator_idx,
        )
        return {"median": probe.median, "iqr": probe.iqr}
    return {}


def _normalization_stats_from_override(normalization_override: dict, norm_mode: str) -> dict:
    def _array(name: str, dtype=np.float32) -> Optional[np.ndarray]:
        value = normalization_override.get(name)
        if value is None:
            return None
        arr = np.asarray(value, dtype=dtype)
        if arr.size == 0:
            return None
        return arr

    if norm_mode == "zscore":
        mean = _array("mean")
        std = _array("std")
        if mean is None or std is None:
            raise ValueError("Loaded zscore normalization stats must include mean and std")
        return {"mean": mean, "std": std}
    if norm_mode == "minmax":
        data_min = _array("data_min")
        data_max = _array("data_max")
        mask = _array("minmax_variable_mask", dtype=bool)
        if data_min is None or data_max is None:
            raise ValueError("Loaded minmax normalization stats must include data_min and data_max")
        return {
            "data_min": data_min,
            "data_max": data_max,
            "minmax_variable_mask": mask,
        }
    if norm_mode == "robust":
        median = _array("median")
        iqr = _array("iqr")
        if median is None or iqr is None:
            raise ValueError("Loaded robust normalization stats must include median and iqr")
        return {"median": median, "iqr": iqr}
    return {}


def _jsonable_array(value: Optional[np.ndarray]) -> Optional[list[float]]:
    if value is None:
        return None
    return np.asarray(value, dtype=np.float64).tolist()


def normalization_metadata(dataset: OneStepDataset, feature_columns: list[str]) -> dict:
    return {
        "mode": dataset.norm_mode,
        "std_floor": float(dataset.std_floor),
        "feature_columns": list(feature_columns),
        "mean": _jsonable_array(getattr(dataset, "_mean", None)),
        "std": _jsonable_array(getattr(dataset, "_std", None)),
        "median": _jsonable_array(getattr(dataset, "_median", None)),
        "iqr": _jsonable_array(getattr(dataset, "_iqr", None)),
        "data_min": _jsonable_array(getattr(dataset, "_data_min", None)),
        "data_max": _jsonable_array(getattr(dataset, "_data_max", None)),
        "minmax_variable_mask": (
            np.asarray(dataset._minmax_variable_mask, dtype=bool).tolist()
            if getattr(dataset, "_minmax_variable_mask", None) is not None
            else None
        ),
        "fit_split": "train",
    }


def make_trajectory_splits(
    cfg: DictConfig,
    *,
    normalization_override: Optional[dict] = None,
) -> TrajectorySplits:
    loaded = load_dataset_arrays(cfg)
    train_arr, train_labels = _apply_train_debug_options(loaded.train, loaded.train_labels, cfg)
    test_arr = loaded.test
    test_labels = loaded.test_labels

    dataset_name = str(cfg.dataset.get("name", "swat"))
    explicit_sensor_idx = cfg.dataset.get("sensor_idx")
    explicit_actuator_idx = cfg.dataset.get("actuator_idx")
    sensor_idx, actuator_idx = build_sensor_actuator_indices(
        dataset_name,
        loaded.feature_columns,
        sensor_idx=[int(i) for i in explicit_sensor_idx] if explicit_sensor_idx is not None else None,
        actuator_idx=[int(i) for i in explicit_actuator_idx] if explicit_actuator_idx is not None else None,
    )

    val_ratio = float(cfg.dataset.val_ratio)
    if not 0.0 <= val_ratio < 0.5:
        raise ValueError(f"val_ratio should be between 0 and 0.5 for {dataset_name}")

    window = int(cfg.dataset.window_size)
    horizon = int(cfg.dataset.horizon)
    min_val_len = window + horizon + 1
    val_len = max(min_val_len, int(len(train_arr) * val_ratio)) if val_ratio > 0 else min_val_len
    val_len = min(val_len, max(len(train_arr) // 2, min_val_len))
    if val_len >= len(train_arr):
        raise ValueError("Not enough samples for validation split")

    train_core = train_arr[:-val_len]
    val_core = train_arr[-val_len:]
    train_labels_core = train_labels[:-val_len] if train_labels is not None else None
    val_labels_core = train_labels[-val_len:] if train_labels is not None else None

    norm_cfg = cfg.dataset.get("normalization", {})
    norm_mode = (norm_cfg.get("mode") or "zscore").lower()
    if norm_mode == "standard":
        norm_mode = "zscore"
    std_floor = float(norm_cfg.get("std_floor", 1e-2))

    if normalization_override is not None:
        override_mode = str(normalization_override.get("normalization_mode", norm_mode)).lower()
        if override_mode == "standard":
            override_mode = "zscore"
        if override_mode != norm_mode:
            raise ValueError(
                "Loaded normalization stats mode does not match config: "
                f"{override_mode} vs {norm_mode}"
            )
        stats = _normalization_stats_from_override(normalization_override, norm_mode)
    else:
        stats = _normalization_stats(train_core, norm_mode, std_floor, sensor_idx, actuator_idx)

    def build(array: np.ndarray, labels: Optional[np.ndarray]) -> OneStepDataset:
        return OneStepDataset(
            array,
            norm_mode=norm_mode,
            std_floor=std_floor,
            labels=labels,
            sensor_idx=sensor_idx,
            actuator_idx=actuator_idx,
            **stats,
        )

    train_base = build(train_core, train_labels_core)
    val_base = build(val_core, val_labels_core)
    test_base = build(test_arr, test_labels)

    history_len = int(cfg.model.get("history_len", cfg.dataset.get("window_size", 1)))
    future_horizon = int(cfg.model.get("horizon", cfg.dataset.get("horizon", 1)))
    sample_stride = int(cfg.model.get("sample_stride", 1))
    target_sensor_only = bool(cfg.model.get("target_sensor_only", True))
    label_mode = str(cfg.model.get("trajectory_label_mode", "horizon_max"))

    train = OneStepHistoryFutureDataset(
        train_base,
        history_len,
        future_horizon,
        sample_stride=sample_stride,
        target_sensor_only=target_sensor_only,
        label_mode=label_mode,
    )
    val = OneStepHistoryFutureDataset(
        val_base,
        history_len,
        future_horizon,
        sample_stride=sample_stride,
        target_sensor_only=target_sensor_only,
        label_mode=label_mode,
    )
    test = OneStepHistoryFutureDataset(
        test_base,
        history_len,
        future_horizon,
        sample_stride=sample_stride,
        target_sensor_only=target_sensor_only,
        label_mode=label_mode,
    )

    return TrajectorySplits(
        train=train,
        val=val,
        test=test,
        num_tags=int(train_base.num_tags),
        feature_columns=loaded.feature_columns,
        sensor_idx=sensor_idx,
        actuator_idx=actuator_idx,
        train_rows=int(train_core.shape[0]),
        val_rows=int(val_core.shape[0]),
        test_rows=int(test_arr.shape[0]),
        normalization=normalization_metadata(train_base, loaded.feature_columns),
    )


def build_dataloaders(cfg: DictConfig, splits: TrajectorySplits):
    batch_size = int(cfg.dataset.batch_size)
    num_workers = int(cfg.dataset.num_workers)
    shuffle = bool(cfg.dataset.shuffle_train)
    train_loader = DataLoader(
        splits.train,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        drop_last=False,
    )
    val_loader = DataLoader(
        splits.val,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    test_loader = DataLoader(
        splits.test,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        drop_last=False,
    )
    return train_loader, val_loader, test_loader
