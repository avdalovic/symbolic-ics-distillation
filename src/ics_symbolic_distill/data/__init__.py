from typing import Any

from .ics_metadata import (
    AttackWindow,
    get_attack_windows,
    is_actuator,
    labels_from_attack_windows,
    normalize_attack_labels,
    split_sensor_actuator_columns,
)
from .normalization import (
    describe_normalization_stats,
    inverse_normalize_features,
    inverse_normalize_targets,
    load_normalization_stats,
    normalize_raw_features,
    normalization_formula,
)

_WINDOWING_EXPORTS = {
    "LoadedArrays",
    "OneStepDataset",
    "OneStepHistoryFutureDataset",
    "TrajectorySplits",
    "build_dataloaders",
    "build_sensor_actuator_indices",
    "load_dataset_arrays",
    "make_trajectory_splits",
}

__all__ = [
    "AttackWindow",
    "LoadedArrays",
    "OneStepDataset",
    "OneStepHistoryFutureDataset",
    "TrajectorySplits",
    "build_dataloaders",
    "build_sensor_actuator_indices",
    "get_attack_windows",
    "is_actuator",
    "labels_from_attack_windows",
    "load_dataset_arrays",
    "load_normalization_stats",
    "make_trajectory_splits",
    "describe_normalization_stats",
    "inverse_normalize_features",
    "inverse_normalize_targets",
    "normalize_raw_features",
    "normalization_formula",
    "normalize_attack_labels",
    "split_sensor_actuator_columns",
]


def __getattr__(name: str) -> Any:
    if name not in _WINDOWING_EXPORTS:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    from . import windowing

    value = getattr(windowing, name)
    globals()[name] = value
    return value
