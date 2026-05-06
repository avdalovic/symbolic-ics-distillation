from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Optional, Sequence

import numpy as np
import pandas as pd


@dataclass(frozen=True)
class AttackWindow:
    start: int
    end: int
    affected_tags: tuple[str, ...]


def is_actuator(dataset: str, label: str) -> bool:
    """Preserve the source-project SWaT/WADI sensor-actuator split heuristic."""

    dataset_name = str(dataset).upper()
    tag = str(label)
    if dataset_name == "SWAT":
        return "IT" not in tag
    if dataset_name == "WADI":
        return "STATUS" in tag
    return False


def build_sensor_actuator_indices(
    dataset_name: str,
    feature_columns: Sequence[str],
    *,
    sensor_idx: Optional[Iterable[int]] = None,
    actuator_idx: Optional[Iterable[int]] = None,
) -> tuple[list[int], list[int]]:
    num_tags = len(feature_columns)

    def _sanitize(indices: Optional[Iterable[int]], name: str) -> Optional[list[int]]:
        if indices is None:
            return None
        out = [int(i) for i in indices]
        if any(i < 0 or i >= num_tags for i in out):
            raise ValueError(f"{name} contains out-of-range indices for num_tags={num_tags}")
        if len(set(out)) != len(out):
            raise ValueError(f"{name} contains duplicates")
        return out

    sensor = _sanitize(sensor_idx, "sensor_idx")
    actuator = _sanitize(actuator_idx, "actuator_idx")

    if sensor is None and actuator is None:
        sensor = [i for i, col in enumerate(feature_columns) if not is_actuator(dataset_name, col)]
        actuator = [i for i, col in enumerate(feature_columns) if is_actuator(dataset_name, col)]
    elif sensor is None:
        actuator_set = set(actuator or [])
        sensor = [i for i in range(num_tags) if i not in actuator_set]
    elif actuator is None:
        sensor_set = set(sensor)
        actuator = [i for i in range(num_tags) if i not in sensor_set]

    overlap = set(sensor).intersection(actuator)
    if overlap:
        raise ValueError(f"sensor_idx and actuator_idx overlap: {sorted(overlap)}")

    if not sensor:
        sensor = list(range(num_tags))
        actuator = []
    return sensor, actuator


def split_sensor_actuator_columns(
    dataset_name: str,
    feature_columns: Sequence[str],
    *,
    sensor_idx: Optional[Iterable[int]] = None,
    actuator_idx: Optional[Iterable[int]] = None,
) -> tuple[list[str], list[str], list[int], list[int]]:
    sensors, actuators = build_sensor_actuator_indices(
        dataset_name,
        feature_columns,
        sensor_idx=sensor_idx,
        actuator_idx=actuator_idx,
    )
    return (
        [str(feature_columns[i]) for i in sensors],
        [str(feature_columns[i]) for i in actuators],
        sensors,
        actuators,
    )


def normalize_attack_labels(labels: Optional[pd.Series | Sequence[object] | np.ndarray]) -> Optional[np.ndarray]:
    if labels is None:
        return None
    series = pd.Series(labels).copy()
    if series.dtype == object:
        lowered = series.astype(str).str.strip().str.lower()
        mapping = {"normal": 0.0, "attack": 1.0, "n": 0.0, "a": 1.0}
        series = lowered.map(mapping).fillna(lowered)
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return numeric.to_numpy(dtype=np.float32)


_SWAT_ATTACK_WINDOWS = [
    (1738, 2672, ("MV101",)),
    (3046, 3490, ("P102",)),
    (4901, 5282, ("LIT101",)),
    (7233, 7431, ("AIT202",)),
    (7685, 8113, ("LIT301",)),
    (11385, 12355, ("DPIT301",)),
    (15361, 16083, ("FIT401",)),
    (90662, 90917, ("MV304",)),
    (93424, 93705, ("LIT301",)),
    (103092, 103797, ("MV303",)),
    (115822, 116080, ("AIT504",)),
    (116123, 116515, ("AIT504",)),
    (116999, 117700, ("LIT101",)),
    (132896, 133362, ("UV401", "AIT502")),
    (142927, 143611, ("DPIT301",)),
    (172268, 172588, ("P203", "P205")),
    (172892, 173499, ("LIT401",)),
    (198273, 199716, ("P101", "LIT301")),
    (227828, 228361, ("LIT401",)),
    (229519, 263727, ("P302",)),
    (280023, 281184, ("P101", "MV201", "LIT101")),
    (302653, 303019, ("LIT401",)),
    (347718, 348315, ("LIT301",)),
    (361243, 361674, ("LIT101",)),
    (371519, 371618, ("P101",)),
    (371893, 372374, ("P101",)),
    (389746, 390262, ("LIT101",)),
    (436672, 437046, ("FIT502",)),
    (437455, 437735, ("AIT402", "AIT502")),
    (438184, 438583, ("FIT401", "AIT502")),
    (438659, 438955, ("FIT401",)),
    (443540, 445191, ("LIT301",)),
]


_WADI_ATTACK_WINDOWS = [
    (5139, 6619, ("1_MV_001_STATUS",)),
    (59069, 59613, ("1_FIT_001_PV",)),
    (61058, 61622, ("2_MV_003_STATUS",)),
    (61667, 61936, ("1_AIT_001_PV",)),
    (
        63046,
        63891,
        (
            "2_MCV_101_CO",
            "2_MCV_201_CO",
            "2_MCV_301_CO",
            "2_MCV_401_CO",
            "2_MCV_501_CO",
            "2_MCV_601_CO",
        ),
    ),
    (70795, 71458, ("2_FIC_101_PV", "2_FIC_201_PV")),
    (74828, 75592, ("1_AIT_002_PV", "2_MV_003_STATUS")),
    (85239, 85779, ("2_MCV_007_CO",)),
    (147297, 147380, ("1_P_006_STATUS",)),
    (148657, 149479, ("1_MV_001_STATUS",)),
    (149793, 150417, ("2_MCV_007_CO",)),
    (151132, 151508, ("2_MCV_007_CO",)),
    (151661, 151853, ("2_PIC_003_CO", "2_PIC_003_SP")),
    (152174, 152742, ("1_P_001_STATUS", "1_P_003_STATUS")),
    (163804, 164221, ("2_MV_003_STATUS",)),
]


def get_attack_windows(dataset_name: str) -> list[AttackWindow]:
    name = str(dataset_name).upper()
    if name == "SWAT":
        source = _SWAT_ATTACK_WINDOWS
    elif name == "WADI":
        source = _WADI_ATTACK_WINDOWS
    else:
        source = []
    return [AttackWindow(start=s, end=e, affected_tags=tuple(tags)) for s, e, tags in source]


def labels_from_attack_windows(length: int, windows: Sequence[AttackWindow]) -> np.ndarray:
    labels = np.zeros(int(length), dtype=np.float32)
    for window in windows:
        start = max(int(window.start), 0)
        end = min(int(window.end), labels.shape[0])
        if start < end:
            labels[start:end] = 1.0
    return labels
