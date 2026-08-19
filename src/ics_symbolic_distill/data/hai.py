from __future__ import annotations

from dataclasses import dataclass
import gzip
import hashlib
import json
import math
from pathlib import Path
from typing import Any, Iterable, Sequence

import numpy as np
import pandas as pd

from ics_symbolic_distill.detection.cusum import CusumParams, run_cusum


HAI_RELEASE = "21.03"
EXPECTED_RAW_FILES = [
    "train1.csv.gz",
    "train2.csv.gz",
    "train3.csv.gz",
    "test1.csv.gz",
    "test2.csv.gz",
    "test3.csv.gz",
    "test4.csv.gz",
    "test5.csv.gz",
]
EXPECTED_TRAIN_FILES = ["train1.state.gz", "train2.state.gz", "train3.state.gz"]
EXPECTED_TEST_FILES = [
    "test1.state.gz",
    "test2.state.gz",
    "test3.state.gz",
    "test4.state.gz",
    "test5.state.gz",
]
EXPECTED_STATE_FILES = EXPECTED_TRAIN_FILES + EXPECTED_TEST_FILES
EXPECTED_ATTACK_COUNT = 50


@dataclass(frozen=True)
class HaiSequence:
    name: str
    path: Path
    split: str
    timestamps: np.ndarray
    frame: pd.DataFrame
    attack_ids: np.ndarray

    @property
    def labels(self) -> np.ndarray:
        return (self.attack_ids > 0).astype(np.int64)


@dataclass(frozen=True)
class HaiPairArrays:
    current_blocks: tuple[np.ndarray, ...]
    next_blocks: tuple[np.ndarray, ...]
    label_blocks: tuple[np.ndarray, ...]
    attack_id_blocks: tuple[np.ndarray, ...]
    timestamp_blocks: tuple[np.ndarray, ...]
    sequence_names: tuple[str, ...]
    feature_columns: tuple[str, ...]

    @property
    def n_pairs(self) -> int:
        return int(sum(block.shape[0] for block in self.current_blocks))

    def flatten_current(self) -> np.ndarray:
        return concat_blocks(self.current_blocks)

    def flatten_next(self) -> np.ndarray:
        return concat_blocks(self.next_blocks)

    def flatten_labels(self) -> np.ndarray:
        return concat_blocks(self.label_blocks).astype(np.int64)

    def flatten_attack_ids(self) -> np.ndarray:
        return concat_blocks(self.attack_id_blocks).astype(np.int64)

    def flatten_timestamps(self) -> np.ndarray:
        return concat_blocks(self.timestamp_blocks).astype(np.int64)


def concat_blocks(blocks: Sequence[np.ndarray]) -> np.ndarray:
    if not blocks:
        return np.asarray([], dtype=np.float64)
    return np.concatenate([np.asarray(block) for block in blocks], axis=0)


def sha256_file(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(chunk_size), b""):
            digest.update(chunk)
    return digest.hexdigest()


def jsonable(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [jsonable(v) for v in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False) + "\n", encoding="utf-8")


def _malicious_to_attack_id(value: Any) -> int:
    if value is None or value is False:
        return 0
    if isinstance(value, str):
        text = value.strip().lower()
        if text in {"", "false", "normal", "benign", "0", "0.0"}:
            return 0
        if text in {"true", "attack", "malicious"}:
            return 1
        try:
            return max(0, int(float(text)))
        except ValueError:
            return 1
    try:
        return max(0, int(float(value)))
    except Exception:
        return 1


def load_state_gz(path: Path, *, split: str | None = None) -> HaiSequence:
    """Load one IPAL HAI ``*.state.gz`` file as an independent sequence."""

    rows: list[dict[str, float]] = []
    timestamps: list[int] = []
    attack_ids: list[int] = []
    path = Path(path)
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                obj = json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
            state = obj.get("state")
            if not isinstance(state, dict):
                raise ValueError(f"{path}:{line_no}: missing JSON object field 'state'")
            timestamp = obj.get("timestamp")
            if timestamp is None:
                raise ValueError(f"{path}:{line_no}: missing field 'timestamp'")
            timestamps.append(int(float(timestamp)))
            attack_ids.append(_malicious_to_attack_id(obj.get("malicious", obj.get("attack", False))))
            rows.append({str(key): float(pd.to_numeric(value, errors="coerce")) for key, value in state.items()})

    if not rows:
        raise ValueError(f"{path}: no JSON state rows found")
    frame = pd.DataFrame(rows)
    for col in frame.columns:
        frame[col] = pd.to_numeric(frame[col], errors="coerce")
    return HaiSequence(
        name=path.name,
        path=path,
        split=split or ("train" if path.name.startswith("train") else "test"),
        timestamps=np.asarray(timestamps, dtype=np.int64),
        frame=frame.astype(np.float32),
        attack_ids=np.asarray(attack_ids, dtype=np.int64),
    )


def load_hai_sequences(data_dir: Path, *, split: str | None = None) -> list[HaiSequence]:
    data_dir = Path(data_dir)
    if split == "train":
        names = EXPECTED_TRAIN_FILES
    elif split == "test":
        names = EXPECTED_TEST_FILES
    elif split is None:
        names = EXPECTED_STATE_FILES
    else:
        raise ValueError(f"unknown split: {split}")
    sequences = []
    for name in names:
        path = data_dir / name
        if not path.exists():
            raise FileNotFoundError(path)
        sequences.append(load_state_gz(path, split="train" if name.startswith("train") else "test"))
    return sequences


def shared_state_keys(sequences: Sequence[HaiSequence]) -> list[str]:
    if not sequences:
        return []
    keys = set(sequences[0].frame.columns)
    for seq in sequences[1:]:
        keys &= set(seq.frame.columns)
    return sorted(keys)


def sequence_key_differences(sequences: Sequence[HaiSequence]) -> list[dict[str, Any]]:
    if not sequences:
        return []
    intersection = set(shared_state_keys(sequences))
    union = sorted(set().union(*(set(seq.frame.columns) for seq in sequences)))
    rows = []
    for seq in sequences:
        keys = set(seq.frame.columns)
        rows.append(
            {
                "sequence": seq.name,
                "split": seq.split,
                "num_keys": len(keys),
                "missing_from_union": sorted(set(union) - keys),
                "extra_relative_to_intersection": sorted(keys - intersection),
            }
        )
    return rows


def make_pair_arrays(sequences: Sequence[HaiSequence], feature_columns: Sequence[str] | None = None) -> HaiPairArrays:
    """Create one-step pairs without crossing file boundaries."""

    columns = list(feature_columns) if feature_columns is not None else shared_state_keys(sequences)
    current_blocks: list[np.ndarray] = []
    next_blocks: list[np.ndarray] = []
    label_blocks: list[np.ndarray] = []
    attack_id_blocks: list[np.ndarray] = []
    timestamp_blocks: list[np.ndarray] = []
    names: list[str] = []
    for seq in sequences:
        if seq.frame.shape[0] < 2:
            continue
        missing = [col for col in columns if col not in seq.frame.columns]
        if missing:
            raise ValueError(f"{seq.name}: missing feature columns {missing[:5]}")
        values = seq.frame[columns].to_numpy(dtype=np.float32)
        current_blocks.append(values[:-1].astype(np.float32, copy=False))
        next_blocks.append(values[1:].astype(np.float32, copy=False))
        label_blocks.append(seq.labels[1:].astype(np.int64, copy=False))
        attack_id_blocks.append(seq.attack_ids[1:].astype(np.int64, copy=False))
        timestamp_blocks.append(seq.timestamps[1:].astype(np.int64, copy=False))
        names.append(seq.name)
    return HaiPairArrays(
        current_blocks=tuple(current_blocks),
        next_blocks=tuple(next_blocks),
        label_blocks=tuple(label_blocks),
        attack_id_blocks=tuple(attack_id_blocks),
        timestamp_blocks=tuple(timestamp_blocks),
        sequence_names=tuple(names),
        feature_columns=tuple(columns),
    )


def train_fit_holdout_indices(pairs: HaiPairArrays, fit_fraction: float = 0.8) -> tuple[np.ndarray, np.ndarray]:
    fit: list[np.ndarray] = []
    holdout: list[np.ndarray] = []
    offset = 0
    for block in pairs.current_blocks:
        n = int(block.shape[0])
        cutoff = int(math.floor(n * float(fit_fraction)))
        fit.append(np.arange(offset, offset + cutoff, dtype=np.int64))
        holdout.append(np.arange(offset + cutoff, offset + n, dtype=np.int64))
        offset += n
    return concat_blocks(fit).astype(np.int64), concat_blocks(holdout).astype(np.int64)


def block_lengths(pairs: HaiPairArrays) -> list[int]:
    return [int(block.shape[0]) for block in pairs.current_blocks]


def split_flat_by_blocks(values: np.ndarray, lengths: Sequence[int]) -> list[np.ndarray]:
    arr = np.asarray(values)
    out = []
    offset = 0
    for length in lengths:
        n = int(length)
        out.append(arr[offset : offset + n])
        offset += n
    if offset != arr.shape[0]:
        raise ValueError(f"block lengths sum to {offset}, but values has length {arr.shape[0]}")
    return out


def fit_cusum_params_sequences(
    residual_blocks: Sequence[np.ndarray],
    *,
    s: float,
    g: float,
) -> CusumParams:
    """Fit CUSUM on several benign sequences, resetting at each boundary."""

    finite_blocks = []
    for block in residual_blocks:
        r = np.asarray(block, dtype=np.float64).reshape(-1)
        finite_blocks.append(r[np.isfinite(r)])
    finite = concat_blocks(finite_blocks)
    if finite.size == 0:
        delta = 1e-6
    else:
        delta = float(np.mean(finite) + np.std(finite))
        if not np.isfinite(delta) or delta <= 0.0:
            delta = 1e-6
    train_max = 0.0
    for block in finite_blocks:
        cusum = 0.0
        for value in block:
            cusum = max(0.0, cusum + float(value) - delta)
            train_max = max(train_max, cusum)
    threshold = float(s) * train_max
    if not np.isfinite(threshold) or threshold <= 0.0:
        threshold = 1e-6
    growth_cap = threshold + float(g) * delta
    if not np.isfinite(growth_cap) or growth_cap <= 0.0:
        growth_cap = 1e-6
    return CusumParams(
        delta=float(delta),
        threshold=float(threshold),
        growth_cap=float(growth_cap),
        max_calib_cusum=float(train_max),
        s=float(s),
        g=float(g),
    )


def run_cusum_sequences(
    residual_blocks: Sequence[np.ndarray],
    params: CusumParams,
) -> tuple[np.ndarray, np.ndarray, list[np.ndarray], list[np.ndarray]]:
    """Run CUSUM with state reset for each independent sequence."""

    cusum_blocks: list[np.ndarray] = []
    alarm_blocks: list[np.ndarray] = []
    for block in residual_blocks:
        cusum, alarm = run_cusum(np.asarray(block, dtype=np.float64), params)
        cusum_blocks.append(cusum)
        alarm_blocks.append(alarm)
    return (
        concat_blocks(cusum_blocks).astype(np.float64),
        concat_blocks(alarm_blocks).astype(np.int64),
        cusum_blocks,
        alarm_blocks,
    )


def sequence_manifest_rows(sequences: Sequence[HaiSequence]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for seq in sequences:
        diffs = np.diff(seq.timestamps)
        rows.append(
            {
                "sequence": seq.name,
                "split": seq.split,
                "path": str(seq.path),
                "rows": int(seq.frame.shape[0]),
                "variables": int(seq.frame.shape[1]),
                "timestamp_start": int(seq.timestamps[0]) if seq.timestamps.size else None,
                "timestamp_end": int(seq.timestamps[-1]) if seq.timestamps.size else None,
                "min_gap_seconds": int(np.min(diffs)) if diffs.size else None,
                "max_gap_seconds": int(np.max(diffs)) if diffs.size else None,
                "non_1s_gap_count": int(np.sum(diffs != 1)) if diffs.size else 0,
                "attack_rows": int(np.sum(seq.labels)),
                "attack_ids": json.dumps(sorted(int(x) for x in np.unique(seq.attack_ids) if int(x) > 0)),
                "sha256": sha256_file(seq.path),
            }
        )
    return rows


def timestamp_overlap_rows(sequences: Sequence[HaiSequence]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for i, left in enumerate(sequences):
        for right in sequences[i + 1 :]:
            start = max(int(left.timestamps[0]), int(right.timestamps[0]))
            end = min(int(left.timestamps[-1]), int(right.timestamps[-1]))
            rows.append(
                {
                    "left": left.name,
                    "right": right.name,
                    "overlap_seconds": max(0, end - start + 1),
                    "overlap_start": start if end >= start else None,
                    "overlap_end": end if end >= start else None,
                }
            )
    return rows


def infer_variable_types(sequences: Sequence[HaiSequence], feature_columns: Sequence[str] | None = None) -> pd.DataFrame:
    columns = list(feature_columns) if feature_columns is not None else shared_state_keys(sequences)
    rows: list[dict[str, Any]] = []
    for col in columns:
        values = []
        unchanged = 0
        transitions = 0
        pair_count = 0
        missing = 0
        nonfinite = 0
        for seq in sequences:
            arr = pd.to_numeric(seq.frame[col], errors="coerce").to_numpy(dtype=np.float64)
            missing += int(np.sum(np.isnan(arr)))
            nonfinite += int(np.sum(~np.isfinite(arr)))
            finite = arr[np.isfinite(arr)]
            values.append(finite)
            if arr.size > 1:
                finite_pair = np.isfinite(arr[:-1]) & np.isfinite(arr[1:])
                same = (arr[:-1] == arr[1:]) & finite_pair
                unchanged += int(np.sum(same))
                transitions += int(np.sum((arr[:-1] != arr[1:]) & finite_pair))
                pair_count += int(np.sum(finite_pair))
        flat = concat_blocks(values)
        if flat.size:
            unique_values = np.unique(flat)
            unique_count = int(unique_values.size)
            min_val = float(np.min(flat))
            max_val = float(np.max(flat))
        else:
            unique_values = np.asarray([], dtype=np.float64)
            unique_count = 0
            min_val = float("nan")
            max_val = float("nan")
        unchanged_fraction = float(unchanged / pair_count) if pair_count else 0.0
        integerish = bool(flat.size and np.all(np.isclose(flat, np.round(flat), atol=1e-6)))
        unique_set = set(np.round(unique_values, 8).tolist())
        if unique_count == 0:
            inferred = "non_numeric"
            selected_model = "excluded"
            reason = "no_finite_values"
        elif unique_count == 1 or transitions == 0:
            inferred = "constant"
            selected_model = "excluded"
            reason = "constant_or_no_observed_transition"
        elif unique_count <= 2 and unique_set.issubset({0.0, 1.0}):
            inferred = "binary"
            selected_model = "persistence"
            reason = "two_value_binary_channel"
        elif unique_count <= 20 and integerish and unchanged_fraction >= 0.90:
            inferred = "discrete"
            selected_model = "persistence"
            reason = "low_cardinality_piecewise_constant"
        elif unique_count <= 100 and unchanged_fraction >= 0.995:
            inferred = "piecewise_constant"
            selected_model = "persistence"
            reason = "mostly_unchanged_piecewise_constant"
        else:
            inferred = "continuous"
            selected_model = "symbolic"
            reason = "continuous_or_frequently_changing"
        rows.append(
            {
                "variable": col,
                "unique_value_count": unique_count,
                "min": min_val,
                "max": max_val,
                "unchanged_step_fraction": unchanged_fraction,
                "transition_count": int(transitions),
                "missing_count": int(missing),
                "nonfinite_count": int(nonfinite),
                "inferred_type": inferred,
                "selected_model": selected_model,
                "reason": reason,
            }
        )
    return pd.DataFrame(rows)


def rounded_sample_size(total_rows: int, fraction: float = 0.025, base: int = 1000) -> tuple[int, dict[str, Any]]:
    raw = int(round(int(total_rows) * float(fraction)))
    rounded = int(round(raw / int(base)) * int(base))
    if rounded <= 0:
        rounded = min(int(total_rows), int(base))
    rounded = min(int(total_rows), rounded)
    return rounded, {
        "total_benign_training_pairs": int(total_rows),
        "fraction": float(fraction),
        "raw_rounded_rows": raw,
        "rounding_base": int(base),
        "sample_size": int(rounded),
    }


def load_attacks(path: Path) -> list[dict[str, Any]]:
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path}: attacks.json must contain a list")
    return payload


def assert_no_cross_sequence_pairs(sequences: Sequence[HaiSequence], pairs: HaiPairArrays) -> None:
    if len(sequences) != len(pairs.current_blocks):
        raise AssertionError("sequence/pair block count mismatch")
    for seq, current, nxt in zip(sequences, pairs.current_blocks, pairs.next_blocks, strict=True):
        if current.shape[0] != max(0, seq.frame.shape[0] - 1):
            raise AssertionError(f"{seq.name}: unexpected pair count")
        frame = seq.frame[list(pairs.feature_columns)]
        if current.shape[0] and not np.allclose(current[-1], frame.iloc[-2].to_numpy(dtype=np.float32), equal_nan=True):
            raise AssertionError(f"{seq.name}: final current row is not the sequence-local penultimate row")
        if nxt.shape[0] and not np.allclose(nxt[0], frame.iloc[1].to_numpy(dtype=np.float32), equal_nan=True):
            raise AssertionError(f"{seq.name}: first next row is not the sequence-local second row")


def file_info(path: Path) -> dict[str, Any]:
    path = Path(path)
    return {
        "name": path.name,
        "path": str(path),
        "bytes": int(path.stat().st_size),
        "sha256": sha256_file(path),
    }


def validate_expected_generated_files(data_dir: Path) -> list[Path]:
    data_dir = Path(data_dir)
    missing = [name for name in EXPECTED_STATE_FILES + ["attacks.json"] if not (data_dir / name).exists()]
    if missing:
        raise FileNotFoundError(f"missing generated HAI files under {data_dir}: {missing}")
    return [data_dir / name for name in EXPECTED_STATE_FILES]


def iter_jsonl_gzip(path: Path) -> Iterable[dict[str, Any]]:
    with gzip.open(path, "rt", encoding="utf-8") as handle:
        for line_no, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            try:
                yield json.loads(text)
            except json.JSONDecodeError as exc:
                raise ValueError(f"{path}:{line_no}: invalid JSON: {exc}") from exc
