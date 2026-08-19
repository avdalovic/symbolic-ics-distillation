#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
TRAIN_SOURCE = RAW / "WADI_14days.csv"
TEST_SOURCE = RAW / "WADI_attackdata.csv"
TRAIN_OUT = RAW / "wadi_train.csv"
TEST_OUT = RAW / "wadi_test.csv"


ATTACK_WINDOWS = [
    (5139, 6624 + 1),
    (59050, 59641),
    (60900, 62641),
    (63040, 63891),
    (70770, 71441),
    (74895, 75591 + 1),
    (85237, 85778 + 1),
    (147296, 147379 + 1),
    (148656, 149489 + 1),
    (149792, 150416 + 1),
    (151131, 151507 + 1),
    (151660, 151851 + 1),
    (152167, 152746 + 1),
    (163590, 164221),
]


NAN_PATCHES = {
    "2B_AIT_004_PV": [
        (61703, 61713, 486.6),
        (384154, 384164, 487.926),
    ],
    "3_AIT_002_PV": [(524280, 524286, 8279.1)],
    "1_AIT_002_PV": [
        (623873, 623879, 0.71646),
        (884845, 884851, 0.62047),
    ],
    "1_AIT_004_PV": [(706470, 706476, 501.642)],
    "3_AIT_004_PV": [(974807, 974818, 1603.980)],
}


def strip_logger_prefix(name: object) -> str:
    text = str(name).strip()
    marker = r"\LOG_DATA\SUTD_WADI\LOG_DATA\\"
    if marker in text:
        return text.split(marker, 1)[1].strip()
    if len(text) > 46 and text[:2] == r"\\":
        return text[46:].strip()
    return text


def normalize_columns(train: pd.DataFrame, test: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    train = train.copy()
    test = test.copy()
    train_cols = list(train.columns)
    test_cols = list(test.columns)
    for i in range(3, len(train_cols)):
        clean = strip_logger_prefix(train_cols[i])
        train_cols[i] = clean
        if i < len(test_cols):
            test_cols[i] = clean
    train.columns = train_cols
    test.columns = test_cols
    return train, test


def label_attacks(test: pd.DataFrame) -> np.ndarray:
    labels = np.zeros(len(test), dtype=np.int64)
    for start, end in ATTACK_WINDOWS:
        if "Time" in test.columns:
            print(f"Label attack from {test['Time'].loc[start]} to {test['Time'].loc[end - 1]}")
        labels[start:end] = 1
    return labels


def apply_nan_patches(train: pd.DataFrame) -> pd.DataFrame:
    out = train.copy()
    for col, patches in NAN_PATCHES.items():
        if col not in out.columns:
            print(f"WARNING: expected WADI column not found for NaN patch: {col}")
            continue
        for start, end, value in patches:
            out.loc[start : end - 1, col] = value
    return out


def main() -> None:
    if not TRAIN_SOURCE.exists() or not TEST_SOURCE.exists():
        raise SystemExit(
            "Missing WADI CSVs. Expected:\n"
            f"  {TRAIN_SOURCE}\n"
            f"  {TEST_SOURCE}\n"
            "Request WADI from iTrust/SUTD and place the downloaded CSVs under data/wadi/raw/."
        )

    train = pd.read_csv(TRAIN_SOURCE, header=3)
    test = pd.read_csv(TEST_SOURCE, header=0)
    train, test = normalize_columns(train, test)

    train["Attack"] = np.zeros(len(train), dtype=np.int64)
    test["Attack"] = label_attacks(test)
    train = apply_nan_patches(train)

    RAW.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)
    print(f"Wrote {TRAIN_OUT} shape={train.shape}")
    print(f"Wrote {TEST_OUT} shape={test.shape}")


if __name__ == "__main__":
    main()
