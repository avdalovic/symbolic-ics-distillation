#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


ROOT = Path(__file__).resolve().parent
RAW = ROOT / "raw"
NORMAL_CSV = RAW / "SWaT_Dataset_Normal_v1.csv"
ATTACK_CSV = RAW / "SWaT_Dataset_Attack_v0.csv"
TRAIN_OUT = RAW / "swat_train.csv"
TEST_OUT = RAW / "swat_test.csv"


ATTACK_WINDOWS = [
    (1738, 2673),
    (3046, 3491),
    (4901, 5283),
    (7233, 7432),
    (7685, 8113),
    (11385, 12355),
    (15361, 16084),
    (90662, 90917),
    (93424, 93705),
    (103092, 103797),
    (115822, 116080),
    (116123, 116515),
    (116999, 117701),
    (132896, 133362),
    (142927, 143611),
    (172268, 172588),
    (172892, 173499),
    (198273, 199716),
    (227828, 228362),
    (229519, 263727),
    (280023, 281185),
    (302653, 303020),
    (347718, 348315),
    (361243, 361674),
    (371519, 371618),
    (371893, 372374),
    (389746, 390262),
    (436672, 437046),
    (437455, 437734),
    (438184, 438584),
    (438659, 438955),
    (443540, 445191),
]


def read_normal(path: Path) -> pd.DataFrame:
    first = pd.read_csv(path, header=0, nrows=1)
    first_col = str(first.columns[0]).strip()
    header = 0 if first_col == "Timestamp" else 1
    return pd.read_csv(path, header=header)


def clean_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    out.columns = out.columns.astype(str).str.strip().str.replace(" ", "", regex=False)
    return out


def numeric_process_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if col not in {"Timestamp", "Normal/Attack"}:
            out[col] = pd.to_numeric(out[col], errors="coerce")
    return out


def label_swat_attacks(test_df: pd.DataFrame) -> np.ndarray:
    labels = np.zeros(len(test_df), dtype=np.int64)
    for start, end in ATTACK_WINDOWS:
        labels[start:end] = 1
    return labels


def print_windows(test_df: pd.DataFrame, labels: np.ndarray) -> None:
    attack_idx = 0
    for i in range(len(test_df)):
        starts = labels[i] and (i == 0 or not labels[i - 1])
        ends = labels[i] and (i == len(test_df) - 1 or not labels[i + 1])
        if starts:
            print(f"Attack {attack_idx} start at {i}: {test_df['Timestamp'].iloc[i]}")
        if ends:
            print(f"Attack {attack_idx} end at {i}: {test_df['Timestamp'].iloc[i]}")
            attack_idx += 1


def main() -> None:
    if not NORMAL_CSV.exists() or not ATTACK_CSV.exists():
        raise SystemExit(
            "Missing SWaT CSVs. Expected:\n"
            f"  {NORMAL_CSV}\n"
            f"  {ATTACK_CSV}\n"
            "Request SWaT from iTrust/SUTD and export the Physical Excel files as CSV."
        )

    train = clean_columns(read_normal(NORMAL_CSV))
    test = clean_columns(pd.read_csv(ATTACK_CSV, header=0)).iloc[1:].reset_index(drop=True)

    train = numeric_process_columns(train)
    test = numeric_process_columns(test)

    train["Normal/Attack"] = 0
    test["Normal/Attack"] = label_swat_attacks(test)

    print_windows(test, test["Normal/Attack"].to_numpy(dtype=np.int64))
    RAW.mkdir(parents=True, exist_ok=True)
    train.to_csv(TRAIN_OUT, index=False)
    test.to_csv(TEST_OUT, index=False)
    print(f"Wrote {TRAIN_OUT} shape={train.shape}")
    print(f"Wrote {TEST_OUT} shape={test.shape}")


if __name__ == "__main__":
    main()
