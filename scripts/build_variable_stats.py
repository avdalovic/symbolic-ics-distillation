#!/usr/bin/env python3
"""Generate paper_artifacts/expressiveness_v1/variable_stats.csv.

Per-variable cardinality on BENIGN TRAINING data for all four datasets. This file
is consumed by audit_geco_expressiveness_class.py, evaluate_geco_class_restriction.py
and the overnight task scripts; it previously existed with no generator, making its
provenance unverifiable. This script is that generator.

Definitions: n_unique counts distinct finite values in the benign training split;
is_constant is n_unique <= 1; is_binary is n_unique <= 2 with values a subset of {0,1}.
"""
from __future__ import annotations
import argparse
from pathlib import Path
import numpy as np, pandas as pd

REPO = Path(__file__).resolve().parents[1]


def stats_from_frame(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    rows = []
    for c in df.columns:
        v = pd.to_numeric(df[c], errors="coerce").dropna().values
        if v.size == 0:
            rows.append({"dataset": dataset, "variable": c, "n_unique": 0,
                         "is_constant": True, "is_binary": False}); continue
        u = np.unique(v)
        rows.append({"dataset": dataset, "variable": c, "n_unique": int(u.size),
                     "is_constant": bool(u.size <= 1),
                     "is_binary": bool(u.size <= 2 and np.all(np.isin(u, [0.0, 1.0])))})
    return pd.DataFrame(rows)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--swat-train", default="data/swat/raw/swat_train.csv")
    ap.add_argument("--wadi-train", default="data/wadi/raw/wadi_train.csv")
    ap.add_argument("--out", default="paper_artifacts/expressiveness_v1/variable_stats.csv")
    args = ap.parse_args()
    frames = []

    b = pd.read_csv(REPO / "results/batadal/variable_classification.csv")
    frames.append(pd.DataFrame({"dataset": "BATADAL", "variable": b["variable"].astype(str),
        "n_unique": b["unique_values"].astype(int),
        "is_constant": b["unique_values"] <= 1,
        "is_binary": (b["unique_values"] <= 2) & (b["min"] >= 0) & (b["max"] <= 1)}))

    h = pd.read_csv(REPO / "artifacts/experiments/seed3/hai_safe_seed4/variable_typing.csv")
    frames.append(pd.DataFrame({"dataset": "HAI_R13", "variable": h["variable"].astype(str),
        "n_unique": h["unique_value_count"].astype(int),
        "is_constant": h["unique_value_count"] <= 1,
        "is_binary": (h["unique_value_count"] <= 2) & (h["min"] >= 0) & (h["max"] <= 1)}))

    for name, path, drop in [("SWaT", args.swat_train, ["Timestamp", "Normal/Attack", "Attack", "Row"]),
                             ("WADI", args.wadi_train, ["Row", "Attack", "Timestamp"])]:
        p = Path(path)
        if not p.exists():
            print(f"{name}: training CSV not found at {p}; skipping (restricted data)"); continue
        df = pd.read_csv(p)
        df = df.drop(columns=[c for c in drop if c in df.columns])
        frames.append(stats_from_frame(df, name))

    out = pd.concat(frames, ignore_index=True)
    dest = REPO / args.out; dest.parent.mkdir(parents=True, exist_ok=True)
    out.to_csv(dest, index=False)
    print(out.groupby("dataset").agg(vars=("variable", "size"), binary=("is_binary", "sum"),
                                     constant=("is_constant", "sum")).to_string())
    print(f"\nWrote {dest}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
