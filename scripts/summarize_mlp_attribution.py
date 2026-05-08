#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ics_symbolic_distill.attribution import (
    build_target_attribution_summary,
    detect_floored_channels,
    rank_of_feature,
    rank_of_feature_excluding,
    target_top_features,
    write_rankings_json_excluding,
)
from ics_symbolic_distill.data.normalization import load_normalization_stats


WATCHED_FEATURES = ["LIT101", "FIT101", "FIT201", "MV101", "P101"]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _print_top(title: str, rows: list[dict], limit: int = 10) -> None:
    print(f"\n{title}")
    for item in rows[:limit]:
        print(f"  {item['rank']:2d}. {item['feature']}={item['value']:.6g}")


def _print_ranks(
    title: str,
    matrix: np.ndarray,
    feature_columns: list[str],
    target_columns: list[str],
    target: str,
    *,
    exclude_indices: list[int] | None = None,
) -> None:
    print(f"\n{title}")
    for feature in WATCHED_FEATURES:
        if exclude_indices is None:
            rank = rank_of_feature(matrix, target, feature, feature_columns, target_columns)
        else:
            rank = rank_of_feature_excluding(
                matrix,
                target,
                feature,
                feature_columns,
                target_columns,
                exclude_indices=exclude_indices,
            )
        print(f"  {feature}: {rank}")


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize MLP attribution with floored-channel-aware rankings.")
    parser.add_argument("--attribution-dir", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--distill-dir", required=True)
    parser.add_argument("--target", default="LIT101")
    parser.add_argument("--tolerance", type=float, default=1e-8)
    args = parser.parse_args()

    attribution_dir = Path(args.attribution_dir)
    checkpoint_dir = Path(args.checkpoint_dir)
    distill_dir = Path(args.distill_dir)
    target = str(args.target)

    columns = _read_json(checkpoint_dir / "columns.json")
    feature_columns = [str(x) for x in columns["feature_columns"]]
    target_columns = [str(x) for x in columns["target_columns"]]
    sensor_idx = [int(i) for i in columns["sensor_idx"]]
    distill_feature_columns = [str(x) for x in _read_json(distill_dir / "distill_feature_columns.json")]
    distill_target_columns = [str(x) for x in _read_json(distill_dir / "distill_target_columns.json")]
    distill_sensor_idx = [int(i) for i in _read_json(distill_dir / "distill_sensor_idx.json")]
    if feature_columns != distill_feature_columns:
        raise ValueError("checkpoint and distillation feature columns differ")
    if target_columns != distill_target_columns:
        raise ValueError("checkpoint and distillation target columns differ")
    if sensor_idx != distill_sensor_idx:
        raise ValueError("checkpoint and distillation sensor_idx differ")
    for j, feature_idx in enumerate(sensor_idx):
        if target_columns[j] != feature_columns[feature_idx]:
            raise ValueError("target_columns[j] must equal feature_columns[sensor_idx[j]]")

    stats = load_normalization_stats(checkpoint_dir / "normalization_stats.npz")
    floored = detect_floored_channels(stats, feature_columns, tolerance=float(args.tolerance))
    exclude_indices = [int(i) for i in floored["indices"]]
    excluded_reason = floored["reason"]

    matrices = {
        "grad_norm_next": np.load(attribution_dir / "attribution_mlp_grad_norm_next.npy"),
        "raw_next_sensitivity": np.load(attribution_dir / "attribution_mlp_sensitivity_raw_next.npy"),
        "raw_delta_sensitivity": np.load(attribution_dir / "attribution_mlp_sensitivity_raw_delta.npy"),
        "corr_mlp_pred_delta": np.load(attribution_dir / "attribution_corr_mlp_pred_delta.npy"),
    }

    write_rankings_json_excluding(
        attribution_dir / "attribution_mlp_grad_norm_next_rankings_nonfloored.json",
        matrices["grad_norm_next"],
        feature_columns,
        target_columns,
        sensor_idx,
        attribution_type="grad_norm_next_nonfloored",
        exclude_indices=exclude_indices,
        excluded_reason=excluded_reason,
    )
    write_rankings_json_excluding(
        attribution_dir / "attribution_mlp_sensitivity_raw_next_rankings_nonfloored.json",
        matrices["raw_next_sensitivity"],
        feature_columns,
        target_columns,
        sensor_idx,
        attribution_type="sensitivity_raw_next_nonfloored",
        exclude_indices=exclude_indices,
        excluded_reason=excluded_reason,
    )
    write_rankings_json_excluding(
        attribution_dir / "attribution_mlp_sensitivity_raw_delta_rankings_nonfloored.json",
        matrices["raw_delta_sensitivity"],
        feature_columns,
        target_columns,
        sensor_idx,
        attribution_type="sensitivity_raw_delta_nonfloored",
        exclude_indices=exclude_indices,
        excluded_reason=excluded_reason,
    )
    write_rankings_json_excluding(
        attribution_dir / "attribution_corr_mlp_pred_delta_rankings_nonfloored.json",
        matrices["corr_mlp_pred_delta"],
        feature_columns,
        target_columns,
        sensor_idx,
        attribution_type="corr_mlp_pred_delta_nonfloored",
        exclude_indices=exclude_indices,
        excluded_reason=excluded_reason,
    )

    summary = build_target_attribution_summary(
        target_name=target,
        matrices=matrices,
        feature_columns=feature_columns,
        target_columns=target_columns,
        sensor_idx=sensor_idx,
        floored=floored,
        watched_features=WATCHED_FEATURES,
    )
    (attribution_dir / f"attribution_summary_{target.lower()}.json").write_text(
        json.dumps(summary, indent=2),
        encoding="utf-8",
    )

    print("Floored channels:")
    print(", ".join(floored["channels"]) if floored["channels"] else "(none)")

    _print_top(
        f"{target} top 10 normalized gradient",
        target_top_features(matrices["grad_norm_next"], target, feature_columns, target_columns, sensor_idx),
    )
    _print_top(
        f"{target} top 10 raw sensitivity",
        target_top_features(matrices["raw_delta_sensitivity"], target, feature_columns, target_columns, sensor_idx),
    )
    _print_top(
        f"{target} top 10 raw sensitivity excluding floored channels",
        target_top_features(
            matrices["raw_delta_sensitivity"],
            target,
            feature_columns,
            target_columns,
            sensor_idx,
            exclude_indices=exclude_indices,
        ),
    )
    _print_top(
        f"{target} top 10 MLP predicted delta correlation",
        target_top_features(matrices["corr_mlp_pred_delta"], target, feature_columns, target_columns, sensor_idx),
    )
    _print_ranks(
        f"{target} ranks, normalized gradient",
        matrices["grad_norm_next"],
        feature_columns,
        target_columns,
        target,
    )
    _print_ranks(
        f"{target} ranks, raw delta sensitivity",
        matrices["raw_delta_sensitivity"],
        feature_columns,
        target_columns,
        target,
    )
    _print_ranks(
        f"{target} ranks, raw delta sensitivity nonfloored",
        matrices["raw_delta_sensitivity"],
        feature_columns,
        target_columns,
        target,
        exclude_indices=exclude_indices,
    )
    _print_ranks(
        f"{target} ranks, MLP predicted delta correlation",
        matrices["corr_mlp_pred_delta"],
        feature_columns,
        target_columns,
        target,
    )
    print("\nRecommendation:")
    print(summary["recommendation"])
    print("\nSummary JSON:")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
