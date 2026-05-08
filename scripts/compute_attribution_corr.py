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
    absolute_pearson_correlation,
    rank_of_feature,
    write_rankings_json,
    write_topk_csv,
)


REPORT_TARGETS = ["LIT101", "FIT101", "FIT201", "LIT301", "DPIT301", "AIT201"]


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _print_top5(title: str, matrix: np.ndarray, feature_columns: list[str], target_columns: list[str]) -> None:
    print(f"\n{title}")
    for target in REPORT_TARGETS:
        if target not in target_columns:
            print(f"{target}: not a target")
            continue
        target_idx = target_columns.index(target)
        order = np.argsort(-matrix[target_idx], kind="mergesort")[:5]
        rendered = ", ".join(f"{feature_columns[i]}={float(matrix[target_idx, i]):.6g}" for i in order)
        print(f"{target}: {rendered}")


def _lit101_report(matrix: np.ndarray, feature_columns: list[str], target_columns: list[str]) -> dict:
    target = "LIT101"
    if target not in target_columns:
        return {"target": target, "available": False}
    target_idx = target_columns.index(target)
    order = np.argsort(-matrix[target_idx], kind="mergesort")[:10]
    top10 = [
        {
            "rank": int(rank),
            "feature": feature_columns[int(feature_idx)],
            "feature_index": int(feature_idx),
            "value": float(matrix[target_idx, int(feature_idx)]),
        }
        for rank, feature_idx in enumerate(order, start=1)
    ]
    ranks = {
        feature: rank_of_feature(matrix, target, feature, feature_columns, target_columns)
        for feature in ["LIT101", "FIT101", "FIT201"]
    }
    return {
        "target": target,
        "available": True,
        "mlp_pred_delta_top10": top10,
        "ranks": ranks,
        "FIT101_in_mlp_pred_delta_top10": ranks["FIT101"] is not None and ranks["FIT101"] <= 10,
        "FIT201_in_mlp_pred_delta_top10": ranks["FIT201"] is not None and ranks["FIT201"] <= 10,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute raw correlation attribution fallbacks.")
    parser.add_argument("--distill-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    root = Path(args.distill_dir)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    feature_columns = [str(x) for x in _read_json(root / "distill_feature_columns.json")]
    target_columns = [str(x) for x in _read_json(root / "distill_target_columns.json")]
    sensor_idx = [int(i) for i in _read_json(root / "distill_sensor_idx.json")]
    for j, feature_idx in enumerate(sensor_idx):
        if target_columns[j] != feature_columns[feature_idx]:
            raise ValueError("target_columns[j] must equal feature_columns[sensor_idx[j]]")

    inputs = np.load(root / "distill_inputs_current_raw.npy")
    pred_next = np.load(root / "distill_pred_next_raw_mlp.npy")
    pred_delta = np.load(root / "distill_pred_delta_raw_mlp.npy")
    actual_delta = np.load(root / "distill_actual_delta_raw.npy")

    matrices = {
        "corr_mlp_pred_next": absolute_pearson_correlation(inputs, pred_next),
        "corr_mlp_pred_delta": absolute_pearson_correlation(inputs, pred_delta),
        "corr_actual_delta": absolute_pearson_correlation(inputs, actual_delta),
    }
    np.save(out_dir / "attribution_corr_mlp_pred_next.npy", matrices["corr_mlp_pred_next"])
    np.save(out_dir / "attribution_corr_mlp_pred_delta.npy", matrices["corr_mlp_pred_delta"])
    np.save(out_dir / "attribution_corr_actual_delta.npy", matrices["corr_actual_delta"])

    ranking_payloads = {
        "corr_mlp_pred_next": write_rankings_json(
            out_dir / "attribution_corr_mlp_pred_next_rankings.json",
            matrices["corr_mlp_pred_next"],
            feature_columns,
            target_columns,
            sensor_idx,
            attribution_type="corr_mlp_pred_next",
        ),
        "corr_mlp_pred_delta": write_rankings_json(
            out_dir / "attribution_corr_mlp_pred_delta_rankings.json",
            matrices["corr_mlp_pred_delta"],
            feature_columns,
            target_columns,
            sensor_idx,
            attribution_type="corr_mlp_pred_delta",
        ),
        "corr_actual_delta": write_rankings_json(
            out_dir / "attribution_corr_actual_delta_rankings.json",
            matrices["corr_actual_delta"],
            feature_columns,
            target_columns,
            sensor_idx,
            attribution_type="corr_actual_delta",
        ),
    }
    write_topk_csv(out_dir / "attribution_corr_top10.csv", ranking_payloads)

    lit101 = _lit101_report(matrices["corr_mlp_pred_delta"], feature_columns, target_columns)
    metadata = {
        "distill_dir": str(root),
        "correlation_definition": "absolute Pearson correlation; constant feature/target columns map to 0",
        "array_shapes": {name: list(value.shape) for name, value in matrices.items()},
        "finite_checks": {name: bool(np.isfinite(value).all()) for name, value in matrices.items()},
        "lit101_report": lit101,
    }
    (out_dir / "attribution_corr_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    _print_top5("Top 5 correlation with MLP predicted next", matrices["corr_mlp_pred_next"], feature_columns, target_columns)
    _print_top5("Top 5 correlation with MLP predicted delta", matrices["corr_mlp_pred_delta"], feature_columns, target_columns)
    _print_top5("Top 5 correlation with actual delta", matrices["corr_actual_delta"], feature_columns, target_columns)
    if lit101.get("available"):
        print("\nLIT101 correlation with MLP predicted delta top 10:")
        for item in lit101["mlp_pred_delta_top10"]:
            print(f"  {item['rank']:2d}. {item['feature']}={item['value']:.6g}")
        print("LIT101 ranks:")
        print(json.dumps(lit101["ranks"], indent=2))
        print(f"FIT101 in MLP predicted delta corr top10: {lit101['FIT101_in_mlp_pred_delta_top10']}")
        print(f"FIT201 in MLP predicted delta corr top10: {lit101['FIT201_in_mlp_pred_delta_top10']}")
    print("\nSummary:")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
