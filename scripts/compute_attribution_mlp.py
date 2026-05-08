#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ics_symbolic_distill.attribution import (
    compute_gradient_attribution,
    load_aligned_mlp_inputs,
    load_mlp_model,
    rank_of_feature,
    write_rankings_json,
    write_topk_csv,
)
from ics_symbolic_distill.utils import get_device


REPORT_TARGETS = ["LIT101", "FIT101", "FIT201", "LIT301", "DPIT301", "AIT201"]


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


def _lit101_report(
    *,
    raw_next: np.ndarray,
    raw_delta: np.ndarray,
    feature_columns: list[str],
    target_columns: list[str],
) -> dict:
    target = "LIT101"
    if target not in target_columns:
        return {"target": target, "available": False}
    target_idx = target_columns.index(target)

    def top10(matrix: np.ndarray) -> list[dict]:
        order = np.argsort(-matrix[target_idx], kind="mergesort")[:10]
        return [
            {
                "rank": int(rank),
                "feature": feature_columns[int(feature_idx)],
                "feature_index": int(feature_idx),
                "value": float(matrix[target_idx, int(feature_idx)]),
            }
            for rank, feature_idx in enumerate(order, start=1)
        ]

    report = {
        "target": target,
        "available": True,
        "raw_next_top10": top10(raw_next),
        "raw_delta_top10": top10(raw_delta),
        "ranks": {},
    }
    for feature in ["LIT101", "FIT101", "FIT201"]:
        report["ranks"][feature] = {
            "raw_next": rank_of_feature(raw_next, target, feature, feature_columns, target_columns),
            "raw_delta": rank_of_feature(raw_delta, target, feature, feature_columns, target_columns),
        }
    report["FIT101_in_raw_delta_top10"] = report["ranks"]["FIT101"]["raw_delta"] is not None and report["ranks"]["FIT101"]["raw_delta"] <= 10
    report["FIT201_in_raw_delta_top10"] = report["ranks"]["FIT201"]["raw_delta"] is not None and report["ranks"]["FIT201"]["raw_delta"] <= 10
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compute current-state MLP input attribution.")
    parser.add_argument("--mlp-export", required=True)
    parser.add_argument("--distill-dir", required=True)
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()

    device = get_device(args.device) if args.device else None
    model, state, _ = load_mlp_model(
        checkpoint=args.checkpoint,
        config=args.config,
        device=device,
    )
    data = load_aligned_mlp_inputs(
        mlp_export=args.mlp_export,
        distill_dir=args.distill_dir,
        checkpoint=args.checkpoint,
    )
    attribution = compute_gradient_attribution(
        model,
        data.inputs_norm_aligned,
        data.normalization_stats,
        data.sensor_idx,
        sample_size=args.sample_size,
        seed=args.seed,
        batch_size=args.batch_size,
        device=next(model.parameters()).device,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    matrices = {
        "grad_norm_next": attribution["grad_norm_next"],
        "sensitivity_raw_next": attribution["sensitivity_raw_next"],
        "sensitivity_raw_delta": attribution["sensitivity_raw_delta"],
    }
    np.save(out_dir / "attribution_mlp_grad_norm_next.npy", matrices["grad_norm_next"])
    np.save(out_dir / "attribution_mlp_sensitivity_raw_next.npy", matrices["sensitivity_raw_next"])
    np.save(out_dir / "attribution_mlp_sensitivity_raw_delta.npy", matrices["sensitivity_raw_delta"])
    np.save(out_dir / "attribution_mlp_sample_indices.npy", attribution["sample_indices"].astype(np.int64))

    ranking_payloads = {
        "grad_norm_next": write_rankings_json(
            out_dir / "attribution_mlp_grad_norm_next_rankings.json",
            matrices["grad_norm_next"],
            data.feature_columns,
            data.target_columns,
            data.sensor_idx,
            attribution_type="grad_norm_next",
        ),
        "sensitivity_raw_next": write_rankings_json(
            out_dir / "attribution_mlp_sensitivity_raw_next_rankings.json",
            matrices["sensitivity_raw_next"],
            data.feature_columns,
            data.target_columns,
            data.sensor_idx,
            attribution_type="sensitivity_raw_next",
        ),
        "sensitivity_raw_delta": write_rankings_json(
            out_dir / "attribution_mlp_sensitivity_raw_delta_rankings.json",
            matrices["sensitivity_raw_delta"],
            data.feature_columns,
            data.target_columns,
            data.sensor_idx,
            attribution_type="sensitivity_raw_delta",
        ),
    }
    write_topk_csv(out_dir / "attribution_mlp_top10.csv", ranking_payloads)

    lit101 = _lit101_report(
        raw_next=matrices["sensitivity_raw_next"],
        raw_delta=matrices["sensitivity_raw_delta"],
        feature_columns=data.feature_columns,
        target_columns=data.target_columns,
    )
    metadata = {
        "mlp_export": str(Path(args.mlp_export)),
        "distill_dir": str(Path(args.distill_dir)),
        "checkpoint": str(Path(args.checkpoint)),
        "config": str(Path(args.config)),
        "checkpoint_epoch": int(state.get("epoch", -1)),
        "checkpoint_best_val": float(state.get("best_val", float("nan"))),
        "input_space": "normalized MLP export inputs",
        "gradient_definition": "mean_n(abs(d y_norm_j[n] / d x_norm_i[n]))",
        "raw_sensitivity_definition": "mean_n(abs(d y_raw_j[n] / d x_raw_i[n])) using saved safe std",
        "delta_sensitivity_definition": "mean_n(abs(d(pred_next_raw_j - current_raw_sensor_j)/d x_raw_i[n]))",
        "sample_size_requested": int(args.sample_size),
        "sample_size_used": int(attribution["num_samples"]),
        "seed": int(args.seed),
        "batch_size": int(args.batch_size),
        "backend": attribution["backend"],
        "alignment_start": int(data.alignment_start),
        "alignment_max_abs_raw": float(data.alignment_max_abs_raw),
        "array_shapes": {name: list(value.shape) for name, value in matrices.items()},
        "lit101_report": lit101,
    }
    (out_dir / "attribution_mlp_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")

    _print_top5("Top 5 raw next-value sensitivities", matrices["sensitivity_raw_next"], data.feature_columns, data.target_columns)
    _print_top5("Top 5 raw delta sensitivities", matrices["sensitivity_raw_delta"], data.feature_columns, data.target_columns)
    if lit101.get("available"):
        print("\nLIT101 raw next sensitivity top 10:")
        for item in lit101["raw_next_top10"]:
            print(f"  {item['rank']:2d}. {item['feature']}={item['value']:.6g}")
        print("LIT101 raw delta sensitivity top 10:")
        for item in lit101["raw_delta_top10"]:
            print(f"  {item['rank']:2d}. {item['feature']}={item['value']:.6g}")
        print("LIT101 ranks:")
        print(json.dumps(lit101["ranks"], indent=2))
        print(f"FIT101 in raw delta top10: {lit101['FIT101_in_raw_delta_top10']}")
        print(f"FIT201 in raw delta top10: {lit101['FIT201_in_raw_delta_top10']}")
    print("\nSummary:")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
