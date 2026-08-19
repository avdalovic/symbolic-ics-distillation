#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


BAT = import_module("batadal_full_for_selection_ablation", REPO_ROOT / "scripts" / "run_batadal_delta_full.py")
V2 = import_module("paper_v2_for_batadal_selection_ablation", REPO_ROOT / "scripts" / "generate_paper_artifacts_v2.py")


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def matrix_from_cache(rows: list[dict[str, Any]], cache: dict[str, dict[str, np.ndarray]], split: str) -> np.ndarray:
    cols = [np.asarray(cache[str(row["target"])][split], dtype=np.float32) for row in rows]
    return np.column_stack(cols).astype(np.float32, copy=False)


def passing_candidates(arrays: dict[str, Any], pareto_root: Path, target: str, max_complexity: int) -> list[dict[str, Any]]:
    csv_path = BAT.pareto_dir(pareto_root, target) / "pareto_front_scored.csv"
    if not csv_path.exists():
        return []
    df = pd.read_csv(csv_path)
    if df.empty:
        return []
    fit_idx = arrays["fit_idx"]
    holdout_idx = arrays["holdout_idx"]
    _, y_all = BAT.target_values(arrays, target, split="train")
    baseline_holdout_mae = float(np.mean(np.abs(y_all[holdout_idx])))
    passing: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        complexity = BAT.safe_float(row.get("complexity"))
        if not np.isfinite(complexity) or complexity > float(max_complexity):
            continue
        equation = str(row.get("sympy_format", row.get("equation", "")))
        pred_train_target = BAT.evaluate_equation(equation, arrays["feature_columns"], arrays["train_current"])
        if not np.isfinite(pred_train_target).all():
            continue
        residual_train = BAT.prediction_residual(arrays, target, equation, split="train")
        residual_fit = residual_train[fit_idx]
        residual_holdout = residual_train[holdout_idx]
        params = BAT.fit_cusum_params(residual_fit, s=BAT.GECO_S, g=BAT.GECO_G)
        _, holdout_alarm = BAT.run_cusum(residual_holdout, params)
        if int(np.sum(holdout_alarm)) > 0:
            continue
        median = float(np.median(residual_train))
        p99 = float(np.percentile(residual_train, 99))
        tail_ratio = p99 / max(median, 1e-9)
        if tail_ratio > 50.0:
            continue
        holdout_metrics = BAT.regression_metrics(y_all[holdout_idx], pred_train_target[holdout_idx])
        passing.append(
            {
                "target": target,
                "variable_type": "sensor",
                "target_mode": BAT.TARGET_MODE,
                "equation": str(row.get("equation", "")),
                "sympy_format": equation,
                "complexity": complexity,
                "loss": BAT.safe_float(row.get("loss")),
                "score": BAT.safe_float(row.get("score")),
                "holdout_r2": holdout_metrics["r2"],
                "holdout_mae": holdout_metrics["mae"],
                "baseline_holdout_mae": baseline_holdout_mae,
                "residual_tail_ratio": tail_ratio,
                "residual_p99": p99,
                "residual_median": median,
                "pareto_row": int(idx),
                "pareto_csv": str(csv_path),
            }
        )
    return passing


def choose_candidate(strategy: str, candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    if strategy == "score":
        key = lambda r: (r["score"], -r["loss"] if np.isfinite(r["loss"]) else -np.inf, -r["complexity"])
    elif strategy == "holdout_r2":
        key = lambda r: (r["holdout_r2"], r["score"], -r["complexity"])
    elif strategy == "holdout_mae":
        key = lambda r: (-r["holdout_mae"], r["score"], -r["complexity"])
    elif strategy == "r2_parsimony_0p01":
        best = max(float(r["holdout_r2"]) for r in candidates)
        near = [r for r in candidates if float(r["holdout_r2"]) >= best - 0.01]
        return max(near, key=lambda r: (-r["complexity"], r["holdout_r2"], r["score"]))
    elif strategy == "r2_parsimony_0p05":
        best = max(float(r["holdout_r2"]) for r in candidates)
        near = [r for r in candidates if float(r["holdout_r2"]) >= best - 0.05]
        return max(near, key=lambda r: (-r["complexity"], r["holdout_r2"], r["score"]))
    else:
        raise ValueError(strategy)
    return max(candidates, key=key)


def build_payload(
    selected_rows: list[dict[str, Any]],
    train_csv: str,
    test_csvs: list[str],
) -> tuple[dict[str, Any], list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, np.ndarray]], str]]]:
    segments = []
    for test_csv in test_csvs:
        arrays = BAT.load_batadal_arrays(argparse.Namespace(train_csv=train_csv, test_csv=test_csv))
        rows, cache = BAT.build_variant_rows(arrays, selected_rows, "geco_matched_plus_actuator_persistence")
        segments.append((arrays, rows, cache, test_csv))
    arrays0, rows0, cache0, _ = segments[0]
    order = [str(row["target"]) for row in rows0]
    for _, rows, _, test_csv in segments[1:]:
        if [str(row["target"]) for row in rows] != order:
            raise RuntimeError(f"Row order mismatch in {test_csv}")
    payload = {
        "variant": "combined_14_attacks_geco_matched_plus_actuator_persistence",
        "train_matrix": matrix_from_cache(rows0, cache0, "train"),
        "test_matrices": [matrix_from_cache(rows, cache, "test") for arrays, rows, cache, _ in segments],
        "label_arrays": [arrays["labels"] for arrays, rows, cache, _ in segments],
        "counts": {
            "num_monitored": len(rows0),
            "num_sensors": sum(1 for row in rows0 if str(row.get("variable_type")) == "sensor"),
            "num_actuators": sum(1 for row in rows0 if str(row.get("variable_type")) == "actuator"),
        },
    }
    return payload, segments


def per_attack_for_point(
    rows: list[dict[str, Any]],
    segments: list[tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, np.ndarray]], str]],
    s: float,
    g: float,
    point_name: str,
) -> pd.DataFrame:
    out = []
    for arrays, segment_rows, cache, test_csv in segments:
        alarm_map: dict[str, np.ndarray] = {}
        alarms = []
        for row in segment_rows:
            target = str(row["target"])
            residuals = cache[target]
            params = BAT.fit_cusum_params(residuals["train"], s=float(s), g=float(g))
            _, alarm = BAT.run_cusum(residuals["test"], params)
            alarm = alarm.astype(np.int8)
            alarm_map[target] = alarm
            alarms.append(alarm)
        alarm_map["system"] = np.max(np.stack(alarms, axis=1), axis=1).astype(np.int8) if alarms else np.zeros_like(arrays["labels"], dtype=np.int8)
        table = BAT.per_attack_table(segment_rows, alarm_map, arrays["attack_windows"])
        table.insert(0, "test_csv", test_csv)
        table.insert(1, "point_name", point_name)
        table.insert(2, "S", float(s))
        table.insert(3, "G", float(g))
        out.append(table)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def point_row(grid: pd.DataFrame, s: float, g: float, name: str) -> dict[str, Any]:
    sub = grid[(np.isclose(grid["S"], s)) & (np.isclose(grid["G"], g))]
    if sub.empty:
        raise RuntimeError(f"Missing grid row S={s} G={g}")
    row = sub.iloc[0].to_dict()
    row["point_name"] = name
    return row


def main() -> int:
    parser = argparse.ArgumentParser(description="BATADAL benign-only equation-selection ablation.")
    parser.add_argument("--pareto-root", default="results/batadal")
    parser.add_argument("--train-csv", default="data/batadal/processed/train.csv")
    parser.add_argument(
        "--test-csvs",
        default="data/batadal/processed/test_dataset04.csv,data/batadal/processed/test_dataset_test.csv",
    )
    parser.add_argument("--out-dir", default="artifacts/experiments/batadal_selection_ablation")
    parser.add_argument("--max-complexity", type=int, default=15)
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    pareto_root = Path(args.pareto_root)
    test_csvs = [tok.strip() for tok in str(args.test_csvs).split(",") if tok.strip()]
    arrays = BAT.load_batadal_arrays(argparse.Namespace(train_csv=args.train_csv, test_csv=test_csvs[0]))

    strategies = ["score", "holdout_r2", "holdout_mae", "r2_parsimony_0p01", "r2_parsimony_0p05"]
    candidate_audit = []
    selected_by_strategy: dict[str, list[dict[str, Any]]] = {name: [] for name in strategies}
    for target in arrays["sensor_names"]:
        candidates = passing_candidates(arrays, pareto_root, target, int(args.max_complexity))
        for cand in candidates:
            candidate_audit.append({**cand, "passed": True})
        for strategy in strategies:
            selected = choose_candidate(strategy, candidates)
            if selected is None:
                continue
            selected = dict(selected)
            selected["selection_reason"] = f"{strategy}_among_training_stable_candidates"
            selected_by_strategy[strategy].append(selected)

    pd.DataFrame(candidate_audit).to_csv(out_dir / "passing_candidates.csv", index=False)

    all_grid_rows = []
    all_summary_rows = []
    all_point_rows = []
    all_selected_rows = []
    all_per_attack_rows = []
    for strategy, selected_rows in selected_by_strategy.items():
        selected_df = pd.DataFrame(selected_rows)
        selected_df.to_csv(out_dir / f"selected_equations_{strategy}.csv", index=False)
        selected_df.assign(strategy=strategy).to_csv(out_dir / f"selected_equations_{strategy}.csv", index=False)
        all_selected_rows.extend(selected_df.assign(strategy=strategy).to_dict("records"))

        payload, segments = build_payload(selected_rows, args.train_csv, test_csvs)
        s_values = sorted({float(v) for v in V2.S_VALUES_EXT} | {1.39, 1.40})
        g_values = sorted({float(v) for v in V2.G_VALUES_EXT} | {2.0, 2.16})
        grid, per_sample_s = V2.evaluate_grid_from_residuals(
            train_matrix=payload["train_matrix"],
            test_matrices=payload["test_matrices"],
            label_arrays=payload["label_arrays"],
            s_values=s_values,
            g_values=g_values,
            expand_steps=1,
            counts=payload["counts"],
        )
        grid.insert(0, "strategy", strategy)
        grid.insert(1, "variant", payload["variant"])
        all_grid_rows.extend(grid.to_dict("records"))

        points = [
            point_row(grid, 1.40, 2.00, "headline_point"),
            point_row(grid, 1.39, 2.16, "geco_point"),
        ]
        fpa8 = grid[grid["FPA"] <= 8].sort_values(["eTaF1", "F1", "Scen"], ascending=False)
        if not fpa8.empty:
            r = fpa8.iloc[0].to_dict()
            r["point_name"] = "best_fpa_le_8"
            points.append(r)
        zero = grid[grid["FPA"] == 0].sort_values(["eTaF1", "F1", "Scen"], ascending=False)
        if not zero.empty:
            r = zero.iloc[0].to_dict()
            r["point_name"] = "best_zero_fpa"
            points.append(r)
        overall = grid.sort_values(["eTaF1", "F1", "Scen"], ascending=False).iloc[0].to_dict()
        overall["point_name"] = "best_overall"
        points.append(overall)
        for row in points:
            row["strategy"] = strategy
            row["variant"] = payload["variant"]
            all_point_rows.append(row)
            all_summary_rows.append(row)
            per_attack = per_attack_for_point(selected_rows, segments, float(row["S"]), float(row["G"]), f"{strategy}:{row['point_name']}")
            per_attack.insert(0, "strategy", strategy)
            all_per_attack_rows.extend(per_attack.to_dict("records"))

        write_json(
            out_dir / f"run_meta_{strategy}.json",
            {
                "strategy": strategy,
                "selection_uses_attack_labels": False,
                "selection_data": "benign training fit/holdout only",
                "stable_candidate_criteria": [
                    "finite train predictions",
                    "zero held-out benign CUSUM alarms at GeCo BATADAL S/G",
                    "residual p99/median tail ratio <= 50",
                ],
                "num_selected_sensor_equations": len(selected_rows),
                "num_monitored_after_geco_exclusion_and_persistence": payload["counts"],
                "grid_rows": int(len(grid)),
                "per_grid_cell_seconds": float(per_sample_s),
            },
        )

    pd.DataFrame(all_selected_rows).to_csv(out_dir / "selected_equations_by_strategy.csv", index=False)
    pd.DataFrame(all_grid_rows).to_csv(out_dir / "detection_grid.csv", index=False)
    pd.DataFrame(all_point_rows).to_csv(out_dir / "operating_points.csv", index=False)
    pd.DataFrame(all_summary_rows).to_csv(out_dir / "summary_rows.csv", index=False)
    pd.DataFrame(all_per_attack_rows).to_csv(out_dir / "per_attack.csv", index=False)

    summary = pd.DataFrame(all_summary_rows)
    cols = ["strategy", "point_name", "S", "G", "Precision", "Recall", "F1", "eTaP", "eTaR", "eTaF1", "FPA", "Scen"]
    print(summary[[c for c in cols if c in summary.columns]].sort_values(["point_name", "eTaF1"], ascending=[True, False]).to_string(index=False))
    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
