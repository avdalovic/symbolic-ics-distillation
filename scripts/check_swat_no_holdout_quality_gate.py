#!/usr/bin/env python
"""Replay SWaT equation selection without the SWaT-only holdout quality gate.

This is a diagnostic/sensitivity check. It does not run PySR. It reads the
existing SWaT delta Pareto fronts and applies the same training-only stability
filter as the frozen SWaT run except for the extra holdout R2/MAE gate.

Kept filters:
- finite complexity <= max_complexity
- finite predictions on all training rows
- zero holdout CUSUM alarms when fit on the first 80% and run on the last 20%
- residual p99 / median <= residual_tail_ratio
- highest PySR score among candidates that pass

Outputs are written to artifacts/swat_1sec/no_holdout_quality_gate_check by
default and do not modify frozen delta_full artifacts.
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ics_symbolic_distill.detection import evaluate_equation, fit_cusum_params, run_cusum


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


DIAG = load_module(REPO_ROOT / "scripts" / "run_swat_1sec_delta_local_diagnostic.py", "swat_delta_diag_for_no_gate")
POST = load_module(REPO_ROOT / "scripts" / "run_swat_1sec_delta_posthoc_ablation.py", "swat_delta_posthoc_for_no_gate")


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
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    return str(value)


def write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def safe_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def candidate_indices_by_score(df: pd.DataFrame) -> list[int]:
    table = df.copy()
    table["complexity_num"] = pd.to_numeric(table.get("complexity"), errors="coerce").fillna(np.inf)
    table["loss_num"] = pd.to_numeric(table.get("loss"), errors="coerce").fillna(np.inf)
    table["score_num"] = pd.to_numeric(table.get("score"), errors="coerce").fillna(-np.inf)
    return table.sort_values(["score_num", "loss_num", "complexity_num"], ascending=[False, True, True]).index.astype(int).tolist()


def pareto_csv(source_root: Path, target: str) -> Path:
    return source_root / "pareto_fronts" / f"{target}_sensors_delta_actuators_next" / "pareto_front_scored.csv"


def variable_model_from_row(row: dict[str, Any], *, source: str) -> Any:
    return POST.VariableModel(
        target=str(row["target"]),
        variable_type=str(row.get("variable_type", "sensor")),
        equation=str(row.get("equation", "")),
        sympy_format=str(row.get("sympy_format", row.get("equation", ""))),
        complexity=safe_float(row.get("complexity")),
        loss=safe_float(row.get("loss")),
        score=safe_float(row.get("score")),
        holdout_r2=safe_float(row.get("holdout_r2")),
        holdout_mae=safe_float(row.get("holdout_mae")),
        baseline_holdout_mae=safe_float(row.get("baseline_holdout_mae")),
        residual_tail_ratio=safe_float(row.get("residual_tail_ratio")),
        source=source,
    )


def select_target_without_holdout_quality_gate(
    arrays: dict[str, Any],
    source_root: Path,
    target: str,
    *,
    max_complexity: int,
    default_s: float,
    default_g: float,
    residual_tail_ratio: float,
    tail_median_floor: float,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, np.ndarray] | None]:
    csv_path = pareto_csv(source_root, target)
    if not csv_path.exists():
        return None, {"target": target, "reason": "missing_pareto_front"}, None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None, {"target": target, "reason": "empty_pareto_front"}, None

    fit_idx = arrays["fit_idx"]
    holdout_idx = arrays["holdout_idx"]
    _, y_all = DIAG.target_values(arrays, target, split="train", target_mode="sensors_delta_actuators_next")
    baseline_holdout_mae = float(np.mean(np.abs(y_all[holdout_idx])))
    reasons: list[str] = []
    passing: list[tuple[float, float, float, int, dict[str, Any], dict[str, np.ndarray]]] = []

    for idx in candidate_indices_by_score(df):
        row = df.loc[idx]
        complexity = safe_float(row.get("complexity"))
        if not np.isfinite(complexity) or complexity > float(max_complexity):
            reasons.append(f"row={idx}:complexity>{max_complexity}")
            continue

        equation = str(row.get("sympy_format", row.get("equation", "")))
        pred_train_target = evaluate_equation(equation, arrays["feature_columns"], arrays["train_current"])
        if not np.isfinite(pred_train_target).all():
            reasons.append(f"row={idx}:nonfinite_train_predictions")
            continue

        residual_train = DIAG.detection_residual(arrays, target, equation, split="train", target_mode="sensors_delta_actuators_next")
        residual_fit = residual_train[fit_idx]
        residual_holdout = residual_train[holdout_idx]

        params = fit_cusum_params(residual_fit, s=float(default_s), g=float(default_g))
        _, holdout_alarm = run_cusum(residual_holdout, params)
        if int(np.sum(holdout_alarm)) > 0:
            reasons.append(f"row={idx}:holdout_cusum_alarm")
            continue

        median = float(np.median(residual_train))
        p99 = float(np.percentile(residual_train, 99))
        tail_ratio = p99 / max(median, float(tail_median_floor))
        if tail_ratio > float(residual_tail_ratio):
            reasons.append(f"row={idx}:tail_ratio>{residual_tail_ratio}:{tail_ratio:.4g}")
            continue

        holdout_metrics = DIAG.regression_metrics(y_all[holdout_idx], pred_train_target[holdout_idx])
        residual_test = DIAG.detection_residual(arrays, target, equation, split="test", target_mode="sensors_delta_actuators_next")
        selected = {
            "config": "FULL_no_holdout_quality_gate",
            "target": target,
            "variable_type": "sensor",
            "target_mode": "sensors_delta_actuators_next",
            "equation": str(row.get("equation", "")),
            "sympy_format": equation,
            "complexity": complexity,
            "loss": safe_float(row.get("loss")),
            "score": safe_float(row.get("score")),
            "holdout_r2": holdout_metrics["r2"],
            "holdout_mae": holdout_metrics["mae"],
            "baseline_holdout_mae": baseline_holdout_mae,
            "residual_tail_ratio": tail_ratio,
            "residual_p99": p99,
            "residual_median": median,
            "selection_reason": "highest_score_among_stable_candidates_no_holdout_quality_gate",
            "pareto_csv": str(csv_path),
            "removed_filter": "holdout_r2_or_mae_improvement_gate",
        }
        score = safe_float(row.get("score"))
        loss = safe_float(row.get("loss"))
        passing.append((score, -loss if np.isfinite(loss) else -np.inf, -complexity, int(idx), selected, {"train": residual_train, "test": residual_test}))

    if not passing:
        return None, {"target": target, "reason": "; ".join(reasons[-8:])}, None
    passing.sort(reverse=True, key=lambda item: item[:4])
    return passing[0][4], None, passing[0][5]


def model_rows_to_df(models: list[Any]) -> pd.DataFrame:
    return pd.DataFrame([model.to_row() for model in models])


def summarize_grid(grid: pd.DataFrame, variant: str) -> list[dict[str, Any]]:
    rows = []
    choices = POST.choose_rows(grid)
    for choice_name in ["geco_s_g", "best_f1_fpa_le_5", "best_f1_fpa_le_15", "best_etaf1_fpa_le_15", "lowest_fpa", "best_f1_overall"]:
        row = choices.get(choice_name)
        if row is None:
            continue
        rows.append({"variant": variant, "selection": choice_name, **row.to_dict()})
    return rows


def markdown_table(df: pd.DataFrame, columns: list[str]) -> str:
    if df.empty:
        return "(no rows)"
    view = df[columns].copy()
    for col in view.columns:
        if pd.api.types.is_float_dtype(view[col]):
            view[col] = view[col].map(lambda x: f"{x:.3f}")
    lines = ["| " + " | ".join(map(str, view.columns)) + " |", "| " + " | ".join(["---"] * len(view.columns)) + " |"]
    for row in view.astype(str).itertuples(index=False, name=None):
        lines.append("| " + " | ".join(row) + " |")
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Check SWaT selection without the holdout R2/MAE quality gate.")
    parser.add_argument("--source", default="artifacts/swat_1sec/delta_full", help="Frozen SWaT delta_full artifact directory.")
    parser.add_argument("--out", default="artifacts/swat_1sec/no_holdout_quality_gate_check")
    parser.add_argument("--experiment", default="configs/experiment/swat_mlp_current_val20.yaml")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--max-complexity", type=int, default=15)
    parser.add_argument("--default-s", type=float, default=1.42)
    parser.add_argument("--default-g", type=float, default=5.98)
    parser.add_argument("--residual-tail-ratio", type=float, default=50.0)
    parser.add_argument("--tail-median-floor", type=float, default=1e-9)
    parser.add_argument("--s-values", default="1.0,1.42,2.0,3.0,5.0,8.0,10.0")
    parser.add_argument("--g-values", default="2.0,5.98,9.0,15.0,25.0")
    return parser.parse_args()


def parse_float_list(text: str) -> list[float]:
    return [float(tok.strip()) for tok in str(text).split(",") if tok.strip()]


def main() -> int:
    args = parse_args()
    source_root = Path(args.source)
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)

    load_args = argparse.Namespace(experiment=args.experiment, train_csv=args.train_csv, test_csv=args.test_csv)
    arrays = DIAG.load_arrays(load_args)
    targets = list(DIAG.SWAT_SENSOR_TARGETS)

    selected_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    residual_cache: dict[str, dict[str, np.ndarray]] = {}
    for target in targets:
        selected, exclusion, residuals = select_target_without_holdout_quality_gate(
            arrays,
            source_root,
            target,
            max_complexity=int(args.max_complexity),
            default_s=float(args.default_s),
            default_g=float(args.default_g),
            residual_tail_ratio=float(args.residual_tail_ratio),
            tail_median_floor=float(args.tail_median_floor),
        )
        if selected is None:
            if exclusion is not None:
                exclusion_rows.append(exclusion)
            continue
        selected_rows.append(selected)
        residual_cache[str(selected["target"])] = residuals or {}

    selected_df = pd.DataFrame(selected_rows)
    exclusion_df = pd.DataFrame(exclusion_rows)
    selected_df.to_csv(out_root / "selected_equations_no_holdout_quality_gate.csv", index=False)
    exclusion_df.to_csv(out_root / "exclusion_reasons_no_holdout_quality_gate.csv", index=False)

    selected_models = [variable_model_from_row(row, source="no_holdout_quality_gate") for row in selected_rows]
    geco_exclusions = set(POST.GECO_EXCLUSIONS)
    geco_models = [model for model in selected_models if model.target not in geco_exclusions]
    actuator_models = POST.make_actuator_persistence_models(arrays["feature_columns"], exclude=geco_exclusions)

    variants = {
        "sensors_only_no_holdout_quality_gate_all": selected_models,
        "geco_matched_sensors_only_no_holdout_quality_gate": geco_models,
        "geco_matched_plus_all_actuator_persistence_no_holdout_quality_gate": geco_models + actuator_models,
    }

    s_values = parse_float_list(args.s_values)
    g_values = parse_float_list(args.g_values)
    summary_rows: list[dict[str, Any]] = []
    for variant, models in variants.items():
        grid, per_sensor, alarm_cache, cache = POST.evaluate_variant(
            arrays,
            models,
            variant=variant,
            s_values=s_values,
            g_values=g_values,
        )
        grid.to_csv(out_root / f"grid_{variant}.csv", index=False)
        per_sensor.to_csv(out_root / f"per_sensor_{variant}.csv", index=False)
        model_rows_to_df(models).to_csv(out_root / f"monitored_models_{variant}.csv", index=False)
        for row in summarize_grid(grid, variant):
            summary_rows.append(row)

        choices = POST.choose_rows(grid)
        choice = choices.get("best_f1_fpa_le_15")
        if choice is None:
            choice = choices.get("lowest_fpa")
        if choice is not None:
            alarms = POST.alarm_map_for_choice(models, cache, arrays["labels"], s=float(choice["S"]), g=float(choice["G"]))
            attack_table = POST.per_attack_table(models, alarms)
            attack_table.to_csv(out_root / f"per_attack_{variant}.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_root / "summary_no_holdout_quality_gate.csv", index=False)

    original_selected = pd.read_csv(source_root / "selected_equations.csv")
    original_targets = set(original_selected["target"].astype(str))
    new_targets = set(selected_df["target"].astype(str)) if not selected_df.empty else set()
    added = sorted(new_targets - original_targets)
    removed = sorted(original_targets - new_targets)

    comparison = {
        "source_artifact": str(source_root),
        "out": str(out_root),
        "targets_attempted": len(targets),
        "original_selected_count": len(original_selected),
        "no_holdout_gate_selected_count": len(selected_df),
        "added_targets_without_gate": added,
        "removed_targets_without_gate": removed,
        "kept_filters": [
            "complexity <= max_complexity",
            "finite predictions on all training rows",
            "zero holdout CUSUM alarms at S/G default",
            "residual p99/median tail ratio <= threshold",
            "highest PySR score among passing candidates",
        ],
        "removed_filter": "holdout R2 >= 0 or holdout MAE improves over baseline",
    }
    write_json(out_root / "run_config.json", {**comparison, "args": vars(args)})

    lines = [
        "# SWaT No-Holdout-Quality-Gate Diagnostic",
        "",
        "This diagnostic replays the existing SWaT 1-second delta Pareto fronts without rerunning PySR.",
        "It removes only the SWaT-specific holdout R2/MAE quality gate and keeps finite-prediction, holdout-CUSUM, and residual-tail filters.",
        "",
        "## Selection Impact",
        "",
        f"- Original selected sensors: `{len(original_selected)} / {len(targets)}`",
        f"- Selected without holdout quality gate: `{len(selected_df)} / {len(targets)}`",
        f"- Newly admitted targets: `{', '.join(added) if added else 'none'}`",
        f"- Lost targets: `{', '.join(removed) if removed else 'none'}`",
        "",
        "## Summary Rows",
        "",
        markdown_table(
            summary,
            ["variant", "selection", "S", "G", "num_monitored", "monitored_sensors", "monitored_actuators", "Precision", "Recall", "F1", "eTaF1", "FPA", "Scen"],
        ),
        "",
        "## Outputs",
        "",
        "- `selected_equations_no_holdout_quality_gate.csv`",
        "- `exclusion_reasons_no_holdout_quality_gate.csv`",
        "- `grid_*.csv`",
        "- `per_sensor_*.csv`",
        "- `per_attack_*.csv`",
        "- `summary_no_holdout_quality_gate.csv`",
    ]
    (out_root / "summary.md").write_text("\n".join(lines) + "\n", encoding="utf-8")

    print("=== SWaT no-holdout-quality-gate diagnostic ===")
    print(f"Original selected: {len(original_selected)} / {len(targets)}")
    print(f"No-gate selected: {len(selected_df)} / {len(targets)}")
    print(f"Newly admitted: {', '.join(added) if added else 'none'}")
    print(f"Outputs: {out_root}")
    if not summary.empty:
        show = summary[["variant", "selection", "S", "G", "num_monitored", "monitored_sensors", "monitored_actuators", "F1", "eTaF1", "FPA", "Scen"]]
        print(show.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
