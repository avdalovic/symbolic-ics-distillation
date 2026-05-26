#!/usr/bin/env python
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

from ics_symbolic_distill.data.ics_metadata import get_attack_windows, is_actuator
from ics_symbolic_distill.detection import compute_detection_metrics, evaluate_equation, fit_cusum_params, run_cusum
from ics_symbolic_distill.detection.metrics import to_intervals
from ics_symbolic_distill.detection.swat1s_delta_sampling import reconstruct_next_from_delta


WADI_GECO_EXCLUSIONS = {
    "2_P_001_STATUS",
    "2_P_002_STATUS",
    "2_LS_002_AL",
    "2_LS_001_AL",
    "2B_AIT_002_PV",
    "2_PIC_003_SP",
    "1_MV_002_STATUS",
    "1_MV_003_STATUS",
}
S_VALUES = [1.0, 1.32, 2.0, 3.0, 5.0]
G_VALUES = [2.0, 5.98, 9.74, 15.0, 25.0]
GECO_S = 1.32
GECO_G = 9.74


def _load_wadi_full_module():
    path = REPO_ROOT / "scripts" / "run_wadi_1sec_delta_full.py"
    spec = importlib.util.spec_from_file_location("run_wadi_1sec_delta_full", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


FULL = _load_wadi_full_module()


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


def normalize_name(name: str) -> str:
    return "".join(ch for ch in str(name).upper() if ch.isalnum())


def resolve_exclusions(requested: set[str], feature_columns: list[str]) -> tuple[set[str], list[str]]:
    """Resolve published WADI exclusion names against dataset-specific column spelling."""

    exact = set(feature_columns).intersection(requested)
    by_norm = {normalize_name(col): col for col in feature_columns}
    resolved = set(exact)
    missing = []
    for name in sorted(requested - exact):
        match = by_norm.get(normalize_name(name))
        if match is None:
            missing.append(name)
        else:
            resolved.add(match)
    return resolved, missing


def load_selected_equations(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(f"Selected equation CSV not found: {path}")
    df = pd.read_csv(path)
    rows = []
    for row in df.to_dict("records"):
        target = str(row["target"])
        rows.append(
            {
                "target": target,
                "variable_type": str(row.get("variable_type", "sensor")),
                "target_mode": str(row.get("target_mode", "sensors_delta_actuators_next")),
                "equation": str(row.get("equation", "")),
                "sympy_format": str(row.get("sympy_format", row.get("equation", ""))),
                "complexity": FULL.safe_float(row.get("complexity")),
                "loss": FULL.safe_float(row.get("loss")),
                "score": FULL.safe_float(row.get("score")),
                "holdout_r2": FULL.safe_float(row.get("holdout_r2")),
                "residual_tail_ratio": FULL.safe_float(row.get("residual_tail_ratio")),
                "source": "selected_sensor_delta",
            }
        )
    return rows


def sensor_residual(arrays: dict[str, Any], row: dict[str, Any], split: str) -> np.ndarray:
    target = str(row["target"])
    idx = arrays["feature_columns"].index(target)
    if split == "train":
        current = arrays["train_current"]
        nxt = arrays["train_next"]
    else:
        current = arrays["test_current"]
        nxt = arrays["test_next"]
    equation_safe = FULL.equation_original_to_safe(str(row["sympy_format"]), arrays["original_to_safe"])
    pred_delta = evaluate_equation(equation_safe, arrays["safe_feature_columns"], current).astype(np.float64)
    pred_next = reconstruct_next_from_delta(current[:, idx], pred_delta)
    residual = np.abs(nxt[:, idx].astype(np.float64) - pred_next)
    return np.where(np.isfinite(residual), residual, 0.0).astype(np.float64)


def actuator_persistence_residual(arrays: dict[str, Any], target: str, split: str) -> np.ndarray:
    idx = arrays["feature_columns"].index(target)
    if split == "train":
        current = arrays["train_current"]
        nxt = arrays["train_next"]
    else:
        current = arrays["test_current"]
        nxt = arrays["test_next"]
    residual = np.abs(nxt[:, idx].astype(np.float64) - current[:, idx].astype(np.float64))
    return np.where(np.isfinite(residual), residual, 0.0).astype(np.float64)


def build_variant_rows(
    arrays: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    *,
    variant: str,
    geco_exclusions: set[str],
) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    if variant == "sensors_only_all":
        sensors = list(selected_rows)
        actuators: list[str] = []
    elif variant == "sensors_only_geco_matched":
        sensors = [row for row in selected_rows if str(row["target"]) not in geco_exclusions]
        actuators = []
    elif variant == "geco_matched_plus_actuator_persistence":
        sensors = [row for row in selected_rows if str(row["target"]) not in geco_exclusions]
        actuators = [name for name in arrays["actuator_names"] if name not in geco_exclusions]
    elif variant == "no_ait_family":
        sensors = [row for row in selected_rows if "AIT" not in str(row["target"]).upper()]
        actuators = []
    else:
        raise ValueError(f"Unknown variant: {variant}")

    rows: list[dict[str, Any]] = []
    cache: dict[str, dict[str, np.ndarray]] = {}
    for row in sensors:
        target = str(row["target"])
        rows.append(row)
        cache[target] = {
            "train": sensor_residual(arrays, row, "train"),
            "test": sensor_residual(arrays, row, "test"),
        }
    for target in actuators:
        row = {
            "target": target,
            "variable_type": "actuator",
            "target_mode": "actuator_persistence_next",
            "equation": target,
            "sympy_format": target,
            "complexity": 1.0,
            "loss": np.nan,
            "score": np.nan,
            "holdout_r2": np.nan,
            "residual_tail_ratio": np.nan,
            "source": "actuator_persistence",
        }
        rows.append(row)
        cache[target] = {
            "train": actuator_persistence_residual(arrays, target, "train"),
            "test": actuator_persistence_residual(arrays, target, "test"),
        }
    return rows, cache


def evaluate_variant(
    labels: np.ndarray,
    rows: list[dict[str, Any]],
    cache: dict[str, dict[str, np.ndarray]],
    variant: str,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[float, float], dict[str, np.ndarray]]]:
    grid_rows = []
    per_sensor_rows = []
    alarm_cache: dict[tuple[float, float], dict[str, np.ndarray]] = {}
    for s in S_VALUES:
        for g in G_VALUES:
            alarms = []
            alarm_map: dict[str, np.ndarray] = {}
            for row in rows:
                target = str(row["target"])
                residuals = cache[target]
                params = fit_cusum_params(residuals["train"], s=float(s), g=float(g))
                cusum, alarm = run_cusum(residuals["test"], params)
                alarm = alarm.astype(np.int8)
                alarm_map[target] = alarm
                alarms.append(alarm)
                burden = FULL.alarm_burden(alarm, labels)
                per_sensor_rows.append(
                    {
                        "variant": variant,
                        "target": target,
                        "variable_type": row.get("variable_type", ""),
                        "source": row.get("source", ""),
                        "S": s,
                        "G": g,
                        "complexity": row.get("complexity"),
                        "equation": row.get("equation"),
                        "holdout_r2": row.get("holdout_r2"),
                        "residual_tail_ratio": row.get("residual_tail_ratio"),
                        "delta": params.delta,
                        "threshold": params.threshold,
                        "growth_cap": params.growth_cap,
                        "max_train_cusum": params.max_calib_cusum,
                        "max_test_cusum": float(np.max(cusum)) if cusum.size else 0.0,
                        **burden,
                    }
                )
            system_alarm = np.max(np.stack(alarms, axis=1), axis=1).astype(np.int8) if alarms else np.zeros_like(labels, dtype=np.int8)
            alarm_map["system"] = system_alarm
            alarm_cache[(float(s), float(g))] = alarm_map
            metrics = compute_detection_metrics(labels, system_alarm, expand_steps=60)
            grid_rows.append(
                {
                    "variant": variant,
                    "S": s,
                    "G": g,
                    "num_monitored": len(rows),
                    "monitored_sensors": sum(str(row.get("variable_type")) == "sensor" for row in rows),
                    "monitored_actuators": sum(str(row.get("variable_type")) == "actuator" for row in rows),
                    "Precision": metrics["point_precision"],
                    "Recall": metrics["point_recall"],
                    "F1": metrics["point_f1"],
                    "eTaP": metrics["eTaP"],
                    "eTaR": metrics["eTaR"],
                    "eTaF1": metrics["eTaF1"],
                    "FPA": metrics["FPA"],
                    "Scen": metrics["scenario_detection_rate"],
                    "attack_interval_count": metrics.get("attack_interval_count", float("nan")),
                    **FULL.point_counts(labels, system_alarm),
                    **{f"system_{k}": v for k, v in FULL.alarm_burden(system_alarm, labels).items()},
                }
            )
    return pd.DataFrame(grid_rows), pd.DataFrame(per_sensor_rows), alarm_cache


def choose_summary_row(variant: str, grid: pd.DataFrame) -> tuple[str, pd.Series]:
    if variant == "sensors_only_geco_matched":
        eligible = grid[grid["FPA"] <= 5].sort_values(["F1", "eTaF1"], ascending=False)
        if not eligible.empty:
            return "best_f1_fpa_le_5", eligible.iloc[0]
    table = grid.sort_values(["F1", "eTaF1"], ascending=False)
    return "best_f1", table.iloc[0]


def first_alarm_delay(alarm: np.ndarray, start_idx: int, end_idx: int, original_start: int) -> int | None:
    segment = alarm[start_idx : end_idx + 1]
    hits = np.flatnonzero(segment)
    if hits.size == 0:
        return None
    return max(0, int(start_idx + int(hits[0]) + 1) - int(original_start))


def per_attack_table(rows: list[dict[str, Any]], alarm_map: dict[str, np.ndarray]) -> pd.DataFrame:
    monitored = {str(row["target"]) for row in rows}
    system = alarm_map.get("system")
    if system is None:
        n = int(next(iter(alarm_map.values())).shape[0]) if alarm_map else 0
        system = np.zeros(n, dtype=np.int8)
    n = int(system.shape[0])
    out = []
    for attack_id, window in enumerate(get_attack_windows("WADI"), start=1):
        start = max(0, int(window.start) - 1)
        end = min(n - 1, int(window.end) - 1)
        affected = [str(tag) for tag in window.affected_tags]
        in_scope = [tag for tag in affected if tag in monitored]
        firing = [
            target
            for target, alarm in alarm_map.items()
            if target != "system" and start <= end and np.any(alarm[start : end + 1] == 1)
        ]
        system_detected = bool(start <= end and np.any(system[start : end + 1] == 1))
        if in_scope and system_detected:
            category = "direct_detected"
        elif not in_scope and system_detected:
            category = "collateral_detected"
        elif not in_scope and not system_detected:
            category = "scope_miss"
        else:
            category = "detection_failure"
        delays = [
            first_alarm_delay(alarm_map[target], start, end, int(window.start))
            for target in firing
            if target in alarm_map
        ]
        delays = [delay for delay in delays if delay is not None]
        out.append(
            {
                "attack_id": attack_id,
                "start": int(window.start),
                "end": int(window.end),
                "affected_tags": ",".join(affected),
                "in_scope_variables": ",".join(in_scope),
                "category": category,
                "firing_variables": ",".join(firing),
                "detected": system_detected,
                "detection_delay_seconds": min(delays) if delays else None,
            }
        )
    return pd.DataFrame(out)


def miss_reason(affected_tags: str, monitored: set[str], detected: bool) -> str:
    if detected:
        return ""
    tags = [tag for tag in str(affected_tags).split(",") if tag]
    if any(tag in monitored for tag in tags):
        return "monitored_but_no_alarm"
    if any(tag in WADI_GECO_EXCLUSIONS for tag in tags):
        return "geco_excluded_variable"
    if any("AIT" in tag.upper() for tag in tags):
        return "ait_not_monitored"
    if any("STATUS" in tag.upper() for tag in tags):
        return "actuator_not_monitored"
    return "not_monitored"


def markdown_summary(summary: pd.DataFrame, coverage_gap: pd.DataFrame, top_fpa: dict[str, pd.DataFrame]) -> str:
    lines = [
        "# WADI 1-second delta posthoc ablation",
        "",
        "All rows reuse fitted WADI delta sensor equations. Actuator channels, when present, use persistence only. GeCo-matched exclusions are the published WADI exclusion list, resolved against local column names.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## GeCo reference",
        "",
        "GeCo published WADI: Prec 92.6, Rec 32.1, F1 47.7, eTaP 91.3, eTaR 56.3, eTaF1 69.7, FPA 0, Scen 78.6.",
        "",
        "## Top FPA contributors",
    ]
    for variant, table in top_fpa.items():
        lines.extend(["", f"### {variant}", "", table.to_markdown(index=False)])
    lines.extend(["", "## Coverage gap", "", coverage_gap.to_markdown(index=False)])
    return "\n".join(lines) + "\n"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WADI 1-second delta posthoc ablations.")
    parser.add_argument("--train-csv", default="data/wadi/raw/wadi_train.csv")
    parser.add_argument("--test-csv", default="data/wadi/raw/wadi_test.csv")
    parser.add_argument("--selected-equations", default="artifacts/wadi_1sec/delta_full/selected_equations.csv")
    parser.add_argument("--pareto-dir", default="artifacts/wadi_1sec/delta_full/pareto_fronts")
    parser.add_argument("--out", default="artifacts/wadi_1sec/delta_posthoc_ablation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    arrays = FULL.load_wadi_1sec_arrays(args)
    selected_rows = load_selected_equations(Path(args.selected_equations))
    geco_exclusions, missing_geco = resolve_exclusions(WADI_GECO_EXCLUSIONS, arrays["feature_columns"])
    write_json(
        out_root / "run_config.json",
        {
            "dataset": "WADI",
            "train_csv": args.train_csv,
            "test_csv": args.test_csv,
            "selected_equations": args.selected_equations,
            "pareto_dir": args.pareto_dir,
            "variants": [
                "sensors_only_all",
                "sensors_only_geco_matched",
                "geco_matched_plus_actuator_persistence",
                "no_ait_family",
            ],
            "s_values": S_VALUES,
            "g_values": G_VALUES,
            "geco_exclusions_requested": sorted(WADI_GECO_EXCLUSIONS),
            "geco_exclusions_resolved": sorted(geco_exclusions),
            "geco_exclusions_missing": missing_geco,
            "data": arrays["metadata"],
        },
    )

    variant_names = [
        "sensors_only_all",
        "sensors_only_geco_matched",
        "geco_matched_plus_actuator_persistence",
        "no_ait_family",
    ]
    summary_rows = []
    per_attack_by_variant: dict[str, pd.DataFrame] = {}
    monitored_by_variant: dict[str, set[str]] = {}
    top_fpa: dict[str, pd.DataFrame] = {}

    for variant in variant_names:
        print(f"[WADI posthoc] evaluating {variant}", flush=True)
        rows, residual_cache = build_variant_rows(arrays, selected_rows, variant=variant, geco_exclusions=geco_exclusions)
        grid, per_sensor, alarm_cache = evaluate_variant(arrays["labels"], rows, residual_cache, variant)
        grid.to_csv(out_root / f"grid_{variant}.csv", index=False)
        per_sensor.to_csv(out_root / f"per_sensor_{variant}.csv", index=False)
        selection_name, selected_grid_row = choose_summary_row(variant, grid)
        key = (float(selected_grid_row["S"]), float(selected_grid_row["G"]))
        attack = per_attack_table(rows, alarm_cache[key])
        attack.to_csv(out_root / f"per_attack_{variant}.csv", index=False)
        per_attack_by_variant[variant] = attack
        monitored_by_variant[variant] = {str(row["target"]) for row in rows}
        selected_per_sensor = per_sensor[(per_sensor["S"] == key[0]) & (per_sensor["G"] == key[1])]
        top_fpa[variant] = selected_per_sensor.sort_values(
            ["num_alarm_intervals", "benign_alarm_rate"], ascending=False
        )[["target", "variable_type", "num_alarm_intervals", "benign_alarm_rate", "total_alarm_rate", "equation"]].head(10)
        counts = attack["category"].value_counts().to_dict()
        summary_rows.append(
            {
                "variant": variant,
                "selection": selection_name,
                "monitored": int(selected_grid_row["num_monitored"]),
                "monitored_sensors": int(selected_grid_row["monitored_sensors"]),
                "monitored_actuators": int(selected_grid_row["monitored_actuators"]),
                "S": float(selected_grid_row["S"]),
                "G": float(selected_grid_row["G"]),
                "Prec": float(selected_grid_row["Precision"]),
                "Rec": float(selected_grid_row["Recall"]),
                "F1": float(selected_grid_row["F1"]),
                "eTaP": float(selected_grid_row["eTaP"]),
                "eTaR": float(selected_grid_row["eTaR"]),
                "eTaF1": float(selected_grid_row["eTaF1"]),
                "FPA": float(selected_grid_row["FPA"]),
                "Scen": float(selected_grid_row["Scen"]),
                "direct_detected": int(counts.get("direct_detected", 0)),
                "collateral_detected": int(counts.get("collateral_detected", 0)),
                "scope_miss": int(counts.get("scope_miss", 0)),
                "detection_failure": int(counts.get("detection_failure", 0)),
            }
        )

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out_root / "summary.csv", index=False)

    gap_rows = []
    windows = get_attack_windows("WADI")
    for attack_id, window in enumerate(windows, start=1):
        row: dict[str, Any] = {
            "attack_id": attack_id,
            "start": int(window.start),
            "end": int(window.end),
            "attacked_tags": ",".join(str(tag) for tag in window.affected_tags),
        }
        for variant in variant_names:
            attack_row = per_attack_by_variant[variant].loc[per_attack_by_variant[variant]["attack_id"] == attack_id].iloc[0]
            detected = bool(attack_row["detected"])
            row[f"detected_by_{variant}"] = detected
            row[f"miss_reason_{variant}"] = miss_reason(row["attacked_tags"], monitored_by_variant[variant], detected)
        gap_rows.append(row)
    coverage_gap = pd.DataFrame(gap_rows)
    coverage_gap.to_csv(out_root / "attack_coverage_gap.csv", index=False)
    (out_root / "summary.md").write_text(markdown_summary(summary, coverage_gap, top_fpa), encoding="utf-8")

    print("\n=== WADI 1-Second Delta Results ===")
    print(summary.to_string(index=False))
    print("\n=== Top FPA Contributors ===")
    for variant, table in top_fpa.items():
        print(f"\n{variant}")
        print(table.to_string(index=False))
    print(f"\nSaved: {out_root / 'summary.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
