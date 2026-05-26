#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
from dataclasses import dataclass
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


S_VALUES = [1.0, 1.42, 2.0, 3.0, 5.0, 8.0, 10.0]
G_VALUES = [2.0, 5.98, 9.0, 15.0, 25.0]
GECO_EXCLUSIONS = {"AIT201", "AIT202", "AIT203", "P201"}
AIT_FAMILY = {"AIT201", "AIT202", "AIT203", "AIT401", "AIT402", "AIT501", "AIT502", "AIT503", "AIT504"}


def _load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DIAG = _load_module(REPO_ROOT / "scripts" / "run_swat_1sec_delta_local_diagnostic.py", "swat1s_delta_diag")


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


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    return DIAG.regression_metrics(y_true, y_pred)


def alarm_burden(alarms: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    return DIAG.alarm_burden(alarms, labels)


def point_counts(labels: np.ndarray, alarms: np.ndarray) -> dict[str, int]:
    return DIAG.point_counts(labels, alarms)


@dataclass
class VariableModel:
    target: str
    variable_type: str
    equation: str
    sympy_format: str
    complexity: float
    loss: float
    score: float
    holdout_r2: float
    holdout_mae: float
    baseline_holdout_mae: float
    residual_tail_ratio: float
    source: str

    def to_row(self) -> dict[str, Any]:
        return {
            "target": self.target,
            "variable_type": self.variable_type,
            "equation": self.equation,
            "sympy_format": self.sympy_format,
            "complexity": self.complexity,
            "loss": self.loss,
            "score": self.score,
            "holdout_r2": self.holdout_r2,
            "holdout_mae": self.holdout_mae,
            "baseline_holdout_mae": self.baseline_holdout_mae,
            "residual_tail_ratio": self.residual_tail_ratio,
            "source": self.source,
        }


def load_arrays(args: argparse.Namespace) -> dict[str, Any]:
    return DIAG.load_arrays(args)


def target_delta(arrays: dict[str, Any], target: str, split: str) -> np.ndarray:
    idx = arrays["feature_columns"].index(target)
    if split == "train":
        return (arrays["train_next"][:, idx] - arrays["train_current"][:, idx]).astype(np.float64)
    return (arrays["test_next"][:, idx] - arrays["test_current"][:, idx]).astype(np.float64)


def residual_for_model(arrays: dict[str, Any], model: VariableModel, split: str) -> np.ndarray:
    feature_columns = arrays["feature_columns"]
    idx = feature_columns.index(model.target)
    if split == "train":
        current = arrays["train_current"]
        nxt = arrays["train_next"]
    else:
        current = arrays["test_current"]
        nxt = arrays["test_next"]
    if model.source == "actuator_persistence":
        pred_next = current[:, idx].astype(np.float64)
    elif model.variable_type == "actuator":
        pred_next = evaluate_equation(model.sympy_format, feature_columns, current).astype(np.float64)
    else:
        pred_delta = evaluate_equation(model.sympy_format, feature_columns, current).astype(np.float64)
        pred_next = reconstruct_next_from_delta(current[:, idx], pred_delta)
    residual = np.abs(nxt[:, idx].astype(np.float64) - pred_next)
    return np.where(np.isfinite(residual), residual, 0.0).astype(np.float64)


def make_actuator_persistence_models(feature_columns: list[str], exclude: set[str]) -> list[VariableModel]:
    out = []
    for target in feature_columns:
        if target in exclude or not is_actuator("SWAT", target):
            continue
        out.append(
            VariableModel(
                target=target,
                variable_type="actuator",
                equation=target,
                sympy_format=target,
                complexity=1.0,
                loss=float("nan"),
                score=float("nan"),
                holdout_r2=float("nan"),
                holdout_mae=float("nan"),
                baseline_holdout_mae=float("nan"),
                residual_tail_ratio=float("nan"),
                source="actuator_persistence",
            )
        )
    return out


def models_from_selected(path: Path, exclude: set[str] | None = None) -> list[VariableModel]:
    exclude = exclude or set()
    df = pd.read_csv(path)
    models = []
    for row in df.to_dict("records"):
        target = str(row["target"])
        if target in exclude:
            continue
        models.append(
            VariableModel(
                target=target,
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
                source="delta_full_selected",
            )
        )
    return models


def candidate_indices_by_score(df: pd.DataFrame) -> list[int]:
    table = df.copy()
    table["complexity_num"] = pd.to_numeric(table.get("complexity"), errors="coerce").fillna(np.inf)
    table["loss_num"] = pd.to_numeric(table.get("loss"), errors="coerce").fillna(np.inf)
    table["score_num"] = pd.to_numeric(table.get("score"), errors="coerce").fillna(-np.inf)
    return table.sort_values(["score_num", "loss_num", "complexity_num"], ascending=[False, True, True]).index.astype(int).tolist()


def candidate_rows_by_burden(df: pd.DataFrame) -> list[int]:
    return list(df.index)


def holdout_alarm_summary(residual_fit: np.ndarray, residual_holdout: np.ndarray) -> tuple[int, float]:
    params = fit_cusum_params(residual_fit, s=1.42, g=5.98)
    _, alarms = run_cusum(residual_holdout, params)
    return len(to_intervals(alarms)), float(np.mean(alarms)) if alarms.size else 0.0


def select_from_pareto(
    arrays: dict[str, Any],
    target: str,
    pareto_csv: Path,
    policy: str,
    max_complexity: int = 15,
) -> tuple[VariableModel | None, str | None]:
    if not pareto_csv.exists():
        return None, "missing_pareto_front"
    df = pd.read_csv(pareto_csv)
    if df.empty:
        return None, "empty_pareto_front"

    feature_columns = arrays["feature_columns"]
    fit_idx = arrays["fit_idx"]
    holdout_idx = arrays["holdout_idx"]
    y_all = target_delta(arrays, target, "train")
    baseline_mae = float(np.mean(np.abs(y_all[holdout_idx])))
    rows = []
    reasons = []
    for idx in df.index:
        row = df.loc[idx]
        complexity = safe_float(row.get("complexity"))
        if not np.isfinite(complexity) or complexity > max_complexity:
            reasons.append(f"row={idx}:complexity>{max_complexity}")
            continue
        equation = str(row.get("sympy_format", row.get("equation", "")))
        pred = evaluate_equation(equation, feature_columns, arrays["train_current"])
        if not np.isfinite(pred).all():
            reasons.append(f"row={idx}:nonfinite_train")
            continue
        residual = np.abs(y_all - pred)
        residual = np.where(np.isfinite(residual), residual, 0.0)
        median = float(np.median(residual))
        p99 = float(np.percentile(residual, 99))
        tail = p99 / max(median, 1e-9)
        holdout_pred = pred[holdout_idx]
        metrics = regression_metrics(y_all[holdout_idx], holdout_pred)
        intervals, frac = holdout_alarm_summary(residual[fit_idx], residual[holdout_idx])
        base = {
            "idx": int(idx),
            "row": row,
            "complexity": complexity,
            "equation": equation,
            "tail": tail,
            "holdout_r2": metrics["r2"],
            "holdout_mae": metrics["mae"],
            "baseline_mae": baseline_mae,
            "holdout_alarm_intervals": intervals,
            "holdout_alarm_fraction": frac,
        }
        if policy == "strict_current":
            if not (metrics["r2"] >= 0.0 or metrics["mae"] < baseline_mae):
                reasons.append(f"row={idx}:quality_fail")
                continue
            if intervals > 0:
                reasons.append(f"row={idx}:holdout_alarm")
                continue
            if tail > 50.0:
                reasons.append(f"row={idx}:tail>{tail:.4g}")
                continue
            rows.append(base)
        elif policy == "relaxed_stability":
            if tail > 100.0:
                reasons.append(f"row={idx}:tail>{tail:.4g}")
                continue
            if intervals > 2:
                reasons.append(f"row={idx}:holdout_intervals>{intervals}")
                continue
            if frac > 0.005:
                reasons.append(f"row={idx}:holdout_fraction>{frac:.4g}")
                continue
            rows.append(base)
        elif policy == "burden_ranked":
            improvement = (baseline_mae - metrics["mae"]) / max(baseline_mae, 1e-12)
            if tail > 150.0:
                reasons.append(f"row={idx}:tail>{tail:.4g}")
                continue
            if not (metrics["r2"] >= 0.05 or improvement >= 0.05):
                reasons.append(f"row={idx}:quality_fail")
                continue
            if frac > 0.01:
                reasons.append(f"row={idx}:holdout_fraction>{frac:.4g}")
                continue
            burden = 1000.0 * frac + 10.0 * intervals + tail / 50.0 - max(metrics["r2"], 0.0)
            base["burden"] = burden
            rows.append(base)
        else:
            raise ValueError(policy)

    if not rows:
        return None, "; ".join(reasons[-8:])
    if policy in {"strict_current", "relaxed_stability"}:
        rows.sort(key=lambda item: (safe_float(item["row"].get("score"), -np.inf), -safe_float(item["row"].get("loss"), np.inf), -item["complexity"]), reverse=True)
    else:
        rows.sort(key=lambda item: (item["burden"], -safe_float(item["row"].get("score"), -np.inf)))
    best = rows[0]
    row = best["row"]
    return (
        VariableModel(
            target=target,
            variable_type="sensor",
            equation=str(row.get("equation", "")),
            sympy_format=best["equation"],
            complexity=best["complexity"],
            loss=safe_float(row.get("loss")),
            score=safe_float(row.get("score")),
            holdout_r2=best["holdout_r2"],
            holdout_mae=best["holdout_mae"],
            baseline_holdout_mae=best["baseline_mae"],
            residual_tail_ratio=best["tail"],
            source=policy,
        ),
        None,
    )


def select_policy_models(arrays: dict[str, Any], pareto_root: Path, policy: str) -> tuple[list[VariableModel], pd.DataFrame]:
    models = []
    exclusions = []
    for target in DIAG.SWAT_SENSOR_TARGETS:
        pareto = pareto_root / f"{target}_sensors_delta_actuators_next" / "pareto_front_scored.csv"
        model, reason = select_from_pareto(arrays, target, pareto, policy)
        if model is None:
            exclusions.append({"policy": policy, "target": target, "reason": reason or "unknown"})
        else:
            models.append(model)
    return models, pd.DataFrame(exclusions)


def residual_cache_for_models(arrays: dict[str, Any], models: list[VariableModel]) -> dict[str, dict[str, np.ndarray]]:
    cache = {}
    for model in models:
        if model.target in cache:
            continue
        cache[model.target] = {
            "train": residual_for_model(arrays, model, "train"),
            "test": residual_for_model(arrays, model, "test"),
        }
    return cache


def _positive_or_floor(values: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(out) & (out > 0.0), out, float(floor))


def fit_batch_base(train_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(train_matrix, dtype=np.float64)
    r = np.where(np.isfinite(r), r, 0.0)
    delta = _positive_or_floor(np.mean(r, axis=0) + np.std(r, axis=0))
    cusum = np.zeros(r.shape[1], dtype=np.float64)
    max_cusum = np.zeros(r.shape[1], dtype=np.float64)
    for i in range(r.shape[0]):
        cusum = np.maximum(0.0, cusum + r[i] - delta)
        max_cusum = np.maximum(max_cusum, cusum)
    return delta, max_cusum


def run_batch_cusum(test_matrix: np.ndarray, delta: np.ndarray, max_train_cusum: np.ndarray, *, s: float, g: float) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    r = np.asarray(test_matrix, dtype=np.float64)
    r = np.where(np.isfinite(r), r, 0.0)
    threshold = _positive_or_floor(float(s) * max_train_cusum)
    growth_cap = _positive_or_floor(threshold + float(g) * delta)
    cusum = np.zeros(r.shape[1], dtype=np.float64)
    max_test = np.zeros(r.shape[1], dtype=np.float64)
    alarms = np.zeros(r.shape, dtype=np.int8)
    for i in range(r.shape[0]):
        raw = np.maximum(0.0, cusum + r[i] - delta)
        cusum = np.minimum(raw, growth_cap)
        max_test = np.maximum(max_test, cusum)
        alarms[i] = (cusum > threshold).astype(np.int8)
    return alarms, threshold, growth_cap, max_test


def stack_residuals(models: list[VariableModel], cache: dict[str, dict[str, np.ndarray]]) -> tuple[np.ndarray, np.ndarray]:
    train = np.column_stack([cache[m.target]["train"] for m in models]).astype(np.float64, copy=False)
    test = np.column_stack([cache[m.target]["test"] for m in models]).astype(np.float64, copy=False)
    return train, test


def evaluate_variant(
    arrays: dict[str, Any],
    models: list[VariableModel],
    *,
    variant: str,
    s_values: list[float] = S_VALUES,
    g_values: list[float] = G_VALUES,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[float, float], dict[str, np.ndarray]], dict[str, dict[str, np.ndarray]]]:
    labels = arrays["labels"]
    cache = residual_cache_for_models(arrays, models)
    train_matrix, test_matrix = stack_residuals(models, cache) if models else (np.empty((labels.shape[0], 0)), np.empty((labels.shape[0], 0)))
    delta, max_train_cusum = fit_batch_base(train_matrix) if models else (np.array([]), np.array([]))
    grid_rows = []
    per_sensor_rows = []
    alarm_cache: dict[tuple[float, float], dict[str, np.ndarray]] = {}
    for s in s_values:
        for g in g_values:
            if models:
                alarm_matrix, threshold, growth_cap, max_test = run_batch_cusum(
                    test_matrix,
                    delta,
                    max_train_cusum,
                    s=float(s),
                    g=float(g),
                )
            else:
                alarm_matrix = np.zeros((labels.shape[0], 0), dtype=np.int8)
                threshold = growth_cap = max_test = np.array([])
            for j, model in enumerate(models):
                alarm = alarm_matrix[:, j].astype(np.int8)
                burden = alarm_burden(alarm, labels)
                per_sensor_rows.append(
                    {
                        "variant": variant,
                        "target": model.target,
                        "variable_type": model.variable_type,
                        "source": model.source,
                        "S": s,
                        "G": g,
                        "complexity": model.complexity,
                        "equation": model.equation,
                        "holdout_r2": model.holdout_r2,
                        "residual_tail_ratio": model.residual_tail_ratio,
                        "delta": float(delta[j]),
                        "threshold": float(threshold[j]),
                        "growth_cap": float(growth_cap[j]),
                        "max_train_cusum": float(max_train_cusum[j]),
                        "max_test_cusum": float(max_test[j]),
                        **burden,
                    }
                )
            system_alarm = np.max(alarm_matrix, axis=1).astype(np.int8) if models else np.zeros_like(labels, dtype=np.int8)
            alarm_cache[(float(s), float(g))] = {"system": system_alarm}
            metrics = compute_detection_metrics(labels, system_alarm, expand_steps=60)
            grid_rows.append(
                {
                    "variant": variant,
                    "S": s,
                    "G": g,
                    "num_monitored": len(models),
                    "monitored_sensors": sum(1 for m in models if m.variable_type == "sensor"),
                    "monitored_actuators": sum(1 for m in models if m.variable_type == "actuator"),
                    "Precision": metrics["point_precision"],
                    "Recall": metrics["point_recall"],
                    "F1": metrics["point_f1"],
                    "eTaP": metrics["eTaP"],
                    "eTaR": metrics["eTaR"],
                    "eTaF1": metrics["eTaF1"],
                    "FPA": metrics["FPA"],
                    "Scen": metrics["scenario_detection_rate"],
                    **point_counts(labels, system_alarm),
                    **{f"system_{k}": v for k, v in alarm_burden(system_alarm, labels).items()},
                }
            )
    return pd.DataFrame(grid_rows), pd.DataFrame(per_sensor_rows), alarm_cache, cache


def alarm_map_for_choice(models: list[VariableModel], cache: dict[str, dict[str, np.ndarray]], labels: np.ndarray, *, s: float, g: float) -> dict[str, np.ndarray]:
    if not models:
        return {"system": np.zeros_like(labels, dtype=np.int8)}
    train_matrix, test_matrix = stack_residuals(models, cache)
    delta, max_train_cusum = fit_batch_base(train_matrix)
    alarm_matrix, _, _, _ = run_batch_cusum(test_matrix, delta, max_train_cusum, s=float(s), g=float(g))
    out = {"system": np.max(alarm_matrix, axis=1).astype(np.int8)}
    for j, model in enumerate(models):
        out[model.target] = alarm_matrix[:, j].astype(np.int8)
    return out


def choose_rows(grid: pd.DataFrame) -> dict[str, pd.Series | None]:
    choices: dict[str, pd.Series | None] = {}
    for key, table in {
        "best_f1_fpa_le_5": grid[grid["FPA"] <= 5].sort_values(["F1", "eTaF1"], ascending=False),
        "best_f1_fpa_le_15": grid[grid["FPA"] <= 15].sort_values(["F1", "eTaF1"], ascending=False),
        "best_etaf1_fpa_le_15": grid[grid["FPA"] <= 15].sort_values(["eTaF1", "F1"], ascending=False),
        "geco_s_g": grid[(grid["S"] == 1.42) & (grid["G"] == 5.98)].sort_values(["F1", "eTaF1"], ascending=False),
        "lowest_fpa": grid.sort_values(["FPA", "F1"], ascending=[True, False]),
        "best_f1_overall": grid.sort_values(["F1", "eTaF1"], ascending=False),
    }.items():
        choices[key] = None if table.empty else table.iloc[0]
    return choices


def first_alarm_delay(alarm: np.ndarray, start_idx: int, end_idx: int, original_start: int) -> int | None:
    segment = alarm[start_idx : end_idx + 1]
    hits = np.flatnonzero(segment)
    if hits.size == 0:
        return None
    return max(0, int(start_idx + int(hits[0]) + 1) - int(original_start))


def per_attack_table(models: list[VariableModel], alarm_map: dict[str, np.ndarray]) -> pd.DataFrame:
    monitored = {m.target for m in models}
    if "system" in alarm_map:
        n = alarm_map["system"].shape[0]
    else:
        n = int(next(iter(alarm_map.values())).shape[0])
    rows = []
    for attack_id, window in enumerate(get_attack_windows("SWAT"), start=1):
        start = max(0, int(window.start) - 1)
        end = min(n - 1, int(window.end) - 1)
        affected = [str(tag) for tag in window.affected_tags]
        in_scope = [tag for tag in affected if tag in monitored]
        firing = [
            target
            for target, alarm in alarm_map.items()
            if target != "system" and np.any(alarm[start : end + 1] == 1)
        ]
        in_scope_firing = [tag for tag in in_scope if tag in firing]
        if in_scope_firing:
            category = "direct_detected"
        elif not in_scope and firing:
            category = "collateral_detected"
        elif not in_scope and not firing:
            category = "scope_miss"
        else:
            category = "detection_failure"
        delays = [
            first_alarm_delay(alarm_map[tag], start, end, int(window.start))
            for tag in firing
            if tag in alarm_map
        ]
        delays = [d for d in delays if d is not None]
        rows.append(
            {
                "attack_id": attack_id,
                "start": int(window.start),
                "end": int(window.end),
                "affected_tags": ",".join(affected),
                "in_scope_sensors": ",".join(in_scope),
                "category": category,
                "firing_sensors": ",".join(firing),
                "detected": bool(firing),
                "detection_delay_seconds": min(delays) if delays else None,
            }
        )
    return pd.DataFrame(rows)


def miss_reason(affected_tags: str, monitored: set[str], detected: bool) -> str:
    if detected:
        return ""
    tags = [tag for tag in str(affected_tags).split(",") if tag]
    in_scope = [tag for tag in tags if tag in monitored]
    if in_scope:
        return "monitored but no alarm"
    if tags and all(tag in AIT_FAMILY for tag in tags):
        return "only AIT excluded"
    if tags and all(is_actuator("SWAT", tag) for tag in tags):
        return "only actuator missing"
    return "not monitored"


def write_variant_outputs(out: Path, name: str, grid: pd.DataFrame, per_sensor: pd.DataFrame, models: list[VariableModel], alarm_cache: dict[tuple[float, float], dict[str, np.ndarray]], residual_cache: dict[str, dict[str, np.ndarray]], labels: np.ndarray) -> tuple[pd.Series, pd.DataFrame]:
    grid.to_csv(out / f"grid_{name}.csv", index=False)
    per_sensor.to_csv(out / f"per_sensor_{name}.csv", index=False)
    choices = choose_rows(grid)
    choice = choices["best_f1_fpa_le_15"]
    if choice is None:
        choice = choices["lowest_fpa"]
    if choice is None:
        choice = choices["best_f1_overall"]
    if choice is None:
        per_attack = pd.DataFrame()
    else:
        alarm_map = alarm_map_for_choice(models, residual_cache, labels, s=float(choice["S"]), g=float(choice["G"]))
        per_attack = per_attack_table(models, alarm_map)
    per_attack.to_csv(out / f"per_attack_{name}.csv", index=False)
    return choice, per_attack


def summarize_variant(name: str, choice: pd.Series, models: list[VariableModel], per_attack: pd.DataFrame) -> dict[str, Any]:
    counts = per_attack["category"].value_counts().to_dict() if not per_attack.empty else {}
    return {
        "variant": name,
        "monitored_sensors": sum(1 for m in models if m.variable_type == "sensor"),
        "monitored_actuators": sum(1 for m in models if m.variable_type == "actuator"),
        "S": float(choice["S"]),
        "G": float(choice["G"]),
        "F1": float(choice["F1"]),
        "eTaF1": float(choice["eTaF1"]),
        "FPA": float(choice["FPA"]),
        "Scen": float(choice["Scen"]),
        "direct_detected": int(counts.get("direct_detected", 0)),
        "collateral_detected": int(counts.get("collateral_detected", 0)),
        "scope_miss": int(counts.get("scope_miss", 0)),
        "detection_failure": int(counts.get("detection_failure", 0)),
    }


def markdown_summary(rows: list[dict[str, Any]]) -> str:
    df = pd.DataFrame(rows)
    if df.empty:
        table = "(no rows)"
    else:
        display = df.copy()
        for col in display.columns:
            if pd.api.types.is_float_dtype(display[col]):
                display[col] = display[col].map(lambda value: f"{value:.3f}")
        headers = [str(col) for col in display.columns]
        table_rows = ["| " + " | ".join(headers) + " |"]
        table_rows.append("| " + " | ".join(["---"] * len(headers)) + " |")
        for row in display.astype(str).itertuples(index=False, name=None):
            table_rows.append("| " + " | ".join(row) + " |")
        table = "\n".join(table_rows)
    lines = [
        "# SWaT 1-second Delta Posthoc Ablation",
        "",
        "Manual no-AIT203 and no-AIT-family variants are diagnostic only. Paper-defensible modes are GeCo-matched exclusions and training-only stability filters.",
        "",
        table,
        "",
    ]
    return "\n".join(lines)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Posthoc replay ablations for SWaT 1-second delta run.")
    parser.add_argument("--experiment", default="configs/experiment/swat_mlp_current_val20.yaml")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--delta-full", default="artifacts/swat_1sec/delta_full")
    parser.add_argument("--out", default="artifacts/swat_1sec/delta_posthoc_ablation")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    arrays = load_arrays(args)
    delta_full = Path(args.delta_full)
    selected_path = delta_full / "selected_equations.csv"
    pareto_root = delta_full / "pareto_fronts"

    write_json(
        out / "run_config.json",
        {
            "delta_full": str(delta_full),
            "s_values": S_VALUES,
            "g_values": G_VALUES,
            "geco_exclusions": sorted(GECO_EXCLUSIONS),
            "ait_family": sorted(AIT_FAMILY),
            "notes": "No PySR fitting is performed; replays use existing Pareto fronts and actuator persistence.",
        },
    )

    summary_rows = []
    variant_tables: dict[str, pd.DataFrame] = {}
    variant_models: dict[str, list[VariableModel]] = {}
    variant_attacks: dict[str, pd.DataFrame] = {}

    base_models = models_from_selected(selected_path)
    original_grid = pd.read_csv(delta_full / "detection_grid.csv")
    original_choice = choose_rows(original_grid)["best_f1_overall"]
    original_attacks = pd.read_csv(delta_full / "per_attack_analysis.csv")
    summary_rows.append(summarize_variant("original_delta_full", original_choice, base_models, original_attacks))

    replay_specs = {
        "no_ait203": {"exclude": {"AIT203"}, "actuators": False},
        "geco_matched_exclusions": {"exclude": GECO_EXCLUSIONS, "actuators": False},
        "no_ait_family": {"exclude": AIT_FAMILY, "actuators": False},
        "no_ait203_plus_actuators": {"exclude": {"AIT203"}, "actuators": True},
        "geco_matched_plus_actuators": {"exclude": GECO_EXCLUSIONS, "actuators": True},
        "no_ait_family_plus_actuators": {"exclude": AIT_FAMILY, "actuators": True},
    }
    plus_actuator_grids = []
    plus_actuator_sensors = []
    plus_actuator_attacks = []
    for name, spec in replay_specs.items():
        models = models_from_selected(selected_path, exclude=set(spec["exclude"]))
        if spec["actuators"]:
            models += make_actuator_persistence_models(arrays["feature_columns"], exclude=set(spec["exclude"]))
        print(f"Running replay {name}: {len(models)} variables", flush=True)
        grid, per_sensor, alarms, residual_cache = evaluate_variant(arrays, models, variant=name)
        file_name = name.replace("_plus_actuators", "")
        if spec["actuators"]:
            plus_actuator_grids.append(grid)
            plus_actuator_sensors.append(per_sensor)
        else:
            choice, per_attack = write_variant_outputs(out, file_name, grid, per_sensor, models, alarms, residual_cache, arrays["labels"])
            summary_rows.append(summarize_variant(name, choice, models, per_attack))
            variant_tables[name] = grid
            variant_models[name] = models
            variant_attacks[name] = per_attack
            continue
        choice, per_attack = write_variant_outputs(out, name, grid, per_sensor, models, alarms, residual_cache, arrays["labels"])
        plus_actuator_attacks.append(per_attack.assign(variant=name))
        summary_rows.append(summarize_variant(name, choice, models, per_attack))
        variant_tables[name] = grid
        variant_models[name] = models
        variant_attacks[name] = per_attack

    if plus_actuator_grids:
        pd.concat(plus_actuator_grids, ignore_index=True).to_csv(out / "grid_sensors_plus_actuators.csv", index=False)
        pd.concat(plus_actuator_sensors, ignore_index=True).to_csv(out / "per_sensor_sensors_plus_actuators.csv", index=False)
        pd.concat(plus_actuator_attacks, ignore_index=True).to_csv(out / "per_attack_sensors_plus_actuators.csv", index=False)

    policy_summaries = []
    for policy in ["strict_current", "relaxed_stability", "burden_ranked"]:
        print(f"Selecting policy {policy}", flush=True)
        models, exclusions = select_policy_models(arrays, pareto_root, policy)
        pd.DataFrame([m.to_row() for m in models]).to_csv(out / f"selected_equations_{policy}.csv", index=False)
        exclusions.to_csv(out / f"exclusion_reasons_{policy}.csv", index=False)
        scopes = {
            f"{policy}_sensors_only": models,
            f"{policy}_geco_matched": [m for m in models if m.target not in GECO_EXCLUSIONS],
            f"{policy}_plus_actuators": models + make_actuator_persistence_models(arrays["feature_columns"], exclude=set()),
        }
        policy_grids = []
        for scope_name, scope_models in scopes.items():
            grid, per_sensor, alarms, residual_cache = evaluate_variant(arrays, scope_models, variant=scope_name)
            choice, per_attack = write_variant_outputs(out, scope_name, grid, per_sensor, scope_models, alarms, residual_cache, arrays["labels"])
            policy_grids.append(grid)
            policy_summaries.append(summarize_variant(scope_name, choice, scope_models, per_attack))
            if policy == "relaxed_stability" and scope_name == f"{policy}_sensors_only":
                summary_rows.append({**summarize_variant("best_relaxed_policy", choice, scope_models, per_attack)})
            if policy == "relaxed_stability" and scope_name == f"{policy}_plus_actuators":
                summary_rows.append({**summarize_variant("best_relaxed_policy_plus_actuators", choice, scope_models, per_attack)})
        pd.concat(policy_grids, ignore_index=True).to_csv(out / f"detection_grid_{policy}.csv", index=False)

    pd.DataFrame(policy_summaries).to_csv(out / "selection_policy_comparison.csv", index=False)

    coverage_variants = {
        "original_delta_full": original_attacks,
        "no_ait203": variant_attacks.get("no_ait203"),
        "geco_matched": variant_attacks.get("geco_matched_exclusions"),
        "sensors_plus_actuators": variant_attacks.get("no_ait203_plus_actuators"),
    }
    coverage_rows = []
    original_monitored = {m.target for m in base_models}
    monitored_sets = {
        "original_delta_full": original_monitored,
        "no_ait203": {m.target for m in variant_models.get("no_ait203", [])},
        "geco_matched": {m.target for m in variant_models.get("geco_matched_exclusions", [])},
        "sensors_plus_actuators": {m.target for m in variant_models.get("no_ait203_plus_actuators", [])},
    }
    template = next(df for df in coverage_variants.values() if df is not None)
    for _, base in template.iterrows():
        row = {
            "attack_id": int(base["attack_id"]),
            "attack_target_variables": base["affected_tags"],
        }
        for name, table in coverage_variants.items():
            if table is None:
                continue
            match = table[table["attack_id"] == int(base["attack_id"])].iloc[0]
            detected = bool(match["detected"]) if "detected" in match else str(match["category"]) in {"direct_detected", "collateral_detected"}
            row[f"detected_by_{name}"] = detected
            row[f"reason_{name}"] = miss_reason(str(match["affected_tags"]), monitored_sets.get(name, set()), detected)
        coverage_rows.append(row)
    pd.DataFrame(coverage_rows).to_csv(out / "attack_coverage_gap.csv", index=False)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(out / "summary.csv", index=False)
    (out / "summary.md").write_text(markdown_summary(summary_rows), encoding="utf-8")

    print("\n=== No-AIT203 replay ===")
    for key, row in choose_rows(variant_tables["no_ait203"]).items():
        if row is not None and key in {"best_f1_fpa_le_5", "best_f1_fpa_le_15", "best_etaf1_fpa_le_15", "geco_s_g"}:
            print(f"{key}: S={row['S']} G={row['G']} F1={row['F1']:.1f} eTaF1={row['eTaF1']:.1f} FPA={row['FPA']:.0f} Scen={row['Scen']:.1f}")
        elif key in {"best_f1_fpa_le_5", "best_f1_fpa_le_15", "best_etaf1_fpa_le_15", "geco_s_g"}:
            print(f"{key}: none")
    print("\n=== Summary ===")
    print(summary.to_string(index=False))
    print(f"\nWrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
