#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
import math
import os
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ics_symbolic_distill.detection import compute_detection_metrics, evaluate_equation, fit_cusum_params, run_cusum
from ics_symbolic_distill.detection.metrics import to_intervals
from ics_symbolic_distill.detection.swat1s_delta_sampling import (
    coverage_stratified_indices,
    reconstruct_next_from_delta,
    sampling_audit_for_uniform,
    uniform_grid_indices,
)


DIAGNOSTIC_TARGETS = [
    "LIT101",
    "FIT201",
    "DPIT301",
    "FIT301",
    "LIT401",
    "AIT504",
    "FIT601",
    "PIT502",
    "AIT201",
    "AIT203",
]

SWAT_SENSOR_TARGETS = [
    "FIT101",
    "LIT101",
    "AIT201",
    "AIT202",
    "AIT203",
    "FIT201",
    "DPIT301",
    "FIT301",
    "LIT301",
    "AIT401",
    "AIT402",
    "FIT401",
    "LIT401",
    "AIT501",
    "AIT502",
    "AIT503",
    "AIT504",
    "FIT501",
    "FIT502",
    "FIT503",
    "FIT504",
    "PIT501",
    "PIT502",
    "PIT503",
    "FIT601",
]

CONFIGS = {
    "A": {"sample_policy": "uniform_grid", "sample_size": 8000, "operator_set": "safe_mul", "scope": "sensors-only"},
    "B": {"sample_policy": "coverage_stratified", "sample_size": 8000, "operator_set": "safe_mul", "scope": "sensors-only"},
    "C": {"sample_policy": "coverage_stratified", "sample_size": 12000, "operator_set": "safe_mul", "scope": "sensors-only"},
    "D": {"sample_policy": "coverage_stratified", "sample_size": 8000, "operator_set": "rich_div", "scope": "sensors-only"},
    "FULL": {"sample_policy": "coverage_stratified", "sample_size": 12000, "operator_set": "safe_mul", "scope": "sensors-only"},
}


def _load_day1_module():
    path = REPO_ROOT / "scripts" / "run_swat_1sec_pysr.py"
    spec = importlib.util.spec_from_file_location("run_swat_1sec_pysr", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


DAY1 = _load_day1_module()


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


def parse_float_list(value: str, *, default: list[float]) -> list[float]:
    if value is None or str(value).strip() == "":
        return default
    out = [float(tok.strip()) for tok in str(value).split(",") if tok.strip()]
    return out or default


def pareto_dir(args: argparse.Namespace, config_name: str, target: str) -> Path:
    root = Path(args.out) / "pareto_fronts"
    if getattr(args, "flat_pareto_layout", False):
        return root / f"{target}_{args.target_mode}"
    return root / config_name / f"{target}_{args.target_mode}"


def safe_float(value: Any) -> float:
    try:
        out = float(value)
    except Exception:
        return float("nan")
    return out if math.isfinite(out) else float("nan")


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    finite = np.isfinite(y) & np.isfinite(pred)
    if not np.any(finite):
        return {"mse": float("inf"), "rmse": float("inf"), "mae": float("inf"), "r2": float("-inf")}
    y = y[finite]
    pred = pred[finite]
    err = pred - y
    mse = float(np.mean(err**2))
    var = float(np.mean((y - np.mean(y)) ** 2))
    return {
        "mse": mse,
        "rmse": float(np.sqrt(mse)),
        "mae": float(np.mean(np.abs(err))),
        "r2": float(1.0 - mse / var) if var > 0.0 else float("-inf"),
    }


def features_in_equation(equation: str, feature_names: list[str]) -> list[str]:
    return DAY1.features_in_equation(str(equation), feature_names)


def operator_params(operator_set: str, *, niterations: int, timeout_minutes: float, max_complexity: int, procs: int, seed: int):
    if operator_set == "safe_mul":
        binary_operators = ["+", "-", "*"]
        extra_sympy_mappings = {}
    elif operator_set == "rich_div":
        binary_operators = ["+", "-", "*", "protected_division(x, y) = x / (abs(y) + 1.0e-6)"]
        extra_sympy_mappings = {"protected_division": lambda x, y: x / (abs(y) + 1.0e-6)}
    else:
        raise ValueError(f"Unknown operator_set={operator_set}")
    return {
        "niterations": int(niterations),
        "binary_operators": binary_operators,
        "unary_operators": [],
        "extra_sympy_mappings": extra_sympy_mappings,
        "maxsize": int(max_complexity),
        "populations": 20,
        "parsimony": 0.01,
        "procs": int(procs),
        "timeout_in_seconds": int(float(timeout_minutes) * 60),
        "temp_equation_file": True,
        "random_state": int(seed),
        "model_selection": "score",
        "verbosity": 0,
        "progress": False,
    }


def make_model(**kwargs):
    from pysr import PySRRegressor

    params = operator_params(**kwargs)
    supported = DAY1._filter_supported_params(PySRRegressor, params)
    return PySRRegressor(**supported), supported


def load_arrays(args: argparse.Namespace):
    train, train_labels, test, test_labels, feature_columns, metadata = DAY1.load_swat_1sec_arrays(args)
    if test_labels is None:
        raise ValueError("SWaT test labels are required")
    train_current = train[:-1].astype(np.float32, copy=False)
    train_next = train[1:].astype(np.float32, copy=False)
    test_current = test[:-1].astype(np.float32, copy=False)
    test_next = test[1:].astype(np.float32, copy=False)
    labels = test_labels[1:].astype(np.int64)
    n_pairs = int(train_current.shape[0])
    cutoff = int(math.floor(n_pairs * 0.8))
    return {
        "train": train,
        "test": test,
        "train_current": train_current,
        "train_next": train_next,
        "test_current": test_current,
        "test_next": test_next,
        "labels": labels,
        "feature_columns": feature_columns,
        "metadata": metadata,
        "fit_idx": np.arange(cutoff, dtype=np.int64),
        "holdout_idx": np.arange(cutoff, n_pairs, dtype=np.int64),
    }


def is_actuator(tag: str) -> bool:
    return bool(DAY1.is_actuator("SWAT", tag))


def target_values(arrays: dict[str, Any], target: str, *, split: str, target_mode: str) -> tuple[np.ndarray, np.ndarray]:
    feature_columns = arrays["feature_columns"]
    idx = int(feature_columns.index(target))
    if split == "train":
        current = arrays["train_current"]
        nxt = arrays["train_next"]
    elif split == "test":
        current = arrays["test_current"]
        nxt = arrays["test_next"]
    else:
        raise ValueError(split)
    if target_mode == "sensors_delta_actuators_next" and not is_actuator(target):
        y = nxt[:, idx] - current[:, idx]
    else:
        y = nxt[:, idx]
    return current, y.astype(np.float64)


def detection_residual(arrays: dict[str, Any], target: str, equation: str, *, split: str, target_mode: str) -> np.ndarray:
    feature_columns = arrays["feature_columns"]
    idx = int(feature_columns.index(target))
    if split == "train":
        current = arrays["train_current"]
        nxt = arrays["train_next"]
    else:
        current = arrays["test_current"]
        nxt = arrays["test_next"]
    pred = evaluate_equation(str(equation), feature_columns, current)
    if target_mode == "sensors_delta_actuators_next" and not is_actuator(target):
        pred_next = reconstruct_next_from_delta(current[:, idx], pred)
    else:
        pred_next = pred
    residual = np.abs(nxt[:, idx].astype(np.float64) - pred_next.astype(np.float64))
    return np.where(np.isfinite(residual), residual, 0.0).astype(np.float64)


def sample_indices_for_target(
    arrays: dict[str, Any],
    target: str,
    *,
    sample_policy: str,
    sample_size: int,
) -> tuple[np.ndarray, dict[str, Any]]:
    fit_idx = arrays["fit_idx"]
    current, y_all = target_values(arrays, target, split="train", target_mode="sensors_delta_actuators_next")
    y_fit = y_all[fit_idx]
    if sample_policy == "uniform_grid":
        local_idx = uniform_grid_indices(fit_idx.shape[0], int(sample_size))
        audit = sampling_audit_for_uniform(target, y_fit, int(sample_size), int(local_idx.shape[0]))
        return fit_idx[local_idx], audit.to_dict()
    if sample_policy == "coverage_stratified":
        feature_columns = arrays["feature_columns"]
        actuator_indices = [idx for idx, name in enumerate(feature_columns) if is_actuator(name)]
        local_idx, audit = coverage_stratified_indices(
            target=target,
            y_delta_fit_pool=y_fit,
            x_current_fit_pool=arrays["train_current"][fit_idx],
            x_next_fit_pool=arrays["train_next"][fit_idx],
            actuator_indices=actuator_indices,
            sample_size=int(sample_size),
            seed=1337,
        )
        return fit_idx[local_idx], audit.to_dict()
    raise ValueError(f"Unknown sample_policy={sample_policy}")


def evaluate_pareto_df(model: Any, equations: pd.DataFrame, x_sample: pd.DataFrame, y_sample: np.ndarray, x_holdout: pd.DataFrame, y_holdout: np.ndarray, feature_columns: list[str]) -> pd.DataFrame:
    rows = []
    for equation_index, row in equations.reset_index(drop=True).iterrows():
        out = row.to_dict()
        try:
            pred_sample = np.asarray(model.predict(x_sample, index=int(equation_index)), dtype=np.float64)
            pred_holdout = np.asarray(model.predict(x_holdout, index=int(equation_index)), dtype=np.float64)
            sample = regression_metrics(y_sample, pred_sample)
            holdout = regression_metrics(y_holdout, pred_holdout)
            out.update(
                {
                    "fit_mse": sample["mse"],
                    "fit_rmse": sample["rmse"],
                    "fit_mae": sample["mae"],
                    "fit_r2_against_constant": sample["r2"],
                    "holdout_mse": holdout["mse"],
                    "holdout_rmse": holdout["rmse"],
                    "holdout_mae": holdout["mae"],
                    "holdout_r2_against_constant": holdout["r2"],
                    "equation_features": json.dumps(features_in_equation(str(row.get("equation", "")), feature_columns)),
                    "evaluation_error": "",
                }
            )
        except Exception as exc:
            out.update(
                {
                    "fit_mse": np.nan,
                    "fit_rmse": np.nan,
                    "fit_mae": np.nan,
                    "fit_r2_against_constant": np.nan,
                    "holdout_mse": np.nan,
                    "holdout_rmse": np.nan,
                    "holdout_mae": np.nan,
                    "holdout_r2_against_constant": np.nan,
                    "equation_features": "[]",
                    "evaluation_error": str(exc),
                }
            )
        rows.append(out)
    return pd.DataFrame(rows)


def run_pysr_target(arrays: dict[str, Any], args: argparse.Namespace, config_name: str, config: dict[str, Any], target: str) -> dict[str, Any]:
    out_dir = pareto_dir(args, config_name, target)
    csv_path = out_dir / "pareto_front_scored.csv"
    if csv_path.exists() and not args.force:
        return {"config": config_name, "target": target, "status": "skipped_existing", "pareto_csv": str(csv_path)}

    started = time.time()
    feature_columns = arrays["feature_columns"]
    current, y_all = target_values(arrays, target, split="train", target_mode=args.target_mode)
    fit_idx = arrays["fit_idx"]
    holdout_idx = arrays["holdout_idx"]
    sample_idx, audit = sample_indices_for_target(
        arrays,
        target,
        sample_policy=str(config["sample_policy"]),
        sample_size=int(config["sample_size"]),
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "sample_indices.npy", sample_idx.astype(np.int64))
    pd.DataFrame([{"config": config_name, **audit}]).to_csv(out_dir / "sample_audit.csv", index=False)

    x_sample = pd.DataFrame(current[sample_idx], columns=feature_columns)
    y_sample = y_all[sample_idx]
    x_holdout = pd.DataFrame(current[holdout_idx], columns=feature_columns)
    y_holdout = y_all[holdout_idx]
    model, params = make_model(
        operator_set=str(config["operator_set"]),
        niterations=int(args.niterations),
        timeout_minutes=float(args.timeout_minutes),
        max_complexity=int(args.max_complexity),
        procs=int(args.parallel_jobs),
        seed=int(args.seed) + abs(hash((config_name, target))) % 100000,
    )
    status = "completed"
    error = ""
    try:
        model.fit(x_sample, y_sample)
        equations = model.equations_.copy()
        pareto = evaluate_pareto_df(model, equations, x_sample, y_sample, x_holdout, y_holdout, feature_columns)
        pareto.to_csv(csv_path, index=False)
    except Exception as exc:
        status = "failed"
        error = str(exc)
    write_json(
        out_dir / "metadata.json",
        {
            "config": config_name,
            "target": target,
            "target_mode": args.target_mode,
            "sample_policy": config["sample_policy"],
            "sample_size": int(config["sample_size"]),
            "operator_set": config["operator_set"],
            "scope": config["scope"],
            "max_complexity": int(args.max_complexity),
            "timeout_minutes": float(args.timeout_minutes),
            "train_fit_pool_rows": int(fit_idx.shape[0]),
            "train_holdout_rows": int(holdout_idx.shape[0]),
            "train_calibration_rows": int(arrays["train_current"].shape[0]),
            "model_params": params,
            "status": status,
            "error": error,
            "elapsed_seconds": time.time() - started,
        },
    )
    return {"config": config_name, "target": target, "status": status, "pareto_csv": str(csv_path), "error": error}


def candidate_indices_by_score(df: pd.DataFrame) -> list[int]:
    table = df.copy()
    table["complexity_num"] = pd.to_numeric(table.get("complexity"), errors="coerce").fillna(np.inf)
    table["loss_num"] = pd.to_numeric(table.get("loss"), errors="coerce").fillna(np.inf)
    table["score_num"] = pd.to_numeric(table.get("score"), errors="coerce").fillna(-np.inf)
    table = table[table["complexity_num"] <= np.inf]
    return table.sort_values(["score_num", "loss_num", "complexity_num"], ascending=[False, True, True]).index.astype(int).tolist()


def select_equation_for_target(arrays: dict[str, Any], args: argparse.Namespace, config_name: str, target: str) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, np.ndarray] | None]:
    csv_path = pareto_dir(args, config_name, target) / "pareto_front_scored.csv"
    if not csv_path.exists():
        return None, {"config": config_name, "target": target, "reason": "missing_pareto_front"}, None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None, {"config": config_name, "target": target, "reason": "empty_pareto_front"}, None

    fit_idx = arrays["fit_idx"]
    holdout_idx = arrays["holdout_idx"]
    _, y_all = target_values(arrays, target, split="train", target_mode=args.target_mode)
    baseline_holdout_mae = float(np.mean(np.abs(y_all[holdout_idx])))
    reasons = []
    passing: list[tuple[float, float, float, int, dict[str, Any], dict[str, np.ndarray]]] = []
    for idx in candidate_indices_by_score(df):
        row = df.loc[idx]
        complexity = safe_float(row.get("complexity"))
        if not np.isfinite(complexity) or complexity > float(args.max_complexity):
            reasons.append(f"row={idx}:complexity>{args.max_complexity}")
            continue
        equation = str(row.get("sympy_format", row.get("equation", "")))
        pred_train_target = evaluate_equation(equation, arrays["feature_columns"], arrays["train_current"])
        if not np.isfinite(pred_train_target).all():
            reasons.append(f"row={idx}:nonfinite_train_predictions")
            continue
        residual_train = detection_residual(arrays, target, equation, split="train", target_mode=args.target_mode)
        residual_fit = residual_train[fit_idx]
        residual_holdout = residual_train[holdout_idx]
        holdout_pred = pred_train_target[holdout_idx]
        holdout_metrics = regression_metrics(y_all[holdout_idx], holdout_pred)
        improves_baseline = holdout_metrics["mae"] < baseline_holdout_mae
        passes_r2 = holdout_metrics["r2"] >= float(args.selection_holdout_r2)
        if not (passes_r2 or improves_baseline):
            reasons.append(f"row={idx}:holdout_quality_fail")
            continue
        params = fit_cusum_params(residual_fit, s=float(args.default_s), g=float(args.default_g))
        _, holdout_alarm = run_cusum(residual_holdout, params)
        if int(np.sum(holdout_alarm)) > 0:
            reasons.append(f"row={idx}:holdout_cusum_alarm")
            continue
        median = float(np.median(residual_train))
        p99 = float(np.percentile(residual_train, 99))
        tail_ratio = p99 / max(median, float(args.tail_median_floor))
        if tail_ratio > float(args.residual_tail_ratio):
            reasons.append(f"row={idx}:tail_ratio>{args.residual_tail_ratio}:{tail_ratio:.4g}")
            continue
        residual_test = detection_residual(arrays, target, equation, split="test", target_mode=args.target_mode)
        selected = {
            "config": config_name,
            "target": target,
            "variable_type": "actuator" if is_actuator(target) else "sensor",
            "target_mode": args.target_mode,
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
            "selection_reason": "highest_score_among_stable_candidates",
            "pareto_csv": str(csv_path),
        }
        residuals = {"train": residual_train, "test": residual_test}
        score = safe_float(row.get("score"))
        loss = safe_float(row.get("loss"))
        passing.append((score, -loss if np.isfinite(loss) else -np.inf, -complexity, int(idx), selected, residuals))
    if passing:
        passing.sort(reverse=True, key=lambda item: item[:4])
        selected = passing[0][4]
        residuals = passing[0][5]
        return selected, None, residuals
    return None, {"config": config_name, "target": target, "reason": "; ".join(reasons[-8:])}, None


def alarm_burden(alarms: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    alarm = np.asarray(alarms, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64) >= 1
    intervals = to_intervals(alarm)
    return {
        "total_alarm_rate": float(np.mean(alarm)),
        "benign_alarm_rate": float(np.mean(alarm[~y])) if np.any(~y) else 0.0,
        "attack_alarm_rate": float(np.mean(alarm[y])) if np.any(y) else 0.0,
        "num_alarm_intervals": int(len(intervals)),
        "longest_alarm_interval": int(max((end - start + 1 for start, end in intervals), default=0)),
    }


def point_counts(labels: np.ndarray, alarms: np.ndarray) -> dict[str, int]:
    y = (np.asarray(labels) >= 1).astype(np.int64)
    a = (np.asarray(alarms) >= 1).astype(np.int64)
    return {
        "TP": int(np.sum((y == 1) & (a == 1))),
        "FP": int(np.sum((y == 0) & (a == 1))),
        "TN": int(np.sum((y == 0) & (a == 0))),
        "FN": int(np.sum((y == 1) & (a == 0))),
    }


def run_detection_grid(args: argparse.Namespace, selected_rows: list[dict[str, Any]], residual_cache: dict[tuple[str, str], dict[str, np.ndarray]], labels: np.ndarray) -> tuple[pd.DataFrame, pd.DataFrame]:
    grid_rows = []
    per_sensor_rows = []
    by_config: dict[str, list[dict[str, Any]]] = {}
    for row in selected_rows:
        by_config.setdefault(str(row["config"]), []).append(row)

    for config_name, rows in by_config.items():
        for s in parse_float_list(args.s_values, default=[1.0, 1.42, 2.0, 3.0]):
            for g in parse_float_list(args.g_values, default=[2.0, 5.98, 9.0]):
                alarms = []
                for row in rows:
                    key = (str(row["config"]), str(row["target"]))
                    residuals = residual_cache[key]
                    params = fit_cusum_params(residuals["train"], s=s, g=g)
                    cusum, alarm = run_cusum(residuals["test"], params)
                    alarms.append(alarm.astype(np.int8))
                    burden = alarm_burden(alarm, labels)
                    per_sensor_rows.append(
                        {
                            **{k: row[k] for k in ["config", "target", "complexity", "equation", "holdout_r2", "holdout_mae", "baseline_holdout_mae", "residual_tail_ratio"]},
                            "S": s,
                            "G": g,
                            "delta": params.delta,
                            "threshold": params.threshold,
                            "growth_cap": params.growth_cap,
                            "max_train_cusum": params.max_calib_cusum,
                            "max_test_cusum": float(np.max(cusum)) if cusum.size else 0.0,
                            **burden,
                        }
                    )
                system_alarm = np.max(np.stack(alarms, axis=1), axis=1).astype(np.int8) if alarms else np.zeros_like(labels, dtype=np.int8)
                metrics = compute_detection_metrics(labels, system_alarm, expand_steps=60)
                config = CONFIGS[config_name]
                grid_rows.append(
                    {
                        "config": config_name,
                        "sample_policy": config["sample_policy"],
                        "sample_size": config["sample_size"],
                        "operator_set": config["operator_set"],
                        "scope": config["scope"],
                        "S": s,
                        "G": g,
                        "num_monitored": len(rows),
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
    return pd.DataFrame(grid_rows), pd.DataFrame(per_sensor_rows)


def write_best_configs(grid: pd.DataFrame, out_root: Path) -> pd.DataFrame:
    rows = []
    eligible_f1_5 = grid[grid["FPA"] <= 5].sort_values(["F1", "eTaF1"], ascending=False)
    eligible_f1 = grid[grid["FPA"] <= 15].sort_values(["F1", "eTaF1"], ascending=False)
    eligible_eta = grid[grid["FPA"] <= 15].sort_values(["eTaF1", "F1"], ascending=False)
    geco = grid[(grid["S"] == 1.42) & (grid["G"] == 5.98)].sort_values(["F1", "eTaF1"], ascending=False)
    for name, table in [
        ("best_f1_fpa_le_5", eligible_f1_5),
        ("best_f1_fpa_le_15", eligible_f1),
        ("best_etaf1_fpa_le_15", eligible_eta),
        ("best_geco_s_g", geco),
        ("lowest_fpa", grid.sort_values(["FPA", "F1", "eTaF1"], ascending=[True, False, False])),
        ("best_f1_overall", grid.sort_values(["F1", "eTaF1"], ascending=False)),
    ]:
        if not table.empty:
            row = table.iloc[0].to_dict()
            row["selection"] = name
            rows.append(row)
    best = pd.DataFrame(rows)
    best.to_csv(out_root / "best_configs.csv", index=False)
    return best


def alarms_for_config(
    selected_rows: list[dict[str, Any]],
    residual_cache: dict[tuple[str, str], dict[str, np.ndarray]],
    *,
    config_name: str,
    s: float,
    g: float,
    labels: np.ndarray,
) -> tuple[dict[str, np.ndarray], np.ndarray]:
    per_sensor: dict[str, np.ndarray] = {}
    alarms = []
    for row in selected_rows:
        if str(row["config"]) != str(config_name):
            continue
        key = (str(row["config"]), str(row["target"]))
        residuals = residual_cache[key]
        params = fit_cusum_params(residuals["train"], s=float(s), g=float(g))
        _, alarm = run_cusum(residuals["test"], params)
        alarm = alarm.astype(np.int8)
        per_sensor[str(row["target"])] = alarm
        alarms.append(alarm)
    system_alarm = np.max(np.stack(alarms, axis=1), axis=1).astype(np.int8) if alarms else np.zeros_like(labels, dtype=np.int8)
    return per_sensor, system_alarm


def first_alarm_delay(alarm: np.ndarray, start_idx: int, end_idx: int, original_start: int) -> int | None:
    segment = alarm[start_idx : end_idx + 1]
    hits = np.flatnonzero(segment)
    if hits.size == 0:
        return None
    return max(0, int(start_idx + int(hits[0]) + 1) - int(original_start))


def write_per_attack_analysis(out: Path, selected_rows: list[dict[str, Any]], per_sensor_alarms: dict[str, np.ndarray]) -> pd.DataFrame:
    from ics_symbolic_distill.data.ics_metadata import get_attack_windows

    monitored = {str(row["target"]) for row in selected_rows}
    if not per_sensor_alarms:
        table = pd.DataFrame()
        table.to_csv(out, index=False)
        return table
    n = int(next(iter(per_sensor_alarms.values())).shape[0])
    rows = []
    for attack_id, window in enumerate(get_attack_windows("SWAT"), start=1):
        original_start = int(window.start)
        original_end = int(window.end)
        start_alarm = max(0, original_start - 1)
        end_alarm = min(n - 1, original_end - 1)
        affected = [str(tag) for tag in window.affected_tags]
        in_scope = [tag for tag in affected if tag in monitored]
        firing = [
            target
            for target, alarm in per_sensor_alarms.items()
            if np.any(alarm[start_alarm : end_alarm + 1] == 1)
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
            first_alarm_delay(per_sensor_alarms[tag], start_alarm, end_alarm, original_start)
            for tag in firing
            if tag in per_sensor_alarms
        ]
        delays = [d for d in delays if d is not None]
        rows.append(
            {
                "attack_id": attack_id,
                "start": original_start,
                "end": original_end,
                "affected_tags": ",".join(affected),
                "in_scope_sensors": ",".join(in_scope),
                "category": category,
                "firing_sensors": ",".join(firing),
                "detection_delay_seconds": min(delays) if delays else None,
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(out, index=False)
    return table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Local SWaT 1-second delta symbolic diagnostic.")
    parser.add_argument("--experiment", default="configs/experiment/swat_mlp_current_val20.yaml")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--out", default="artifacts/swat_1sec/delta_local_diagnostic")
    parser.add_argument("--targets", default=",".join(DIAGNOSTIC_TARGETS))
    parser.add_argument("--configs", default="A,B,C")
    parser.add_argument("--target-mode", default="sensors_delta_actuators_next", choices=["sensors_delta_actuators_next"])
    parser.add_argument("--niterations", type=int, default=400)
    parser.add_argument("--timeout-minutes", type=float, default=20.0)
    parser.add_argument("--max-complexity", type=int, default=12)
    parser.add_argument("--parallel-jobs", type=int, default=max(1, min(2, (os.cpu_count() or 2) // 4)))
    parser.add_argument("--target-parallel-jobs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--default-s", type=float, default=1.42)
    parser.add_argument("--default-g", type=float, default=5.98)
    parser.add_argument("--selection-holdout-r2", type=float, default=0.0)
    parser.add_argument("--residual-tail-ratio", type=float, default=100.0)
    parser.add_argument("--tail-median-floor", type=float, default=1e-9)
    parser.add_argument("--s-values", default="1.0,1.42,2.0,3.0")
    parser.add_argument("--g-values", default="2.0,5.98,9.0")
    parser.add_argument("--flat-pareto-layout", action="store_true")
    parser.add_argument("--force", action="store_true")
    return parser.parse_args()


def parse_targets(value: str) -> list[str]:
    requested = str(value).strip()
    if requested in {"all-sensors", "all_sensor", "sensors"}:
        return list(SWAT_SENSOR_TARGETS)
    return [tok.strip() for tok in requested.split(",") if tok.strip()]


def main() -> int:
    args = parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    arrays = load_arrays(args)
    targets = parse_targets(args.targets)
    configs = [tok.strip() for tok in str(args.configs).split(",") if tok.strip()]
    configs = [name for name in configs if name in CONFIGS]
    if not configs:
        raise ValueError("No valid configs requested")

    write_json(
        out_root / "run_config.json",
        {
            "targets": targets,
            "configs": {name: CONFIGS[name] for name in configs},
            "target_mode": args.target_mode,
            "niterations": int(args.niterations),
            "timeout_minutes": float(args.timeout_minutes),
            "max_complexity": int(args.max_complexity),
            "parallel_jobs": int(args.parallel_jobs),
            "target_parallel_jobs": int(args.target_parallel_jobs),
            "selection_holdout_r2": float(args.selection_holdout_r2),
            "residual_tail_ratio": float(args.residual_tail_ratio),
            "data": arrays["metadata"],
        },
    )

    run_rows = []
    for config_name in configs:
        config = CONFIGS[config_name]
        print(f"=== PySR config {config_name}: {config} ===", flush=True)
        eligible_targets = [
            target
            for target in targets
            if not (config["scope"] == "sensors-only" and is_actuator(target))
        ]
        target_workers = max(1, int(args.target_parallel_jobs))
        if target_workers == 1:
            for target in eligible_targets:
                print(f"[{config_name}] target={target}", flush=True)
                run_rows.append(run_pysr_target(arrays, args, config_name, config, target))
        else:
            print(f"[{config_name}] running {len(eligible_targets)} targets with {target_workers} target workers", flush=True)
            with ProcessPoolExecutor(max_workers=target_workers) as pool:
                future_map = {
                    pool.submit(run_pysr_target, arrays, args, config_name, config, target): target
                    for target in eligible_targets
                }
                for future in as_completed(future_map):
                    target = future_map[future]
                    try:
                        result = future.result()
                    except Exception as exc:
                        result = {"config": config_name, "target": target, "status": "failed", "error": str(exc)}
                    print(f"[{config_name}] done target={target} status={result.get('status')}", flush=True)
                    run_rows.append(result)
    pd.DataFrame(run_rows).to_csv(out_root / "pysr_run_status.csv", index=False)

    sample_audits = []
    for config_name in configs:
        for target in targets:
            audit_path = pareto_dir(args, config_name, target) / "sample_audit.csv"
            if audit_path.exists():
                sample_audits.append(pd.read_csv(audit_path))
    if sample_audits:
        pd.concat(sample_audits, ignore_index=True).to_csv(out_root / "sample_audit.csv", index=False)

    selected_rows = []
    exclusion_rows = []
    residual_cache: dict[tuple[str, str], dict[str, np.ndarray]] = {}
    for config_name in configs:
        for target in targets:
            selected, exclusion, residuals = select_equation_for_target(arrays, args, config_name, target)
            if selected is None:
                if exclusion is not None:
                    exclusion_rows.append(exclusion)
                continue
            selected_rows.append(selected)
            residual_cache[(config_name, target)] = residuals or {}

    selected_df = pd.DataFrame(selected_rows)
    exclusion_df = pd.DataFrame(exclusion_rows)
    selected_df.to_csv(out_root / "selected_equations.csv", index=False)
    exclusion_df.to_csv(out_root / "exclusion_reasons.csv", index=False)

    grid, per_sensor = run_detection_grid(args, selected_rows, residual_cache, arrays["labels"])
    grid.to_csv(out_root / "diagnostic_grid.csv", index=False)
    grid.to_csv(out_root / "detection_grid.csv", index=False)
    per_sensor.to_csv(out_root / "per_sensor_alarm_stats.csv", index=False)
    best = write_best_configs(grid, out_root)

    per_attack_choice = best[best["selection"] == "best_f1_fpa_le_15"]
    if per_attack_choice.empty:
        per_attack_choice = best[best["selection"] == "lowest_fpa"]
    if not per_attack_choice.empty:
        best_row = per_attack_choice.iloc[0]
        per_sensor_alarms, system_alarm = alarms_for_config(
            selected_rows,
            residual_cache,
            config_name=str(best_row["config"]),
            s=float(best_row["S"]),
            g=float(best_row["G"]),
            labels=arrays["labels"],
        )
        np.save(out_root / "system_alarms.npy", system_alarm)
        np.savez_compressed(out_root / "per_sensor_alarms.npz", **per_sensor_alarms)
        selected_for_best = [row for row in selected_rows if str(row["config"]) == str(best_row["config"])]
        write_per_attack_analysis(out_root / "per_attack_analysis.csv", selected_for_best, per_sensor_alarms)
        write_json(
            out_root / "per_attack_config.json",
            {
                "selection": str(best_row["selection"]),
                "note": "Used best_f1_fpa_le_15 when available; otherwise lowest_fpa fallback.",
                "config": str(best_row["config"]),
                "S": float(best_row["S"]),
                "G": float(best_row["G"]),
                "FPA": float(best_row["FPA"]),
                "F1": float(best_row["F1"]),
                "eTaF1": float(best_row["eTaF1"]),
                "Scen": float(best_row["Scen"]),
            },
        )

    print("\n=== Delta Local Diagnostic Results ===")
    if not selected_df.empty:
        print("\nSelected equations:")
        display_cols = ["config", "target", "complexity", "equation", "loss", "holdout_r2", "residual_tail_ratio", "selection_reason"]
        print(selected_df[display_cols].to_string(index=False))
    print("Best configs:")
    print(best[["selection", "config", "sample_policy", "sample_size", "operator_set", "S", "G", "num_monitored", "Precision", "Recall", "F1", "eTaP", "eTaR", "eTaF1", "FPA", "Scen"]].to_string(index=False) if not best.empty else "(none)")
    print("\nBad sensors by benign alarm intervals:")
    if not per_sensor.empty:
        print(per_sensor.sort_values(["num_alarm_intervals", "benign_alarm_rate"], ascending=False).head(20)[["config", "target", "S", "G", "num_alarm_intervals", "benign_alarm_rate", "attack_alarm_rate", "FPA" if "FPA" in per_sensor.columns else "threshold"]].to_string(index=False))
    print("\nCurrent actual-next comparison anchors:")
    print("  division-safe sensors+actuators S=3.0 G=9.0: F1=30.7 eTaF1=45.7 FPA=8 Scen=71.9")
    print("  identity-fallback sensors-only S=1.0 G=2.0: F1=81.6 eTaF1=41.8 FPA=169 Scen=65.6")
    print("  AIT203-excluded check: F1=81.0 eTaF1=54.2 FPA=21 Scen=68.8")
    print(f"\nWrote {out_root / 'diagnostic_grid.csv'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
