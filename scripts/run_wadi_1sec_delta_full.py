#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import re
import subprocess
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

from ics_symbolic_distill.data.ics_metadata import get_attack_windows, is_actuator, normalize_attack_labels
from ics_symbolic_distill.detection import compute_detection_metrics, evaluate_equation, fit_cusum_params, run_cusum
from ics_symbolic_distill.detection.metrics import to_intervals
from ics_symbolic_distill.detection.swat1s_delta_sampling import coverage_stratified_indices, reconstruct_next_from_delta


TARGET_MODE = "sensors_delta_actuators_next"
S_VALUES = [1.0, 1.32, 2.0, 3.0, 5.0]
G_VALUES = [2.0, 5.98, 9.74, 15.0, 25.0]
GECO_S = 1.32
GECO_G = 9.74
SELECTED_EQUATION_COLUMNS = [
    "target",
    "variable_type",
    "target_mode",
    "equation",
    "sympy_format",
    "complexity",
    "loss",
    "score",
    "holdout_r2",
    "holdout_mae",
    "baseline_holdout_mae",
    "residual_tail_ratio",
    "residual_p99",
    "residual_median",
    "selection_reason",
    "pareto_csv",
]


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


def with_target_first_column(table: pd.DataFrame, target: str) -> pd.DataFrame:
    out = table.copy()
    if "target" in out.columns:
        out = out.drop(columns=["target"])
    out.insert(0, "target", target)
    return out


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


def make_variable_name_mapping(feature_columns: list[str]) -> tuple[dict[str, str], dict[str, str], list[str]]:
    original_to_safe = {str(col): f"V{i}" for i, col in enumerate(feature_columns)}
    safe_to_original = {safe: original for original, safe in original_to_safe.items()}
    safe_feature_columns = [original_to_safe[col] for col in feature_columns]
    return original_to_safe, safe_to_original, safe_feature_columns


def _replace_symbols(text: Any, mapping: dict[str, str]) -> str:
    out = str(text)
    for src, dst in sorted(mapping.items(), key=lambda item: len(item[0]), reverse=True):
        out = re.sub(rf"(?<![A-Za-z0-9_]){re.escape(src)}(?![A-Za-z0-9_])", dst, out)
    return out


def equation_safe_to_original(equation: Any, safe_to_original: dict[str, str]) -> str:
    return _replace_symbols(equation, safe_to_original)


def equation_original_to_safe(equation: Any, original_to_safe: dict[str, str]) -> str:
    return _replace_symbols(equation, original_to_safe)


def write_variable_name_mapping(out_root: Path, arrays: dict[str, Any]) -> None:
    rows = [
        {"original_name": original, "safe_name": safe}
        for original, safe in arrays["original_to_safe"].items()
    ]
    pd.DataFrame(rows).to_csv(out_root / "variable_name_mapping.csv", index=False)


def load_wadi_1sec_arrays(args: argparse.Namespace) -> dict[str, Any]:
    train_path = Path(args.train_csv)
    test_path = Path(args.test_csv)
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"WADI CSVs not found: train={train_path} test={test_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_labels = train_df["Attack"].copy() if "Attack" in train_df.columns else None
    test_labels = test_df["Attack"].copy() if "Attack" in test_df.columns else None
    drop = {c for c in ["Row", "Attack"] if c in train_df.columns}
    train_features = train_df.drop(columns=sorted(drop))
    test_features = test_df.drop(columns=[c for c in sorted(drop) if c in test_df.columns])
    numeric_cols = train_features.select_dtypes(include=[np.number]).columns.tolist()
    numeric_cols = [c for c in numeric_cols if c in test_features.columns]
    if not numeric_cols:
        raise ValueError("No shared numeric WADI process variables found")
    train_sel = train_features[numeric_cols]
    test_sel = test_features[numeric_cols]
    keep_mask = train_sel.notna().any(axis=0)
    if not bool(np.all(keep_mask.to_numpy())):
        numeric_cols = train_sel.columns[keep_mask].tolist()
        train_sel = train_sel[numeric_cols]
        test_sel = test_sel[numeric_cols]
    if train_sel.isna().any().any() or test_sel.isna().any().any():
        med = train_sel.median(axis=0, numeric_only=True)
        train_sel = train_sel.fillna(med).fillna(0.0)
        test_sel = test_sel.fillna(med).fillna(0.0)
    train = train_sel.to_numpy(dtype=np.float32)
    test = test_sel.to_numpy(dtype=np.float32)
    labels = normalize_attack_labels(test_labels)
    if labels is None:
        labels = labels_from_wadi_windows(test.shape[0])
    labels = labels.astype(np.int64)
    train_current = train[:-1].astype(np.float32, copy=False)
    train_next = train[1:].astype(np.float32, copy=False)
    test_current = test[:-1].astype(np.float32, copy=False)
    test_next = test[1:].astype(np.float32, copy=False)
    n_pairs = int(train_current.shape[0])
    cutoff = int(math.floor(n_pairs * 0.8))
    feature_columns = [str(c) for c in train_sel.columns]
    original_to_safe, safe_to_original, safe_feature_columns = make_variable_name_mapping(feature_columns)
    sensor_names = [c for c in feature_columns if not is_actuator("WADI", c)]
    actuator_names = [c for c in feature_columns if is_actuator("WADI", c)]
    return {
        "train": train,
        "test": test,
        "train_current": train_current,
        "train_next": train_next,
        "test_current": test_current,
        "test_next": test_next,
        "labels": labels[1:].astype(np.int64),
        "feature_columns": feature_columns,
        "safe_feature_columns": safe_feature_columns,
        "original_to_safe": original_to_safe,
        "safe_to_original": safe_to_original,
        "sensor_names": sensor_names,
        "actuator_names": actuator_names,
        "fit_idx": np.arange(cutoff, dtype=np.int64),
        "holdout_idx": np.arange(cutoff, n_pairs, dtype=np.int64),
        "metadata": {
            "train_csv": str(train_path),
            "test_csv": str(test_path),
            "train_rows": int(train.shape[0]),
            "test_rows": int(test.shape[0]),
            "num_features": len(feature_columns),
            "num_sensors": len(sensor_names),
            "num_actuators": len(actuator_names),
        },
    }


def labels_from_wadi_windows(length: int) -> np.ndarray:
    labels = np.zeros(int(length), dtype=np.int64)
    for window in get_attack_windows("WADI"):
        start = max(0, int(window.start))
        end = min(int(length), int(window.end))
        if start < end:
            labels[start:end] = 1
    return labels


def target_values(arrays: dict[str, Any], target: str, *, split: str) -> tuple[np.ndarray, np.ndarray]:
    idx = arrays["feature_columns"].index(target)
    if split == "train":
        current = arrays["train_current"]
        nxt = arrays["train_next"]
    elif split == "test":
        current = arrays["test_current"]
        nxt = arrays["test_next"]
    else:
        raise ValueError(split)
    if is_actuator("WADI", target):
        y = nxt[:, idx]
    else:
        y = nxt[:, idx] - current[:, idx]
    return current, y.astype(np.float64)


def prediction_residual(arrays: dict[str, Any], target: str, equation: str, *, split: str) -> np.ndarray:
    idx = arrays["feature_columns"].index(target)
    if split == "train":
        current = arrays["train_current"]
        nxt = arrays["train_next"]
    else:
        current = arrays["test_current"]
        nxt = arrays["test_next"]
    equation_safe = equation_original_to_safe(str(equation), arrays["original_to_safe"])
    pred = evaluate_equation(equation_safe, arrays["safe_feature_columns"], current)
    if is_actuator("WADI", target):
        pred_next = pred
    else:
        pred_next = reconstruct_next_from_delta(current[:, idx], pred)
    residual = np.abs(nxt[:, idx].astype(np.float64) - pred_next.astype(np.float64))
    return np.where(np.isfinite(residual), residual, 0.0).astype(np.float64)


def sample_indices_for_target(arrays: dict[str, Any], target: str, sample_size: int) -> tuple[np.ndarray, dict[str, Any]]:
    fit_idx = arrays["fit_idx"]
    current, y_all = target_values(arrays, target, split="train")
    y_fit = y_all[fit_idx]
    actuator_indices = [arrays["feature_columns"].index(name) for name in arrays["actuator_names"]]
    local_idx, audit = coverage_stratified_indices(
        target=target,
        y_delta_fit_pool=y_fit,
        x_current_fit_pool=current[fit_idx],
        x_next_fit_pool=arrays["train_next"][fit_idx],
        actuator_indices=actuator_indices,
        sample_size=int(sample_size),
        seed=1337,
    )
    return fit_idx[local_idx], audit.to_dict()


def operator_params(*, niterations: int, timeout_minutes: float, max_complexity: int, procs: int, seed: int) -> dict[str, Any]:
    return {
        "niterations": int(niterations),
        "binary_operators": ["+", "-", "*"],
        "unary_operators": [],
        "extra_sympy_mappings": {},
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


def make_model(**kwargs: Any):
    from pysr import PySRRegressor

    params = operator_params(**kwargs)
    supported = DAY1._filter_supported_params(PySRRegressor, params)
    return PySRRegressor(**supported), supported


def stable_target_seed(target: str, base_seed: int) -> int:
    """Return a deterministic target-specific seed independent of hash randomization."""

    offset = sum((i + 1) * ord(ch) for i, ch in enumerate(str(target))) % 100000
    return int(base_seed) + int(offset)


def features_in_equation(equation: str, feature_names: list[str]) -> list[str]:
    return DAY1.features_in_equation(str(equation), feature_names)


def evaluate_pareto_df(
    model: Any,
    equations: pd.DataFrame,
    x_sample: pd.DataFrame,
    y_sample: np.ndarray,
    x_holdout: pd.DataFrame,
    y_holdout: np.ndarray,
    safe_feature_columns: list[str],
    safe_to_original: dict[str, str],
) -> pd.DataFrame:
    rows = []
    for equation_index, row in equations.reset_index(drop=True).iterrows():
        out = row.to_dict()
        equation_safe = str(row.get("equation", ""))
        sympy_safe = str(row.get("sympy_format", row.get("equation", "")))
        out["equation"] = equation_safe_to_original(equation_safe, safe_to_original)
        out["sympy_format"] = equation_safe_to_original(sympy_safe, safe_to_original)
        if "lambda_format" in out:
            out["lambda_format"] = equation_safe_to_original(out["lambda_format"], safe_to_original)
        try:
            pred_sample = np.asarray(model.predict(x_sample, index=int(equation_index)), dtype=np.float64)
            pred_holdout = np.asarray(model.predict(x_holdout, index=int(equation_index)), dtype=np.float64)
            sample = regression_metrics(y_sample, pred_sample)
            holdout = regression_metrics(y_holdout, pred_holdout)
            features_original = [
                safe_to_original.get(feature, feature)
                for feature in features_in_equation(equation_safe, safe_feature_columns)
            ]
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
                    "equation_features": json.dumps(features_original),
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


def pareto_dir(out: Path, target: str) -> Path:
    return out / "pareto_fronts" / f"{target}_{TARGET_MODE}"


def run_pysr_target(args_dict: dict[str, Any], target: str) -> dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    arrays = load_wadi_1sec_arrays(args)
    out_root = Path(args.out)
    out_dir = pareto_dir(out_root, target)
    csv_path = out_dir / "pareto_front_scored.csv"
    if csv_path.exists() and not args.force:
        return {"target": target, "status": "skipped_existing", "pareto_csv": str(csv_path)}
    started = time.time()
    status = "completed"
    error = ""
    try:
        feature_columns = arrays["feature_columns"]
        safe_feature_columns = arrays["safe_feature_columns"]
        current, y_all = target_values(arrays, target, split="train")
        fit_idx = arrays["fit_idx"]
        holdout_idx = arrays["holdout_idx"]
        sample_idx, audit = sample_indices_for_target(arrays, target, int(args.sample_size))
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "sample_indices.npy", sample_idx.astype(np.int64))
        pd.DataFrame([audit]).to_csv(out_dir / "sample_audit.csv", index=False)
        x_sample = pd.DataFrame(current[sample_idx], columns=safe_feature_columns)
        y_sample = y_all[sample_idx]
        x_holdout = pd.DataFrame(current[holdout_idx], columns=safe_feature_columns)
        y_holdout = y_all[holdout_idx]
        model, params = make_model(
            niterations=int(args.niterations),
            timeout_minutes=float(args.timeout_minutes),
            max_complexity=int(args.max_complexity),
            procs=int(args.pysr_procs),
            seed=stable_target_seed(target, int(args.seed)),
        )
        model.fit(x_sample, y_sample)
        pareto = evaluate_pareto_df(
            model,
            model.equations_.copy(),
            x_sample,
            y_sample,
            x_holdout,
            y_holdout,
            safe_feature_columns,
            arrays["safe_to_original"],
        )
        pareto.to_csv(csv_path, index=False)
    except Exception as exc:
        status = "failed"
        error = str(exc)
    write_json(
        out_dir / "metadata.json",
        {
            "target": target,
            "dataset": "WADI",
            "target_mode": TARGET_MODE,
            "sample_policy": "coverage_stratified",
            "sample_size": int(args.sample_size),
            "operator_set": "safe_mul",
            "max_complexity": int(args.max_complexity),
            "timeout_minutes": float(args.timeout_minutes),
            "status": status,
            "error": error,
            "elapsed_seconds": time.time() - started,
        },
    )
    return {"target": target, "status": status, "pareto_csv": str(csv_path), "error": error}


def candidate_indices_by_score(df: pd.DataFrame) -> list[int]:
    table = df.copy()
    table["complexity_num"] = pd.to_numeric(table.get("complexity"), errors="coerce").fillna(np.inf)
    table["loss_num"] = pd.to_numeric(table.get("loss"), errors="coerce").fillna(np.inf)
    table["score_num"] = pd.to_numeric(table.get("score"), errors="coerce").fillna(-np.inf)
    return table.sort_values(["score_num", "loss_num", "complexity_num"], ascending=[False, True, True]).index.astype(int).tolist()


def select_equation(arrays: dict[str, Any], out_root: Path, target: str, max_complexity: int) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, np.ndarray] | None]:
    csv_path = pareto_dir(out_root, target) / "pareto_front_scored.csv"
    if not csv_path.exists():
        return None, {"target": target, "reason": "missing_pareto_front"}, None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None, {"target": target, "reason": "empty_pareto_front"}, None
    fit_idx = arrays["fit_idx"]
    holdout_idx = arrays["holdout_idx"]
    _, y_all = target_values(arrays, target, split="train")
    baseline_holdout_mae = float(np.mean(np.abs(y_all[holdout_idx])))
    passing = []
    reasons = []
    for idx in candidate_indices_by_score(df):
        row = df.loc[idx]
        complexity = safe_float(row.get("complexity"))
        if not np.isfinite(complexity) or complexity > float(max_complexity):
            reasons.append(f"row={idx}:complexity>{max_complexity}")
            continue
        equation = str(row.get("sympy_format", row.get("equation", "")))
        equation_safe = equation_original_to_safe(equation, arrays["original_to_safe"])
        pred_train_target = evaluate_equation(equation_safe, arrays["safe_feature_columns"], arrays["train_current"])
        if not np.isfinite(pred_train_target).all():
            reasons.append(f"row={idx}:nonfinite_train_predictions")
            continue
        residual_train = prediction_residual(arrays, target, equation, split="train")
        residual_fit = residual_train[fit_idx]
        residual_holdout = residual_train[holdout_idx]
        params = fit_cusum_params(residual_fit, s=GECO_S, g=GECO_G)
        _, holdout_alarm = run_cusum(residual_holdout, params)
        if int(np.sum(holdout_alarm)) > 0:
            reasons.append(f"row={idx}:holdout_cusum_alarm")
            continue
        median = float(np.median(residual_train))
        p99 = float(np.percentile(residual_train, 99))
        tail_ratio = p99 / max(median, 1e-9)
        if tail_ratio > 50.0:
            reasons.append(f"row={idx}:tail_ratio>50:{tail_ratio:.4g}")
            continue
        holdout_metrics = regression_metrics(y_all[holdout_idx], pred_train_target[holdout_idx])
        residual_test = prediction_residual(arrays, target, equation, split="test")
        selected = {
            "target": target,
            "variable_type": "actuator" if is_actuator("WADI", target) else "sensor",
            "target_mode": TARGET_MODE,
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
        passing.append((selected["score"], -selected["loss"] if np.isfinite(selected["loss"]) else -np.inf, -complexity, selected, {"train": residual_train, "test": residual_test}))
    if not passing:
        return None, {"target": target, "reason": "; ".join(reasons[-8:])}, None
    passing.sort(reverse=True, key=lambda item: item[:3])
    return passing[0][3], None, passing[0][4]


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


def evaluate_detection(labels: np.ndarray, selected_rows: list[dict[str, Any]], residual_cache: dict[str, dict[str, np.ndarray]], *, variant: str) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[float, float], dict[str, np.ndarray]]]:
    grid_rows = []
    per_sensor_rows = []
    alarm_cache: dict[tuple[float, float], dict[str, np.ndarray]] = {}
    for s in S_VALUES:
        for g in G_VALUES:
            alarms = []
            per_sensor_alarms = {}
            for row in selected_rows:
                target = str(row["target"])
                residuals = residual_cache[target]
                params = fit_cusum_params(residuals["train"], s=float(s), g=float(g))
                cusum, alarm = run_cusum(residuals["test"], params)
                alarm = alarm.astype(np.int8)
                per_sensor_alarms[target] = alarm
                alarms.append(alarm)
                per_sensor_rows.append(
                    {
                        "variant": variant,
                        "target": target,
                        "variable_type": row.get("variable_type", ""),
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
                        **alarm_burden(alarm, labels),
                    }
                )
            system_alarm = np.max(np.stack(alarms, axis=1), axis=1).astype(np.int8) if alarms else np.zeros_like(labels, dtype=np.int8)
            alarm_cache[(float(s), float(g))] = {"system": system_alarm, **per_sensor_alarms}
            metrics = compute_detection_metrics(labels, system_alarm, expand_steps=60)
            grid_rows.append(
                {
                    "variant": variant,
                    "S": s,
                    "G": g,
                    "num_monitored": len(selected_rows),
                    "monitored_sensors": sum(str(r.get("variable_type")) == "sensor" for r in selected_rows),
                    "monitored_actuators": sum(str(r.get("variable_type")) == "actuator" for r in selected_rows),
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
    return pd.DataFrame(grid_rows), pd.DataFrame(per_sensor_rows), alarm_cache



def target_result_from_artifacts(out_root: Path, target: str, *, returncode: int | None = None) -> dict[str, Any]:
    out_dir = pareto_dir(out_root, target)
    csv_path = out_dir / "pareto_front_scored.csv"
    metadata_path = out_dir / "metadata.json"
    result: dict[str, Any] = {
        "target": target,
        "status": "completed" if csv_path.exists() else "failed",
        "pareto_csv": str(csv_path),
        "error": "" if csv_path.exists() else f"missing_pareto_front_after_exit_{returncode}",
    }
    if returncode is not None:
        result["exit_code"] = int(returncode)
    if metadata_path.exists():
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            result["status"] = str(metadata.get("status", result["status"]))
            result["error"] = str(metadata.get("error", result.get("error", "")))
            result["elapsed_seconds"] = metadata.get("elapsed_seconds")
        except Exception as exc:
            result["metadata_error"] = str(exc)
    if csv_path.exists() and result.get("status") == "failed" and not result.get("error"):
        result["status"] = "completed"
    return result


def write_pysr_status(out_root: Path, targets: list[str], status_by_target: dict[str, dict[str, Any]]) -> None:
    rows = [status_by_target[target] for target in targets]
    pd.DataFrame(rows).to_csv(out_root / "pysr_run_status.csv", index=False)


def build_single_target_command(args: argparse.Namespace, target: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--train-csv",
        str(args.train_csv),
        "--test-csv",
        str(args.test_csv),
        "--out",
        str(args.out),
        "--sample-size",
        str(args.sample_size),
        "--timeout-minutes",
        str(args.timeout_minutes),
        "--target-wall-timeout-minutes",
        str(args.target_wall_timeout_minutes),
        "--max-complexity",
        str(args.max_complexity),
        "--niterations",
        str(args.niterations),
        "--target-parallel-jobs",
        "1",
        "--pysr-procs",
        str(args.pysr_procs),
        "--seed",
        str(args.seed),
        "--single-target",
        target,
    ]
    if args.force:
        cmd.append("--force")
    return cmd


def run_pysr_targets_isolated(args: argparse.Namespace, targets: list[str], out_root: Path) -> list[dict[str, Any]]:
    """Run each PySR target in its own Python process.

    Julia/PySR can terminate a worker process abruptly under memory pressure. A
    ProcessPoolExecutor treats that as a broken pool and can strand all pending
    futures. Isolating each target in a subprocess lets the parent continue and
    keeps completed Pareto fronts resume-safe.
    """

    workers = max(1, int(args.target_parallel_jobs))
    status_by_target: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    for target in targets:
        csv_path = pareto_dir(out_root, target) / "pareto_front_scored.csv"
        if csv_path.exists() and not args.force:
            status_by_target[target] = {
                "target": target,
                "status": "skipped_existing",
                "pareto_csv": str(csv_path),
                "error": "",
            }
        else:
            status_by_target[target] = {
                "target": target,
                "status": "pending",
                "pareto_csv": str(csv_path),
                "error": "",
            }
            pending.append(target)
    write_pysr_status(out_root, targets, status_by_target)

    wall_limit_seconds = float(args.target_wall_timeout_minutes) * 60.0
    active: dict[int, tuple[subprocess.Popen[Any], str, float]] = {}
    while pending or active:
        while pending and len(active) < workers:
            target = pending.pop(0)
            cmd = build_single_target_command(args, target)
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT))
            active[int(proc.pid)] = (proc, target, time.time())
            status_by_target[target] = {
                **status_by_target[target],
                "status": "running",
                "pid": int(proc.pid),
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            print(f"[WADI] launched target={target} pid={proc.pid}", flush=True)
            write_pysr_status(out_root, targets, status_by_target)

        time.sleep(5.0)
        for pid, (proc, target, started) in list(active.items()):
            elapsed = time.time() - started
            returncode = proc.poll()
            if returncode is None and wall_limit_seconds > 0 and elapsed > wall_limit_seconds:
                print(f"[WADI] timeout target={target} pid={pid}; terminating isolated child", flush=True)
                proc.terminate()
                try:
                    returncode = proc.wait(timeout=30.0)
                except subprocess.TimeoutExpired:
                    proc.kill()
                    returncode = proc.wait()
                write_json(
                    pareto_dir(out_root, target) / "metadata.json",
                    {
                        "target": target,
                        "dataset": "WADI",
                        "target_mode": TARGET_MODE,
                        "sample_policy": "coverage_stratified",
                        "sample_size": int(args.sample_size),
                        "operator_set": "safe_mul",
                        "max_complexity": int(args.max_complexity),
                        "timeout_minutes": float(args.timeout_minutes),
                        "status": "failed",
                        "error": f"target_wall_timeout>{float(args.target_wall_timeout_minutes):.3f}min",
                        "elapsed_seconds": elapsed,
                    },
                )
            if returncode is None:
                continue
            result = target_result_from_artifacts(out_root, target, returncode=int(returncode))
            result["pid"] = pid
            result["wall_seconds"] = elapsed
            status_by_target[target] = result
            print(
                f"[WADI] done target={target} status={result.get('status')} exit_code={returncode}",
                flush=True,
            )
            write_pysr_status(out_root, targets, status_by_target)
            del active[pid]

    return [status_by_target[target] for target in targets]

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="WADI 1-second delta symbolic PySR and detection.")
    parser.add_argument("--train-csv", default="data/wadi/raw/wadi_train.csv")
    parser.add_argument("--test-csv", default="data/wadi/raw/wadi_test.csv")
    parser.add_argument("--out", default="artifacts/wadi_1sec/delta_full")
    parser.add_argument("--sample-size", type=int, default=20000)
    parser.add_argument("--timeout-minutes", type=float, default=60.0)
    parser.add_argument("--target-wall-timeout-minutes", type=float, default=75.0)
    parser.add_argument("--max-complexity", type=int, default=15)
    parser.add_argument("--niterations", type=int, default=400)
    parser.add_argument("--target-parallel-jobs", type=int, default=2)
    parser.add_argument("--pysr-procs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-pysr", action="store_true")
    parser.add_argument("--single-target", default=None, help="Run one PySR target in an isolated child process and exit.")
    return parser.parse_args(argv)


def main() -> int:
    args = parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    if args.single_target:
        result = run_pysr_target(vars(args), str(args.single_target))
        print(f"[WADI child] target={result.get('target')} status={result.get('status')} error={result.get('error', '')}", flush=True)
        return 0 if result.get("status") in {"completed", "skipped_existing"} else 2

    arrays = load_wadi_1sec_arrays(args)
    write_variable_name_mapping(out_root, arrays)
    targets = list(arrays["sensor_names"])
    write_json(
        out_root / "run_config.json",
        {
            "dataset": "WADI",
            "target_mode": TARGET_MODE,
            "sample_policy": "coverage_stratified",
            "sample_size": int(args.sample_size),
            "operator_set": "safe_mul",
            "niterations": int(args.niterations),
            "timeout_minutes": float(args.timeout_minutes),
            "target_wall_timeout_minutes": float(args.target_wall_timeout_minutes),
            "max_complexity": int(args.max_complexity),
            "target_parallel_jobs": int(args.target_parallel_jobs),
            "pysr_procs": int(args.pysr_procs),
            "geco_s": GECO_S,
            "geco_g": GECO_G,
            "variable_name_mapping": str(out_root / "variable_name_mapping.csv"),
            "data": arrays["metadata"],
        },
    )
    print("=== WADI 1-second data ===", flush=True)
    print(arrays["metadata"], flush=True)
    print(f"PySR sensor targets: {len(targets)}", flush=True)

    if not args.skip_pysr:
        run_pysr_targets_isolated(args, targets, out_root)
    sample_audits = []
    for target in targets:
        audit_path = pareto_dir(out_root, target) / "sample_audit.csv"
        if audit_path.exists():
            table = pd.read_csv(audit_path)
            sample_audits.append(with_target_first_column(table, target))
    if sample_audits:
        pd.concat(sample_audits, ignore_index=True).to_csv(out_root / "sample_audit.csv", index=False)

    selected_rows = []
    exclusion_rows = []
    residual_cache: dict[str, dict[str, np.ndarray]] = {}
    for target in targets:
        selected, exclusion, residuals = select_equation(arrays, out_root, target, int(args.max_complexity))
        if selected is None:
            if exclusion is not None:
                exclusion_rows.append(exclusion)
            continue
        selected_rows.append(selected)
        residual_cache[target] = residuals or {}
    selected_df = pd.DataFrame(selected_rows, columns=SELECTED_EQUATION_COLUMNS)
    exclusion_df = pd.DataFrame(exclusion_rows)
    selected_df.to_csv(out_root / "selected_equations.csv", index=False)
    exclusion_df.to_csv(out_root / "exclusion_reasons.csv", index=False)
    grid, per_sensor, _ = evaluate_detection(arrays["labels"], selected_rows, residual_cache, variant="sensors_only_all")
    grid.to_csv(out_root / "detection_grid.csv", index=False)
    per_sensor.to_csv(out_root / "per_sensor_alarm_stats.csv", index=False)

    print("\n=== WADI 1-second delta full ===")
    print(f"Selected sensors: {len(selected_rows)} / {len(targets)}")
    geco = grid[(grid["S"] == GECO_S) & (grid["G"] == GECO_G)]
    best = grid.sort_values(["F1", "eTaF1"], ascending=False).head(5)
    print("Best rows:")
    print(best[["S", "G", "num_monitored", "Precision", "Recall", "F1", "eTaP", "eTaR", "eTaF1", "FPA", "Scen"]].to_string(index=False))
    if not geco.empty:
        print("GeCo S/G row:")
        print(geco[["S", "G", "num_monitored", "Precision", "Recall", "F1", "eTaP", "eTaR", "eTaF1", "FPA", "Scen"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
