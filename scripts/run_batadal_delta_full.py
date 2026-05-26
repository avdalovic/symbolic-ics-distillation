#!/usr/bin/env python
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import importlib.util
import json
import keyword
import math
import sys
import time
from pathlib import Path
from typing import Any, NamedTuple

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ics_symbolic_distill.data.ics_metadata import is_actuator, normalize_attack_labels
from ics_symbolic_distill.detection import compute_detection_metrics, evaluate_equation, fit_cusum_params, run_cusum
from ics_symbolic_distill.detection.metrics import to_intervals
from ics_symbolic_distill.detection.swat1s_delta_sampling import coverage_stratified_indices, reconstruct_next_from_delta


TARGET_MODE = "sensors_delta_actuators_next"
S_VALUES = [1.0, 1.39, 2.0, 3.0, 5.0]
G_VALUES = [1.0, 2.16, 5.0, 9.0]
GECO_S = 1.39
GECO_G = 2.16
BATADAL_EXPAND_STEPS = 1
GECO_EXCLUSIONS = {"P_J280"}
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


class BatadalAttackWindow(NamedTuple):
    attack_id: str
    start: int
    end: int
    affected_tags: tuple[str, ...]


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


def process_columns(df: pd.DataFrame) -> list[str]:
    drop = {"timestamp", "DATETIME", "ATT_FLAG", "attack_id", "attack_target", "Attack", "Row"}
    features = df.drop(columns=[c for c in drop if c in df.columns])
    numeric_cols = features.select_dtypes(include=[np.number]).columns.tolist()
    bad = [str(col) for col in numeric_cols if not str(col).isidentifier() or keyword.iskeyword(str(col))]
    if bad:
        raise ValueError(f"BATADAL variable names are not valid identifiers: {bad}")
    return [str(col) for col in numeric_cols]


def attack_windows_from_test(test_df: pd.DataFrame, labels_pair: np.ndarray) -> list[BatadalAttackWindow]:
    if "attack_id" not in test_df.columns:
        return []
    attack_ids = test_df["attack_id"].iloc[1:].reset_index(drop=True)
    attack_targets = test_df.get("attack_target", pd.Series([""] * len(test_df))).iloc[1:].reset_index(drop=True)
    windows = []
    for start, end in to_intervals(labels_pair):
        ids = attack_ids.iloc[start : end + 1]
        ids = [str(x).strip() for x in ids.tolist() if str(x).strip() and str(x).lower() != "nan"]
        attack_id = ids[0] if ids else str(len(windows) + 1)
        if attack_id.endswith(".0"):
            attack_id = attack_id[:-2]
        tags = attack_targets.iloc[start : end + 1]
        affected = []
        for tag in tags.tolist():
            text = str(tag).strip()
            if text and text.lower() != "nan" and text not in affected:
                affected.append(text)
        windows.append(BatadalAttackWindow(attack_id=attack_id, start=int(start), end=int(end), affected_tags=tuple(affected)))
    return windows


def load_batadal_arrays(args: argparse.Namespace) -> dict[str, Any]:
    train_path = Path(args.train_csv)
    test_path = Path(args.test_csv)
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"BATADAL CSVs not found: train={train_path} test={test_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    feature_columns = [c for c in process_columns(train_df) if c in process_columns(test_df)]
    if not feature_columns:
        raise ValueError("No shared numeric BATADAL process variables found")
    train_sel = train_df[feature_columns].copy()
    test_sel = test_df[feature_columns].copy()
    if train_sel.isna().any().any() or test_sel.isna().any().any():
        med = train_sel.median(axis=0, numeric_only=True)
        train_sel = train_sel.fillna(med).fillna(0.0)
        test_sel = test_sel.fillna(med).fillna(0.0)
    train = train_sel.to_numpy(dtype=np.float32)
    test = test_sel.to_numpy(dtype=np.float32)
    labels_raw = normalize_attack_labels(test_df["ATT_FLAG"]) if "ATT_FLAG" in test_df.columns else None
    if labels_raw is None:
        labels_raw = np.zeros(test.shape[0], dtype=np.float32)
    labels = labels_raw[1:].astype(np.int64)
    train_current = train[:-1].astype(np.float32, copy=False)
    train_next = train[1:].astype(np.float32, copy=False)
    test_current = test[:-1].astype(np.float32, copy=False)
    test_next = test[1:].astype(np.float32, copy=False)
    n_pairs = int(train_current.shape[0])
    cutoff = int(math.floor(n_pairs * 0.8))
    sensor_names = [c for c in feature_columns if not is_actuator("BATADAL", c)]
    actuator_names = [c for c in feature_columns if is_actuator("BATADAL", c)]
    attack_windows = attack_windows_from_test(test_df, labels)
    return {
        "train": train,
        "test": test,
        "train_current": train_current,
        "train_next": train_next,
        "test_current": test_current,
        "test_next": test_next,
        "labels": labels,
        "feature_columns": feature_columns,
        "sensor_names": sensor_names,
        "actuator_names": actuator_names,
        "attack_windows": attack_windows,
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
            "num_attack_windows_in_test": len(attack_windows),
        },
    }


def write_variable_classification(out_root: Path, arrays: dict[str, Any]) -> None:
    train = pd.DataFrame(arrays["train"], columns=arrays["feature_columns"])
    rows = []
    for col in arrays["feature_columns"]:
        series = train[col]
        rows.append(
            {
                "variable": col,
                "type": "actuator" if is_actuator("BATADAL", col) else "sensor",
                "unique_values": int(series.nunique(dropna=True)),
                "min": float(series.min()),
                "max": float(series.max()),
            }
        )
    pd.DataFrame(rows).to_csv(out_root / "variable_classification.csv", index=False)


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
    if is_actuator("BATADAL", target):
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
    pred = evaluate_equation(str(equation), arrays["feature_columns"], current)
    if is_actuator("BATADAL", target):
        pred_next = pred
    else:
        pred_next = reconstruct_next_from_delta(current[:, idx], pred)
    residual = np.abs(nxt[:, idx].astype(np.float64) - pred_next.astype(np.float64))
    return np.where(np.isfinite(residual), residual, 0.0).astype(np.float64)


def sample_size_for_fit(value: str, fit_pool_size: int) -> int:
    text = str(value).strip().lower()
    if text == "all":
        return int(fit_pool_size)
    requested = int(float(text))
    return min(max(requested, 0), int(fit_pool_size))


def sample_indices_for_target(arrays: dict[str, Any], target: str, sample_size_value: str) -> tuple[np.ndarray, dict[str, Any]]:
    fit_idx = arrays["fit_idx"]
    current, y_all = target_values(arrays, target, split="train")
    y_fit = y_all[fit_idx]
    actuator_indices = [arrays["feature_columns"].index(name) for name in arrays["actuator_names"]]
    sample_size = sample_size_for_fit(sample_size_value, len(fit_idx))
    local_idx, audit = coverage_stratified_indices(
        target=target,
        y_delta_fit_pool=y_fit,
        x_current_fit_pool=current[fit_idx],
        x_next_fit_pool=arrays["train_next"][fit_idx],
        actuator_indices=actuator_indices,
        sample_size=sample_size,
        seed=1337,
        transition_radius=5,
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
    offset = sum((i + 1) * ord(ch) for i, ch in enumerate(str(target))) % 100000
    return int(base_seed) + int(offset)


def features_in_equation(equation: str, feature_names: list[str]) -> list[str]:
    return DAY1.features_in_equation(str(equation), feature_names)


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


def pareto_dir(out: Path, target: str) -> Path:
    return out / "pareto_fronts" / f"{target}_{TARGET_MODE}"


def run_pysr_target(args_dict: dict[str, Any], target: str) -> dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    arrays = load_batadal_arrays(args)
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
        current, y_all = target_values(arrays, target, split="train")
        holdout_idx = arrays["holdout_idx"]
        sample_idx, audit = sample_indices_for_target(arrays, target, str(args.sample_size))
        out_dir.mkdir(parents=True, exist_ok=True)
        np.save(out_dir / "sample_indices.npy", sample_idx.astype(np.int64))
        pd.DataFrame([audit]).to_csv(out_dir / "sample_audit.csv", index=False)
        x_sample = pd.DataFrame(current[sample_idx], columns=feature_columns)
        y_sample = y_all[sample_idx]
        x_holdout = pd.DataFrame(current[holdout_idx], columns=feature_columns)
        y_holdout = y_all[holdout_idx]
        model, params = make_model(
            niterations=int(args.niterations),
            timeout_minutes=float(args.timeout_minutes),
            max_complexity=int(args.max_complexity),
            procs=int(args.pysr_procs),
            seed=stable_target_seed(target, int(args.seed)),
        )
        model.fit(x_sample, y_sample)
        pareto = evaluate_pareto_df(model, model.equations_.copy(), x_sample, y_sample, x_holdout, y_holdout, feature_columns)
        pareto.to_csv(csv_path, index=False)
    except Exception as exc:
        status = "failed"
        error = str(exc)
    write_json(
        out_dir / "metadata.json",
        {
            "target": target,
            "dataset": "BATADAL",
            "target_mode": TARGET_MODE,
            "sample_policy": "coverage_stratified",
            "sample_size": str(args.sample_size),
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
        pred_train_target = evaluate_equation(equation, arrays["feature_columns"], arrays["train_current"])
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
            "variable_type": "actuator" if is_actuator("BATADAL", target) else "sensor",
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
                        "source": row.get("source", "selected_sensor_delta"),
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
            metrics = compute_detection_metrics(labels, system_alarm, expand_steps=BATADAL_EXPAND_STEPS)
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
                    "attack_interval_count": metrics.get("attack_interval_count", float("nan")),
                    **point_counts(labels, system_alarm),
                    **{f"system_{k}": v for k, v in alarm_burden(system_alarm, labels).items()},
                }
            )
    return pd.DataFrame(grid_rows), pd.DataFrame(per_sensor_rows), alarm_cache


def build_variant_rows(arrays: dict[str, Any], selected_rows: list[dict[str, Any]], variant: str) -> tuple[list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    if variant == "sensors_only_all":
        sensors = list(selected_rows)
        actuators: list[str] = []
    elif variant == "sensors_only_geco_matched":
        sensors = [row for row in selected_rows if str(row["target"]) not in GECO_EXCLUSIONS]
        actuators = []
    elif variant == "geco_matched_plus_actuator_persistence":
        sensors = [row for row in selected_rows if str(row["target"]) not in GECO_EXCLUSIONS]
        actuators = [name for name in arrays["actuator_names"] if name not in GECO_EXCLUSIONS]
    else:
        raise ValueError(f"Unknown variant: {variant}")
    rows: list[dict[str, Any]] = []
    cache: dict[str, dict[str, np.ndarray]] = {}
    for row in sensors:
        target = str(row["target"])
        rows.append(row)
        cache[target] = {
            "train": prediction_residual(arrays, target, str(row["sympy_format"]), split="train"),
            "test": prediction_residual(arrays, target, str(row["sympy_format"]), split="test"),
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


def choose_summary_row(variant: str, grid: pd.DataFrame) -> tuple[str, pd.Series]:
    if variant == "sensors_only_geco_matched":
        eligible = grid[grid["FPA"] <= 5].sort_values(["F1", "eTaF1"], ascending=False)
        if not eligible.empty:
            return "best_f1_fpa_le_5", eligible.iloc[0]
    table = grid.sort_values(["F1", "eTaF1"], ascending=False)
    return "best_f1", table.iloc[0]


def per_attack_table(rows: list[dict[str, Any]], alarm_map: dict[str, np.ndarray], attack_windows: list[BatadalAttackWindow]) -> pd.DataFrame:
    monitored = {str(row["target"]) for row in rows}
    system = alarm_map.get("system")
    if system is None:
        n = int(next(iter(alarm_map.values())).shape[0]) if alarm_map else 0
        system = np.zeros(n, dtype=np.int8)
    n = int(system.shape[0])
    out = []
    for window in attack_windows:
        start = max(0, int(window.start))
        end = min(n - 1, int(window.end))
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
        out.append(
            {
                "attack_id": window.attack_id,
                "start": int(window.start),
                "end": int(window.end),
                "affected_tags": ",".join(affected),
                "in_scope_variables": ",".join(in_scope),
                "category": category,
                "firing_variables": ",".join(firing),
                "detected": system_detected,
            }
        )
    return pd.DataFrame(out)


def markdown_summary(summary: pd.DataFrame, top_fpa: dict[str, pd.DataFrame], physics_rows: pd.DataFrame) -> str:
    lines = [
        "# BATADAL 1-hour delta posthoc ablation",
        "",
        "All rows reuse fitted BATADAL delta sensor equations. Actuator channels, when present, use persistence only. GeCo-matched exclusions remove P_J280.",
        "",
        "## Summary",
        "",
        summary.to_markdown(index=False),
        "",
        "## References",
        "",
        "GeCo published BATADAL: Prec 93.8, Rec 73.4, F1 82.3, eTaP 97.0, eTaR 88.1, eTaF1 92.4, FPA 0, Scen 100.0.",
        "SIMPLE published BATADAL: Prec 52.0, Rec 43.3, F1 47.2, eTaP 49.0, eTaR 42.8, eTaF1 45.7, FPA 4, Scen 71.4.",
        "",
        "## Top FPA Contributors",
    ]
    for variant, table in top_fpa.items():
        lines.extend(["", f"### {variant}", "", table.to_markdown(index=False)])
    lines.extend(["", "## Physics Recovery Candidates", "", physics_rows.to_markdown(index=False)])
    return "\n".join(lines) + "\n"


def run_posthoc_ablation(arrays: dict[str, Any], selected_rows: list[dict[str, Any]], out_root: Path) -> pd.DataFrame:
    posthoc_root = out_root.parent / "delta_posthoc_ablation"
    posthoc_root.mkdir(parents=True, exist_ok=True)
    variant_names = ["sensors_only_all", "sensors_only_geco_matched", "geco_matched_plus_actuator_persistence"]
    summary_rows = []
    top_fpa: dict[str, pd.DataFrame] = {}
    for variant in variant_names:
        print(f"[BATADAL posthoc] evaluating {variant}", flush=True)
        rows, residual_cache = build_variant_rows(arrays, selected_rows, variant)
        grid, per_sensor, alarm_cache = evaluate_detection(arrays["labels"], rows, residual_cache, variant=variant)
        grid.to_csv(posthoc_root / f"grid_{variant}.csv", index=False)
        per_sensor.to_csv(posthoc_root / f"per_sensor_{variant}.csv", index=False)
        selection_name, selected_grid_row = choose_summary_row(variant, grid)
        key = (float(selected_grid_row["S"]), float(selected_grid_row["G"]))
        attack = per_attack_table(rows, alarm_cache[key], arrays["attack_windows"])
        attack.to_csv(posthoc_root / f"per_attack_{variant}.csv", index=False)
        top_columns = ["target", "variable_type", "source", "num_alarm_intervals", "benign_alarm_rate", "total_alarm_rate", "equation"]
        if per_sensor.empty:
            top_fpa[variant] = pd.DataFrame(columns=top_columns)
        else:
            selected_per_sensor = per_sensor[(per_sensor["S"] == key[0]) & (per_sensor["G"] == key[1])]
            top_fpa[variant] = selected_per_sensor.sort_values(["num_alarm_intervals", "benign_alarm_rate"], ascending=False)[top_columns].head(10)
        counts = attack["category"].value_counts().to_dict() if not attack.empty else {}
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
                "attack_interval_count": int(selected_grid_row.get("attack_interval_count", len(arrays["attack_windows"]))),
                "direct_detected": int(counts.get("direct_detected", 0)),
                "collateral_detected": int(counts.get("collateral_detected", 0)),
                "scope_miss": int(counts.get("scope_miss", 0)),
                "detection_failure": int(counts.get("detection_failure", 0)),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(posthoc_root / "summary.csv", index=False)
    physics_rows = pd.DataFrame(
        [
            {
                "target": row["target"],
                "complexity": row.get("complexity"),
                "score": row.get("score"),
                "holdout_r2": row.get("holdout_r2"),
                "equation": row.get("equation"),
            }
            for row in selected_rows
            if str(row.get("target", "")).startswith(("L_T", "F_PU"))
        ]
    ).head(20)
    (posthoc_root / "summary.md").write_text(markdown_summary(summary, top_fpa, physics_rows), encoding="utf-8")
    write_json(
        posthoc_root / "run_config.json",
        {
            "dataset": "BATADAL",
            "variants": variant_names,
            "s_values": S_VALUES,
            "g_values": G_VALUES,
            "geco_exclusions": sorted(GECO_EXCLUSIONS),
            "expand_steps": BATADAL_EXPAND_STEPS,
            "data": arrays["metadata"],
        },
    )
    return summary


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="BATADAL 1-hour delta symbolic PySR and detection.")
    parser.add_argument("--train-csv", default="data/batadal/processed/train.csv")
    parser.add_argument("--test-csv", default="data/batadal/processed/test_dataset04.csv")
    parser.add_argument("--out", default="artifacts/batadal/delta_full")
    parser.add_argument("--sample-size", default="all")
    parser.add_argument("--timeout-minutes", type=float, default=30.0)
    parser.add_argument("--max-complexity", type=int, default=15)
    parser.add_argument("--niterations", type=int, default=400)
    parser.add_argument("--target-parallel-jobs", type=int, default=4)
    parser.add_argument("--pysr-procs", type=int, default=1)
    parser.add_argument("--seed", type=int, default=1337)
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-pysr", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    arrays = load_batadal_arrays(args)
    write_variable_classification(out_root, arrays)
    targets = list(arrays["sensor_names"])
    write_json(
        out_root / "run_config.json",
        {
            "dataset": "BATADAL",
            "target_mode": TARGET_MODE,
            "sample_policy": "coverage_stratified",
            "sample_size": str(args.sample_size),
            "operator_set": "safe_mul",
            "niterations": int(args.niterations),
            "timeout_minutes": float(args.timeout_minutes),
            "max_complexity": int(args.max_complexity),
            "target_parallel_jobs": int(args.target_parallel_jobs),
            "pysr_procs": int(args.pysr_procs),
            "geco_s": GECO_S,
            "geco_g": GECO_G,
            "expand_steps": BATADAL_EXPAND_STEPS,
            "data": arrays["metadata"],
        },
    )
    print("=== BATADAL 1-hour data ===", flush=True)
    print(arrays["metadata"], flush=True)
    print(f"PySR sensor targets: {len(targets)}", flush=True)

    if not args.skip_pysr:
        args_dict = vars(args).copy()
        run_rows = []
        workers = max(1, int(args.target_parallel_jobs))
        with ProcessPoolExecutor(max_workers=workers) as pool:
            future_map = {pool.submit(run_pysr_target, args_dict, target): target for target in targets}
            for future in as_completed(future_map):
                target = future_map[future]
                try:
                    result = future.result()
                except Exception as exc:
                    result = {"target": target, "status": "failed", "error": str(exc)}
                print(f"[BATADAL] done target={target} status={result.get('status')}", flush=True)
                run_rows.append(result)
                pd.DataFrame(run_rows).to_csv(out_root / "pysr_run_status.csv", index=False)

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
    posthoc_summary = run_posthoc_ablation(arrays, selected_rows, out_root)

    print("\n=== BATADAL Delta Results ===")
    print(posthoc_summary.to_string(index=False))
    print("\n=== BATADAL Delta Full ===")
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
