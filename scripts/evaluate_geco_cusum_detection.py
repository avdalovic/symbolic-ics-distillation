#!/usr/bin/env python
from __future__ import annotations

import argparse
import subprocess
import sys
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ics_symbolic_distill.detection import (
    compute_detection_metrics,
    evaluate_equation,
    fit_cusum_params,
    load_pareto_front,
    run_cusum,
    to_intervals,
)
from ics_symbolic_distill.detection.io import (
    DetectionSplit,
    load_distillation_split,
    load_model_export_raw_split,
    package_versions,
    write_json,
)
from ics_symbolic_distill.detection.symbolic_eval import equation_features


@dataclass(frozen=True)
class DetectorSpec:
    name: str
    family: str
    variant: str
    target_source: str
    target_mode: str


GECO_REFERENCE = {
    "IDS": "GeCo reference",
    "Prec.": 94.8,
    "Rec.": 79.0,
    "F1": 86.2,
    "eTaP": 83.1,
    "eTaR": 60.7,
    "eTaF1": 70.2,
    "FPA": 4,
    "Scen.": 86.1,
}


def detector_specs(symbolic_target_mode: str) -> list[DetectorSpec]:
    mode = str(symbolic_target_mode).lower()
    specs = [DetectorSpec("MLP", "MLP", "neural", "neural_mlp", "next")]
    if mode in {"next", "both"}:
        specs.extend(
            [
                DetectorSpec("Sym-Raw-next", "Sym-Raw", "next", "actual_next", "next"),
                DetectorSpec("Sym-MLP-next", "Sym-MLP", "next", "mlp_next", "next"),
            ]
        )
    if mode in {"delta", "both"}:
        specs.extend(
            [
                DetectorSpec("Sym-Raw-delta", "Sym-Raw", "delta", "actual_delta", "delta"),
                DetectorSpec("Sym-MLP-delta", "Sym-MLP", "delta", "mlp_delta", "delta"),
            ]
        )
    return specs


def parse_csv_list(value: str | None) -> list[str]:
    if value is None:
        return []
    return [item.strip() for item in str(value).split(",") if item.strip()]


def filter_detector_specs(specs: list[DetectorSpec], requested: str | None) -> list[DetectorSpec]:
    names = parse_csv_list(requested)
    if not names:
        return specs
    by_name = {spec.name: spec for spec in specs}
    selected = []
    missing = []
    for name in names:
        if name in by_name:
            selected.append(by_name[name])
        else:
            missing.append(name)
    if missing:
        print(f"Warning: requested detectors not available for this symbolic-target-mode: {missing}")
    if not selected:
        raise ValueError(f"No requested detectors are available: {names}")
    return selected


def select_candidate_sensors(
    all_sensors: list[str],
    *,
    include_sensors: str | None,
    exclude_sensors: str | None,
) -> tuple[list[str], dict[str, Any]]:
    include = parse_csv_list(include_sensors)
    exclude = parse_csv_list(exclude_sensors)
    available = set(all_sensors)
    unknown_include = [name for name in include if name not in available]
    unknown_exclude = [name for name in exclude if name not in available]
    if unknown_include:
        print(f"Warning: requested include sensors are not target sensors and will be ignored: {unknown_include}")
    if unknown_exclude:
        print(f"Warning: requested exclude sensors are not target sensors and will be ignored: {unknown_exclude}")
    if include:
        initial = [name for name in include if name in available]
    else:
        initial = list(all_sensors)
    exclude_set = {name for name in exclude if name in available}
    candidates = [name for name in initial if name not in exclude_set]
    manual_excluded = [name for name in initial if name in exclude_set]
    return candidates, {
        "include_sensors": include,
        "exclude_sensors": exclude,
        "unknown_include_sensors": unknown_include,
        "unknown_exclude_sensors": unknown_exclude,
        "manual_excluded_sensors": manual_excluded,
    }


def is_sensor_scope_run(args: argparse.Namespace) -> bool:
    return bool(args.detectors or args.include_sensors or args.exclude_sensors or args.sensor_scope_name)


def variance_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float | None:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    mask = np.isfinite(y) & np.isfinite(pred)
    if not np.any(mask):
        return None
    y = y[mask]
    pred = pred[mask]
    return float(1.0 - np.var(y - pred) / (np.var(y) + 1e-10))


def finite_or_none(value: Any) -> float | None:
    try:
        val = float(value)
    except Exception:
        return None
    return val if np.isfinite(val) else None


def git_commit() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def check_compatible(name: str, left: DetectionSplit, right: DetectionSplit) -> None:
    if left.feature_columns != right.feature_columns:
        raise ValueError(f"{name}: feature column order differs")
    if left.target_columns != right.target_columns:
        raise ValueError(f"{name}: target column order differs")
    if left.sensor_idx != right.sensor_idx:
        raise ValueError(f"{name}: sensor_idx differs")


def load_split_inputs(args: argparse.Namespace) -> tuple[DetectionSplit, DetectionSplit, DetectionSplit, dict[str, Any]]:
    if args.calibration_split == "train" and not args.allow_train_calibration:
        raise ValueError(
            "--calibration-split train is unsafe for final evaluation. Pass --allow-train-calibration to use it explicitly."
        )
    calib_root = Path(args.calib_root or args.distill_root)
    if not calib_root.exists():
        if args.allow_train_calibration:
            calib_root = Path(args.distill_root)
        else:
            raise FileNotFoundError(
                f"Missing calibration split {calib_root}. Provide --calib-root or pass --allow-train-calibration explicitly."
            )
    calib = load_distillation_split(calib_root, require_labels=False, split_name=args.calibration_split)
    train_diag = load_distillation_split(args.train_root, require_labels=False) if args.train_root else calib
    if args.test_distill_root:
        test = load_distillation_split(args.test_distill_root, require_labels=True)
    else:
        test = load_model_export_raw_split(args.test_export, require_labels=True)
    check_compatible("train/calib", train_diag, calib)
    check_compatible("calib/test", calib, test)
    used_train = args.calibration_split == "train" or (calib.source == train_diag.source and bool(args.allow_train_calibration))
    info = {
        "calibration_source": calib.source,
        "calibration_split": args.calibration_split,
        "used_train_calibration": bool(used_train),
    }
    return train_diag, calib, test, info


def _raw_current_sensor(split: DetectionSplit, target_index: int) -> np.ndarray:
    return split.x_current_raw[:, split.sensor_idx[target_index]]


def symbolic_prediction(
    *,
    spec: DetectorSpec,
    sensor: str,
    target_index: int,
    split: DetectionSplit,
    equation: str,
) -> np.ndarray:
    pred = evaluate_equation(equation, split.feature_columns, split.x_current_raw)
    if spec.target_mode == "delta":
        pred = _raw_current_sensor(split, target_index) + pred
    return np.asarray(pred, dtype=np.float64)


def select_symbolic_equation(
    *,
    sensor: str,
    spec: DetectorSpec,
    audit_root: str | Path,
) -> tuple[dict[str, Any] | None, str | None]:
    row, front = load_pareto_front(sensor, spec.target_source, audit_root)
    if row is None or front is None:
        return None, None
    equation = str(row.get("equation", ""))
    diag = {
        "sensor": sensor,
        "detector": spec.name,
        "family": spec.family,
        "variant": spec.variant,
        "target_source": spec.target_source,
        "target_mode": spec.target_mode,
        "equation": equation,
        "selected_score": finite_or_none(row.get("score", row.get("_selection_score"))),
        "selected_loss": finite_or_none(row.get("loss")),
        "selected_complexity": finite_or_none(row.get("complexity")),
        "selected_row_index": int(row.get("_row_index", -1)),
        "source_csv": str(row.get("_source_csv", "")),
        "pareto_max_complexity": finite_or_none(pd.to_numeric(front.get("complexity"), errors="coerce").max())
        if "complexity" in front.columns
        else None,
    }
    return diag, equation


def residual_stats(values: np.ndarray, mask: np.ndarray | None = None) -> tuple[float | None, float | None]:
    arr = np.asarray(values, dtype=np.float64).reshape(-1)
    if mask is not None:
        arr = arr[np.asarray(mask, dtype=bool).reshape(-1)]
    arr = arr[np.isfinite(arr)]
    if arr.size == 0:
        return None, None
    return float(np.mean(arr)), float(np.std(arr))


def alarm_burden(alarms: np.ndarray, labels: np.ndarray | None = None) -> dict[str, float | None]:
    alarm = (np.asarray(alarms).reshape(-1) >= 0.5).astype(np.int64)
    intervals = to_intervals(alarm)
    first = intervals[0][0] if intervals else None
    longest = max((end - start + 1 for start, end in intervals), default=0)
    out: dict[str, float | None] = {
        "total_alarm_rate": float(np.mean(alarm)) if alarm.size else 0.0,
        "num_alarm_intervals": float(len(intervals)),
        "longest_alarm_interval": float(longest),
        "first_alarm_timestep": None if first is None else float(first),
    }
    if labels is not None:
        y = np.asarray(labels).reshape(-1) >= 0.5
        benign = ~y
        attack = y
        out["benign_alarm_rate"] = float(np.mean(alarm[benign])) if np.any(benign) else None
        out["attack_alarm_rate"] = float(np.mean(alarm[attack])) if np.any(attack) else None
    return out


def evaluate_sensor(
    *,
    sensor: str,
    train_diag: DetectionSplit,
    calib: DetectionSplit,
    test: DetectionSplit,
    audit_root: str | Path,
    specs: list[DetectorSpec],
    s: float,
    g: float,
) -> dict[str, Any]:
    if sensor not in calib.target_columns:
        raise ValueError(f"Sensor {sensor} is not in target_columns")
    j = int(calib.target_columns.index(sensor))
    labels = test.labels
    detector_results: dict[str, dict[str, Any]] = {}
    skipped: list[dict[str, str]] = []
    equation_diagnostics: list[dict[str, Any]] = []

    actual_diag = train_diag.actual_next_raw[:, j]
    actual_calib = calib.actual_next_raw[:, j]
    actual_test = test.actual_next_raw[:, j]
    mlp_diag = train_diag.mlp_pred_next_raw[:, j]
    mlp_calib = calib.mlp_pred_next_raw[:, j]
    mlp_test = test.mlp_pred_next_raw[:, j]

    for spec in specs:
        if spec.name == "MLP":
            pred_diag, pred_calib, pred_test = mlp_diag, mlp_calib, mlp_test
            diag = {
                "sensor": sensor,
                "detector": spec.name,
                "family": spec.family,
                "variant": spec.variant,
                "target_source": spec.target_source,
                "target_mode": spec.target_mode,
                "status": "ok",
            }
        else:
            diag, equation = select_symbolic_equation(sensor=sensor, spec=spec, audit_root=audit_root)
            if diag is None or equation is None:
                skipped.append({"detector": spec.name, "sensor": sensor, "reason": f"missing_{spec.target_source}_equation"})
                continue
            pred_diag = symbolic_prediction(spec=spec, sensor=sensor, target_index=j, split=train_diag, equation=equation)
            pred_calib = symbolic_prediction(spec=spec, sensor=sensor, target_index=j, split=calib, equation=equation)
            pred_test = symbolic_prediction(spec=spec, sensor=sensor, target_index=j, split=test, equation=equation)
            nonfinite = {
                "diag": int((~np.isfinite(pred_diag)).sum()),
                "calib": int((~np.isfinite(pred_calib)).sum()),
                "test": int((~np.isfinite(pred_test)).sum()),
            }
            if any(nonfinite.values()):
                diag.update({"status": "nonfinite_prediction", "nonfinite_counts": nonfinite})
                equation_diagnostics.append(diag)
                skipped.append({"detector": spec.name, "sensor": sensor, "reason": "nonfinite_prediction"})
                continue
            diag["equation_features"] = equation_features(equation, train_diag.feature_columns)
            equation_diagnostics.append(diag)

        benign_mask = labels < 0.5 if labels is not None else None
        calib_r2 = variance_r2(actual_calib, pred_calib)
        test_benign_r2 = variance_r2(actual_test[benign_mask], pred_test[benign_mask]) if benign_mask is not None else None
        fidelity_calib_r2 = None
        fidelity_test_benign_r2 = None
        if spec.family == "Sym-MLP":
            fidelity_calib_r2 = variance_r2(mlp_calib, pred_calib)
            fidelity_test_benign_r2 = (
                variance_r2(mlp_test[benign_mask], pred_test[benign_mask]) if benign_mask is not None else None
            )
        diag.update(
            {
                "diag_r2": variance_r2(actual_diag, pred_diag),
                "calib_r2": calib_r2,
                "test_benign_r2": test_benign_r2,
                "fidelity_calib_r2": fidelity_calib_r2,
                "fidelity_test_benign_r2": fidelity_test_benign_r2,
                "status": "ok",
            }
        )

        r_calib = np.abs(actual_calib - pred_calib)
        r_test = np.abs(actual_test - pred_test)
        params = fit_cusum_params(r_calib, s=s, g=g)
        calib_cusum, calib_alarms = run_cusum(r_calib, params)
        test_cusum, test_alarms = run_cusum(r_test, params)
        metrics = compute_detection_metrics(labels, test_alarms) if labels is not None else {}
        calib_alarm_rate = float(np.mean(calib_alarms)) if calib_alarms.size else 0.0
        detector_results[spec.name] = {
            "sensor": sensor,
            "spec": spec,
            "pred_diag": pred_diag,
            "pred_calib": pred_calib,
            "pred_test": pred_test,
            "residual_calib": r_calib,
            "residual_test": r_test,
            "calib_cusum": calib_cusum,
            "calib_alarms": calib_alarms,
            "cusum": test_cusum,
            "alarms": test_alarms,
            "params": params,
            "metrics": metrics,
            "diagnostics": diag,
            "calib_alarm_rate": calib_alarm_rate,
        }

    return {
        "sensor": sensor,
        "detectors": detector_results,
        "skipped": skipped,
        "equation_diagnostics": equation_diagnostics,
        "target_index": j,
    }


def include_sensor(
    payload: dict[str, Any],
    *,
    filter_mode: str,
    r2_threshold: float,
    max_calib_alarm_rate: float,
) -> tuple[bool, str | None]:
    if filter_mode == "none":
        return True, None
    diag = payload["diagnostics"]
    if filter_mode == "calib-r2":
        calib_r2 = diag.get("calib_r2")
        if calib_r2 is None or not np.isfinite(float(calib_r2)) or float(calib_r2) < r2_threshold:
            return False, "calib_r2_below_threshold"
        return True, None
    if filter_mode == "alarm-burden":
        if float(payload.get("calib_alarm_rate", 0.0)) > max_calib_alarm_rate:
            return False, "calib_alarm_rate_above_threshold"
        return True, None
    raise ValueError(f"Unsupported sensor filter: {filter_mode}")


def aggregate_alarms(per_sensor_alarms: np.ndarray, *, aggregation: str, k: int) -> np.ndarray:
    if per_sensor_alarms.ndim != 2:
        raise ValueError("per_sensor_alarms must be [N, S]")
    if aggregation == "or":
        threshold = 1
    elif aggregation == "kofn":
        threshold = int(k)
    else:
        raise ValueError("--system-aggregation must be or or kofn")
    return (per_sensor_alarms.sum(axis=1) >= threshold).astype(np.int64)


def system_ablation_rows(
    *,
    results: list[dict[str, Any]],
    labels: np.ndarray,
    specs: list[DetectorSpec],
    filter_modes: list[str],
    aggregations: list[tuple[str, int]],
    r2_threshold: float,
    max_calib_alarm_rate: float,
) -> tuple[pd.DataFrame, list[dict[str, Any]], dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    excluded: list[dict[str, Any]] = []
    alarms_payload: dict[str, Any] = {}
    for spec in specs:
        for filter_mode in filter_modes:
            included_payloads = []
            for res in results:
                payload = res["detectors"].get(spec.name)
                if payload is None:
                    continue
                include, reason = include_sensor(
                    payload,
                    filter_mode=filter_mode,
                    r2_threshold=r2_threshold,
                    max_calib_alarm_rate=max_calib_alarm_rate,
                )
                if include:
                    included_payloads.append(payload)
                else:
                    diag = payload["diagnostics"]
                    excluded.append(
                        {
                            "detector": spec.name,
                            "variant": spec.variant,
                            "sensor": res["sensor"],
                            "filter_mode": filter_mode,
                            "reason": reason,
                            "calib_r2": diag.get("calib_r2"),
                            "calib_alarm_rate": payload.get("calib_alarm_rate"),
                            "sensor_filter_threshold": r2_threshold,
                            "max_calib_alarm_rate": max_calib_alarm_rate,
                        }
                    )
            if not included_payloads:
                continue
            sensor_names = [payload["sensor"] for payload in included_payloads]
            per_sensor_alarms = np.stack([payload["alarms"] for payload in included_payloads], axis=1)
            per_sensor_cusum = np.stack([payload["cusum"] for payload in included_payloads], axis=1)
            for aggregation, k_value in aggregations:
                if aggregation == "kofn" and len(included_payloads) < k_value:
                    continue
                system_alarm = aggregate_alarms(per_sensor_alarms, aggregation=aggregation, k=k_value)
                metrics = compute_detection_metrics(labels, system_alarm)
                burden = alarm_burden(system_alarm, labels)
                row = {
                    "detector": spec.name,
                    "family": spec.family,
                    "variant": spec.variant,
                    "target_source": spec.target_source,
                    "sensor_filter": filter_mode,
                    "aggregation": "or" if aggregation == "or" else "kofn",
                    "k": 1 if aggregation == "or" else int(k_value),
                    "num_valid_sensors": len(included_payloads),
                    "num_excluded_sensors": len([x for x in excluded if x["detector"] == spec.name and x["filter_mode"] == filter_mode]),
                    "system_total_alarm_rate": burden["total_alarm_rate"],
                    "system_benign_alarm_rate": burden.get("benign_alarm_rate"),
                    "system_attack_alarm_rate": burden.get("attack_alarm_rate"),
                    "num_system_alarm_intervals": burden["num_alarm_intervals"],
                    "longest_system_alarm_interval": burden["longest_alarm_interval"],
                    "first_system_alarm_timestep": burden["first_alarm_timestep"],
                    **metrics,
                }
                rows.append(row)
                key = f"{spec.name}|{filter_mode}|{row['aggregation']}|{row['k']}"
                alarms_payload[key] = {
                    "sensor_names": sensor_names,
                    "per_sensor_alarms": per_sensor_alarms,
                    "per_sensor_cusum": per_sensor_cusum,
                    "system_alarm": system_alarm,
                }
    return pd.DataFrame(rows), excluded, alarms_payload


def per_sensor_alarm_burden_rows(results: list[dict[str, Any]], labels: np.ndarray) -> list[dict[str, Any]]:
    rows = []
    benign = labels < 0.5
    attack = labels >= 0.5
    for result in results:
        for detector, payload in result["detectors"].items():
            params = payload["params"]
            r_calib = payload["residual_calib"]
            r_test = payload["residual_test"]
            diag = payload["diagnostics"]
            calib_mean, calib_std = residual_stats(r_calib)
            benign_mean, benign_std = residual_stats(r_test, benign)
            attack_mean, attack_std = residual_stats(r_test, attack)
            burden = alarm_burden(payload["alarms"], labels)
            rows.append(
                {
                    "sensor": result["sensor"],
                    "detector": detector,
                    "variant": diag.get("variant"),
                    "target_source": diag.get("target_source"),
                    "delta": params.delta,
                    "threshold": params.threshold,
                    "growth_cap": params.growth_cap,
                    "max_calib_cusum": params.max_calib_cusum,
                    "max_test_cusum": float(np.max(payload["cusum"])) if payload["cusum"].size else 0.0,
                    "max_test_cusum_over_threshold": float(np.max(payload["cusum"]) / max(params.threshold, 1e-12)),
                    "calib_residual_mean": calib_mean,
                    "calib_residual_std": calib_std,
                    "test_benign_residual_mean": benign_mean,
                    "test_benign_residual_std": benign_std,
                    "test_attack_residual_mean": attack_mean,
                    "test_attack_residual_std": attack_std,
                    "calib_r2": diag.get("calib_r2"),
                    "test_benign_r2": diag.get("test_benign_r2"),
                    "benign_alarm_rate": burden.get("benign_alarm_rate"),
                    "attack_alarm_rate": burden.get("attack_alarm_rate"),
                    "total_alarm_rate": burden.get("total_alarm_rate"),
                    "num_alarm_intervals": burden.get("num_alarm_intervals"),
                    "longest_alarm_interval": burden.get("longest_alarm_interval"),
                    "first_alarm_timestep": burden.get("first_alarm_timestep"),
                    "valid_for_system_alarm": True,
                }
            )
    return rows


def per_sensor_metric_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        for detector, payload in result["detectors"].items():
            diag = payload["diagnostics"]
            rows.append(
                {
                    "sensor": result["sensor"],
                    "detector": detector,
                    "variant": diag.get("variant"),
                    "target_source": diag.get("target_source"),
                    "delta": payload["params"].delta,
                    "threshold": payload["params"].threshold,
                    "growth_cap": payload["params"].growth_cap,
                    "max_calib_cusum": payload["params"].max_calib_cusum,
                    "calib_r2": diag.get("calib_r2"),
                    "test_benign_r2": diag.get("test_benign_r2"),
                    "fidelity_calib_r2": diag.get("fidelity_calib_r2"),
                    "fidelity_test_benign_r2": diag.get("fidelity_test_benign_r2"),
                    "calib_alarm_rate": payload.get("calib_alarm_rate"),
                    **payload["metrics"],
                }
            )
    return rows


def cusum_param_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    return [
        {"sensor": result["sensor"], "detector": detector, **payload["params"].to_dict()}
        for result in results
        for detector, payload in result["detectors"].items()
    ]


def selected_equation_rows(results: list[dict[str, Any]]) -> list[dict[str, Any]]:
    rows = []
    for result in results:
        rows.extend(result["equation_diagnostics"])
    return rows


def agreement_analysis(results: list[dict[str, Any]], labels: np.ndarray, sym_detector: str) -> dict[str, Any]:
    attack_intervals = to_intervals(labels)
    out: dict[str, Any] = {
        "sym_detector": sym_detector,
        "fidelity_calib_r2_per_sensor": {},
        "fidelity_test_benign_r2_per_sensor": {},
        "worst_fidelity_sensors": [],
        "gap_summary_per_sensor": {},
        "per_attack_agreement_matrix": {"agree_both": 0, "mlp_only": 0, "sym_only": 0, "agree_neither": 0},
    }
    for result in results:
        if "MLP" not in result["detectors"] or sym_detector not in result["detectors"]:
            continue
        sensor = result["sensor"]
        mlp = result["detectors"]["MLP"]
        sym = result["detectors"][sym_detector]
        diag = sym["diagnostics"]
        out["fidelity_calib_r2_per_sensor"][sensor] = diag.get("fidelity_calib_r2")
        out["fidelity_test_benign_r2_per_sensor"][sensor] = diag.get("fidelity_test_benign_r2")
        gap = np.abs(mlp["residual_test"] - sym["residual_test"])
        benign = labels < 0.5
        attack = labels >= 0.5
        out["gap_summary_per_sensor"][sensor] = {
            "mean_gap_benign": float(np.mean(gap[benign])) if np.any(benign) else None,
            "mean_gap_attack": float(np.mean(gap[attack])) if np.any(attack) else None,
            "max_gap": float(np.max(gap)) if gap.size else None,
            "fidelity_calib_r2": diag.get("fidelity_calib_r2"),
            "fidelity_test_benign_r2": diag.get("fidelity_test_benign_r2"),
        }
        for start, end in attack_intervals:
            mlp_alarm = bool(np.any(mlp["alarms"][start : end + 1] == 1))
            sym_alarm = bool(np.any(sym["alarms"][start : end + 1] == 1))
            if mlp_alarm and sym_alarm:
                out["per_attack_agreement_matrix"]["agree_both"] += 1
            elif mlp_alarm and not sym_alarm:
                out["per_attack_agreement_matrix"]["mlp_only"] += 1
            elif sym_alarm and not mlp_alarm:
                out["per_attack_agreement_matrix"]["sym_only"] += 1
            else:
                out["per_attack_agreement_matrix"]["agree_neither"] += 1
    sortable = [
        (sensor, value)
        for sensor, value in out["fidelity_test_benign_r2_per_sensor"].items()
        if value is not None and np.isfinite(float(value))
    ]
    out["worst_fidelity_sensors"] = [
        {"sensor": sensor, "fidelity_test_benign_r2": float(value)}
        for sensor, value in sorted(sortable, key=lambda item: item[1])[:5]
    ]
    return out


def write_agreement_markdown(path: Path, agreements: list[dict[str, Any]]) -> None:
    lines = ["# MLP and Symbolic-MLP Agreement", ""]
    for agreement in agreements:
        lines.extend([f"## {agreement['sym_detector']}", "", "Per-attack alarm agreement:"])
        for key, value in agreement["per_attack_agreement_matrix"].items():
            lines.append(f"- {key}: {value}")
        lines.extend(["", "Worst benign-test fidelity sensors:"])
        for item in agreement["worst_fidelity_sensors"]:
            lines.append(f"- {item['sensor']}: R2={item['fidelity_test_benign_r2']:.6g}")
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def write_alarms_npz(path: Path, payload: dict[str, Any]) -> None:
    arrays: dict[str, np.ndarray] = {}
    for key, data in payload.items():
        safe = key.lower().replace("-", "_").replace("|", "__")
        arrays[f"{safe}__sensor_names"] = np.asarray(data["sensor_names"], dtype=str)
        arrays[f"{safe}__per_sensor_alarms"] = np.asarray(data["per_sensor_alarms"], dtype=np.int8)
        arrays[f"{safe}__per_sensor_cusum"] = np.asarray(data["per_sensor_cusum"], dtype=np.float32)
        arrays[f"{safe}__system_alarm"] = np.asarray(data["system_alarm"], dtype=np.int8)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **arrays)


def write_system_markdown(path: Path, system_df: pd.DataFrame) -> None:
    lines = [
        "# System Detection Ablation Summary",
        "",
        "The GeCo row is a visual reference from a separate setup. This repository uses 10-second downsampled SWaT data; direct equality with 1-second results should not be assumed.",
        "",
    ]
    main = system_df[(system_df["sensor_filter"] == "none") & (system_df["aggregation"] == "or")].copy()
    headers = ["IDS", "Prec.", "Rec.", "F1", "eTaP", "eTaR", "eTaF1", "FPA", "Scen."]
    lines.extend(["| " + " | ".join(headers) + " |", "| " + " | ".join(["---"] * len(headers)) + " |"])
    lines.append("| GeCo reference | 94.8 | 79.0 | 86.2 | 83.1 | 60.7 | 70.2 | 4 | 86.1 |")
    for _, row in main.iterrows():
        lines.append(
            f"| {row['detector']} | {row['point_precision']:.1f} | {row['point_recall']:.1f} | "
            f"{row['point_f1']:.1f} | {row['eTaP']:.1f} | {row['eTaR']:.1f} | "
            f"{row['eTaF1']:.1f} | {row['FPA']:.0f} | {row['scenario_detection_rate']:.1f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def print_system_table(system_df: pd.DataFrame) -> None:
    main = system_df[(system_df["sensor_filter"] == "none") & (system_df["aggregation"] == "or")]
    if main.empty:
        main = system_df
    print("\nIDS              Prec.  Rec.   F1     eTaP   eTaR   eTaF1  FPA   Scen.")
    print("GeCo reference    94.8   79.0   86.2   83.1   60.7    70.2     4   86.1")
    for _, row in main.iterrows():
        print(
            f"{row['detector']:<16} {row['point_precision']:5.1f}  {row['point_recall']:5.1f}  "
            f"{row['point_f1']:5.1f}  {row['eTaP']:5.1f}  {row['eTaR']:5.1f}  "
            f"{row['eTaF1']:6.1f}  {row['FPA']:4.0f}  {row['scenario_detection_rate']:5.1f}"
        )


def write_lit101_figure(path: Path, result: dict[str, Any], test: DetectionSplit, detectors: list[str], title: str) -> None:
    sensor = result["sensor"]
    target_idx = test.target_columns.index(sensor)
    feature_idx = test.sensor_idx[target_idx]
    labels = test.labels if test.labels is not None else np.zeros(test.x_current_raw.shape[0], dtype=np.int64)
    intervals = to_intervals(labels)
    t = np.arange(test.x_current_raw.shape[0])
    colors = {"MLP": "#1f77b4", "Sym-Raw-next": "#2ca02c", "Sym-MLP-next": "#d62728", "Sym-Raw-delta": "#17becf", "Sym-MLP-delta": "#9467bd"}
    fig, axes = plt.subplots(4, 1, figsize=(13, 9), sharex=True)
    for ax in axes:
        for start, end in intervals:
            ax.axvspan(start, end, color="#f4b6b6", alpha=0.35, linewidth=0)
    axes[0].plot(t, test.x_current_raw[:, feature_idx], color="black", linewidth=0.9, label=f"{sensor} current")
    axes[0].set_title(title)
    axes[0].set_ylabel("Value")
    axes[0].legend(loc="upper right")
    for detector in detectors:
        if detector not in result["detectors"]:
            continue
        payload = result["detectors"][detector]
        color = colors.get(detector)
        axes[1].plot(t, payload["residual_test"], linewidth=0.8, label=detector, color=color)
        axes[1].axhline(payload["params"].delta, linestyle="--", linewidth=0.8, color=color)
        axes[2].plot(t, payload["cusum"], linewidth=0.8, label=detector, color=color)
        axes[2].axhline(payload["params"].threshold, linestyle="-", linewidth=0.8, color=color)
    axes[1].set_ylabel("Abs residual")
    axes[1].legend(loc="upper right")
    axes[2].set_ylabel("CUSUM")
    axes[2].legend(loc="upper right")
    sym = next((name for name in detectors if name.startswith("Sym-MLP") and name in result["detectors"]), None)
    if sym and "MLP" in result["detectors"]:
        gap = np.abs(result["detectors"]["MLP"]["residual_test"] - result["detectors"][sym]["residual_test"])
        axes[3].plot(t, gap, color=colors.get(sym), linewidth=0.8, label=f"|MLP residual - {sym} residual|")
        axes[3].legend(loc="upper right")
    axes[3].set_ylabel("Gap")
    axes[3].set_xlabel("Downsampled test timestep")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=150)
    plt.close(fig)


def lit101_s_sweep(result: dict[str, Any], labels: np.ndarray, out: Path, g: float) -> pd.DataFrame:
    rows = []
    residual_calib = result["detectors"]["MLP"]["residual_calib"]
    residual_test = result["detectors"]["MLP"]["residual_test"]
    for s_value in [0.5, 0.8, 1.0, 1.42, 2.0, 3.0, 5.0]:
        params = fit_cusum_params(residual_calib, s=s_value, g=g)
        _, alarms = run_cusum(residual_test, params)
        metrics = compute_detection_metrics(labels, alarms)
        rows.append({"S": s_value, "G": g, "eTaF1": metrics["eTaF1"], **metrics})
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False)
    print("\nLIT101 MLP S sensitivity (G fixed):")
    print(df[["S", "eTaF1"]].to_string(index=False))
    return df


def write_paper_artifacts(system_df: pd.DataFrame, lit101_result: dict[str, Any] | None) -> None:
    paper = REPO_ROOT / "paper_artifacts"
    paper.mkdir(parents=True, exist_ok=True)
    system_df.to_csv(paper / "detection_ablation_summary.csv", index=False)
    main = system_df[(system_df["sensor_filter"] == "none") & (system_df["aggregation"] == "or")]
    main.to_csv(paper / "detection_summary.csv", index=False)
    if lit101_result is not None:
        rows = []
        for detector, payload in lit101_result["detectors"].items():
            diag = payload.get("diagnostics") or {}
            rows.append(
                {
                    "sensor": "LIT101",
                    "detector": detector,
                    "variant": diag.get("variant"),
                    "calib_r2": diag.get("calib_r2"),
                    "test_benign_r2": diag.get("test_benign_r2"),
                    "fidelity_calib_r2": diag.get("fidelity_calib_r2"),
                    "fidelity_test_benign_r2": diag.get("fidelity_test_benign_r2"),
                    "delta": payload["params"].delta,
                    "threshold": payload["params"].threshold,
                    **payload["metrics"],
                }
            )
        pd.DataFrame(rows).to_csv(paper / "lit101_detection_diagnostic_summary.csv", index=False)


def active_sensor_entries(
    *,
    results: list[dict[str, Any]],
    specs: list[DetectorSpec],
    system_df: pd.DataFrame,
    excluded: list[dict[str, Any]],
    scope_info: dict[str, Any],
    sensor_filter: str,
) -> list[dict[str, Any]]:
    candidate_sensors = [res["sensor"] for res in results]
    skipped_all = [item for res in results for item in res["skipped"]]
    entries: list[dict[str, Any]] = []
    for spec in specs:
        detector_rows = system_df[system_df["detector"] == spec.name]
        if detector_rows.empty:
            continue
        skipped = sorted({item["sensor"] for item in skipped_all if item["detector"] == spec.name})
        quality_excluded = sorted(
            {
                item["sensor"]
                for item in excluded
                if item["detector"] == spec.name and item["filter_mode"] == sensor_filter
            }
        )
        active = [
            res["sensor"]
            for res in results
            if spec.name in res["detectors"] and res["sensor"] not in quality_excluded
        ]
        for _, row in detector_rows.iterrows():
            entries.append(
                {
                    "sensor_scope_name": row.get("sensor_scope_name", ""),
                    "detector": spec.name,
                    "variant": spec.variant,
                    "target_source": spec.target_source,
                    "sensor_filter": row["sensor_filter"],
                    "sensor_filter_threshold": row.get("sensor_filter_threshold"),
                    "aggregation": row["aggregation"],
                    "k": int(row["k"]),
                    "candidate_sensors": candidate_sensors,
                    "manual_excluded_sensors": scope_info.get("manual_excluded_sensors", []),
                    "quality_excluded_sensors": quality_excluded,
                    "skipped_sensors": skipped,
                    "active_sensors": active,
                    "num_candidate_sensors": len(candidate_sensors),
                    "num_manual_excluded_sensors": len(scope_info.get("manual_excluded_sensors", [])),
                    "num_quality_excluded_sensors": len(quality_excluded),
                    "num_skipped_sensors": len(skipped),
                    "num_active_sensors": len(active),
                    "unknown_include_sensors": scope_info.get("unknown_include_sensors", []),
                    "unknown_exclude_sensors": scope_info.get("unknown_exclude_sensors", []),
                }
            )
    return entries


def remaining_sensor_burden(
    burden_df: pd.DataFrame,
    active_entries: list[dict[str, Any]],
) -> pd.DataFrame:
    frames = []
    for entry in active_entries:
        active = set(entry["active_sensors"])
        mask = (burden_df["detector"] == entry["detector"]) & (burden_df["sensor"].isin(active))
        cols = [
            "sensor",
            "detector",
            "variant",
            "calib_r2",
            "test_benign_r2",
            "benign_alarm_rate",
            "attack_alarm_rate",
            "total_alarm_rate",
            "num_alarm_intervals",
            "longest_alarm_interval",
            "max_test_cusum_over_threshold",
        ]
        sub = burden_df.loc[mask, [col for col in cols if col in burden_df.columns]].copy()
        sub["sensor_scope_name"] = entry["sensor_scope_name"]
        sub["sensor_filter"] = entry["sensor_filter"]
        sub["aggregation"] = entry["aggregation"]
        sub["k"] = entry["k"]
        frames.append(sub)
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    return out.sort_values(["detector", "benign_alarm_rate"], ascending=[True, False])


def _summary_row_for_scope(row: pd.Series, entry: dict[str, Any]) -> dict[str, Any]:
    return {
        "sensor_scope_name": entry["sensor_scope_name"],
        "detector": row["detector"],
        "variant": row["variant"],
        "aggregation": row["aggregation"],
        "k": int(row["k"]),
        "sensor_filter": row["sensor_filter"],
        "sensor_filter_threshold": row.get("sensor_filter_threshold"),
        "num_candidate_sensors": entry["num_candidate_sensors"],
        "num_manual_excluded_sensors": entry["num_manual_excluded_sensors"],
        "num_quality_excluded_sensors": entry["num_quality_excluded_sensors"],
        "num_skipped_sensors": entry["num_skipped_sensors"],
        "num_active_sensors": entry["num_active_sensors"],
        "active_sensors": ",".join(entry["active_sensors"]),
        "manual_excluded_sensors": ",".join(entry["manual_excluded_sensors"]),
        "quality_excluded_sensors": ",".join(entry["quality_excluded_sensors"]),
        "skipped_sensors": ",".join(entry["skipped_sensors"]),
        "Prec.": row["point_precision"],
        "Rec.": row["point_recall"],
        "F1": row["point_f1"],
        "eTaP": row["eTaP"],
        "eTaR": row["eTaR"],
        "eTaF1": row["eTaF1"],
        "FPA": row["FPA"],
        "Scen.": row["scenario_detection_rate"],
        "system_benign_alarm_rate": row["system_benign_alarm_rate"],
        "system_attack_alarm_rate": row["system_attack_alarm_rate"],
        "system_total_alarm_rate": row["system_total_alarm_rate"],
        "num_system_alarm_intervals": row["num_system_alarm_intervals"],
        "longest_system_alarm_interval": row["longest_system_alarm_interval"],
    }


def update_sensor_scope_summary(
    *,
    out_dir: Path,
    system_df: pd.DataFrame,
    active_entries: list[dict[str, Any]],
) -> pd.DataFrame:
    root = out_dir.parent if out_dir.name.startswith("sensor_scope_") else out_dir
    summary_path = root / "sensor_scope_ablation_summary.csv"
    rows: list[dict[str, Any]] = []

    all_valid_path = root / "system_detection_ablation_summary.csv"
    if all_valid_path.exists():
        all_valid = pd.read_csv(all_valid_path)
        match = all_valid[
            (all_valid["detector"] == "Sym-Raw-delta")
            & (all_valid["sensor_filter"] == "none")
            & (all_valid["aggregation"] == "or")
            & (all_valid["k"] == 1)
        ]
        if not match.empty:
            row = match.iloc[0]
            rows.append(
                {
                    "sensor_scope_name": "all_valid",
                    "detector": row["detector"],
                    "variant": row["variant"],
                    "aggregation": row["aggregation"],
                    "k": int(row["k"]),
                    "sensor_filter": row["sensor_filter"],
                    "sensor_filter_threshold": "",
                    "num_candidate_sensors": int(row["num_valid_sensors"]),
                    "num_manual_excluded_sensors": 0,
                    "num_quality_excluded_sensors": 0,
                    "num_skipped_sensors": "",
                    "num_active_sensors": int(row["num_valid_sensors"]),
                    "active_sensors": "",
                    "manual_excluded_sensors": "",
                    "quality_excluded_sensors": "",
                    "skipped_sensors": "",
                    "Prec.": row["point_precision"],
                    "Rec.": row["point_recall"],
                    "F1": row["point_f1"],
                    "eTaP": row["eTaP"],
                    "eTaR": row["eTaR"],
                    "eTaF1": row["eTaF1"],
                    "FPA": row["FPA"],
                    "Scen.": row["scenario_detection_rate"],
                    "system_benign_alarm_rate": row["system_benign_alarm_rate"],
                    "system_attack_alarm_rate": row["system_attack_alarm_rate"],
                    "system_total_alarm_rate": row["system_total_alarm_rate"],
                    "num_system_alarm_intervals": row["num_system_alarm_intervals"],
                    "longest_system_alarm_interval": row["longest_system_alarm_interval"],
                }
            )

    by_entry = {
        (entry["detector"], entry["sensor_filter"], entry["aggregation"], int(entry["k"])): entry
        for entry in active_entries
    }
    for _, row in system_df.iterrows():
        key = (row["detector"], row["sensor_filter"], row["aggregation"], int(row["k"]))
        entry = by_entry.get(key)
        if entry:
            rows.append(_summary_row_for_scope(row, entry))

    new_df = pd.DataFrame(rows)
    if summary_path.exists():
        old = pd.read_csv(summary_path)
        if not old.empty and not new_df.empty:
            keys = ["sensor_scope_name", "detector", "aggregation", "k", "sensor_filter"]
            new_keys = set(map(tuple, new_df[keys].astype(str).to_numpy()))
            keep = ~old[keys].astype(str).apply(tuple, axis=1).isin(new_keys)
            new_df = pd.concat([old[keep], new_df], ignore_index=True)
    if not new_df.empty:
        order = {
            "all_valid": 0,
            "geco_ait_exclusion": 1,
            "geco_plus_quality": 2,
        }
        new_df["_order"] = new_df["sensor_scope_name"].map(order).fillna(99)
        new_df = new_df.sort_values(["_order", "aggregation", "k"]).drop(columns=["_order"])
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(summary_path, index=False)
    write_sensor_scope_markdown(root / "sensor_scope_ablation_summary.md", new_df)
    paper = REPO_ROOT / "paper_artifacts"
    paper.mkdir(parents=True, exist_ok=True)
    new_df.to_csv(paper / "sensor_scope_ablation_summary.csv", index=False)
    return new_df


def write_sensor_scope_markdown(path: Path, df: pd.DataFrame) -> None:
    if df.empty:
        return
    lines = [
        "# Sensor-Scope Ablation Summary",
        "",
        "These rows are ablations, not replacements for the all-valid main result.",
        "",
        "| scope | agg | active | Prec. | Rec. | F1 | eTaP | eTaR | eTaF1 | FPA | Scen. | benign_alarm | longest_alarm |",
        "| --- | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for _, row in df.iterrows():
        agg = "OR" if row["aggregation"] == "or" else f"k={int(row['k'])}"
        lines.append(
            f"| {row['sensor_scope_name']} | {agg} | {int(row['num_active_sensors']) if str(row['num_active_sensors']) else ''} | "
            f"{row['Prec.']:.1f} | {row['Rec.']:.1f} | {row['F1']:.1f} | {row['eTaP']:.1f} | "
            f"{row['eTaR']:.1f} | {row['eTaF1']:.1f} | {row['FPA']:.0f} | {row['Scen.']:.1f} | "
            f"{row['system_benign_alarm_rate']:.3f} | {row['longest_system_alarm_interval']:.0f} |"
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def run_config(args: argparse.Namespace, train_diag: DetectionSplit, calib: DetectionSplit, test: DetectionSplit, calib_info: dict[str, Any]) -> dict[str, Any]:
    attack_count = len(to_intervals(test.labels)) if test.labels is not None else None
    return {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "cli_args": vars(args),
        "git_commit": git_commit(),
        "package_versions": package_versions(),
        "calibration_source": calib_info["calibration_source"],
        "calibration_split": calib_info["calibration_split"],
        "used_train_calibration": calib_info["used_train_calibration"],
        "s": float(args.s),
        "g": float(args.g),
        "symbolic_target_mode": args.symbolic_target_mode,
        "detectors": parse_csv_list(args.detectors),
        "sensor_scope_name": args.sensor_scope_name,
        "include_sensors": parse_csv_list(args.include_sensors),
        "exclude_sensors": parse_csv_list(args.exclude_sensors),
        "sensor_filter": args.sensor_filter,
        "sensor_filter_threshold": args.sensor_filter_threshold,
        "max_calib_alarm_rate": args.max_calib_alarm_rate,
        "system_aggregation": args.system_aggregation,
        "k": int(args.k),
        "label_attack_interval_count": attack_count,
        "inputs": {
            "train_root": {"path": str(args.train_root) if args.train_root else None, "source": train_diag.source},
            "calib_root": {"path": str(args.calib_root or args.distill_root), "source": calib.source},
            "test_distill_root": str(args.test_distill_root) if args.test_distill_root else None,
            "test_export": {"path": str(args.test_export), "source": test.source},
            "audit_root": str(args.audit_root),
        },
        "data_shapes": {
            "diag_x": list(train_diag.x_current_raw.shape),
            "calib_x": list(calib.x_current_raw.shape),
            "test_x": list(test.x_current_raw.shape),
            "test_labels": None if test.labels is None else list(test.labels.shape),
        },
    }


def run_one_sensor_outputs(
    *,
    args: argparse.Namespace,
    result: dict[str, Any],
    test: DetectionSplit,
    out_dir: Path,
    specs: list[DetectorSpec],
) -> None:
    labels = test.labels
    assert labels is not None
    system_df, excluded, alarms = system_ablation_rows(
        results=[result],
        labels=labels,
        specs=specs,
        filter_modes=[args.sensor_filter],
        aggregations=[(args.system_aggregation, args.k)],
        r2_threshold=args.sensor_filter_threshold,
        max_calib_alarm_rate=args.max_calib_alarm_rate,
    )
    system_df.to_csv(out_dir / "lit101_detection_summary.csv", index=False)
    pd.DataFrame(per_sensor_metric_rows([result])).to_csv(out_dir / "lit101_per_sensor_metrics.csv", index=False)
    pd.DataFrame(selected_equation_rows([result])).to_csv(out_dir / "lit101_selected_equations.csv", index=False)
    write_json(out_dir / "lit101_skipped_sensors.json", result["skipped"])
    write_json(out_dir / "lit101_excluded_sensors.json", excluded)
    if args.sweep_lit101_s:
        lit101_s_sweep(result, labels, out_dir / "lit101_s_sweep.csv", g=args.g)
    if args.make_lit101_figure:
        write_lit101_figure(
            out_dir / "LIT101_diagnostic_next.pdf",
            result,
            test,
            ["MLP", "Sym-Raw-next", "Sym-MLP-next"],
            "LIT101 next-value detectors",
        )
        write_lit101_figure(
            out_dir / "LIT101_diagnostic_delta.pdf",
            result,
            test,
            ["MLP", "Sym-Raw-delta", "Sym-MLP-delta"],
            "LIT101 delta detectors",
        )
    print_system_table(system_df)


def run_all_sensor_outputs(
    *,
    args: argparse.Namespace,
    results: list[dict[str, Any]],
    test: DetectionSplit,
    out_dir: Path,
    specs: list[DetectorSpec],
    scope_info: dict[str, Any],
) -> None:
    labels = test.labels
    assert labels is not None
    focused = is_sensor_scope_run(args)
    filter_modes = [args.sensor_filter] if focused else ["none", "calib-r2", "alarm-burden"]
    aggregations = [(args.system_aggregation, args.k)] if focused else [("or", 1), ("kofn", 2), ("kofn", 3)]
    system_df, excluded, alarms = system_ablation_rows(
        results=results,
        labels=labels,
        specs=specs,
        filter_modes=filter_modes,
        aggregations=aggregations,
        r2_threshold=args.sensor_filter_threshold,
        max_calib_alarm_rate=args.max_calib_alarm_rate,
    )
    scope_name = args.sensor_scope_name or ("manual_scope" if focused else "all_valid")
    if not system_df.empty:
        system_df.insert(0, "sensor_scope_name", scope_name)
        system_df["sensor_filter_threshold"] = args.sensor_filter_threshold
        system_df["max_calib_alarm_rate"] = args.max_calib_alarm_rate
    per_sensor_df = pd.DataFrame(per_sensor_metric_rows(results))
    burden_df = pd.DataFrame(per_sensor_alarm_burden_rows(results, labels))
    params_df = pd.DataFrame(cusum_param_rows(results))
    equations_df = pd.DataFrame(selected_equation_rows(results))
    skipped = [item for res in results for item in res["skipped"]]

    system_df.to_csv(out_dir / "system_detection_ablation_summary.csv", index=False)
    system_df.to_csv(out_dir / "system_alarm_burden.csv", index=False)
    if focused:
        system_df.to_csv(out_dir / "system_detection_summary.csv", index=False)
    else:
        system_df[(system_df["sensor_filter"] == "none") & (system_df["aggregation"] == "or")].to_csv(
            out_dir / "system_detection_summary_all_valid.csv", index=False
        )
        system_df[system_df["sensor_filter"] == "calib-r2"].to_csv(
            out_dir / "system_detection_summary_filter_calib_r2.csv", index=False
        )
        system_df[system_df["sensor_filter"] == "alarm-burden"].to_csv(
            out_dir / "system_detection_summary_filter_alarm_burden.csv", index=False
        )
        # Backward-compatible summary name points to the all-valid OR main result.
        system_df[(system_df["sensor_filter"] == "none") & (system_df["aggregation"] == "or")].to_csv(
            out_dir / "system_detection_summary.csv", index=False
        )
    write_system_markdown(out_dir / "system_detection_ablation_summary.md", system_df)
    write_system_markdown(out_dir / "system_detection_summary.md", system_df)
    per_sensor_df.to_csv(out_dir / "per_sensor_metrics.csv", index=False)
    burden_df.to_csv(out_dir / "per_sensor_alarm_burden.csv", index=False)
    params_df.to_csv(out_dir / "per_sensor_cusum_params.csv", index=False)
    equations_df.to_csv(out_dir / "selected_equations_diagnostics.csv", index=False)
    write_json(out_dir / "skipped_sensors.json", skipped)
    write_json(out_dir / "excluded_sensors.json", excluded)
    write_alarms_npz(out_dir / "system_alarms.npz", alarms)
    if focused:
        active_entries = active_sensor_entries(
            results=results,
            specs=specs,
            system_df=system_df,
            excluded=excluded,
            scope_info=scope_info,
            sensor_filter=args.sensor_filter,
        )
        write_json(out_dir / "active_sensors.json", active_entries)
        remaining_df = remaining_sensor_burden(burden_df, active_entries)
        remaining_df.to_csv(out_dir / "remaining_sensor_burden.csv", index=False)
        scope_summary = update_sensor_scope_summary(out_dir=out_dir, system_df=system_df, active_entries=active_entries)
        write_sensor_scope_markdown(out_dir / "system_detection_summary.md", scope_summary[scope_summary["sensor_scope_name"] == scope_name])

    agreements = []
    for detector in ["Sym-MLP-next", "Sym-MLP-delta"]:
        if any(detector in res["detectors"] for res in results):
            agreement = agreement_analysis(results, labels, detector)
            suffix = "next" if detector.endswith("next") else "delta"
            write_json(out_dir / f"mlp_symmlp_agreement_{suffix}.json", agreement)
            agreements.append(agreement)
    write_agreement_markdown(out_dir / "mlp_symmlp_agreement.md", agreements)
    print_system_table(system_df)
    if args.write_paper_artifacts:
        lit101_result = next((res for res in results if res["sensor"] == "LIT101"), None)
        write_paper_artifacts(system_df, lit101_result)


def run(args: argparse.Namespace) -> int:
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    train_diag, calib, test, calib_info = load_split_inputs(args)
    if test.labels is None:
        raise ValueError("Test labels are required for detection metrics")
    attack_count = len(to_intervals(test.labels))
    if attack_count != 36:
        print(f"SWaT attack interval count from labels: {attack_count} (expected 36 in the original 1-second benchmark)")
    else:
        print("SWaT attack interval count from labels: 36")
    print(f"CUSUM calibration source: {calib.source} split={calib_info['calibration_split']}")

    specs = filter_detector_specs(detector_specs(args.symbolic_target_mode), args.detectors)
    all_target_sensors = list(calib.target_columns)
    base_sensors = [args.sensor] if args.sensor else []
    if args.all_sensors:
        base_sensors = all_target_sensors
    sensors, scope_info = select_candidate_sensors(
        base_sensors,
        include_sensors=args.include_sensors,
        exclude_sensors=args.exclude_sensors,
    )
    scope_info["sensor_scope_name"] = args.sensor_scope_name or ("manual_scope" if is_sensor_scope_run(args) else "all_valid")
    scope_info["all_target_sensors"] = all_target_sensors
    scope_info["requested_detectors"] = [spec.name for spec in specs]
    if not sensors:
        raise ValueError("Provide --sensor or --all-sensors")
    config = run_config(args, train_diag, calib, test, calib_info)
    config["sensor_scope"] = scope_info
    config["detector_variants"] = [spec.name for spec in specs]
    config["candidate_sensors"] = sensors
    write_json(out_dir / "run_config.json", config)
    print(f"Detector variants: {[spec.name for spec in specs]}")
    if is_sensor_scope_run(args):
        print(f"Sensor scope: {scope_info['sensor_scope_name']}")
        print(f"Manual excluded sensors: {scope_info['manual_excluded_sensors']}")
        if scope_info["unknown_include_sensors"] or scope_info["unknown_exclude_sensors"]:
            print(
                "Unknown requested sensors: "
                f"include={scope_info['unknown_include_sensors']} exclude={scope_info['unknown_exclude_sensors']}"
            )
    print(f"Candidate sensors ({len(sensors)}): {', '.join(sensors)}")

    results = []
    for sensor in sensors:
        print(f"\nEvaluating sensor {sensor}")
        result = evaluate_sensor(
            sensor=sensor,
            train_diag=train_diag,
            calib=calib,
            test=test,
            audit_root=args.audit_root,
            specs=specs,
            s=args.s,
            g=args.g,
        )
        results.append(result)
        if sensor == "LIT101":
            for diag in result["equation_diagnostics"]:
                if diag.get("status") == "ok":
                    print(f"{diag['detector']} selected {diag['target_source']}: {diag.get('equation')}")
                    print(
                        f"  calib_r2={diag.get('calib_r2')} test_benign_r2={diag.get('test_benign_r2')} "
                        f"fidelity_calib_r2={diag.get('fidelity_calib_r2')} "
                        f"fidelity_test_benign_r2={diag.get('fidelity_test_benign_r2')}"
                    )
            for detector, payload in result["detectors"].items():
                p = payload["params"]
                print(
                    f"  {detector} CUSUM: delta={p.delta:.6g} T={p.threshold:.6g} "
                    f"cap={p.growth_cap:.6g} max_calib={p.max_calib_cusum:.6g}"
                )
        if args.sensor and not args.all_sensors:
            run_one_sensor_outputs(args=args, result=result, test=test, out_dir=out_dir, specs=specs)

    if args.all_sensors:
        run_all_sensor_outputs(args=args, results=results, test=test, out_dir=out_dir, specs=specs, scope_info=scope_info)
    return 0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate MLP and symbolic predictors with GECO-style CUSUM.")
    parser.add_argument("--distill-root", default="artifacts/model_exports/swat/distillation/val20_overlap")
    parser.add_argument("--train-root", default=None, help="Optional diagnostic split for equation R2 reporting.")
    parser.add_argument("--calib-root", default=None, help="Held-out calibration split. Defaults to --distill-root.")
    parser.add_argument("--test-distill-root", default=None)
    parser.add_argument("--test-export", default="artifacts/model_exports/swat/mlp_current_h1_val20/test")
    parser.add_argument("--audit-root", default="artifacts/symbolic_equations/swat/full_sensor_audit")
    parser.add_argument("--out", default="artifacts/detection/swat")
    parser.add_argument("--sensor", default=None)
    parser.add_argument("--all-sensors", action="store_true")
    parser.add_argument("--include-sensors", default=None, help="Comma-separated target sensors eligible for evaluation.")
    parser.add_argument("--exclude-sensors", default=None, help="Comma-separated target sensors to remove from eligibility.")
    parser.add_argument("--sensor-scope-name", default=None, help="Name recorded for manual sensor-scope ablations.")
    parser.add_argument(
        "--detectors",
        default=None,
        help="Comma-separated detectors to evaluate, e.g. MLP,Sym-Raw-delta.",
    )
    parser.add_argument("--s", type=float, default=1.42)
    parser.add_argument("--g", type=float, default=5.98)
    parser.add_argument("--workers", type=int, default=1, help="Accepted for CLI compatibility; current implementation is serial.")
    parser.add_argument("--sweep-lit101-s", action="store_true")
    parser.add_argument("--make-lit101-figure", action="store_true")
    parser.add_argument("--write-paper-artifacts", action="store_true")
    parser.add_argument("--symbolic-target-mode", default="both", choices=["next", "delta", "both"])
    parser.add_argument("--sensor-filter", default="none", choices=["none", "calib-r2", "alarm-burden"])
    parser.add_argument("--sensor-filter-threshold", type=float, default=0.3)
    parser.add_argument("--max-calib-alarm-rate", type=float, default=0.05)
    parser.add_argument("--system-aggregation", default="or", choices=["or", "kofn"])
    parser.add_argument("--k", type=int, default=2)
    parser.add_argument("--calibration-split", default="val")
    parser.add_argument("--allow-train-calibration", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    raise SystemExit(run(parse_args()))
