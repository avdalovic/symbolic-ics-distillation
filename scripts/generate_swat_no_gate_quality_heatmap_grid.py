#!/usr/bin/env python
"""Generate a dense SWaT no-holdout-gate S/G grid for quality-filter figures.

This script does not run PySR. It reads the no-gate selected monitored models
created by scripts/check_swat_no_holdout_quality_gate.py and evaluates the
final-style GeCo-matched + actuator-persistence monitor set over a dense S/G
CUSUM grid.
"""
from __future__ import annotations

import argparse
import importlib.util
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def load_module(path: Path, name: str):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


POST = load_module(REPO_ROOT / "scripts" / "run_swat_1sec_delta_posthoc_ablation.py", "swat_posthoc_for_no_gate_dense")

_GLOBALS: dict[str, Any] = {}


def safe_float(value: Any) -> float:
    try:
        return float(value)
    except Exception:
        return float("nan")


def model_from_row(row: dict[str, Any]) -> Any:
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
        source=str(row.get("source", "no_holdout_quality_gate")),
    )


def parse_grid(text: str, default: list[float]) -> list[float]:
    if not str(text).strip():
        return default
    return [float(tok.strip()) for tok in str(text).split(",") if tok.strip()]


def default_s_values() -> list[float]:
    return [0.0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.42, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]


def default_g_values() -> list[float]:
    return [0.0, 0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 5.98, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 25.0]


def init_worker(train_matrix: np.ndarray, test_matrix: np.ndarray, labels: np.ndarray, delta: np.ndarray, max_train_cusum: np.ndarray) -> None:
    _GLOBALS["train_matrix"] = train_matrix
    _GLOBALS["test_matrix"] = test_matrix
    _GLOBALS["labels"] = labels
    _GLOBALS["delta"] = delta
    _GLOBALS["max_train_cusum"] = max_train_cusum


def evaluate_point(pair: tuple[float, float]) -> dict[str, Any]:
    s, g = pair
    labels = _GLOBALS["labels"]
    alarm_matrix, threshold, growth_cap, max_test = POST.run_batch_cusum(
        _GLOBALS["test_matrix"],
        _GLOBALS["delta"],
        _GLOBALS["max_train_cusum"],
        s=float(s),
        g=float(g),
    )
    system_alarm = np.max(alarm_matrix, axis=1).astype(np.int8)
    metrics = POST.compute_detection_metrics(labels, system_alarm, expand_steps=60)
    return {
        "variant": "geco_matched_plus_all_actuator_persistence_no_holdout_quality_gate_dense",
        "S": float(s),
        "G": float(g),
        "num_monitored": int(alarm_matrix.shape[1]),
        "monitored_sensors": 14,
        "monitored_actuators": 25,
        "Precision": metrics["point_precision"],
        "Recall": metrics["point_recall"],
        "F1": metrics["point_f1"],
        "eTaP": metrics["eTaP"],
        "eTaR": metrics["eTaR"],
        "eTaF1": metrics["eTaF1"],
        "FPA": metrics["FPA"],
        "Scen": metrics["scenario_detection_rate"],
        **POST.point_counts(labels, system_alarm),
        **{f"system_{k}": v for k, v in POST.alarm_burden(system_alarm, labels).items()},
        "threshold_min": float(np.min(threshold)) if threshold.size else np.nan,
        "threshold_max": float(np.max(threshold)) if threshold.size else np.nan,
        "growth_cap_min": float(np.min(growth_cap)) if growth_cap.size else np.nan,
        "growth_cap_max": float(np.max(growth_cap)) if growth_cap.size else np.nan,
        "max_test_cusum_max": float(np.max(max_test)) if max_test.size else np.nan,
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Dense SWaT no-gate quality-filter S/G grid.")
    parser.add_argument("--models", default="artifacts/swat_1sec/no_holdout_quality_gate_check/monitored_models_geco_matched_plus_all_actuator_persistence_no_holdout_quality_gate.csv")
    parser.add_argument("--out", default="artifacts/swat_1sec/no_holdout_quality_gate_check/dense_grid_geco_matched_plus_all_actuator_persistence_no_holdout_quality_gate.csv")
    parser.add_argument("--experiment", default="configs/experiment/swat_mlp_current_val20.yaml")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--s-values", default="")
    parser.add_argument("--g-values", default="")
    parser.add_argument("--jobs", type=int, default=max(1, min(8, (mp.cpu_count() or 2) // 4)))
    args = parser.parse_args()

    model_df = pd.read_csv(args.models)
    models = [model_from_row(row) for row in model_df.to_dict("records")]
    load_args = argparse.Namespace(experiment=args.experiment, train_csv=args.train_csv, test_csv=args.test_csv)
    arrays = POST.load_arrays(load_args)
    cache = POST.residual_cache_for_models(arrays, models)
    train_matrix, test_matrix = POST.stack_residuals(models, cache)
    delta, max_train_cusum = POST.fit_batch_base(train_matrix)
    labels = arrays["labels"]

    s_values = parse_grid(args.s_values, default_s_values())
    g_values = parse_grid(args.g_values, default_g_values())
    pairs = [(s, g) for s in s_values for g in g_values]
    print(f"Evaluating {len(pairs)} S/G points for {len(models)} monitored variables with jobs={args.jobs}", flush=True)

    if int(args.jobs) <= 1:
        init_worker(train_matrix, test_matrix, labels, delta, max_train_cusum)
        rows = [evaluate_point(pair) for pair in pairs]
    else:
        ctx = mp.get_context("fork")
        with ctx.Pool(
            processes=int(args.jobs),
            initializer=init_worker,
            initargs=(train_matrix, test_matrix, labels, delta, max_train_cusum),
        ) as pool:
            rows = list(pool.imap_unordered(evaluate_point, pairs, chunksize=1))

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame(rows).sort_values(["S", "G"]).reset_index(drop=True)
    df.to_csv(out, index=False)
    print(f"Saved: {out}")
    fpa5 = df[df["FPA"] <= 5]
    if not fpa5.empty:
        best = fpa5.sort_values(["eTaF1", "F1"], ascending=False).iloc[0]
        print("Best FPA<=5:", best[["S", "G", "F1", "eTaF1", "FPA", "Scen"]].to_dict())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
