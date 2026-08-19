#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import re
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
from ics_symbolic_distill.detection.swat1s_delta_sampling import reconstruct_next_from_delta


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


SWAT = import_module("guard_swat_posthoc", REPO_ROOT / "scripts" / "run_swat_1sec_delta_posthoc_ablation.py")
WADI_FULL = import_module("guard_wadi_full", REPO_ROOT / "scripts" / "run_wadi_1sec_delta_full.py")
WADI_POST = import_module("guard_wadi_posthoc", REPO_ROOT / "scripts" / "run_wadi_1sec_delta_posthoc_ablation.py")
BAT = import_module("guard_batadal_full", REPO_ROOT / "scripts" / "run_batadal_delta_full.py")
HAI = import_module("guard_hai_full", REPO_ROOT / "scripts" / "run_hai_1sec_delta_full.py")
SEED_GRID = import_module("guard_seed_grid", REPO_ROOT / "scripts" / "evaluate_seed_grid.py")
V2 = import_module("guard_paper_v2", REPO_ROOT / "scripts" / "generate_paper_artifacts_v2.py")


RULE_NAME = "score_self_attractor_holdout_mae_fallback"
HEADLINE = {
    "swat": (1.20, 15.0),
    "wadi": (1.20, 25.0),
    "batadal": (1.40, 2.00),
    "hai": (2.50, 12.0),
}
GECO = {
    "swat": (1.42, 5.98),
    "wadi": (1.32, 9.74),
    "batadal": (1.39, 2.16),
    "hai": (8.02147642080159, 1.4376682451456457),
}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def contains_self(target: str, equation: str) -> bool:
    return bool(re.search(rf"(?<![A-Za-z0-9_]){re.escape(str(target))}(?![A-Za-z0-9_])", str(equation)))


def finite_float(value: Any, default: float = float("nan")) -> float:
    try:
        out = float(value)
    except Exception:
        return default
    return out if math.isfinite(out) else default


def should_replace(row: pd.Series) -> bool:
    target = str(row["target"])
    equation = str(row.get("sympy_format", row.get("equation", "")))
    holdout_mae = finite_float(row.get("holdout_mae"))
    baseline = finite_float(row.get("baseline_holdout_mae"))
    return contains_self(target, equation) and baseline > 0 and holdout_mae > baseline


def metric_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Precision" not in out.columns and "Prec" in out.columns:
        out["Precision"] = out["Prec"]
    if "Recall" not in out.columns and "Rec" in out.columns:
        out["Recall"] = out["Rec"]
    return out


def pick_best_holdout_mae(candidates: list[dict[str, Any]]) -> dict[str, Any] | None:
    if not candidates:
        return None
    return min(candidates, key=lambda r: (finite_float(r.get("holdout_mae"), float("inf")), finite_float(r.get("complexity"), float("inf"))))


def passes_replacement_guard(
    old_row: pd.Series,
    candidate: dict[str, Any],
    *,
    min_delta_r2: float,
    min_mae_improvement: float,
) -> bool:
    old_mae = finite_float(old_row.get("holdout_mae"), float("inf"))
    new_mae = finite_float(candidate.get("holdout_mae"), float("inf"))
    old_r2 = finite_float(old_row.get("holdout_r2"), float("-inf"))
    new_r2 = finite_float(candidate.get("holdout_r2"), float("-inf"))
    if not (np.isfinite(old_mae) and np.isfinite(new_mae) and np.isfinite(old_r2) and np.isfinite(new_r2)):
        return False
    if new_mae > old_mae * (1.0 - min_mae_improvement):
        return False
    if new_r2 < old_r2 + min_delta_r2:
        return False
    return True


def swat_candidate_rows(arrays: dict[str, Any], pareto_root: Path, target: str, max_complexity: int) -> list[dict[str, Any]]:
    pareto = pareto_root / f"{target}_sensors_delta_actuators_next" / "pareto_front_scored.csv"
    if not pareto.exists():
        return []
    df = pd.read_csv(pareto)
    feature_columns = arrays["feature_columns"]
    fit_idx = arrays["fit_idx"]
    holdout_idx = arrays["holdout_idx"]
    idx_target = feature_columns.index(target)
    y_all = (arrays["train_next"][:, idx_target] - arrays["train_current"][:, idx_target]).astype(np.float64)
    baseline_mae = float(np.mean(np.abs(y_all[holdout_idx])))
    out: list[dict[str, Any]] = []
    for idx, row in df.iterrows():
        complexity = finite_float(row.get("complexity"))
        if not np.isfinite(complexity) or complexity > max_complexity:
            continue
        equation = str(row.get("sympy_format", row.get("equation", "")))
        pred = evaluate_equation(equation, feature_columns, arrays["train_current"]).astype(np.float64)
        if not np.isfinite(pred).all():
            continue
        residual_train = np.abs(y_all - pred)
        median = float(np.median(residual_train))
        p99 = float(np.percentile(residual_train, 99))
        tail = p99 / max(median, 1e-9)
        if tail > 50.0:
            continue
        intervals, _ = SWAT.holdout_alarm_summary(residual_train[fit_idx], residual_train[holdout_idx])
        if intervals > 0:
            continue
        metrics = SWAT.regression_metrics(y_all[holdout_idx], pred[holdout_idx])
        out.append(
            {
                "target": target,
                "variable_type": "sensor",
                "target_mode": "sensors_delta_actuators_next",
                "equation": str(row.get("equation", "")),
                "sympy_format": equation,
                "complexity": complexity,
                "loss": finite_float(row.get("loss")),
                "score": finite_float(row.get("score")),
                "holdout_r2": metrics["r2"],
                "holdout_mae": metrics["mae"],
                "baseline_holdout_mae": baseline_mae,
                "residual_tail_ratio": tail,
                "residual_p99": p99,
                "residual_median": median,
                "selection_reason": RULE_NAME,
                "pareto_csv": str(pareto),
            }
        )
    return out


def wadi_candidate_rows(arrays: dict[str, Any], pareto_root: Path, target: str, max_complexity: int) -> list[dict[str, Any]]:
    pareto = WADI_FULL.pareto_dir(pareto_root, target) / "pareto_front_scored.csv"
    if not pareto.exists():
        return []
    df = pd.read_csv(pareto)
    fit_idx = arrays["fit_idx"]
    holdout_idx = arrays["holdout_idx"]
    _, y_all = WADI_FULL.target_values(arrays, target, split="train")
    baseline_mae = float(np.mean(np.abs(y_all[holdout_idx])))
    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        complexity = finite_float(row.get("complexity"))
        if not np.isfinite(complexity) or complexity > max_complexity:
            continue
        equation = str(row.get("sympy_format", row.get("equation", "")))
        equation_safe = WADI_FULL.equation_original_to_safe(equation, arrays["original_to_safe"])
        pred = evaluate_equation(equation_safe, arrays["safe_feature_columns"], arrays["train_current"]).astype(np.float64)
        if not np.isfinite(pred).all():
            continue
        residual_train = WADI_FULL.prediction_residual(arrays, target, equation, split="train")
        params = fit_cusum_params(residual_train[fit_idx], s=WADI_FULL.GECO_S, g=WADI_FULL.GECO_G)
        _, holdout_alarm = run_cusum(residual_train[holdout_idx], params)
        if int(np.sum(holdout_alarm)) > 0:
            continue
        median = float(np.median(residual_train))
        p99 = float(np.percentile(residual_train, 99))
        tail = p99 / max(median, 1e-9)
        if tail > 50.0:
            continue
        metrics = WADI_FULL.regression_metrics(y_all[holdout_idx], pred[holdout_idx])
        out.append(
            {
                "target": target,
                "variable_type": "sensor",
                "target_mode": WADI_FULL.TARGET_MODE,
                "equation": str(row.get("equation", "")),
                "sympy_format": equation,
                "complexity": complexity,
                "loss": finite_float(row.get("loss")),
                "score": finite_float(row.get("score")),
                "holdout_r2": metrics["r2"],
                "holdout_mae": metrics["mae"],
                "baseline_holdout_mae": baseline_mae,
                "residual_tail_ratio": tail,
                "residual_p99": p99,
                "residual_median": median,
                "selection_reason": RULE_NAME,
                "pareto_csv": str(pareto),
            }
        )
    return out


def batadal_candidate_rows(arrays: dict[str, Any], pareto_root: Path, target: str, max_complexity: int) -> list[dict[str, Any]]:
    # Reuse the audited helper from the BATADAL-specific script if present.
    mod = import_module("guard_batadal_selection", REPO_ROOT / "scripts" / "analyze_batadal_selection_ablation.py")
    return mod.passing_candidates(arrays, pareto_root, target, max_complexity)


def hai_candidate_rows(arrays: dict[str, Any], pareto_root: Path, target: str, max_complexity: int) -> list[dict[str, Any]]:
    pareto = HAI.pareto_dir(pareto_root, target) / "pareto_front_scored.csv"
    if not pareto.exists():
        return []
    df = pd.read_csv(pareto)
    holdout_idx = arrays["holdout_idx"]
    _, y_all = HAI.target_values(arrays, target, split="train")
    baseline_mae = float(np.mean(np.abs(y_all[holdout_idx]))) if holdout_idx.size else 0.0
    out: list[dict[str, Any]] = []
    for _, row in df.iterrows():
        complexity = finite_float(row.get("complexity"))
        if not np.isfinite(complexity) or complexity > max_complexity:
            continue
        equation = str(row.get("sympy_format", row.get("equation", "")))
        equation_safe = HAI.equation_original_to_safe(equation, arrays["original_to_safe"])
        pred = evaluate_equation(equation_safe, arrays["safe_feature_columns"], arrays["train_current"]).astype(np.float64)
        if not np.isfinite(pred).all():
            continue
        residual_train = HAI.prediction_residual(arrays, target, equation, split="train")
        residual_fit_blocks = HAI.train_residual_blocks(arrays, residual_train, fit_only=True)
        residual_holdout_blocks = HAI.train_residual_blocks(arrays, residual_train, fit_only=False)
        params = HAI.fit_cusum_params_sequences(residual_fit_blocks, s=GECO["hai"][0], g=GECO["hai"][1])
        _, holdout_alarm, _, _ = HAI.run_cusum_sequences(residual_holdout_blocks, params)
        if int(np.sum(holdout_alarm)) > 0:
            continue
        median = float(np.median(residual_train))
        p99 = float(np.percentile(residual_train, 99))
        tail = p99 / max(median, 1e-9)
        if tail > 50.0:
            continue
        metrics = HAI.regression_metrics(y_all[holdout_idx], pred[holdout_idx]) if holdout_idx.size else {"r2": 0.0, "mae": 0.0}
        out.append(
            {
                "target": target,
                "variable_type": "sensor",
                "target_mode": HAI.TARGET_MODE,
                "equation": str(row.get("equation", "")),
                "sympy_format": equation,
                "complexity": complexity,
                "loss": finite_float(row.get("loss")),
                "score": finite_float(row.get("score")),
                "holdout_r2": metrics["r2"],
                "holdout_mae": metrics["mae"],
                "baseline_holdout_mae": baseline_mae,
                "residual_tail_ratio": tail,
                "residual_p99": p99,
                "residual_median": median,
                "selection_reason": RULE_NAME,
                "pareto_csv": str(pareto),
            }
        )
    return out


def apply_guard(
    dataset: str,
    selected_path: Path,
    arrays: dict[str, Any],
    pareto_root: Path,
    out_dir: Path,
    max_complexity: int,
    min_delta_r2: float,
    min_mae_improvement: float,
) -> tuple[Path, pd.DataFrame]:
    df = pd.read_csv(selected_path)
    audit_rows = []
    for idx, row in df.iterrows():
        if not should_replace(row):
            continue
        target = str(row["target"])
        if dataset == "swat":
            candidates = swat_candidate_rows(arrays, pareto_root, target, max_complexity)
        elif dataset == "wadi":
            candidates = wadi_candidate_rows(arrays, pareto_root, target, max_complexity)
        elif dataset == "batadal":
            candidates = batadal_candidate_rows(arrays, pareto_root, target, max_complexity)
        elif dataset == "hai":
            candidates = hai_candidate_rows(arrays, pareto_root, target, max_complexity)
        else:
            raise ValueError(dataset)
        candidates = [
            c
            for c in candidates
            if passes_replacement_guard(
                row,
                c,
                min_delta_r2=min_delta_r2,
                min_mae_improvement=min_mae_improvement,
            )
        ]
        replacement = pick_best_holdout_mae(candidates)
        if replacement is None:
            audit_rows.append(
                {
                    "dataset": dataset,
                    "target": target,
                    "status": "flagged_no_replacement_passing_guard",
                    "old_equation": row.get("equation"),
                    "old_holdout_mae": row.get("holdout_mae"),
                    "old_holdout_r2": row.get("holdout_r2"),
                    "old_baseline_holdout_mae": row.get("baseline_holdout_mae"),
                }
            )
            continue
        for col, value in replacement.items():
            if col in df.columns:
                df.at[idx, col] = value
        audit_rows.append(
            {
                "dataset": dataset,
                "target": target,
                "status": "replaced",
                "old_equation": row.get("equation"),
                "new_equation": replacement.get("equation"),
                "old_complexity": row.get("complexity"),
                "new_complexity": replacement.get("complexity"),
                "old_score": row.get("score"),
                "new_score": replacement.get("score"),
                "old_holdout_r2": row.get("holdout_r2"),
                "new_holdout_r2": replacement.get("holdout_r2"),
                "old_holdout_mae": row.get("holdout_mae"),
                "new_holdout_mae": replacement.get("holdout_mae"),
                "old_baseline_holdout_mae": row.get("baseline_holdout_mae"),
                "new_baseline_holdout_mae": replacement.get("baseline_holdout_mae"),
            }
        )
    out_path = out_dir / dataset / "selected_equations_guarded.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    return out_path, pd.DataFrame(audit_rows)


def evaluate_swat(selected_path: Path, args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    cmd_args = argparse.Namespace(
        dataset="swat",
        run_dir=str(out_dir / "swat"),
        selected_csv=str(selected_path),
        train_csv=str(args.swat_train),
        test_csv=str(args.swat_test),
        out_dir=str(out_dir / "swat_eval"),
        seed=0,
        swat_experiment="configs/experiment/swat_mlp_current_val20.yaml",
        selected_path=str(selected_path),
    )
    payload = SEED_GRID.build_swat_payload(cmd_args)
    s_values = sorted({float(v) for v in V2.S_VALUES_EXT} | {HEADLINE["swat"][0], GECO["swat"][0]})
    g_values = sorted({float(v) for v in V2.G_VALUES_EXT} | {HEADLINE["swat"][1], GECO["swat"][1]})
    grid, _ = V2.evaluate_grid_from_residuals(
        train_matrix=payload["train_matrix"],
        test_matrices=payload["test_matrices"],
        label_arrays=payload["label_arrays"],
        s_values=s_values,
        g_values=g_values,
        expand_steps=60,
        counts=payload["counts"],
    )
    grid.insert(0, "dataset", "SWaT")
    return grid


def evaluate_wadi(selected_path: Path, args: argparse.Namespace, out_dir: Path) -> pd.DataFrame:
    cmd_args = argparse.Namespace(
        selected_path=str(selected_path),
        train_csv=str(args.wadi_train),
        test_csv=str(args.wadi_test),
    )
    payload = SEED_GRID.build_wadi_payload(cmd_args)
    s_values = sorted({float(v) for v in V2.S_VALUES_EXT} | {HEADLINE["wadi"][0], GECO["wadi"][0]})
    g_values = sorted({float(v) for v in V2.G_VALUES_EXT} | {HEADLINE["wadi"][1], GECO["wadi"][1]})
    grid, _ = V2.evaluate_grid_from_residuals(
        train_matrix=payload["train_matrix"],
        test_matrices=payload["test_matrices"],
        label_arrays=payload["label_arrays"],
        s_values=s_values,
        g_values=g_values,
        expand_steps=60,
        counts=payload["counts"],
    )
    grid.insert(0, "dataset", "WADI")
    return grid


def evaluate_batadal(selected_path: Path, out_dir: Path) -> pd.DataFrame:
    cmd_args = argparse.Namespace(
        selected_path=str(selected_path),
        train_csv="data/batadal/processed/train.csv",
        test_csv="data/batadal/processed/test_dataset04.csv,data/batadal/processed/test_dataset_test.csv",
    )
    payload = SEED_GRID.build_batadal_payload(cmd_args)
    s_values = sorted({float(v) for v in V2.S_VALUES_EXT} | {HEADLINE["batadal"][0], GECO["batadal"][0]})
    g_values = sorted({float(v) for v in V2.G_VALUES_EXT} | {HEADLINE["batadal"][1], GECO["batadal"][1]})
    grid, _ = V2.evaluate_grid_from_residuals(
        train_matrix=payload["train_matrix"],
        test_matrices=payload["test_matrices"],
        label_arrays=payload["label_arrays"],
        s_values=s_values,
        g_values=g_values,
        expand_steps=1,
        counts=payload["counts"],
    )
    grid.insert(0, "dataset", "BATADAL")
    return grid


def matrix_from_cache(rows: list[dict[str, Any]], cache: dict[str, dict[str, np.ndarray]], split: str) -> np.ndarray:
    return np.column_stack([cache[str(row["target"])][split] for row in rows]).astype(np.float32)


def evaluate_hai(selected_path: Path, arrays: dict[str, Any]) -> pd.DataFrame:
    rows = pd.read_csv(selected_path).to_dict("records")
    cache: dict[str, dict[str, np.ndarray]] = {}
    for row in rows:
        target = str(row["target"])
        if str(row.get("variable_type")) == "persistence" or str(row.get("target_mode")) == "persistence_next":
            selected_row, residuals = HAI.persistence_selected_row(arrays, target)
            cache[target] = residuals
        else:
            cache[target] = {
                "train": HAI.prediction_residual(arrays, target, str(row["sympy_format"]), split="train"),
                "test": HAI.prediction_residual(arrays, target, str(row["sympy_format"]), split="test"),
            }
    s_values = sorted({float(v) for v in V2.S_VALUES_EXT} | {HEADLINE["hai"][0], GECO["hai"][0]})
    g_values = sorted({float(v) for v in V2.G_VALUES_EXT} | {HEADLINE["hai"][1], GECO["hai"][1]})
    grid, _ = V2.evaluate_grid_from_residuals(
        train_matrix=matrix_from_cache(rows, cache, "train"),
        test_matrices=[matrix_from_cache(rows, cache, "test")],
        label_arrays=[arrays["labels"]],
        s_values=s_values,
        g_values=g_values,
        expand_steps=60,
        counts={
            "num_monitored": len(rows),
            "num_sensors": sum(str(r.get("variable_type")) == "sensor" for r in rows),
            "num_actuators": sum(str(r.get("variable_type")) == "persistence" for r in rows),
        },
    )
    grid.insert(0, "dataset", "HAI")
    return grid


def summarize_grid(dataset_key: str, grid: pd.DataFrame) -> pd.DataFrame:
    grid = metric_aliases(grid)
    rows = []
    for name, (s, g) in [("headline_point", HEADLINE[dataset_key]), ("geco_point", GECO[dataset_key])]:
        sub = grid[np.isclose(grid["S"], s) & np.isclose(grid["G"], g)]
        if not sub.empty:
            row = sub.iloc[0].to_dict()
            row["point_name"] = name
            rows.append(row)
    fpa8 = grid[grid["FPA"] <= 8].sort_values(["eTaF1", "F1", "Scen"], ascending=False)
    if not fpa8.empty:
        row = fpa8.iloc[0].to_dict()
        row["point_name"] = "best_fpa_le_8"
        rows.append(row)
    zero = grid[grid["FPA"] == 0].sort_values(["eTaF1", "F1", "Scen"], ascending=False)
    if not zero.empty:
        row = zero.iloc[0].to_dict()
        row["point_name"] = "best_zero_fpa"
        rows.append(row)
    return pd.DataFrame(rows)


def main() -> int:
    parser = argparse.ArgumentParser(description="Evaluate a global self-attractor holdout-MAE selection guard without changing original results.")
    parser.add_argument("--out-dir", default="artifacts/experiments/global_selection_guard")
    parser.add_argument("--max-complexity", type=int, default=15)
    parser.add_argument("--min-delta-r2", type=float, default=0.0)
    parser.add_argument("--min-mae-improvement", type=float, default=0.0)
    parser.add_argument("--swat-train", default="data/swat/raw/swat_train.csv")
    parser.add_argument("--swat-test", default="data/swat/raw/swat_test.csv")
    parser.add_argument("--wadi-train", default="data/wadi/raw/wadi_train.csv")
    parser.add_argument("--wadi-test", default="data/wadi/raw/wadi_test.csv")
    args = parser.parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    audits = []
    grids = []
    summaries = []

    # SWaT
    if Path(args.swat_train).exists() and Path(args.swat_test).exists():
        swat_arrays = SWAT.load_arrays(argparse.Namespace(experiment="configs/experiment/swat_mlp_current_val20.yaml", train_csv=str(args.swat_train), test_csv=str(args.swat_test)))
        selected, audit = apply_guard("swat", REPO_ROOT / "results/swat/selected_equations.csv", swat_arrays, REPO_ROOT / "results/swat/pareto_fronts", out_dir, args.max_complexity, args.min_delta_r2, args.min_mae_improvement)
        audits.append(audit)
        grid = evaluate_swat(selected, args, out_dir)
        grid.to_csv(out_dir / "swat" / "detection_grid.csv", index=False)
        grids.append(grid.assign(dataset_key="swat"))
        summaries.append(summarize_grid("swat", grid).assign(dataset_key="swat"))

    # WADI
    if Path(args.wadi_train).exists() and Path(args.wadi_test).exists():
        wadi_arrays = WADI_FULL.load_wadi_1sec_arrays(argparse.Namespace(train_csv=str(args.wadi_train), test_csv=str(args.wadi_test)))
        selected, audit = apply_guard("wadi", REPO_ROOT / "results/wadi/selected_equations.csv", wadi_arrays, REPO_ROOT / "results/wadi", out_dir, args.max_complexity, args.min_delta_r2, args.min_mae_improvement)
        audits.append(audit)
        grid = evaluate_wadi(selected, args, out_dir)
        grid.to_csv(out_dir / "wadi" / "detection_grid.csv", index=False)
        grids.append(grid.assign(dataset_key="wadi"))
        summaries.append(summarize_grid("wadi", grid).assign(dataset_key="wadi"))

    # BATADAL
    bat_arrays = BAT.load_batadal_arrays(argparse.Namespace(train_csv="data/batadal/processed/train.csv", test_csv="data/batadal/processed/test_dataset04.csv"))
    selected, audit = apply_guard("batadal", REPO_ROOT / "results/batadal/selected_equations.csv", bat_arrays, REPO_ROOT / "results/batadal", out_dir, args.max_complexity, args.min_delta_r2, args.min_mae_improvement)
    audits.append(audit)
    grid = evaluate_batadal(selected, out_dir)
    grid.to_csv(out_dir / "batadal" / "detection_grid.csv", index=False)
    grids.append(grid.assign(dataset_key="batadal"))
    summaries.append(summarize_grid("batadal", grid).assign(dataset_key="batadal"))

    # HAI
    hai_arrays = HAI.load_hai_arrays(argparse.Namespace(data_dir="data/hai/ipal", output_dir="artifacts/experiments/hai_baseline_seed0", seed=0, sample_size=None, timeout_minutes=60.0, target_wall_timeout_minutes=75.0, max_complexity=15, niterations=400, max_workers=2, pysr_procs=1, resume=True, force=False, skip_pysr=True, skip_detection_eval=False, single_target=None, smoke_targets=None, geco_model=None, promote_results=False, enable_division=False, sample_index_source_dir=None))
    selected, audit = apply_guard("hai", REPO_ROOT / "artifacts/experiments/hai_baseline_seed0/selected_equations.csv", hai_arrays, REPO_ROOT / "artifacts/experiments/hai_baseline_seed0", out_dir, args.max_complexity, args.min_delta_r2, args.min_mae_improvement)
    audits.append(audit)
    grid = evaluate_hai(selected, hai_arrays)
    grid.to_csv(out_dir / "hai" / "detection_grid.csv", index=False)
    grids.append(grid.assign(dataset_key="hai"))
    summaries.append(summarize_grid("hai", grid).assign(dataset_key="hai"))

    audit_table = pd.concat([a for a in audits if not a.empty], ignore_index=True) if audits else pd.DataFrame()
    audit_table.to_csv(out_dir / "replacement_audit.csv", index=False)
    grid_table = pd.concat(grids, ignore_index=True) if grids else pd.DataFrame()
    grid_table.to_csv(out_dir / "detection_grid_all.csv", index=False)
    summary_table = pd.concat(summaries, ignore_index=True) if summaries else pd.DataFrame()
    summary_table.to_csv(out_dir / "summary_rows.csv", index=False)
    write_json(
        out_dir / "run_meta.json",
        {
            "rule": RULE_NAME,
            "min_delta_r2": args.min_delta_r2,
            "min_mae_improvement": args.min_mae_improvement,
            "selection_uses_attack_labels": False,
            "selection_data": "stored Pareto fronts plus benign training/holdout stability checks",
            "original_results_modified": False,
            "out_dir": str(out_dir),
            "replacement_count": int(len(audit_table)),
        },
    )
    cols = ["dataset_key", "point_name", "S", "G", "Precision", "Recall", "F1", "eTaP", "eTaR", "eTaF1", "FPA", "Scen"]
    printable = metric_aliases(summary_table)
    print(printable[[c for c in cols if c in printable.columns]].to_string(index=False))
    print(f"Wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
