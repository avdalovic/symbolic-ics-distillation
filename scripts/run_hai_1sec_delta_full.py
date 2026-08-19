#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import inspect
import json
import math
import os
import re
import resource
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ics_symbolic_distill.data.hai import (  # noqa: E402
    EXPECTED_ATTACK_COUNT,
    assert_no_cross_sequence_pairs,
    block_lengths,
    fit_cusum_params_sequences,
    infer_variable_types,
    load_attacks,
    load_hai_sequences,
    make_pair_arrays,
    rounded_sample_size,
    run_cusum_sequences,
    sequence_manifest_rows,
    shared_state_keys,
    split_flat_by_blocks,
    train_fit_holdout_indices,
    validate_expected_generated_files,
    write_json,
)
from ics_symbolic_distill.detection import compute_detection_metrics, evaluate_equation  # noqa: E402
from ics_symbolic_distill.detection.metrics import to_intervals  # noqa: E402
from ics_symbolic_distill.detection.selection_guards import state_dependence_for_delta_equation  # noqa: E402
from ics_symbolic_distill.detection.swat1s_delta_sampling import coverage_stratified_indices, reconstruct_next_from_delta  # noqa: E402


TARGET_MODE = "sensors_delta_discrete_persistence"
SAFE_FEATURE_PREFIX = "V"
GECO_HAI_S = 8.02147642080159
GECO_HAI_G = 1.4376682451456457
S_VALUES = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0, 1.2, 1.4, 1.6, 1.8, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0]
G_VALUES = [0.1, 0.5, 1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 4.5, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 12.0, 15.0, 20.0, 25.0]
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


def rel(path: Path) -> str:
    try:
        return str(Path(path).resolve().relative_to(REPO_ROOT.resolve()))
    except ValueError:
        return str(path)


def run_text(cmd: list[str]) -> str | None:
    try:
        return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
    except Exception:
        return None


def git_info() -> dict[str, Any]:
    return {
        "commit": run_text(["git", "rev-parse", "HEAD"]),
        "branch": run_text(["git", "branch", "--show-current"]),
        "status_short": run_text(["git", "status", "--short"]),
    }


def cpu_model() -> str:
    path = Path("/proc/cpuinfo")
    if path.exists():
        for line in path.read_text(errors="ignore").splitlines():
            if line.lower().startswith("model name"):
                return line.split(":", 1)[-1].strip()
    return os.uname().machine


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
    original_to_safe = {str(col): f"{SAFE_FEATURE_PREFIX}{i}" for i, col in enumerate(feature_columns)}
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


def features_in_equation(equation: str, feature_names: Sequence[str]) -> list[str]:
    return DAY1.features_in_equation(str(equation), feature_names)


def grammar_operators(args: argparse.Namespace | None = None, *, enable_division: bool | None = None) -> list[str]:
    if enable_division is None:
        enable_division = bool(getattr(args, "enable_division", False)) if args is not None else False
    return ["+", "-", "*", "/"] if enable_division else ["+", "-", "*"]


def grammar_name(args: argparse.Namespace | None = None, *, enable_division: bool | None = None) -> str:
    return "division" if "/" in grammar_operators(args, enable_division=enable_division) else "safe_mul"


def operator_params(
    *,
    niterations: int,
    timeout_minutes: float,
    max_complexity: int,
    procs: int,
    seed: int,
    binary_operators: Sequence[str] | None = None,
) -> dict[str, Any]:
    return {
        "niterations": int(niterations),
        "binary_operators": list(binary_operators or ["+", "-", "*"]),
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

    requested = operator_params(**kwargs)
    signature = inspect.signature(PySRRegressor.__init__)
    supported = {
        name
        for name, value in signature.parameters.items()
        if name != "self" and value.kind in {value.POSITIONAL_OR_KEYWORD, value.KEYWORD_ONLY}
    }
    params = {key: value for key, value in requested.items() if key in supported}
    return PySRRegressor(**params), params


def stable_target_seed(target: str, base_seed: int) -> int:
    offset = sum((i + 1) * ord(ch) for i, ch in enumerate(str(target))) % 100000
    return int(base_seed) + int(offset)


def pareto_dir(out_root: Path, target: str) -> Path:
    return out_root / "pareto_fronts" / f"{target}_{TARGET_MODE}"


def payload_dir(out_root: Path, target: str) -> Path:
    return out_root / "target_payloads" / target


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


def candidate_indices_by_score(df: pd.DataFrame) -> list[int]:
    table = df.copy()
    table["complexity_num"] = pd.to_numeric(table.get("complexity"), errors="coerce").fillna(np.inf)
    table["loss_num"] = pd.to_numeric(table.get("loss"), errors="coerce").fillna(np.inf)
    table["score_num"] = pd.to_numeric(table.get("score"), errors="coerce").fillna(-np.inf)
    return table.sort_values(["score_num", "loss_num", "complexity_num"], ascending=[False, True, True]).index.astype(int).tolist()


def evaluate_pareto_df(
    model: Any,
    equations: pd.DataFrame,
    x_sample: pd.DataFrame,
    y_sample: np.ndarray,
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
        try:
            pred_sample = np.asarray(model.predict(x_sample, index=int(equation_index)), dtype=np.float64)
            sample = regression_metrics(y_sample, pred_sample)
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
                    "equation_features": "[]",
                    "evaluation_error": str(exc),
                }
            )
        rows.append(out)
    return pd.DataFrame(rows)


def parse_geco_point(model_path: str | None) -> tuple[float, float, str]:
    if not model_path:
        return GECO_HAI_S, GECO_HAI_G, "default_expected_hai_model_point"
    text = Path(model_path).read_text(encoding="utf-8", errors="replace")
    try:
        payload = json.loads(text)
        settings = payload.get("settings", {}) if isinstance(payload, dict) else {}
    except Exception:
        settings = {}
    s_json = safe_float(settings.get("threshold_factor"))
    g_json = safe_float(settings.get("cusum_factor"))
    if np.isfinite(s_json) and np.isfinite(g_json):
        return s_json, g_json, "geco_model"
    def first(patterns: list[str]) -> float | None:
        for pattern in patterns:
            match = re.search(pattern, text, flags=re.IGNORECASE | re.MULTILINE)
            if not match:
                continue
            value = safe_float(match.group(1))
            if np.isfinite(value):
                return value
        return None
    s = first([r"(?:threshold_factor|S_scale|scale|threshold_scale|s)\s*['\"]?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", r"['\"]S['\"]\s*:\s*([0-9]+(?:\.[0-9]+)?)"])
    g = first([r"(?:cusum_factor|G_cap|growth_factor|growth|cap|g)\s*['\"]?\s*[:=]\s*([0-9]+(?:\.[0-9]+)?)", r"['\"]G['\"]\s*:\s*([0-9]+(?:\.[0-9]+)?)"])
    return s or GECO_HAI_S, g or GECO_HAI_G, "geco_model" if s is not None and g is not None else "geco_model_missing_values_used_expected"


def make_detection_points(geco_s: float, geco_g: float, *, smoke: bool = False) -> list[tuple[float, float, str]]:
    points: list[tuple[float, float, str]] = []
    if smoke:
        base = [(1.0, 1.0), (geco_s, geco_g)]
    else:
        base = [(s, g) for s in S_VALUES for g in G_VALUES] + [(geco_s, geco_g)]
    seen = set()
    for s, g in base:
        key = (round(float(s), 12), round(float(g), 12))
        if key in seen:
            continue
        seen.add(key)
        label = "geco_operating_point" if abs(float(s) - float(geco_s)) < 1e-12 and abs(float(g) - float(geco_g)) < 1e-12 else "grid"
        points.append((float(s), float(g), label))
    return points


def load_hai_arrays(args: argparse.Namespace) -> dict[str, Any]:
    data_dir = Path(args.data_dir)
    validate_expected_generated_files(data_dir)
    train_sequences = load_hai_sequences(data_dir, split="train")
    test_sequences = load_hai_sequences(data_dir, split="test")
    all_sequences = train_sequences + test_sequences
    feature_columns = shared_state_keys(all_sequences)
    train_pairs = make_pair_arrays(train_sequences, feature_columns)
    test_pairs = make_pair_arrays(test_sequences, feature_columns)
    assert_no_cross_sequence_pairs(train_sequences, train_pairs)
    assert_no_cross_sequence_pairs(test_sequences, test_pairs)
    fit_idx, holdout_idx = train_fit_holdout_indices(train_pairs, fit_fraction=0.8)
    train_current = train_pairs.flatten_current().astype(np.float32, copy=False)
    train_next = train_pairs.flatten_next().astype(np.float32, copy=False)
    test_current = test_pairs.flatten_current().astype(np.float32, copy=False)
    test_next = test_pairs.flatten_next().astype(np.float32, copy=False)
    original_to_safe, safe_to_original, safe_feature_columns = make_variable_name_mapping(feature_columns)
    typing = infer_variable_types(train_sequences, feature_columns)
    symbolic_targets = typing.loc[typing["selected_model"] == "symbolic", "variable"].astype(str).tolist()
    persistence_targets = typing.loc[typing["selected_model"] == "persistence", "variable"].astype(str).tolist()
    sample_size, sample_derivation = rounded_sample_size(int(train_pairs.n_pairs), fraction=0.025, base=1000)
    if args.sample_size is not None:
        sample_size = int(args.sample_size)
        sample_derivation["sample_size_override"] = sample_size
    return {
        "data_dir": data_dir,
        "train_sequences": train_sequences,
        "test_sequences": test_sequences,
        "all_sequences": all_sequences,
        "train_pairs": train_pairs,
        "test_pairs": test_pairs,
        "train_current": train_current,
        "train_next": train_next,
        "test_current": test_current,
        "test_next": test_next,
        "labels": test_pairs.flatten_labels().astype(np.int64),
        "attack_ids": test_pairs.flatten_attack_ids().astype(np.int64),
        "test_timestamps": test_pairs.flatten_timestamps().astype(np.int64),
        "feature_columns": feature_columns,
        "original_to_safe": original_to_safe,
        "safe_to_original": safe_to_original,
        "safe_feature_columns": safe_feature_columns,
        "variable_typing": typing,
        "symbolic_targets": symbolic_targets,
        "persistence_targets": persistence_targets,
        "fit_idx": fit_idx,
        "holdout_idx": holdout_idx,
        "train_block_lengths": block_lengths(train_pairs),
        "test_block_lengths": block_lengths(test_pairs),
        "sample_size": sample_size,
        "sample_derivation": sample_derivation,
        "attacks": load_attacks(data_dir / "attacks.json"),
        "metadata": {
            "train_sequences": len(train_sequences),
            "test_sequences": len(test_sequences),
            "train_pairs": int(train_pairs.n_pairs),
            "test_pairs": int(test_pairs.n_pairs),
            "num_features": len(feature_columns),
            "num_symbolic_targets": len(symbolic_targets),
            "num_persistence_targets": len(persistence_targets),
            "sample_derivation": sample_derivation,
        },
    }


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
    pred_delta = evaluate_equation(equation_safe, arrays["safe_feature_columns"], current)
    pred_next = reconstruct_next_from_delta(current[:, idx], pred_delta)
    residual = np.abs(nxt[:, idx].astype(np.float64) - pred_next.astype(np.float64))
    return np.where(np.isfinite(residual), residual, 0.0).astype(np.float64)


def persistence_residual(arrays: dict[str, Any], target: str, *, split: str) -> np.ndarray:
    idx = arrays["feature_columns"].index(target)
    if split == "train":
        current = arrays["train_current"]
        nxt = arrays["train_next"]
    else:
        current = arrays["test_current"]
        nxt = arrays["test_next"]
    residual = np.abs(nxt[:, idx].astype(np.float64) - current[:, idx].astype(np.float64))
    return np.where(np.isfinite(residual), residual, 0.0).astype(np.float64)


def sample_indices_for_target(arrays: dict[str, Any], target: str, sample_size: int, seed: int) -> tuple[np.ndarray, dict[str, Any]]:
    fit_idx = arrays["fit_idx"]
    current, y_all = target_values(arrays, target, split="train")
    y_fit = y_all[fit_idx]
    actuator_indices = [arrays["feature_columns"].index(name) for name in arrays["persistence_targets"]]
    local_idx, audit = coverage_stratified_indices(
        target=target,
        y_delta_fit_pool=y_fit,
        x_current_fit_pool=current[fit_idx],
        x_next_fit_pool=arrays["train_next"][fit_idx],
        actuator_indices=actuator_indices,
        sample_size=int(sample_size),
        seed=int(seed),
    )
    return fit_idx[local_idx], audit.to_dict()


def prepare_target_payload(arrays: dict[str, Any], out_root: Path, target: str, args: argparse.Namespace) -> None:
    payload_root = payload_dir(out_root, target)
    payload_path = payload_root / "sample.npz"
    metadata_path = payload_root / "metadata.json"
    if payload_path.exists() and metadata_path.exists() and args.resume and not args.force:
        return
    source_dir = getattr(args, "sample_index_source_dir", None)
    if source_dir:
        source_path = pareto_dir(Path(source_dir), target) / "sample_indices.npy"
        if not source_path.exists():
            raise FileNotFoundError(f"missing source sample indices for {target}: {source_path}")
        sample_idx = np.load(source_path).astype(np.int64)
        audit = {
            "target": target,
            "sample_policy": "copied_from_sample_index_source_dir",
            "source_sample_indices": rel(source_path),
            "source_sample_indices_sha256": file_sha256(source_path),
            "sample_size_requested": int(arrays["sample_size"]),
            "sample_size_actual": int(sample_idx.shape[0]),
        }
    else:
        sample_idx, audit = sample_indices_for_target(arrays, target, int(arrays["sample_size"]), int(args.seed))
    current, y_all = target_values(arrays, target, split="train")
    payload_root.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        payload_path,
        x_sample=current[sample_idx].astype(np.float32),
        y_sample=y_all[sample_idx].astype(np.float64),
    )
    write_json(
        metadata_path,
        {
            "target": target,
            "safe_feature_columns": arrays["safe_feature_columns"],
            "safe_to_original": arrays["safe_to_original"],
            "sample_size": int(sample_idx.shape[0]),
            "sample_audit": audit,
        },
    )
    pdir = pareto_dir(out_root, target)
    pdir.mkdir(parents=True, exist_ok=True)
    np.save(pdir / "sample_indices.npy", sample_idx.astype(np.int64))
    pd.DataFrame([audit]).to_csv(pdir / "sample_audit.csv", index=False)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_sample_index_manifest(out_root: Path, targets: Sequence[str], *, seed: int, sample_size: int) -> Path:
    rows: list[dict[str, Any]] = []
    for target in targets:
        pdir = pareto_dir(out_root, str(target))
        sample_path = pdir / "sample_indices.npy"
        audit_path = pdir / "sample_audit.csv"
        row: dict[str, Any] = {
            "target": str(target),
            "seed": int(seed),
            "requested_sample_size": int(sample_size),
            "sample_indices_path": rel(sample_path),
            "sample_audit_path": rel(audit_path),
            "status": "missing",
            "sample_size": None,
            "min_index": None,
            "max_index": None,
            "first_index": None,
            "last_index": None,
            "sample_indices_sha256": None,
        }
        if sample_path.exists():
            sample_idx = np.load(sample_path)
            row.update(
                {
                    "status": "present",
                    "sample_size": int(sample_idx.shape[0]),
                    "min_index": int(np.min(sample_idx)) if sample_idx.size else None,
                    "max_index": int(np.max(sample_idx)) if sample_idx.size else None,
                    "first_index": int(sample_idx[0]) if sample_idx.size else None,
                    "last_index": int(sample_idx[-1]) if sample_idx.size else None,
                    "sample_indices_sha256": file_sha256(sample_path),
                }
            )
        rows.append(row)
    path = out_root / "sample_index_manifest.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def run_pysr_target(args_dict: dict[str, Any], target: str) -> dict[str, Any]:
    args = argparse.Namespace(**args_dict)
    out_root = Path(args.output_dir)
    out_dir = pareto_dir(out_root, target)
    csv_path = out_dir / "pareto_front_scored.csv"
    if csv_path.exists() and args.resume and not args.force:
        return {"target": target, "status": "skipped_existing", "pareto_csv": str(csv_path), "error": ""}
    started = time.time()
    status = "completed"
    error = ""
    params: dict[str, Any] = {}
    try:
        payload_root = payload_dir(out_root, target)
        payload = np.load(payload_root / "sample.npz")
        metadata = json.loads((payload_root / "metadata.json").read_text(encoding="utf-8"))
        safe_feature_columns = [str(x) for x in metadata["safe_feature_columns"]]
        safe_to_original = {str(k): str(v) for k, v in metadata["safe_to_original"].items()}
        x_sample = pd.DataFrame(payload["x_sample"], columns=safe_feature_columns)
        y_sample = np.asarray(payload["y_sample"], dtype=np.float64)
        out_dir.mkdir(parents=True, exist_ok=True)
        model, params = make_model(
            niterations=int(args.niterations),
            timeout_minutes=float(args.timeout_minutes),
            max_complexity=int(args.max_complexity),
            procs=int(args.pysr_procs),
            seed=stable_target_seed(target, int(args.seed)),
            binary_operators=grammar_operators(args),
        )
        model.fit(x_sample, y_sample)
        pareto = evaluate_pareto_df(
            model,
            model.equations_.copy(),
            x_sample,
            y_sample,
            safe_feature_columns,
            safe_to_original,
        )
        pareto.to_csv(csv_path, index=False)
    except Exception as exc:
        status = "failed"
        error = str(exc)
    write_json(
        out_dir / "metadata.json",
        {
            "target": target,
            "dataset": "HAI-21.03",
            "target_mode": TARGET_MODE,
            "sample_policy": "coverage_stratified_sequence_safe",
            "operator_set": grammar_name(args),
            "binary_operators": grammar_operators(args),
            "pysr_params": params,
            "max_complexity": int(args.max_complexity),
            "timeout_minutes": float(args.timeout_minutes),
            "status": status,
            "error": error,
            "elapsed_seconds": time.time() - started,
        },
    )
    return {"target": target, "status": status, "pareto_csv": str(csv_path), "error": error}


def build_single_target_command(args: argparse.Namespace, target: str) -> list[str]:
    cmd = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--data-dir",
        str(args.data_dir),
        "--output-dir",
        str(args.output_dir),
        "--timeout-minutes",
        str(args.timeout_minutes),
        "--target-wall-timeout-minutes",
        str(args.target_wall_timeout_minutes),
        "--max-complexity",
        str(args.max_complexity),
        "--niterations",
        str(args.niterations),
        "--max-workers",
        "1",
        "--pysr-procs",
        str(args.pysr_procs),
        "--seed",
        str(args.seed),
        "--single-target",
        target,
    ]
    if getattr(args, "enable_division", False):
        cmd.append("--enable-division")
    if args.resume:
        cmd.append("--resume")
    if args.force:
        cmd.append("--force")
    if args.sample_size is not None:
        cmd.extend(["--sample-size", str(args.sample_size)])
    if getattr(args, "sample_index_source_dir", None):
        cmd.extend(["--sample-index-source-dir", str(args.sample_index_source_dir)])
    return cmd


def run_pysr_targets_isolated(args: argparse.Namespace, targets: list[str], out_root: Path, arrays: dict[str, Any]) -> list[dict[str, Any]]:
    workers = max(1, int(args.max_workers))
    status_by_target: dict[str, dict[str, Any]] = {}
    pending: list[str] = []
    for target in targets:
        csv_path = pareto_dir(out_root, target) / "pareto_front_scored.csv"
        if csv_path.exists() and args.resume and not args.force:
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
    log_dir = out_root / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    active: dict[int, tuple[subprocess.Popen[Any], str, float, Any]] = {}
    while pending or active:
        while pending and len(active) < workers:
            target = pending.pop(0)
            prepare_target_payload(arrays, out_root, target, args)
            cmd = build_single_target_command(args, target)
            log_handle = (log_dir / f"{target}.log").open("a", encoding="utf-8")
            log_handle.write(f"\n=== launch {time.strftime('%Y-%m-%dT%H:%M:%S')} ===\n")
            log_handle.write(" ".join(cmd) + "\n")
            log_handle.flush()
            proc = subprocess.Popen(cmd, cwd=str(REPO_ROOT), stdout=log_handle, stderr=subprocess.STDOUT)
            active[int(proc.pid)] = (proc, target, time.time(), log_handle)
            status_by_target[target] = {
                **status_by_target[target],
                "status": "running",
                "pid": int(proc.pid),
                "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
            }
            print(f"[HAI] launched target={target} pid={proc.pid}", flush=True)
            write_pysr_status(out_root, targets, status_by_target)
        time.sleep(5.0)
        for pid, (proc, target, started, log_handle) in list(active.items()):
            elapsed = time.time() - started
            returncode = proc.poll()
            if returncode is None and wall_limit_seconds > 0 and elapsed > wall_limit_seconds:
                print(f"[HAI] timeout target={target} pid={pid}; terminating isolated child", flush=True)
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
                        "dataset": "HAI-21.03",
                        "target_mode": TARGET_MODE,
                        "sample_policy": "coverage_stratified_sequence_safe",
                        "operator_set": grammar_name(args),
                        "binary_operators": grammar_operators(args),
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
            print(f"[HAI] done target={target} status={result.get('status')} exit_code={returncode}", flush=True)
            write_pysr_status(out_root, targets, status_by_target)
            log_handle.close()
            del active[pid]
    return [status_by_target[target] for target in targets]


def write_pysr_status(out_root: Path, targets: list[str], status_by_target: dict[str, dict[str, Any]]) -> None:
    rows = [status_by_target[target] for target in targets]
    pd.DataFrame(rows).to_csv(out_root / "pysr_run_status.csv", index=False)


def train_residual_blocks(arrays: dict[str, Any], residual_train: np.ndarray, *, fit_only: bool | None = None) -> list[np.ndarray]:
    blocks = split_flat_by_blocks(residual_train, arrays["train_block_lengths"])
    if fit_only is None:
        return blocks
    out = []
    for block in blocks:
        cutoff = int(math.floor(block.shape[0] * 0.8))
        out.append(block[:cutoff] if fit_only else block[cutoff:])
    return out


def select_symbolic_equation(
    arrays: dict[str, Any],
    out_root: Path,
    target: str,
    *,
    max_complexity: int,
    geco_s: float,
    geco_g: float,
    state_dependence_guard: bool = True,
) -> tuple[dict[str, Any] | None, dict[str, Any] | None, dict[str, np.ndarray] | None]:
    csv_path = pareto_dir(out_root, target) / "pareto_front_scored.csv"
    if not csv_path.exists():
        return None, {"target": target, "reason": "missing_pareto_front"}, None
    df = pd.read_csv(csv_path)
    if df.empty:
        return None, {"target": target, "reason": "empty_pareto_front"}, None
    holdout_idx = arrays["holdout_idx"]
    _, y_all = target_values(arrays, target, split="train")
    baseline_holdout_mae = float(np.mean(np.abs(y_all[holdout_idx]))) if holdout_idx.size else 0.0
    reasons = []
    passing = []
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
        if state_dependence_guard:
            guard = state_dependence_for_delta_equation(
                target=target,
                equation=equation,
                feature_names=arrays["feature_columns"],
            )
            if not guard.state_dependent:
                reasons.append(f"row={idx}:state_dependence_guard:{guard.reason}:{guard.simplified_next}")
                continue
        residual_train = prediction_residual(arrays, target, equation, split="train")
        residual_fit_blocks = train_residual_blocks(arrays, residual_train, fit_only=True)
        residual_holdout_blocks = train_residual_blocks(arrays, residual_train, fit_only=False)
        params = fit_cusum_params_sequences(residual_fit_blocks, s=float(geco_s), g=float(geco_g))
        _, holdout_alarm, _, _ = run_cusum_sequences(residual_holdout_blocks, params)
        if int(np.sum(holdout_alarm)) > 0:
            reasons.append(f"row={idx}:holdout_cusum_alarm")
            continue
        median = float(np.median(residual_train))
        p99 = float(np.percentile(residual_train, 99))
        tail_ratio = p99 / max(median, 1e-9)
        if tail_ratio > 50.0:
            reasons.append(f"row={idx}:tail_ratio>50:{tail_ratio:.4g}")
            continue
        holdout_metrics = regression_metrics(y_all[holdout_idx], pred_train_target[holdout_idx]) if holdout_idx.size else {"r2": 0.0, "mae": 0.0}
        residual_test = prediction_residual(arrays, target, equation, split="test")
        selected = {
            "target": target,
            "variable_type": "sensor",
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


def persistence_selected_row(arrays: dict[str, Any], target: str) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    residual_train = persistence_residual(arrays, target, split="train")
    residual_test = persistence_residual(arrays, target, split="test")
    median = float(np.median(residual_train))
    p99 = float(np.percentile(residual_train, 99))
    return (
        {
            "target": target,
            "variable_type": "persistence",
            "target_mode": TARGET_MODE,
            "equation": target,
            "sympy_format": target,
            "complexity": 1.0,
            "loss": float(np.mean(residual_train**2)),
            "score": 0.0,
            "holdout_r2": 0.0,
            "holdout_mae": float(np.mean(residual_train[arrays["holdout_idx"]])) if arrays["holdout_idx"].size else 0.0,
            "baseline_holdout_mae": float(np.mean(residual_train[arrays["holdout_idx"]])) if arrays["holdout_idx"].size else 0.0,
            "residual_tail_ratio": p99 / max(median, 1e-9),
            "residual_p99": p99,
            "residual_median": median,
            "selection_reason": "persistence_for_discrete_or_piecewise_constant_variable",
            "pareto_csv": "",
        },
        {"train": residual_train, "test": residual_test},
    )


def alarm_burden(alarms: np.ndarray, labels: np.ndarray) -> dict[str, float]:
    alarm = np.asarray(alarms, dtype=np.int64)
    y = np.asarray(labels, dtype=np.int64) >= 1
    intervals = to_intervals(alarm)
    return {
        "total_alarm_rate": float(np.mean(alarm)) if alarm.size else 0.0,
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


def evaluate_detection(
    arrays: dict[str, Any],
    selected_rows: list[dict[str, Any]],
    residual_cache: dict[str, dict[str, np.ndarray]],
    *,
    points: list[tuple[float, float, str]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[tuple[float, float], dict[str, np.ndarray]]]:
    grid_rows = []
    per_sensor_rows = []
    alarm_cache: dict[tuple[float, float], dict[str, np.ndarray]] = {}
    labels = arrays["labels"]
    for s, g, point_kind in points:
        alarms = []
        per_sensor_alarms: dict[str, np.ndarray] = {}
        for row in selected_rows:
            target = str(row["target"])
            residuals = residual_cache[target]
            params = fit_cusum_params_sequences(train_residual_blocks(arrays, residuals["train"], fit_only=None), s=float(s), g=float(g))
            _, alarm, _, _ = run_cusum_sequences(split_flat_by_blocks(residuals["test"], arrays["test_block_lengths"]), params)
            alarm = alarm.astype(np.int8)
            per_sensor_alarms[target] = alarm
            alarms.append(alarm)
            per_sensor_rows.append(
                {
                    "target": target,
                    "variable_type": row.get("variable_type", ""),
                    "S": s,
                    "G": g,
                    "point_kind": point_kind,
                    "complexity": row.get("complexity"),
                    "equation": row.get("equation"),
                    "holdout_r2": row.get("holdout_r2"),
                    "residual_tail_ratio": row.get("residual_tail_ratio"),
                    "delta": params.delta,
                    "threshold": params.threshold,
                    "growth_cap": params.growth_cap,
                    "max_train_cusum": params.max_calib_cusum,
                    **alarm_burden(alarm, labels),
                }
            )
        system_alarm = np.max(np.stack(alarms, axis=1), axis=1).astype(np.int8) if alarms else np.zeros_like(labels, dtype=np.int8)
        alarm_cache[(float(s), float(g))] = {"system": system_alarm, **per_sensor_alarms}
        metrics = compute_detection_metrics(labels, system_alarm, expand_steps=60)
        grid_rows.append(
            {
                "dataset": "HAI",
                "variant": "seed0_baseline",
                "S": s,
                "G": g,
                "point_kind": point_kind,
                "num_monitored": len(selected_rows),
                "monitored_symbolic": sum(str(r.get("variable_type")) == "sensor" for r in selected_rows),
                "monitored_persistence": sum(str(r.get("variable_type")) == "persistence" for r in selected_rows),
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


def write_per_attack(arrays: dict[str, Any], system_alarm: np.ndarray, out_path: Path) -> pd.DataFrame:
    attack_ids = arrays["attack_ids"]
    timestamps = arrays["test_timestamps"]
    attack_points = {int(item["id"]): json.dumps(item.get("attack_point", [])) for item in arrays["attacks"]}
    rows = []
    for attack_id in sorted(int(x) for x in np.unique(attack_ids) if int(x) > 0):
        mask = attack_ids == attack_id
        idx = np.flatnonzero(mask)
        alarm_inside = system_alarm[mask]
        rows.append(
            {
                "attack_id": attack_id,
                "attack_point": attack_points.get(attack_id, "[]"),
                "start_index": int(idx[0]),
                "end_index": int(idx[-1]),
                "start_timestamp": int(timestamps[idx[0]]),
                "end_timestamp": int(timestamps[idx[-1]]),
                "duration_steps": int(idx.size),
                "detected": bool(np.any(alarm_inside)),
                "alarm_steps": int(np.sum(alarm_inside)),
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(out_path, index=False)
    return table


def write_per_file_metrics(arrays: dict[str, Any], system_alarm: np.ndarray, out_path: Path) -> pd.DataFrame:
    label_blocks = split_flat_by_blocks(arrays["labels"], arrays["test_block_lengths"])
    alarm_blocks = split_flat_by_blocks(system_alarm, arrays["test_block_lengths"])
    rows = []
    for seq, labels, alarms in zip(arrays["test_sequences"], label_blocks, alarm_blocks, strict=True):
        metrics = compute_detection_metrics(labels, alarms, expand_steps=60)
        rows.append(
            {
                "sequence": seq.name,
                "rows": int(len(labels)),
                "attack_rows": int(np.sum(labels)),
                "Precision": metrics["point_precision"],
                "Recall": metrics["point_recall"],
                "F1": metrics["point_f1"],
                "eTaP": metrics["eTaP"],
                "eTaR": metrics["eTaR"],
                "eTaF1": metrics["eTaF1"],
                "FPA": metrics["FPA"],
                "Scen": metrics["scenario_detection_rate"],
            }
        )
    table = pd.DataFrame(rows)
    table.to_csv(out_path, index=False)
    return table


def write_residual_stats(selected_rows: list[dict[str, Any]], residual_cache: dict[str, dict[str, np.ndarray]], out_path: Path) -> pd.DataFrame:
    rows = []
    for row in selected_rows:
        target = str(row["target"])
        for split in ["train", "test"]:
            residual = residual_cache[target][split]
            rows.append(
                {
                    "target": target,
                    "split": split,
                    "variable_type": row.get("variable_type", ""),
                    "count": int(residual.shape[0]),
                    "mean": float(np.mean(residual)),
                    "std": float(np.std(residual)),
                    "median": float(np.median(residual)),
                    "p95": float(np.percentile(residual, 95)),
                    "p99": float(np.percentile(residual, 99)),
                    "max": float(np.max(residual)),
                    "nonfinite_count": int(np.sum(~np.isfinite(residual))),
                }
            )
    table = pd.DataFrame(rows)
    table.to_csv(out_path, index=False)
    return table


def select_smoke_targets(arrays: dict[str, Any], requested: list[str]) -> list[str]:
    available = set(arrays["feature_columns"])
    chosen = [target for target in requested if target in available]
    if len(chosen) >= len(requested) and chosen:
        return chosen
    typing = arrays["variable_typing"]
    symbolic = typing[typing["selected_model"] == "symbolic"]["variable"].astype(str).tolist()
    persistence = typing[typing["selected_model"] == "persistence"]["variable"].astype(str).tolist()
    buckets = [
        [v for v in symbolic if "LIT" in v or "_LT" in v or "LEVEL" in v.upper()],
        [v for v in symbolic if "TIT" in v or "_TT" in v or "TEMP" in v.upper()],
        [v for v in symbolic if "FIT" in v or "_FT" in v or "FLOW" in v.upper()],
        persistence,
        [v for v in symbolic if v.startswith("P2_") or v.startswith("P3_") or v.startswith("P4_")],
    ]
    for bucket in buckets:
        for target in bucket:
            if target not in chosen:
                chosen.append(target)
                break
    return chosen[:5]


def promote_compact_results(out_root: Path, results_dir: Path) -> None:
    results_dir.mkdir(parents=True, exist_ok=True)
    names = [
        "selected_equations.csv",
        "detection_grid.csv",
        "per_sensor_residual_stats.csv",
        "per_attack.csv",
        "per_file_metrics.csv",
        "variable_typing.csv",
        "sequence_manifest.csv",
        "geco_operating_point.csv",
        "run_metadata.json",
    ]
    for name in names:
        src = out_root / name
        if src.exists():
            dst = results_dir / name
            dst.write_bytes(src.read_bytes())


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="HAI 21.03 one-second delta symbolic PySR and CUSUM detection.")
    parser.add_argument("--data-dir", default="data/hai/ipal")
    parser.add_argument("--output-dir", default="artifacts/experiments/hai_baseline_seed0")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--sample-size", type=int, default=None)
    parser.add_argument("--timeout-minutes", type=float, default=60.0)
    parser.add_argument("--target-wall-timeout-minutes", type=float, default=75.0)
    parser.add_argument("--max-complexity", type=int, default=15)
    parser.add_argument("--niterations", type=int, default=400)
    parser.add_argument("--max-workers", type=int, default=2)
    parser.add_argument("--pysr-procs", type=int, default=1)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--force", action="store_true")
    parser.add_argument("--skip-pysr", action="store_true")
    parser.add_argument("--skip-detection-eval", action="store_true", help="Stop after equation selection and residual export.")
    parser.add_argument("--single-target", default=None)
    parser.add_argument("--smoke-targets", nargs="*", default=None)
    parser.add_argument("--geco-model", default=None)
    parser.add_argument("--promote-results", action="store_true")
    parser.add_argument("--enable-division", action="store_true", help="Use PySR grammar {+, -, *, /}. Default is {+, -, *}.")
    parser.add_argument("--sample-index-source-dir", default=None, help="Copy per-target sample_indices.npy from another HAI run.")
    parser.add_argument(
        "--disable-state-dependence-guard",
        dest="state_dependence_guard",
        action="store_false",
        default=True,
        help="Allow delta equations whose reconstructed next-state prediction simplifies to a constant.",
    )
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    if args.smoke_targets is not None:
        if args.output_dir == "artifacts/experiments/hai_baseline_seed0":
            args.output_dir = "artifacts/experiments/hai_smoke_seed0"
        args.timeout_minutes = min(float(args.timeout_minutes), 2.0)
        args.target_wall_timeout_minutes = min(float(args.target_wall_timeout_minutes), 5.0)
        args.niterations = min(int(args.niterations), 20)
        args.max_complexity = min(int(args.max_complexity), 7)
        args.max_workers = min(int(args.max_workers), 2)
        args.sample_size = min(int(args.sample_size or 1000), 1000)

    out_root = Path(args.output_dir)
    out_root.mkdir(parents=True, exist_ok=True)

    if args.single_target:
        result = run_pysr_target(vars(args), str(args.single_target))
        print(f"[HAI child] target={result.get('target')} status={result.get('status')} error={result.get('error', '')}", flush=True)
        return 0 if result.get("status") in {"completed", "skipped_existing"} else 2

    started = time.time()
    arrays = load_hai_arrays(args)
    geco_s, geco_g, geco_source = parse_geco_point(args.geco_model)
    smoke = args.smoke_targets is not None
    targets = list(arrays["symbolic_targets"])
    if smoke:
        targets = select_smoke_targets(arrays, [str(x) for x in args.smoke_targets or []])
    print("=== HAI 21.03 data ===", flush=True)
    print(arrays["metadata"], flush=True)
    print(f"Sample derivation: {arrays['sample_derivation']}", flush=True)
    print(f"PySR symbolic targets: {len(targets)}", flush=True)
    print(f"Persistence targets: {len(arrays['persistence_targets'])}", flush=True)
    print(f"GeCo HAI point: S={geco_s} G={geco_g} source={geco_source}", flush=True)

    sequence_rows = sequence_manifest_rows(arrays["all_sequences"])
    for row in sequence_rows:
        row["path"] = rel(Path(row["path"]))
    pd.DataFrame(sequence_rows).to_csv(out_root / "sequence_manifest.csv", index=False)
    variable_typing = arrays["variable_typing"].copy()
    variable_typing.to_csv(out_root / "variable_typing_initial.csv", index=False)
    pd.DataFrame(
        [
            {"original_name": original, "safe_name": safe}
            for original, safe in arrays["original_to_safe"].items()
        ]
    ).to_csv(out_root / "variable_name_mapping.csv", index=False)

    run_metadata = {
        "dataset": "HAI-21.03",
        "git": git_info(),
        "seed": int(args.seed),
        "sample_size": int(arrays["sample_size"]),
        "sample_derivation": arrays["sample_derivation"],
        "target_count": len(targets),
        "persistence_target_count": len(arrays["persistence_targets"]),
        "grammar": {"name": grammar_name(args), "binary_operators": grammar_operators(args), "unary_operators": []},
        "pysr_config": operator_params(
            niterations=int(args.niterations),
            timeout_minutes=float(args.timeout_minutes),
            max_complexity=int(args.max_complexity),
            procs=int(args.pysr_procs),
            seed=int(args.seed),
            binary_operators=grammar_operators(args),
        ),
        "sample_index_source_dir": str(args.sample_index_source_dir) if args.sample_index_source_dir else None,
        "state_dependence_guard": bool(args.state_dependence_guard),
        "max_workers": int(args.max_workers),
        "cpu_model": cpu_model(),
        "geco_operating_point": {"S": geco_s, "G": geco_g, "source": geco_source},
        "data": arrays["metadata"],
        "smoke": bool(smoke),
        "started_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    source_manifest = REPO_ROOT / "data" / "hai" / "SOURCE_MANIFEST.json"
    if source_manifest.exists():
        run_metadata["source_manifest"] = json.loads(source_manifest.read_text(encoding="utf-8"))
    write_json(out_root / "run_metadata.json", run_metadata)

    if not args.skip_pysr:
        run_pysr_targets_isolated(args, targets, out_root, arrays)
    sample_manifest = write_sample_index_manifest(
        out_root,
        targets,
        seed=int(args.seed),
        sample_size=int(arrays["sample_size"]),
    )
    run_metadata["sample_index_manifest"] = rel(sample_manifest)
    write_json(out_root / "run_metadata.json", run_metadata)

    selected_rows: list[dict[str, Any]] = []
    exclusion_rows: list[dict[str, Any]] = []
    residual_cache: dict[str, dict[str, np.ndarray]] = {}
    for target in targets:
        selected, exclusion, residuals = select_symbolic_equation(
            arrays,
            out_root,
            target,
            max_complexity=int(args.max_complexity),
            geco_s=float(geco_s),
            geco_g=float(geco_g),
            state_dependence_guard=bool(args.state_dependence_guard),
        )
        if selected is None:
            if exclusion is not None:
                exclusion_rows.append(exclusion)
            continue
        selected_rows.append(selected)
        residual_cache[target] = residuals or {}

    if not smoke:
        for target in arrays["persistence_targets"]:
            selected, residuals = persistence_selected_row(arrays, target)
            selected_rows.append(selected)
            residual_cache[target] = residuals

    selected_df = pd.DataFrame(selected_rows, columns=SELECTED_EQUATION_COLUMNS)
    selected_df.to_csv(out_root / "selected_equations.csv", index=False)
    pd.DataFrame(exclusion_rows).to_csv(out_root / "exclusion_reasons.csv", index=False)

    final_typing = variable_typing.copy()
    selected_set = {str(row["target"]) for row in selected_rows}
    exclusions = {str(row["target"]): str(row["reason"]) for row in exclusion_rows}
    final_models = []
    final_reasons = []
    for _, row in final_typing.iterrows():
        variable = str(row["variable"])
        if variable in selected_set:
            final_models.append(str(row["selected_model"]))
            final_reasons.append("selected_for_monitoring")
        elif variable in exclusions:
            final_models.append("excluded")
            final_reasons.append(exclusions[variable])
        else:
            final_models.append(str(row["selected_model"]))
            final_reasons.append(str(row["reason"]))
    final_typing["selected_model"] = final_models
    final_typing["reason"] = final_reasons
    final_typing.to_csv(out_root / "variable_typing.csv", index=False)

    if args.skip_detection_eval:
        write_residual_stats(selected_rows, residual_cache, out_root / "per_sensor_residual_stats.csv")
        run_metadata["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
        run_metadata["wall_seconds"] = time.time() - started
        run_metadata["symbolic_selected_count"] = sum(str(row.get("variable_type")) == "sensor" for row in selected_rows)
        run_metadata["persistence_selected_count"] = sum(str(row.get("variable_type")) == "persistence" for row in selected_rows)
        run_metadata["excluded_symbolic_count"] = len(exclusion_rows)
        run_metadata["peak_memory_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
        run_metadata["detection_eval_skipped"] = True
        write_json(out_root / "run_metadata.json", run_metadata)
        print("\n=== HAI 21.03 discovery complete; detection evaluation skipped ===")
        print(f"Selected symbolic: {run_metadata['symbolic_selected_count']} / {len(targets)}")
        print(f"Selected persistence: {run_metadata['persistence_selected_count']}")
        print(f"Excluded symbolic: {run_metadata['excluded_symbolic_count']}")
        return 0

    points = make_detection_points(float(geco_s), float(geco_g), smoke=smoke)
    grid, per_sensor, alarm_cache = evaluate_detection(arrays, selected_rows, residual_cache, points=points)
    grid.to_csv(out_root / "detection_grid.csv", index=False)
    per_sensor.to_csv(out_root / "per_sensor_alarm_stats.csv", index=False)
    geco_row = grid[grid["point_kind"] == "geco_operating_point"].copy()
    geco_row.to_csv(out_root / "geco_operating_point.csv", index=False)
    if not geco_row.empty:
        key = (float(geco_row.iloc[0]["S"]), float(geco_row.iloc[0]["G"]))
        system_alarm = alarm_cache[key]["system"]
    elif alarm_cache:
        system_alarm = next(iter(alarm_cache.values()))["system"]
    else:
        system_alarm = np.zeros_like(arrays["labels"], dtype=np.int8)
    write_per_attack(arrays, system_alarm, out_root / "per_attack.csv")
    write_per_file_metrics(arrays, system_alarm, out_root / "per_file_metrics.csv")
    write_residual_stats(selected_rows, residual_cache, out_root / "per_sensor_residual_stats.csv")

    run_metadata["finished_at"] = time.strftime("%Y-%m-%dT%H:%M:%S")
    run_metadata["wall_seconds"] = time.time() - started
    run_metadata["symbolic_selected_count"] = sum(str(row.get("variable_type")) == "sensor" for row in selected_rows)
    run_metadata["persistence_selected_count"] = sum(str(row.get("variable_type")) == "persistence" for row in selected_rows)
    run_metadata["excluded_symbolic_count"] = len(exclusion_rows)
    run_metadata["peak_memory_kb"] = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    write_json(out_root / "run_metadata.json", run_metadata)

    if args.promote_results and not smoke:
        promote_compact_results(out_root, REPO_ROOT / "results" / "hai")

    print("\n=== HAI 21.03 seed baseline ===")
    print(f"Selected symbolic: {run_metadata['symbolic_selected_count']} / {len(targets)}")
    print(f"Selected persistence: {run_metadata['persistence_selected_count']}")
    if not geco_row.empty:
        print("GeCo operating point:")
        print(geco_row[["S", "G", "num_monitored", "Precision", "Recall", "F1", "eTaP", "eTaR", "eTaF1", "FPA", "Scen"]].to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
