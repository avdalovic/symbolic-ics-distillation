#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
import math
import multiprocessing as mp
import os
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from omegaconf import OmegaConf

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


_CTX: dict[str, Any] = {}


def is_actuator(dataset: str, label: str) -> bool:
    if str(dataset).upper() == "SWAT":
        return "IT" not in str(label)
    return False


def normalize_attack_labels(labels: pd.Series | Sequence[object] | np.ndarray | None) -> np.ndarray | None:
    if labels is None:
        return None
    series = pd.Series(labels).copy()
    if series.dtype == object:
        lowered = series.astype(str).str.strip().str.lower()
        mapping = {"normal": 0.0, "attack": 1.0, "n": 0.0, "a": 1.0}
        series = lowered.map(mapping).fillna(lowered)
    numeric = pd.to_numeric(series, errors="coerce").fillna(0.0)
    return numeric.to_numpy(dtype=np.float32)


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
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(k): jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [jsonable(v) for v in value]
    try:
        if pd.isna(value):
            return None
    except Exception:
        pass
    try:
        import sympy as sp

        if isinstance(value, sp.Basic):
            return str(value)
    except Exception:
        pass
    return str(value)


def write_json(path: str | Path, payload: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(jsonable(payload), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return out


def git_info() -> dict[str, Any]:
    def run(cmd: list[str]) -> str | None:
        try:
            return subprocess.check_output(cmd, cwd=REPO_ROOT, text=True, stderr=subprocess.DEVNULL).strip()
        except Exception:
            return None

    return {
        "commit": run(["git", "rev-parse", "HEAD"]),
        "branch": run(["git", "branch", "--show-current"]),
        "status_short": run(["git", "status", "--short"]),
    }


def _config_root_from_experiment(experiment_path: Path) -> Path:
    if experiment_path.parent.name != "experiment":
        return experiment_path.parent.parent
    return experiment_path.parent.parent


def _merge_named(root: Path, section: str, name: str | None, cfg: Any) -> Any:
    if not name:
        return cfg
    path = root / section / f"{name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Missing {section} config: {path}")
    return OmegaConf.merge(cfg, OmegaConf.load(path))


def load_experiment_config_local(path: str | Path) -> tuple[Any, Path]:
    exp_path = Path(path).expanduser().resolve()
    if not exp_path.exists():
        raise FileNotFoundError(f"Experiment config not found: {exp_path}")
    payload = OmegaConf.load(exp_path)
    exp = payload.get("experiment", payload)
    root = _config_root_from_experiment(exp_path)
    cfg = OmegaConf.create()
    cfg = _merge_named(root, "dataset", exp.get("dataset_cfg"), cfg)
    cfg = _merge_named(root, "model", exp.get("model_cfg"), cfg)
    cfg = _merge_named(root, "train", exp.get("train_cfg"), cfg)
    cfg = _merge_named(root, "export", exp.get("export_cfg"), cfg)
    cfg = _merge_named(root, "evaluation", exp.get("evaluation_cfg"), cfg)
    overrides = exp.get("overrides") or []
    if overrides:
        cfg = OmegaConf.merge(cfg, OmegaConf.from_dotlist([str(x) for x in overrides]))
    cfg = OmegaConf.merge(cfg, {"experiment": OmegaConf.to_container(exp, resolve=True)})
    return cfg, exp_path


def operator_config(operator_set: str) -> dict[str, Any]:
    token = str(operator_set).strip().lower()
    if token == "restricted":
        return {
            "operator_set": "restricted",
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": [],
            "extra_sympy_mappings": {},
            "default_maxsize": 25,
        }
    if token == "rich":
        return {
            "operator_set": "rich",
            "binary_operators": ["+", "-", "*", "/"],
            "unary_operators": ["square(x) = x^2", "abs_op(x) = abs(x)"],
            "extra_sympy_mappings": {"square": lambda x: x**2, "abs_op": lambda x: abs(x)},
            "default_maxsize": 25,
        }
    if token == "affine_like":
        return {
            "operator_set": "affine_like",
            "binary_operators": ["+", "-", "*"],
            "unary_operators": [],
            "extra_sympy_mappings": {},
            "default_maxsize": 15,
        }
    raise ValueError("--operator-set must be one of: restricted, rich, affine_like")


def _filter_supported_params(cls: Any, params: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(cls.__init__)
    supported = {
        name
        for name, value in signature.parameters.items()
        if name != "self" and value.kind in {value.POSITIONAL_OR_KEYWORD, value.KEYWORD_ONLY}
    }
    return {key: value for key, value in params.items() if key in supported}


def make_pysr_model(
    *,
    niterations: int,
    timeout: int,
    seed: int,
    operator_set: str,
    maxsize: int | None,
    procs: int,
    verbosity: int,
):
    from pysr import PySRRegressor

    ops = operator_config(operator_set)
    requested = {
        "niterations": int(niterations),
        "binary_operators": ops["binary_operators"],
        "unary_operators": ops["unary_operators"],
        "extra_sympy_mappings": ops["extra_sympy_mappings"],
        "maxsize": int(maxsize) if maxsize is not None else int(ops["default_maxsize"]),
        "populations": 30,
        "parsimony": 0.01,
        "procs": int(procs),
        "timeout_in_seconds": int(timeout),
        "temp_equation_file": True,
        "random_state": int(seed),
        "model_selection": "score",
        "verbosity": int(verbosity),
        "progress": False,
    }
    params = _filter_supported_params(PySRRegressor, requested)
    return PySRRegressor(**params), params


def features_in_equation(equation: str, feature_names: Sequence[str]) -> list[str]:
    used: list[str] = []
    text = str(equation)
    for name in feature_names:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(str(name))}(?![A-Za-z0-9_])", text):
            used.append(str(name))
    return used


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y.shape != pred.shape:
        raise ValueError(f"prediction shape mismatch: {pred.shape} vs {y.shape}")
    err = pred - y
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    var = float(np.mean((y - np.mean(y)) ** 2))
    r2 = None if var <= 0.0 else float(1.0 - mse / var)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2_against_constant": r2}


def dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(str)
    return out


def parse_targets(value: str | None, feature_columns: Sequence[str]) -> list[str]:
    if value is None or not str(value).strip():
        return [str(name) for name in feature_columns]
    tokens = [tok.strip() for tok in re.split(r"[\s,]+", str(value)) if tok.strip()]
    missing = [tok for tok in tokens if tok not in feature_columns]
    if missing:
        raise ValueError(f"Requested targets not present in feature columns: {missing}")
    return tokens


def grid_sample_indices(n_rows: int, sample_size: int) -> np.ndarray:
    n = int(n_rows)
    requested = int(sample_size)
    if requested <= 0 or requested >= n:
        return np.arange(n, dtype=np.int64)
    # Use one shared temporal grid for every target. This mirrors the old
    # 10-second audit more closely than per-target random samples while keeping
    # the 1-second PySR fit set small.
    return np.linspace(0, n - 1, num=requested, dtype=np.int64)


def _prepare_swat_dataframe(df: pd.DataFrame, *, time_column: str | None, label_column: str | None) -> tuple[pd.DataFrame, np.ndarray | None]:
    if time_column and time_column in df.columns:
        df = df.drop(columns=[time_column])
    labels = None
    if label_column and label_column in df.columns:
        labels = normalize_attack_labels(df[label_column])
        df = df.drop(columns=[label_column])
    numeric = df.select_dtypes(include=[np.number])
    if numeric.empty:
        raise ValueError("SWaT CSV must contain numeric feature columns after dropping metadata columns")
    return numeric, labels


def load_swat_1sec_arrays(args: argparse.Namespace) -> tuple[np.ndarray, np.ndarray | None, np.ndarray, np.ndarray | None, list[str], dict[str, Any]]:
    cfg, cfg_path = load_experiment_config_local(args.experiment)
    cfg = OmegaConf.create(OmegaConf.to_container(cfg, resolve=True))
    cfg.dataset.sampling_stride = 1
    if args.train_csv:
        cfg.dataset.train_csv = args.train_csv
    if args.test_csv:
        cfg.dataset.test_csv = args.test_csv
    train_path = Path(str(cfg.dataset.train_csv))
    test_path = Path(str(cfg.dataset.test_csv))
    if not train_path.exists() or not test_path.exists():
        raise FileNotFoundError(f"SWaT CSVs not found: train={train_path} test={test_path}")
    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)
    train_sel, train_labels = _prepare_swat_dataframe(
        train_df,
        time_column=cfg.dataset.get("time_column"),
        label_column=cfg.dataset.get("label_column"),
    )
    test_sel, test_labels = _prepare_swat_dataframe(
        test_df,
        time_column=cfg.dataset.get("time_column"),
        label_column=cfg.dataset.get("label_column"),
    )
    tag_columns = cfg.dataset.get("tag_columns")
    if tag_columns:
        missing = set(str(c) for c in tag_columns) - set(str(c) for c in train_sel.columns)
        if missing:
            raise ValueError(f"Configured SWaT tag columns missing from train CSV: {sorted(missing)}")
        feature_columns = [str(c) for c in tag_columns]
        train_sel = train_sel[feature_columns]
    else:
        feature_columns = [str(c) for c in train_sel.columns]
    test_sel = test_sel[feature_columns]
    metadata = {
        "experiment_config": str(cfg_path),
        "train_csv": str(cfg.dataset.train_csv),
        "test_csv": str(cfg.dataset.test_csv),
        "sampling_stride": int(cfg.dataset.sampling_stride),
        "train_rows": int(train_sel.shape[0]),
        "test_rows": int(test_sel.shape[0]),
        "feature_columns": feature_columns,
    }
    return (
        train_sel.to_numpy(dtype=np.float32),
        train_labels,
        test_sel.to_numpy(dtype=np.float32),
        test_labels,
        feature_columns,
        metadata,
    )


def target_arrays(target: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, str, str, int]:
    feature_columns: list[str] = _CTX["feature_columns"]
    train_current: np.ndarray = _CTX["train_current"]
    train_next: np.ndarray = _CTX["train_next"]
    train_fit_idx: np.ndarray = _CTX["train_fit_idx"]
    train_holdout_idx: np.ndarray = _CTX["train_holdout_idx"]
    sensor_target_source = str(_CTX.get("sensor_target_source", "actual_delta"))
    target_idx = int(feature_columns.index(target))
    if is_actuator("SWAT", target):
        y_all = train_next[:, target_idx]
        target_source = "actual_next"
        variable_type = "actuator"
    elif sensor_target_source == "actual_next":
        y_all = train_next[:, target_idx]
        target_source = "actual_next"
        variable_type = "sensor"
    else:
        y_all = train_next[:, target_idx] - train_current[:, target_idx]
        target_source = "actual_delta"
        variable_type = "sensor"
    return (
        train_current[train_fit_idx],
        y_all[train_fit_idx],
        train_current[train_holdout_idx],
        y_all[train_holdout_idx],
        target_source,
        variable_type,
        target_idx,
    )


def evaluate_pareto(
    model: Any,
    equations: pd.DataFrame,
    *,
    x_fit: pd.DataFrame,
    y_fit: np.ndarray,
    x_holdout: pd.DataFrame,
    y_holdout: np.ndarray,
    feature_columns: list[str],
) -> pd.DataFrame:
    out = equations.copy()
    fit_metrics = {"fit_mse": [], "fit_rmse": [], "fit_mae": [], "fit_r2_against_constant": []}
    holdout_metrics = {
        "holdout_mse": [],
        "holdout_rmse": [],
        "holdout_mae": [],
        "holdout_r2_against_constant": [],
    }
    support: list[str] = []
    errors: list[str | None] = []
    for equation_index, row in out.iterrows():
        equation = str(row.get("equation", ""))
        support.append(",".join(features_in_equation(equation, feature_columns)))
        try:
            pred_fit = np.asarray(model.predict(x_fit, index=int(equation_index)), dtype=np.float64)
            pred_holdout = np.asarray(model.predict(x_holdout, index=int(equation_index)), dtype=np.float64)
            fit = regression_metrics(y_fit, pred_fit)
            holdout = regression_metrics(y_holdout, pred_holdout)
            for key in fit_metrics:
                fit_metrics[key].append(fit[key.removeprefix("fit_")])
            for key in holdout_metrics:
                holdout_metrics[key].append(holdout[key.removeprefix("holdout_")])
            errors.append(None)
        except Exception as exc:
            for values in fit_metrics.values():
                values.append(None)
            for values in holdout_metrics.values():
                values.append(None)
            errors.append(str(exc))
    for key, values in fit_metrics.items():
        out[key] = values
    for key, values in holdout_metrics.items():
        out[key] = values
    out["equation_features"] = support
    out["evaluation_error"] = errors
    return out


def fit_one_target(payload: tuple[dict[str, Any], str]) -> dict[str, Any]:
    args_dict, target = payload
    args = argparse.Namespace(**args_dict)
    feature_columns: list[str] = _CTX["feature_columns"]
    sensor_source = str(getattr(args, "sensor_target_source", "actual_delta"))
    target_source_for_dir = "actual_next" if is_actuator("SWAT", target) or sensor_source == "actual_next" else "actual_delta"
    out_dir = Path(args.out) / f"{target}_{target_source_for_dir}"
    done = (out_dir / "pareto_front_scored.csv").exists() and (out_dir / "metadata.json").exists()
    if done and not bool(args.overwrite):
        return {"target": target, "status": "skipped_existing", "out_dir": str(out_dir)}

    started = time.time()
    x_fit_pool, y_fit_pool, x_holdout, y_holdout, target_source, variable_type, target_idx = target_arrays(target)
    sample_idx_local = np.asarray(_CTX["fit_sample_idx_local"], dtype=np.int64)
    x_sample = x_fit_pool[sample_idx_local]
    y_sample = y_fit_pool[sample_idx_local]
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "fit_sample_indices.npy", _CTX["train_fit_idx"][sample_idx_local])

    if not np.isfinite(x_sample).all() or not np.isfinite(y_sample).all():
        raise ValueError(f"{target}: non-finite sample values")
    if not np.isfinite(x_holdout).all() or not np.isfinite(y_holdout).all():
        raise ValueError(f"{target}: non-finite holdout values")

    model, pysr_params = make_pysr_model(
        niterations=int(args.niterations),
        timeout=int(args.timeout),
        seed=int(args.seed),
        operator_set=str(args.operator_set),
        maxsize=args.maxsize,
        procs=int(args.pysr_procs),
        verbosity=int(args.verbosity),
    )
    x_sample_df = pd.DataFrame(x_sample, columns=feature_columns)
    x_holdout_df = pd.DataFrame(x_holdout, columns=feature_columns)

    status = "ok"
    error = None
    try:
        fit_signature = inspect.signature(model.fit)
        if "variable_names" in fit_signature.parameters:
            model.fit(x_sample, y_sample, variable_names=feature_columns)
        else:
            model.fit(x_sample_df, y_sample)

        equations = evaluate_pareto(
            model,
            dataframe_for_csv(model.equations_),
            x_fit=x_sample_df,
            y_fit=y_sample,
            x_holdout=x_holdout_df,
            y_holdout=y_holdout,
            feature_columns=feature_columns,
        )
        equations.to_csv(out_dir / "pareto_front.csv", index=False)
        equations.to_csv(out_dir / "pareto_front_scored.csv", index=False)
        best = model.get_best()
        best_payload = best.to_dict() if hasattr(best, "to_dict") else dict(best)
        equation = str(best_payload.get("equation", ""))
        (out_dir / "best_equation.txt").write_text(equation + "\n", encoding="utf-8")
        write_json(out_dir / "best_equation.json", best_payload)
    except Exception as exc:
        status = "failed"
        error = str(exc)
        write_json(
            out_dir / "error.json",
            {
                "target": target,
                "target_source": target_source,
                "variable_type": variable_type,
                "error": error,
            },
        )
        equation = None
        best_payload = None

    elapsed = float(time.time() - started)
    metadata = {
        "target": target,
        "target_feature_index": target_idx,
        "target_source": target_source,
        "variable_type": variable_type,
        "status": status,
        "error": error,
        "best_equation": equation,
        "best_equation_payload": best_payload,
        "feature_columns": feature_columns,
        "pysr_parameters": pysr_params,
        "operator_set": args.operator_set,
        "sample_size_requested": int(args.sample_size),
        "sample_size_used": int(sample_idx_local.shape[0]),
        "sample_strategy": "shared_temporal_grid",
        "train_fit_pool_rows": int(x_fit_pool.shape[0]),
        "train_holdout_rows": int(x_holdout.shape[0]),
        "train_calibration_rows": int(_CTX["train_current"].shape[0]),
        "train_fit_fraction": float(args.fit_frac),
        "calibration_protocol": "CUSUM calibration uses all one-step training residuals, not the PySR fit subset.",
        "data_frequency": "1 second",
        "elapsed_seconds": elapsed,
        "output_dir": str(out_dir),
    }
    write_json(out_dir / "metadata.json", metadata)
    return {
        "target": target,
        "target_source": target_source,
        "variable_type": variable_type,
        "status": status,
        "out_dir": str(out_dir),
        "elapsed_seconds": elapsed,
        "error": error,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run template-free PySR state equations on 1-second SWaT variables.")
    parser.add_argument("--experiment", default="configs/experiment/swat_mlp_current_val20.yaml")
    parser.add_argument("--train-csv", default=None)
    parser.add_argument("--test-csv", default=None)
    parser.add_argument("--out", default="artifacts/swat_1sec/pareto_fronts")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--sample-size", type=int, default=8000)
    parser.add_argument("--fit-frac", type=float, default=0.8)
    parser.add_argument("--niterations", type=int, default=400)
    parser.add_argument("--timeout", type=int, default=2700)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--operator-set", default="restricted", choices=["restricted", "rich", "affine_like"])
    parser.add_argument(
        "--sensor-target-source",
        default="actual_delta",
        choices=["actual_delta", "actual_next"],
        help="Target source for sensor variables. Actuators always use actual_next.",
    )
    parser.add_argument("--maxsize", type=int, default=None)
    parser.add_argument("--pysr-procs", type=int, default=1)
    parser.add_argument("--verbosity", type=int, default=0)
    parser.add_argument("--targets", default=None, help="Optional comma/space separated target list. Default: all 51 variables.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--list-only", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_root = Path(args.out)
    out_root.mkdir(parents=True, exist_ok=True)
    train, train_labels, test, test_labels, feature_columns, data_meta = load_swat_1sec_arrays(args)
    if train.shape[1] != 51:
        raise ValueError(f"Expected 51 SWaT feature columns, got {train.shape[1]}")
    if train.shape[0] < 3:
        raise ValueError("Training split is too short for one-step dynamics")

    train_current = train[:-1].astype(np.float32, copy=False)
    train_next = train[1:].astype(np.float32, copy=False)
    n_pairs = int(train_current.shape[0])
    cutoff = int(math.floor(n_pairs * float(args.fit_frac)))
    cutoff = min(max(cutoff, 1), n_pairs - 1)
    train_fit_idx = np.arange(cutoff, dtype=np.int64)
    train_holdout_idx = np.arange(cutoff, n_pairs, dtype=np.int64)
    fit_sample_idx_local = grid_sample_indices(train_fit_idx.shape[0], int(args.sample_size))
    targets = parse_targets(args.targets, feature_columns)
    sensors = [name for name in feature_columns if not is_actuator("SWAT", name)]
    actuators = [name for name in feature_columns if is_actuator("SWAT", name)]
    pending = []
    existing = []
    for target in targets:
        target_source = "actual_next" if is_actuator("SWAT", target) or args.sensor_target_source == "actual_next" else "actual_delta"
        run_dir = out_root / f"{target}_{target_source}"
        if (run_dir / "pareto_front_scored.csv").exists() and (run_dir / "metadata.json").exists() and not args.overwrite:
            existing.append(target)
        else:
            pending.append(target)

    manifest = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "git": git_info(),
        "data": data_meta,
        "output_root": str(out_root),
        "num_feature_columns": len(feature_columns),
        "feature_columns": feature_columns,
        "sensors": sensors,
        "actuators": actuators,
        "num_targets_requested": len(targets),
        "targets_requested": targets,
        "existing_targets": existing,
        "pending_targets": pending,
        "num_existing": len(existing),
        "num_pending": len(pending),
        "train_one_step_pairs": n_pairs,
        "train_fit_pool_rows": int(train_fit_idx.shape[0]),
        "train_holdout_rows": int(train_holdout_idx.shape[0]),
        "train_calibration_rows": n_pairs,
        "fit_fraction": float(args.fit_frac),
        "sample_size": int(args.sample_size),
        "sample_strategy": "shared_temporal_grid",
        "sample_size_used": int(fit_sample_idx_local.shape[0]),
        "sample_first_fit_pool_index": int(fit_sample_idx_local[0]) if fit_sample_idx_local.size else None,
        "sample_last_fit_pool_index": int(fit_sample_idx_local[-1]) if fit_sample_idx_local.size else None,
        "niterations": int(args.niterations),
        "timeout": int(args.timeout),
        "workers": int(args.workers),
        "pysr_procs_per_worker": int(args.pysr_procs),
        "operator_set": args.operator_set,
        "sensor_target_source": args.sensor_target_source,
        "maxsize": args.maxsize,
        "calibration_protocol": "Fit PySR on first 80% pool subsample; calibrate CUSUM on all 100% of training residuals.",
        "pid": os.getpid(),
    }
    write_json(out_root / "run_manifest.json", manifest)

    print(f"Loaded SWaT 1-second train={train.shape} test={test.shape}")
    print(f"Targets requested: {len(targets)} ({len(sensors)} sensors, {len(actuators)} actuators in feature set)")
    print(f"Existing completed targets: {len(existing)}")
    print(f"Pending targets: {len(pending)}")
    print("Pending target list:", ", ".join(pending) if pending else "(none)")
    print(f"PySR fit pool rows={len(train_fit_idx)} sample_size={args.sample_size}; holdout rows={len(train_holdout_idx)}")
    print(f"CUSUM calibration rows later={n_pairs} all one-step training pairs")
    if args.list_only or not pending:
        return 0

    global _CTX
    _CTX = {
        "train_current": train_current,
        "train_next": train_next,
        "train_fit_idx": train_fit_idx,
        "train_holdout_idx": train_holdout_idx,
        "fit_sample_idx_local": fit_sample_idx_local,
        "feature_columns": feature_columns,
        "sensor_target_source": args.sensor_target_source,
    }
    payload = vars(args)
    ctx = mp.get_context("fork")
    started = time.time()
    results: list[dict[str, Any]] = []
    state_dir = out_root / ".run_state"
    state_dir.mkdir(parents=True, exist_ok=True)
    with ctx.Pool(processes=int(args.workers)) as pool:
        for result in pool.imap_unordered(fit_one_target, [(payload, target) for target in pending]):
            results.append(result)
            print(json.dumps(jsonable(result), sort_keys=True), flush=True)
            write_json(state_dir / "latest_results.json", results)
    elapsed = float(time.time() - started)
    completed = [row for row in results if row.get("status") == "ok"]
    failed = [row for row in results if row.get("status") == "failed"]
    skipped = [row for row in results if row.get("status") == "skipped_existing"]
    run_status = {
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "elapsed_seconds": elapsed,
        "num_completed": len(completed),
        "num_failed": len(failed),
        "num_skipped": len(skipped) + len(existing),
        "completed": completed,
        "failed": failed,
        "skipped_existing": skipped,
        "preexisting_targets": existing,
        "output_root": str(out_root),
    }
    write_json(out_root / "run_status.json", run_status)
    if failed:
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
