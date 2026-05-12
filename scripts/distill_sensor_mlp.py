#!/usr/bin/env python
from __future__ import annotations

import argparse
import inspect
import json
import math
import re
import sys
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def to_jsonable(value: Any) -> Any:
    """Recursively convert common scientific Python objects into JSON values."""
    if value is None or isinstance(value, (str, bool, int)):
        return value
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.generic):
        return to_jsonable(value.item())
    if isinstance(value, np.ndarray):
        return [to_jsonable(item) for item in value.tolist()]
    if isinstance(value, pd.Timestamp):
        return value.isoformat()
    if value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, dict):
        return {str(to_jsonable(key)): to_jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_jsonable(item) for item in value]
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
    if callable(value):
        return repr(value)
    return str(value)


def write_json(path: str | Path, payload: Any) -> Path:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(to_jsonable(payload), indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")
    return out


def normalize_target_source(value: str) -> str:
    token = str(value).strip().lower()
    aliases = {
        "mlp": "mlp_delta",
        "mlp_delta": "mlp_delta",
        "pred": "mlp_delta",
        "pred_delta": "mlp_delta",
        "mlp_next": "mlp_next",
        "pred_next": "mlp_next",
        "actual": "actual_delta",
        "actual_delta": "actual_delta",
        "actual_next": "actual_next",
    }
    if token not in aliases:
        raise ValueError("--target-source must be one of: mlp_delta, actual_delta, mlp_next, actual_next")
    return aliases[token]


def load_distillation_arrays(distill_dir: str | Path, target_source: str) -> dict[str, Any]:
    root = Path(distill_dir)
    feature_columns = [str(x) for x in read_json(root / "distill_feature_columns.json")]
    target_columns = [str(x) for x in read_json(root / "distill_target_columns.json")]
    sensor_idx = [int(i) for i in read_json(root / "distill_sensor_idx.json")]
    for target_i, feature_i in enumerate(sensor_idx):
        if target_columns[target_i] != feature_columns[feature_i]:
            raise ValueError("target_columns[j] must equal feature_columns[sensor_idx[j]]")
    x = np.load(root / "distill_inputs_current_raw.npy").astype(np.float32)
    target_source_method = "loaded"
    y_path: Path | None
    derived_from: list[str] = []
    if target_source == "mlp_delta":
        y_path = root / "distill_pred_delta_raw_mlp.npy"
        y_all = np.load(y_path).astype(np.float32)
    elif target_source == "actual_delta":
        y_path = root / "distill_actual_delta_raw.npy"
        y_all = np.load(y_path).astype(np.float32)
    elif target_source == "mlp_next":
        y_path = root / "distill_pred_next_raw_mlp.npy"
        if y_path.exists():
            y_all = np.load(y_path).astype(np.float32)
        else:
            delta_path = root / "distill_pred_delta_raw_mlp.npy"
            y_all = x[:, sensor_idx] + np.load(delta_path).astype(np.float32)
            y_path = None
            target_source_method = "derived_current_plus_delta"
            derived_from = ["distill_inputs_current_raw.npy", "distill_pred_delta_raw_mlp.npy"]
    elif target_source == "actual_next":
        y_path = root / "distill_actual_next_raw.npy"
        if y_path.exists():
            y_all = np.load(y_path).astype(np.float32)
        else:
            delta_path = root / "distill_actual_delta_raw.npy"
            y_all = x[:, sensor_idx] + np.load(delta_path).astype(np.float32)
            y_path = None
            target_source_method = "derived_current_plus_delta"
            derived_from = ["distill_inputs_current_raw.npy", "distill_actual_delta_raw.npy"]
    else:
        raise ValueError(f"Unsupported target_source: {target_source}")
    if x.ndim != 2 or x.shape[1] != len(feature_columns):
        raise ValueError(f"Expected X shape [N, {len(feature_columns)}], got {x.shape}")
    if y_all.ndim != 2 or y_all.shape[1] != len(target_columns):
        raise ValueError(f"Expected Y shape [N, {len(target_columns)}], got {y_all.shape}")
    if x.shape[0] != y_all.shape[0]:
        raise ValueError(f"X/Y sample mismatch: {x.shape[0]} vs {y_all.shape[0]}")
    return {
        "x": x,
        "y_all": y_all,
        "target_source_path": str(y_path) if y_path is not None else None,
        "target_source_method": target_source_method,
        "target_source_derived_from": derived_from,
        "feature_columns": feature_columns,
        "target_columns": target_columns,
        "sensor_idx": sensor_idx,
        "metadata": read_json(root / "metadata.json") if (root / "metadata.json").exists() else {},
    }


def select_feature_indices(
    *,
    mode: str,
    sensor: str,
    feature_columns: Sequence[str],
    attribution_dir: str | Path,
    top_k: int,
) -> tuple[list[int], str | None]:
    if mode == "unconstrained":
        return list(range(len(feature_columns))), None
    if mode != "topk":
        raise ValueError("--mode must be one of: topk, unconstrained")

    rankings_path = Path(attribution_dir) / "attribution_corr_mlp_pred_delta_rankings_nonfloored.json"
    rankings = read_json(rankings_path)
    for target in rankings.get("targets", []):
        if str(target.get("target")) == str(sensor):
            indices = [int(item["feature_index"]) for item in target.get("top_features", [])[: int(top_k)]]
            if not indices:
                raise ValueError(f"No top-k features found for target {sensor} in {rankings_path}")
            return indices, str(rankings_path)
    raise ValueError(f"Target {sensor} not found in attribution rankings: {rankings_path}")


def subsample_indices(n_samples: int, sample_size: int, seed: int) -> np.ndarray:
    n = int(n_samples)
    requested = int(sample_size)
    if requested <= 0 or requested >= n:
        return np.arange(n, dtype=np.int64)
    rng = np.random.default_rng(int(seed))
    return np.sort(rng.choice(n, size=requested, replace=False)).astype(np.int64)


def parse_sample_size(value: str | int) -> int | None:
    token = str(value).strip().lower()
    if token in {"all", "0", "none"}:
        return None
    parsed = int(token)
    if parsed < 0:
        raise ValueError("--sample-size must be a positive integer, 0, or all")
    return parsed


def choose_sample_indices(pool_indices: np.ndarray, sample_size: int | None, seed: int) -> np.ndarray:
    pool = np.asarray(pool_indices, dtype=np.int64)
    if sample_size is None or int(sample_size) == 0 or int(sample_size) >= pool.shape[0]:
        return pool.copy()
    offsets = subsample_indices(pool.shape[0], int(sample_size), seed)
    return pool[offsets]


def train_eval_indices(n_samples: int, eval_split: str, eval_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    n = int(n_samples)
    split = str(eval_split).lower()
    if split == "none":
        return np.arange(n, dtype=np.int64), np.array([], dtype=np.int64)
    frac = float(eval_frac)
    if not (0.0 < frac < 1.0):
        raise ValueError("--eval-frac must be between 0 and 1 for random/temporal splits")
    n_eval = int(round(n * frac))
    n_eval = min(max(n_eval, 1), n - 1)
    if split == "temporal":
        cutoff = n - n_eval
        return np.arange(cutoff, dtype=np.int64), np.arange(cutoff, n, dtype=np.int64)
    if split == "random":
        rng = np.random.default_rng(int(seed))
        perm = rng.permutation(n)
        eval_idx = np.sort(perm[:n_eval]).astype(np.int64)
        train_idx = np.sort(perm[n_eval:]).astype(np.int64)
        return train_idx, eval_idx
    raise ValueError("--eval-split must be one of: none, random, temporal")


def _filter_supported_params(cls, params: dict[str, Any]) -> dict[str, Any]:
    signature = inspect.signature(cls.__init__)
    supported = {
        name
        for name, value in signature.parameters.items()
        if name != "self" and value.kind in {value.POSITIONAL_OR_KEYWORD, value.KEYWORD_ONLY}
    }
    return {key: value for key, value in params.items() if key in supported}


def _sanitize_params(params: dict[str, Any]) -> dict[str, Any]:
    return to_jsonable(params)


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


def resolve_pysr_param_conflicts(params: dict[str, Any]) -> dict[str, Any]:
    out = dict(params)
    if out.get("temp_equation_file") and out.get("output_directory"):
        # PySR 1.5.x asserts that temporary equation files cannot be combined
        # with output_directory. We save our own CSV/JSON artifacts below, so
        # prefer temporary PySR internals over leaving extra backend files.
        out.pop("output_directory")
    return out


def make_pysr_model(
    *,
    output_directory: Path,
    niterations: int,
    timeout: int,
    seed: int,
    operator_set: str,
    maxsize: int | None,
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
        "procs": 4,
        "timeout_in_seconds": int(timeout),
        "temp_equation_file": True,
        "random_state": int(seed),
        "model_selection": "score",
        "output_directory": str(output_directory),
        "verbosity": 1,
        "progress": False,
    }
    params = _filter_supported_params(PySRRegressor, requested)
    params = resolve_pysr_param_conflicts(params)
    output_directory.mkdir(parents=True, exist_ok=True)
    return PySRRegressor(**params), params


def fit_model(model, x: np.ndarray, y: np.ndarray, variable_names: list[str]):
    fit_signature = inspect.signature(model.fit)
    if "variable_names" in fit_signature.parameters:
        return model.fit(x, y, variable_names=variable_names)
    return model.fit(pd.DataFrame(x, columns=variable_names), y)


def dataframe_for_csv(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    for col in out.columns:
        if out[col].dtype == object:
            out[col] = out[col].map(str)
    return out


def best_equation_payload(model) -> dict[str, Any]:
    best = model.get_best()
    payload = {}
    if hasattr(best, "to_dict"):
        payload = best.to_dict()
    elif isinstance(best, dict):
        payload = dict(best)
    else:
        payload = {"equation": str(best)}
    return {str(k): (str(v) if callable(v) else v) for k, v in payload.items()}


def features_in_equation(equation: str, feature_names: Sequence[str]) -> list[str]:
    text = str(equation)
    used = []
    for name in feature_names:
        if re.search(rf"(?<![A-Za-z0-9_]){re.escape(str(name))}(?![A-Za-z0-9_])", text):
            used.append(str(name))
    return used


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y.shape != pred.shape:
        raise ValueError(f"Prediction shape mismatch: {pred.shape} vs {y.shape}")
    err = pred - y
    mse = float(np.mean(err**2))
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(mse))
    variance = float(np.mean((y - np.mean(y)) ** 2))
    r2 = None if variance <= 0.0 else float(1.0 - mse / variance)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2_against_constant": r2}


def evaluate_equations(
    model: Any,
    equations: pd.DataFrame,
    *,
    x_train: pd.DataFrame,
    y_train: np.ndarray,
    x_holdout: pd.DataFrame | None,
    y_holdout: np.ndarray | None,
    selected_features: Sequence[str],
) -> pd.DataFrame:
    out = equations.copy()
    train_metrics: dict[str, list[Any]] = {"fit_mse": [], "fit_rmse": [], "fit_mae": [], "fit_r2_against_constant": []}
    holdout_metrics: dict[str, list[Any]] = {
        "holdout_mse": [],
        "holdout_rmse": [],
        "holdout_mae": [],
        "holdout_r2_against_constant": [],
    }
    feature_support: list[str] = []
    eval_errors: list[str | None] = []

    for equation_index, row in out.iterrows():
        equation = str(row.get("equation", ""))
        feature_support.append(",".join(features_in_equation(equation, selected_features)))
        try:
            train_pred = np.asarray(model.predict(x_train, index=int(equation_index)), dtype=np.float64)
            metrics = regression_metrics(y_train, train_pred)
            train_metrics["fit_mse"].append(metrics["mse"])
            train_metrics["fit_rmse"].append(metrics["rmse"])
            train_metrics["fit_mae"].append(metrics["mae"])
            train_metrics["fit_r2_against_constant"].append(metrics["r2_against_constant"])
            if x_holdout is not None and y_holdout is not None and len(y_holdout) > 0:
                holdout_pred = np.asarray(model.predict(x_holdout, index=int(equation_index)), dtype=np.float64)
                metrics = regression_metrics(y_holdout, holdout_pred)
                holdout_metrics["holdout_mse"].append(metrics["mse"])
                holdout_metrics["holdout_rmse"].append(metrics["rmse"])
                holdout_metrics["holdout_mae"].append(metrics["mae"])
                holdout_metrics["holdout_r2_against_constant"].append(metrics["r2_against_constant"])
            else:
                holdout_metrics["holdout_mse"].append(None)
                holdout_metrics["holdout_rmse"].append(None)
                holdout_metrics["holdout_mae"].append(None)
                holdout_metrics["holdout_r2_against_constant"].append(None)
            eval_errors.append(None)
        except Exception as exc:
            for values in train_metrics.values():
                values.append(None)
            for values in holdout_metrics.values():
                values.append(None)
            eval_errors.append(str(exc))

    for name, values in train_metrics.items():
        out[name] = values
    for name, values in holdout_metrics.items():
        out[name] = values
    out["equation_features"] = feature_support
    out["evaluation_error"] = eval_errors
    return out


def main() -> int:
    parser = argparse.ArgumentParser(description="Distill one MLP-predicted sensor delta with PySR.")
    parser.add_argument("--sensor", required=True, help="Target sensor name, e.g. LIT101.")
    parser.add_argument("--distill-dir", required=True)
    parser.add_argument("--attribution-dir", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--mode", required=True, choices=["topk", "unconstrained"])
    parser.add_argument("--top-k", type=int, default=8)
    parser.add_argument("--sample-size", default="5000", help="Number of fit rows, or all/0 for all train rows.")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--niterations", type=int, default=400)
    parser.add_argument("--timeout", type=int, default=1800)
    parser.add_argument("--target", default=None, help="Backward-compatible alias for --target-source.")
    parser.add_argument("--target-source", default=None, help="mlp_delta, actual_delta, mlp_next, or actual_next.")
    parser.add_argument("--operator-set", default="restricted", choices=["restricted", "rich", "affine_like"])
    parser.add_argument("--maxsize", type=int, default=None)
    parser.add_argument("--eval-split", default="none", choices=["none", "random", "temporal"])
    parser.add_argument("--eval-frac", type=float, default=0.2)
    args = parser.parse_args()

    target_arg = args.target_source if args.target_source is not None else args.target
    target_source = normalize_target_source(target_arg or "mlp_delta")
    requested_sample_size = parse_sample_size(args.sample_size)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    loaded = load_distillation_arrays(args.distill_dir, target_source)
    feature_columns = loaded["feature_columns"]
    target_columns = loaded["target_columns"]
    if args.sensor not in target_columns:
        raise ValueError(f"Sensor {args.sensor} not present in target columns")
    target_index = int(target_columns.index(args.sensor))
    target_feature_index = int(loaded["sensor_idx"][target_index])

    selected_indices, attribution_ranking_path = select_feature_indices(
        mode=args.mode,
        sensor=args.sensor,
        feature_columns=feature_columns,
        attribution_dir=args.attribution_dir,
        top_k=args.top_k,
    )
    selected_features = [feature_columns[i] for i in selected_indices]
    x_full = loaded["x"][:, selected_indices]
    y_full = loaded["y_all"][:, target_index]
    train_pool_idx, holdout_idx = train_eval_indices(x_full.shape[0], args.eval_split, args.eval_frac, args.seed)
    sample_idx = choose_sample_indices(train_pool_idx, requested_sample_size, args.seed)
    x_sample = x_full[sample_idx]
    y_sample = y_full[sample_idx]
    x_holdout = x_full[holdout_idx] if holdout_idx.size else None
    y_holdout = y_full[holdout_idx] if holdout_idx.size else None

    if not np.isfinite(x_sample).all() or not np.isfinite(y_sample).all():
        raise ValueError("Selected sample contains NaN or inf")
    if x_holdout is not None and (not np.isfinite(x_holdout).all() or not np.isfinite(y_holdout).all()):
        raise ValueError("Holdout sample contains NaN or inf")

    model, pysr_params = make_pysr_model(
        output_directory=out_dir / "pysr",
        niterations=args.niterations,
        timeout=args.timeout,
        seed=args.seed,
        operator_set=args.operator_set,
        maxsize=args.maxsize,
    )
    print("PySR parameters:")
    print(json.dumps(_sanitize_params(pysr_params), indent=2, sort_keys=True, allow_nan=False))
    print(
        f"Sensor={args.sensor} target_source={target_source} mode={args.mode} "
        f"operator_set={args.operator_set} eval_split={args.eval_split}"
    )
    print(f"Selected features ({len(selected_features)}): {', '.join(selected_features)}")
    x_sample_df = pd.DataFrame(x_sample, columns=selected_features)
    x_full_df = pd.DataFrame(x_full, columns=selected_features)
    x_holdout_df = pd.DataFrame(x_holdout, columns=selected_features) if x_holdout is not None else None
    fit_model(model, x_sample, y_sample, selected_features)

    equations = evaluate_equations(
        model,
        dataframe_for_csv(model.equations_),
        x_train=x_sample_df,
        y_train=y_sample,
        x_holdout=x_holdout_df,
        y_holdout=y_holdout,
        selected_features=selected_features,
    )
    equations.to_csv(out_dir / "pareto_front.csv", index=False)
    equations.to_csv(out_dir / "pareto_front_scored.csv", index=False)
    best = best_equation_payload(model)
    equation = str(best.get("equation", ""))
    used_features = features_in_equation(equation, selected_features)
    (out_dir / "best_equation.txt").write_text(equation + "\n", encoding="utf-8")
    write_json(out_dir / "best_equation.json", best)
    write_json(
        out_dir / "selected_features.json",
        {
            "mode": args.mode,
            "top_k": int(args.top_k),
            "selected_feature_indices": selected_indices,
            "selected_features": selected_features,
            "attribution_ranking_path": attribution_ranking_path,
        },
    )

    sample_pred = np.asarray(model.predict(x_sample_df), dtype=np.float64)
    full_pred = np.asarray(model.predict(x_full_df), dtype=np.float64)
    sample_mse = float(np.mean((sample_pred - y_sample.astype(np.float64)) ** 2))
    full_mse = float(np.mean((full_pred - y_full.astype(np.float64)) ** 2))
    holdout_mse = None
    holdout_metrics = None
    if x_holdout_df is not None and y_holdout is not None:
        holdout_pred = np.asarray(model.predict(x_holdout_df), dtype=np.float64)
        holdout_metrics = regression_metrics(y_holdout, holdout_pred)
        holdout_mse = holdout_metrics["mse"]
    best_loss = None
    if "loss" in best:
        try:
            best_loss = float(best["loss"])
        except Exception:
            best_loss = None

    metadata = {
        "sensor": args.sensor,
        "target_index": target_index,
        "target_feature_index": target_feature_index,
        "target_source": target_source,
        "target_source_method": loaded["target_source_method"],
        "mode": args.mode,
        "top_k": int(args.top_k),
        "selected_feature_indices": selected_indices,
        "selected_features": selected_features,
        "equation_used_features": used_features,
        "operator_set": args.operator_set,
        "maxsize": args.maxsize,
        "sample_size_requested": "all" if requested_sample_size is None else int(requested_sample_size),
        "sample_size_used": int(sample_idx.shape[0]),
        "train_pool_size": int(train_pool_idx.shape[0]),
        "holdout_size": int(holdout_idx.shape[0]),
        "eval_split": args.eval_split,
        "eval_frac": float(args.eval_frac),
        "seed": int(args.seed),
        "niterations": int(args.niterations),
        "timeout": int(args.timeout),
        "pysr_parameters": _sanitize_params(pysr_params),
        "distill_dir": str(Path(args.distill_dir)),
        "attribution_dir": str(Path(args.attribution_dir)),
        "target_source_path": loaded["target_source_path"],
        "target_source_derived_from": loaded["target_source_derived_from"],
        "attribution_ranking_path": attribution_ranking_path,
        "best_equation": equation,
        "best_equation_payload": best,
        "best_loss": best_loss,
        "sample_mse": sample_mse,
        "full_mse": full_mse,
        "holdout_mse": holdout_mse,
        "holdout_metrics": holdout_metrics,
        "target_mean": float(np.mean(y_full)),
        "target_std": float(np.std(y_full)),
        "target_min": float(np.min(y_full)),
        "target_max": float(np.max(y_full)),
        "sample_indices_path": "sample_indices.npy",
        "pareto_front_path": "pareto_front.csv",
    }
    np.save(out_dir / "sample_indices.npy", sample_idx.astype(np.int64))
    np.save(out_dir / "train_pool_indices.npy", train_pool_idx.astype(np.int64))
    np.save(out_dir / "holdout_indices.npy", holdout_idx.astype(np.int64))
    write_json(out_dir / "metadata.json", metadata)

    print("\nBest equation:")
    print(equation)
    print(f"Equation used features: {', '.join(used_features) if used_features else '(none detected)'}")
    print(
        f"best_loss={best_loss} sample_mse={sample_mse:.12g} "
        f"full_mse={full_mse:.12g} holdout_mse={holdout_mse}"
    )
    print(f"Wrote outputs to {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
