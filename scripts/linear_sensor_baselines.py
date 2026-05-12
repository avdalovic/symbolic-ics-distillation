#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

REPO_ROOT = Path(__file__).resolve().parents[1]
DISTILL_SCRIPT = REPO_ROOT / "scripts" / "distill_sensor_mlp.py"
spec = importlib.util.spec_from_file_location("distill_sensor_mlp", DISTILL_SCRIPT)
distill_sensor_mlp = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(distill_sensor_mlp)

DEFAULT_TARGET_SOURCES = ["actual_next", "actual_delta", "mlp_next", "mlp_delta"]


def read_json(path: str | Path) -> Any:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def load_target_sensors(distill_dir: str | Path) -> list[str]:
    return [str(item) for item in read_json(Path(distill_dir) / "distill_target_columns.json")]


def dedupe(values: Sequence[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []
    for value in values:
        item = str(value)
        if item not in seen:
            out.append(item)
            seen.add(item)
    return out


def split_indices(n_samples: int, eval_split: str, eval_frac: float, seed: int) -> tuple[np.ndarray, np.ndarray]:
    return distill_sensor_mlp.train_eval_indices(n_samples, eval_split, eval_frac, seed)


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    if y.shape != pred.shape:
        raise ValueError(f"Prediction shape mismatch: {pred.shape} vs {y.shape}")
    err = pred - y
    mse = float(np.mean(err**2))
    rmse = float(np.sqrt(mse))
    mae = float(np.mean(np.abs(err)))
    variance = float(np.mean((y - np.mean(y)) ** 2))
    r2 = None if variance <= 0.0 else float(1.0 - mse / variance)
    return {"mse": mse, "rmse": rmse, "mae": mae, "r2": r2}


def feature_sets_for_sensor(sensor: str, feature_columns: Sequence[str], support_config: dict[str, Any]) -> dict[str, list[str]]:
    local = [name for name in support_config.get(sensor, {}).get("local_features", []) if name in feature_columns]
    self_feature = [sensor] if sensor in feature_columns else []
    feature_sets = {
        "self_only": self_feature,
        "self_plus_local": dedupe(self_feature + local),
        "all51": list(feature_columns),
    }
    if local:
        feature_sets = {"local_support": dedupe(local), **feature_sets}
    return feature_sets


def fit_linear_baseline(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    train_idx: np.ndarray,
    holdout_idx: np.ndarray,
    *,
    model_type: str,
    ridge_alpha: float = 1.0,
) -> dict[str, Any]:
    if model_type == "ols":
        model = LinearRegression()
    elif model_type == "ridge":
        model = Ridge(alpha=float(ridge_alpha))
    else:
        raise ValueError("model_type must be ols or ridge")
    model.fit(x[train_idx], y[train_idx])
    train_pred = model.predict(x[train_idx])
    holdout_pred = model.predict(x[holdout_idx])
    train_metrics = regression_metrics(y[train_idx], train_pred)
    holdout_metrics = regression_metrics(y[holdout_idx], holdout_pred)
    coefficients = {name: float(value) for name, value in zip(feature_names, model.coef_.reshape(-1), strict=True)}
    return {
        "model_type": model_type,
        "coefficients": coefficients,
        "intercept": float(model.intercept_),
        "train_mse": train_metrics["mse"],
        "holdout_mse": holdout_metrics["mse"],
        "train_rmse": train_metrics["rmse"],
        "holdout_rmse": holdout_metrics["rmse"],
        "train_mae": train_metrics["mae"],
        "holdout_mae": holdout_metrics["mae"],
        "train_r2": train_metrics["r2"],
        "holdout_r2": holdout_metrics["r2"],
    }


def build_rows(
    *,
    sensors: Sequence[str],
    target_sources: Sequence[str],
    distill_dir: Path,
    support_config_path: Path,
    eval_split: str,
    eval_frac: float,
    seed: int,
    ridge_alpha: float,
) -> list[dict[str, Any]]:
    support_config = read_json(support_config_path)
    rows: list[dict[str, Any]] = []
    for target_source in target_sources:
        loaded = distill_sensor_mlp.load_distillation_arrays(distill_dir, target_source)
        feature_columns = list(loaded["feature_columns"])
        target_columns = list(loaded["target_columns"])
        x_all = loaded["x"].astype(np.float64)
        train_idx, holdout_idx = split_indices(x_all.shape[0], eval_split, eval_frac, seed)
        for sensor in sensors:
            if sensor not in target_columns:
                raise ValueError(f"{sensor} not present in target columns")
            target_index = int(target_columns.index(sensor))
            y = loaded["y_all"][:, target_index].astype(np.float64)
            feature_sets = feature_sets_for_sensor(sensor, feature_columns, support_config)
            for feature_set, selected_features in feature_sets.items():
                if not selected_features:
                    continue
                feature_indices = [feature_columns.index(name) for name in selected_features]
                x = x_all[:, feature_indices]
                for model_type in ["ols", "ridge"]:
                    result = fit_linear_baseline(
                        x,
                        y,
                        selected_features,
                        train_idx,
                        holdout_idx,
                        model_type=model_type,
                        ridge_alpha=ridge_alpha,
                    )
                    rows.append(
                        {
                            "sensor": sensor,
                            "target_source": target_source,
                            "feature_set": feature_set,
                            "model_type": model_type,
                            "selected_features": json.dumps(selected_features),
                            "coefficients": json.dumps(result["coefficients"], sort_keys=True),
                            "intercept": result["intercept"],
                            "train_mse": result["train_mse"],
                            "holdout_mse": result["holdout_mse"],
                            "train_rmse": result["train_rmse"],
                            "holdout_rmse": result["holdout_rmse"],
                            "train_mae": result["train_mae"],
                            "holdout_mae": result["holdout_mae"],
                            "train_r2": result["train_r2"],
                            "holdout_r2": result["holdout_r2"],
                        }
                    )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit linear diagnostic baselines for selected SWaT sensors.")
    parser.add_argument("--sensors", nargs="+", default=None)
    parser.add_argument("--target-sources", nargs="+", default=DEFAULT_TARGET_SOURCES)
    parser.add_argument("--distill-dir", default="artifacts/model_exports/swat/distillation/val20_overlap")
    parser.add_argument("--out", required=True)
    parser.add_argument("--support-config", default="configs/swat_sensor_local_support.json")
    parser.add_argument("--eval-split", default="temporal", choices=["none", "random", "temporal"])
    parser.add_argument("--eval-frac", type=float, default=0.2)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args()

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sensors = args.sensors if args.sensors is not None else load_target_sensors(args.distill_dir)
    rows = build_rows(
        sensors=sensors,
        target_sources=args.target_sources,
        distill_dir=Path(args.distill_dir),
        support_config_path=Path(args.support_config),
        eval_split=args.eval_split,
        eval_frac=args.eval_frac,
        seed=args.seed,
        ridge_alpha=args.ridge_alpha,
    )
    df = pd.DataFrame(rows)
    df.to_csv(out_path, index=False)
    print(f"Wrote {out_path} ({len(df)} rows)")
    for _, row in df[(df["feature_set"] == "local_support") & (df["model_type"] == "ols")].iterrows():
        print(
            f"{row['sensor']} {row['target_source']} OLS local_support "
            f"holdout_r2={row['holdout_r2']:.6g} holdout_mse={row['holdout_mse']:.8g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
