#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import sys
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

REPO_ROOT = Path(__file__).resolve().parents[1]
DISTILL_SCRIPT = REPO_ROOT / "scripts" / "distill_sensor_mlp.py"
spec = importlib.util.spec_from_file_location("distill_sensor_mlp", DISTILL_SCRIPT)
distill_sensor_mlp = importlib.util.module_from_spec(spec)
assert spec.loader is not None
spec.loader.exec_module(distill_sensor_mlp)


TARGET_SOURCES = ["actual_next", "actual_delta", "mlp_next", "mlp_delta"]
FEATURE_SETS = {
    "physics_delta_support": ["FIT101", "FIT201"],
    "physics_next_support": ["LIT101", "FIT101", "FIT201"],
    "extended_local_support": ["LIT101", "FIT101", "FIT201", "MV101", "P101"],
    "all51": None,
}


def regression_metrics(y_true: np.ndarray, y_pred: np.ndarray) -> dict[str, float | None]:
    y = np.asarray(y_true, dtype=np.float64).reshape(-1)
    pred = np.asarray(y_pred, dtype=np.float64).reshape(-1)
    err = pred - y
    mse = float(np.mean(err**2))
    variance = float(np.mean((y - np.mean(y)) ** 2))
    r2 = None if variance <= 0.0 else float(1.0 - mse / variance)
    return {"mse": mse, "r2": r2}


def split_temporal(n_samples: int, eval_frac: float) -> tuple[np.ndarray, np.ndarray]:
    n_eval = int(round(int(n_samples) * float(eval_frac)))
    n_eval = min(max(n_eval, 1), int(n_samples) - 1)
    cutoff = int(n_samples) - n_eval
    return np.arange(cutoff, dtype=np.int64), np.arange(cutoff, int(n_samples), dtype=np.int64)


def fit_linear_baseline(
    x: np.ndarray,
    y: np.ndarray,
    feature_names: Sequence[str],
    train_idx: np.ndarray,
    holdout_idx: np.ndarray,
    *,
    model_type: str,
    ridge_alpha: float = 1.0,
) -> dict:
    if model_type == "ols":
        model = LinearRegression()
    elif model_type == "ridge":
        model = Ridge(alpha=float(ridge_alpha))
    else:
        raise ValueError("model_type must be ols or ridge")
    model.fit(x[train_idx], y[train_idx])
    train_metrics = regression_metrics(y[train_idx], model.predict(x[train_idx]))
    holdout_metrics = regression_metrics(y[holdout_idx], model.predict(x[holdout_idx]))
    coefficients = {name: float(value) for name, value in zip(feature_names, model.coef_.reshape(-1), strict=True)}
    return {
        "model_type": model_type,
        "coefficients": coefficients,
        "intercept": float(model.intercept_),
        "train_mse": train_metrics["mse"],
        "holdout_mse": holdout_metrics["mse"],
        "train_r2": train_metrics["r2"],
        "holdout_r2": holdout_metrics["r2"],
    }


def build_rows(distill_dir: Path, sensor: str, eval_frac: float, ridge_alpha: float) -> list[dict]:
    rows: list[dict] = []
    for target_source in TARGET_SOURCES:
        loaded = distill_sensor_mlp.load_distillation_arrays(distill_dir, target_source)
        feature_columns = list(loaded["feature_columns"])
        target_columns = list(loaded["target_columns"])
        if sensor not in target_columns:
            raise ValueError(f"{sensor} not present in target columns")
        target_index = int(target_columns.index(sensor))
        y = loaded["y_all"][:, target_index].astype(np.float64)
        train_idx, holdout_idx = split_temporal(y.shape[0], eval_frac)
        for feature_set, names in FEATURE_SETS.items():
            selected = feature_columns if names is None else list(names)
            missing = [name for name in selected if name not in feature_columns]
            if missing:
                raise ValueError(f"Missing features for {feature_set}: {missing}")
            feature_idx = [feature_columns.index(name) for name in selected]
            x = loaded["x"][:, feature_idx].astype(np.float64)
            for model_type in ["ols", "ridge"]:
                result = fit_linear_baseline(
                    x,
                    y,
                    selected,
                    train_idx,
                    holdout_idx,
                    model_type=model_type,
                    ridge_alpha=ridge_alpha,
                )
                rows.append(
                    {
                        "target_source": target_source,
                        "sensor": sensor,
                        "feature_set": feature_set,
                        "model_type": model_type,
                        "selected_features": json.dumps(selected),
                        "coefficients": json.dumps(result["coefficients"], sort_keys=True),
                        "intercept": result["intercept"],
                        "train_mse": result["train_mse"],
                        "holdout_mse": result["holdout_mse"],
                        "train_r2": result["train_r2"],
                        "holdout_r2": result["holdout_r2"],
                    }
                )
    return rows


def main() -> int:
    parser = argparse.ArgumentParser(description="Fit LIT101 linear coefficient baselines.")
    parser.add_argument("--distill-dir", default="artifacts/model_exports/swat/distillation/val20_overlap")
    parser.add_argument("--out", required=True)
    parser.add_argument("--sensor", default="LIT101")
    parser.add_argument("--eval-frac", type=float, default=0.2)
    parser.add_argument("--ridge-alpha", type=float, default=1.0)
    args = parser.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    rows = build_rows(Path(args.distill_dir), args.sensor, args.eval_frac, args.ridge_alpha)
    df = pd.DataFrame(rows)
    csv_path = out_dir / "lit101_linear_baselines.csv"
    df.to_csv(csv_path, index=False)
    print(f"Wrote {csv_path}")
    for _, row in df[
        (df["model_type"] == "ols")
        & (df["feature_set"].isin(["physics_delta_support", "physics_next_support"]))
    ].iterrows():
        print(
            f"{row['target_source']} {row['feature_set']} OLS "
            f"holdout_mse={row['holdout_mse']:.8g} holdout_r2={row['holdout_r2']:.6g} "
            f"coefficients={row['coefficients']} intercept={row['intercept']:.8g}"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
