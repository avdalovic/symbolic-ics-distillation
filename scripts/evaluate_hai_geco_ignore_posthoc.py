#!/usr/bin/env python
from __future__ import annotations

import argparse
import ctypes
import importlib.util
import json
import math
import subprocess
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

GECO_IGNORED = {
    "P1_PCV02Z",
    "P2_SIT01",
    "P2_SIT02",
    "P2_VT01",
    "P2_VXT02",
    "P2_VXT03",
    "P2_VYT02",
}


def load_run_hai():
    path = REPO_ROOT / "scripts" / "run_hai_1sec_delta_full.py"
    spec = importlib.util.spec_from_file_location("run_hai_1sec_delta_full", path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def metric_summary(grid: pd.DataFrame, *, fpa_gate: int = 8) -> dict[str, Any]:
    out: dict[str, Any] = {}
    geco = grid[grid["point_kind"] == "geco_operating_point"]
    if not geco.empty:
        out["geco_operating_point"] = geco.iloc[0].to_dict()
    best = grid.sort_values(["eTaF1", "F1", "Scen", "Precision"], ascending=False).iloc[0]
    out["best_overall_eTaF1"] = best.to_dict()
    gated = grid[grid["FPA"] <= fpa_gate]
    out["low_fpa_gate"] = {
        "gate": int(fpa_gate),
        "rows": int(len(gated)),
        "best": None if gated.empty else gated.sort_values(["eTaF1", "F1", "Scen", "Precision"], ascending=False).iloc[0].to_dict(),
        "min_fpa": float(grid["FPA"].min()) if not grid.empty else None,
    }
    return out


def build_fast_cusum(out_dir: Path):
    c_path = out_dir / "fast_cusum.c"
    so_path = out_dir / "fast_cusum.so"
    source = r"""
    #include <stdint.h>
    #include <math.h>

    double train_max_blocks(const double* residual, const int64_t* lengths, int64_t nblocks, double delta) {
        int64_t offset = 0;
        double train_max = 0.0;
        for (int64_t b = 0; b < nblocks; b++) {
            double cusum = 0.0;
            for (int64_t i = 0; i < lengths[b]; i++) {
                double value = residual[offset + i];
                if (!isfinite(value)) {
                    value = 0.0;
                }
                cusum = cusum + value - delta;
                if (cusum < 0.0) {
                    cusum = 0.0;
                }
                if (cusum > train_max) {
                    train_max = cusum;
                }
            }
            offset += lengths[b];
        }
        return train_max;
    }

    void run_alarm_blocks(const double* residual, const int64_t* lengths, int64_t nblocks,
                          double delta, double threshold, double growth_cap, uint8_t* alarm) {
        int64_t offset = 0;
        for (int64_t b = 0; b < nblocks; b++) {
            double cusum = 0.0;
            for (int64_t i = 0; i < lengths[b]; i++) {
                double value = residual[offset + i];
                if (!isfinite(value)) {
                    value = 0.0;
                }
                cusum = cusum + value - delta;
                if (cusum < 0.0) {
                    cusum = 0.0;
                }
                if (cusum > growth_cap) {
                    cusum = growth_cap;
                }
                alarm[offset + i] = cusum > threshold ? 1 : 0;
            }
            offset += lengths[b];
        }
    }
    """
    existing_source = c_path.read_text(encoding="utf-8") if c_path.exists() else ""
    if not so_path.exists() or existing_source != source:
        c_path.write_text(source, encoding="utf-8")
        subprocess.run(["cc", "-O3", "-shared", "-fPIC", str(c_path), "-o", str(so_path), "-lm"], check=True)
    lib = ctypes.CDLL(str(so_path))
    arr_f64 = np.ctypeslib.ndpointer(dtype=np.float64, ndim=1, flags="C_CONTIGUOUS")
    arr_i64 = np.ctypeslib.ndpointer(dtype=np.int64, ndim=1, flags="C_CONTIGUOUS")
    arr_u8 = np.ctypeslib.ndpointer(dtype=np.uint8, ndim=1, flags="C_CONTIGUOUS")
    lib.train_max_blocks.argtypes = [arr_f64, arr_i64, ctypes.c_int64, ctypes.c_double]
    lib.train_max_blocks.restype = ctypes.c_double
    lib.run_alarm_blocks.argtypes = [arr_f64, arr_i64, ctypes.c_int64, ctypes.c_double, ctypes.c_double, ctypes.c_double, arr_u8]
    lib.run_alarm_blocks.restype = None
    return lib


def fit_channel_params(lib: Any, train_residual: np.ndarray, train_lengths: np.ndarray, s: float, g: float) -> dict[str, float]:
    finite = train_residual[np.isfinite(train_residual)]
    if finite.size == 0:
        delta = 1e-6
    else:
        delta = float(np.mean(finite) + np.std(finite))
        if not math.isfinite(delta) or delta <= 0.0:
            delta = 1e-6
    train_max = float(lib.train_max_blocks(train_residual, train_lengths, int(train_lengths.shape[0]), float(delta)))
    threshold = float(s) * train_max
    if not math.isfinite(threshold) or threshold <= 0.0:
        threshold = 1e-6
    growth_cap = threshold + float(g) * delta
    if not math.isfinite(growth_cap) or growth_cap <= 0.0:
        growth_cap = 1e-6
    return {"delta": delta, "threshold": threshold, "growth_cap": growth_cap, "max_calib_cusum": train_max}


def alarm_for_point(lib: Any, test_residual: np.ndarray, test_lengths: np.ndarray, params: dict[str, float]) -> np.ndarray:
    alarm = np.zeros(test_residual.shape[0], dtype=np.uint8)
    lib.run_alarm_blocks(
        test_residual,
        test_lengths,
        int(test_lengths.shape[0]),
        float(params["delta"]),
        float(params["threshold"]),
        float(params["growth_cap"]),
        alarm,
    )
    return alarm


def point_counts(labels: np.ndarray, alarms: np.ndarray) -> dict[str, int]:
    y = (np.asarray(labels) >= 1).astype(np.int64)
    a = (np.asarray(alarms) >= 1).astype(np.int64)
    return {
        "TP": int(np.sum((y == 1) & (a == 1))),
        "FP": int(np.sum((y == 0) & (a == 1))),
        "TN": int(np.sum((y == 0) & (a == 0))),
        "FN": int(np.sum((y == 1) & (a == 0))),
    }


def evaluate_variant_fast(
    run_hai: Any,
    lib: Any,
    arrays: dict[str, Any],
    rows: list[dict[str, Any]],
    residual_cache: dict[str, dict[str, np.ndarray]],
    points: list[tuple[float, float, str]],
    variant: str,
) -> tuple[pd.DataFrame, dict[tuple[float, float], np.ndarray]]:
    from ics_symbolic_distill.detection.metrics import compute_detection_metrics

    labels = arrays["labels"]
    test_lengths = np.asarray(arrays["test_block_lengths"], dtype=np.int64)
    train_lengths = np.asarray(arrays["train_block_lengths"], dtype=np.int64)
    cache: dict[tuple[float, float], np.ndarray] = {}
    grid_rows: list[dict[str, Any]] = []
    for s, g, point_kind in points:
        system_alarm = np.zeros(labels.shape[0], dtype=np.uint8)
        for row in rows:
            target = str(row["target"])
            residuals = residual_cache[target]
            params = fit_channel_params(lib, residuals["train"], train_lengths, float(s), float(g))
            system_alarm |= alarm_for_point(lib, residuals["test"], test_lengths, params)
        cache[(float(s), float(g))] = system_alarm
        metrics = compute_detection_metrics(labels, system_alarm, expand_steps=60)
        grid_rows.append(
            {
                "dataset": "HAI",
                "variant": variant,
                "S": float(s),
                "G": float(g),
                "point_kind": point_kind,
                "num_monitored": len(rows),
                "monitored_symbolic": sum(str(r.get("variable_type")) == "sensor" for r in rows),
                "monitored_persistence": sum(str(r.get("variable_type")) == "persistence" for r in rows),
                "Precision": metrics["point_precision"],
                "Recall": metrics["point_recall"],
                "F1": metrics["point_f1"],
                "eTaP": metrics["eTaP"],
                "eTaR": metrics["eTaR"],
                "eTaF1": metrics["eTaF1"],
                "FPA": metrics["FPA"],
                "Scen": metrics["scenario_detection_rate"],
                **point_counts(labels, system_alarm),
            }
        )
    return pd.DataFrame(grid_rows), cache


def make_detection_points(run_hai: Any, geco_s: float, geco_g: float, *, smoke: bool = False, s_max: float | None = None) -> list[tuple[float, float, str]]:
    if s_max is None:
        return run_hai.make_detection_points(float(geco_s), float(geco_g), smoke=smoke)
    s_values = list(run_hai.S_VALUES)
    if float(s_max) > max(s_values):
        s_values.extend(np.arange(5.5, float(s_max) + 1e-9, 0.5).tolist())
    s_values.append(float(geco_s))
    g_values = list(run_hai.G_VALUES)
    points: list[tuple[float, float, str]] = []
    seen: set[tuple[float, float]] = set()
    for s in s_values:
        for g in g_values:
            key = (round(float(s), 12), round(float(g), 12))
            if key in seen:
                continue
            seen.add(key)
            kind = "geco_operating_point" if abs(float(s) - float(geco_s)) < 1e-12 and abs(float(g) - float(geco_g)) < 1e-12 else "grid"
            points.append((float(s), float(g), kind))
    key = (round(float(geco_s), 12), round(float(geco_g), 12))
    if key not in seen:
        points.append((float(geco_s), float(geco_g), "geco_operating_point"))
    return points


def select_variant(rows: list[dict[str, Any]], variant: str) -> list[dict[str, Any]]:
    if variant == "all":
        return list(rows)
    if variant == "without_geco_ignored":
        return [row for row in rows if str(row["target"]) not in GECO_IGNORED]
    if variant == "geco_ignored_only":
        return [row for row in rows if str(row["target"]) in GECO_IGNORED]
    raise ValueError(f"unknown variant: {variant}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Post-hoc HAI evaluation with GeCo ignored variables removed or isolated.")
    parser.add_argument("--baseline-dir", default="artifacts/experiments/hai_baseline_seed0")
    parser.add_argument("--output-dir", default="artifacts/experiments/hai_posthoc_geco_ignore")
    parser.add_argument("--data-dir", default="data/hai/ipal")
    parser.add_argument("--geco-model", default=None)
    parser.add_argument("--variants", nargs="+", default=["without_geco_ignored", "geco_ignored_only"])
    parser.add_argument("--s-max", type=float, default=None, help="Optionally extend the HAI scale-factor grid to this value.")
    args = parser.parse_args()

    run_hai = load_run_hai()
    baseline_dir = REPO_ROOT / args.baseline_dir
    out_dir = REPO_ROOT / args.output_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    lib = build_fast_cusum(out_dir)

    selected_df = pd.read_csv(baseline_dir / "selected_equations.csv")
    selected_rows = selected_df.to_dict(orient="records")
    arrays = run_hai.load_hai_arrays(argparse.Namespace(data_dir=str(REPO_ROOT / args.data_dir), sample_size=None))
    geco_s, geco_g, geco_source = run_hai.parse_geco_point(args.geco_model)
    points = make_detection_points(run_hai, float(geco_s), float(geco_g), smoke=False, s_max=args.s_max)

    residual_cache: dict[str, dict[str, Any]] = {}
    for row in selected_rows:
        target = str(row["target"])
        if str(row.get("variable_type")) == "persistence":
            residual_cache[target] = {
                "train": run_hai.persistence_residual(arrays, target, split="train"),
                "test": run_hai.persistence_residual(arrays, target, split="test"),
            }
        else:
            equation = str(row.get("sympy_format") or row.get("equation"))
            residual_cache[target] = {
                "train": run_hai.prediction_residual(arrays, target, equation, split="train"),
                "test": run_hai.prediction_residual(arrays, target, equation, split="test"),
            }

    summary: dict[str, Any] = {
        "baseline_dir": str(baseline_dir.relative_to(REPO_ROOT)),
        "geco_ignored": sorted(GECO_IGNORED),
        "geco_point": {"S": float(geco_s), "G": float(geco_g), "source": geco_source},
        "s_max": None if args.s_max is None else float(args.s_max),
        "num_detection_points": len(points),
        "variants": {},
    }
    for variant in args.variants:
        rows = select_variant(selected_rows, variant)
        grid, alarm_cache = evaluate_variant_fast(run_hai, lib, arrays, rows, residual_cache, points, variant)
        grid.to_csv(out_dir / f"{variant}_detection_grid.csv", index=False)
        geco_row = grid[grid["point_kind"] == "geco_operating_point"]
        geco_row.to_csv(out_dir / f"{variant}_geco_operating_point.csv", index=False)
        if not geco_row.empty:
            key = (float(geco_row.iloc[0]["S"]), float(geco_row.iloc[0]["G"]))
            system_alarm = alarm_cache[key]
            run_hai.write_per_attack(arrays, system_alarm, out_dir / f"{variant}_per_attack.csv")
            run_hai.write_per_file_metrics(arrays, system_alarm, out_dir / f"{variant}_per_file_metrics.csv")
        summary["variants"][variant] = {
            "num_monitored": len(rows),
            "targets": [str(row["target"]) for row in rows],
            **metric_summary(grid, fpa_gate=8),
        }

    (out_dir / "summary.json").write_text(json.dumps(summary, indent=2, allow_nan=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, allow_nan=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
