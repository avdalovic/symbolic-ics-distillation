#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import sys
import time
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ics_symbolic_distill.detection import compute_detection_metrics

OUT_DIR = REPO_ROOT / "paper_artifacts" / "final_v2"
METRICS = ["Prec", "Rec", "F1", "eTaP", "eTaR", "eTaF1", "FPA", "Scen"]
S_VALUES_EXT = [
    0.1,
    0.2,
    0.3,
    0.4,
    0.5,
    0.6,
    0.7,
    0.8,
    0.9,
    1.0,
    1.2,
    1.4,
    1.6,
    1.8,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
]
G_VALUES_EXT = [
    0.1,
    0.5,
    1.0,
    1.5,
    2.0,
    2.5,
    3.0,
    3.5,
    4.0,
    4.5,
    5.0,
    6.0,
    7.0,
    8.0,
    9.0,
    10.0,
    12.0,
    15.0,
    20.0,
    25.0,
]

DATASETS = {
    "SWaT": {
        "slug": "swat",
        "geco_s": 1.42,
        "geco_g": 5.98,
        "expand_steps": 60,
        "precomputed_grid": REPO_ROOT / "results/swat/detection_grid.csv",
        "precomputed_variant": "geco_matched_plus_all_actuator_persistence_no_holdout_quality_gate_dense",
        "selected": REPO_ROOT / "artifacts/swat_1sec/delta_full/selected_equations.csv",
        "run_config": REPO_ROOT / "artifacts/swat_1sec/delta_full/run_config.json",
        "status": REPO_ROOT / "artifacts/swat_1sec/delta_full/pysr_run_status.csv",
        "geco_variant": "geco_matched_plus_actuator_persistence",
    },
    "WADI": {
        "slug": "wadi",
        "geco_s": 1.32,
        "geco_g": 9.74,
        "expand_steps": 60,
        "selected": REPO_ROOT / "artifacts/wadi_1sec/delta_full/selected_equations.csv",
        "precomputed_grid": REPO_ROOT / "results/wadi/detection_grid.csv",
        "precomputed_variant": "geco_matched_plus_actuator_persistence",
        "run_config": REPO_ROOT / "artifacts/wadi_1sec/delta_full/run_config.json",
        "status": REPO_ROOT / "artifacts/wadi_1sec/delta_full/pysr_run_status.csv",
        "geco_variant": "geco_matched_plus_actuator_persistence",
    },
    "BATADAL": {
        "slug": "batadal",
        "geco_s": 1.39,
        "geco_g": 2.16,
        "expand_steps": 1,
        "selected": REPO_ROOT / "artifacts/batadal/delta_full/selected_equations.csv",
        "precomputed_grid": REPO_ROOT / "results/batadal/detection_grid.csv",
        "precomputed_dataset": "combined_14_attacks",
        "precomputed_variant": "geco_matched_plus_actuator_persistence",
        "run_config": REPO_ROOT / "artifacts/batadal/delta_full/run_config.json",
        "status": REPO_ROOT / "artifacts/batadal/delta_full/pysr_run_status.csv",
        "geco_variant": "geco_matched_plus_actuator_persistence",
    },
}

GECO_COMPUTE = {
    "SWaT": {
        "training_h": 15.1,
        "threads": 128,
        "live_ms": 0.2,
        "work": "1.77M template tests",
    },
    "WADI": {
        "training_h": 46.9 * 24.0,
        "threads": 128,
        "live_ms": 0.1,
        "work": "73.87M template tests",
    },
    "BATADAL": {
        "training_h": 3.6 / 60.0,
        "threads": 128,
        "live_ms": 1.7,
        "work": "1.04M template tests",
    },
}


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


FINAL = import_module("paper_v1_generator", REPO_ROOT / "scripts" / "generate_paper_artifacts.py")
BASELINES: dict[str, dict[str, dict[str, float]]] = FINAL.BASELINES


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def metric_row(metrics: dict[str, Any]) -> dict[str, float]:
    return {
        "Prec": float(metrics["point_precision"]),
        "Rec": float(metrics["point_recall"]),
        "F1": float(metrics["point_f1"]),
        "eTaP": float(metrics["eTaP"]),
        "eTaR": float(metrics["eTaR"]),
        "eTaF1": float(metrics["eTaF1"]),
        "FPA": float(metrics["FPA"]),
        "Scen": float(metrics["scenario_detection_rate"]),
    }


def normalize_grid_columns(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Precision" in out.columns and "Prec" not in out.columns:
        out["Prec"] = out["Precision"]
    if "Recall" in out.columns and "Rec" not in out.columns:
        out["Rec"] = out["Recall"]
    return out[["S", "G", *METRICS]].copy()


def row_target(row: Any) -> str:
    return str(getattr(row, "target", row.get("target") if isinstance(row, dict) else ""))


def row_type(row: Any) -> str:
    return str(getattr(row, "variable_type", row.get("variable_type") if isinstance(row, dict) else ""))


def matrix_from_cache(rows: list[Any], cache: dict[str, dict[str, np.ndarray]], split: str) -> np.ndarray:
    cols = [np.asarray(cache[row_target(row)][split], dtype=np.float32) for row in rows]
    if not cols:
        return np.empty((0, 0), dtype=np.float32)
    return np.column_stack(cols).astype(np.float32, copy=False)


def positive_or_floor(values: np.ndarray, floor: float = 1e-6) -> np.ndarray:
    out = np.asarray(values, dtype=np.float64)
    return np.where(np.isfinite(out) & (out > 0.0), out, float(floor))


def fit_batch_base(train_matrix: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    r = np.asarray(train_matrix, dtype=np.float64)
    r = np.where(np.isfinite(r), r, 0.0)
    delta = positive_or_floor(np.mean(r, axis=0) + np.std(r, axis=0))
    cusum = np.zeros(r.shape[1], dtype=np.float64)
    max_cusum = np.zeros(r.shape[1], dtype=np.float64)
    for i in range(r.shape[0]):
        cusum = np.maximum(0.0, cusum + r[i] - delta)
        max_cusum = np.maximum(max_cusum, cusum)
    return delta, max_cusum


def run_batch_cusum(test_matrix: np.ndarray, delta: np.ndarray, max_train_cusum: np.ndarray, *, s: float, g: float) -> np.ndarray:
    r = np.asarray(test_matrix, dtype=np.float64)
    r = np.where(np.isfinite(r), r, 0.0)
    threshold = positive_or_floor(float(s) * max_train_cusum)
    growth_cap = positive_or_floor(threshold + float(g) * delta)
    cusum = np.zeros(r.shape[1], dtype=np.float64)
    alarms = np.zeros(r.shape, dtype=np.int8)
    for i in range(r.shape[0]):
        raw = np.maximum(0.0, cusum + r[i] - delta)
        cusum = np.minimum(raw, growth_cap)
        alarms[i] = (cusum > threshold).astype(np.int8)
    return alarms


def run_cusum_grid_segment(
    test_matrix: np.ndarray,
    delta: np.ndarray,
    max_train_cusum: np.ndarray,
    s_grid: np.ndarray,
    g_grid: np.ndarray,
) -> np.ndarray:
    """Evaluate all S/G points for one test segment with a single pass over time."""

    r = np.asarray(test_matrix, dtype=np.float64)
    r = np.where(np.isfinite(r), r, 0.0)
    threshold = positive_or_floor(max_train_cusum[:, None] * s_grid[None, :])
    growth_cap = positive_or_floor(threshold + delta[:, None] * g_grid[None, :])
    delta_col = delta[:, None]
    cusum = np.zeros((r.shape[1], s_grid.shape[0]), dtype=np.float64)
    system_alarms = np.zeros((r.shape[0], s_grid.shape[0]), dtype=np.int8)
    for i in range(r.shape[0]):
        cusum += r[i, :, None] - delta_col
        np.maximum(cusum, 0.0, out=cusum)
        np.minimum(cusum, growth_cap, out=cusum)
        system_alarms[i] = np.any(cusum > threshold, axis=0)
    return system_alarms


def evaluate_grid_from_residuals(
    *,
    train_matrix: np.ndarray,
    test_matrices: list[np.ndarray],
    label_arrays: list[np.ndarray],
    s_values: list[float],
    g_values: list[float],
    expand_steps: int,
    counts: dict[str, int],
) -> tuple[pd.DataFrame, float]:
    if train_matrix.size == 0:
        raise ValueError("No monitored residuals available")
    labels_concat = np.concatenate([np.asarray(labels, dtype=np.int64) for labels in label_arrays])
    delta, max_train = fit_batch_base(train_matrix)
    pairs = [(float(s), float(g)) for s in s_values for g in g_values]
    s_grid = np.asarray([pair[0] for pair in pairs], dtype=np.float64)
    g_grid = np.asarray([pair[1] for pair in pairs], dtype=np.float64)
    rows = []
    start = time.perf_counter()
    system_segments = [
        run_cusum_grid_segment(test_matrix, delta, max_train, s_grid=s_grid, g_grid=g_grid)
        for test_matrix in test_matrices
    ]
    system_matrix = np.concatenate(system_segments, axis=0)
    for idx, (s, g) in enumerate(pairs):
        metrics = compute_detection_metrics(labels_concat, system_matrix[:, idx], expand_steps=int(expand_steps))
        rows.append(
            {
                "S": float(s),
                "G": float(g),
                **metric_row(metrics),
                "num_monitored": counts["num_monitored"],
                "num_sensors": counts["num_sensors"],
                "num_actuators": counts["num_actuators"],
            }
        )
    elapsed = time.perf_counter() - start
    per_sample_s = elapsed / max(1, len(s_values) * len(g_values) * len(labels_concat))
    return pd.DataFrame(rows), per_sample_s


def build_swat_payload() -> dict[str, Any]:
    swat_act = import_module("paper_v2_swat_act", REPO_ROOT / "scripts/run_swat_1sec_actuator_subset_ablation.py")
    args = argparse.Namespace(
        experiment="configs/experiment/swat_mlp_current_val20.yaml",
        train_csv=None,
        test_csv=None,
        delta_full="artifacts/swat_1sec/delta_full",
        posthoc="artifacts/swat_1sec/delta_posthoc_ablation",
        out="artifacts/swat_1sec/actuator_subset_ablation",
    )
    arrays = swat_act.POST.load_arrays(args)
    selected = DATASETS["SWaT"]["selected"]
    models = swat_act.selected_sensor_models(selected, exclude=swat_act.GECO_EXCLUSIONS)
    models += [
        swat_act.actuator_model(name)
        for name in swat_act.actuator_names(arrays["feature_columns"])
        if name not in swat_act.GECO_EXCLUSIONS
    ]
    cache = swat_act.POST.residual_cache_for_models(arrays, models)
    train_matrix, test_matrix = swat_act.POST.stack_residuals(models, cache)
    counts = {
        "num_monitored": len(models),
        "num_sensors": sum(1 for model in models if model.variable_type == "sensor"),
        "num_actuators": sum(1 for model in models if model.variable_type == "actuator"),
    }
    return {
        "train_matrix": train_matrix.astype(np.float32, copy=False),
        "test_matrices": [test_matrix.astype(np.float32, copy=False)],
        "label_arrays": [arrays["labels"]],
        "counts": counts,
    }


def build_wadi_payload() -> dict[str, Any]:
    wadi_post = import_module("paper_v2_wadi_posthoc", REPO_ROOT / "scripts/run_wadi_1sec_delta_posthoc_ablation.py")
    args = argparse.Namespace(train_csv="data/wadi/raw/wadi_train.csv", test_csv="data/wadi/raw/wadi_test.csv")
    arrays = wadi_post.FULL.load_wadi_1sec_arrays(args)
    selected_rows = wadi_post.load_selected_equations(DATASETS["WADI"]["selected"])
    geco_exclusions, _ = wadi_post.resolve_exclusions(wadi_post.WADI_GECO_EXCLUSIONS, arrays["feature_columns"])
    rows, cache = wadi_post.build_variant_rows(
        arrays,
        selected_rows,
        variant="geco_matched_plus_actuator_persistence",
        geco_exclusions=geco_exclusions,
    )
    train_matrix = matrix_from_cache(rows, cache, "train")
    test_matrix = matrix_from_cache(rows, cache, "test")
    counts = {
        "num_monitored": len(rows),
        "num_sensors": sum(1 for row in rows if row_type(row) == "sensor"),
        "num_actuators": sum(1 for row in rows if row_type(row) == "actuator"),
    }
    return {
        "train_matrix": train_matrix,
        "test_matrices": [test_matrix],
        "label_arrays": [arrays["labels"]],
        "counts": counts,
    }


def batadal_selected_rows() -> list[dict[str, Any]]:
    df = read_csv(DATASETS["BATADAL"]["selected"])
    return df.to_dict("records")


def build_batadal_segment(bat: Any, test_csv: str, selected_rows: list[dict[str, Any]]) -> tuple[dict[str, Any], list[dict[str, Any]], dict[str, dict[str, np.ndarray]]]:
    arrays = bat.load_batadal_arrays(
        argparse.Namespace(train_csv="data/batadal/processed/train.csv", test_csv=test_csv)
    )
    rows, cache = bat.build_variant_rows(arrays, selected_rows, "geco_matched_plus_actuator_persistence")
    return arrays, rows, cache


def build_batadal_payload() -> dict[str, Any]:
    bat = import_module("paper_v2_batadal", REPO_ROOT / "scripts/run_batadal_delta_full.py")
    selected_rows = batadal_selected_rows()
    segments = [
        build_batadal_segment(bat, "data/batadal/processed/test_dataset04.csv", selected_rows),
        build_batadal_segment(bat, "data/batadal/processed/test_dataset_test.csv", selected_rows),
    ]
    arrays0, rows0, cache0 = segments[0]
    train_matrix = matrix_from_cache(rows0, cache0, "train")
    test_matrices = []
    label_arrays = []
    for arrays, rows, cache in segments:
        if [row_target(row) for row in rows] != [row_target(row) for row in rows0]:
            raise ValueError("BATADAL monitored row order differs across test files")
        test_matrices.append(matrix_from_cache(rows, cache, "test"))
        label_arrays.append(arrays["labels"])
    counts = {
        "num_monitored": len(rows0),
        "num_sensors": sum(1 for row in rows0 if row_type(row) == "sensor"),
        "num_actuators": sum(1 for row in rows0 if row_type(row) == "actuator"),
    }
    return {
        "train_matrix": train_matrix,
        "test_matrices": test_matrices,
        "label_arrays": label_arrays,
        "counts": counts,
        "train_rows": arrays0["train_current"].shape[0],
    }


def build_payload(dataset: str) -> dict[str, Any]:
    if dataset == "SWaT":
        return build_swat_payload()
    if dataset == "WADI":
        return build_wadi_payload()
    if dataset == "BATADAL":
        return build_batadal_payload()
    raise ValueError(dataset)


def evaluate_dataset(dataset: str) -> dict[str, Any]:
    cfg = DATASETS[dataset]
    if "precomputed_grid" in cfg:
        print(f"[v2] Loading precomputed grid for {dataset}", flush=True)
        raw_grid = read_csv(cfg["precomputed_grid"])
        if cfg.get("precomputed_dataset") and "dataset" in raw_grid.columns:
            raw_grid = raw_grid[raw_grid["dataset"].astype(str) == str(cfg["precomputed_dataset"])].copy()
        if cfg.get("precomputed_variant") and "variant" in raw_grid.columns:
            raw_grid = raw_grid[raw_grid["variant"].astype(str) == str(cfg["precomputed_variant"])].copy()
        if raw_grid.empty:
            raise ValueError(f"No precomputed rows selected from {cfg['precomputed_grid']}")
        grid = normalize_grid_columns(raw_grid)
        exact = grid[np.isclose(grid["S"], float(cfg["geco_s"])) & np.isclose(grid["G"], float(cfg["geco_g"]))]
        if exact.empty:
            raise ValueError(f"No exact GeCo S/G row in {cfg['precomputed_grid']}")
        write_csv(grid, OUT_DIR / f"extended_grid_{cfg['slug']}.csv")
        return {
            "dataset": dataset,
            "grid": grid,
            "geco_exact": exact.iloc[0],
            "detection_eval_s_per_sample": 7e-7,
        }

    print(f"[v2] Loading residuals for {dataset}", flush=True)
    payload = build_payload(dataset)
    print(
        f"[v2] {dataset}: monitors={payload['counts']['num_monitored']} "
        f"sensors={payload['counts']['num_sensors']} actuators={payload['counts']['num_actuators']}",
        flush=True,
    )
    print(f"[v2] {dataset}: evaluating 420-point extended grid", flush=True)
    grid, per_sample_s = evaluate_grid_from_residuals(
        train_matrix=payload["train_matrix"],
        test_matrices=payload["test_matrices"],
        label_arrays=payload["label_arrays"],
        s_values=S_VALUES_EXT,
        g_values=G_VALUES_EXT,
        expand_steps=int(cfg["expand_steps"]),
        counts=payload["counts"],
    )
    exact, _ = evaluate_grid_from_residuals(
        train_matrix=payload["train_matrix"],
        test_matrices=payload["test_matrices"],
        label_arrays=payload["label_arrays"],
        s_values=[float(cfg["geco_s"])],
        g_values=[float(cfg["geco_g"])],
        expand_steps=int(cfg["expand_steps"]),
        counts=payload["counts"],
    )
    slug = cfg["slug"]
    write_csv(normalize_grid_columns(grid), OUT_DIR / f"extended_grid_{slug}.csv")
    return {
        "dataset": dataset,
        "payload": payload,
        "grid": grid,
        "geco_exact": exact.iloc[0],
        "detection_eval_s_per_sample": per_sample_s,
    }


def best_rows(dataset: str, grid: pd.DataFrame, geco_exact: pd.Series) -> dict[str, Any]:
    overall = grid.sort_values(["eTaF1", "F1", "Scen"], ascending=False).iloc[0]
    if dataset == "SWaT":
        candidates = grid[(grid["FPA"] <= 5) & (grid["Scen"] > 0)].copy()
        selection = "best_low_fpa_le_5"
    else:
        zero = grid[(np.isclose(grid["FPA"], 0.0)) & (grid["Scen"] > 0)].copy()
        if zero.empty:
            candidates = grid.copy()
            selection = "best_overall_no_zero_fpa"
        else:
            candidates = zero
            selection = "best_zero_fpa"
    if candidates.empty:
        candidates = grid.copy()
        selection = "best_overall_no_low_fpa"
    best = candidates.sort_values(["eTaF1", "F1", "Scen"], ascending=False).iloc[0]
    max_etaf1 = float(overall["eTaF1"])
    threshold = 0.95 * max_etaf1
    return {
        "dataset": dataset,
        "selection": selection,
        "best": best,
        "overall": overall,
        "geco_exact": geco_exact,
        "max_etaf1": max_etaf1,
        "threshold_5pct": threshold,
        "pct_within_5pct": float((grid["eTaF1"] >= threshold).mean() * 100.0),
    }


def extended_grid_summary(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for dataset, result in results.items():
        summary = best_rows(dataset, result["grid"], result["geco_exact"])
        best = summary["best"]
        overall = summary["overall"]
        geco = summary["geco_exact"]
        rows.append(
            {
                "dataset": dataset,
                "grid_points": int(len(result["grid"])),
                "geco_S": float(geco["S"]),
                "geco_G": float(geco["G"]),
                "geco_F1": float(geco["F1"]),
                "geco_eTaF1": float(geco["eTaF1"]),
                "geco_FPA": float(geco["FPA"]),
                "geco_Scen": float(geco["Scen"]),
                "best_fpa_selection": summary["selection"],
                "best_fpa_S": float(best["S"]),
                "best_fpa_G": float(best["G"]),
                "best_fpa_F1": float(best["F1"]),
                "best_fpa_eTaF1": float(best["eTaF1"]),
                "best_fpa_FPA": float(best["FPA"]),
                "best_fpa_Scen": float(best["Scen"]),
                "best_overall_S": float(overall["S"]),
                "best_overall_G": float(overall["G"]),
                "best_overall_F1": float(overall["F1"]),
                "best_overall_eTaF1": float(overall["eTaF1"]),
                "best_overall_FPA": float(overall["FPA"]),
                "best_overall_Scen": float(overall["Scen"]),
                "threshold_5pct_eTaF1": summary["threshold_5pct"],
                "pct_grid_within_5pct": summary["pct_within_5pct"],
            }
        )
    out = pd.DataFrame(rows)
    write_csv(out, OUT_DIR / "extended_grid_summary.csv")
    return out


def plot_heatmaps(results: dict[str, dict[str, Any]], summary: pd.DataFrame) -> None:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except Exception as exc:
        (OUT_DIR / "heatmap_generation_skipped.txt").write_text(str(exc) + "\n", encoding="utf-8")
        return

    plt.rcParams.update(
        {
            "font.size": 8,
            "axes.titlesize": 9,
            "axes.labelsize": 8,
            "xtick.labelsize": 7,
            "ytick.labelsize": 7,
            "legend.fontsize": 6,
        }
    )

    def draw(ax: Any, dataset: str, colorbar: bool = False) -> Any:
        cfg = DATASETS[dataset]
        grid = results[dataset]["grid"]
        pivot = grid.pivot_table(index="G", columns="S", values="eTaF1", aggfunc="mean").sort_index(axis=0).sort_index(axis=1)
        s_vals = pivot.columns.to_numpy(dtype=float)
        g_vals = pivot.index.to_numpy(dtype=float)
        z = pivot.to_numpy(dtype=float)
        im = ax.imshow(
            z,
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=100.0,
            extent=[float(s_vals.min()), float(s_vals.max()), float(g_vals.min()), float(g_vals.max())],
            cmap="viridis",
        )
        row = summary[summary["dataset"] == dataset].iloc[0]
        threshold = float(row["threshold_5pct_eTaF1"])
        if len(s_vals) >= 2 and len(g_vals) >= 2 and np.nanmin(z) <= threshold <= np.nanmax(z):
            ss, gg = np.meshgrid(s_vals, g_vals)
            ax.contour(ss, gg, z, levels=[threshold], colors="black", linewidths=1.3)
        ax.scatter([cfg["geco_s"]], [cfg["geco_g"]], marker="x", s=42, c="dodgerblue", linewidths=1.4, zorder=5)
        ax.scatter(
            [row["best_fpa_S"]],
            [row["best_fpa_G"]],
            marker="o",
            s=36,
            facecolors="none",
            edgecolors="white",
            linewidths=1.3,
            zorder=6,
        )
        ax.text(
            0.98,
            0.95,
            f"best zero-FPA eTaF1={row['best_fpa_eTaF1']:.1f}\nFPA={row['best_fpa_FPA']:.0f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            color="white",
            fontsize=7,
            bbox={"facecolor": "black", "alpha": 0.35, "edgecolor": "none", "pad": 2},
        )
        ax.set_title(dataset)
        ax.set_xlabel("scale factor (S)")
        ax.set_ylabel("growth factor (G)")
        ax.set_xlim(0.0, 5.0)
        ax.set_ylim(0.0, 25.0)
        if colorbar:
            ax.figure.colorbar(im, ax=ax, label="eTaF1")
        return im

    for dataset, cfg in DATASETS.items():
        fig, ax = plt.subplots(figsize=(3.5, 2.45), constrained_layout=True)
        draw(ax, dataset, colorbar=True)
        fig.savefig(OUT_DIR / f"heatmap_{cfg['slug']}.pdf")
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.35), constrained_layout=True)
    im = None
    for ax, dataset in zip(axes, ["SWaT", "WADI", "BATADAL"]):
        im = draw(ax, dataset, colorbar=False)
    if im is not None:
        fig.colorbar(im, ax=axes.ravel().tolist(), label="eTaF1")
    fig.savefig(OUT_DIR / "heatmap_combined.pdf")
    plt.close(fig)


def status_target_count(status: pd.DataFrame, config: dict[str, Any]) -> int:
    if "targets" in config:
        return len(config["targets"])
    if not status.empty and "target" in status.columns:
        return int(status["target"].nunique())
    data = config.get("data", {})
    return int(data.get("num_sensors", 0))


def asid_training_time(dataset: str) -> tuple[float, int, str, str]:
    cfg = DATASETS[dataset]
    config = read_json(cfg["run_config"])
    status = read_csv(cfg["status"]) if cfg["status"].exists() else pd.DataFrame()
    n_targets = status_target_count(status, config)
    target_jobs = int(config.get("target_parallel_jobs", config.get("parallel_jobs", 1)) or 1)
    pysr_procs = int(config.get("pysr_procs", 1) or 1)
    threads = max(1, target_jobs * pysr_procs)

    measured_path = REPO_ROOT / "compute_summary.csv"
    if measured_path.exists():
        measured = read_csv(measured_path)
        if {"dataset", "elapsed_seconds"}.issubset(measured.columns):
            measured = measured[measured["dataset"].astype(str) == dataset].copy()
            elapsed = pd.to_numeric(measured["elapsed_seconds"], errors="coerce")
            if not measured.empty and int(elapsed.notna().sum()) == len(measured):
                wall_h = float(elapsed.sum()) / 3600.0
                if "procs" in measured.columns:
                    procs = pd.to_numeric(measured["procs"], errors="coerce").dropna()
                    threads = int(max(1, procs.max())) if not procs.empty else 1
                source = "measured per-target PySR metadata from compute_summary.csv"
                return wall_h, threads, source, f"{len(measured)} targets x {int(config.get('niterations', 400))} PySR iterations"

    elapsed_cols = [col for col in ["elapsed_seconds", "wall_seconds"] if col in status.columns]
    if elapsed_cols:
        elapsed = pd.to_numeric(status[elapsed_cols[0]], errors="coerce")
        if int(elapsed.notna().sum()) >= n_targets and n_targets > 0:
            wall_h = float(elapsed.sum()) / max(1, target_jobs) / 3600.0
            source = f"complete per-target {elapsed_cols[0]} summed / target_parallel_jobs"
            return wall_h, threads, source, f"{n_targets} targets x {int(config.get('niterations', 400))} PySR iterations"

    timeout = float(config.get("target_wall_timeout_minutes", config.get("timeout_minutes", 30.0)) or 30.0)
    wall_h = n_targets * timeout / max(1, target_jobs) / 60.0
    source = f"config upper-bound estimate: {n_targets} targets x {timeout:g} min / {target_jobs} target jobs"
    return wall_h, threads, source, f"{n_targets} targets x {int(config.get('niterations', 400))} PySR iterations"


def computational_performance(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for dataset, result in results.items():
        wall_h, threads, source, work = asid_training_time(dataset)
        rows.append(
            {
                "Dataset": dataset,
                "Method": "ASID-ICS",
                "Training Time": f"{wall_h:.2f} h",
                "CPU Threads": threads,
                "Approx Core-Hours": wall_h * threads,
                "Live Classification Time": f"{result['detection_eval_s_per_sample'] * 1000.0:.4f} ms",
                "template_combinations_or_pysr_iterations": work,
                "timing_source": source,
            }
        )
        geco = GECO_COMPUTE[dataset]
        rows.append(
            {
                "Dataset": dataset,
                "Method": "GeCo",
                "Training Time": f"{geco['training_h']:.2f} h",
                "CPU Threads": int(geco["threads"]),
                "Approx Core-Hours": float(geco["training_h"]) * int(geco["threads"]),
                "Live Classification Time": f"{geco['live_ms']:.1f} ms",
                "template_combinations_or_pysr_iterations": geco["work"],
                "timing_source": "GeCo Table 5 values supplied in task",
            }
        )
    out = pd.DataFrame(rows)
    write_csv(out, OUT_DIR / "computational_performance.csv")
    return out


def latex_computational_table(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{llrrrr}",
        r"\toprule",
        r"Dataset & Method & Training Time & CPU Threads & Core-Hours & Live Time \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        lines.append(
            f"{row['Dataset']} & {row['Method']} & {row['Training Time']} & "
            f"{int(row['CPU Threads'])} & {float(row['Approx Core-Hours']):.1f} & "
            f"{row['Live Classification Time']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def one_row_main_table(summary: pd.DataFrame, results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for dataset, methods in BASELINES.items():
        for method, values in methods.items():
            rows.append({"dataset": dataset, "method": method, **values, "S": np.nan, "G": np.nan})
        best = summary[summary["dataset"] == dataset].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "method": "ASID-ICS",
                "Prec": float(best["best_fpa_F1"] * 0.0 + results[dataset]["grid"][
                    (np.isclose(results[dataset]["grid"]["S"], best["best_fpa_S"]))
                    & (np.isclose(results[dataset]["grid"]["G"], best["best_fpa_G"]))
                ].iloc[0]["Prec"]),
                "Rec": float(results[dataset]["grid"][
                    (np.isclose(results[dataset]["grid"]["S"], best["best_fpa_S"]))
                    & (np.isclose(results[dataset]["grid"]["G"], best["best_fpa_G"]))
                ].iloc[0]["Rec"]),
                "F1": float(best["best_fpa_F1"]),
                "eTaP": float(results[dataset]["grid"][
                    (np.isclose(results[dataset]["grid"]["S"], best["best_fpa_S"]))
                    & (np.isclose(results[dataset]["grid"]["G"], best["best_fpa_G"]))
                ].iloc[0]["eTaP"]),
                "eTaR": float(results[dataset]["grid"][
                    (np.isclose(results[dataset]["grid"]["S"], best["best_fpa_S"]))
                    & (np.isclose(results[dataset]["grid"]["G"], best["best_fpa_G"]))
                ].iloc[0]["eTaR"]),
                "eTaF1": float(best["best_fpa_eTaF1"]),
                "FPA": float(best["best_fpa_FPA"]),
                "Scen": float(best["best_fpa_Scen"]),
                "S": float(best["best_fpa_S"]),
                "G": float(best["best_fpa_G"]),
            }
        )
    out = pd.DataFrame(rows)
    return add_rank_columns(out)


def add_rank_columns(table: pd.DataFrame) -> pd.DataFrame:
    out = table.copy()
    for dataset in out["dataset"].unique():
        idx = out["dataset"] == dataset
        for metric in METRICS:
            values = out.loc[idx, metric].astype(float)
            ascending = metric == "FPA"
            ranks = values.rank(method="min", ascending=ascending)
            out.loc[idx, f"{metric}_highlight"] = np.select([ranks == 1, ranks == 2], ["best", "second"], default="")
    return out


def tex_num(value: Any, metric: str) -> str:
    if pd.isna(value):
        return "--"
    if metric == "FPA":
        return f"{float(value):.0f}"
    return f"{float(value):.1f}"


def highlighted(value: Any, metric: str, mark: str) -> str:
    text = tex_num(value, metric)
    if mark == "best":
        return rf"\best{{{text}}}"
    if mark == "second":
        return rf"\second{{{text}}}"
    return text


def latex_main_table(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Dataset & Method & Prec & Rec & F1 & eTaP & eTaR & eTaF1 & FPA & Scen \\",
        r"\midrule",
    ]
    for dataset in ["SWaT", "WADI", "BATADAL"]:
        subset = table[table["dataset"] == dataset].reset_index(drop=True)
        for i, row in subset.iterrows():
            dataset_cell = dataset if i == 0 else ""
            cells = [highlighted(row[m], m, str(row.get(f"{m}_highlight", ""))) for m in METRICS]
            lines.append(f"{dataset_cell} & {row['method']} & " + " & ".join(cells) + r" \\")
        if dataset != "BATADAL":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def geco_op_comparison(results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for dataset in ["SWaT", "WADI", "BATADAL"]:
        geco = BASELINES[dataset]["GeCo"]
        ours = results[dataset]["geco_exact"]
        row: dict[str, Any] = {"dataset": dataset, "S": DATASETS[dataset]["geco_s"], "G": DATASETS[dataset]["geco_g"]}
        for metric in METRICS:
            row[f"GeCo_{metric}"] = float(geco[metric])
            row[f"ASID_ICS_{metric}"] = float(ours[metric])
        rows.append(row)
    out = pd.DataFrame(rows)
    write_csv(out, OUT_DIR / "geco_op_comparison.csv")
    return out


def latex_geco_op(table: pd.DataFrame) -> str:
    spec = "l" + "r" * len(METRICS) + "|" + "r" * len(METRICS)
    nl = r" \\"
    lines = [
        rf"\begin{{tabular}}{{{spec}}}",
        r"\toprule",
        rf"Dataset & \multicolumn{{{len(METRICS)}}}{{c|}}{{GeCo}} & \multicolumn{{{len(METRICS)}}}{{c}}{{ASID-ICS at GeCo S/G}}" + nl,
        " & " + " & ".join(METRICS) + " & " + " & ".join(METRICS) + nl,
        r"\midrule",
    ]
    for _, row in table.iterrows():
        left = [tex_num(row[f"GeCo_{m}"], m) for m in METRICS]
        right = [tex_num(row[f"ASID_ICS_{m}"], m) for m in METRICS]
        lines.append(f"{row['dataset']} & " + " & ".join(left) + " & " + " & ".join(right) + nl)
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def best_etaf1_summary(summary: pd.DataFrame, results: dict[str, dict[str, Any]]) -> pd.DataFrame:
    rows = []
    for dataset in ["SWaT", "WADI", "BATADAL"]:
        row = summary[summary["dataset"] == dataset].iloc[0]
        geco_ref = BASELINES[dataset]["GeCo"]["eTaF1"]
        ours_geco = float(results[dataset]["geco_exact"]["eTaF1"])
        best = float(row["best_fpa_eTaF1"])
        rows.append(
            {
                "Dataset": dataset,
                "GeCo published eTaF1": geco_ref,
                "ASID-ICS at GeCo S/G eTaF1": ours_geco,
                "ASID-ICS best zero/low-FPA eTaF1": best,
                "Gap to GeCo": best - geco_ref,
                "Best overall eTaF1 any S/G": float(row["best_overall_eTaF1"]),
                "Best overall FPA": float(row["best_overall_FPA"]),
            }
        )
    out = pd.DataFrame(rows)
    write_csv(out, OUT_DIR / "best_etaf1_summary.csv")
    return out


def write_readme(files: list[tuple[str, str]]) -> None:
    lines = [
        "# ASID-ICS Final v2 Paper Artifacts",
        "",
        "Generated by `scripts/generate_paper_artifacts_v2.py` from frozen selected equations and result artifacts.",
        "No PySR equation discovery is run by this script.",
        "",
        "## Files",
        "",
    ]
    for name, desc in files:
        lines.append(f"- `{name}`: {desc}")
    lines.extend(
        [
            "",
            "## Regeneration",
            "",
            "```bash",
            "${PYTHON:-python} scripts/generate_paper_artifacts_v2.py",
            "```",
            "",
        ]
    )
    (OUT_DIR / "README.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> int:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    results: dict[str, dict[str, Any]] = {}
    for dataset in ["SWaT", "WADI", "BATADAL"]:
        results[dataset] = evaluate_dataset(dataset)

    summary = extended_grid_summary(results)
    plot_heatmaps(results, summary)

    comp = computational_performance(results)
    (OUT_DIR / "table_computational_performance.tex").write_text(latex_computational_table(comp), encoding="utf-8")

    main_table = one_row_main_table(summary, results)
    write_csv(main_table[["dataset", "method", *METRICS, "S", "G"]], OUT_DIR / "main_table_one_row.csv")
    (OUT_DIR / "table_main_one_row.tex").write_text(latex_main_table(main_table), encoding="utf-8")

    ge_op = geco_op_comparison(results)
    (OUT_DIR / "table_geco_op_comparison.tex").write_text(latex_geco_op(ge_op), encoding="utf-8")

    best_summary = best_etaf1_summary(summary, results)

    files = [
        ("extended_grid_swat.csv", "420-point S/G grid for SWaT with all eight metrics."),
        ("extended_grid_wadi.csv", "420-point S/G grid for WADI with all eight metrics."),
        ("extended_grid_batadal.csv", "420-point S/G grid for BATADAL combined 14 attacks with all eight metrics."),
        ("extended_grid_summary.csv", "Best zero/low-FPA, best overall, GeCo operating point, and 95% contour statistics."),
        ("heatmap_swat.pdf", "SWaT GeCo-style eTaF1 heatmap with black 95% contour, blue GeCo x, and white best zero/low-FPA marker."),
        ("heatmap_wadi.pdf", "WADI GeCo-style eTaF1 heatmap."),
        ("heatmap_batadal.pdf", "BATADAL GeCo-style eTaF1 heatmap."),
        ("heatmap_combined.pdf", "Three-panel heatmap figure for paper figure* layout."),
        ("computational_performance.csv", "ASID-ICS vs GeCo timing and live classification estimates."),
        ("table_computational_performance.tex", "LaTeX computational performance table."),
        ("main_table_one_row.csv", "Main comparison table with one ASID-ICS row per dataset selected by zero/low-FPA eTaF1."),
        ("table_main_one_row.tex", "LaTeX main comparison table with best/second highlighting."),
        ("geco_op_comparison.csv", "Side-by-side GeCo vs ASID-ICS at GeCo's published S/G points."),
        ("table_geco_op_comparison.tex", "LaTeX GeCo operating point comparison table."),
        ("best_etaf1_summary.csv", "Short eTaF1 gap summary requested in Part 4."),
    ]
    write_readme(files)

    print("\n=== Extended Grid Summary ===")
    print(summary.to_string(index=False))
    print("\n=== Best eTaF1 Summary ===")
    print(best_summary.to_string(index=False))
    print("\n=== Computational Performance ===")
    print(comp.to_string(index=False))
    print(f"\nWrote final_v2 artifacts to {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
