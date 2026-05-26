#!/usr/bin/env python
from __future__ import annotations

import argparse
import importlib.util
import json
import math
import os
import sys
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))


OUT_DIR = REPO_ROOT / "paper_artifacts" / "final"
METRICS = ["Prec", "Rec", "F1", "eTaP", "eTaR", "eTaF1", "FPA", "Scen"]
HIGHER_IS_BETTER = {"Prec", "Rec", "F1", "eTaP", "eTaR", "eTaF1", "Scen"}
LOWER_IS_BETTER = {"FPA"}

DATASETS = {
    "SWaT": {
        "slug": "swat",
        "geco_s": 1.42,
        "geco_g": 5.98,
        "grid": REPO_ROOT / "artifacts/swat_1sec/actuator_subset_ablation/grid_geco_matched_plus_all_actuator_persistence.csv",
        "selected": REPO_ROOT / "artifacts/swat_1sec/delta_full/selected_equations.csv",
        "run_config": REPO_ROOT / "artifacts/swat_1sec/delta_full/run_config.json",
        "status": REPO_ROOT / "artifacts/swat_1sec/delta_full/pysr_run_status.csv",
        "variant_in": "geco_matched_plus_all_actuator_persistence",
        "variant_out": "geco_matched_plus_actuator_persistence",
    },
    "WADI": {
        "slug": "wadi",
        "geco_s": 1.32,
        "geco_g": 9.74,
        "grid": REPO_ROOT / "artifacts/wadi_1sec/delta_posthoc_ablation/grid_geco_matched_plus_actuator_persistence.csv",
        "selected": REPO_ROOT / "artifacts/wadi_1sec/delta_full/selected_equations.csv",
        "run_config": REPO_ROOT / "artifacts/wadi_1sec/delta_full/run_config.json",
        "status": REPO_ROOT / "artifacts/wadi_1sec/delta_full/pysr_run_status.csv",
        "variant_in": "geco_matched_plus_actuator_persistence",
        "variant_out": "geco_matched_plus_actuator_persistence",
    },
    "BATADAL": {
        "slug": "batadal",
        "geco_s": 1.39,
        "geco_g": 2.16,
        "grid": REPO_ROOT / "artifacts/batadal/full_14_attack_eval/grid_combined_14_attacks.csv",
        "selected": REPO_ROOT / "artifacts/batadal/delta_full/selected_equations.csv",
        "run_config": REPO_ROOT / "artifacts/batadal/delta_full/run_config.json",
        "status": REPO_ROOT / "artifacts/batadal/delta_full/pysr_run_status.csv",
        "variant_in": "geco_matched_plus_actuator_persistence",
        "variant_out": "geco_matched_plus_actuator_persistence",
    },
}

BASELINES = {
    "SWaT": {
        "GeCo": {"Prec": 94.8, "Rec": 79.0, "F1": 86.2, "eTaP": 83.1, "eTaR": 60.7, "eTaF1": 70.2, "FPA": 4, "Scen": 86.1},
        "SIMPLE": {"Prec": 70.7, "Rec": 86.7, "F1": 77.9, "eTaP": 58.7, "eTaR": 47.2, "eTaF1": 52.3, "FPA": 18, "Scen": 75.0},
        "TABOR": {"Prec": 81.5, "Rec": 74.7, "F1": 77.9, "eTaP": 49.1, "eTaR": 18.9, "eTaF1": 27.3, "FPA": 27, "Scen": 55.6},
        "Invariant": {"Prec": 97.3, "Rec": 69.1, "F1": 80.8, "eTaP": 54.7, "eTaR": 29.8, "eTaF1": 38.6, "FPA": 182, "Scen": 86.1},
        "Seq2SeqNN": {"Prec": 44.0, "Rec": 10.9, "F1": 17.5, "eTaP": 42.8, "eTaR": 47.2, "eTaF1": 44.9, "FPA": 36, "Scen": 75.0},
        "PASAD": {"Prec": 32.4, "Rec": 71.5, "F1": 44.6, "eTaP": 16.0, "eTaR": 4.9, "eTaF1": 7.5, "FPA": 14, "Scen": 44.4},
    },
    "WADI": {
        "GeCo": {"Prec": 92.6, "Rec": 32.1, "F1": 47.7, "eTaP": 91.3, "eTaR": 56.3, "eTaF1": 69.7, "FPA": 0, "Scen": 78.6},
        "SIMPLE": {"Prec": 58.2, "Rec": 43.6, "F1": 49.8, "eTaP": 57.0, "eTaR": 52.1, "eTaF1": 54.4, "FPA": 8, "Scen": 64.3},
        "TABOR": {"Prec": 19.1, "Rec": 43.7, "F1": 26.6, "eTaP": 14.9, "eTaR": 13.0, "eTaF1": 13.9, "FPA": 3, "Scen": 57.1},
        "Invariant": {"Prec": 90.0, "Rec": 21.9, "F1": 35.2, "eTaP": 92.3, "eTaR": 32.6, "eTaF1": 48.1, "FPA": 2, "Scen": 42.9},
        "Seq2SeqNN": {"Prec": 44.4, "Rec": 13.4, "F1": 20.5, "eTaP": 45.4, "eTaR": 31.3, "eTaF1": 37.1, "FPA": 7, "Scen": 64.3},
        "PASAD": {"Prec": 16.4, "Rec": 23.9, "F1": 19.5, "eTaP": 5.4, "eTaR": 4.3, "eTaF1": 4.8, "FPA": 3, "Scen": 35.7},
    },
    "BATADAL": {
        "GeCo": {"Prec": 93.8, "Rec": 73.4, "F1": 82.3, "eTaP": 97.0, "eTaR": 88.1, "eTaF1": 92.4, "FPA": 0, "Scen": 100.0},
        "SIMPLE": {"Prec": 52.0, "Rec": 43.3, "F1": 47.2, "eTaP": 49.0, "eTaR": 42.8, "eTaF1": 45.7, "FPA": 4, "Scen": 71.4},
        "TABOR": {"Prec": 78.5, "Rec": 6.9, "F1": 12.7, "eTaP": 77.7, "eTaR": 14.3, "eTaF1": 24.1, "FPA": 2, "Scen": 14.3},
        "Invariant": {"Prec": 27.2, "Rec": 45.5, "F1": 34.0, "eTaP": 18.2, "eTaR": 74.9, "eTaF1": 29.3, "FPA": 865, "Scen": 100.0},
        "Seq2SeqNN": {"Prec": 34.2, "Rec": 5.6, "F1": 9.6, "eTaP": 27.0, "eTaR": 6.9, "eTaF1": 11.0, "FPA": 1, "Scen": 14.3},
        "PASAD": {"Prec": 20.1, "Rec": 52.1, "F1": 29.1, "eTaP": 10.5, "eTaR": 21.5, "eTaF1": 14.1, "FPA": 32, "Scen": 78.6},
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


def require_file(path: Path) -> Path:
    if not path.exists():
        raise FileNotFoundError(path)
    return path


def read_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(require_file(path))


def write_csv(df: pd.DataFrame, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def metric_value(row: pd.Series | dict[str, Any], metric: str) -> float:
    aliases = {
        "Prec": ["Prec", "Precision"],
        "Rec": ["Rec", "Recall"],
    }
    for key in aliases.get(metric, [metric]):
        if key in row:
            return float(row[key])
    return float("nan")


def normalized_metric_row(row: pd.Series | dict[str, Any]) -> dict[str, float]:
    return {metric: metric_value(row, metric) for metric in METRICS}


def variant_grid(dataset: str) -> pd.DataFrame:
    cfg = DATASETS[dataset]
    df = read_csv(cfg["grid"])
    if "variant" in df.columns:
        df = df[df["variant"].astype(str) == cfg["variant_in"]].copy()
    if "dataset" in df.columns and dataset == "BATADAL":
        df = df[df["dataset"].astype(str) == "combined_14_attacks"].copy()
    if df.empty:
        raise ValueError(f"No grid rows for {dataset} variant {cfg['variant_in']}")
    return df.reset_index(drop=True)


def exact_or_nearest_rows(dataset: str, grid: pd.DataFrame, s: float, g: float) -> list[tuple[pd.Series, str]]:
    exact = grid[np.isclose(grid["S"], s) & np.isclose(grid["G"], g)]
    if not exact.empty:
        return [(exact.iloc[0], "exact")]
    if dataset == "SWaT":
        nearest_s = sorted(grid["S"].unique(), key=lambda value: abs(float(value) - s))[0]
        at_s = grid[np.isclose(grid["S"], nearest_s)].copy()
        nearest_g = sorted(at_s["G"].unique(), key=lambda value: abs(float(value) - g))[:2]
        return [
            (at_s[np.isclose(at_s["G"], gg)].iloc[0], f"nearest_grid_s={nearest_s};nearest_g={gg}")
            for gg in nearest_g
        ]
    scored = grid.assign(_dist=(grid["S"].astype(float) - s) ** 2 + (grid["G"].astype(float) - g) ** 2)
    row = scored.sort_values("_dist").iloc[0]
    return [(row, "nearest_grid_point")]


def primary_operating_points() -> pd.DataFrame:
    rows = []
    for dataset, cfg in DATASETS.items():
        grid = variant_grid(dataset)
        for row, note in exact_or_nearest_rows(dataset, grid, cfg["geco_s"], cfg["geco_g"]):
            metric_row = normalized_metric_row(row)
            rows.append(
                {
                    "dataset": dataset,
                    "variant": cfg["variant_out"],
                    "requested_S": cfg["geco_s"],
                    "requested_G": cfg["geco_g"],
                    "S": float(row["S"]),
                    "G": float(row["G"]),
                    **metric_row,
                    "num_monitored": int(row.get("num_monitored", np.nan)),
                    "num_sensors": int(row.get("monitored_sensors", np.nan)),
                    "num_actuators": int(row.get("monitored_actuators", np.nan)),
                    "grid_point_note": note,
                }
            )
    return pd.DataFrame(rows)


def best_zero_or_low_fpa_rows() -> pd.DataFrame:
    rows = []
    for dataset, cfg in DATASETS.items():
        grid = variant_grid(dataset)
        zero = grid[np.isclose(grid["FPA"], 0.0)].copy()
        if zero.empty:
            candidates = grid[grid["FPA"] <= 5].copy()
            selection = "ASID-ICS best low-FPA"
            if candidates.empty:
                candidates = grid.copy()
                selection = "ASID-ICS best eTaF1"
        else:
            candidates = zero
            selection = "ASID-ICS best zero-FPA"
        row = candidates.sort_values(["eTaF1", "F1", "Scen"], ascending=False).iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "method": selection,
                "variant": cfg["variant_out"],
                "S": float(row["S"]),
                "G": float(row["G"]),
                **normalized_metric_row(row),
                "num_monitored": int(row.get("num_monitored", np.nan)),
                "num_sensors": int(row.get("monitored_sensors", np.nan)),
                "num_actuators": int(row.get("monitored_actuators", np.nan)),
            }
        )
    return pd.DataFrame(rows)


def comparison_table(primary: pd.DataFrame, best: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for dataset, methods in BASELINES.items():
        for method, values in methods.items():
            rows.append({"dataset": dataset, "method": method, "row_type": "published_baseline", **values})
        ge_op = primary[primary["dataset"] == dataset].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "method": "ASID-ICS (GeCo op)",
                "row_type": "ours_geco_operating_point",
                **normalized_metric_row(ge_op),
            }
        )
        best_row = best[best["dataset"] == dataset].iloc[0]
        rows.append(
            {
                "dataset": dataset,
                "method": str(best_row["method"]).replace("ASID-ICS ", "ASID-ICS (") + ")",
                "row_type": "ours_best_zero_or_low_fpa",
                **normalized_metric_row(best_row),
            }
        )
    table = pd.DataFrame(rows)
    for dataset in table["dataset"].unique():
        idx = table["dataset"] == dataset
        for metric in METRICS:
            values = table.loc[idx, metric].astype(float)
            ascending = metric in LOWER_IS_BETTER
            ranks = values.rank(method="min", ascending=ascending)
            table.loc[idx, f"{metric}_rank"] = ranks.astype(int)
            table.loc[idx, f"{metric}_highlight"] = np.select(
                [ranks == 1, ranks == 2],
                ["best", "second"],
                default="",
            )
    return table


def tex_num(value: Any, metric: str) -> str:
    if pd.isna(value):
        return "--"
    if metric == "FPA":
        return f"{float(value):.0f}"
    return f"{float(value):.1f}"


def highlighted(value: Any, metric: str, highlight: str) -> str:
    text = tex_num(value, metric)
    if highlight == "best":
        return rf"\best{{{text}}}"
    if highlight == "second":
        return rf"\second{{{text}}}"
    return text


def latex_detection_table(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{llrrrrrrrr}",
        r"\toprule",
        r"Dataset & Method & Prec & Rec & F1 & eTaP & eTaR & eTaF1 & FPA & Scen \\",
        r"\midrule",
    ]
    for dataset in ["SWaT", "WADI", "BATADAL"]:
        ds = table[table["dataset"] == dataset].reset_index(drop=True)
        for i, row in ds.iterrows():
            dataset_cell = dataset if i == 0 else ""
            metric_cells = [
                highlighted(row[metric], metric, str(row.get(f"{metric}_highlight", "")))
                for metric in METRICS
            ]
            lines.append(f"{dataset_cell} & {row['method']} & " + " & ".join(metric_cells) + r" \\")
        if dataset != "BATADAL":
            lines.append(r"\midrule")
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def normalize_per_attack(df: pd.DataFrame, dataset: str) -> pd.DataFrame:
    out = df.copy()
    if "detection_category" in out.columns and "category" not in out.columns:
        out["category"] = out["detection_category"]
    if "firing_sensors" in out.columns and "firing_variables" not in out.columns:
        out["firing_variables"] = out["firing_sensors"]
    keep = ["attack_id", "affected_tags", "category", "firing_variables"]
    for col in keep:
        if col not in out.columns:
            out[col] = ""
    out = out[keep].copy()
    out.insert(0, "dataset", dataset)
    return out


def per_attack_swat() -> pd.DataFrame:
    swat_act = import_module(
        "paper_swat_actuator_subset",
        REPO_ROOT / "scripts/run_swat_1sec_actuator_subset_ablation.py",
    )
    args = argparse.Namespace(
        experiment="configs/experiment/swat_mlp_current_val20.yaml",
        train_csv=None,
        test_csv=None,
        delta_full="artifacts/swat_1sec/delta_full",
        posthoc="artifacts/swat_1sec/delta_posthoc_ablation",
        out="artifacts/swat_1sec/actuator_subset_ablation",
    )
    arrays = swat_act.POST.load_arrays(args)
    selected = REPO_ROOT / "artifacts/swat_1sec/delta_full/selected_equations.csv"
    models = swat_act.selected_sensor_models(selected, exclude=swat_act.GECO_EXCLUSIONS)
    models += [
        swat_act.actuator_model(name)
        for name in swat_act.actuator_names(arrays["feature_columns"])
        if name not in swat_act.GECO_EXCLUSIONS
    ]
    cache = swat_act.POST.residual_cache_for_models(arrays, models)
    alarms = swat_act.POST.alarm_map_for_choice(models, cache, arrays["labels"], s=1.42, g=5.98)
    return normalize_per_attack(swat_act.POST.per_attack_table(models, alarms), "SWaT")


def per_attack_wadi() -> pd.DataFrame:
    path = REPO_ROOT / "artifacts/wadi_1sec/hyperparameter_eval/per_attack_geco_op.csv"
    if path.exists():
        return normalize_per_attack(read_csv(path), "WADI")
    path = REPO_ROOT / "artifacts/wadi_1sec/delta_posthoc_ablation/per_attack_geco_matched_plus_actuator_persistence.csv"
    return normalize_per_attack(read_csv(path), "WADI")


def per_attack_batadal() -> pd.DataFrame:
    path = REPO_ROOT / "artifacts/batadal/full_14_attack_eval/per_attack_standard_operating_points.csv"
    df = read_csv(path)
    df = df[
        (df["variant"].astype(str) == "geco_matched_plus_actuator_persistence")
        & np.isclose(df["S"], 1.39)
        & np.isclose(df["G"], 2.16)
    ].copy()
    if "global_start" in df.columns:
        df = df.sort_values(["global_start", "global_end"])
    else:
        df = df.sort_values(["dataset", "attack_id"])
    df["source_attack_id"] = df["attack_id"]
    df["attack_id"] = np.arange(1, len(df) + 1)
    return normalize_per_attack(df, "BATADAL")


def heatmap_outputs() -> pd.DataFrame:
    summary_rows = []
    for dataset, cfg in DATASETS.items():
        grid = variant_grid(dataset)
        heat = grid[["S", "G", "F1", "eTaF1", "FPA", "Scen"]].copy()
        heat = heat.sort_values(["S", "G"]).reset_index(drop=True)
        write_csv(heat, OUT_DIR / f"heatmap_{cfg['slug']}.csv")
        zero = heat[np.isclose(heat["FPA"], 0.0)]
        if zero.empty:
            candidates = heat[heat["FPA"] <= 5]
            selection = "low_fpa_le_5" if not candidates.empty else "all_grid"
            if candidates.empty:
                candidates = heat
        else:
            candidates = zero
            selection = "zero_fpa"
        best = candidates.sort_values(["eTaF1", "F1", "Scen"], ascending=False).iloc[0]
        ge = exact_or_nearest_rows(dataset, grid, cfg["geco_s"], cfg["geco_g"])[0][0]
        max_ref = float(best["eTaF1"])
        pct = float((heat["eTaF1"] >= 0.95 * max_ref).mean() * 100.0) if max_ref > 0 else 0.0
        summary_rows.append(
            {
                "dataset": dataset,
                "selection_basis": selection,
                "max_etaf1_at_fpa0_or_low_fpa": max_ref,
                "best_S": float(best["S"]),
                "best_G": float(best["G"]),
                "best_FPA": float(best["FPA"]),
                "geco_published_S": cfg["geco_s"],
                "geco_published_G": cfg["geco_g"],
                "geco_grid_S": float(ge["S"]),
                "geco_grid_G": float(ge["G"]),
                "geco_published_etaf1": float(ge["eTaF1"]),
                "pct_grid_within_5pct": pct,
            }
        )
    return pd.DataFrame(summary_rows)


def geco_style_hyperparameter_search() -> pd.DataFrame:
    rows = []
    for dataset, cfg in DATASETS.items():
        grid = variant_grid(dataset).sort_values(["S", "G"]).reset_index(drop=True)
        ge = exact_or_nearest_rows(dataset, grid, cfg["geco_s"], cfg["geco_g"])[0][0]
        overall = grid.sort_values(["eTaF1", "F1", "Scen"], ascending=False).iloc[0]
        zero = grid[np.isclose(grid["FPA"], 0.0)].copy()
        if zero.empty:
            candidates = grid[grid["FPA"] <= 5].copy()
            selection = "best_low_fpa_le_5" if not candidates.empty else "best_overall_no_fpa_constraint"
            if candidates.empty:
                candidates = grid.copy()
        else:
            candidates = zero
            selection = "best_zero_fpa"
        best = candidates.sort_values(["eTaF1", "F1", "Scen"], ascending=False).iloc[0]
        threshold = 0.95 * float(overall["eTaF1"])
        rows.append(
            {
                "dataset": dataset,
                "variant": cfg["variant_out"],
                "geco_S": float(ge["S"]),
                "geco_G": float(ge["G"]),
                "geco_F1": float(ge["F1"]),
                "geco_eTaF1": float(ge["eTaF1"]),
                "geco_FPA": float(ge["FPA"]),
                "geco_Scen": float(ge["Scen"]),
                "max_overall_S": float(overall["S"]),
                "max_overall_G": float(overall["G"]),
                "max_overall_F1": float(overall["F1"]),
                "max_overall_eTaF1": float(overall["eTaF1"]),
                "max_overall_FPA": float(overall["FPA"]),
                "max_overall_Scen": float(overall["Scen"]),
                "best_fpa_selection": selection,
                "best_fpa_S": float(best["S"]),
                "best_fpa_G": float(best["G"]),
                "best_fpa_F1": float(best["F1"]),
                "best_fpa_eTaF1": float(best["eTaF1"]),
                "best_fpa_FPA": float(best["FPA"]),
                "best_fpa_Scen": float(best["Scen"]),
                "within_5pct_threshold_eTaF1": threshold,
                "grid_points": int(len(grid)),
                "grid_points_within_5pct_of_overall_max": int((grid["eTaF1"] >= threshold).sum()),
                "pct_grid_within_5pct_of_overall_max": float((grid["eTaF1"] >= threshold).mean() * 100.0),
            }
        )
    return pd.DataFrame(rows)


def plot_geco_style_heatmaps(search: pd.DataFrame) -> list[tuple[str, str]]:
    try:
        import matplotlib

        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from matplotlib.lines import Line2D
    except Exception as exc:
        note = OUT_DIR / "figure_generation_skipped.txt"
        note.write_text(f"Matplotlib unavailable; skipped heatmap figures: {exc}\n", encoding="utf-8")
        return [("figure_generation_skipped.txt", "Notes why GeCo-style heatmap figures were not generated.")]

    figure_files: list[tuple[str, str]] = []
    fig_dir = OUT_DIR / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    def draw_dataset(ax: Any, dataset: str) -> Any:
        cfg = DATASETS[dataset]
        grid = variant_grid(dataset)
        pivot = (
            grid.pivot_table(index="G", columns="S", values="eTaF1", aggfunc="mean")
            .sort_index(axis=0)
            .sort_index(axis=1)
        )
        s_vals = pivot.columns.to_numpy(dtype=float)
        g_vals = pivot.index.to_numpy(dtype=float)
        z = pivot.to_numpy(dtype=float) / 100.0
        im = ax.imshow(
            z,
            origin="lower",
            aspect="auto",
            vmin=0.0,
            vmax=1.0,
            extent=[float(s_vals.min()), float(s_vals.max()), float(g_vals.min()), float(g_vals.max())],
            cmap="viridis",
        )
        row = search[search["dataset"] == dataset].iloc[0]
        threshold = float(row["within_5pct_threshold_eTaF1"]) / 100.0
        if len(s_vals) >= 2 and len(g_vals) >= 2 and np.nanmin(z) <= threshold <= np.nanmax(z):
            ss, gg = np.meshgrid(s_vals, g_vals)
            ax.contour(ss, gg, z, levels=[threshold], colors="black", linewidths=2.0)
        ax.scatter([cfg["geco_s"]], [cfg["geco_g"]], marker="x", s=90, c="dodgerblue", linewidths=2.4, zorder=5)
        ax.scatter([row["best_fpa_S"]], [row["best_fpa_G"]], marker="o", s=46, c="black", edgecolors="white", linewidths=0.8, zorder=6)
        ax.set_xlim(left=0.0, right=max(float(s_vals.max()), float(cfg["geco_s"]), float(row["best_fpa_S"])))
        ax.set_ylim(bottom=0.0, top=max(float(g_vals.max()), float(cfg["geco_g"]), float(row["best_fpa_G"])))
        ax.set_title(dataset)
        ax.set_xlabel("scale factor (S)")
        ax.set_ylabel("growth factor (G)")
        ax.legend(
            handles=[
                Line2D([0], [0], marker="x", color="dodgerblue", linestyle="None", markersize=7, markeredgewidth=2.0, label="GeCo S/G"),
                Line2D([0], [0], marker="o", color="black", linestyle="None", markersize=5, label="best zero/low-FPA"),
                Line2D([0], [0], color="black", linewidth=2.0, label="within 5% max eTaF1"),
            ],
            loc="lower right",
            fontsize=7,
            framealpha=0.72,
        )
        ax.text(
            0.98,
            0.95,
            f"best {row['best_fpa_selection']}\nFPA={row['best_fpa_FPA']:.0f}",
            transform=ax.transAxes,
            ha="right",
            va="top",
            fontsize=8,
            color="white",
            bbox={"facecolor": "black", "alpha": 0.35, "pad": 3, "edgecolor": "none"},
        )
        return im

    fig, axes = plt.subplots(1, 3, figsize=(13.5, 3.8), constrained_layout=True)
    image = None
    for ax, dataset in zip(axes, ["SWaT", "WADI", "BATADAL"]):
        image = draw_dataset(ax, dataset)
    fig.suptitle("ASID-ICS GeCo-style S/G Hyperparameter Search", fontsize=13)
    if image is not None:
        fig.colorbar(image, ax=axes.ravel().tolist(), label="eTaF1")
    combined_png = fig_dir / "figure_geco_style_hyperparameter_search.png"
    combined_pdf = fig_dir / "figure_geco_style_hyperparameter_search.pdf"
    fig.savefig(combined_png, dpi=220)
    fig.savefig(combined_pdf)
    plt.close(fig)
    figure_files.extend(
        [
            ("figures/figure_geco_style_hyperparameter_search.png", "Combined GeCo-style eTaF1 heatmap over S/G for all datasets."),
            ("figures/figure_geco_style_hyperparameter_search.pdf", "PDF version of the combined GeCo-style hyperparameter heatmap."),
        ]
    )

    for dataset, cfg in DATASETS.items():
        fig, ax = plt.subplots(1, 1, figsize=(5.0, 3.6), constrained_layout=True)
        image = draw_dataset(ax, dataset)
        fig.suptitle(f"{dataset} S/G Sensitivity", fontsize=12)
        fig.colorbar(image, ax=ax, label="eTaF1")
        png = fig_dir / f"figure_geco_style_hyperparameter_{cfg['slug']}.png"
        pdf = fig_dir / f"figure_geco_style_hyperparameter_{cfg['slug']}.pdf"
        fig.savefig(png, dpi=220)
        fig.savefig(pdf)
        plt.close(fig)
        figure_files.extend(
            [
                (f"figures/figure_geco_style_hyperparameter_{cfg['slug']}.png", f"{dataset} GeCo-style S/G heatmap PNG."),
                (f"figures/figure_geco_style_hyperparameter_{cfg['slug']}.pdf", f"{dataset} GeCo-style S/G heatmap PDF."),
            ]
        )
    return figure_files


def performance_table() -> pd.DataFrame:
    rows = [
        {
            "dataset": "SWaT",
            "method": "ASID-ICS",
            "wall_clock_hours": 25 * 12.0 / 60.0 / 8.0,
            "cpu_threads": 8,
            "cpu_core_hours": 25 * 12.0 / 60.0,
            "template_combinations_or_pysr_iterations": "25 targets x 400 PySR iterations",
            "timing_source": "prompt estimate: 25 sensors, 12 min avg, 8 parallel jobs",
        },
        {
            "dataset": "WADI",
            "method": "ASID-ICS",
            "wall_clock_hours": 85 * 75.0 / 60.0 / 2.0,
            "cpu_threads": 2,
            "cpu_core_hours": 85 * 75.0 / 60.0,
            "template_combinations_or_pysr_iterations": "85 targets x 400 PySR iterations; 75 min target wall cap",
            "timing_source": "config estimate upper bound: 85 sensors, 75 min, 2 target jobs",
        },
        {
            "dataset": "BATADAL",
            "method": "ASID-ICS",
            "wall_clock_hours": 31 * 30.0 / 60.0 / 4.0,
            "cpu_threads": 4,
            "cpu_core_hours": 31 * 30.0 / 60.0,
            "template_combinations_or_pysr_iterations": "31 targets x 400 PySR iterations; 30 min target cap",
            "timing_source": "config estimate upper bound: 31 sensors, 30 min, 4 target jobs",
        },
        {
            "dataset": "SWaT",
            "method": "GeCo",
            "wall_clock_hours": 15.1,
            "cpu_threads": 128,
            "cpu_core_hours": 15.1 * 128,
            "template_combinations_or_pysr_iterations": "1.77M template combinations",
            "timing_source": "GeCo published value supplied in task",
        },
        {
            "dataset": "WADI",
            "method": "GeCo",
            "wall_clock_hours": 46.9 * 24.0,
            "cpu_threads": 128,
            "cpu_core_hours": 46.9 * 24.0 * 128,
            "template_combinations_or_pysr_iterations": "not transcribed locally",
            "timing_source": "GeCo published value supplied in task",
        },
        {
            "dataset": "BATADAL",
            "method": "GeCo",
            "wall_clock_hours": np.nan,
            "cpu_threads": 128,
            "cpu_core_hours": np.nan,
            "template_combinations_or_pysr_iterations": "not available in local artifacts/task prompt",
            "timing_source": "missing local source; left blank rather than inventing",
        },
    ]
    return pd.DataFrame(rows)


def latex_performance(table: pd.DataFrame) -> str:
    lines = [
        r"\begin{tabular}{llrrrl}",
        r"\toprule",
        r"Dataset & Method & Wall-clock h & Threads & Core h & Search budget \\",
        r"\midrule",
    ]
    for _, row in table.iterrows():
        wall = "--" if pd.isna(row["wall_clock_hours"]) else f"{float(row['wall_clock_hours']):.2f}"
        core = "--" if pd.isna(row["cpu_core_hours"]) else f"{float(row['cpu_core_hours']):.1f}"
        lines.append(
            f"{row['dataset']} & {row['method']} & {wall} & {int(row['cpu_threads'])} & {core} & {row['template_combinations_or_pysr_iterations']} \\\\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def latex_escape(text: Any) -> str:
    out = str(text)
    replacements = {
        "\\": r"\textbackslash{}",
        "&": r"\&",
        "%": r"\%",
        "$": r"\$",
        "#": r"\#",
        "_": r"\_",
        "{": r"\{",
        "}": r"\}",
        "~": r"\textasciitilde{}",
        "^": r"\textasciicircum{}",
    }
    return "".join(replacements.get(ch, ch) for ch in out)


def selected_equations(dataset: str, path: Path) -> pd.DataFrame:
    df = read_csv(path)
    score_col = "holdout_r2" if "holdout_r2" in df.columns else "score"
    top = df.sort_values(score_col, ascending=False).head(5).copy()
    if dataset == "SWaT":
        physics = df[df["target"].isin(["LIT101", "LIT301"])].copy()
        top = pd.concat([top, physics], ignore_index=True).drop_duplicates(subset=["target"], keep="first")
    out = top.copy()
    out.insert(0, "dataset", dataset)
    rename = {"score": "PySR_score"}
    out = out.rename(columns=rename)
    keep = [
        col
        for col in ["dataset", "target", "equation", "sympy_format", "complexity", "holdout_r2", "PySR_score"]
        if col in out.columns
    ]
    return out[keep]


def physics_recovery_tex(swat_equations: pd.DataFrame) -> str:
    lit101 = swat_equations[swat_equations["target"] == "LIT101"]
    lit301 = swat_equations[swat_equations["target"] == "LIT301"]
    rows = []
    if not lit101.empty:
        rows.append(
            {
                "Target": "LIT101",
                "ASID-ICS": str(lit101.iloc[0]["equation"]),
                "GeCo reference": "0.9999*LIT101 + 0.192*FIT101 - 0.220*FIT201 + 0.056*P101 - 0.047",
                "Known physics": "delta_LIT101 = c * (FIT101 - FIT201)",
            }
        )
    if not lit301.empty:
        rows.append(
            {
                "Target": "LIT301",
                "ASID-ICS": str(lit301.iloc[0]["equation"]),
                "GeCo reference": "not transcribed locally",
                "Known physics": "delta_LIT301 = c * (inflow - outflow)",
            }
        )
    lines = [
        r"\begin{tabular}{llll}",
        r"\toprule",
        r"Target & ASID-ICS delta equation & GeCo reference & Known physics \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            " & ".join(
                [
                    latex_escape(row["Target"]),
                    latex_escape(row["ASID-ICS"]),
                    latex_escape(row["GeCo reference"]),
                    latex_escape(row["Known physics"]),
                ]
            )
            + r" \\"
        )
    lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    return "\n".join(lines)


def wadi_contamination_sensitivity() -> pd.DataFrame:
    rows = []
    ge = read_csv(REPO_ROOT / "artifacts/wadi_1sec/hyperparameter_eval/geco_operating_point.csv")
    clean = read_csv(REPO_ROOT / "artifacts/wadi_1sec/clean_input_redistill/geco_operating_point_clean_input.csv")
    mapping = {
        "geco_matched_plus_persistence_as_is": "as-is primary",
        "geco_matched_plus_persistence_clean": "drop contaminated target",
        "clean_input_geco_matched_plus_persistence": "clean-input re-distill",
    }
    for source, label in mapping.items():
        table = clean if source.startswith("clean_input") else ge
        row = table[
            (table.iloc[:, 0].astype(str) == source)
            & np.isclose(table["S"], 1.32)
            & np.isclose(table["G"], 9.74)
        ]
        if row.empty and "variant" in table.columns:
            row = table[
                (table["variant"].astype(str) == source)
                & np.isclose(table["S"], 1.32)
                & np.isclose(table["G"], 9.74)
            ]
        if row.empty:
            continue
        r = row.iloc[0]
        rows.append(
            {
                "condition": label,
                "S": float(r["S"]),
                "G": float(r["G"]),
                **normalized_metric_row(r),
                "note": {
                    "as-is primary": "2_LT_002_PV equation uses TOTAL_CONS_REQUIRED_FLOW",
                    "drop contaminated target": "2_LT_002_PV excluded entirely",
                    "clean-input re-distill": "2_LT_002_PV re-distilled without problematic inputs",
                }[label],
            }
        )
    return pd.DataFrame(rows)


def write_readme(files: list[tuple[str, str]]) -> None:
    lines = [
        "# Final ASID-ICS Paper Artifacts",
        "",
        "This directory is the single source of truth for ACSAC paper numbers generated from frozen experiment artifacts. No PySR fitting or equation discovery is performed by the generator.",
        "",
        "## Regeneration",
        "",
        "```bash",
        "bash paper_artifacts/final/reproduce.sh",
        "```",
        "",
        "Equivalent direct command:",
        "",
        "```bash",
        "python scripts/generate_paper_artifacts.py --out paper_artifacts/final",
        "```",
        "",
        "## Files",
        "",
    ]
    for name, desc in sorted(files):
        lines.append(f"- `{name}`: {desc}")
    (OUT_DIR / "README.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def write_reproduce() -> None:
    path = OUT_DIR / "reproduce.sh"
    path.write_text(
        "#!/usr/bin/env bash\n"
        "set -euo pipefail\n"
        "cd \"$(dirname \"$0\")/../..\"\n"
        "${PYTHON:-python} scripts/generate_paper_artifacts.py --out paper_artifacts/final\n",
        encoding="utf-8",
    )
    os.chmod(path, 0o755)


def main() -> int:
    global OUT_DIR
    parser = argparse.ArgumentParser(description="Generate final ASID-ICS paper artifacts from frozen experiment outputs.")
    parser.add_argument("--out", default=str(OUT_DIR), help="Output directory for final paper artifacts.")
    args = parser.parse_args()
    OUT_DIR = Path(args.out)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    files: list[tuple[str, str]] = []

    primary = primary_operating_points()
    write_csv(primary, OUT_DIR / "primary_operating_points.csv")
    files.append(("primary_operating_points.csv", "ASID-ICS at each dataset's published GeCo S/G operating point."))

    best = best_zero_or_low_fpa_rows()
    write_csv(best, OUT_DIR / "best_zero_fpa_operating_points.csv")
    files.append(("best_zero_fpa_operating_points.csv", "Best ASID-ICS zero-FPA row per dataset, or best low-FPA fallback."))

    asid_ops = pd.concat(
        [
            primary.assign(method="ASID-ICS at GeCo operating point"),
            best.assign(method=best["method"]),
        ],
        ignore_index=True,
        sort=False,
    )
    write_csv(asid_ops, OUT_DIR / "asid_ics_operating_points.csv")
    files.append(("asid_ics_operating_points.csv", "Both GeCo-operating-point and best zero-FPA ASID-ICS rows."))

    comp = comparison_table(primary, best)
    write_csv(comp, OUT_DIR / "full_comparison_table.csv")
    (OUT_DIR / "table_detection_comparison.tex").write_text(latex_detection_table(comp), encoding="utf-8")
    files.append(("full_comparison_table.csv", "GeCo Table 2 equivalent with ASID-ICS rows and rank/highlight columns."))
    files.append(("table_detection_comparison.tex", "LaTeX detection comparison table using \\best{} and \\second{} markers."))

    per_attack_tables = {
        "swat": per_attack_swat(),
        "wadi": per_attack_wadi(),
        "batadal": per_attack_batadal(),
    }
    for slug, table in per_attack_tables.items():
        write_csv(table, OUT_DIR / f"per_attack_{slug}.csv")
        files.append((f"per_attack_{slug}.csv", f"Per-attack ASID-ICS categories at the GeCo operating point for {slug.upper()}."))

    heat_summary = heatmap_outputs()
    write_csv(heat_summary, OUT_DIR / "heatmap_summary.csv")
    files.append(("heatmap_summary.csv", "Sensitivity-grid statistics for Figure 10-style analysis."))
    for cfg in DATASETS.values():
        files.append((f"heatmap_{cfg['slug']}.csv", "S/G sensitivity grid with F1, eTaF1, FPA, and Scen."))

    geco_search = geco_style_hyperparameter_search()
    write_csv(geco_search, OUT_DIR / "geco_style_hyperparameter_search.csv")
    files.append(("geco_style_hyperparameter_search.csv", "GeCo-style S/G search summary: GeCo point, global max, best zero/low-FPA point, and within-5% region."))
    files.extend(plot_geco_style_heatmaps(geco_search))

    perf = performance_table()
    write_csv(perf, OUT_DIR / "computational_performance.csv")
    (OUT_DIR / "table_computational_performance.tex").write_text(latex_performance(perf), encoding="utf-8")
    files.append(("computational_performance.csv", "ASID-ICS and GeCo computational performance summary."))
    files.append(("table_computational_performance.tex", "LaTeX computational performance table."))

    eq_tables = {}
    for dataset, cfg in DATASETS.items():
        eq = selected_equations(dataset, cfg["selected"])
        eq_tables[dataset] = eq
        write_csv(eq, OUT_DIR / f"selected_equations_{cfg['slug']}.csv")
        files.append((f"selected_equations_{cfg['slug']}.csv", f"Top selected symbolic equations for {dataset}."))
    (OUT_DIR / "table_physics_recovery.tex").write_text(physics_recovery_tex(eq_tables["SWaT"]), encoding="utf-8")
    files.append(("table_physics_recovery.tex", "SWaT physics-recovery equations and references."))

    wadi_sens = wadi_contamination_sensitivity()
    write_csv(wadi_sens, OUT_DIR / "wadi_contamination_sensitivity.csv")
    files.append(("wadi_contamination_sensitivity.csv", "WADI as-is, drop-contaminated, and clean-input sensitivity rows."))

    write_reproduce()
    files.append(("reproduce.sh", "Regenerates all final artifacts end-to-end."))
    write_readme(files)

    print("=== Primary operating points ===")
    print(primary[["dataset", "variant", "S", "G", *METRICS, "num_monitored"]].to_string(index=False))
    print("\n=== Best zero-FPA / low-FPA rows ===")
    print(best[["dataset", "method", "S", "G", *METRICS, "num_monitored"]].to_string(index=False))
    print("\n=== Heatmap summary ===")
    print(heat_summary.to_string(index=False))
    print("\n=== GeCo-style hyperparameter search ===")
    print(geco_search.to_string(index=False))
    print(f"\nSaved final paper artifacts to: {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
