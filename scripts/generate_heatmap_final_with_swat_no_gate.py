#!/usr/bin/env python
"""Generate final paper heatmaps with the SWaT no-gate dense grid."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("pdf")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ics_symbolic_distill.detection import compute_detection_metrics


OUT_DIR = REPO_ROOT / "paper_artifacts" / "final" / "figures"
PAPER_DIR = REPO_ROOT / "paper_artifacts" / "final"
SWAT_NOGATE_DIR = REPO_ROOT / "artifacts" / "swat_1sec" / "no_holdout_quality_gate_check"


DATASETS = {
    "SWaT": {
        "slug": "swat",
        "grid": PAPER_DIR / "heatmap_swat_no_holdout_quality_gate_dense.csv",
        "geco_s": 1.42,
        "geco_g": 5.98,
        "best_s": 1.2,
        "best_g": 15.0,
        "selection": "best low-FPA",
        "source": "SWaT no-holdout-quality-gate dense grid",
    },
    "WADI": {
        "slug": "wadi",
        "grid": REPO_ROOT / "paper_artifacts" / "final_v2" / "extended_grid_wadi.csv",
        "geco_s": 1.32,
        "geco_g": 9.74,
        "best_s": None,
        "best_g": None,
        "selection": "best zero-FPA",
        "source": "final_v2 extended WADI grid",
    },
    "BATADAL": {
        "slug": "batadal",
        "grid": REPO_ROOT / "paper_artifacts" / "final_v2" / "extended_grid_batadal.csv",
        "geco_s": 1.39,
        "geco_g": 2.16,
        "best_s": None,
        "best_g": None,
        "selection": "best zero-FPA",
        "source": "final_v2 extended BATADAL grid",
    },
}


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def metric_col(df: pd.DataFrame, *names: str) -> str:
    for name in names:
        if name in df.columns:
            return name
    raise KeyError(f"Missing any of columns: {names}")


def read_grid(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path)
    rename = {}
    for canonical, aliases in {
        "Precision": ("Precision", "Prec", "precision"),
        "Recall": ("Recall", "Rec", "recall"),
        "F1": ("F1", "f1"),
        "eTaF1": ("eTaF1", "etaf1", "eTa-F1"),
        "FPA": ("FPA", "fpa"),
        "Scen": ("Scen", "Scen.", "scenario", "scenario_coverage"),
    }.items():
        for alias in aliases:
            if alias in df.columns:
                rename[alias] = canonical
                break
    out = df.rename(columns=rename).copy()
    required = {"S", "G", "F1", "eTaF1", "FPA", "Scen"}
    missing = required.difference(out.columns)
    if missing:
        raise ValueError(f"{path} missing required columns {sorted(missing)}")
    for col in required:
        out[col] = pd.to_numeric(out[col], errors="coerce")
    return out.dropna(subset=["S", "G", "eTaF1", "FPA"]).reset_index(drop=True)


def choose_best_row(dataset: str, grid: pd.DataFrame, cfg: dict[str, Any]) -> pd.Series:
    if cfg.get("best_s") is not None and cfg.get("best_g") is not None:
        rows = grid[np.isclose(grid["S"], float(cfg["best_s"])) & np.isclose(grid["G"], float(cfg["best_g"]))]
        if rows.empty:
            raise ValueError(f"Configured best point missing for {dataset}: S={cfg['best_s']} G={cfg['best_g']}")
        return rows.iloc[0]
    zero = grid[np.isclose(grid["FPA"], 0.0)].copy()
    if zero.empty:
        low = grid[grid["FPA"] <= 5].copy()
        if low.empty:
            low = grid.copy()
        return low.sort_values(["eTaF1", "F1", "Scen"], ascending=False).iloc[0]
    return zero.sort_values(["eTaF1", "F1", "Scen"], ascending=False).iloc[0]


def grid_to_matrices(grid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    pivot_etaf1 = grid.pivot_table(index="G", columns="S", values="eTaF1", aggfunc="mean").sort_index(axis=0).sort_index(axis=1)
    pivot_fpa = grid.pivot_table(index="G", columns="S", values="FPA", aggfunc="mean").reindex_like(pivot_etaf1)
    s_vals = pivot_etaf1.columns.to_numpy(dtype=float)
    g_vals = pivot_etaf1.index.to_numpy(dtype=float)
    z = pivot_etaf1.to_numpy(dtype=float)
    fpa = pivot_fpa.to_numpy(dtype=float)
    return s_vals, g_vals, z, fpa


def cell_edges(values: np.ndarray, *, lo: float, hi: float) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    if vals.size == 1:
        return np.asarray([lo, hi], dtype=float)
    mids = (vals[:-1] + vals[1:]) / 2.0
    edges = np.concatenate([[vals[0] - (mids[0] - vals[0])], mids, [vals[-1] + (vals[-1] - mids[-1])]])
    edges[0] = min(edges[0], lo)
    edges[-1] = max(edges[-1], hi)
    return edges


def draw_hatching(ax: plt.Axes, s_vals: np.ndarray, g_vals: np.ndarray, fpa: np.ndarray) -> None:
    s_edges = cell_edges(s_vals, lo=0.0, hi=5.0)
    g_edges = cell_edges(g_vals, lo=0.0, hi=25.0)
    for i in range(len(g_vals)):
        for j in range(len(s_vals)):
            if np.isfinite(fpa[i, j]) and fpa[i, j] > 0:
                ax.add_patch(
                    Rectangle(
                        (s_edges[j], g_edges[i]),
                        s_edges[j + 1] - s_edges[j],
                        g_edges[i + 1] - g_edges[i],
                        facecolor="none",
                        edgecolor=(0, 0, 0, 0.62),
                        hatch="////",
                        lw=0.0,
                        zorder=3,
                    )
                )


def draw_dataset(ax: plt.Axes, dataset: str, grid: pd.DataFrame, best: pd.Series, *, colorbar: bool = False) -> Any:
    cfg = DATASETS[dataset]
    s_vals, g_vals, z, fpa = grid_to_matrices(grid)
    s_edges = cell_edges(s_vals, lo=0.0, hi=5.0)
    g_edges = cell_edges(g_vals, lo=0.0, hi=25.0)
    im = ax.pcolormesh(
        s_edges,
        g_edges,
        z,
        shading="auto",
        vmin=0.0,
        vmax=100.0,
        cmap="YlGnBu",
        zorder=1,
    )
    draw_hatching(ax, s_vals, g_vals, fpa)

    threshold = 0.95 * float(np.nanmax(z))
    if np.nanmin(z) <= threshold <= np.nanmax(z) and len(s_vals) >= 2 and len(g_vals) >= 2:
        ss, gg = np.meshgrid(s_vals, g_vals)
        ax.contour(ss, gg, z, levels=[threshold], colors="black", linewidths=1.05, zorder=4)

    ax.scatter(
        [float(cfg["geco_s"])],
        [float(cfg["geco_g"])],
        marker="X",
        s=44,
        c="#b22222",
        edgecolors="#7a1111",
        linewidths=0.8,
        zorder=6,
    )
    ax.scatter(
        [float(best["S"])],
        [float(best["G"])],
        marker="o",
        s=34,
        facecolors="white",
        edgecolors="black",
        linewidths=1.0,
        zorder=7,
    )

    ax.text(
        0.975,
        0.93,
        f"eTaF1={float(best['eTaF1']):.1f}\nFPA={float(best['FPA']):.0f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.2},
        zorder=8,
    )

    ax.text(0.5, 1.035, dataset, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xlim(0.0, 5.0)
    ax.set_ylim(0.0, 25.0)
    ax.set_xticks([0, 1, 2, 3, 4, 5])
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.tick_params(axis="both", labelsize=6.2, pad=1.0, length=2.2, width=0.6)
    ax.set_xlabel("scale factor (S)", fontsize=6.7, labelpad=1.2)
    ax.set_ylabel("growth factor (G)", fontsize=6.7, labelpad=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    if colorbar:
        cbar = ax.figure.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
        cbar.set_label("eTaF1", fontsize=6.7)
        cbar.ax.tick_params(labelsize=6.2, pad=1)
    return im


def variable_model_from_row(post: Any, row: dict[str, Any]) -> Any:
    def num(name: str) -> float:
        try:
            return float(row.get(name, float("nan")))
        except Exception:
            return float("nan")

    return post.VariableModel(
        target=str(row["target"]),
        variable_type=str(row.get("variable_type", "sensor")),
        equation=str(row.get("equation", "")),
        sympy_format=str(row.get("sympy_format", row.get("equation", ""))),
        complexity=num("complexity"),
        loss=num("loss"),
        score=num("score"),
        holdout_r2=num("holdout_r2"),
        holdout_mae=num("holdout_mae"),
        baseline_holdout_mae=num("baseline_holdout_mae"),
        residual_tail_ratio=num("residual_tail_ratio"),
        source=str(row.get("source", "no_holdout_quality_gate")),
    )


def verify_swat_no_gate_point(swat_grid: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    point = swat_grid[np.isclose(swat_grid["S"], 1.2) & np.isclose(swat_grid["G"], 15.0)]
    if point.empty:
        raise ValueError("Missing SWaT no-gate verification point S=1.2 G=15.0")
    point = point.iloc[0]

    post = import_module("final_heatmap_swat_post", REPO_ROOT / "scripts" / "run_swat_1sec_delta_posthoc_ablation.py")
    arrays = post.load_arrays(
        argparse.Namespace(
            experiment="configs/experiment/swat_mlp_current_val20.yaml",
            train_csv=None,
            test_csv=None,
        )
    )
    models_df = pd.read_csv(SWAT_NOGATE_DIR / "monitored_models_geco_matched_plus_all_actuator_persistence_no_holdout_quality_gate.csv")
    models = [variable_model_from_row(post, row) for row in models_df.to_dict("records")]
    cache = post.residual_cache_for_models(arrays, models)
    alarms = post.alarm_map_for_choice(models, cache, arrays["labels"], s=1.2, g=15.0)
    metrics = compute_detection_metrics(arrays["labels"], alarms["system"], expand_steps=60)
    per_attack = post.per_attack_table(models, alarms)
    for col, value in {
        "S": 1.2,
        "G": 15.0,
        "F1": metrics["point_f1"],
        "eTaF1": metrics["eTaF1"],
        "FPA": metrics["FPA"],
        "Scen": metrics["scenario_detection_rate"],
        "grid_F1": point["F1"],
        "grid_eTaF1": point["eTaF1"],
        "grid_FPA": point["FPA"],
        "grid_Scen": point["Scen"],
    }.items():
        per_attack[col] = value

    counts = per_attack["category"].value_counts().to_dict()
    verification = pd.DataFrame(
        [
            {
                "dataset": "SWaT",
                "variant": "geco_matched_plus_all_actuator_persistence_no_holdout_quality_gate_dense",
                "S": 1.2,
                "G": 15.0,
                "Precision": metrics["point_precision"],
                "Recall": metrics["point_recall"],
                "F1": metrics["point_f1"],
                "eTaF1": metrics["eTaF1"],
                "FPA": metrics["FPA"],
                "Scen": metrics["scenario_detection_rate"],
                "grid_F1": float(point["F1"]),
                "grid_eTaF1": float(point["eTaF1"]),
                "grid_FPA": float(point["FPA"]),
                "grid_Scen": float(point["Scen"]),
                "direct_detected": int(counts.get("direct_detected", 0)),
                "collateral_detected": int(counts.get("collateral_detected", 0)),
                "scope_miss": int(counts.get("scope_miss", 0)),
                "detection_failure": int(counts.get("detection_failure", 0)),
                "per_attack_csv": str(SWAT_NOGATE_DIR / "per_attack_geco_matched_plus_all_actuator_persistence_no_holdout_quality_gate_S1p2_G15.csv"),
            }
        ]
    )
    return verification, per_attack


def save_figures(grids: dict[str, pd.DataFrame], best_rows: dict[str, pd.Series], out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    for dataset, cfg in DATASETS.items():
        fig, ax = plt.subplots(figsize=(2.45, 2.35), constrained_layout=True)
        draw_dataset(ax, dataset, grids[dataset], best_rows[dataset], colorbar=True)
        fig.savefig(out_dir / f"heatmap_final_{cfg['slug']}.pdf", bbox_inches="tight", dpi=300)
        fig.savefig(out_dir / f"heatmap_final_{cfg['slug']}.png", bbox_inches="tight", dpi=300)
        plt.close(fig)

    fig, axes = plt.subplots(1, 3, figsize=(7.2, 2.38), constrained_layout=False)
    fig.subplots_adjust(left=0.055, right=0.905, top=0.86, bottom=0.24, wspace=0.22)
    im = None
    for i, (ax, dataset) in enumerate(zip(axes, ["SWaT", "WADI", "BATADAL"])):
        im = draw_dataset(ax, dataset, grids[dataset], best_rows[dataset], colorbar=False)
        if i > 0:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
    cax = fig.add_axes([0.925, 0.245, 0.018, 0.61])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("eTaF1", fontsize=6.7)
    cbar.ax.tick_params(labelsize=6.2, pad=1)
    handles = [
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#b22222", markeredgecolor="#7a1111", markersize=5.5, label="GeCo published S/G"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="white", markeredgewidth=1.0, linestyle="None", markersize=4.8, label="ASID-ICS best eTaF1 (FPA constrained)"),
        Line2D([0], [0], color="black", linewidth=1.05, label="95% of max eTaF1"),
        Patch(facecolor="white", edgecolor=(0, 0, 0, 0.62), hatch="////", label="FPA > 0"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.49, 0.035), ncol=4, frameon=True, fontsize=6.4, handlelength=1.55, columnspacing=0.8)
    fig.savefig(out_dir / "heatmap_final.pdf", bbox_inches="tight", dpi=300)
    fig.savefig(out_dir / "heatmap_final.png", bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate final ASID-ICS S/G heatmaps using SWaT no-gate dense data.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "hatch.linewidth": 0.55,
        }
    )

    grids = {dataset: read_grid(cfg["grid"]) for dataset, cfg in DATASETS.items()}
    best_rows = {dataset: choose_best_row(dataset, grids[dataset], cfg) for dataset, cfg in DATASETS.items()}
    verification, per_attack = verify_swat_no_gate_point(grids["SWaT"])
    SWAT_NOGATE_DIR.mkdir(parents=True, exist_ok=True)
    per_attack_path = SWAT_NOGATE_DIR / "per_attack_geco_matched_plus_all_actuator_persistence_no_holdout_quality_gate_S1p2_G15.csv"
    per_attack.to_csv(per_attack_path, index=False)
    verification_path = PAPER_DIR / "swat_no_gate_dense_S1p2_G15_verification.csv"
    verification.to_csv(verification_path, index=False)

    summary_rows = []
    for dataset, cfg in DATASETS.items():
        grid = grids[dataset]
        best = best_rows[dataset]
        geco = grid.assign(_dist=(grid["S"] - float(cfg["geco_s"])) ** 2 + (grid["G"] - float(cfg["geco_g"])) ** 2).sort_values("_dist").iloc[0]
        overall = grid.sort_values(["eTaF1", "F1"], ascending=False).iloc[0]
        summary_rows.append(
            {
                "dataset": dataset,
                "source": cfg["source"],
                "rows": len(grid),
                "S_min": float(grid["S"].min()),
                "S_max": float(grid["S"].max()),
                "G_min": float(grid["G"].min()),
                "G_max": float(grid["G"].max()),
                "best_selection": cfg["selection"],
                "best_S": float(best["S"]),
                "best_G": float(best["G"]),
                "best_F1": float(best["F1"]),
                "best_eTaF1": float(best["eTaF1"]),
                "best_FPA": float(best["FPA"]),
                "best_Scen": float(best["Scen"]),
                "overall_max_eTaF1": float(overall["eTaF1"]),
                "geco_S": float(cfg["geco_s"]),
                "geco_G": float(cfg["geco_g"]),
                "geco_grid_S": float(geco["S"]),
                "geco_grid_G": float(geco["G"]),
                "geco_grid_eTaF1": float(geco["eTaF1"]),
                "geco_grid_FPA": float(geco["FPA"]),
            }
        )
    summary = pd.DataFrame(summary_rows)
    summary_path = PAPER_DIR / "heatmap_final_summary.csv"
    summary.to_csv(summary_path, index=False)
    save_figures(grids, best_rows, Path(args.out_dir))

    print("SWaT no-gate verification at S=1.2, G=15.0:")
    print(verification.to_string(index=False))
    print()
    print("Final heatmap summary:")
    print(summary.to_string(index=False))
    for path in [
        Path(args.out_dir) / "heatmap_final_swat.pdf",
        Path(args.out_dir) / "heatmap_final_wadi.pdf",
        Path(args.out_dir) / "heatmap_final_batadal.pdf",
        Path(args.out_dir) / "heatmap_final.pdf",
        verification_path,
        per_attack_path,
        summary_path,
    ]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
