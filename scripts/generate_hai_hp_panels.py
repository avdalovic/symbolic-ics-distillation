#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("pdf")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.path import Path as MplPath
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
OUT_DIR = REPO_ROOT / "paper_artifacts" / "final_v2"
HP_DIR = REPO_ROOT / "artifacts" / "experiments" / "hai_hp"
GECO_HAI_S = 8.02147642080159
GECO_HAI_G = 1.4376682451456457
DEFAULT_CONTOUR_RATIO = 0.90


PANEL_CONFIGS: dict[str, dict[str, Any]] = {
    "SWaT": {
        "grid": REPO_ROOT / "paper_artifacts/final_v2/extended_grid_swat.csv",
        "geco_s": 1.42,
        "geco_g": 5.98,
        "chosen_s": 1.2,
        "chosen_g": 15.0,
    },
    "WADI": {
        "grid": REPO_ROOT / "paper_artifacts/final_v2/extended_grid_wadi.csv",
        "geco_s": 1.32,
        "geco_g": 9.74,
        "chosen_s": 1.2,
        "chosen_g": 25.0,
    },
    "BATADAL": {
        "grid": REPO_ROOT / "paper_artifacts/selected_models/batadal/detection_grid.csv",
        "geco_s": 1.39,
        "geco_g": 2.16,
        "chosen_s": 1.4,
        "chosen_g": 2.0,
    },
    "HAI": {
        "grid": REPO_ROOT / "paper_artifacts/selected_models/hai/detection_grid_r13.csv",
        "fallback_grid": REPO_ROOT / "artifacts/experiments/hai_division_seed0/analysis/G_safe_R_13_grid.csv",
        "geco_s": GECO_HAI_S,
        "geco_g": GECO_HAI_G,
    },
}


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


def grid_path(cfg: dict[str, Any]) -> Path:
    path = Path(cfg["grid"])
    if path.exists():
        return path
    fallback = cfg.get("fallback_grid")
    if fallback is not None and Path(fallback).exists():
        return Path(fallback)
    return path


def heatmap_rows(grid: pd.DataFrame) -> pd.DataFrame:
    if "point_kind" in grid.columns:
        regular = grid[grid["point_kind"].astype(str) == "grid"].copy()
        if not regular.empty:
            grid = regular
    out = grid.copy()
    if out.empty:
        return out
    s_counts = out.groupby("S").size()
    g_counts = out.groupby("G").size()
    keep_s = set(s_counts[s_counts >= 0.9 * float(s_counts.max())].index)
    keep_g = set(g_counts[g_counts >= 0.9 * float(g_counts.max())].index)
    return out[out["S"].isin(keep_s) & out["G"].isin(keep_g)].copy()


def grid_to_matrices(grid: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    grid = heatmap_rows(grid)
    pivot_etaf1 = grid.pivot_table(index="G", columns="S", values="eTaF1", aggfunc="mean").sort_index(axis=0).sort_index(axis=1)
    pivot_fpa = grid.pivot_table(index="G", columns="S", values="FPA", aggfunc="mean").reindex_like(pivot_etaf1)
    return (
        pivot_etaf1.columns.to_numpy(dtype=float),
        pivot_etaf1.index.to_numpy(dtype=float),
        pivot_etaf1.to_numpy(dtype=float),
        pivot_fpa.to_numpy(dtype=float),
    )


def cell_edges(values: np.ndarray, *, lo: float, hi: float) -> np.ndarray:
    vals = np.asarray(values, dtype=float)
    if vals.size == 1:
        return np.asarray([lo, hi], dtype=float)
    mids = (vals[:-1] + vals[1:]) / 2.0
    edges = np.concatenate([[vals[0] - (mids[0] - vals[0])], mids, [vals[-1] + (vals[-1] - mids[-1])]])
    edges[0] = min(edges[0], lo)
    edges[-1] = max(edges[-1], hi)
    return edges


def draw_hatching(ax: plt.Axes, s_vals: np.ndarray, g_vals: np.ndarray, fpa: np.ndarray, *, x_hi: float, y_hi: float) -> None:
    s_edges = cell_edges(s_vals, lo=0.0, hi=x_hi)
    g_edges = cell_edges(g_vals, lo=0.0, hi=y_hi)
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


def split_contour_paths(paths: list[MplPath]) -> list[np.ndarray]:
    subpaths: list[np.ndarray] = []
    for path in paths:
        vertices = path.vertices
        codes = path.codes
        if codes is None:
            if len(vertices) > 1:
                subpaths.append(np.asarray(vertices, dtype=float))
            continue
        current: list[np.ndarray] = []
        for vertex, code in zip(vertices, codes):
            if code == MplPath.MOVETO:
                if len(current) > 1:
                    subpaths.append(np.asarray(current, dtype=float))
                current = [vertex]
            elif code == MplPath.LINETO:
                current.append(vertex)
            elif code == MplPath.CLOSEPOLY:
                if len(current) > 1:
                    subpaths.append(np.asarray(current, dtype=float))
                current = []
        if len(current) > 1:
            subpaths.append(np.asarray(current, dtype=float))
    return subpaths


def nearest_row(grid: pd.DataFrame, s: float, g: float) -> pd.Series:
    exact = grid[np.isclose(grid["S"], float(s)) & np.isclose(grid["G"], float(g))]
    if not exact.empty:
        return exact.iloc[0]
    return grid.assign(_dist=(grid["S"] - float(s)) ** 2 + (grid["G"] - float(g)) ** 2).sort_values("_dist").iloc[0]


def neighborhood_table(grid: pd.DataFrame, *, fpa_gate: int = 8) -> pd.DataFrame:
    s_vals, g_vals, _, _ = grid_to_matrices(grid)
    by_point = {(float(row.S), float(row.G)): row for row in grid.itertuples(index=False)}
    rows = []
    for row in grid[grid["FPA"] <= fpa_gate].itertuples(index=False):
        s_idx = int(np.where(np.isclose(s_vals, float(row.S)))[0][0])
        g_idx = int(np.where(np.isclose(g_vals, float(row.G)))[0][0])
        neighbors = []
        for gi in range(max(0, g_idx - 1), min(len(g_vals), g_idx + 2)):
            for si in range(max(0, s_idx - 1), min(len(s_vals), s_idx + 2)):
                neighbor = by_point.get((float(s_vals[si]), float(g_vals[gi])))
                if neighbor is not None:
                    neighbors.append(neighbor)
        etaf1 = np.asarray([float(n.eTaF1) for n in neighbors], dtype=float)
        fpa = np.asarray([float(n.FPA) for n in neighbors], dtype=float)
        rows.append(
            {
                "S": float(row.S),
                "G": float(row.G),
                "Precision": float(row.Precision),
                "Recall": float(row.Recall),
                "F1": float(row.F1),
                "eTaP": float(row.eTaP) if hasattr(row, "eTaP") else np.nan,
                "eTaR": float(row.eTaR) if hasattr(row, "eTaR") else np.nan,
                "eTaF1": float(row.eTaF1),
                "FPA": float(row.FPA),
                "Scen": float(row.Scen),
                "neighborhood_size": int(len(neighbors)),
                "neighborhood_mean_eTaF1": float(np.mean(etaf1)),
                "neighborhood_min_eTaF1": float(np.min(etaf1)),
                "neighborhood_max_FPA": float(np.max(fpa)),
                "neighborhood_all_FPA_le_8": bool(np.all(fpa <= fpa_gate)),
            }
        )
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    return out.sort_values(
        ["neighborhood_all_FPA_le_8", "eTaF1", "neighborhood_min_eTaF1", "F1", "Scen"],
        ascending=[False, False, False, False, False],
    ).reset_index(drop=True)


def choose_hai_point(table: pd.DataFrame, *, requested_s: float | None, requested_g: float | None) -> tuple[float, float, str]:
    if requested_s is not None and requested_g is not None:
        return float(requested_s), float(requested_g), "user_supplied"
    row = table.sort_values(["eTaF1", "F1", "Scen"], ascending=False).iloc[0]
    return float(row["S"]), float(row["G"]), "auto_best_etaf1_with_FPA_le_8"


def mark_chosen_point(table: pd.DataFrame, *, chosen_s: float, chosen_g: float) -> pd.DataFrame:
    out = table.copy()
    out.insert(0, "chosen_point", np.isclose(out["S"], chosen_s) & np.isclose(out["G"], chosen_g))
    return out


def draw_panel(
    ax: plt.Axes,
    *,
    title: str,
    grid: pd.DataFrame,
    geco_s: float,
    geco_g: float,
    chosen: pd.Series,
    x_hi: float,
    y_hi: float,
    contour_anchor: tuple[float, float] | None = None,
    contour_ratio: float = DEFAULT_CONTOUR_RATIO,
) -> Any:
    s_vals, g_vals, z, fpa = grid_to_matrices(grid)
    s_edges = cell_edges(s_vals, lo=0.0, hi=x_hi)
    g_edges = cell_edges(g_vals, lo=0.0, hi=y_hi)
    im = ax.pcolormesh(s_edges, g_edges, z, shading="auto", vmin=0.0, vmax=100.0, cmap="YlGnBu", zorder=1)
    draw_hatching(ax, s_vals, g_vals, fpa, x_hi=x_hi, y_hi=y_hi)
    threshold = contour_ratio * float(np.nanmax(z))
    if np.nanmin(z) <= threshold <= np.nanmax(z) and len(s_vals) >= 2 and len(g_vals) >= 2:
        ss, gg = np.meshgrid(s_vals, g_vals)
        contour = ax.contour(ss, gg, z, levels=[threshold], colors="black", linewidths=1.05, zorder=4)
        if contour_anchor is not None:
            subpaths = split_contour_paths(contour.get_paths())
            contour.remove()
            if subpaths:
                anchor = np.asarray(contour_anchor, dtype=float)
                selected = min(subpaths, key=lambda path: float(np.min(np.linalg.norm(path - anchor, axis=1))))
                ax.plot(selected[:, 0], selected[:, 1], color="black", linewidth=1.05, zorder=4)
    ax.scatter([float(geco_s)], [float(geco_g)], marker="X", s=44, c="#b22222", edgecolors="#7a1111", linewidths=0.8, zorder=6)
    ax.scatter([float(chosen["S"])], [float(chosen["G"])], marker="o", s=34, facecolors="white", edgecolors="black", linewidths=1.0, zorder=7)
    ax.text(
        0.975,
        0.93,
        f"eTaF1={float(chosen['eTaF1']):.1f}\nFPA={float(chosen['FPA']):.0f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=5.8,
        bbox={"facecolor": "white", "alpha": 0.78, "edgecolor": "none", "pad": 1.2},
        zorder=8,
    )
    ax.text(0.5, 1.035, title, transform=ax.transAxes, ha="center", va="bottom", fontsize=8, fontweight="bold")
    ax.set_xlim(0.0, x_hi)
    ax.set_ylim(0.0, y_hi)
    if x_hi <= 5.0:
        ax.set_xticks([0, 1, 2, 3, 4, 5])
    elif x_hi <= 8.5:
        ax.set_xticks([0, 2, 4, 6, 8])
    else:
        ax.set_xticks([0, 2, 4, 6, 8, 10])
    ax.set_yticks([0, 5, 10, 15, 20, 25])
    ax.tick_params(axis="both", labelsize=6.2, pad=1.0, length=2.2, width=0.6)
    ax.set_xlabel("scale factor (S)", fontsize=6.7, labelpad=1.2)
    ax.set_ylabel("growth factor (G)", fontsize=6.7, labelpad=1.2)
    for spine in ax.spines.values():
        spine.set_linewidth(0.7)
    return im


def save_single_panel(
    path: Path,
    *,
    title: str,
    grid: pd.DataFrame,
    geco_s: float,
    geco_g: float,
    chosen: pd.Series,
    contour_ratio: float,
) -> None:
    x_hi = max(5.0, float(grid["S"].max()), float(geco_s))
    y_hi = max(25.0, float(grid["G"].max()), float(geco_g))
    fig, ax = plt.subplots(figsize=(2.65, 2.35), constrained_layout=True)
    contour_anchor = (float(chosen["S"]), float(chosen["G"])) if title.startswith("HAI") else None
    im = draw_panel(
        ax,
        title=title,
        grid=grid,
        geco_s=geco_s,
        geco_g=geco_g,
        chosen=chosen,
        x_hi=x_hi,
        y_hi=y_hi,
        contour_anchor=contour_anchor,
        contour_ratio=contour_ratio,
    )
    cbar = fig.colorbar(im, ax=ax, fraction=0.046, pad=0.03)
    cbar.set_label("eTaF1", fontsize=6.7)
    cbar.ax.tick_params(labelsize=6.2, pad=1)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def save_fig4(path: Path, grids: dict[str, pd.DataFrame], chosen_rows: dict[str, pd.Series], *, contour_ratio: float) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(9.25, 2.38), constrained_layout=False)
    fig.subplots_adjust(left=0.048, right=0.91, top=0.86, bottom=0.25, wspace=0.23)
    im = None
    for i, (ax, title) in enumerate(zip(axes, ["SWaT", "WADI", "BATADAL", "HAI"])):
        cfg = PANEL_CONFIGS[title]
        x_hi = max(5.0, float(grids[title]["S"].max()), float(cfg["geco_s"]))
        y_hi = max(25.0, float(grids[title]["G"].max()), float(cfg["geco_g"]))
        im = draw_panel(
            ax,
            title=title,
            grid=grids[title],
            geco_s=float(cfg["geco_s"]),
            geco_g=float(cfg["geco_g"]),
            chosen=chosen_rows[title],
            x_hi=x_hi,
            y_hi=y_hi,
            contour_anchor=(float(chosen_rows[title]["S"]), float(chosen_rows[title]["G"])) if title == "HAI" else None,
            contour_ratio=contour_ratio,
        )
        if i > 0:
            ax.set_ylabel("")
            ax.tick_params(labelleft=False)
    cax = fig.add_axes([0.928, 0.252, 0.016, 0.60])
    cbar = fig.colorbar(im, cax=cax)
    cbar.set_label("eTaF1", fontsize=6.7)
    cbar.ax.tick_params(labelsize=6.2, pad=1)
    handles = [
        Line2D([0], [0], marker="X", color="none", markerfacecolor="#b22222", markeredgecolor="#7a1111", markersize=5.5, label="GeCo published S/G"),
        Line2D([0], [0], marker="o", color="black", markerfacecolor="white", markeredgewidth=1.0, linestyle="None", markersize=4.8, label="ASID-ICS operating point (Table II)"),
        Line2D([0], [0], color="black", linewidth=1.05, label=f"{contour_ratio * 100:.0f}% of max eTaF1"),
        Patch(facecolor="white", edgecolor=(0, 0, 0, 0.62), hatch="////", label="FPA > 0"),
    ]
    fig.legend(handles=handles, loc="lower center", bbox_to_anchor=(0.50, 0.035), ncol=4, frameon=True, fontsize=6.4, handlelength=1.55, columnspacing=0.8)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate HAI hyperparameter panels and point-selection table.")
    parser.add_argument("--chosen-s", type=float, default=None)
    parser.add_argument("--chosen-g", type=float, default=None)
    parser.add_argument("--contour-ratio", type=float, default=DEFAULT_CONTOUR_RATIO)
    parser.add_argument("--fig4-name", default="fig4_v2.pdf")
    args = parser.parse_args()
    if not 0.0 < args.contour_ratio <= 1.0:
        raise ValueError("--contour-ratio must be in (0, 1]")

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "hatch.linewidth": 0.55,
        }
    )
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    HP_DIR.mkdir(parents=True, exist_ok=True)

    grids = {name: read_grid(grid_path(cfg)) for name, cfg in PANEL_CONFIGS.items()}
    r13_table = neighborhood_table(grids["HAI"], fpa_gate=8)
    chosen_s, chosen_g, chosen_policy = choose_hai_point(r13_table, requested_s=args.chosen_s, requested_g=args.chosen_g)
    chosen_hai_r13 = nearest_row(grids["HAI"], chosen_s, chosen_g)
    r13_table = mark_chosen_point(r13_table, chosen_s=float(chosen_hai_r13["S"]), chosen_g=float(chosen_hai_r13["G"]))
    chosen_rows = {
        "SWaT": nearest_row(grids["SWaT"], 1.2, 15.0),
        "WADI": nearest_row(grids["WADI"], 1.2, 25.0),
        "BATADAL": nearest_row(grids["BATADAL"], 1.4, 2.0),
        "HAI": chosen_hai_r13,
    }
    # Optional context: the same point on the unrestricted 54-channel HAI grid.
    # Present only when a local discovery run exists; not required for Figure 4.
    all54_grid_path = REPO_ROOT / "artifacts/experiments/hai_baseline_seed0/posthoc_rosters_smax10/all_detection_grid.csv"
    if not all54_grid_path.exists():
        all54_grid_path = REPO_ROOT / "artifacts/experiments/hai_baseline_seed0/detection_grid.csv"
    chosen_all54 = nearest_row(read_grid(all54_grid_path), chosen_s, chosen_g) if all54_grid_path.exists() else None

    r13_table.to_csv(HP_DIR / "hai_point_selection_table.csv", index=False)
    chosen_payload = {
        "chosen_policy": chosen_policy,
        "chosen_point": {
            "S": float(chosen_hai_r13["S"]),
            "G": float(chosen_hai_r13["G"]),
            "Precision": float(chosen_hai_r13["Precision"]),
            "Recall": float(chosen_hai_r13["Recall"]),
            "F1": float(chosen_hai_r13["F1"]),
            "eTaF1": float(chosen_hai_r13["eTaF1"]),
            "FPA": float(chosen_hai_r13["FPA"]),
            "Scen": float(chosen_hai_r13["Scen"]),
        },
        "all54_same_point": None if chosen_all54 is None else {
            "S": float(chosen_all54["S"]),
            "G": float(chosen_all54["G"]),
            "Precision": float(chosen_all54["Precision"]),
            "Recall": float(chosen_all54["Recall"]),
            "F1": float(chosen_all54["F1"]),
            "eTaF1": float(chosen_all54["eTaF1"]),
            "FPA": float(chosen_all54["FPA"]),
            "Scen": float(chosen_all54["Scen"]),
        },
    }
    (HP_DIR / "hai_chosen_point.json").write_text(json.dumps(chosen_payload, indent=2, allow_nan=False), encoding="utf-8")
    pd.DataFrame([chosen_payload["chosen_point"]]).to_csv(HP_DIR / "hai_chosen_point.csv", index=False)

    save_single_panel(
        OUT_DIR / "hai_panel_r13.pdf",
        title="HAI",
        grid=grids["HAI"],
        geco_s=GECO_HAI_S,
        geco_g=GECO_HAI_G,
        chosen=chosen_hai_r13,
        contour_ratio=args.contour_ratio,
    )
    save_single_panel(
        OUT_DIR / "hai_panel.pdf",
        title="HAI",
        grid=grids["HAI"],
        geco_s=GECO_HAI_S,
        geco_g=GECO_HAI_G,
        chosen=chosen_hai_r13,
        contour_ratio=args.contour_ratio,
    )
    if chosen_all54 is not None:
        save_single_panel(
            OUT_DIR / "hai_panel_all54.pdf",
            title="HAI all-54",
            grid=read_grid(all54_grid_path),
            geco_s=GECO_HAI_S,
            geco_g=GECO_HAI_G,
            chosen=chosen_all54,
            contour_ratio=args.contour_ratio,
        )
    save_fig4(OUT_DIR / args.fig4_name, grids, chosen_rows, contour_ratio=args.contour_ratio)

    print(json.dumps(chosen_payload, indent=2, allow_nan=False))
    print("Chosen HAI R_13 operating point:")
    print(r13_table[r13_table["chosen_point"]].to_string(index=False))
    print("Top HAI R_13 FPA<=8 candidates, with neighborhood stability:")
    print(r13_table.head(12).to_string(index=False))
    print(f"Saved {OUT_DIR / 'hai_panel_r13.pdf'}")
    print(f"Saved {OUT_DIR / 'hai_panel_all54.pdf'}")
    print(f"Saved {OUT_DIR / args.fig4_name}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
