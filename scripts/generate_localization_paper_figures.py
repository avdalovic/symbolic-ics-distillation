#!/usr/bin/env python
from __future__ import annotations

import argparse
import math
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ics_symbolic_distill.detection import evaluate_equation, fit_cusum_params, run_cusum
from ics_symbolic_distill.detection.swat1s_delta_sampling import reconstruct_next_from_delta


DATASETS = ["SWaT", "WADI", "HAI", "BATADAL"]
RADII = [(0, "<=0"), (1, "<=1"), (2, "<=2"), (3, "<=3")]
COLORS = {
    "SWaT": "#0072B2",
    "WADI": "#D55E00",
    "HAI": "#E69F00",
    "BATADAL": "#009E73",
}


def _bool_series(series: pd.Series) -> np.ndarray:
    return series.astype(str).str.lower().isin({"true", "1", "yes"}).to_numpy()


def _distance_array(series: pd.Series) -> np.ndarray:
    return pd.to_numeric(series, errors="coerce").replace([np.inf, -np.inf], np.inf).to_numpy(dtype=float)


def load_per_attack(paths: dict[str, Path]) -> dict[str, pd.DataFrame]:
    frames = {}
    for dataset, path in paths.items():
        if not path.exists():
            raise FileNotFoundError(f"Missing {dataset} localization table: {path}")
        frames[dataset] = pd.read_csv(path)
    return frames


def summarize(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame]:
    gecko_style_rows = []
    conservative_rows = []
    for dataset in DATASETS:
        df = frames[dataset]
        distances = _distance_array(df["nearest_distance"])
        detected = _bool_series(df["detected"])
        finite = np.isfinite(distances)
        connected_detected = detected & finite

        for radius, label in RADII:
            gecko_style_rows.append(
                {
                    "dataset": dataset,
                    "radius": label,
                    "radius_hops": radius,
                    "percentage": float(np.mean(distances[connected_detected] <= radius) * 100.0)
                    if np.any(connected_detected)
                    else math.nan,
                    "denominator": int(np.sum(connected_detected)),
                    "convention": "detected_attacks_with_finite_graph_distance",
                }
            )
            conservative_rows.append(
                {
                    "dataset": dataset,
                    "radius": label,
                    "radius_hops": radius,
                    "percentage": float(np.mean(distances <= radius) * 100.0) if len(df) else math.nan,
                    "denominator": int(len(df)),
                    "convention": "all_attack_scenarios",
                }
            )

    summary_rows = []
    for dataset in DATASETS:
        df = frames[dataset]
        distances = _distance_array(df["nearest_distance"])
        detected = _bool_series(df["detected"])
        finite = np.isfinite(distances)
        connected_detected = detected & finite
        summary_rows.append(
            {
                "dataset": dataset,
                "scenarios": int(len(df)),
                "detected": int(np.sum(detected)),
                "connected_detected": int(np.sum(connected_detected)),
                "detected_disconnected": int(np.sum(detected & ~finite)),
                "geco_style_D_eq_0": float(np.mean(distances[connected_detected] == 0) * 100.0)
                if np.any(connected_detected)
                else math.nan,
                "geco_style_D_le_1": float(np.mean(distances[connected_detected] <= 1) * 100.0)
                if np.any(connected_detected)
                else math.nan,
                "geco_style_D_le_2": float(np.mean(distances[connected_detected] <= 2) * 100.0)
                if np.any(connected_detected)
                else math.nan,
                "geco_style_D_le_3": float(np.mean(distances[connected_detected] <= 3) * 100.0)
                if np.any(connected_detected)
                else math.nan,
                "mean_finite_hops": float(np.mean(distances[connected_detected]))
                if np.any(connected_detected)
                else math.nan,
                "conservative_D_eq_0": float(np.mean(distances == 0) * 100.0) if len(df) else math.nan,
                "conservative_D_le_1": float(np.mean(distances <= 1) * 100.0) if len(df) else math.nan,
                "conservative_D_le_2": float(np.mean(distances <= 2) * 100.0) if len(df) else math.nan,
                "conservative_D_le_3": float(np.mean(distances <= 3) * 100.0) if len(df) else math.nan,
            }
        )

    values = pd.concat([pd.DataFrame(gecko_style_rows), pd.DataFrame(conservative_rows)], ignore_index=True)
    summary = pd.DataFrame(summary_rows)
    return values, summary


def plot_geco_style(values: pd.DataFrame, out_dir: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.7, 2.45))
    x = np.arange(len(RADII))
    for dataset in DATASETS:
        rows = values[
            (values["dataset"] == dataset)
            & (values["convention"] == "detected_attacks_with_finite_graph_distance")
        ].sort_values("radius_hops")
        ax.plot(
            x,
            rows["percentage"].to_numpy(dtype=float),
            color=COLORS[dataset],
            lw=2.1,
            label=dataset,
        )

    ax.set_xticks(x)
    ax.set_xticklabels([label for _, label in RADII])
    ax.set_ylim(40, 102)
    ax.set_xlim(-0.03, len(RADII) - 0.97)
    ax.set_xlabel("Distance to attack of alerts")
    ax.set_ylabel("Attacks in distance [%]")
    ax.grid(axis="y", alpha=0.22, lw=0.6)
    ax.legend(
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncol=4,
        frameon=True,
        fontsize=8,
        columnspacing=0.9,
        handlelength=2.3,
        handletextpad=0.35,
    )
    fig.tight_layout(pad=0.5)
    fig.savefig(out_dir / "localization_connected_detected_geco_style.pdf")
    fig.savefig(out_dir / "localization_connected_detected_geco_style.png", dpi=300)
    plt.close(fig)


def plot_local_swat_graph(graph_path: Path, per_attack_path: Path, out_dir: Path) -> None:
    graph = nx.read_graphml(graph_path)
    per_attack = pd.read_csv(per_attack_path)
    candidates = per_attack[
        (_bool_series(per_attack["detected"]))
        & (per_attack["n_alert_nodes"].astype(int) == 1)
        & (_distance_array(per_attack["nearest_distance"]) == 0)
        & (per_attack["attacked_tags"].astype(str) == "LIT101")
    ]
    if candidates.empty:
        candidates = per_attack[
            (_bool_series(per_attack["detected"]))
            & (per_attack["n_alert_nodes"].astype(int) == 1)
            & (_distance_array(per_attack["nearest_distance"]) == 0)
        ]
    if candidates.empty:
        raise RuntimeError("Could not find a single-alert exact-target SWaT localization example.")

    row = candidates.sort_values("attack_id").iloc[0]
    attack_id = str(row["attack_id"])
    attacked = {tag for tag in str(row["attacked_tags"]).split(";") if tag}
    alert = {tag for tag in str(row["alert_nodes"]).split(";") if tag}
    focus = sorted(attacked | alert)[0]
    predecessors = set(graph.predecessors(focus)) if focus in graph else set()
    nodes = sorted(predecessors | attacked | alert)
    subgraph = nx.DiGraph()
    subgraph.add_nodes_from(nodes)
    for predecessor in sorted(predecessors):
        subgraph.add_edge(predecessor, focus)

    pos = {}
    inputs = sorted(n for n in nodes if n != focus)
    if len(inputs) == 1:
        pos[inputs[0]] = (-1.2, 0.0)
    else:
        for i, node in enumerate(inputs):
            y = 0.52 - i * (1.04 / max(1, len(inputs) - 1))
            pos[node] = (-1.25, y)
    pos[focus] = (0.75, 0.0)

    fig, ax = plt.subplots(figsize=(4.4, 2.45))
    nx.draw_networkx_edges(
        subgraph,
        pos,
        ax=ax,
        edge_color="#777777",
        width=1.15,
        arrows=True,
        arrowsize=12,
        connectionstyle="arc3,rad=0.0",
    )
    ordinary = [n for n in subgraph.nodes if n not in alert]
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        nodelist=ordinary,
        node_color="white",
        edgecolors="#555555",
        linewidths=1.0,
        node_size=1050,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        nodelist=sorted(alert),
        node_color="#d8ecff",
        edgecolors="#1f77b4",
        linewidths=2.0,
        node_size=1250,
        ax=ax,
    )
    nx.draw_networkx_nodes(
        subgraph,
        pos,
        nodelist=sorted(attacked),
        node_color="none",
        edgecolors="#d62728",
        linewidths=2.5,
        node_size=1430,
        ax=ax,
    )
    nx.draw_networkx_labels(subgraph, pos, font_size=8.5, font_weight="bold", ax=ax)
    ax.set_title(f"SWaT attack {attack_id} alert", fontsize=9.5)
    ax.text(
        0.5,
        -0.12,
        "Blue = ASID alert, red outline = attacked tag",
        transform=ax.transAxes,
        ha="center",
        va="top",
        fontsize=8,
    )
    ax.set_xlim(-1.85, 1.35)
    ax.set_ylim(-0.9, 0.9)
    ax.axis("off")
    fig.tight_layout(pad=0.25)
    fig.savefig(out_dir / f"swat_attack{attack_id}_lit101_local_dependency_graph.pdf")
    fig.savefig(out_dir / f"swat_attack{attack_id}_lit101_local_dependency_graph.png", dpi=300)
    plt.close(fig)

    pd.DataFrame(
        [
            {
                "dataset": "SWaT",
                "attack_id": attack_id,
                "attacked_tags": ";".join(sorted(attacked)),
                "alert_nodes": ";".join(sorted(alert)),
                "nearest_distance": row["nearest_distance"],
                "n_alert_nodes": int(row["n_alert_nodes"]),
                "figure": f"swat_attack{attack_id}_lit101_local_dependency_graph.pdf",
                "note": "Readable equation-local dependency graph.",
            }
        ]
    ).to_csv(out_dir / "swat_lit101_showcase_summary.csv", index=False)


def _load_swat_numeric(path: Path) -> tuple[pd.DataFrame, np.ndarray, list[str]]:
    df = pd.read_csv(path)
    columns = [col for col in df.columns if col not in {"Timestamp", "Normal/Attack"}]
    for col in columns:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    labels = (
        df["Normal/Attack"].astype(str).str.lower().isin({"true", "1", "attack", "a"}).to_numpy(dtype=np.int8)
        if "Normal/Attack" in df.columns
        else np.zeros(len(df), dtype=np.int8)
    )
    return df, labels, columns


def plot_swat_lit101_trace(
    train_csv: Path,
    test_csv: Path,
    selected_csv: Path,
    attack_targets_csv: Path,
    out_dir: Path,
    *,
    s: float,
    g: float,
) -> None:
    if not train_csv.exists() or not test_csv.exists() or not selected_csv.exists():
        return
    train_df, _, feature_columns = _load_swat_numeric(train_csv)
    test_df, labels_raw, test_columns = _load_swat_numeric(test_csv)
    if feature_columns != test_columns:
        raise ValueError("SWaT train/test feature columns differ.")
    selected = pd.read_csv(selected_csv)
    row = selected[selected["target"].astype(str) == "LIT101"].iloc[0]
    equation = str(row.get("sympy_format", row.get("equation")))

    train = train_df[feature_columns].to_numpy(dtype=np.float64)
    test = test_df[feature_columns].to_numpy(dtype=np.float64)
    train_current, train_next = train[:-1], train[1:]
    test_current, test_next = test[:-1], test[1:]
    idx = feature_columns.index("LIT101")

    pred_train_delta = evaluate_equation(equation, feature_columns, train_current)
    pred_train_next = reconstruct_next_from_delta(train_current[:, idx], pred_train_delta)
    train_residual = np.abs(train_next[:, idx] - pred_train_next)
    train_residual = np.where(np.isfinite(train_residual), train_residual, 0.0)

    pred_test_delta = evaluate_equation(equation, feature_columns, test_current)
    pred_test_next = reconstruct_next_from_delta(test_current[:, idx], pred_test_delta)
    observed = test_next[:, idx]
    residual = np.abs(observed - pred_test_next)
    residual = np.where(np.isfinite(residual), residual, 0.0)

    params = fit_cusum_params(train_residual, s=float(s), g=float(g))
    cusum, alarm = run_cusum(residual, params)

    attacks = pd.read_csv(attack_targets_csv)
    attack = attacks[(attacks["attack_id"].astype(str) == "3") & (attacks["attacked_tags"].astype(str) == "LIT101")]
    if attack.empty:
        attack = attacks[attacks["attacked_tags"].astype(str) == "LIT101"].sort_values("attack_id").head(1)
    start = int(attack.iloc[0]["start"])
    end = int(attack.iloc[0]["end"])
    crossings = np.flatnonzero((cusum >= params.threshold) & np.r_[True, cusum[:-1] < params.threshold])
    crossings = crossings[(crossings >= start) & (crossings <= end)]
    first_crossing = int(crossings[0]) if crossings.size else None

    pad = 450
    lo = max(0, start - pad)
    hi = min(len(cusum) - 1, end + pad)
    x = np.arange(lo, hi + 1) - start
    fig, axes = plt.subplots(3, 1, figsize=(6.1, 4.25), sharex=True)
    axes[0].plot(x, observed[lo : hi + 1], color="#009E73", lw=1.05, label="measured")
    axes[0].plot(x, pred_test_next[lo : hi + 1], color="#D55E00", lw=1.05, label="predicted")
    axes[0].set_ylabel("LIT101")
    axes[0].legend(loc="upper right", fontsize=8, frameon=True)

    axes[1].plot(x, residual[lo : hi + 1], color="#555555", lw=0.95)
    axes[1].set_ylabel("residual")

    axes[2].plot(x, cusum[lo : hi + 1], color="#0072B2", lw=1.0)
    axes[2].axhline(float(params.threshold), color="#D55E00", lw=0.95, ls="--", label="threshold")
    if first_crossing is not None:
        axes[2].axvline(first_crossing - start, color="#0072B2", lw=0.8, ls=":")
    axes[2].set_ylabel("CUSUM")
    axes[2].set_xlabel("seconds from attack start")
    axes[2].legend(loc="upper right", fontsize=8, frameon=True)

    for ax in axes:
        ax.axvspan(0, end - start, color="#f4a6b7", alpha=0.35)
        ax.grid(alpha=0.18, lw=0.45)
    axes[0].set_title("SWaT attack 3 alert", fontsize=10)
    fig.tight_layout(pad=0.55)
    fig.savefig(out_dir / "swat_attack3_lit101_trace.pdf")
    fig.savefig(out_dir / "swat_attack3_lit101_trace.png", dpi=300)
    plt.close(fig)

    pd.DataFrame(
        [
            {
                "dataset": "SWaT",
                "attack_id": 3,
                "target": "LIT101",
                "equation": equation,
                "attack_start_index": start,
                "attack_end_index": end,
                "threshold": float(params.threshold),
                "max_cusum_in_attack": float(np.max(cusum[start : end + 1])),
                "first_crossing_index": first_crossing,
                "detection_delay_seconds": None if first_crossing is None else int(first_crossing - start),
                "max_threshold_ratio": float(np.max(cusum[start : end + 1]) / max(float(params.threshold), 1e-12)),
                "figure": "swat_attack3_lit101_trace.pdf",
            }
        ]
    ).to_csv(out_dir / "swat_lit101_trace_summary.csv", index=False)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate paper localization figures from saved ASID artifacts.")
    parser.add_argument("--out", default="paper_artifacts/localization")
    parser.add_argument("--swat", default="artifacts/localization/swat_localization_per_attack.csv")
    parser.add_argument("--wadi", default="artifacts/localization/wadi_localization_per_attack.csv")
    parser.add_argument("--hai", default="paper_artifacts/localization_runs/hai_localization_per_attack.csv")
    parser.add_argument("--batadal", default="paper_artifacts/localization_runs/batadal_localization_per_attack.csv")
    parser.add_argument("--swat-graph", default="artifacts/localization/swat_dependency_graph.graphml")
    parser.add_argument("--swat-train-csv", default="data/swat/raw/swat_train.csv")
    parser.add_argument("--swat-test-csv", default="data/swat/raw/swat_test.csv")
    parser.add_argument("--swat-selected", default="paper_artifacts/selected_models/swat/selected_equations.csv")
    parser.add_argument("--swat-attack-targets", default="artifacts/localization/swat_attack_targets.csv")
    parser.add_argument("--swat-s", type=float, default=1.20)
    parser.add_argument("--swat-g", type=float, default=15.0)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "SWaT": Path(args.swat),
        "WADI": Path(args.wadi),
        "HAI": Path(args.hai),
        "BATADAL": Path(args.batadal),
    }
    frames = load_per_attack(paths)
    values, summary = summarize(frames)
    values.to_csv(out_dir / "localization_connected_detected_geco_style_values.csv", index=False)
    summary.to_csv(out_dir / "localization_all4_summary.csv", index=False)
    plot_geco_style(values, out_dir)
    plot_local_swat_graph(Path(args.swat_graph), paths["SWaT"], out_dir)
    plot_swat_lit101_trace(
        Path(args.swat_train_csv),
        Path(args.swat_test_csv),
        Path(args.swat_selected),
        Path(args.swat_attack_targets),
        out_dir,
        s=float(args.swat_s),
        g=float(args.swat_g),
    )

    print("\nGeCo-style localization, omitting undetected and graph-disconnected attacks")
    cols = [
        "dataset",
        "scenarios",
        "detected",
        "connected_detected",
        "detected_disconnected",
        "geco_style_D_eq_0",
        "geco_style_D_le_1",
        "geco_style_D_le_2",
        "geco_style_D_le_3",
        "mean_finite_hops",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.1f}"))

    print("\nConservative end-to-end denominator, all attack scenarios")
    cols = [
        "dataset",
        "scenarios",
        "conservative_D_eq_0",
        "conservative_D_le_1",
        "conservative_D_le_2",
        "conservative_D_le_3",
    ]
    print(summary[cols].to_string(index=False, float_format=lambda x: f"{x:.1f}"))
    print(f"\nWrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
