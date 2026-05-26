#!/usr/bin/env python
"""Generate GeCo-style detection trace figures from frozen ASID-ICS artifacts."""

from __future__ import annotations

import argparse
import importlib.util
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable

import matplotlib

matplotlib.use("pdf")

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Patch
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from ics_symbolic_distill.data.ics_metadata import AttackWindow, get_attack_windows
from ics_symbolic_distill.detection import evaluate_equation, fit_cusum_params, run_cusum
from ics_symbolic_distill.detection.swat1s_delta_sampling import reconstruct_next_from_delta


OUT_DIR = REPO_ROOT / "paper_artifacts" / "final" / "figures"


@dataclass(frozen=True)
class TracePayload:
    dataset: str
    target: str
    x: np.ndarray
    actual: np.ndarray
    predicted: np.ndarray
    residual: np.ndarray
    cusum: np.ndarray
    threshold: float
    windows: list[tuple[float, float, tuple[str, ...]]]
    residual_clip: float | None
    x_label: str
    title_label: str


def import_module(name: str, path: Path) -> Any:
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def selected_row(path: Path, target: str) -> pd.Series:
    df = pd.read_csv(path)
    rows = df[df["target"].astype(str) == target]
    if rows.empty:
        raise ValueError(f"No selected equation for {target} in {path}")
    return rows.iloc[0]


def inclusive_window_indices(start: int, end: int, n: int) -> tuple[int, int]:
    lo = max(0, int(start))
    hi = min(int(end), n - 1)
    if hi < lo:
        raise ValueError(f"Empty trace window after clipping: {start=} {end=} {n=}")
    return lo, hi


def overlapping_windows(
    windows: Iterable[AttackWindow | Any],
    *,
    lo: int,
    hi: int,
    x0: int,
    scale: float,
) -> list[tuple[float, float, tuple[str, ...]]]:
    out = []
    for window in windows:
        start = int(window.start)
        end = int(window.end)
        if end < lo or start > hi:
            continue
        tags = tuple(str(tag) for tag in getattr(window, "affected_tags", ()))
        out.append(((max(start, lo) - x0) / scale, (min(end, hi) - x0) / scale, tags))
    return out


def sensor_prediction(arrays: dict[str, Any], target: str, equation: str, split: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    feature_columns = arrays["feature_columns"]
    idx = feature_columns.index(target)
    if split == "train":
        current = arrays["train_current"]
        nxt = arrays["train_next"]
    elif split == "test":
        current = arrays["test_current"]
        nxt = arrays["test_next"]
    else:
        raise ValueError(split)
    pred_delta = evaluate_equation(equation, feature_columns, current).astype(np.float64)
    predicted = reconstruct_next_from_delta(current[:, idx], pred_delta).astype(np.float64)
    actual = nxt[:, idx].astype(np.float64)
    residual = np.abs(actual - predicted)
    residual = np.where(np.isfinite(residual), residual, 0.0)
    return actual, predicted, residual


def build_swat_lit101_trace() -> TracePayload:
    post = import_module("trace_swat_posthoc", REPO_ROOT / "scripts" / "run_swat_1sec_delta_posthoc_ablation.py")
    arrays = post.load_arrays(
        argparse.Namespace(
            experiment="configs/experiment/swat_mlp_current_val20.yaml",
            train_csv=None,
            test_csv=None,
        )
    )
    row = selected_row(REPO_ROOT / "artifacts" / "swat_1sec" / "delta_full" / "selected_equations.csv", "LIT101")
    equation = str(row["sympy_format"])
    train_actual, train_pred, train_residual = sensor_prediction(arrays, "LIT101", equation, "train")
    test_actual, test_pred, test_residual = sensor_prediction(arrays, "LIT101", equation, "test")
    del train_actual, train_pred
    params = fit_cusum_params(train_residual, s=1.42, g=15.0)
    cusum, _ = run_cusum(test_residual, params)

    attack = get_attack_windows("SWAT")[2]
    # Show the two immediately preceding attacks plus the LIT101 manipulation,
    # matching the compact two-hour visual rhythm of GeCo's SWaT trace.
    lo, hi = inclusive_window_indices(attack.start - 3600, attack.start + 4200, len(test_residual))
    x = (np.arange(lo, hi + 1) - lo) / 3600.0
    windows = overlapping_windows(get_attack_windows("SWAT"), lo=lo, hi=hi, x0=lo, scale=3600.0)
    return TracePayload(
        dataset="SWaT",
        target="LIT101",
        x=x,
        actual=test_actual[lo : hi + 1],
        predicted=test_pred[lo : hi + 1],
        residual=test_residual[lo : hi + 1],
        cusum=cusum[lo : hi + 1],
        threshold=float(params.threshold),
        windows=windows,
        residual_clip=10.0,
        x_label="time [h]",
        title_label="SWaT: LIT101",
    )


def build_batadal_lt1_trace() -> TracePayload:
    bat = import_module("trace_batadal", REPO_ROOT / "scripts" / "run_batadal_delta_full.py")
    arrays = bat.load_batadal_arrays(
        argparse.Namespace(
            train_csv="data/batadal/processed/train.csv",
            test_csv="data/batadal/processed/test_dataset04.csv",
        )
    )
    row = selected_row(REPO_ROOT / "artifacts" / "batadal" / "delta_full" / "selected_equations.csv", "L_T1")
    equation = str(row["sympy_format"])
    train_actual, train_pred, train_residual = sensor_prediction(arrays, "L_T1", equation, "train")
    test_actual, test_pred, test_residual = sensor_prediction(arrays, "L_T1", equation, "test")
    del train_actual, train_pred
    params = fit_cusum_params(train_residual, s=1.39, g=5.0)
    cusum, _ = run_cusum(test_residual, params)

    windows_src = arrays["attack_windows"]
    attack2 = windows_src[1]
    attack5 = windows_src[4]
    # Include one preceding non-L_T1 attack, both L_T1 attacks, and one following
    # non-L_T1 attack to show localization rather than just detection.
    lo, hi = inclusive_window_indices(attack2.start - 48, attack5.end + 48, len(test_residual))
    x = (np.arange(lo, hi + 1) - lo).astype(float)
    windows = overlapping_windows(windows_src, lo=lo, hi=hi, x0=lo, scale=1.0)
    return TracePayload(
        dataset="BATADAL",
        target="L_T1",
        x=x,
        actual=test_actual[lo : hi + 1],
        predicted=test_pred[lo : hi + 1],
        residual=test_residual[lo : hi + 1],
        cusum=cusum[lo : hi + 1],
        threshold=float(params.threshold),
        windows=windows,
        residual_clip=None,
        x_label="time [h]",
        title_label="BATADAL: L_T1",
    )


def shade_attacks(ax: plt.Axes, payload: TracePayload) -> None:
    for start, end, tags in payload.windows:
        if payload.target in tags:
            ax.axvspan(start, end, color="#f04b5f", alpha=0.22, lw=0, zorder=0)
        else:
            ax.axvspan(start, end, color="#f0c45c", alpha=0.26, lw=0, zorder=0)


def style_axis(ax: plt.Axes) -> None:
    ax.grid(False)
    ax.tick_params(axis="both", labelsize=7, pad=1.5, width=0.7, length=2.5)
    for spine in ax.spines.values():
        spine.set_linewidth(0.75)
        spine.set_color("black")


def plot_trace_column(axes: list[plt.Axes], payload: TracePayload, *, show_ylabel: bool = True) -> None:
    ax0, ax1, ax2 = axes
    for ax in axes:
        shade_attacks(ax, payload)
        style_axis(ax)

    ax0.plot(payload.x, payload.actual, color="#0b5d2a", lw=0.8, label="actual")
    ax0.plot(payload.x, payload.predicted, color="#c9253d", lw=0.8, ls="--", label="predicted")

    residual = payload.residual
    if payload.residual_clip is not None:
        residual = np.minimum(residual, payload.residual_clip)
        ylabel = "|err|"
        ax1.set_ylim(0.0, payload.residual_clip * 1.08)
    else:
        ylabel = "|err|"
        ymax = max(float(np.nanpercentile(residual, 99.5)) * 1.35, float(np.nanmax(residual)) * 0.7, 1e-6)
        ax1.set_ylim(0.0, ymax)
    ax1.fill_between(payload.x, 0.0, residual, color="#8c92c9", alpha=0.55, lw=0)
    ax1.plot(payload.x, residual, color="#333333", lw=0.35, alpha=0.7)

    ax2.plot(payload.x, payload.cusum, color="#008b8b", lw=0.8)
    ax2.axhline(payload.threshold, color="#d62728", lw=0.8, ls="--")
    ax2.set_ylim(0.0, max(float(np.nanmax(payload.cusum)) * 1.12, payload.threshold * 1.2, 1e-6))

    if show_ylabel:
        ax0.set_ylabel(payload.target, fontsize=7)
        ax1.set_ylabel(ylabel, fontsize=7)
        ax2.set_ylabel("CUSUM", fontsize=7)
    else:
        ax0.set_ylabel("")
        ax1.set_ylabel("")
        ax2.set_ylabel("")
    ax0.tick_params(labelbottom=False)
    ax1.tick_params(labelbottom=False)
    ax2.set_xlabel(payload.x_label, fontsize=7)
    ax0.set_xlim(float(payload.x[0]), float(payload.x[-1]))
    ax1.set_xlim(float(payload.x[0]), float(payload.x[-1]))
    ax2.set_xlim(float(payload.x[0]), float(payload.x[-1]))


def legend_handles() -> list[Any]:
    return [
        Patch(facecolor="#f04b5f", alpha=0.22, label="sensor attacked"),
        Patch(facecolor="#f0c45c", alpha=0.26, label="other attack"),
        Line2D([0], [0], color="#0b5d2a", lw=0.9, label="original"),
        Line2D([0], [0], color="#c9253d", lw=0.9, ls="--", label="prediction"),
        Patch(facecolor="#8c92c9", alpha=0.55, label="residue"),
        Line2D([0], [0], color="#008b8b", lw=0.9, label="cusum"),
        Line2D([0], [0], color="#d62728", lw=0.9, ls="--", label="alert threshold"),
    ]


def save_single_trace(payload: TracePayload, path: Path) -> None:
    fig, axes = plt.subplots(3, 1, figsize=(3.35, 2.78), sharex=True, constrained_layout=False)
    plot_trace_column(list(axes), payload, show_ylabel=True)
    fig.subplots_adjust(left=0.15, right=0.985, top=0.84, bottom=0.13, hspace=0.10)
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=5.6,
        handlelength=1.05,
        columnspacing=0.55,
        labelspacing=0.35,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def save_combined_trace(swat: TracePayload, batadal: TracePayload, path: Path) -> None:
    fig = plt.figure(figsize=(7.2, 2.9), constrained_layout=True)
    subfigs = fig.subfigures(1, 2, wspace=0.06)
    for subfig, payload in zip(subfigs, [swat, batadal]):
        axes = subfig.subplots(3, 1, sharex=True)
        plot_trace_column(list(axes), payload, show_ylabel=True)
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.5, 1.055),
        ncol=7,
        frameon=False,
        fontsize=6.2,
        handlelength=1.35,
        columnspacing=0.72,
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, bbox_inches="tight", dpi=300)
    plt.close(fig)


def write_trace_audit(swat: TracePayload, batadal: TracePayload, out_dir: Path) -> Path:
    rows = []
    for payload in [swat, batadal]:
        for start, end, tags in payload.windows:
            rows.append(
                {
                    "dataset": payload.dataset,
                    "target": payload.target,
                    "window_start_plot_time": start,
                    "window_end_plot_time": end,
                    "affected_tags": ",".join(tags),
                    "target_attacked": payload.target in tags,
                    "max_cusum_in_window": float(np.max(payload.cusum[(payload.x >= start) & (payload.x <= end)]))
                    if np.any((payload.x >= start) & (payload.x <= end))
                    else 0.0,
                    "threshold": float(payload.threshold),
                }
            )
    path = out_dir / "detection_trace_localization_audit.csv"
    pd.DataFrame(rows).to_csv(path, index=False)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description="Generate GeCo-style ASID-ICS detection trace figures.")
    parser.add_argument("--out-dir", default=str(OUT_DIR))
    args = parser.parse_args()

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    out_dir = Path(args.out_dir)
    swat = build_swat_lit101_trace()
    batadal = build_batadal_lt1_trace()

    outputs = [
        out_dir / "detection_trace_swat.pdf",
        out_dir / "detection_trace_batadal.pdf",
        out_dir / "detection_trace_geco_style_localization.pdf",
        out_dir / "detection_trace_combined.pdf",
    ]
    save_single_trace(swat, outputs[0])
    save_single_trace(batadal, outputs[1])
    save_combined_trace(swat, batadal, outputs[2])
    save_combined_trace(swat, batadal, outputs[3])
    audit = write_trace_audit(swat, batadal, out_dir)

    for path in [*outputs, audit]:
        print(f"Saved: {path}")


if __name__ == "__main__":
    main()
