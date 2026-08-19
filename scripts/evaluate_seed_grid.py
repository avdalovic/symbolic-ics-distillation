#!/usr/bin/env python
from __future__ import annotations

import argparse
import hashlib
import importlib.util
import json
import platform
import subprocess
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


HEADLINE_POINTS = {
    "swat": (1.2, 15.0),
    "wadi": (1.2, 25.0),
    "batadal": (1.4, 2.0),
}
GECO_POINTS = {
    "swat": (1.42, 5.98),
    "wadi": (1.32, 9.74),
    "batadal": (1.39, 2.16),
}
EXPAND_STEPS = {
    "swat": 60,
    "wadi": 60,
    "batadal": 1,
}


def import_module(name: str, path: Path):
    spec = importlib.util.spec_from_file_location(name, path)
    if spec is None or spec.loader is None:
        raise RuntimeError(f"Cannot import {path}")
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


V2 = import_module("seed_grid_v2", REPO_ROOT / "scripts" / "generate_paper_artifacts_v2.py")


def rel(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(REPO_ROOT))
    except Exception:
        return str(path)


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False), encoding="utf-8")


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def git_sha() -> str | None:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO_ROOT, text=True).strip()
    except Exception:
        return None


def metric_aliases(df: pd.DataFrame) -> pd.DataFrame:
    out = df.copy()
    if "Prec" in out.columns and "Precision" not in out.columns:
        out["Precision"] = out["Prec"]
    if "Rec" in out.columns and "Recall" not in out.columns:
        out["Recall"] = out["Rec"]
    ordered = [
        "dataset",
        "variant",
        "point_kind",
        "S",
        "G",
        "Precision",
        "Recall",
        "F1",
        "eTaP",
        "eTaR",
        "eTaF1",
        "FPA",
        "Scen",
        "num_monitored",
        "num_sensors",
        "num_actuators",
    ]
    existing = [col for col in ordered if col in out.columns]
    rest = [col for col in out.columns if col not in existing]
    return out[existing + rest]


def row_target(row: Any) -> str:
    return str(getattr(row, "target", row.get("target") if isinstance(row, dict) else ""))


def row_type(row: Any) -> str:
    return str(getattr(row, "variable_type", row.get("variable_type") if isinstance(row, dict) else ""))


def matrix_from_cache(rows: list[Any], cache: dict[str, dict[str, np.ndarray]], split: str) -> np.ndarray:
    cols = [np.asarray(cache[row_target(row)][split], dtype=np.float32) for row in rows]
    if not cols:
        return np.empty((0, 0), dtype=np.float32)
    return np.column_stack(cols).astype(np.float32, copy=False)


def write_sample_index_manifest(run_dir: Path, out_dir: Path) -> Path:
    rows: list[dict[str, Any]] = []
    for sample_path in sorted((run_dir / "pareto_fronts").rglob("sample_indices.npy")):
        indices = np.load(sample_path)
        target = sample_path.parent.name.split("_sensors_delta")[0]
        config = ""
        try:
            parent = sample_path.parent.parent
            if parent.name != "pareto_fronts":
                config = parent.name
        except Exception:
            config = ""
        rows.append(
            {
                "config": config,
                "target": target,
                "sample_indices_path": rel(sample_path),
                "status": "present",
                "sample_size": int(indices.shape[0]),
                "min_index": int(np.min(indices)) if indices.size else None,
                "max_index": int(np.max(indices)) if indices.size else None,
                "first_index": int(indices[0]) if indices.size else None,
                "last_index": int(indices[-1]) if indices.size else None,
                "sample_indices_sha256": file_sha256(sample_path),
            }
        )
    out = out_dir / "sample_index_manifest.csv"
    pd.DataFrame(rows).to_csv(out, index=False)
    return out


def load_selected(run_dir: Path) -> pd.DataFrame:
    path = run_dir / "selected_equations.csv"
    if not path.exists():
        raise FileNotFoundError(path)
    return pd.read_csv(path)


def build_swat_payload(args: argparse.Namespace) -> dict[str, Any]:
    swat = import_module("seed_grid_swat_post", REPO_ROOT / "scripts" / "run_swat_1sec_delta_posthoc_ablation.py")
    arrays = swat.load_arrays(
        argparse.Namespace(
            experiment=args.swat_experiment,
            train_csv=args.train_csv,
            test_csv=args.test_csv,
        )
    )
    selected_path = Path(args.selected_path)
    models = swat.models_from_selected(selected_path, exclude=swat.GECO_EXCLUSIONS)
    models += swat.make_actuator_persistence_models(arrays["feature_columns"], exclude=swat.GECO_EXCLUSIONS)
    cache = swat.residual_cache_for_models(arrays, models)
    train_matrix, test_matrix = swat.stack_residuals(models, cache)
    return {
        "variant": "geco_matched_plus_actuator_persistence",
        "train_matrix": train_matrix.astype(np.float32, copy=False),
        "test_matrices": [test_matrix.astype(np.float32, copy=False)],
        "label_arrays": [arrays["labels"]],
        "counts": {
            "num_monitored": len(models),
            "num_sensors": sum(1 for model in models if model.variable_type == "sensor"),
            "num_actuators": sum(1 for model in models if model.variable_type == "actuator"),
        },
        "data": arrays.get("metadata", {}),
    }


def build_wadi_payload(args: argparse.Namespace) -> dict[str, Any]:
    wadi = import_module("seed_grid_wadi_post", REPO_ROOT / "scripts" / "run_wadi_1sec_delta_posthoc_ablation.py")
    arrays = wadi.FULL.load_wadi_1sec_arrays(argparse.Namespace(train_csv=args.train_csv, test_csv=args.test_csv))
    selected_rows = wadi.load_selected_equations(Path(args.selected_path))
    geco_exclusions, missing = wadi.resolve_exclusions(wadi.WADI_GECO_EXCLUSIONS, arrays["feature_columns"])
    rows, cache = wadi.build_variant_rows(
        arrays,
        selected_rows,
        variant="geco_matched_plus_actuator_persistence",
        geco_exclusions=geco_exclusions,
    )
    return {
        "variant": "geco_matched_plus_actuator_persistence",
        "train_matrix": matrix_from_cache(rows, cache, "train"),
        "test_matrices": [matrix_from_cache(rows, cache, "test")],
        "label_arrays": [arrays["labels"]],
        "counts": {
            "num_monitored": len(rows),
            "num_sensors": sum(1 for row in rows if row_type(row) == "sensor"),
            "num_actuators": sum(1 for row in rows if row_type(row) == "actuator"),
        },
        "data": arrays.get("metadata", {}),
        "missing_geco_exclusions": missing,
    }


def build_batadal_payload(args: argparse.Namespace) -> dict[str, Any]:
    bat = import_module("seed_grid_batadal", REPO_ROOT / "scripts" / "run_batadal_delta_full.py")
    selected_rows = pd.read_csv(Path(args.selected_path)).to_dict("records")
    test_csvs = [tok.strip() for tok in str(args.test_csv).split(",") if tok.strip()]
    if not test_csvs:
        raise ValueError("BATADAL requires at least one --test-csv")
    segments = []
    for test_csv in test_csvs:
        arrays = bat.load_batadal_arrays(argparse.Namespace(train_csv=args.train_csv, test_csv=test_csv))
        rows, cache = bat.build_variant_rows(arrays, selected_rows, "geco_matched_plus_actuator_persistence")
        segments.append((arrays, rows, cache, test_csv))
    arrays0, rows0, cache0, _ = segments[0]
    train_matrix = matrix_from_cache(rows0, cache0, "train")
    row_order = [row_target(row) for row in rows0]
    test_matrices = []
    label_arrays = []
    per_file = []
    for arrays, rows, cache, test_csv in segments:
        if [row_target(row) for row in rows] != row_order:
            raise ValueError("BATADAL monitored row order differs across test files")
        test_matrices.append(matrix_from_cache(rows, cache, "test"))
        label_arrays.append(arrays["labels"])
        per_file.append({"test_csv": test_csv, **arrays.get("metadata", {})})
    return {
        "variant": "combined_14_attacks_geco_matched_plus_actuator_persistence",
        "train_matrix": train_matrix,
        "test_matrices": test_matrices,
        "label_arrays": label_arrays,
        "counts": {
            "num_monitored": len(rows0),
            "num_sensors": sum(1 for row in rows0 if row_type(row) == "sensor"),
            "num_actuators": sum(1 for row in rows0 if row_type(row) == "actuator"),
        },
        "data": {"train_csv": args.train_csv, "test_segments": per_file},
    }


def build_payload(args: argparse.Namespace) -> dict[str, Any]:
    if args.dataset == "swat":
        return build_swat_payload(args)
    if args.dataset == "wadi":
        return build_wadi_payload(args)
    if args.dataset == "batadal":
        return build_batadal_payload(args)
    raise ValueError(args.dataset)


def evaluate_points(payload: dict[str, Any], dataset: str, points: list[tuple[float, float, str]]) -> pd.DataFrame:
    rows = []
    for s, g, name in points:
        grid, _ = V2.evaluate_grid_from_residuals(
            train_matrix=payload["train_matrix"],
            test_matrices=payload["test_matrices"],
            label_arrays=payload["label_arrays"],
            s_values=[float(s)],
            g_values=[float(g)],
            expand_steps=EXPAND_STEPS[dataset],
            counts=payload["counts"],
        )
        row = grid.iloc[0].to_dict()
        row["point_name"] = name
        rows.append(row)
    return pd.DataFrame(rows)


def choose_summary_rows(grid: pd.DataFrame, points: pd.DataFrame) -> pd.DataFrame:
    rows = []
    if not points.empty:
        rows.extend(points.to_dict("records"))
    eligible = grid[grid["FPA"] <= 8].sort_values(["eTaF1", "F1", "Scen"], ascending=False)
    if not eligible.empty:
        row = eligible.iloc[0].to_dict()
        row["point_name"] = "best_fpa_le_8"
        rows.append(row)
    overall = grid.sort_values(["eTaF1", "F1", "Scen"], ascending=False)
    if not overall.empty:
        row = overall.iloc[0].to_dict()
        row["point_name"] = "best_overall"
        rows.append(row)
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Evaluate a seed run with the fast CUSUM grid.")
    parser.add_argument("--dataset", choices=["swat", "wadi", "batadal"], required=True)
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--selected-csv", default=None)
    parser.add_argument("--train-csv", required=True)
    parser.add_argument("--test-csv", required=True, help="For BATADAL, pass comma-separated test CSVs.")
    parser.add_argument("--out-dir", default=None)
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--swat-experiment", default="configs/experiment/swat_mlp_current_val20.yaml")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    run_dir = Path(args.run_dir)
    out_dir = Path(args.out_dir) if args.out_dir else run_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    selected_path = Path(args.selected_csv) if args.selected_csv else run_dir / "selected_equations.csv"
    if not selected_path.exists():
        raise FileNotFoundError(selected_path)
    args.selected_path = str(selected_path)

    started = time.time()
    sample_manifest = write_sample_index_manifest(run_dir, out_dir)
    payload = build_payload(args)
    grid, per_sample_s = V2.evaluate_grid_from_residuals(
        train_matrix=payload["train_matrix"],
        test_matrices=payload["test_matrices"],
        label_arrays=payload["label_arrays"],
        s_values=V2.S_VALUES_EXT,
        g_values=V2.G_VALUES_EXT,
        expand_steps=EXPAND_STEPS[args.dataset],
        counts=payload["counts"],
    )
    grid["dataset"] = args.dataset.upper() if args.dataset != "swat" else "SWaT"
    grid["variant"] = payload["variant"]
    grid["point_kind"] = "grid"
    grid = metric_aliases(grid)
    grid.to_csv(out_dir / "detection_grid_fast.csv", index=False)

    headline_s, headline_g = HEADLINE_POINTS[args.dataset]
    geco_s, geco_g = GECO_POINTS[args.dataset]
    point_rows = evaluate_points(
        payload,
        args.dataset,
        [
            (headline_s, headline_g, "headline_point"),
            (geco_s, geco_g, "geco_point"),
        ],
    )
    point_rows["dataset"] = args.dataset.upper() if args.dataset != "swat" else "SWaT"
    point_rows["variant"] = payload["variant"]
    point_rows["point_kind"] = "operating_point"
    point_rows = metric_aliases(point_rows)

    summary = choose_summary_rows(grid, point_rows)
    summary["dataset"] = args.dataset.upper() if args.dataset != "swat" else "SWaT"
    summary["variant"] = payload["variant"]
    summary = metric_aliases(summary)
    point_rows.to_csv(out_dir / "operating_points.csv", index=False)
    summary.to_csv(out_dir / "seed_summary_rows.csv", index=False)

    run_config = read_json(run_dir / "run_config.json")
    meta = {
        "dataset": args.dataset,
        "seed": args.seed if args.seed is not None else run_config.get("seed"),
        "run_dir": rel(run_dir),
        "out_dir": rel(out_dir),
        "git_sha": git_sha(),
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "wall_seconds": time.time() - started,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "selected_equations": rel(selected_path),
        "selected_equations_sha256": file_sha256(selected_path),
        "sample_index_manifest": rel(sample_manifest),
        "detection_grid": rel(out_dir / "detection_grid_fast.csv"),
        "operating_points": rel(out_dir / "operating_points.csv"),
        "summary_rows": rel(out_dir / "seed_summary_rows.csv"),
        "grid_rows": int(len(grid)),
        "variant": payload["variant"],
        "counts": payload["counts"],
        "headline_point": {"S": headline_s, "G": headline_g},
        "geco_point": {"S": geco_s, "G": geco_g},
        "detection_eval_s_per_sample": per_sample_s,
        "data": payload.get("data", {}),
        "run_config": run_config,
    }
    if "missing_geco_exclusions" in payload:
        meta["missing_geco_exclusions"] = payload["missing_geco_exclusions"]
    write_json(out_dir / "run_meta.json", meta)

    print(json.dumps({"run_dir": rel(run_dir), "grid_rows": len(grid), "summary": summary.to_dict("records")}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
