#!/usr/bin/env python
from __future__ import annotations

import argparse
import json
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = REPO_ROOT / "src"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from evaluate_hai_2x2 import residual_cache_for_rows  # noqa: E402
from evaluate_hai_geco_ignore_posthoc import build_fast_cusum, evaluate_variant_fast, load_run_hai  # noqa: E402


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else {}


def write_json(path: Path, payload: dict[str, Any]) -> None:
    path.write_text(json.dumps(payload, indent=2, allow_nan=False), encoding="utf-8")


def elapsed_from_started(started_at: str | None) -> float | None:
    if not started_at:
        return None
    try:
        started = datetime.strptime(started_at, "%Y-%m-%dT%H:%M:%S")
    except ValueError:
        return None
    return max(0.0, time.time() - time.mktime(started.timetuple()))


def main() -> int:
    parser = argparse.ArgumentParser(description="Fast-finalize a completed HAI run from selected equations.")
    parser.add_argument("--run-dir", default="artifacts/experiments/hai_division_seed0")
    parser.add_argument("--data-dir", default="data/hai/ipal")
    parser.add_argument("--analysis-dir", default="artifacts/experiments/hai_division_seed0/analysis")
    parser.add_argument("--variant", default="seed0_division_fast_finalized")
    parser.add_argument("--geco-model", default=None)
    parser.add_argument("--parent-terminated", action="store_true")
    args = parser.parse_args()

    run_dir = REPO_ROOT / args.run_dir
    analysis_dir = REPO_ROOT / args.analysis_dir
    analysis_dir.mkdir(parents=True, exist_ok=True)
    selected_path = run_dir / "selected_equations.csv"
    if not selected_path.exists():
        raise FileNotFoundError(selected_path)
    selected_rows = pd.read_csv(selected_path).to_dict(orient="records")

    run_hai = load_run_hai()
    arrays = run_hai.load_hai_arrays(argparse.Namespace(data_dir=str(REPO_ROOT / args.data_dir), sample_size=None))
    residual_cache = residual_cache_for_rows(run_hai, arrays, selected_rows)
    lib = build_fast_cusum(analysis_dir)
    geco_s, geco_g, _ = run_hai.parse_geco_point(args.geco_model)

    grid_path = analysis_dir / "G_div_R_all_grid.csv"
    grid_source = grid_path
    if grid_path.exists():
        grid = pd.read_csv(grid_path).copy()
        grid["variant"] = args.variant
        _, alarm_cache = evaluate_variant_fast(
            run_hai,
            lib,
            arrays,
            selected_rows,
            residual_cache,
            [(float(geco_s), float(geco_g), "geco_operating_point")],
            args.variant,
        )
    else:
        points = run_hai.make_detection_points(float(geco_s), float(geco_g), smoke=False)
        grid, alarm_cache = evaluate_variant_fast(run_hai, lib, arrays, selected_rows, residual_cache, points, args.variant)
        grid_source = run_dir / "detection_grid.csv"
    grid.to_csv(run_dir / "detection_grid.csv", index=False)
    geco_row = grid[grid["point_kind"].astype(str) == "geco_operating_point"].copy()
    geco_row.to_csv(run_dir / "geco_operating_point.csv", index=False)

    system_alarm = alarm_cache[(float(geco_s), float(geco_g))]
    run_hai.write_per_attack(arrays, system_alarm, run_dir / "per_attack.csv")
    run_hai.write_per_file_metrics(arrays, system_alarm, run_dir / "per_file_metrics.csv")
    run_hai.write_residual_stats(selected_rows, residual_cache, run_dir / "per_sensor_residual_stats.csv")

    metadata_path = run_dir / "run_metadata.json"
    metadata = read_json(metadata_path)
    finished_at = time.strftime("%Y-%m-%dT%H:%M:%S")
    metadata["finished_at"] = finished_at
    metadata["wall_seconds"] = elapsed_from_started(metadata.get("started_at"))
    metadata["symbolic_selected_count"] = sum(str(row.get("variable_type")) == "sensor" for row in selected_rows)
    metadata["persistence_selected_count"] = sum(str(row.get("variable_type")) == "persistence" for row in selected_rows)
    metadata["excluded_symbolic_count"] = int(len(pd.read_csv(run_dir / "exclusion_reasons.csv"))) if (run_dir / "exclusion_reasons.csv").exists() else None
    metadata["fast_finalized"] = {
        "at": finished_at,
        "script": "scripts/finalize_hai_run_fast.py",
        "variant": args.variant,
        "grid_source": str(grid_source.relative_to(REPO_ROOT)),
        "parent_evaluator_terminated": bool(args.parent_terminated),
    }
    write_json(metadata_path, metadata)
    print(
        json.dumps(
            {
                "run_dir": str(run_dir.relative_to(REPO_ROOT)),
                "grid_rows": int(len(grid)),
                "geco_rows": int(len(geco_row)),
                "selected": len(selected_rows),
                "symbolic_selected": metadata["symbolic_selected_count"],
                "persistence_selected": metadata["persistence_selected_count"],
                "excluded_symbolic": metadata["excluded_symbolic_count"],
            },
            indent=2,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
