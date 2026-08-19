#!/usr/bin/env python3
from __future__ import annotations

import csv
import json
import math
import re
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "paper_artifacts" / "timing_final_seed42"
LOGS = ROOT / "logs"


DATASETS = {
    "batadal": {
        "log": LOGS / "batadal_seed42_timev.log",
        "run_id": "paper_artifacts/timing_validation/batadal_seed42_timev",
        "workers": 20,
        "cgroup": False,
        "note": "BATADAL measured earlier with /usr/bin/time -v; no cgroup pre/post was captured for that validation run.",
    },
    "swat": {
        "log": LOGS / "swat_seed42_timev.log",
        "run_id": "paper_artifacts/timing_final_seed42/swat_seed42_timev",
        "workers": 20,
        "cgroup": True,
        "note": "",
    },
    "wadi": {
        "log": LOGS / "wadi_seed42_timev.log",
        "run_id": "paper_artifacts/timing_final_seed42/wadi_seed42_timev",
        "workers": 8,
        "cgroup": True,
        "note": "",
    },
    "hai": {
        "log": LOGS / "hai_seed42_timev.log",
        "run_id": "paper_artifacts/timing_final_seed42/hai_seed42_timev",
        "workers": 40,
        "cgroup": True,
        "note": "",
    },
}


OLD_NUMBERS = [
    {
        "description": "Original manuscript summed/estimated ASID training times",
        "values": "SWaT 4.8h, WADI 13.9h, BATADAL 40m",
        "status": "retired",
        "reason": "These were not /usr/bin/time CPU-work measurements and are superseded for compute reporting.",
    },
    {
        "description": "Per-target metadata wall-time sums",
        "values": "SWaT 2.0h, WADI 7.8h, HAI 20.6h, BATADAL 2.2h",
        "status": "retired_as_cpu_work",
        "reason": "They are useful internal per-target fit wall-time sums, but BATADAL /usr/bin/time shows they undercount process CPU work.",
    },
]


def parse_elapsed_seconds(text: str) -> float:
    parts = text.strip().split(":")
    if len(parts) == 2:
        minutes, seconds = parts
        return float(minutes) * 60.0 + float(seconds)
    if len(parts) == 3:
        hours, minutes, seconds = parts
        return float(hours) * 3600.0 + float(minutes) * 60.0 + float(seconds)
    raise ValueError(f"Unrecognized elapsed time: {text!r}")


def read_timev(path: Path) -> dict[str, float | str]:
    text = path.read_text(errors="replace")
    patterns = {
        "user_s": r"User time \(seconds\):\s*([0-9.]+)",
        "sys_s": r"System time \(seconds\):\s*([0-9.]+)",
        "elapsed_text": r"Elapsed \(wall clock\) time \(h:mm:ss or m:ss\):\s*([^\n]+)",
        "cpu_percent_text": r"Percent of CPU this job got:\s*([^\n]+)",
        "max_rss_kb": r"Maximum resident set size \(kbytes\):\s*([0-9]+)",
        "exit_status": r"Exit status:\s*([0-9]+)",
    }
    out: dict[str, float | str] = {}
    for key, pattern in patterns.items():
        match = re.search(pattern, text)
        if not match:
            raise ValueError(f"Missing {key} in {path}")
        out[key] = match.group(1).strip()
    out["user_s"] = float(out["user_s"])
    out["sys_s"] = float(out["sys_s"])
    out["elapsed_s"] = parse_elapsed_seconds(str(out["elapsed_text"]))
    out["max_rss_kb"] = int(str(out["max_rss_kb"]))
    out["exit_status"] = int(str(out["exit_status"]))
    return out


def read_cpu_stat(path: Path) -> dict[str, int]:
    values: dict[str, int] = {}
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) == 2:
            values[parts[0]] = int(parts[1])
    return values


def cgroup_core_hours(ds: str) -> float | None:
    pre = OUT / f"pre_{ds}_cpu.stat"
    post = OUT / f"post_{ds}_cpu.stat"
    if not pre.exists() or not post.exists():
        return None
    before = read_cpu_stat(pre)
    after = read_cpu_stat(post)
    if "usage_usec" not in before or "usage_usec" not in after:
        return None
    return (after["usage_usec"] - before["usage_usec"]) / 1_000_000.0 / 3600.0


def build_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    for ds, meta in DATASETS.items():
        log = Path(meta["log"])
        if not log.exists():
            continue
        parsed = read_timev(log)
        cpu_s = float(parsed["user_s"]) + float(parsed["sys_s"])
        core_h = cpu_s / 3600.0
        wall_min = float(parsed["elapsed_s"]) / 60.0
        cgroup_h = cgroup_core_hours(ds) if meta["cgroup"] else None
        if cgroup_h is None or core_h == 0:
            cgroup_delta_pct = None
            cgroup_check = "not_available"
        else:
            cgroup_delta_pct = abs(cgroup_h - core_h) / core_h * 100.0
            cgroup_check = "pass" if cgroup_delta_pct <= 5.0 else "flag"
        rows.append(
            {
                "dataset": ds.upper() if ds != "swat" else "SWaT",
                "run_id": meta["run_id"],
                "seed": 42,
                "wall_minutes": wall_min,
                "core_hours_timev": core_h,
                "user_seconds": parsed["user_s"],
                "system_seconds": parsed["sys_s"],
                "average_cores_used": core_h / (wall_min / 60.0) if wall_min > 0 else math.nan,
                "peak_memory_gb": int(parsed["max_rss_kb"]) / 1024.0 / 1024.0,
                "workers": meta["workers"],
                "cgroup_core_hours": "" if cgroup_h is None else cgroup_h,
                "cgroup_delta_percent": "" if cgroup_delta_pct is None else cgroup_delta_pct,
                "cgroup_check": cgroup_check,
                "exit_status": parsed["exit_status"],
                "log_path": str(log),
                "note": meta["note"],
            }
        )
    return rows


def write_outputs(rows: list[dict[str, object]]) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    fieldnames = [
        "dataset",
        "run_id",
        "seed",
        "wall_minutes",
        "core_hours_timev",
        "user_seconds",
        "system_seconds",
        "average_cores_used",
        "peak_memory_gb",
        "workers",
        "cgroup_core_hours",
        "cgroup_delta_percent",
        "cgroup_check",
        "exit_status",
        "log_path",
        "note",
    ]
    with (OUT / "timing_manifest.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)

    payload = {
        "timing_boundary": "/usr/bin/time -v around the discovery command. CPU work is (user time + system time) / 3600.",
        "live_cost_boundary": "Existing sub-millisecond live costs cover equation evaluation and CUSUM updates on numeric arrays, not IPAL/raw-file parsing or network ingestion.",
        "rows": rows,
        "retired_numbers": OLD_NUMBERS,
    }
    (OUT / "timing_manifest.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    with (OUT / "retired_timing_numbers.csv").open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=["description", "values", "status", "reason"])
        writer.writeheader()
        writer.writerows(OLD_NUMBERS)

    ordered = ["SWaT", "WADI", "HAI", "BATADAL"]
    by_dataset = {str(row["dataset"]): row for row in rows}
    lines = [
        "# Morning Report: Final Timing Seed 42",
        "",
        "Timing boundary: `/usr/bin/time -v` around each discovery command. CPU work is `(user time + system time) / 3600`.",
        "",
        "| Dataset | Wall min | Core-hours | Avg cores | Peak GB | Workers | cgroup check |",
        "|---|---:|---:|---:|---:|---:|---|",
    ]
    for dataset in ordered:
        row = by_dataset.get(dataset)
        if not row:
            continue
        lines.append(
            f"| {dataset} | {float(row['wall_minutes']):.1f} | {float(row['core_hours_timev']):.2f} | "
            f"{float(row['average_cores_used']):.1f} | {float(row['peak_memory_gb']):.2f} | "
            f"{int(row['workers'])} | {row['cgroup_check']} |"
        )
    lines.extend(
        [
            "",
            "BATADAL was measured earlier in `logs/batadal_seed42_timev.log`; it has no cgroup before/after pair.",
            "",
            "Retired numbers:",
        ]
    )
    for item in OLD_NUMBERS:
        lines.append(f"- {item['values']}: {item['reason']}")
    lines.extend(
        [
            "",
            "Live-cost note: the current sub-millisecond live numbers cover equation evaluation and CUSUM updates on numeric arrays, not file parsing or telemetry ingestion.",
        ]
    )
    (OUT / "MORNING_REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> int:
    rows = build_rows()
    write_outputs(rows)
    print(f"Wrote {OUT / 'timing_manifest.csv'}")
    print(f"Wrote {OUT / 'MORNING_REPORT.md'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
