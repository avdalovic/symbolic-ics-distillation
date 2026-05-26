# Artifact Guide

This file maps the paper claims to the public artifact files. It is intentionally
short; the large raw datasets and generated run logs are not part of the public
repository.

## Public Result Snapshot

The `results/` directory contains compact frozen outputs for each dataset:

```text
results/{swat,wadi,batadal}/selected_equations.csv
results/{swat,wadi,batadal}/detection_grid.csv
results/{swat,wadi,batadal}/per_sensor_residual_stats.csv
results/{swat,wadi,batadal}/per_attack.csv
results/{swat,wadi,batadal}/pareto_fronts/*/pareto_front_scored.csv
```

These files let reviewers inspect the selected symbolic equations, the Pareto
fronts returned by PySR, the per-sensor CUSUM/alarm statistics, and the
per-attack coverage categories without access to restricted raw data.

## Paper Tables and Figures

The current paper-ready artifacts are under `paper_artifacts/final_v2/`:

- `main_table_one_row.csv` and `table_main_one_row.tex`: main detection table.
- `geco_op_comparison.csv`: ASID-ICS evaluated at GeCo's published S/G values.
- `extended_grid_{swat,wadi,batadal}.csv`: full S/G sensitivity grids.
- `extended_grid_summary.csv`: best low-FPA/zero-FPA rows and contour summary.
- `heatmap_combined.pdf`: three-dataset hyperparameter sensitivity figure.
- `computational_performance.csv` and `table_computational_performance.tex`.

The older `paper_artifacts/final/` directory is retained for traceability, but
`paper_artifacts/final_v2/` is the current paper export.

## Configuration Summary

Reviewer-facing configuration summaries are in `configs/asid_ics/`:

```text
configs/asid_ics/swat.json
configs/asid_ics/wadi.json
configs/asid_ics/batadal.json
```

They record the operator grammar, target type, sample size, quality-filter
criteria, CUSUM operating point, and monitored-variable counts.

## Reproduction Levels

1. Inspect paper outputs without raw data:

```bash
python - <<'PY'
from pathlib import Path
required = [
    "paper_artifacts/final_v2/main_table_one_row.csv",
    "paper_artifacts/final_v2/extended_grid_summary.csv",
    "results/swat/selected_equations.csv",
    "results/wadi/selected_equations.csv",
    "results/batadal/selected_equations.csv",
]
for path in required:
    p = Path(path)
    print(f"{path}: {'ok' if p.exists() else 'missing'}")
PY
```

2. Regenerate paper tables from frozen local run outputs:

```bash
python scripts/generate_paper_artifacts_v2.py
```

This requires authorized SWaT/WADI data and the local ignored `artifacts/`
directories. It does not rerun PySR.

3. Rerun symbolic discovery:

```bash
python scripts/run_wadi_1sec_delta_full.py --out artifacts/wadi_1sec/delta_full
python scripts/run_batadal_delta_full.py --out artifacts/batadal/delta_full
```

SWaT one-second delta discovery and posthoc scripts:

```text
scripts/run_swat_1sec_delta_local_diagnostic.py
scripts/run_swat_1sec_delta_posthoc_ablation.py
scripts/check_swat_no_holdout_quality_gate.py
```

## No Test-Set Training

ASID-ICS uses only benign training data for symbolic discovery, quality
filtering, actuator persistence, and CUSUM calibration. Attack labels are used
only to score alarms and to populate the sensitivity grids. The full S/G grids
are included so reviewers can verify that the reported operating points are not
isolated undocumented settings.
