# Artifact Guide

This document explains how the repository's compact public artifacts relate to generated audit outputs.

## Curated Artifacts

The directory `paper_artifacts/` contains anonymized derived summaries:

- `symbolic_audit_summary.csv`: compact cross-sensor symbolic audit table.
- `linear_baseline_summary.csv`: compact OLS/Ridge diagnostic baseline table.
- `lit101_case_study.csv`: LIT101 rows extracted from the cross-sensor summary.
- `selected_pareto_fronts/`: selected Pareto fronts for case-study and proxy-equation examples.

These files are derived from generated outputs under `artifacts/symbolic_equations/`, but the full generated run directories are not committed.

## Mapping to Generated Files

After regenerating the audit, the source files are:

```text
artifacts/symbolic_equations/swat/full_sensor_audit/cross_sensor_summary.csv
artifacts/symbolic_equations/swat/full_sensor_audit/linear_sensor_baselines.csv
artifacts/symbolic_equations/swat/full_sensor_audit/*/pareto_front_scored.csv
```

The curated `paper_artifacts/` files keep only columns needed for reporting symbolic equations, holdout metrics, local-support labels, and diagnostic linear baselines.

## Regeneration

Run validation first:

```bash
python -m compileall src scripts tests
python -m pytest -q
```

Then run the full audit:

```bash
TARGET_SOURCES="actual_next actual_delta mlp_next mlp_delta" \
NITERATIONS=400 \
TIMEOUT=1800 \
SAMPLE_SIZE=all \
OPERATOR_SET=restricted \
MODE=unconstrained \
EVAL_SPLIT=temporal \
EVAL_FRAC=0.2 \
RESUME=1 \
RUN_LINEAR_BASELINES=1 \
RUN_REPORT=1 \
OUT_ROOT=artifacts/symbolic_equations/swat/full_sensor_audit \
bash scripts/run_full_sensor_audit.sh
```

Inspect completion status:

```bash
cat artifacts/symbolic_equations/swat/full_sensor_audit/run_status.json
```

## Interpretation

`symbolic_audit_summary.csv` contains one row per sensor and target source. Useful columns include:

- `simplest_nonconstant_equation`
- `score_selected_equation`
- `lowest_loss_equation`
- `best_local_physical_equation`
- `best_local_holdout_r2`
- `score_selected_holdout_r2`
- `lowest_loss_holdout_r2`
- `notes`

PySR score-selected equations are not automatically treated as final physical explanations. Candidate equations should be evaluated using local support, holdout performance, simplicity, and physical plausibility. Equations using off-process variables are flagged as possible proxy fits.

## Exclusions

The following are intentionally excluded from git:

- raw SWaT/WADI data
- trained checkpoints and model weights
- `.npy` and `.npz` distillation arrays
- full PySR output directories
- run logs and `nohup` outputs
- local manifests containing machine-specific metadata
