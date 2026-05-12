# Day 4 Full Sensor Audit

This audit checks whether symbolic regression can recover simple, physically interpretable structure from current-state MLP exports on SWaT.

## Goal

The audit runs PySR for every target sensor predicted by the MLP. The target list is read from:

```text
distill_target_columns.json
```

Do not derive this list from the 51 input columns; that list includes actuators and input-only variables that are not model prediction targets.

For each sensor, it fits four target sources:

- `actual_next`
- `actual_delta`
- `mlp_next`
- `mlp_delta`

The main PySR setting is unconstrained over all 51 current raw SWaT features. Process-local feature sets and GeCo references are used only after the fact to label equations as local, partially local, off-process, or unknown-support.

## Why Four Targets

`actual_next` and `actual_delta` show what is recoverable from the exported validation trajectory itself. `mlp_next` and `mlp_delta` show what the trained MLP predicts. Comparing actual targets with MLP targets helps distinguish physical structure in the data from structure learned by the model.

## Default Operators

The default operator set is `restricted`:

```text
+ - * /
```

No square, abs, or other rich unary operators are used by default. This keeps the first audit focused on simple low-complexity relationships before allowing more expressive equations.

## Equation Selection

The PySR score-selected equation is not automatically trusted as the final physical equation. Reports distinguish:

- lowest-loss equation
- PySR score-selected equation
- simplest nonconstant equation
- best local physical equation

Final candidates should be judged by local support, holdout performance, simplicity, and physical plausibility. Off-process features are flagged as possible proxy fits.

## Launch

Use an environment with the repo installed, then run:

```bash
NITERATIONS=400 TIMEOUT=1800 SAMPLE_SIZE=all bash scripts/run_full_sensor_audit.sh
```

By default, `scripts/run_full_sensor_audit.sh` derives `SENSORS` from `distill_target_columns.json`. You can inspect that list with:

```bash
python scripts/list_target_sensors.py \
  --distill-dir artifacts/model_exports/swat/distillation/val20_overlap
```

To run a smaller smoke test:

```bash
SENSORS="LIT101 FIT101" \
TARGET_SOURCES="actual_delta mlp_delta" \
NITERATIONS=5 \
TIMEOUT=120 \
SAMPLE_SIZE=200 \
OUT_ROOT=artifacts/symbolic_equations/swat/full_sensor_audit_smoke \
bash scripts/run_full_sensor_audit.sh
```

## Monitor

```bash
tail -f artifacts/symbolic_equations/swat/full_sensor_audit/nohup.out
tail -f artifacts/symbolic_equations/swat/full_sensor_audit/logs/LIT101_actual_delta_restricted_unconstrained.log
```

## Outputs

Outputs are written under:

```text
artifacts/symbolic_equations/swat/full_sensor_audit/
```

Key files:

- `run_manifest.json`
- `run_status.json`
- per-run `pareto_front_scored.csv`
- per-run `metadata.json`
- `linear_sensor_baselines.csv`
- `cross_sensor_summary.csv`
- `cross_sensor_report.md`
- optional all-target memo under `docs/day4_all_target_results_memo.md` after the full audit has completed and been reviewed

These generated outputs are intentionally ignored by git.

## Data and Artifacts

SWaT data is access restricted and must be obtained from iTrust by authorized users. Raw data, checkpoints, distillation arrays, PySR outputs, logs, and generated reports are not committed. Public reference artifacts should be included only when their license allows it.
