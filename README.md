# Symbolic ICS Distillation

This repository contains a reproducible pipeline for symbolic distillation of neural anomaly-model predictions on industrial control system time series.

The code supports:

- SWaT/WADI dataset loading and preprocessing configuration
- current-state and sequence model training/export utilities
- distillation-array preparation
- PySR symbolic regression audits
- cross-sensor reporting and compact derived-result artifacts

The repository is intended to be publishable without raw data, trained weights, or large generated artifacts.

## Data Access

Raw SWaT and WADI data are not included. Authorized users should place local copies under:

```text
data/swat/raw/swat_train.csv
data/swat/raw/swat_test.csv
data/wadi/raw/wadi_train.csv
data/wadi/raw/wadi_test.csv
```

SWaT access is restricted; obtain the data from the appropriate dataset provider. Raw data, checkpoints, model exports, distillation arrays, and full PySR run outputs are intentionally ignored by git.

Check configured data paths with:

```bash
python scripts/check_data_paths.py \
  --config configs/experiment/swat_gru.yaml
```

## Environment

Create the conda environment:

```bash
conda env create -f environment.yml
conda activate symbolic-ics
```

If the environment already exists:

```bash
conda env update -f environment.yml --prune
conda activate symbolic-ics
```

Fallback pip setup:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[dev,distill]"
```

## Validation

```bash
python -m compileall src scripts tests
python -m pytest -q
```

## Symbolic Audit

The symbolic audit consumes prepared distillation arrays, not raw data directly. The expected distillation directory is:

```text
artifacts/model_exports/swat/distillation/val20_overlap/
```

If `SENSORS` is omitted, the runner derives the target list from:

```text
distill_target_columns.json
```

This avoids mixing predicted target sensors with input-only variables or actuators.

### Smoke Test

```bash
SENSORS="LIT101 FIT101" \
TARGET_SOURCES="actual_delta mlp_delta" \
NITERATIONS=5 \
TIMEOUT=120 \
SAMPLE_SIZE=200 \
OUT_ROOT=artifacts/symbolic_equations/swat/full_sensor_audit_smoke \
bash scripts/run_full_sensor_audit.sh
```

### Full Audit

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

The main PySR audit is unconstrained over all current SWaT input variables. Process context and GeCo-style references are used only for post-hoc interpretation.

Generated reports:

```text
artifacts/symbolic_equations/swat/full_sensor_audit/run_status.json
artifacts/symbolic_equations/swat/full_sensor_audit/cross_sensor_report.md
artifacts/symbolic_equations/swat/full_sensor_audit/cross_sensor_summary.csv
artifacts/symbolic_equations/swat/full_sensor_audit/linear_sensor_baselines.csv
```

## Curated Result Artifacts

Small anonymized derived summaries are included under:

```text
paper_artifacts/
```

These files support reproducibility checks and paper-table construction without committing raw SWaT data, trained checkpoints, `.npy` arrays, full PySR logs, or local run manifests.

See:

```text
docs/artifacts.md
docs/day4_full_sensor_audit.md
```
