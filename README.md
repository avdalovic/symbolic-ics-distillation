# Symbolic ICS Distillation

This repository contains a clean, reproducible codebase for training deep-learning anomaly models on industrial control system time-series data and exporting neural predictions for later symbolic distillation.

The first supported path is a SWaT GRU forecasting model with the same preprocessing, row sampling, train/validation split, train-only normalization, column ordering, and trajectory target indexing as the source project. In the symbolic distillation workflow, these trained neural models act as teacher models for symbolic regressors.

## Layout

```text
configs/                  YAML configs for datasets, models, training, export, evaluation
src/ics_symbolic_distill/  Python package
scripts/                  CLI entry points
artifacts/                Local outputs ignored by git
data/                     Dataset instructions only; raw data is not tracked
tests/                    Smoke tests
```

## Install

```bash
python -m pip install -e ".[dev]"
```

## Data

Raw SWaT and WADI files are not included. Place authorized local copies under:

```text
data/swat/raw/swat_train.csv
data/swat/raw/swat_test.csv
data/wadi/raw/wadi_train.csv
data/wadi/raw/wadi_test.csv
```

Check configured paths with:

```bash
python scripts/check_data_paths.py \
  --config configs/experiment/swat_gru.yaml
```

## Train SWaT GRU

```bash
python scripts/train_model.py \
  --experiment configs/experiment/swat_gru.yaml
```

Outputs:

```text
artifacts/checkpoints/swat/gru_h1/best.pth
artifacts/checkpoints/swat/gru_h1/resolved_config.yaml
artifacts/checkpoints/swat/gru_h1/normalization_stats.npz
artifacts/checkpoints/swat/gru_h1/columns.json
artifacts/checkpoints/swat/gru_h1/manifest.json
```

## Export Model Predictions

Using a trained checkpoint:

```bash
python scripts/export_model_predictions.py \
  --checkpoint artifacts/checkpoints/swat/gru_h1/best.pth \
  --config artifacts/checkpoints/swat/gru_h1/resolved_config.yaml \
  --split val \
  --normal-only \
  --out artifacts/model_exports/swat/gru_h1/val
```

Using the experiment config:

```bash
python scripts/export_model_predictions.py \
  --experiment configs/experiment/swat_gru.yaml \
  --split val \
  --normal-only \
  --out artifacts/model_exports/swat/gru_h1/val
```

Expected export files:

```text
val_inputs.npy
val_neural_preds.npy
val_actual_next.npy
metadata.json
```

## Current-State MLP Config

The current-state MLP follow-up config is available at:

```text
configs/experiment/swat_mlp_current.yaml
```

It uses `architecture: mlp`, `history_len: 1`, `horizon: 1`, and the same SWaT dataset preprocessing settings as the GRU model.

## Environment

Create a conda environment:

```bash
conda env create -f environment.yml
conda activate symbolic-ics
python -m pip install -e ".[dev]"
```

## Symbolic Distillation

Symbolic distillation configs are placeholders under `configs/distill/`. The PySR workflow is intentionally not implemented yet.

## Smoke Tests

```bash
python -m pytest -q
```
