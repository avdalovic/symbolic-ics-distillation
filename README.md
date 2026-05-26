# ASID-ICS

Automatic Symbolic Intrusion Detection for Industrial Control Systems.

ASID-ICS learns compact state-space equations from benign ICS telemetry. The method
finds equation with evolutionary symbolic regression
over a fixed operator grammar: addition, subtraction, and multiplication.


## What Is Included

- `src/ics_symbolic_distill/detection/`: CUSUM, metrics, equation evaluation,
  and coverage-stratified sampling.
- `scripts/run_swat_1sec_delta_local_diagnostic.py`: SWaT one-second delta
  symbolic discovery and selection workflow.
- `scripts/run_wadi_1sec_delta_full.py`: WADI one-second delta symbolic
  discovery, WADI-safe variable-name mapping, quality filtering, and detection.
- `scripts/run_batadal_delta_full.py`: BATADAL delta symbolic discovery,
  quality filtering, and detection.
- `scripts/generate_paper_artifacts_v2.py`: regenerates the final paper tables
  and sensitivity figures from frozen local experiment outputs.
- `configs/asid_ics/`: per-dataset public configuration summaries.
- `results/`: compact reviewer-facing result snapshot: selected equations,
  Pareto-front CSVs, detection grids, per-sensor residual/alarm statistics, and
  per-attack summaries.
- `paper_artifacts/final_v2/`: paper-ready CSV, LaTeX, and figure outputs.

## Data

Raw SWaT and WADI data must be obtained from the dataset providers. Place
authorized local copies at:

```text
data/swat/raw/swat_train.csv
data/swat/raw/swat_test.csv
data/wadi/raw/wadi_train.csv
data/wadi/raw/wadi_test.csv
```

BATADAL is public; local BATADAL preparation is handled by
`scripts/prepare_batadal.py`.

Raw data, model checkpoints, large arrays, logs, and full generated run
directories are ignored by git. Before a public push, `git ls-files data`
should list only `data/README.md`.

## Install

```bash
conda env create -f environment.yml
conda activate symbolic-ics
python -m pip install -e ".[dev,distill]"
```

Fallback:

```bash
python -m pip install -r requirements.txt
python -m pip install -e ".[dev,distill]"
```

## Validate

```bash
python -m compileall src scripts tests
python -m pytest -q
```

## Reproduce Main Table Artifacts

The committed paper table is in:

```text
paper_artifacts/final_v2/main_table_one_row.csv
paper_artifacts/final_v2/table_main_one_row.tex
```

To regenerate the table and heatmaps from frozen local experiment outputs:

```bash
python scripts/generate_paper_artifacts_v2.py
```

This command does not rerun PySR. It requires the local raw datasets and frozen
experiment outputs under `artifacts/`. Reviewers without SWaT/WADI access can
inspect the committed `results/` and `paper_artifacts/final_v2/` files.

## Rerun Discovery

Full reruns require authorized raw data and can take hours.

```bash
python scripts/run_wadi_1sec_delta_full.py \
  --out artifacts/wadi_1sec/delta_full \
  --target-parallel-jobs 4
```

```bash
python scripts/run_batadal_delta_full.py \
  --out artifacts/batadal/delta_full
```

SWaT and WADI posthoc detection/sensitivity scripts are listed in
`ARTIFACTS.md`.

## Notes for Reviewers

Attack labels are not used for equation discovery, equation selection, quality
filtering, or CUSUM calibration. Labels are used only for evaluation and for the
reported S/G sensitivity grids. The public artifacts include the full evaluated
grids and the GeCo operating-point rows so the reported operating points can be
checked against nearby settings.

See `ARTIFACTS.md` for a short map from paper claims to files.
