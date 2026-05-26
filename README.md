# ASID-ICS

Automatic Symbolic Intrusion Detection for Industrial Control Systems.

ASID-ICS learns compact state-space equations from benign ICS telemetry. The
method finds equations with evolutionary symbolic regression over a fixed
operator grammar: addition, subtraction, and multiplication.


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
  and sensitivity figures from the committed `results/` snapshot.
- `configs/asid_ics/`: per-dataset public configuration summaries.
- `results/`: compact reviewer-facing result snapshot: selected equations,
  Pareto-front CSVs, detection grids, per-sensor residual/alarm statistics, and
  per-attack summaries.
- `paper_artifacts/final_v2/`: paper-ready CSV, LaTeX, and figure outputs.

## Data

Raw SWaT and WADI data must be obtained from the dataset providers. See:

```text
data/swat/README.md
data/wadi/README.md
```

After processing, the expected local paths are:

```text
data/swat/raw/swat_train.csv
data/swat/raw/swat_test.csv
data/wadi/raw/wadi_train.csv
data/wadi/raw/wadi_test.csv
```

BATADAL is public and the processed CSVs used by the paper are included under:

```text
data/batadal/processed/
```

Raw data, model checkpoints, large arrays, logs, and full generated run
directories are ignored by git. `data/swat/raw/` and `data/wadi/raw/` remain
local-only.

## Install

The lightweight artifact install works on Linux, macOS, and Windows with
Python 3.11 or newer. It is enough to regenerate the paper tables/heatmaps and
run the tests.

Linux/macOS:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e ".[dev,plot]"
```

Windows PowerShell:

```powershell
py -3.11 -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -U pip
python -m pip install -r requirements.txt
python -m pip install -e ".[dev,plot]"
```

Conda is also supported:

```bash
conda env create -f environment.yml
conda activate symbolic-ics-artifact
```

Full PySR discovery reruns require the additional symbolic-regression stack:

```bash
python -m pip install -r requirements-full.txt
python -m pip install -e ".[dev,plot,distill]"
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

To regenerate the table and heatmaps from the committed result snapshot:

```bash
python scripts/generate_paper_artifacts_v2.py
```

This command does not rerun PySR and does not require raw SWaT/WADI data. It
uses the committed `results/{swat,wadi,batadal}/detection_grid.csv` files.
The regenerated main table should contain these ASID-ICS rows:

```text
SWaT:    F1=86.3446, eTaF1=67.6095, FPA=4
WADI:    F1=63.3663, eTaF1=71.3327, FPA=0
BATADAL: F1=66.3768, eTaF1=84.9644, FPA=0
```

To check that the data filenames are in place after setup:

```bash
python scripts/check_data_paths.py --config configs/experiment/swat_mlp_current_val20.yaml
test -f data/wadi/raw/wadi_train.csv && test -f data/wadi/raw/wadi_test.csv
test -f data/batadal/processed/train.csv
```

## Rerun Discovery

Full reruns require authorized raw data and can take hours.

```bash
python scripts/run_swat_1sec_delta_local_diagnostic.py \
  --out artifacts/swat_1sec/delta_full \
  --flat-pareto-layout
```

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
