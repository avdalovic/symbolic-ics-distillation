# Artifact Appendix

This artifact reproduces every table and the sensitivity figure of the ACID
paper. Each item below names the paper element, the committed file that contains
its values, and the command that regenerates it.

All recorded results were produced with the seeds reported in the paper.
Everything downstream of equation discovery is deterministic: given the
committed equations and grids, every command below returns the same numbers on
every run.

## Setup

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e ".[dev,plot]"
python -m pytest -q
```

Expected: `51 passed, 1 skipped`. Python 3.11 or newer; no plant data is needed
for anything except Section "Full pipeline" at the end.

## Table 2 — Detection performance against published IIDSs

```bash
cat paper_artifacts/main_results/comparison_table.csv
python scripts/build_main_results.py     # regenerates it from the detection grids
```

The ACID rows are read directly from the committed per-plant detection grids at
the operating points printed in the table, so the comparison is consistent with
the grids by construction:

| Dataset | Prec | Rec | F1 | eTaP | eTaR | eTaF1 | FPA | Scen. |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| SWaT | 97.3 | 77.6 | 86.3 | 88.8 | 54.6 | 67.6 | 4 | 75.0 |
| WADI | 93.9 | 47.8 | 63.4 | 95.6 | 56.9 | 71.3 | 0 | 64.3 |
| BATADAL | 95.5 | 53.9 | 68.9 | 98.2 | 80.0 | 88.2 | 0 | 100.0 |
| HAI | 55.2 | 65.5 | 59.9 | 53.7 | 80.3 | 64.4 | 7 | 96.0 |

Rows marked `wolsing2025gecos` are the published baseline results (measured with
the IPAL framework) and are reproduced verbatim for the side-by-side comparison.

## Table 3 — Knowledge-based comparison on SWaT

```bash
cat paper_artifacts/main_results/knowledge_swat.csv
```

The ACID row equals the SWaT row of Table 2; the knowledge-based row is the
published value.

## Table 4 — Recovered equations for three SWaT sensors

```bash
grep -E "LIT101|LIT301|DPIT301" paper_artifacts/main_results/swat/equations.csv
python -c "import json; m=json.load(open('artifacts/experiments/geco_model_inspection/SWaT.model'))['CI']; \
  [print(t, m[t]['combination'], [round(p,3) for p in m[t]['parameters']]) for t in ('LIT101','LIT301','DPIT301')]"
```

The first command prints ACID's deployed delta equations, for example
`(FIT201 - FIT101) * -0.19228067` for LIT101. The second prints the published
GeCo model entries for the same sensors, as quoted in the table.

## Table 5 — Cost of equation discovery and live classification

```bash
cat paper_artifacts/timing_final_seed42/timing_manifest.csv
cat paper_artifacts/timing_final_seed42/hai_live_timing/hai_live_detection_timing.csv
```

`timing_manifest.csv` records the discovery measurement, taken with
`/usr/bin/time -v` on a single machine (`lscpu.txt` and `uname_a.txt` record the
hardware and OS; `scripts/run_final_timing_seed42.sh` is the driver): elapsed
8.2 / 72.9 / 8.1 / 42.7 minutes and 5.7 / 46.3 / 6.4 / 32.6 core-hours for
SWaT / WADI / BATADAL / HAI. The live-classification measurement is
`hai_live_detection_timing.csv`: 0.0036 ms per snapshot over 402,000 test
snapshots and 47 channels, covering fixed-equation residual construction plus
the CUSUM update and excluding parsing, loading, and calibration. Deployed
detection is a fixed arithmetic expression per channel, so its cost scales with
the channel count alone. GeCo costs are the published values.

## Figure 4 — Sensitivity to the CUSUM parameters

```bash
python scripts/generate_hai_hp_panels.py \
  --chosen-s 2.5 --chosen-g 12.0 --contour-ratio 0.90 --fig4-name fig4_regenerated.pdf
```

Writes `paper_artifacts/final_v2/fig4_regenerated.pdf`, reproducing the
committed `fig4_v2.pdf`, from the committed detection
grids (`paper_artifacts/final_v2/extended_grid_{swat,wadi}.csv`,
`paper_artifacts/selected_models/batadal/detection_grid.csv`,
`paper_artifacts/selected_models/hai/detection_grid_r13.csv`). The contour is
90% of the maximum eTaF1 per grid; hatching marks cells with at least one
false-positive alarm interval.

## Table 6 — Pareto front for LIT101

```bash
cat results/swat/pareto_fronts/LIT101_sensors_delta_actuators_next/pareto_front_scored.csv
```

The complete scored front. The complexity-5 row,
`(FIT201 - FIT101) * -0.192` with holdout R² 0.402 and score 0.214, is the
selected equation. The corresponding fronts for every sensor of every plant are
under `results/<plant>/pareto_fronts/`.

## Table 7 — Sensitivity to the residual-tail threshold

```bash
cat paper_artifacts/overnight_v1/task3d_tail_frontier_both.csv
```

The recorded sweep over the bound. To re-run it end to end on BATADAL, which
ships with this repository:

```bash
python scripts/make_relaxed_guard.py
python scripts/overnight/task3d_tail_frontier_both.py --datasets batadal
```

The run prints `FIDELITY @50: 88.1531 vs 88.1531 -> FAITHFUL`: at the deployed
bound, the re-selection returns exactly the deployed equations and reproduces
the reported eTaF1. Fresh runs write to `task3d_*_run.csv`; the recorded sweep
is never overwritten.

## Table 8 — Detection at different CUSUM operating points

```bash
cat paper_artifacts/main_results/operating_points.csv
```

"Ours" and "At GeCo S/G" rows are both read from the committed grids; the GeCo
rows are the published values. Every plant's full grid is available at
`paper_artifacts/main_results/<plant>/grid.csv` so any other operating point can
be inspected as well.

## Table 9 — Alert localization

```bash
cat paper_artifacts/localization/localization_all4_summary.csv
python scripts/generate_localization_paper_figures.py --out /tmp/localization_check
```

The second command rebuilds the summary and the localization figures from the
committed per-attack localization records
(`artifacts/localization/` and `paper_artifacts/localization_runs/`). The
distance population per plant is the detected attacks whose alerting variable
has a finite path to the attacked variable in the learned dependency graph
(SWaT 23, WADI 8, BATADAL 11, HAI 33), as stated in the table caption.

## Tables 11 and 12 — Stability across three retrainings

```bash
cat paper_artifacts/seed_stability_v1/seed014_stability_tables.csv
python scripts/build_seed_stability_tables.py   # regenerates Tables 11-14
```

Mean ± standard deviation over 3 seeds at the ACID operating points
(Table 11) and at GeCo's published parameters (Table 12). FPA, an integer count,
is reported as median [min–max]. The per-seed rows behind every aggregate are in
`seed014_points_guarded.csv`.

## Table 13 — Input-set stability

```bash
cat paper_artifacts/seed_stability_v1/seed014_equation_stability_summary.csv
```

Jaccard similarity of each equation's input-variable set across the three
retrainings, averaged over the run pairs, summarized per plant (SWaT 0.680 mean
over 15 targets, WADI 0.658 over 61, BATADAL 0.758 over 31, HAI 0.724 over 45).
Per-target values, including each run's equation and input set, are in
`seed014_equation_stability_per_target.csv`.

## Table 14 — Equations recovered across retrainings

```bash
cat paper_artifacts/seed_stability_v1/seed014_showcase_equations.csv
```

The LIT101, LIT301, and DPIT301 equations from each retraining. LIT101 recovers
`c (FIT101 − FIT201)` with c = 0.19228, 0.19225, 0.19225. The same
`build_seed_stability_tables.py` run recomputes this file, and Tables 11-13,
from the three retrainings' committed equation selections.

## Table 15 — Equation structures beyond the template families

```bash
python scripts/audit_geco_expressiveness_class.py
python scripts/build_expressiveness_report.py
cat paper_artifacts/expressiveness_v1/showcase_out_of_class.csv
```

The audit classifies every deployed equation of every retraining against the
additive and multiplicative template families of the published GeCo models
(read directly from `artifacts/experiments/geco_model_inspection/*.model`) and
writes per-target results to `equation_class_per_target.csv`. The showcase file
contains the four equations quoted in the table (SWaT FIT601, WADI
2\_LS\_301\_AL, BATADAL P\_J415, HAI P1\_FT03Z) together with the corresponding
published GeCo equations.

## Full pipeline (restricted data)

The commands above reproduce all reported numbers from the committed equations
and grids. To repeat equation discovery itself, obtain the datasets (see the
README in each directory under `data/`; BATADAL is included), place them at

```
data/swat/raw/swat_train.csv    data/swat/raw/swat_test.csv
data/wadi/raw/wadi_train.csv    data/wadi/raw/wadi_test.csv
data/hai/ipal/*.state.gz
```

and run, per plant (hours each; `--seed` selects the retraining):

```bash
python scripts/run_swat_1sec_delta_local_diagnostic.py --out artifacts/swat/full --flat-pareto-layout
python scripts/run_wadi_1sec_delta_full.py  --out artifacts/wadi/full  --target-parallel-jobs 4
python scripts/run_batadal_delta_full.py    --out artifacts/batadal/full
python scripts/run_hai_1sec_delta_full.py   --output-dir artifacts/hai/full --max-workers 40
```

Evolutionary search is stochastic, so rediscovered equations vary between runs;
Tables 11–14 quantify exactly this variation across the three reported seeds,
and the recovered physical relationships (Table 14) are stable across them.

## Notes

**Benign-only training.** Equation discovery, selection, quality filtering, and
detector calibration use benign data exclusively. Attack labels are used only to
score the result. The full S/G grids are committed so the reported operating
points can be checked against every neighboring setting.

**HAI channel set.** HAI is evaluated on the channel set that excludes the seven
channels the published GeCo configuration also ignores
(`artifacts/experiments/geco_model_inspection/HAI.model`, `settings.ignore`), so
both methods are measured on the same channels.

