# ACID: Automatic and Comprehensible Intrusion Detection for ICS

ACID detects process-level attacks on Industrial Control Systems by learning how
the physical process evolves and reporting when the observed behaviour stops
agreeing with the learned dynamics.

The detector is built from equations that an operator can read. ACID learns those
equations directly from benign telemetry with evolutionary symbolic regression:
the search determines which process variables enter each equation, how they
combine, and the constants that relate them. No equation form is supplied by hand.

This repository is the artifact for the ACID paper. It contains the method, the
deployed equations, every result file behind the reported numbers, and the
scripts that regenerate them. `ARTIFACT.md` is the reproduction guide.

## Method

For each sensor `i`, ACID learns the one-step change rather than the absolute next
value:

```
delta x_i[t] = x_i[t+1] - x_i[t] = F_i(x[t], u[t])
```

where `x` are sensor readings and `u` are actuator states. Predicting the change
isolates the quantity physical laws govern, and prevents the search from settling
on a near-identity mapping that scores well while modelling nothing.

**Search.** Expressions are assembled from the operator grammar `{+, -, *}` over
all process variables. Division is excluded: zero-valued control signals are
routine in ICS telemetry and produce singular predictions. Each sensor gets a
fixed search budget, and the search returns a Pareto front trading accuracy
against expression size.

**Selection.** A candidate is deployed only if it satisfies four criteria
evaluated on benign data alone. It must produce finite predictions; its
reconstructed next state `x_i[t] + F_i` must still depend on the measured process
state; it must raise no alarm on a held-out benign split; and its residual tail,
measured as the 99th percentile over the median, must stay bounded. Among the
admissible candidates ACID keeps the highest-scoring one. A sensor with no
admissible candidate is left unmonitored. Actuator channels are monitored by
persistence.

**Detection.** Each monitored channel produces a residual, and a CUSUM test
accumulates it. The system alarms when any channel's statistic exceeds its
calibrated threshold. Two parameters govern sensitivity: `S` scales the threshold
above the benign maximum, and `G` caps accumulation.

Attack labels are never used for discovery, selection, filtering, or calibration.
They are used only to score the result.

**Example.** On the SWaT water-storage subsystem, ACID recovers conservation of
mass from data alone:

```
delta LIT101 = 0.192 * (FIT101 - FIT201)
```

The search selected inflow `FIT101` and outflow `FIT201` from all available
variables, composed them as a difference, and fitted the coefficient. The same
relationship and coefficient reappear under independent retraining (0.19228, 0.19225, 0.19225).

## Results

ACID at each plant's operating point:

| Plant | S | G | F1 | eTaF1 | FPA | Scenarios |
|---|---:|---:|---:|---:|---:|---:|
| SWaT | 1.20 | 15.0 | 86.34 | 67.61 | 4 | 75.0% |
| WADI | 1.20 | 25.0 | 63.37 | 71.33 | 0 | 64.3% |
| BATADAL | 1.40 | 2.00 | 68.94 | 88.15 | 0 | 100.0% |
| HAI | 2.50 | 12.0 | 59.92 | 64.36 | 7 | 96.0% |

`paper_artifacts/main_results/comparison_table.csv` places these alongside six
published baselines (GeCo, SIMPLE, TABOR, Invariant, Seq2SeqNN, PASAD).

Equation discovery costs 8.2 minutes on SWaT, 72.9 on WADI, 8.1 on BATADAL and
42.7 on HAI, on a single machine. Detection itself is a fixed arithmetic
expression per channel and runs in microseconds per sample.

## Layout

```
src/ics_symbolic_distill/             Detection core: CUSUM, metrics, equation
                                      evaluation, selection criteria, sampling
scripts/                              Discovery pipelines and the scripts that
                                      regenerate each paper table and figure
configs/                              Per-plant configuration
results/                              Selected equations, Pareto fronts,
                                      detection grids, per-attack outcomes
paper_artifacts/main_results/         Tables 2, 3, 8 and per-plant results
paper_artifacts/final_v2/             Sensitivity grids and Figure 4
paper_artifacts/selected_models/      Deployed BATADAL and HAI models
paper_artifacts/seed_stability_v1/    Tables 11-14 (retraining stability)
paper_artifacts/expressiveness_v1/    Table 15 (equation structures)
paper_artifacts/localization/         Table 9 (alert localization)
paper_artifacts/timing_final_seed42/  Table 5 (construction cost)
data/                                 Dataset instructions; BATADAL included
tests/
```

The core is `src/ics_symbolic_distill/detection/`: `cusum.py` (detector),
`metrics.py` (F1, eTaF1, FPA, scenario coverage), `symbolic_eval.py` (equation
evaluation), `selection_guards.py` (state-dependence criterion), and
`swat1s_delta_sampling.py` (coverage-stratified training sample).

## Environment

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python -m pip install -e ".[dev,plot]"
```

Python 3.11 or newer. Rerunning equation discovery additionally needs the
symbolic-regression stack:

```bash
python -m pip install -r requirements-full.txt
python -m pip install -e ".[dev,plot,distill]"
```

## Validation

```bash
python -m compileall -q src scripts tests
python -m pytest -q
```

## Data access

BATADAL is public and included under `data/batadal/processed/`. SWaT and WADI
require an authorized request to iTrust, Singapore University of Technology and
Design. HAI is public but must be downloaded and transcribed. See the README in
each directory under `data/`.

Raw telemetry is not redistributed here. The committed result files are enough to
check every reported table without it.

## Reproduction

`ARTIFACT.md` maps every table and figure of the paper to the committed file
that contains its values and the command that regenerates it. All reported
numbers reproduce from the committed equations and grids without plant data;
BATADAL additionally supports an end-to-end detector re-run, and the restricted
datasets enable full equation rediscovery.
