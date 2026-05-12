#!/usr/bin/env bash
set -euo pipefail

DISTILL_DIR="${DISTILL_DIR:-artifacts/model_exports/swat/distillation/val20_overlap}"
ATTRIBUTION_DIR="${ATTRIBUTION_DIR:-artifacts/model_exports/swat/attribution/mlp_current_h1_val20}"
OUT_ROOT="${OUT_ROOT:-artifacts/symbolic_equations/swat/lit101_target_audit}"
SENSOR="${SENSOR:-LIT101}"
SAMPLE_SIZE="${SAMPLE_SIZE:-all}"
SEED="${SEED:-0}"
NITERATIONS="${NITERATIONS:-400}"
TIMEOUT="${TIMEOUT:-1800}"
EVAL_FRAC="${EVAL_FRAC:-0.2}"
RUN_RICH="${RUN_RICH:-0}"

mkdir -p "${OUT_ROOT}/logs"

run_one() {
  local target_source="$1"
  local operator_set="$2"
  local run_name="${target_source}_${operator_set}_unconstrained"
  echo "==== $(date -Is) START ${run_name} ===="
  python scripts/distill_sensor_mlp.py \
    --sensor "${SENSOR}" \
    --target-source "${target_source}" \
    --mode unconstrained \
    --operator-set "${operator_set}" \
    --sample-size "${SAMPLE_SIZE}" \
    --eval-split temporal \
    --eval-frac "${EVAL_FRAC}" \
    --seed "${SEED}" \
    --niterations "${NITERATIONS}" \
    --timeout "${TIMEOUT}" \
    --distill-dir "${DISTILL_DIR}" \
    --attribution-dir "${ATTRIBUTION_DIR}" \
    --out "${OUT_ROOT}/${run_name}" 2>&1 | tee "${OUT_ROOT}/logs/${run_name}.log"
  echo "==== $(date -Is) END ${run_name} ===="
}

for target_source in actual_next actual_delta mlp_next mlp_delta; do
  run_one "${target_source}" restricted
done

if [[ "${RUN_RICH}" == "1" ]]; then
  for target_source in actual_next actual_delta mlp_next mlp_delta; do
    run_one "${target_source}" rich
  done
fi

python scripts/linear_lit101_baselines.py \
  --distill-dir "${DISTILL_DIR}" \
  --sensor "${SENSOR}" \
  --eval-frac "${EVAL_FRAC}" \
  --out "${OUT_ROOT}" 2>&1 | tee "${OUT_ROOT}/logs/linear_baselines.log"

python scripts/report_lit101_target_audit.py \
  --audit-dir "${OUT_ROOT}" \
  --sensor "${SENSOR}" 2>&1 | tee "${OUT_ROOT}/logs/report.log"
