#!/usr/bin/env bash
set -euo pipefail

DISTILL_DIR="${DISTILL_DIR:-artifacts/model_exports/swat/distillation/val20_overlap}"
ATTRIBUTION_DIR="${ATTRIBUTION_DIR:-artifacts/model_exports/swat/attribution/mlp_current_h1_val20}"
OUT_ROOT="${OUT_ROOT:-artifacts/symbolic_equations/swat/mlp_pilot}"
SAMPLE_SIZE="${SAMPLE_SIZE:-5000}"
SEED="${SEED:-0}"
NITERATIONS="${NITERATIONS:-400}"
TIMEOUT="${TIMEOUT:-1800}"
TOP_K="${TOP_K:-8}"

mkdir -p "${OUT_ROOT}/logs"

run_distill() {
  local run_name="$1"
  local sensor="$2"
  local mode="$3"
  local target="$4"
  local extra_args=()
  if [[ "${mode}" == "topk" ]]; then
    extra_args+=(--top-k "${TOP_K}")
  fi

  echo "==== $(date -Is) START ${run_name} ===="
  python scripts/distill_sensor_mlp.py \
    --sensor "${sensor}" \
    --mode "${mode}" \
    --target-source "${target}" \
    --operator-set rich \
    --distill-dir "${DISTILL_DIR}" \
    --attribution-dir "${ATTRIBUTION_DIR}" \
    --out "${OUT_ROOT}/${run_name}" \
    --sample-size "${SAMPLE_SIZE}" \
    --seed "${SEED}" \
    --niterations "${NITERATIONS}" \
    --timeout "${TIMEOUT}" \
    "${extra_args[@]}" 2>&1 | tee "${OUT_ROOT}/logs/${run_name}.log"
  echo "==== $(date -Is) END ${run_name} ===="
}

run_distill "LIT101_mlp_delta_unconstrained" "LIT101" "unconstrained" "mlp_delta"
run_distill "LIT101_mlp_delta_topk8" "LIT101" "topk" "mlp_delta"
run_distill "FIT101_mlp_delta_topk8" "FIT101" "topk" "mlp_delta"
run_distill "FIT201_mlp_delta_topk8" "FIT201" "topk" "mlp_delta"
run_distill "LIT301_mlp_delta_topk8" "LIT301" "topk" "mlp_delta"
run_distill "DPIT301_mlp_delta_topk8" "DPIT301" "topk" "mlp_delta"
run_distill "LIT101_actual_delta_unconstrained" "LIT101" "unconstrained" "actual_delta"

python scripts/report_pilot_mlp.py --out-root "${OUT_ROOT}" 2>&1 | tee "${OUT_ROOT}/logs/final_report.log"
