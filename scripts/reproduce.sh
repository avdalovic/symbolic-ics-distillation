#!/usr/bin/env bash
set -euo pipefail

SWAT_RAW_DIR="${SWAT_RAW_DIR:-data/swat/raw}"
DISTILL_DIR="${DISTILL_DIR:-artifacts/model_exports/swat/distillation/val20_overlap}"
OUT_ROOT="${OUT_ROOT:-artifacts/symbolic_equations/swat/full_sensor_audit}"
NITERATIONS="${NITERATIONS:-400}"
TIMEOUT="${TIMEOUT:-1800}"
SAMPLE_SIZE="${SAMPLE_SIZE:-all}"

echo "Symbolic ICS reproducibility entry point"
echo "Python: $(command -v python)"

if [[ ! -f "${SWAT_RAW_DIR}/swat_train.csv" || ! -f "${SWAT_RAW_DIR}/swat_test.csv" ]]; then
  cat <<EOF
SWaT raw data was not found under:
  ${SWAT_RAW_DIR}/swat_train.csv
  ${SWAT_RAW_DIR}/swat_test.csv

SWaT is access restricted and must be obtained from iTrust by authorized users.
Raw data is intentionally not included in this repository.
EOF
fi

required_distill_files=(
  "distill_inputs_current_raw.npy"
  "distill_pred_next_raw_mlp.npy"
  "distill_pred_delta_raw_mlp.npy"
  "distill_actual_next_raw.npy"
  "distill_actual_delta_raw.npy"
  "distill_feature_columns.json"
  "distill_target_columns.json"
  "metadata.json"
)

missing=()
for file in "${required_distill_files[@]}"; do
  if [[ ! -f "${DISTILL_DIR}/${file}" ]]; then
    missing+=("${DISTILL_DIR}/${file}")
  fi
done

if [[ "${#missing[@]}" -gt 0 ]]; then
  echo "Required distillation arrays are missing:"
  printf '  %s\n' "${missing[@]}"
  cat <<EOF

Create them first by running the project's SWaT preprocessing/training/export flow,
then prepare the distillation overlap directory. This script does not invent or
replace those scientific preprocessing steps.
EOF
  exit 1
fi

python -m pytest -q

OUT_ROOT="${OUT_ROOT}" \
DISTILL_DIR="${DISTILL_DIR}" \
NITERATIONS="${NITERATIONS}" \
TIMEOUT="${TIMEOUT}" \
SAMPLE_SIZE="${SAMPLE_SIZE}" \
bash scripts/run_full_sensor_audit.sh
