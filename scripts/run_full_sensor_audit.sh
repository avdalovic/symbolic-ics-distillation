#!/usr/bin/env bash
set -uo pipefail

TARGET_SOURCES="${TARGET_SOURCES:-actual_next actual_delta mlp_next mlp_delta}"
NITERATIONS="${NITERATIONS:-400}"
TIMEOUT="${TIMEOUT:-1800}"
SAMPLE_SIZE="${SAMPLE_SIZE:-all}"
OPERATOR_SET="${OPERATOR_SET:-restricted}"
EVAL_SPLIT="${EVAL_SPLIT:-temporal}"
EVAL_FRAC="${EVAL_FRAC:-0.2}"
MODE="${MODE:-unconstrained}"
SEED="${SEED:-0}"
DISTILL_DIR="${DISTILL_DIR:-artifacts/model_exports/swat/distillation/val20_overlap}"
ATTRIBUTION_DIR="${ATTRIBUTION_DIR:-artifacts/model_exports/swat/attribution/mlp_current_h1_val20}"
OUT_ROOT="${OUT_ROOT:-artifacts/symbolic_equations/swat/full_sensor_audit}"
SUPPORT_CONFIG="${SUPPORT_CONFIG:-configs/swat_sensor_local_support.json}"
RESUME="${RESUME:-1}"
RUN_LINEAR_BASELINES="${RUN_LINEAR_BASELINES:-1}"
RUN_REPORT="${RUN_REPORT:-1}"
SENSORS="${SENSORS:-$(python scripts/list_target_sensors.py --distill-dir "${DISTILL_DIR}")}"

read -r -a SENSOR_ARR <<< "${SENSORS}"
read -r -a TARGET_ARR <<< "${TARGET_SOURCES}"

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/.run_state"
: > "${OUT_ROOT}/.run_state/completed.txt"
: > "${OUT_ROOT}/.run_state/skipped.txt"
: > "${OUT_ROOT}/.run_state/failed.txt"

START_TS="$(date +%s)"
export SENSORS TARGET_SOURCES NITERATIONS TIMEOUT SAMPLE_SIZE OPERATOR_SET EVAL_SPLIT EVAL_FRAC MODE SEED
export DISTILL_DIR ATTRIBUTION_DIR OUT_ROOT SUPPORT_CONFIG RESUME RUN_LINEAR_BASELINES RUN_REPORT START_TS

python - <<'PY'
import json
import os
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

def run(cmd):
    try:
        return subprocess.check_output(cmd, text=True, stderr=subprocess.STDOUT).strip()
    except Exception as exc:
        return str(exc)

out_root = Path(os.environ["OUT_ROOT"])
manifest = {
    "timestamp": datetime.now(timezone.utc).isoformat(),
    "git_commit": run(["git", "rev-parse", "HEAD"]),
    "git_status_short": run(["git", "status", "--short"]),
    "hostname": platform.node(),
    "python_version": sys.version,
    "python_executable": sys.executable,
    "settings": {
        "sensors": os.environ["SENSORS"].split(),
        "target_sources": os.environ["TARGET_SOURCES"].split(),
        "niterations": int(os.environ["NITERATIONS"]),
        "timeout": int(os.environ["TIMEOUT"]),
        "sample_size": os.environ["SAMPLE_SIZE"],
        "operator_set": os.environ["OPERATOR_SET"],
        "eval_split": os.environ["EVAL_SPLIT"],
        "eval_frac": float(os.environ["EVAL_FRAC"]),
        "mode": os.environ["MODE"],
        "seed": int(os.environ["SEED"]),
        "distill_dir": os.environ["DISTILL_DIR"],
        "attribution_dir": os.environ["ATTRIBUTION_DIR"],
        "support_config": os.environ["SUPPORT_CONFIG"],
        "output_root": os.environ["OUT_ROOT"],
        "resume": os.environ["RESUME"],
        "run_linear_baselines": os.environ["RUN_LINEAR_BASELINES"],
        "run_report": os.environ["RUN_REPORT"],
    },
}
(out_root / "run_manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
print(f"Wrote {out_root / 'run_manifest.json'}")
PY

record() {
  local file="$1"
  local value="$2"
  printf '%s\n' "${value}" >> "${OUT_ROOT}/.run_state/${file}.txt"
}

run_complete() {
  local run_dir="$1"
  [[ -f "${run_dir}/pareto_front_scored.csv" && -f "${run_dir}/metadata.json" && -f "${run_dir}/best_equation.json" ]]
}

run_one() {
  local sensor="$1"
  local target_source="$2"
  local run_name="${sensor}_${target_source}_${OPERATOR_SET}_${MODE}"
  local run_dir="${OUT_ROOT}/${run_name}"
  local log_path="${OUT_ROOT}/logs/${run_name}.log"

  if [[ "${RESUME}" == "1" ]] && run_complete "${run_dir}"; then
    echo "==== $(date -Is) SKIP existing ${run_name} ====" | tee -a "${log_path}"
    record skipped "${run_name}"
    return 0
  fi

  echo "==== $(date -Is) START ${run_name} ====" | tee "${log_path}"
  python scripts/distill_sensor_mlp.py \
    --sensor "${sensor}" \
    --target-source "${target_source}" \
    --mode "${MODE}" \
    --operator-set "${OPERATOR_SET}" \
    --sample-size "${SAMPLE_SIZE}" \
    --eval-split "${EVAL_SPLIT}" \
    --eval-frac "${EVAL_FRAC}" \
    --seed "${SEED}" \
    --niterations "${NITERATIONS}" \
    --timeout "${TIMEOUT}" \
    --distill-dir "${DISTILL_DIR}" \
    --attribution-dir "${ATTRIBUTION_DIR}" \
    --out "${run_dir}" 2>&1 | tee -a "${log_path}"
  local rc=${PIPESTATUS[0]}
  if [[ "${rc}" -eq 0 ]]; then
    echo "==== $(date -Is) END ${run_name} ====" | tee -a "${log_path}"
    record completed "${run_name}"
  else
    echo "==== $(date -Is) FAIL ${run_name} rc=${rc} ====" | tee -a "${log_path}"
    record failed "${run_name}"
  fi
  return 0
}

for sensor in "${SENSOR_ARR[@]}"; do
  for target_source in "${TARGET_ARR[@]}"; do
    run_one "${sensor}" "${target_source}"
  done
done

if [[ "${RUN_LINEAR_BASELINES}" == "1" ]]; then
  log_path="${OUT_ROOT}/logs/linear_sensor_baselines.log"
  echo "==== $(date -Is) START linear_sensor_baselines ====" | tee "${log_path}"
  python scripts/linear_sensor_baselines.py \
    --sensors "${SENSOR_ARR[@]}" \
    --target-sources "${TARGET_ARR[@]}" \
    --distill-dir "${DISTILL_DIR}" \
    --out "${OUT_ROOT}/linear_sensor_baselines.csv" \
    --support-config "${SUPPORT_CONFIG}" \
    --eval-split "${EVAL_SPLIT}" \
    --eval-frac "${EVAL_FRAC}" \
    --seed "${SEED}" 2>&1 | tee -a "${log_path}"
  rc=${PIPESTATUS[0]}
  if [[ "${rc}" -ne 0 ]]; then
    record failed "linear_sensor_baselines"
  fi
fi

if [[ "${RUN_REPORT}" == "1" ]]; then
  log_path="${OUT_ROOT}/logs/report_cross_sensor.log"
  echo "==== $(date -Is) START report_cross_sensor ====" | tee "${log_path}"
  python scripts/report_cross_sensor.py \
    --audit-root "${OUT_ROOT}" \
    --distill-dir "${DISTILL_DIR}" \
    --support-config "${SUPPORT_CONFIG}" \
    --linear-baselines "${OUT_ROOT}/linear_sensor_baselines.csv" \
    --out "${OUT_ROOT}/cross_sensor_report.md" \
    --summary-csv "${OUT_ROOT}/cross_sensor_summary.csv" 2>&1 | tee -a "${log_path}"
  rc=${PIPESTATUS[0]}
  if [[ "${rc}" -ne 0 ]]; then
    record failed "report_cross_sensor"
  fi
fi

END_TS="$(date +%s)"
export END_TS
python - <<'PY'
import json
import os
from pathlib import Path

out_root = Path(os.environ["OUT_ROOT"])
state = out_root / ".run_state"

def read_lines(name):
    path = state / f"{name}.txt"
    if not path.exists():
        return []
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]

failed = read_lines("failed")
status = {
    "completed_runs": read_lines("completed"),
    "skipped_runs": read_lines("skipped"),
    "failed_runs": failed,
    "total_wall_clock_seconds": int(os.environ["END_TS"]) - int(os.environ["START_TS"]),
    "report_files": {
        "manifest": str(out_root / "run_manifest.json"),
        "linear_baselines": str(out_root / "linear_sensor_baselines.csv"),
        "cross_sensor_report": str(out_root / "cross_sensor_report.md"),
        "cross_sensor_summary": str(out_root / "cross_sensor_summary.csv"),
    },
}
(out_root / "run_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
print(f"Wrote {out_root / 'run_status.json'}")
print(json.dumps(status, indent=2))
PY

if [[ -s "${OUT_ROOT}/.run_state/failed.txt" ]]; then
  exit 1
fi
exit 0
