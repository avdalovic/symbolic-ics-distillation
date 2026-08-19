#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PY="${PY:-python}"
SWAT_TRAIN="${SWAT_TRAIN:-$ROOT/data/swat/raw/swat_train.csv}"
SWAT_TEST="${SWAT_TEST:-$ROOT/data/swat/raw/swat_test.csv}"
WADI_TRAIN="${WADI_TRAIN:-$ROOT/data/wadi/raw/wadi_train.csv}"
WADI_TEST="${WADI_TEST:-$ROOT/data/wadi/raw/wadi_test.csv}"
OUT_ROOT="$ROOT/paper_artifacts/timing_final_seed42"
LOG_DIR="$ROOT/logs"
CGROUP_CPU="/sys/fs/cgroup/cpu.stat"

mkdir -p "$OUT_ROOT" "$LOG_DIR"

cd "$ROOT" || exit 1

echo "=== $(date --iso-8601=seconds) FINAL TIMING seed 42 start ===" | tee "$OUT_ROOT/runner_status.log"
git rev-parse HEAD > "$OUT_ROOT/git_head.txt"
git status --short > "$OUT_ROOT/git_status_short.txt"
cat /proc/loadavg > "$OUT_ROOT/loadavg_start.txt"
lscpu > "$OUT_ROOT/lscpu.txt"
free -h > "$OUT_ROOT/free_h.txt"
uname -a > "$OUT_ROOT/uname_a.txt"

run_one() {
  local ds="$1"
  shift
  local log="$LOG_DIR/${ds}_seed42_timev.log"
  local out_dir="$OUT_ROOT/${ds}_seed42_timev"
  mkdir -p "$out_dir"

  echo "=== $(date --iso-8601=seconds) START ${ds} ===" | tee -a "$OUT_ROOT/runner_status.log"
  cat "$CGROUP_CPU" > "$OUT_ROOT/pre_${ds}_cpu.stat"
  /usr/bin/time -v "$@" > "$log" 2>&1
  local status=$?
  cat "$CGROUP_CPU" > "$OUT_ROOT/post_${ds}_cpu.stat"
  echo "$status" > "$OUT_ROOT/${ds}_exit_status.txt"
  echo "=== $(date --iso-8601=seconds) DONE ${ds} status=${status} ===" | tee -a "$OUT_ROOT/runner_status.log"
  if [ "$status" -ne 0 ]; then
    echo "Timing run failed for ${ds}; see ${log}" | tee -a "$OUT_ROOT/runner_status.log"
    exit "$status"
  fi
}

run_one swat \
  "$PY" scripts/run_swat_1sec_delta_local_diagnostic.py \
    --seed 42 \
    --out "$OUT_ROOT/swat_seed42_timev" \
    --train-csv "$SWAT_TRAIN" \
    --test-csv "$SWAT_TEST" \
    --targets all-sensors \
    --configs FULL \
    --flat-pareto-layout \
    --parallel-jobs 1 \
    --target-parallel-jobs 20 \
    --max-complexity 12 \
    --niterations 400 \
    --timeout-minutes 20

run_one wadi \
  "$PY" scripts/run_wadi_1sec_delta_full.py \
    --seed 42 \
    --out "$OUT_ROOT/wadi_seed42_timev" \
    --train-csv "$WADI_TRAIN" \
    --test-csv "$WADI_TEST" \
    --target-parallel-jobs 8 \
    --pysr-procs 1

run_one hai \
  "$PY" scripts/run_hai_1sec_delta_full.py \
    --seed 42 \
    --max-workers 40 \
    --resume \
    --skip-detection-eval \
    --output-dir "$OUT_ROOT/hai_seed42_timev"

"$PY" scripts/extract_final_timing_seed42.py

echo "=== $(date --iso-8601=seconds) FINAL TIMING seed 42 complete ===" | tee -a "$OUT_ROOT/runner_status.log"
