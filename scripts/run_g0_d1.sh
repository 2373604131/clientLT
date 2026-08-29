#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-DATA}"
OUT_ROOT="${OUT_ROOT:-output/g0_d1_seed42}"
GPU="${GPU:-0}"
STAGE="${STAGE:-all}"
NUM_WORKERS="${NUM_WORKERS:-8}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-128}"
RANDOM_SUPPORT_COUNT="${RANDOM_SUPPORT_COUNT:-20}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
DRY_RUN="${DRY_RUN:-0}"

command=(
  "${PYTHON_BIN}" -u scripts/run_g0_d1.py
  --stage "${STAGE}"
  --output-root "${OUT_ROOT}"
  --data-root "${DATA_ROOT}"
  --python-bin "${PYTHON_BIN}"
  --gpu "${GPU}"
  --num-workers "${NUM_WORKERS}"
  --test-batch-size "${TEST_BATCH_SIZE}"
  --random-support-count "${RANDOM_SUPPORT_COUNT}"
)

if [[ "${SKIP_COMPLETED}" == "1" ]]; then
  command+=(--skip-completed)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  command+=(--dry-run)
fi

echo "G0 -> D1 foreground runner"
echo "  stage: ${STAGE}"
echo "  output: ${OUT_ROOT}"
echo "  GPU: ${GPU}"
echo "  python: ${PYTHON_BIN}"
printf '  command:'
printf ' %q' "${command[@]}"
printf '\n'

"${command[@]}"
