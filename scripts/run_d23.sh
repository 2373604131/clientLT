#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-DATA}"
OUT_ROOT="${OUT_ROOT:-output/d23_seed42}"
FREEZE="${FREEZE:-output/g0_d1_seed42/lora_freeze.json}"
GPU="${GPU:-0}"
STAGE="${STAGE:-all}"
NUM_WORKERS="${NUM_WORKERS:-8}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
SKIP_COMPLETED="${SKIP_COMPLETED:-0}"
DRY_RUN="${DRY_RUN:-0}"

command=(
  "${PYTHON_BIN}" -u scripts/run_d23.py
  --stage "${STAGE}"
  --output-root "${OUT_ROOT}"
  --freeze "${FREEZE}"
  --data-root "${DATA_ROOT}"
  --python-bin "${PYTHON_BIN}"
  --gpu "${GPU}"
  --num-workers "${NUM_WORKERS}"
  --eval-batch-size "${EVAL_BATCH_SIZE}"
)
if [[ "${SKIP_COMPLETED}" == "1" ]]; then
  command+=(--skip-completed)
fi
if [[ "${DRY_RUN}" == "1" ]]; then
  command+=(--dry-run)
fi

echo "D2/D3/D2b/P0 foreground diagnostic runner"
echo "  stage: ${STAGE}"
echo "  shared dump: ${OUT_ROOT}/dump_seed42"
echo "  GPU: ${GPU}"
echo "  python: ${PYTHON_BIN}"
printf '  command:'
printf ' %q' "${command[@]}"
printf '\n'
"${command[@]}"
