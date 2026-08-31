#!/usr/bin/env bash
set -euo pipefail

# Foreground by design: every Python error remains visible in the terminal.
PYTHON_BIN="${PYTHON_BIN:-python}"
STAGE="${STAGE:-sca}"
GPU="${GPU:-0}"
DATA_ROOT="${DATA_ROOT:-DATA}"
OUT_ROOT="${OUT_ROOT:-output/online_sca_seed42}"
FREEZE="${FREEZE:-output/g0_d1_seed42/lora_freeze.json}"
MATCHED_BETA="${MATCHED_BETA:-0.5}"
EXTRA_ARGS=()
if [[ "${SKIP_SUMMARY:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-summary)
fi
if [[ "${SKIP_FACTORIAL_ANALYSIS:-0}" == "1" ]]; then
  EXTRA_ARGS+=(--skip-factorial-analysis)
fi

exec "${PYTHON_BIN}" -u scripts/run_online_sca.py \
  --stage "${STAGE}" \
  --gpu "${GPU}" \
  --data-root "${DATA_ROOT}" \
  --output-root "${OUT_ROOT}" \
  --freeze-file "${FREEZE}" \
  --matched-beta "${MATCHED_BETA}" \
  "${EXTRA_ARGS[@]}"
