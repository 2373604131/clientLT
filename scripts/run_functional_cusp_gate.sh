#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DUMP_DIR:-}" ]]; then
  echo "DUMP_DIR is required" >&2
  exit 2
fi

if [[ -z "${OUTPUT_DIR:-}" ]]; then
  echo "OUTPUT_DIR is required" >&2
  exit 2
fi

ARGS=(--dump-dir "${DUMP_DIR}")
ARGS+=(--output-dir "${OUTPUT_DIR}")
ARGS+=(--rank-max "${RANK_MAX:-8}")
ARGS+=(--probe-rel-step "${PROBE_REL_STEP:-0.1}")
ARGS+=(--steer-ratio "${STEER_RATIO:-0.25}")
ARGS+=(--class-count-power "${CLASS_COUNT_POWER:-0.5}")
ARGS+=(--probe-batch-size "${PROBE_BATCH_SIZE:-2048}")

"${PYTHON_BIN:-python}" scripts/run_functional_cusp_gate.py "${ARGS[@]}"
