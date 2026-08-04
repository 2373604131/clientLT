#!/usr/bin/env bash
set -euo pipefail

if [[ -z "${DUMP_DIR:-}" || -z "${OUTPUT_DIR:-}" ]]; then
  echo "DUMP_DIR and OUTPUT_DIR are required" >&2
  exit 2
fi

ARGS=(--dump-dir "${DUMP_DIR}" --output-dir "${OUTPUT_DIR}")
ARGS+=(--batch-size "${BATCH_SIZE:-2048}")
ARGS+=(--gamma "${GAMMA:-0.5}" --tau "${TAU:-0.0}")
ARGS+=(--min-support-clients "${MIN_SUPPORT_CLIENTS:-2}")
ARGS+=(--max-edges-per-class "${MAX_EDGES_PER_CLASS:-3}" --max-total-edges "${MAX_TOTAL_EDGES:-60}")
ARGS+=(--repair-ratio "${REPAIR_RATIO:-0.25}")
ARGS+=(--min-deficit-closure "${MIN_DEFICIT_CLOSURE:-0.0}")
ARGS+=(--max-non-target-margin-drop "${MAX_NON_TARGET_MARGIN_DROP:-0.05}")
ARGS+=(--max-semantic-repair-drift "${MAX_SEMANTIC_REPAIR_DRIFT:-0.01}")

"${PYTHON_BIN:-python}" scripts/run_boundary_gate.py "${ARGS[@]}"
