#!/usr/bin/env bash
set -euo pipefail

ARGS=(--stage "${STAGE:-all}")
ARGS+=(--output-root "${OUTPUT_ROOT:-output/cusp_minimal_seed42}")
ARGS+=(--topology "${TOPOLOGY:-clientlt}")
ARGS+=(--data "${DATA:-DATA}")
ARGS+=(--python-bin "${PYTHON_BIN:-python}")

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  ARGS+=(--cuda-visible-devices "${CUDA_VISIBLE_DEVICES}")
fi

if [[ "${DRY_RUN:-1}" == "0" ]]; then
  ARGS+=(--run)
else
  ARGS+=(--dry-run)
fi

"${PYTHON_BIN:-python}" scripts/run_cusp_minimal.py "${ARGS[@]}"
