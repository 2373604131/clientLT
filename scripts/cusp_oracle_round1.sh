#!/usr/bin/env bash
set -euo pipefail

# Linux-compatible wrapper.  The Python launcher is the single source of truth
# and also works on Windows PowerShell.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

ARGS=()
if [[ "${DRY_RUN:-1}" == "0" ]]; then
  ARGS+=(--run)
else
  ARGS+=(--dry-run)
fi
ARGS+=(--stage "${STAGE:-all}")
ARGS+=(--python-bin "${PYTHON_BIN:-python}")
ARGS+=(--data "${DATA:-DATA/}")
ARGS+=(--output-root "${OUTPUT_ROOT:-output/cusp_minimal_seed42}")
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  ARGS+=(--cuda-visible-devices "${CUDA_VISIBLE_DEVICES}")
fi

"${PYTHON_BIN:-python}" scripts/cusp_oracle_round1.py "${ARGS[@]}"
