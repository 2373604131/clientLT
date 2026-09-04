#!/usr/bin/env bash
set -euo pipefail

# Local/interactive wrapper. For Slurm arrays use eri_closure_slurm_array.sbatch.
PYTHON_BIN="${PYTHON_BIN:-python}"
ERI_STAGE="${ERI_STAGE:-train}"
ERI_OUTPUT_ROOT="${ERI_OUTPUT_ROOT:-output/eri_closure_v1}"
DATA_ROOT="${DATA_ROOT:-DATA}"
ERI_CASES="${ERI_CASES:-clientlt_fedavg matched_dirichlet_fedavg}"
ERI_SEEDS="${ERI_SEEDS:-1 2 3 42 2026}"
EXTRA=()
if [[ "${ERI_SKIP_COMPLETED:-0}" == "1" ]]; then
  EXTRA+=(--skip-completed)
fi

exec "${PYTHON_BIN}" -u scripts/run_eri_closure.py \
  --stage "${ERI_STAGE}" \
  --output-root "${ERI_OUTPUT_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --cases ${ERI_CASES} \
  --seeds ${ERI_SEEDS} \
  "${EXTRA[@]}"
