#!/usr/bin/env bash
set -euo pipefail

# Safe default: one same-node, two-GPU job for the two Phase-1 topology cells.
# Override these environment variables only when expanding the confirmation:
#   ERI_STAGE, ERI_CASES (comma-separated), ERI_SEEDS (comma-separated),
#   ERI_LAUNCH_MODE, ERI_MAX_PARALLEL, DATA_ROOT, ERI_OUTPUT_ROOT, PYTHON_BIN.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="${REPO_DIR:-$(cd "${SCRIPT_DIR}/.." && pwd)}"
DATA_ROOT="${DATA_ROOT:-${REPO_DIR}/DATA}"
ERI_OUTPUT_ROOT="${ERI_OUTPUT_ROOT:-${REPO_DIR}/output/eri_closure_v1}"
ERI_STAGE="${ERI_STAGE:-train}"
ERI_CASES="${ERI_CASES:-clientlt_fedavg,matched_dirichlet_fedavg}"
ERI_SEEDS="${ERI_SEEDS:-42}"
ERI_MAX_PARALLEL="${ERI_MAX_PARALLEL:-2}"
ERI_LAUNCH_MODE="${ERI_LAUNCH_MODE:-two_gpu}"
PYTHON_BIN="${PYTHON_BIN:-python}"

if ! command -v sbatch >/dev/null 2>&1; then
  echo "sbatch is not available in PATH" >&2
  exit 2
fi
if [[ ! -f "${REPO_DIR}/scripts/run_eri_closure.py" ]]; then
  echo "Cannot find the ERI runner below REPO_DIR=${REPO_DIR}" >&2
  exit 2
fi
if [[ ! -f "${DATA_ROOT}/cifar-100/cifar-100-python/train" && ! -f "${DATA_ROOT}/cifar-100-python/train" ]]; then
  echo "Cannot find CIFAR-100 train below DATA_ROOT=${DATA_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${ERI_OUTPUT_ROOT}/protocol/eri_protocol.json" ]]; then
  echo "Run the protocol stage first; missing ${ERI_OUTPUT_ROOT}/protocol/eri_protocol.json" >&2
  exit 2
fi
if [[ ! "${ERI_MAX_PARALLEL}" =~ ^[1-9][0-9]*$ ]]; then
  echo "ERI_MAX_PARALLEL must be a positive integer" >&2
  exit 2
fi

IFS=',' read -r -a CASE_LIST <<< "${ERI_CASES}"
IFS=',' read -r -a SEED_LIST <<< "${ERI_SEEDS}"
TASK_COUNT=$(( ${#CASE_LIST[@]} * ${#SEED_LIST[@]} ))
if (( TASK_COUNT < 1 )); then
  echo "No ERI tasks were requested" >&2
  exit 2
fi
LAST_TASK=$(( TASK_COUNT - 1 ))

export REPO_DIR DATA_ROOT ERI_OUTPUT_ROOT ERI_STAGE ERI_CASES ERI_SEEDS PYTHON_BIN
if [[ "${ERI_LAUNCH_MODE}" == "two_gpu" ]]; then
  if (( TASK_COUNT != 2 || ${#CASE_LIST[@]} != 2 || ${#SEED_LIST[@]} != 1 )); then
    echo "two_gpu mode requires exactly two cases and one seed" >&2
    exit 2
  fi
  echo "Submitting one same-node job with two explicitly pinned GPU processes"
  echo "stage=${ERI_STAGE}; cases=${ERI_CASES}; seed=${ERI_SEEDS}"
  echo "output=${ERI_OUTPUT_ROOT}"
  exec sbatch "${REPO_DIR}/scripts/eri_closure_2gpu.sbatch"
fi
if [[ "${ERI_LAUNCH_MODE}" != "array" ]]; then
  echo "ERI_LAUNCH_MODE must be two_gpu or array" >&2
  exit 2
fi
echo "Submitting ${TASK_COUNT} ERI array task(s), maximum parallelism=${ERI_MAX_PARALLEL}"
echo "stage=${ERI_STAGE}; cases=${ERI_CASES}; seeds=${ERI_SEEDS}"
echo "output=${ERI_OUTPUT_ROOT}"
exec sbatch --array="0-${LAST_TASK}%${ERI_MAX_PARALLEL}" "${REPO_DIR}/scripts/eri_closure_slurm_array.sbatch"
