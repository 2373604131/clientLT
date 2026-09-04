#!/usr/bin/env bash
# One node, two GPUs, two explicitly pinned ERI condition processes.
# The job starts only after Slurm can allocate both GPUs on the same node.
#SBATCH --job-name=eri-2gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:2
#SBATCH --cpus-per-task=16
# N16R4 supplies 55 GB host memory per GPU (110 GB for this job). Do not add
# --mem or --mem-per-*; the site job_submit policy supplies the valid amount.
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --output=eri_%x_%j.out
#SBATCH --error=eri_%x_%j.err

set -euo pipefail

REPO_DIR="${REPO_DIR:?Set REPO_DIR to the shared CAPT checkout}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the shared DATA directory}"
ERI_OUTPUT_ROOT="${ERI_OUTPUT_ROOT:?Set ERI_OUTPUT_ROOT to shared experiment storage}"
ERI_STAGE="${ERI_STAGE:-train}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ERI_CASES="${ERI_CASES:-clientlt_fedavg,matched_dirichlet_fedavg}"
ERI_SEEDS="${ERI_SEEDS:-42}"

cd "${REPO_DIR}"
if [[ ! -f "scripts/run_eri_closure.py" ]]; then
  echo "Invalid REPO_DIR (missing scripts/run_eri_closure.py): ${REPO_DIR}" >&2
  exit 2
fi
if [[ ! -f "${DATA_ROOT}/cifar-100/cifar-100-python/train" && ! -f "${DATA_ROOT}/cifar-100-python/train" ]]; then
  echo "Invalid DATA_ROOT (cannot find CIFAR-100 train): ${DATA_ROOT}" >&2
  exit 2
fi
if [[ ! -f "${ERI_OUTPUT_ROOT}/protocol/eri_protocol.json" ]]; then
  echo "Missing frozen ERI protocol: ${ERI_OUTPUT_ROOT}/protocol/eri_protocol.json" >&2
  exit 2
fi

IFS=',' read -r -a CASE_LIST <<< "${ERI_CASES}"
IFS=',' read -r -a SEED_LIST <<< "${ERI_SEEDS}"
if (( ${#CASE_LIST[@]} != 2 )); then
  echo "Two-GPU mode requires exactly two comma-separated ERI_CASES; got ${ERI_CASES}" >&2
  exit 2
fi
if (( ${#SEED_LIST[@]} != 1 )); then
  echo "Two-GPU mode requires exactly one ERI seed; got ${ERI_SEEDS}" >&2
  exit 2
fi

# Slurm exposes the two allocated physical devices through CUDA_VISIBLE_DEVICES.
# Split that allocation explicitly: condition 0 gets the first visible device,
# condition 1 gets the second. Each child therefore sees exactly one cuda:0.
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  IFS=',' read -r -a ALLOCATED_GPUS <<< "${CUDA_VISIBLE_DEVICES}"
else
  ALLOCATED_GPUS=(0 1)
fi
if (( ${#ALLOCATED_GPUS[@]} != 2 )); then
  echo "Expected exactly two allocated GPUs, CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}" >&2
  exit 2
fi

SEED="${SEED_LIST[0]}"
LOG_DIR="${ERI_OUTPUT_ROOT}/slurm_logs/job_${SLURM_JOB_ID:-manual}"
mkdir -p "${LOG_DIR}"
PIDS=()

terminate_children() {
  for pid in "${PIDS[@]:-}"; do
    kill "${pid}" 2>/dev/null || true
  done
}
trap terminate_children INT TERM

for slot in 0 1; do
  CASE="${CASE_LIST[$slot]}"
  GPU_TOKEN="${ALLOCATED_GPUS[$slot]}"
  LOG_FILE="${LOG_DIR}/${CASE}_seed${SEED}_gpu${slot}.log"
  echo "Launching ${CASE}/seed${SEED}: logical_slot=${slot}, allocated_device=${GPU_TOKEN}, log=${LOG_FILE}"
  (
    export CUDA_VISIBLE_DEVICES="${GPU_TOKEN}"
    export OMP_NUM_THREADS=8
    "${PYTHON_BIN}" -u scripts/run_eri_closure.py \
      --stage "${ERI_STAGE}" \
      --output-root "${ERI_OUTPUT_ROOT}" \
      --data-root "${DATA_ROOT}" \
      --case "${CASE}" \
      --seeds "${SEED}" \
      --gpu 0 \
      --device cuda
  ) >"${LOG_FILE}" 2>&1 &
  PIDS+=("$!")
done

STATUS=0
for slot in 0 1; do
  if ! wait "${PIDS[$slot]}"; then
    echo "FAILED: ${CASE_LIST[$slot]}/seed${SEED}; see ${LOG_DIR}" >&2
    STATUS=1
  else
    echo "FINISHED: ${CASE_LIST[$slot]}/seed${SEED}"
  fi
done
exit "${STATUS}"
