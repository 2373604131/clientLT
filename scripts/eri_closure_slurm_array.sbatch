#!/usr/bin/env bash
# Submit with an explicit array size, e.g. sbatch --array=0-9 scripts/eri_closure_slurm_array.sbatch
# ERI_STAGE: train | analyze | replay. Protocol/verify/summary are single jobs.
# Adjust the SBATCH resources and module/conda section to the local cluster.
#SBATCH --job-name=eri-closure
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
# N16R4 assigns 55 GB host memory per requested GPU. Do not set --mem or
# --mem-per-* here; the site job_submit policy supplies the valid value.
#SBATCH --time=24:00:00
#SBATCH --export=ALL
#SBATCH --output=eri_%x_%A_%a.out
#SBATCH --error=eri_%x_%A_%a.err

set -euo pipefail

REPO_DIR="${REPO_DIR:?Set REPO_DIR to the shared CAPT checkout}"
DATA_ROOT="${DATA_ROOT:?Set DATA_ROOT to the shared DATA directory}"
ERI_OUTPUT_ROOT="${ERI_OUTPUT_ROOT:?Set ERI_OUTPUT_ROOT to shared experiment storage}"
ERI_STAGE="${ERI_STAGE:-train}"
PYTHON_BIN="${PYTHON_BIN:-python}"
ERI_CASES="${ERI_CASES:-clientlt_fedavg,matched_dirichlet_fedavg}"
ERI_SEEDS="${ERI_SEEDS:-42}"

# Site-specific examples (uncomment/adapt one):
# module purge; module load cuda/12.1 anaconda/2024
# source /path/to/miniconda3/etc/profile.d/conda.sh; conda activate capt

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
if (( ${#CASE_LIST[@]} == 0 || ${#SEED_LIST[@]} == 0 )); then
  echo "ERI_CASES and ERI_SEEDS must be non-empty comma-separated lists" >&2
  exit 2
fi
TASK="${SLURM_ARRAY_TASK_ID:-0}"
CASE_INDEX=$(( TASK % ${#CASE_LIST[@]} ))
SEED_INDEX=$(( TASK / ${#CASE_LIST[@]} ))
EXPECTED_TASKS=$(( ${#CASE_LIST[@]} * ${#SEED_LIST[@]} ))
if (( TASK < 0 || TASK >= EXPECTED_TASKS )); then
  echo "Array task ${TASK} exceeds ${#CASE_LIST[@]} cases x ${#SEED_LIST[@]} seeds" >&2
  exit 2
fi

CASE="${CASE_LIST[$CASE_INDEX]}"
SEED="${SEED_LIST[$SEED_INDEX]}"
echo "ERI ${ERI_STAGE}: case=${CASE}, seed=${SEED}, task=${TASK}/${EXPECTED_TASKS}"
echo "repo=${REPO_DIR}"
echo "data=${DATA_ROOT}"
echo "output=${ERI_OUTPUT_ROOT}"
echo "CUDA_VISIBLE_DEVICES=${CUDA_VISIBLE_DEVICES:-<unset>}"

"${PYTHON_BIN}" -u scripts/run_eri_closure.py \
  --stage "${ERI_STAGE}" \
  --output-root "${ERI_OUTPUT_ROOT}" \
  --data-root "${DATA_ROOT}" \
  --case "${CASE}" \
  --seeds "${SEED}" \
  --gpu 0 \
  --device cuda
