#!/bin/bash
#
# CAPT partition comparison: fine-class Dirichlet vs Client-LT.
#
# This keeps CAPT's original cluster aggregation path.  The shared training
# settings follow Experiment D; only --partition and its partition-specific
# arguments differ between the two conditions.  Experiment D's FedAvg
# counterfactual diagnostics are deliberately not enabled here: they are not
# executed by the CAPT/cluster training branch.

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Override any of these from the environment, for example:
# GPU=1 SEEDS="1" ROUND=5 DRY_RUN=1 bash scripts/capt.sh
DATA="${DATA:-DATA/}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPU="${GPU:-0}"
DRY_RUN="${DRY_RUN:-0}"

# Keep the original CAPT path. Do not change these to PromptFL/fedavg when
# interpreting this as a CAPT comparison.
MODEL="cluster"
TRAINER="CAPT"
DATASET="cifar100_LT"
CFG="vit_b16"
NUM_CLASSES="100"

# Shared Experiment-D training protocol.
LR="0.001"
GAMMA="1"
USERS="30"
FRAC="1.0"
ROUND="100"
LOCAL_EPOCHS="3"
BATCH_SIZE="32"
TEST_BATCH_SIZE="64"
NUM_WORKERS="${NUM_WORKERS:-8}"
GLOBAL_EVAL_INTERVAL="1"

NCTX="4"
N_GENERAL="1"
CTXINIT="False"
CSC="True"
SIMCLUST="4"
DISCLUSTERS="4"

IMB_FACTOR="0.01"
IMB_TYPE="exp"
TAIL_CLASS_RATIO="0.2"

# Experiment-D partition settings. Alpha has different semantics for the two
# protocols, so numerical equality is a protocol setting rather than a claim
# of matched client topology.
ALPHA="${ALPHA:-0.5}"
SPECIALIZATION_LAMBDA="${SPECIALIZATION_LAMBDA:-0.75}"
HEAD_LEAKAGE_SCALE="${HEAD_LEAKAGE_SCALE:-3.0}"
HEAD_CLIENT_RATIO="${HEAD_CLIENT_RATIO:-0.9}"
TAIL_CLIENT_RATIO="${TAIL_CLIENT_RATIO:-0.1}"
HEAD_CLASS_RATIO="${HEAD_CLASS_RATIO:-0.8}"

SEEDS="${SEEDS:-1 42 2026}"
PARTITIONS="${PARTITIONS:-noniid-labeldir-fine client-longtail}"

DATASET_CONFIG="configs/datasets/${DATASET}.yaml"
TRAINER_CONFIG="configs/trainers/CAPT/${CFG}.yaml"
BASE_OUTPUT_DIR="output/${DATASET}/${TRAINER}_${MODEL}_${CFG}_batchSize${BATCH_SIZE}/ExperimentD_AlignedPartitionCompare"
SCHEDULE_DIR="output/capt_shared_schedules"

if [[ ! -f "federated_main.py" ]]; then
  echo "federated_main.py not found from ${REPO_ROOT}" >&2
  exit 1
fi
if [[ ! -f "${DATASET_CONFIG}" ]]; then
  echo "Dataset config not found: ${DATASET_CONFIG}" >&2
  exit 1
fi
if [[ ! -f "${TRAINER_CONFIG}" ]]; then
  echo "CAPT config not found: ${TRAINER_CONFIG}" >&2
  exit 1
fi

print_command() {
  local -n command_ref="$1"
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU}"
  printf '%q ' "${command_ref[@]}"
  printf '\n'
}

prepare_schedule() {
  local seed="$1"
  local schedule_file="$2"
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  "${PYTHON_BIN}" scripts/create_client_schedule.py \
    --path "${schedule_file}" \
    --num_rounds "${ROUND}" \
    --num_users "${USERS}" \
    --frac "${FRAC}" \
    --seed "${seed}"
}

build_common_cmd() {
  local seed="$1"
  local output_dir="$2"
  local schedule_file="$3"
  CMD=(
    "${PYTHON_BIN}" federated_main.py
    --root "${DATA}"
    --model "${MODEL}"
    --trainer "${TRAINER}"
    --dataset "${DATASET}"
    --seed "${seed}"
    --split_seed "${seed}"
    --num_users "${USERS}"
    --frac "${FRAC}"
    --round "${ROUND}"
    --local_epochs "${LOCAL_EPOCHS}"
    --lr "${LR}"
    --gamma "${GAMMA}"
    --n_ctx "${NCTX}"
    --n_general "${N_GENERAL}"
    --ctx_init "${CTXINIT}"
    --csc "${CSC}"
    --dataset-config-file "${DATASET_CONFIG}"
    --config-file "${TRAINER_CONFIG}"
    --output-dir "${output_dir}"
    --imb_factor "${IMB_FACTOR}"
    --imb_type "${IMB_TYPE}"
    --tail_class_ratio "${TAIL_CLASS_RATIO}"
    --train_batch_size "${BATCH_SIZE}"
    --test_batch_size "${TEST_BATCH_SIZE}"
    --global_eval_interval "${GLOBAL_EVAL_INTERVAL}"
    --num_classes "${NUM_CLASSES}"
    --n_simclusters "${SIMCLUST}"
    --n_disclusters "${DISCLUSTERS}"
    --client_schedule_file "${schedule_file}"
    --client_schedule_seed "${seed}"
  )
}

append_partition_args() {
  local partition="$1"
  if [[ "${partition}" == "noniid-labeldir-fine" ]]; then
    CMD+=(--partition "${partition}" --beta "${ALPHA}")
  elif [[ "${partition}" == "client-longtail" ]]; then
    CMD+=(
      --partition "${partition}"
      --head_client_ratio "${HEAD_CLIENT_RATIO}"
      --tail_client_ratio "${TAIL_CLIENT_RATIO}"
      --head_class_ratio "${HEAD_CLASS_RATIO}"
      --specialization_lambda "${SPECIALIZATION_LAMBDA}"
      --intra_group_alpha "${ALPHA}"
      --head_leakage_scale "${HEAD_LEAKAGE_SCALE}"
    )
  else
    echo "Unsupported partition '${partition}'. Use noniid-labeldir-fine or client-longtail." >&2
    exit 1
  fi
  CMD+=(DATALOADER.NUM_WORKERS "${NUM_WORKERS}")
}

run_condition() {
  local partition="$1"
  local seed="$2"
  local output_dir="$3"
  local schedule_file="$4"

  build_common_cmd "${seed}" "${output_dir}" "${schedule_file}"
  append_partition_args "${partition}"

  echo "partition=${partition} seed=${seed} output=${output_dir}"
  print_command CMD
  if [[ "${DRY_RUN}" == "1" ]]; then
    return 0
  fi
  if [[ -d "${output_dir}" ]]; then
    echo "Output directory already exists; skipping to protect prior results: ${output_dir}"
    return 0
  fi
  mkdir -p "${output_dir}"
  CUDA_VISIBLE_DEVICES="${GPU}" "${CMD[@]}" 2>&1 | tee "${output_dir}/run.log"
}

for seed in ${SEEDS}; do
  schedule_file="${SCHEDULE_DIR}/capt_users${USERS}_frac${FRAC}_round${ROUND}_seed${seed}.json"
  prepare_schedule "${seed}" "${schedule_file}"
  for partition in ${PARTITIONS}; do
    if [[ "${partition}" == "noniid-labeldir-fine" ]]; then
      output_dir="${BASE_OUTPUT_DIR}/partition=noniid-labeldir-fine_beta=${ALPHA}_IF=${IMB_FACTOR}_localE=${LOCAL_EPOCHS}_seed=${seed}"
    else
      output_dir="${BASE_OUTPUT_DIR}/partition=client-longtail_lambda=${SPECIALIZATION_LAMBDA}_alpha=${ALPHA}_rho=${HEAD_LEAKAGE_SCALE}_IF=${IMB_FACTOR}_localE=${LOCAL_EPOCHS}_seed=${seed}"
    fi
    run_condition "${partition}" "${seed}" "${output_dir}" "${schedule_file}"
  done
done

echo "CAPT partition comparison complete."
