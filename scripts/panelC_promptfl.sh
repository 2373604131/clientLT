#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA="${DATA:-DATA/}"
GPU="${GPU:-0}"
PYTHON_BIN="${PYTHON_BIN:-python}"
DRY_RUN="${DRY_RUN:-0}"

MODEL="fedavg"
TRAINER="PromptFL"

DATASETS="cifar100_LT"
CFG="vit_b16"

LR="0.001"
GAMMA="1"

USERS="30"
FRAC="1.0"
ROUND="100"
LOCAL_EPOCHS="3"

BATCH_SIZE="32"
TEST_BATCH_SIZE="64"
NUM_WORKERS="0"

GLOBAL_EVAL_INTERVAL="5"
UPDATE_RETENTION_INTERVAL="5"
LOG_UPDATE_RETENTION="False"

NCTX="4"
N_GENERAL="1"
CTXINIT="False"
CSC="True"

IMB_FACTOR="0.01"
IMB_TYPE="exp"

HEAD_CLIENT_RATIO="0.9"
TAIL_CLIENT_RATIO="0.1"
HEAD_CLASS_RATIO="0.8"
TAIL_CLASS_RATIO="0.2"

SPECIALIZATION_LAMBDA="0.75"
HEAD_LEAKAGE_SCALE="3.0"

SEEDS="1 42 2026"
ALPHAS="0.1 0.25 0.5 0.75 1.0"

ISOLATE_LOCAL_OPTIMIZER_STATE="True"
FEDERATED_SINGLE_SCHEDULER_STEP="True"

TOTAL_RUNS="30"
PLANNED_RUNS=0

print_command() {
  local -n command_ref=$1
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${GPU}"
  printf '%q ' "${command_ref[@]}"
  printf '\n'
}

run_PanelC_condition() {
  local protocol="$1"
  local parameter_name="$2"
  local parameter_value="$3"
  local seed="$4"
  local dir="$5"
  local schedule_file="$6"
  shift 6

  PLANNED_RUNS=$((PLANNED_RUNS + 1))
  local run_id
  run_id="$(printf "%02d" "${PLANNED_RUNS}")"
  local finished="${dir}/finished.flag"
  local -a cmd=("$@")

  echo "[PanelC ${run_id}/${TOTAL_RUNS}]"
  echo "protocol=${protocol}"
  echo "${parameter_name}=${parameter_value}"
  echo "seed=${seed}"
  echo "output directory=${dir}"
  echo "schedule file=${schedule_file}"
  print_command cmd

  if [[ "${DRY_RUN}" == "1" ]]; then
    return
  fi

  if [[ -f "${finished}" ]]; then
    echo "Results completed at ${dir} (skip)"
    return
  fi

  mkdir -p "${dir}"
  {
    echo "[PanelC ${run_id}/${TOTAL_RUNS}]"
    echo "protocol=${protocol}"
    echo "${parameter_name}=${parameter_value}"
    echo "seed=${seed}"
    echo "output directory=${dir}"
    echo "schedule file=${schedule_file}"
    echo "start time=$(date -Is)"
    print_command cmd
  } >> "${dir}/run.log"

  if CUDA_VISIBLE_DEVICES="${GPU}" "${cmd[@]}" 2>&1 | tee -a "${dir}/run.log"; then
    if [[ -s "${dir}/round_metrics.csv" ]]; then
      touch "${finished}"
    else
      echo "round_metrics.csv missing or empty at ${dir}" | tee -a "${dir}/run.log"
      exit 1
    fi
  else
    echo "Experiment failed for ${dir}" | tee -a "${dir}/run.log"
    exit 1
  fi
}

if [[ ! -f "federated_main.py" ]]; then
  echo "federated_main.py not found from ${REPO_ROOT}"
  exit 1
fi

for DATASET in ${DATASETS}; do
  case "${DATASET}" in
    cifar100_LT)
      NUM_CLASSES="100"
      ;;
    *)
      echo "Unknown PanelC dataset: ${DATASET}"
      exit 1
      ;;
  esac

  DATASET_CONFIG="configs/datasets/${DATASET}.yaml"
  TRAINER_CONFIG="configs/trainers/PromptFL/${CFG}.yaml"

  if [[ ! -f "${DATASET_CONFIG}" ]]; then
    echo "Dataset config not found: ${DATASET_CONFIG}"
    exit 1
  fi
  if [[ ! -f "${TRAINER_CONFIG}" ]]; then
    echo "Trainer config not found: ${TRAINER_CONFIG}"
    exit 1
  fi

  BASE_OUTPUT_DIR="output/${DATASET}/${TRAINER}_${MODEL}_${CFG}_batchSize${BATCH_SIZE}/PanelC_users${USERS}_localE${LOCAL_EPOCHS}"

  for SEED in ${SEEDS}; do
    SCHEDULE_FILE="output/panelC_shared_schedules/users${USERS}_frac${FRAC}_round${ROUND}_seed${SEED}.json"

    for ALPHA in ${ALPHAS}; do
      PARTITION="noniid-labeldir-fine"
      DIR="${BASE_OUTPUT_DIR}/partition=${PARTITION}_alpha=${ALPHA}_IF=${IMB_FACTOR}_localE=${LOCAL_EPOCHS}_seed=${SEED}"
      CMD=(
        "${PYTHON_BIN}" federated_main.py
        --root "${DATA}"
        --model "${MODEL}"
        --trainer "${TRAINER}"
        --dataset "${DATASET}"
        --seed "${SEED}"
        --split_seed "${SEED}"
        --num_users "${USERS}"
        --frac "${FRAC}"
        --round "${ROUND}"
        --local_epochs "${LOCAL_EPOCHS}"
        --isolate_local_optimizer_state "${ISOLATE_LOCAL_OPTIMIZER_STATE}"
        --federated_single_scheduler_step "${FEDERATED_SINGLE_SCHEDULER_STEP}"
        --lr "${LR}"
        --gamma "${GAMMA}"
        --n_ctx "${NCTX}"
        --n_general "${N_GENERAL}"
        --ctx_init "${CTXINIT}"
        --csc "${CSC}"
        --dataset-config-file "${DATASET_CONFIG}"
        --config-file "${TRAINER_CONFIG}"
        --output-dir "${DIR}"
        --imb_factor "${IMB_FACTOR}"
        --imb_type "${IMB_TYPE}"
        --train_batch_size "${BATCH_SIZE}"
        --test_batch_size "${TEST_BATCH_SIZE}"
        --global_eval_interval "${GLOBAL_EVAL_INTERVAL}"
        --num_classes "${NUM_CLASSES}"
        --tail_class_ratio "${TAIL_CLASS_RATIO}"
        --client_schedule_file "${SCHEDULE_FILE}"
        --client_schedule_seed "${SEED}"
        --log_update_retention "${LOG_UPDATE_RETENTION}"
        --update_retention_interval "${UPDATE_RETENTION_INTERVAL}"
        --update_retention_param_key prompt_learner.class_aware_ctx
        --partition "${PARTITION}"
        --beta "${ALPHA}"
        DATALOADER.NUM_WORKERS "${NUM_WORKERS}"
      )
      run_PanelC_condition "Dirichlet" "alpha" "${ALPHA}" "${SEED}" "${DIR}" "${SCHEDULE_FILE}" "${CMD[@]}"
    done

    for ALPHA in ${ALPHAS}; do
      PARTITION="client-longtail"
      DIR="${BASE_OUTPUT_DIR}/partition=${PARTITION}_lambda=${SPECIALIZATION_LAMBDA}_alpha=${ALPHA}_rho=${HEAD_LEAKAGE_SCALE}_IF=${IMB_FACTOR}_localE=${LOCAL_EPOCHS}_seed=${SEED}"
      CMD=(
        "${PYTHON_BIN}" federated_main.py
        --root "${DATA}"
        --model "${MODEL}"
        --trainer "${TRAINER}"
        --dataset "${DATASET}"
        --seed "${SEED}"
        --split_seed "${SEED}"
        --num_users "${USERS}"
        --frac "${FRAC}"
        --round "${ROUND}"
        --local_epochs "${LOCAL_EPOCHS}"
        --isolate_local_optimizer_state "${ISOLATE_LOCAL_OPTIMIZER_STATE}"
        --federated_single_scheduler_step "${FEDERATED_SINGLE_SCHEDULER_STEP}"
        --lr "${LR}"
        --gamma "${GAMMA}"
        --n_ctx "${NCTX}"
        --n_general "${N_GENERAL}"
        --ctx_init "${CTXINIT}"
        --csc "${CSC}"
        --dataset-config-file "${DATASET_CONFIG}"
        --config-file "${TRAINER_CONFIG}"
        --output-dir "${DIR}"
        --imb_factor "${IMB_FACTOR}"
        --imb_type "${IMB_TYPE}"
        --train_batch_size "${BATCH_SIZE}"
        --test_batch_size "${TEST_BATCH_SIZE}"
        --global_eval_interval "${GLOBAL_EVAL_INTERVAL}"
        --num_classes "${NUM_CLASSES}"
        --tail_class_ratio "${TAIL_CLASS_RATIO}"
        --client_schedule_file "${SCHEDULE_FILE}"
        --client_schedule_seed "${SEED}"
        --log_update_retention True
        --update_retention_interval "${UPDATE_RETENTION_INTERVAL}"
        --update_retention_param_key prompt_learner.class_aware_ctx
        --partition "${PARTITION}"
        --head_client_ratio "${HEAD_CLIENT_RATIO}"
        --tail_client_ratio "${TAIL_CLIENT_RATIO}"
        --head_class_ratio "${HEAD_CLASS_RATIO}"
        --tail_class_ratio "${TAIL_CLASS_RATIO}"
        --specialization_lambda "${SPECIALIZATION_LAMBDA}"
        --intra_group_alpha "${ALPHA}"
        --head_leakage_scale "${HEAD_LEAKAGE_SCALE}"
        DATALOADER.NUM_WORKERS "${NUM_WORKERS}"
      )
      run_PanelC_condition "Client-LT" "alpha" "${ALPHA}" "${SEED}" "${DIR}" "${SCHEDULE_FILE}" "${CMD[@]}"
    done
  done
done

if [[ "${PLANNED_RUNS}" -ne "${TOTAL_RUNS}" ]]; then
  echo "PanelC matrix error: expected ${TOTAL_RUNS} runs, got ${PLANNED_RUNS}"
  exit 1
fi
