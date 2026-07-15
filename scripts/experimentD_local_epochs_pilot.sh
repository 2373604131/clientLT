#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA="${DATA:-DATA/}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-3 4}"
DRY_RUN="${DRY_RUN:-0}"
RERUN_FAILED="${RERUN_FAILED:-0}"
INCLUDE_LOCAL_E5="${INCLUDE_LOCAL_E5:-0}"

MODEL="fedavg"
TRAINER="PromptFL"
DATASET="cifar100_LT"
NUM_CLASSES="100"
TAIL_CLASS_COUNT="20"
CFG="vit_b16"

LR="0.001"
GAMMA="1"
USERS="30"
FRAC="1.0"
ROUND="30"
BATCH_SIZE="32"
TEST_BATCH_SIZE="64"
NUM_WORKERS="${NUM_WORKERS:-8}"
GLOBAL_EVAL_INTERVAL="1"
UPDATE_RETENTION_INTERVAL="1"
LOG_UPDATE_RETENTION="False"

NCTX="4"
N_GENERAL="1"
CTXINIT="False"
CSC="True"

IMB_FACTOR="0.01"
IMB_TYPE="exp"

PARTITION="client-longtail"
HEAD_CLIENT_RATIO="0.9"
TAIL_CLIENT_RATIO="0.1"
HEAD_CLASS_RATIO="0.8"
TAIL_CLASS_RATIO="0.2"
SPECIALIZATION_LAMBDA="0.75"
INTRA_GROUP_ALPHA="0.5"
HEAD_LEAKAGE_SCALE="3.0"

SEEDS="${SEEDS:-1}"
LOCAL_EPOCHS_LIST="1 3"
if [[ "${INCLUDE_LOCAL_E5}" == "1" ]]; then
  LOCAL_EPOCHS_LIST="${LOCAL_EPOCHS_LIST} 5"
fi

ISOLATE_LOCAL_OPTIMIZER_STATE="True"
FEDERATED_SINGLE_SCHEDULER_STEP="True"
EXPERIMENT_D_ROUNDS="5,10,20,30"
EXPERIMENT_D_INCLUDE_NORMALIZED="True"
EXPERIMENT_D_LOG_UPDATE_NORM="True"
EXPERIMENT_D_REQUIRE_FULL_PARTICIPATION="True"
EXPERIMENT_D_VERIFY_FEDAVG="True"
EXPERIMENT_D_EVAL_MODE="class_filtered"

DATASET_CONFIG="configs/datasets/${DATASET}.yaml"
TRAINER_CONFIG="configs/trainers/PromptFL/${CFG}.yaml"
BASE_OUTPUT_DIR="output/${DATASET}/${TRAINER}_${MODEL}_${CFG}_batchSize${BATCH_SIZE}/ExperimentD_LocalEpochPilot"

read -r -a GPU_ARRAY <<< "${GPUS}"
GPU_COUNT="${#GPU_ARRAY[@]}"
if [[ "${GPU_COUNT}" -le 0 ]]; then
  echo "No GPUs configured in GPUS='${GPUS}'"
  exit 1
fi

if [[ ! -f "federated_main.py" ]]; then
  echo "federated_main.py not found from ${REPO_ROOT}"
  exit 1
fi
if [[ ! -f "${DATASET_CONFIG}" ]]; then
  echo "Dataset config not found: ${DATASET_CONFIG}"
  exit 1
fi
if [[ ! -f "${TRAINER_CONFIG}" ]]; then
  echo "Trainer config not found: ${TRAINER_CONFIG}"
  exit 1
fi

print_command() {
  local gpu="$1"
  local -n command_ref=$2
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu}"
  printf '%q ' "${command_ref[@]}"
  printf '\n'
}

diagnostic_round_count() {
  local compact="${EXPERIMENT_D_ROUNDS// /}"
  if [[ -z "${compact}" ]]; then
    echo "0"
    return 0
  fi
  local -a rounds
  IFS=',' read -r -a rounds <<< "${compact}"
  echo "${#rounds[@]}"
}

csv_data_rows() {
  local file="$1"
  if [[ ! -f "${file}" ]]; then
    echo "-1"
    return 0
  fi
  local lines
  lines="$(wc -l < "${file}")"
  lines="${lines//[[:space:]]/}"
  if [[ -z "${lines}" || "${lines}" -le 0 ]]; then
    echo "0"
  else
    echo "$((lines - 1))"
  fi
}

check_csv_data_rows() {
  local file="$1"
  local expected="$2"
  local rows
  rows="$(csv_data_rows "${file}")"
  if [[ "${rows}" -ne "${expected}" ]]; then
    echo "Output validation failed: ${file} has ${rows} data rows, expected ${expected}"
    return 1
  fi
  echo "Output validation ok: ${file} has ${rows} data rows"
  return 0
}

validate_run_outputs() {
  local dir="$1"
  local diag_rounds
  diag_rounds="$(diagnostic_round_count)"
  local expected_per_class=$((TAIL_CLASS_COUNT * diag_rounds))
  local expected_update_norms=$((USERS * ROUND))
  local status=0

  check_csv_data_rows "${dir}/round_metrics.csv" "${ROUND}" || status=1
  check_csv_data_rows "${dir}/experiment_d/experiment_d_per_class.csv" "${expected_per_class}" || status=1
  check_csv_data_rows "${dir}/experiment_d/experiment_d_round_summary.csv" "${diag_rounds}" || status=1
  check_csv_data_rows "${dir}/experiment_d/client_update_norms.csv" "${expected_update_norms}" || status=1
  check_csv_data_rows "${dir}/experiment_d/client_update_norm_summary.csv" "${ROUND}" || status=1
  check_csv_data_rows "${dir}/experiment_d/runtime_metrics.csv" "${ROUND}" || status=1
  return "${status}"
}

backup_existing_run_dir() {
  local dir="$1"
  if [[ ! -d "${dir}" ]]; then
    return 0
  fi
  local stamp backup suffix
  stamp="$(date +%Y%m%d_%H%M%S)"
  backup="${dir}.rerun_backup_${stamp}"
  suffix=1
  while [[ -e "${backup}" ]]; do
    backup="${dir}.rerun_backup_${stamp}_${suffix}"
    suffix=$((suffix + 1))
  done
  echo "RERUN_FAILED=1: moving existing output directory to ${backup}"
  mv "${dir}" "${backup}"
}

build_cmd() {
  local seed="$1"
  local local_epochs="$2"
  local dir="$3"
  local schedule_file="$4"
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
    --local_epochs "${local_epochs}"
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
    --output-dir "${dir}"
    --imb_factor "${IMB_FACTOR}"
    --imb_type "${IMB_TYPE}"
    --partition "${PARTITION}"
    --head_client_ratio "${HEAD_CLIENT_RATIO}"
    --tail_client_ratio "${TAIL_CLIENT_RATIO}"
    --head_class_ratio "${HEAD_CLASS_RATIO}"
    --tail_class_ratio "${TAIL_CLASS_RATIO}"
    --specialization_lambda "${SPECIALIZATION_LAMBDA}"
    --intra_group_alpha "${INTRA_GROUP_ALPHA}"
    --head_leakage_scale "${HEAD_LEAKAGE_SCALE}"
    --train_batch_size "${BATCH_SIZE}"
    --test_batch_size "${TEST_BATCH_SIZE}"
    --global_eval_interval "${GLOBAL_EVAL_INTERVAL}"
    --num_classes "${NUM_CLASSES}"
    --client_schedule_file "${schedule_file}"
    --client_schedule_seed "${seed}"
    --log_update_retention "${LOG_UPDATE_RETENTION}"
    --update_retention_interval "${UPDATE_RETENTION_INTERVAL}"
    --experimentD_enable True
    --experimentD_rounds "${EXPERIMENT_D_ROUNDS}"
    --experimentD_include_normalized "${EXPERIMENT_D_INCLUDE_NORMALIZED}"
    --experimentD_log_update_norm "${EXPERIMENT_D_LOG_UPDATE_NORM}"
    --experimentD_require_full_participation "${EXPERIMENT_D_REQUIRE_FULL_PARTICIPATION}"
    --experimentD_verify_fedavg "${EXPERIMENT_D_VERIFY_FEDAVG}"
    --experimentD_eval_mode "${EXPERIMENT_D_EVAL_MODE}"
    DATALOADER.NUM_WORKERS "${NUM_WORKERS}"
  )
}

task_count() {
  local count=0
  local _seed _local_epochs
  for _local_epochs in ${LOCAL_EPOCHS_LIST}; do
    for _seed in ${SEEDS}; do
      count=$((count + 1))
    done
  done
  echo "${count}"
}

TOTAL_RUNS="$(task_count)"

handle_task() {
  local gpu="$1"
  local slot="$2"
  local run_id="$3"
  local seed="$4"
  local local_epochs="$5"
  local dir="$6"
  local schedule_file="$7"
  local run_label
  run_label="$(printf "%02d" "${run_id}")"

  build_cmd "${seed}" "${local_epochs}" "${dir}" "${schedule_file}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[ExperimentD Pilot ${run_label}/${TOTAL_RUNS}] gpu=${gpu} slot=${slot}"
    echo "partition=${PARTITION}"
    echo "local_epochs=${local_epochs}"
    echo "seed=${seed}"
    echo "output directory=${dir}"
    echo "schedule file=${schedule_file}"
    print_command "${gpu}" CMD
    return 0
  fi

  local finished="${dir}/finished.flag"
  local failed="${dir}/failed.flag"
  local lock="${dir}/running.lock"

  if [[ -f "${finished}" ]]; then
    echo "[GPU ${gpu}] Results completed at ${dir} (skip)"
    return 0
  fi
  if [[ -d "${lock}" ]]; then
    echo "[GPU ${gpu}] Task already running at ${dir} (skip)"
    return 0
  fi
  if [[ -d "${dir}" && "${RERUN_FAILED}" == "1" ]]; then
    backup_existing_run_dir "${dir}"
  elif [[ -d "${dir}" && ! -f "${failed}" ]]; then
    echo "[GPU ${gpu}] Incomplete output directory exists at ${dir} (skip; set RERUN_FAILED=1 to back it up and rerun cleanly)"
    return 0
  fi
  if [[ -f "${failed}" && "${RERUN_FAILED}" != "1" ]]; then
    echo "[GPU ${gpu}] Previous failure at ${dir} (skip; set RERUN_FAILED=1 to retry)"
    return 0
  fi

  mkdir -p "${dir}"
  if ! mkdir "${lock}" 2>/dev/null; then
    echo "[GPU ${gpu}] Task already running at ${dir} (skip)"
    return 0
  fi
  rm -f "${failed}"

  {
    echo "[ExperimentD Pilot ${run_label}/${TOTAL_RUNS}]"
    echo "worker_gpu=${gpu}"
    echo "partition=${PARTITION}"
    echo "local_epochs=${local_epochs}"
    echo "seed=${seed}"
    echo "output directory=${dir}"
    echo "schedule file=${schedule_file}"
    echo "start time=$(date -Is)"
    print_command "${gpu}" CMD
  } >> "${dir}/run.log"

  echo "[GPU ${gpu}] Starting [ExperimentD Pilot ${run_label}/${TOTAL_RUNS}] local_epochs=${local_epochs} seed=${seed}"
  if CUDA_VISIBLE_DEVICES="${gpu}" "${CMD[@]}" 2>&1 | tee -a "${dir}/run.log"; then
    if validate_run_outputs "${dir}" >> "${dir}/run.log" 2>&1; then
      touch "${finished}"
      echo "[GPU ${gpu}] Finished ${dir}" | tee -a "${dir}/run.log"
      rm -rf "${lock}"
    else
      touch "${failed}"
      echo "[GPU ${gpu}] Output validation failed for ${dir}" | tee -a "${dir}/run.log"
      rm -rf "${lock}"
      return 1
    fi
  else
    touch "${failed}"
    echo "[GPU ${gpu}] Experiment failed for ${dir}" | tee -a "${dir}/run.log"
    rm -rf "${lock}"
    return 1
  fi
}

enumerate_tasks() {
  local callback="$1"
  local gpu="$2"
  local slot="${3:-all}"
  local run_id=0
  local seed local_epochs schedule_file dir

  for local_epochs in ${LOCAL_EPOCHS_LIST}; do
    for seed in ${SEEDS}; do
      run_id=$((run_id + 1))
      schedule_file="output/experimentD_shared_schedules/pilot_users${USERS}_frac${FRAC}_round${ROUND}_seed${seed}.json"
      dir="${BASE_OUTPUT_DIR}/client-longtail_lambda=${SPECIALIZATION_LAMBDA}_alpha=${INTRA_GROUP_ALPHA}_rho=${HEAD_LEAKAGE_SCALE}_localE=${local_epochs}_seed=${seed}"
      if [[ "${slot}" == "all" || $(((run_id - 1) % GPU_COUNT)) -eq "${slot}" ]]; then
        "${callback}" "${gpu}" "${slot}" "${run_id}" "${seed}" "${local_epochs}" "${dir}" "${schedule_file}"
      fi
    done
  done

  if [[ "${run_id}" -ne "${TOTAL_RUNS}" ]]; then
    echo "ExperimentD pilot matrix error: expected ${TOTAL_RUNS} runs, got ${run_id}"
    return 1
  fi
}

worker() {
  local gpu="$1"
  local slot="$2"
  echo "[GPU ${gpu}] ExperimentD pilot worker started (slot ${slot}/${GPU_COUNT})"
  enumerate_tasks handle_task "${gpu}" "${slot}"
  echo "[GPU ${gpu}] ExperimentD pilot worker finished"
}

prepare_shared_schedules() {
  local seed schedule_file
  for seed in ${SEEDS}; do
    schedule_file="output/experimentD_shared_schedules/pilot_users${USERS}_frac${FRAC}_round${ROUND}_seed${seed}.json"
    echo "Preparing ExperimentD pilot shared schedule for seed=${seed}: ${schedule_file}"
    "${PYTHON_BIN}" scripts/create_client_schedule.py \
      --path "${schedule_file}" \
      --num_rounds "${ROUND}" \
      --num_users "${USERS}" \
      --frac "${FRAC}" \
      --seed "${seed}"
  done
}

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "ExperimentD local-epochs pilot dry run; no files, schedules, locks, or Python training processes will be created."
  for slot in "${!GPU_ARRAY[@]}"; do
    worker "${GPU_ARRAY[slot]}" "${slot}"
  done
  exit 0
fi

prepare_shared_schedules

pids=()
for slot in "${!GPU_ARRAY[@]}"; do
  worker "${GPU_ARRAY[slot]}" "${slot}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done

if [[ "${status}" -ne 0 ]]; then
  echo "At least one ExperimentD pilot worker failed."
  exit "${status}"
fi

echo "All ExperimentD pilot workers finished."
