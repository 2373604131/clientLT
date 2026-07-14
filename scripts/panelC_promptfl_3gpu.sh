#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

DATA="${DATA:-DATA/}"
PYTHON_BIN="${PYTHON_BIN:-python}"
GPUS="${GPUS:-0 1 2}"
DRY_RUN="${DRY_RUN:-0}"
RERUN_FAILED="${RERUN_FAILED:-0}"

MODEL="fedavg"
TRAINER="PromptFL"

DATASET="cifar100_LT"
NUM_CLASSES="100"
CFG="vit_b16"

LR="0.001"
GAMMA="1"

USERS="50"
FRAC="1.0"
ROUND="100"
LOCAL_EPOCHS="5"

BATCH_SIZE="32"
TEST_BATCH_SIZE="64"
NUM_WORKERS="4"

GLOBAL_EVAL_INTERVAL="5"
UPDATE_RETENTION_INTERVAL="5"

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
DATASET_CONFIG="configs/datasets/${DATASET}.yaml"
TRAINER_CONFIG="configs/trainers/PromptFL/${CFG}.yaml"
BASE_OUTPUT_DIR="output/${DATASET}/${TRAINER}_${MODEL}_${CFG}_batchSize${BATCH_SIZE}/PanelC"

read -r -a GPU_ARRAY <<< "${GPUS}"
GPU_COUNT="${#GPU_ARRAY[@]}"
if [[ "${GPU_COUNT}" -le 0 ]]; then
  echo "No GPUs configured in GPUS='${GPUS}'"
  exit 1
fi

print_command() {
  local gpu="$1"
  local -n command_ref=$2
  printf 'CUDA_VISIBLE_DEVICES=%q ' "${gpu}"
  printf '%q ' "${command_ref[@]}"
  printf '\n'
}

build_common_cmd() {
  local seed="$1"
  local dir="$2"
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
    --train_batch_size "${BATCH_SIZE}"
    --test_batch_size "${TEST_BATCH_SIZE}"
    --global_eval_interval "${GLOBAL_EVAL_INTERVAL}"
    --num_classes "${NUM_CLASSES}"
    --tail_class_ratio "${TAIL_CLASS_RATIO}"
    --client_schedule_file "${schedule_file}"
    --client_schedule_seed "${seed}"
    --log_update_retention True
    --update_retention_interval "${UPDATE_RETENTION_INTERVAL}"
    --update_retention_param_key prompt_learner.class_aware_ctx
  )
}

append_dirichlet_args() {
  local beta="$1"
  CMD+=(
    --partition noniid-labeldir-fine
    --beta "${beta}"
    DATALOADER.NUM_WORKERS "${NUM_WORKERS}"
  )
}

append_clientlt_args() {
  local alpha="$1"
  CMD+=(
    --partition client-longtail
    --head_client_ratio "${HEAD_CLIENT_RATIO}"
    --tail_client_ratio "${TAIL_CLIENT_RATIO}"
    --head_class_ratio "${HEAD_CLASS_RATIO}"
    --tail_class_ratio "${TAIL_CLASS_RATIO}"
    --specialization_lambda "${SPECIALIZATION_LAMBDA}"
    --intra_group_alpha "${alpha}"
    --head_leakage_scale "${HEAD_LEAKAGE_SCALE}"
    DATALOADER.NUM_WORKERS "${NUM_WORKERS}"
  )
}

prepare_task_command() {
  local protocol="$1"
  local parameter_value="$2"
  local seed="$3"
  local dir="$4"
  local schedule_file="$5"

  build_common_cmd "${seed}" "${dir}" "${schedule_file}"
  if [[ "${protocol}" == "Dirichlet" ]]; then
    append_dirichlet_args "${parameter_value}"
  elif [[ "${protocol}" == "Client-LT" ]]; then
    append_clientlt_args "${parameter_value}"
  else
    echo "Unknown protocol: ${protocol}"
    exit 1
  fi
}

handle_task() {
  local gpu="$1"
  local run_id="$2"
  local protocol="$3"
  local parameter_name="$4"
  local parameter_value="$5"
  local seed="$6"
  local dir="$7"
  local schedule_file="$8"
  local run_label
  run_label="$(printf "%02d" "${run_id}")"

  prepare_task_command "${protocol}" "${parameter_value}" "${seed}" "${dir}" "${schedule_file}"

  if [[ "${DRY_RUN}" == "1" ]]; then
    echo "[PanelC ${run_label}/${TOTAL_RUNS}] gpu=${gpu}"
    echo "protocol=${protocol}"
    echo "${parameter_name}=${parameter_value}"
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
    echo "[PanelC ${run_label}/${TOTAL_RUNS}]"
    echo "worker_gpu=${gpu}"
    echo "protocol=${protocol}"
    echo "${parameter_name}=${parameter_value}"
    echo "seed=${seed}"
    echo "output directory=${dir}"
    echo "schedule file=${schedule_file}"
    echo "start time=$(date -Is)"
    print_command "${gpu}" CMD
  } >> "${dir}/run.log"

  echo "[GPU ${gpu}] Starting [PanelC ${run_label}/${TOTAL_RUNS}] ${protocol} ${parameter_name}=${parameter_value} seed=${seed}"

  if CUDA_VISIBLE_DEVICES="${gpu}" "${CMD[@]}" 2>&1 | tee -a "${dir}/run.log"; then
    if [[ -s "${dir}/round_metrics.csv" ]]; then
      touch "${finished}"
      echo "[GPU ${gpu}] Finished ${dir}" | tee -a "${dir}/run.log"
      rm -rf "${lock}"
    else
      touch "${failed}"
      echo "[GPU ${gpu}] round_metrics.csv missing or empty at ${dir}" | tee -a "${dir}/run.log"
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
  local schedule_file
  local dir

  for seed in ${SEEDS}; do
    schedule_file="output/panelC_shared_schedules/users${USERS}_frac${FRAC}_round${ROUND}_seed${seed}.json"

    for alpha in ${ALPHAS}; do
      run_id=$((run_id + 1))
      if [[ "${slot}" == "all" || $(( (run_id - 1) % GPU_COUNT )) -eq "${slot}" ]]; then
        dir="${BASE_OUTPUT_DIR}/partition=noniid-labeldir-fine_alpha=${alpha}_IF=${IMB_FACTOR}_localE=${LOCAL_EPOCHS}_seed=${seed}"
        "${callback}" "${gpu}" "${run_id}" "Dirichlet" "alpha" "${alpha}" "${seed}" "${dir}" "${schedule_file}"
      fi
    done

    for alpha in ${ALPHAS}; do
      run_id=$((run_id + 1))
      if [[ "${slot}" == "all" || $(( (run_id - 1) % GPU_COUNT )) -eq "${slot}" ]]; then
        dir="${BASE_OUTPUT_DIR}/partition=client-longtail_lambda=${SPECIALIZATION_LAMBDA}_alpha=${alpha}_rho=${HEAD_LEAKAGE_SCALE}_IF=${IMB_FACTOR}_localE=${LOCAL_EPOCHS}_seed=${seed}"
        "${callback}" "${gpu}" "${run_id}" "Client-LT" "alpha" "${alpha}" "${seed}" "${dir}" "${schedule_file}"
      fi
    done
  done

  if [[ "${run_id}" -ne "${TOTAL_RUNS}" ]]; then
    echo "PanelC matrix error: expected ${TOTAL_RUNS} runs, got ${run_id}"
    return 1
  fi
}

worker() {
  local gpu="$1"
  local slot="$2"
  echo "[GPU ${gpu}] worker started (slot ${slot}/${GPU_COUNT})"
  enumerate_tasks handle_task "${gpu}" "${slot}"
  echo "[GPU ${gpu}] worker finished"
}

prepare_shared_schedules() {
  local schedule_file
  for seed in ${SEEDS}; do
    schedule_file="output/panelC_shared_schedules/users${USERS}_frac${FRAC}_round${ROUND}_seed${seed}.json"
    echo "Preparing shared schedule for seed=${seed}: ${schedule_file}"
    "${PYTHON_BIN}" scripts/create_client_schedule.py \
      --path "${schedule_file}" \
      --num_rounds "${ROUND}" \
      --num_users "${USERS}" \
      --frac "${FRAC}" \
      --seed "${seed}"
  done
}

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

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "PanelC 3-GPU dry run; no files or directories will be created."
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
  echo "At least one PanelC worker failed."
  exit "${status}"
fi

echo "All PanelC workers finished."
