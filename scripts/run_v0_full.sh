#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Complete V0 headroom gate:
#   1. run no-test FedAvg/ClipLoRA jobs and dump rounds 20/50/80;
#   2. run one offline oracle unit for every seed/round;
#   3. aggregate all nine units.
#
# The current CIFAR100-LT dataset has no independent validation split.  The
# executable default is therefore the explicitly optimistic engineering gate.
# Set SELECTION_SOURCE=val only after the dataset exposes a held-out `val` set.

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA_ROOT="${DATA_ROOT:-DATA}"
OUT_ROOT="${OUT_ROOT:-output/v0_oracle_full}"
SEEDS_TEXT="${SEEDS:-1 42 2026}"
if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  # Slurm commonly exports a comma-separated list of physical GPU ids or UUIDs.
  # Preserve those allocation tokens when binding one seed worker per GPU.
  DEFAULT_GPU_IDS="${CUDA_VISIBLE_DEVICES//,/ }"
else
  DEFAULT_GPU_IDS="0"
fi
GPU_IDS_TEXT="${GPU_IDS:-${DEFAULT_GPU_IDS}}"
ROUNDS="${ROUNDS:-80}"
DUMP_ROUNDS_TEXT="${DUMP_ROUNDS:-20 50 80}"
NUM_USERS="${NUM_USERS:-30}"
LOCAL_EPOCHS="${LOCAL_EPOCHS:-3}"
NUM_WORKERS="${NUM_WORKERS:-8}"
SELECTION_SOURCE="${SELECTION_SOURCE:-train}"
ALLOW_OPTIMISTIC_SELECTION="${ALLOW_OPTIMISTIC_SELECTION:-1}"
GRID="${GRID:-pilot}"
PARTITION="${PARTITION:-client-longtail-controlled}"

read -r -a SEED_ARRAY <<< "${SEEDS_TEXT}"
read -r -a GPU_ARRAY <<< "${GPU_IDS_TEXT}"
read -r -a DUMP_ROUND_ARRAY <<< "${DUMP_ROUNDS_TEXT}"

if [[ "${#GPU_ARRAY[@]}" -lt 1 ]]; then
  echo "GPU_IDS must contain at least one GPU id" >&2
  exit 2
fi
if [[ "${#SEED_ARRAY[@]}" -lt 1 ]]; then
  echo "SEEDS must contain at least one seed" >&2
  exit 2
fi
if [[ "${SELECTION_SOURCE}" != "train" && "${SELECTION_SOURCE}" != "val" ]]; then
  echo "SELECTION_SOURCE must be train or val" >&2
  exit 2
fi
if [[ "${SELECTION_SOURCE}" == "train" && "${ALLOW_OPTIMISTIC_SELECTION}" != "1" ]]; then
  echo "train selection requires ALLOW_OPTIMISTIC_SELECTION=1" >&2
  exit 2
fi
if [[ "${GRID}" != "pilot" && "${GRID}" != "v0b" && "${GRID}" != "formal" ]]; then
  echo "GRID must be pilot, v0b, or formal" >&2
  exit 2
fi

comma_dump_rounds="$(IFS=,; echo "${DUMP_ROUND_ARRAY[*]}")"
last_dump_round="${DUMP_ROUND_ARRAY[${#DUMP_ROUND_ARRAY[@]}-1]}"
if [[ "${last_dump_round}" -ne "${ROUNDS}" ]]; then
  echo "The largest DUMP_ROUNDS value (${last_dump_round}) must equal ROUNDS (${ROUNDS})" >&2
  exit 2
fi

mkdir -p "${OUT_ROOT}/logs" "${OUT_ROOT}/schedules" "${OUT_ROOT}/dumps" "${OUT_ROOT}/oracle"
ORACLE_TAG="${SELECTION_SOURCE}_${GRID}"

make_schedule() {
  local seed="$1"
  local schedule="${OUT_ROOT}/schedules/full_users${NUM_USERS}_rounds${ROUNDS}_seed${seed}.json"
  "${PYTHON_BIN}" scripts/create_client_schedule.py \
    --path "${schedule}" \
    --num_rounds "${ROUNDS}" \
    --num_users "${NUM_USERS}" \
    --frac 1.0 \
    --seed "${seed}"
}

dump_complete() {
  local seed="$1"
  local round_id
  local round_tag
  for round_id in "${DUMP_ROUND_ARRAY[@]}"; do
    printf -v round_tag '%03d' "${round_id}"
    if [[ ! -s "${OUT_ROOT}/dumps/seed${seed}/v0_oracle/round_${round_tag}/round_state.pt" || \
          ! -s "${OUT_ROOT}/dumps/seed${seed}/v0_oracle/round_${round_tag}/metadata.json" ]]; then
      return 1
    fi
  done
  return 0
}

run_dump() {
  local seed="$1"
  local gpu="$2"
  local schedule="${OUT_ROOT}/schedules/full_users${NUM_USERS}_rounds${ROUNDS}_seed${seed}.json"
  local dump_root="${OUT_ROOT}/dumps/seed${seed}"
  local log_file="${OUT_ROOT}/logs/dump_seed${seed}.log"

  if dump_complete "${seed}"; then
    echo "V0 dumps already complete: seed=${seed}"
    return 0
  fi

  mkdir -p "${dump_root}"
  echo "Starting V0 dump: seed=${seed} gpu=${gpu} log=${log_file}"
  CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u federated_main.py \
    --root "${DATA_ROOT}" \
    --model fedavg \
    --trainer ClipLora \
    --dataset cifar100_LT \
    --seed "${seed}" \
    --split_seed "${seed}" \
    --num_users "${NUM_USERS}" \
    --frac 1.0 \
    --round "${ROUNDS}" \
    --local_epochs "${LOCAL_EPOCHS}" \
    --client_schedule_seed "${seed}" \
    --client_schedule_file "${schedule}" \
    --isolate_local_optimizer_state True \
    --federated_single_scheduler_step True \
    --lr 0.001 \
    --gamma 1 \
    --n_ctx 4 \
    --n_general 1 \
    --ctx_init False \
    --csc True \
    --dataset-config-file configs/datasets/cifar100_LT.yaml \
    --config-file configs/trainers/PromptFL/vit_b16.yaml \
    --output-dir "${dump_root}" \
    --imb_factor 0.01 \
    --imb_type exp \
    --train_batch_size 32 \
    --test_batch_size 64 \
    --global_eval_interval 1 \
    --num_classes 100 \
    --tail_class_ratio 0.2 \
    --head_client_ratio 0.9 \
    --tail_client_ratio 0.1 \
    --head_class_ratio 0.8 \
    --partition "${PARTITION}" \
    --beta 0.5 \
    --intra_group_alpha 0.5 \
    --controlled_tail_min_purity 0.8 \
    --specialization_lambda 1.0 \
    --head_leakage_scale 0.0 \
    --encoder vision \
    --cliplora_position top3 \
    --cliplora_rank 2 \
    --cliplora_alpha 1 \
    --cliplora_dropout_rate 0.0 \
    --cliplora_params q v \
    --cliplora_lr_policy constant \
    --cliplora_precision fp32 \
    --cliplora_aggregation fedavg \
    --experimentD_enable False \
    --v0_dump_enable True \
    --v0_dump_rounds "${comma_dump_rounds}" \
    DATALOADER.NUM_WORKERS "${NUM_WORKERS}" \
    >"${log_file}" 2>&1

  if ! dump_complete "${seed}"; then
    echo "Dump process exited but required artifacts are missing for seed=${seed}" >&2
    return 1
  fi
}

oracle_complete() {
  local output_dir="$1"
  [[ -s "${output_dir}/v0_manifest.json" && -s "${output_dir}/test_metrics.csv" ]]
}

run_oracles() {
  local seed="$1"
  local gpu="$2"
  local round_id
  local round_tag
  local dump_dir
  local output_dir
  local log_file
  local -a selection_args
  local -a grid_args

  selection_args=(--selection-source "${SELECTION_SOURCE}")
  if [[ "${SELECTION_SOURCE}" == "train" ]]; then
    selection_args+=(--allow-optimistic-selection)
  fi
  if [[ "${GRID}" == "pilot" ]]; then
    grid_args=(
      --gammas 0 0.2 0.4
      --lambda-head 0 1 4
      --lambda-mid 0 1
      --solver-iterations 2
      --random-count 5
      --convex-random-count 8
    )
  elif [[ "${GRID}" == "v0b" ]]; then
    grid_args=(
      --gammas 0.2 0.4 0.8 1.0
      --lambda-head 0 1 4
      --lambda-mid 0 1
      --solver-iterations 4
      --random-count 20
      --convex-random-count 32
      --oracle-starts fedavg support equal best_random
    )
  else
    grid_args=()
  fi

  for round_id in "${DUMP_ROUND_ARRAY[@]}"; do
    printf -v round_tag '%03d' "${round_id}"
    dump_dir="${OUT_ROOT}/dumps/seed${seed}/v0_oracle/round_${round_tag}"
    output_dir="${OUT_ROOT}/oracle/${ORACLE_TAG}/seed${seed}/round_${round_tag}"
    log_file="${OUT_ROOT}/logs/oracle_${ORACLE_TAG}_seed${seed}_round${round_tag}.log"
    if oracle_complete "${output_dir}"; then
      echo "V0 oracle already complete: seed=${seed} round=${round_id}"
      continue
    fi
    mkdir -p "${output_dir}"
    echo "Starting V0 oracle: seed=${seed} round=${round_id} gpu=${gpu} log=${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u scripts/run_v0_oracle.py \
      --dump-dir "${dump_dir}" \
      --output-dir "${output_dir}" \
      "${selection_args[@]}" \
      "${grid_args[@]}" \
      >"${log_file}" 2>&1
  done
}

run_seed_worker() {
  local seed="$1"
  local gpu="$2"
  make_schedule "${seed}"
  run_dump "${seed}" "${gpu}"
  run_oracles "${seed}" "${gpu}"
}

echo "V0 full runner"
echo "  output: ${OUT_ROOT}"
echo "  seeds: ${SEED_ARRAY[*]}"
echo "  dump rounds: ${DUMP_ROUND_ARRAY[*]}"
echo "  GPUs: ${GPU_ARRAY[*]}"
echo "  selection: ${SELECTION_SOURCE}"
echo "  grid: ${GRID}"

run_gpu_worker() {
  local gpu="$1"
  local slot="$2"
  local stride="$3"
  local index
  local seed
  for ((index=slot; index<${#SEED_ARRAY[@]}; index+=stride)); do
    seed="${SEED_ARRAY[$index]}"
    run_seed_worker "${seed}" "${gpu}"
  done
}

worker_count="${#GPU_ARRAY[@]}"
if [[ "${worker_count}" -gt "${#SEED_ARRAY[@]}" ]]; then
  worker_count="${#SEED_ARRAY[@]}"
fi
pids=()
for ((slot=0; slot<worker_count; slot+=1)); do
  run_gpu_worker "${GPU_ARRAY[$slot]}" "${slot}" "${worker_count}" &
  pids+=("$!")
done

status=0
for pid in "${pids[@]}"; do
  if ! wait "${pid}"; then
    status=1
  fi
done
if [[ "${status}" -ne 0 ]]; then
  echo "At least one V0 seed worker failed. Inspect ${OUT_ROOT}/logs/." >&2
  exit "${status}"
fi

input_dirs=()
for seed in "${SEED_ARRAY[@]}"; do
  for round_id in "${DUMP_ROUND_ARRAY[@]}"; do
    printf -v round_tag '%03d' "${round_id}"
    input_dirs+=("${OUT_ROOT}/oracle/${ORACLE_TAG}/seed${seed}/round_${round_tag}")
  done
done

"${PYTHON_BIN}" scripts/summarize_v0_oracle.py \
  --input-dirs "${input_dirs[@]}" \
  --output-dir "${OUT_ROOT}/summary/${ORACLE_TAG}"

echo "V0 complete. Verdict: ${OUT_ROOT}/summary/${ORACLE_TAG}/v0_verdict.json"
