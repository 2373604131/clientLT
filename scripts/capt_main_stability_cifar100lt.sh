#!/bin/bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${PROJECT_ROOT}"

# Matched CAPT baseline for the current FedTEF-v2 main setting.
#
# Default plan:
#   - CIFAR100-LT
#   - CAPT with cluster aggregation
#   - partitions: client-longtail + noniid-labeldir
#   - seeds: 1, 42, 3407
#   - frac: 0.4
#   - rounds: 100
#
# Example:
#   GPU_IDS="0 1 2" bash scripts/capt_main_stability_cifar100lt.sh

DATA="${DATA:-DATA/}"
DATASET="${DATASET:-cifar100_LT}"
NUM_CLASSES="${NUM_CLASSES:-100}"
MODEL="${MODEL:-cluster}"
TRAINER="${TRAINER:-CAPT}"
TRAINER_CONFIG="${TRAINER_CONFIG:-CAPT}"
CFG="${CFG:-vit_b16}"
OUT_ROOT="${OUT_ROOT:-output/${DATASET}/capt_main_matched}"

SEED_LIST="${SEEDS:-1 42 3407}"
PARTITION_LIST="${PARTITIONS:-client-longtail noniid-labeldir}"
FRAC_LIST="${FRACS:-0.4}"
GPU_IDS="${GPU_IDS:-0 1}"

ROUNDS="${ROUNDS:-100}"
USERS="${USERS:-20}"
SCHEDULE_SEED_MODE="${SCHEDULE_SEED_MODE:-seed}"
SCHEDULE_SEED_FIXED="${SCHEDULE_SEED_FIXED:-2026}"
SKIP_FINISHED="${SKIP_FINISHED:-true}"
DRY_RUN="${DRY_RUN:-false}"

LR="${LR:-0.001}"
GAMMA="${GAMMA:-1}"
BETA="${BETA:-0.5}"
BATCH_SIZE="${BATCH_SIZE:-32}"
TEST_BATCH_SIZE="${TEST_BATCH_SIZE:-64}"
NUM_WORKERS="${NUM_WORKERS:-4}"
GLOBAL_EVAL_INTERVAL="${GLOBAL_EVAL_INTERVAL:-5}"
IMB_FACTOR="${IMB_FACTOR:-0.01}"
IMB_TYPE="${IMB_TYPE:-exp}"
NCTX="${NCTX:-4}"
N_GENERAL="${N_GENERAL:-1}"
CTXINIT="${CTXINIT:-False}"
CSC="${CSC:-True}"
SIMCLUST="${SIMCLUST:-4}"
DISCLUSTERS="${DISCLUSTERS:-4}"

# Keep the same main client-longtail topology knobs as the FedTEF-v2 main runs.
SPECIALIZATION_LAMBDA="${SPECIALIZATION_LAMBDA:-1.0}"
INTRA_GROUP_ALPHA="${INTRA_GROUP_ALPHA:-0.5}"
HEAD_LEAKAGE_SCALE="${HEAD_LEAKAGE_SCALE:-3.0}"
HEAD_CLIENT_RATIO="${HEAD_CLIENT_RATIO:-0.9}"
TAIL_CLIENT_RATIO="${TAIL_CLIENT_RATIO:-0.1}"
HEAD_CLASS_RATIO="${HEAD_CLASS_RATIO:-0.8}"
TAIL_CLASS_RATIO="${TAIL_CLASS_RATIO:-0.2}"

read -r -a SEED_ARRAY <<< "${SEED_LIST}"
read -r -a PARTITION_ARRAY <<< "${PARTITION_LIST}"
read -r -a FRAC_ARRAY <<< "${FRAC_LIST}"
read -r -a GPU_ID_ARRAY <<< "${GPU_IDS}"

schedule_seed_for() {
  local seed="$1"
  if [[ "${SCHEDULE_SEED_MODE}" == "seed" ]]; then
    echo "${seed}"
  else
    echo "${SCHEDULE_SEED_FIXED}"
  fi
}

schedule_path_for() {
  local seed="$1"
  local frac="$2"
  echo "${OUT_ROOT}/schedules/seed${seed}_frac${frac}_rounds${ROUNDS}_users${USERS}.json"
}

run_dir_for() {
  local seed="$1"
  local frac="$2"
  local partition="$3"
  echo "${OUT_ROOT}/seed${seed}/frac${frac}/CAPT_${partition}"
}

ensure_schedule() {
  local schedule="$1"
  local schedule_seed="$2"
  local frac="$3"
  if [[ -f "${schedule}" ]]; then
    return 0
  fi
  mkdir -p "$(dirname "${schedule}")"
  SCHEDULE_PATH="${schedule}" \
  SCHEDULE_ROUNDS="${ROUNDS}" \
  SCHEDULE_USERS="${USERS}" \
  SCHEDULE_FRAC="${frac}" \
  SCHEDULE_SEED_VALUE="${schedule_seed}" \
  python -c '
import json
import os
import numpy as np

path = os.environ["SCHEDULE_PATH"]
num_rounds = int(os.environ["SCHEDULE_ROUNDS"])
num_users = int(os.environ["SCHEDULE_USERS"])
frac = float(os.environ["SCHEDULE_FRAC"])
seed = int(os.environ["SCHEDULE_SEED_VALUE"])
m = max(int(frac * num_users), 1)
rng = np.random.default_rng(seed)
schedule = [
    [int(x) for x in rng.choice(num_users, m, replace=False).tolist()]
    for _ in range(num_rounds)
]
payload = {
    "num_rounds": num_rounds,
    "num_users": num_users,
    "frac": frac,
    "clients_per_round": m,
    "seed": seed,
    "schedule": schedule,
}
with open(path, "w", encoding="utf-8") as f:
    json.dump(payload, f, indent=2)
print(f"Created fixed client schedule at {path}")
'
}

is_finished() {
  local dir="$1"
  python - "$dir" "$ROUNDS" <<'PY'
import csv
import os
import sys

run_dir = sys.argv[1]
rounds = int(sys.argv[2])
path = os.path.join(run_dir, "round_metrics.csv")
if not os.path.exists(path):
    sys.exit(1)
with open(path, newline="", encoding="utf-8") as f:
    rows = list(csv.DictReader(f))
if not rows:
    sys.exit(1)
try:
    last_epoch = max(int(float(row.get("epoch", -1))) for row in rows)
except ValueError:
    sys.exit(1)
sys.exit(0 if last_epoch >= rounds - 1 else 1)
PY
}

run_one() {
  local seed="$1"
  local frac="$2"
  local partition="$3"
  local gpu="$4"
  local schedule_seed
  local schedule
  local dir
  local log_file

  schedule_seed="$(schedule_seed_for "${seed}")"
  schedule="$(schedule_path_for "${seed}" "${frac}")"
  dir="$(run_dir_for "${seed}" "${frac}" "${partition}")"
  log_file="${dir}/console_gpu${gpu}.log"

  ensure_schedule "${schedule}" "${schedule_seed}" "${frac}"
  mkdir -p "${dir}"

  if [[ "${SKIP_FINISHED}" == "true" ]] && is_finished "${dir}"; then
    echo "Skip finished: seed=${seed} frac=${frac} partition=${partition}"
    return 0
  fi

  echo "========================================================="
  echo "CAPT matched baseline job"
  echo "Seed/frac/partition: ${seed}/${frac}/${partition}"
  echo "GPU: ${gpu}"
  echo "Output: ${dir}"
  echo "Schedule: ${schedule}"
  echo "Config: model=${MODEL}, trainer=${TRAINER}, cfg=${CFG}"
  echo "========================================================="

  if [[ "${DRY_RUN}" == "true" ]]; then
    return 0
  fi

  RUN_CMD=(
    python federated_main.py
    --root "${DATA}"
    --model "${MODEL}"
    --dataset "${DATASET}"
    --seed "${seed}"
    --client_schedule_file "${schedule}"
    --client_schedule_seed "${schedule_seed}"
    --num_users "${USERS}"
    --frac "${frac}"
    --lr "${LR}"
    --csc "${CSC}"
    --gamma "${GAMMA}"
    --trainer "${TRAINER}"
    --round "${ROUNDS}"
    --partition "${partition}"
    --beta "${BETA}"
    --n_ctx "${NCTX}"
    --dataset-config-file "configs/datasets/${DATASET}.yaml"
    --config-file "configs/trainers/${TRAINER_CONFIG}/${CFG}.yaml"
    --output-dir "${dir}"
    --imb_factor "${IMB_FACTOR}"
    --imb_type "${IMB_TYPE}"
    --ctx_init "${CTXINIT}"
    --train_batch_size "${BATCH_SIZE}"
    --test_batch_size "${TEST_BATCH_SIZE}"
    --global_eval_interval "${GLOBAL_EVAL_INTERVAL}"
    --num_classes "${NUM_CLASSES}"
    --n_general "${N_GENERAL}"
    --n_simclusters "${SIMCLUST}"
    --n_disclusters "${DISCLUSTERS}"
    --head_client_ratio "${HEAD_CLIENT_RATIO}"
    --tail_client_ratio "${TAIL_CLIENT_RATIO}"
    --head_class_ratio "${HEAD_CLASS_RATIO}"
    --tail_class_ratio "${TAIL_CLASS_RATIO}"
    --specialization_lambda "${SPECIALIZATION_LAMBDA}"
    --intra_group_alpha "${INTRA_GROUP_ALPHA}"
    --head_leakage_scale "${HEAD_LEAKAGE_SCALE}"
    DATALOADER.NUM_WORKERS "${NUM_WORKERS}"
  )

  if ! CUDA_VISIBLE_DEVICES="${gpu}" "${RUN_CMD[@]}" 2>&1 | tee "${log_file}"; then
    echo "CAPT matched run failed. Showing the last 120 log lines:"
    echo "Log file: ${log_file}"
    echo "---------------------------------------------------------"
    tail -n 120 "${log_file}" || true
    echo "---------------------------------------------------------"
    exit 1
  fi
}

declare -a JOBS=()
for seed in "${SEED_ARRAY[@]}"; do
  for frac in "${FRAC_ARRAY[@]}"; do
    for partition in "${PARTITION_ARRAY[@]}"; do
      JOBS+=("${seed}|${frac}|${partition}")
    done
  done
done

echo "========================================================="
echo "CAPT matched baseline runner"
echo "Output root: ${OUT_ROOT}"
echo "Seeds: ${SEED_ARRAY[*]}"
echo "Fracs: ${FRAC_ARRAY[*]}"
echo "Partitions: ${PARTITION_ARRAY[*]}"
echo "GPUs: ${GPU_ID_ARRAY[*]}"
echo "Jobs: ${#JOBS[@]}"
echo "Skip finished: ${SKIP_FINISHED}"
echo "Dry run: ${DRY_RUN}"
echo "========================================================="

run_worker() {
  local gpu="$1"
  local slot="$2"
  local stride="$3"
  local job
  local seed
  local frac
  local partition
  for ((i = slot; i < ${#JOBS[@]}; i += stride)); do
    job="${JOBS[$i]}"
    IFS='|' read -r seed frac partition <<< "${job}"
    run_one "${seed}" "${frac}" "${partition}" "${gpu}"
  done
}

if [[ "${#GPU_ID_ARRAY[@]}" -le 1 ]]; then
  run_worker "${GPU_ID_ARRAY[0]}" 0 1
else
  pids=()
  for ((slot = 0; slot < ${#GPU_ID_ARRAY[@]}; slot += 1)); do
    run_worker "${GPU_ID_ARRAY[$slot]}" "${slot}" "${#GPU_ID_ARRAY[@]}" &
    pids+=("$!")
  done
  status=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      status=1
    fi
  done
  if [[ "${status}" -ne 0 ]]; then
    echo "At least one CAPT matched baseline worker failed."
    exit "${status}"
  fi
fi

echo "CAPT matched baseline runner finished."
echo "Output root: ${OUT_ROOT}"
