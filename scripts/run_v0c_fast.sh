#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

# Fast, offline-only V0 oracle screening. It reuses existing V0 dumps, starts
# with round 80, shares one candidate bank across all lambda scalarizations,
# and promotes only a shortlist to the full safe split.

PYTHON_BIN="${PYTHON_BIN:-python}"
V0_ROOT="${V0_ROOT:-output/v0_oracle_full}"
DUMP_ROOT="${DUMP_ROOT:-${V0_ROOT}/dumps}"
SEEDS_TEXT="${SEEDS:-1 42 2026}"
ROUNDS_TEXT="${ROUNDS:-80}"
SELECTION_SOURCE="${SELECTION_SOURCE:-train}"
ALLOW_OPTIMISTIC_SELECTION="${ALLOW_OPTIMISTIC_SELECTION:-1}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-256}"
OPT_PER_CLASS="${OPT_PER_CLASS:-20}"
SAFE_TOP_K="${SAFE_TOP_K:-8}"
RANDOM_COUNT="${RANDOM_COUNT:-20}"
CONVEX_RANDOM_COUNT="${CONVEX_RANDOM_COUNT:-16}"
RANK_MAX="${RANK_MAX:-8}"
ORACLE_TAG="${SELECTION_SOURCE}_v0c_fast"

if [[ -n "${CUDA_VISIBLE_DEVICES:-}" ]]; then
  DEFAULT_GPU_IDS="${CUDA_VISIBLE_DEVICES//,/ }"
else
  DEFAULT_GPU_IDS="0"
fi
GPU_IDS_TEXT="${GPU_IDS:-${DEFAULT_GPU_IDS}}"

read -r -a SEED_ARRAY <<< "${SEEDS_TEXT}"
read -r -a ROUND_ARRAY <<< "${ROUNDS_TEXT}"
read -r -a GPU_ARRAY <<< "${GPU_IDS_TEXT}"

if [[ "${#GPU_ARRAY[@]}" -lt 1 ]]; then
  echo "No GPU ids are available" >&2
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

for seed in "${SEED_ARRAY[@]}"; do
  for round_id in "${ROUND_ARRAY[@]}"; do
    printf -v round_tag '%03d' "${round_id}"
    dump_dir="${DUMP_ROOT}/seed${seed}/v0_oracle/round_${round_tag}"
    if [[ ! -s "${dump_dir}/round_state.pt" || ! -s "${dump_dir}/metadata.json" ]]; then
      echo "Missing V0 dump; V0c is offline-only and will not retrain: ${dump_dir}" >&2
      exit 1
    fi
  done
done

mkdir -p "${V0_ROOT}/logs" "${V0_ROOT}/oracle/${ORACLE_TAG}" "${V0_ROOT}/summary/${ORACLE_TAG}"

unit_complete() {
  local directory="$1"
  [[ -s "${directory}/v0_manifest.json" && -s "${directory}/test_metrics.csv" && -s "${directory}/v0_verdict.json" ]]
}

run_seed() {
  local seed="$1"
  local gpu="$2"
  local round_id
  local round_tag
  local dump_dir
  local output_dir
  local log_file
  local unit_start
  local unit_elapsed
  local -a selection_args=(--selection-source "${SELECTION_SOURCE}")
  if [[ "${SELECTION_SOURCE}" == "train" ]]; then
    selection_args+=(--allow-optimistic-selection)
  fi

  for round_id in "${ROUND_ARRAY[@]}"; do
    printf -v round_tag '%03d' "${round_id}"
    dump_dir="${DUMP_ROOT}/seed${seed}/v0_oracle/round_${round_tag}"
    output_dir="${V0_ROOT}/oracle/${ORACLE_TAG}/seed${seed}/round_${round_tag}"
    log_file="${V0_ROOT}/logs/oracle_${ORACLE_TAG}_seed${seed}_round${round_tag}.log"
    if unit_complete "${output_dir}"; then
      echo "Skip completed V0c unit: seed=${seed} round=${round_id}"
      continue
    fi
    mkdir -p "${output_dir}"
    unit_start="$(date +%s)"
    echo "Start V0c: seed=${seed} round=${round_id} gpu=${gpu} log=${log_file}"
    CUDA_VISIBLE_DEVICES="${gpu}" "${PYTHON_BIN}" -u scripts/run_v0_oracle.py \
      --dump-dir "${dump_dir}" \
      --output-dir "${output_dir}" \
      "${selection_args[@]}" \
      --search-mode pooled \
      --opt-per-class "${OPT_PER_CLASS}" \
      --gammas 0.2 0.4 0.8 1.0 \
      --lambda-head 0 1 4 \
      --lambda-mid 0 1 \
      --rank-max "${RANK_MAX}" \
      --random-count "${RANDOM_COUNT}" \
      --convex-random-count "${CONVEX_RANDOM_COUNT}" \
      --axis-scales 0.5 1.0 \
      --safe-top-k "${SAFE_TOP_K}" \
      --progress-every 5 \
      --eval-batch-size "${EVAL_BATCH_SIZE}" \
      >"${log_file}" 2>&1
    unit_elapsed="$(( $(date +%s) - unit_start ))"
    echo "Complete V0c unit: seed=${seed} round=${round_id} elapsed=${unit_elapsed}s"
  done
}

run_gpu_worker() {
  local gpu="$1"
  local slot="$2"
  local stride="$3"
  local index
  for ((index=slot; index<${#SEED_ARRAY[@]}; index+=stride)); do
    run_seed "${SEED_ARRAY[$index]}" "${gpu}"
  done
}

worker_count="${#GPU_ARRAY[@]}"
if [[ "${worker_count}" -gt "${#SEED_ARRAY[@]}" ]]; then
  worker_count="${#SEED_ARRAY[@]}"
fi

echo "V0c fast offline oracle screening"
echo "  dumps: ${DUMP_ROOT}"
echo "  seeds: ${SEED_ARRAY[*]}"
echo "  rounds: ${ROUND_ARRAY[*]}"
echo "  GPUs: ${GPU_ARRAY[*]}"
echo "  balanced opt cap: ${OPT_PER_CLASS} per class"
echo "  full-safe top-k: ${SAFE_TOP_K}"
echo "  random/convex: ${RANDOM_COUNT}/${CONVEX_RANDOM_COUNT}"
echo "  eval batch size: ${EVAL_BATCH_SIZE}"

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
  echo "At least one V0c worker failed; inspect ${V0_ROOT}/logs/oracle_${ORACLE_TAG}_*.log" >&2
  exit "${status}"
fi

input_dirs=()
for seed in "${SEED_ARRAY[@]}"; do
  for round_id in "${ROUND_ARRAY[@]}"; do
    printf -v round_tag '%03d' "${round_id}"
    input_dirs+=("${V0_ROOT}/oracle/${ORACLE_TAG}/seed${seed}/round_${round_tag}")
  done
done

"${PYTHON_BIN}" scripts/summarize_v0_oracle.py \
  --input-dirs "${input_dirs[@]}" \
  --output-dir "${V0_ROOT}/summary/${ORACLE_TAG}"

echo "V0c complete: ${V0_ROOT}/summary/${ORACLE_TAG}/v0_verdict.json"
