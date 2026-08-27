#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
V0_ROOT="${V0_ROOT:-output/v0_oracle_full/dumps}"
OUT_DIR="${OUT_DIR:-output/v1_mode_stability_full}"
SEEDS_TEXT="${SEEDS:-1 42 2026}"
ROUNDS_TEXT="${ROUNDS:-20 50 80}"
POLL_SECONDS="${POLL_SECONDS:-30}"
WAIT_TIMEOUT_SECONDS="${WAIT_TIMEOUT_SECONDS:-172800}"
RANKS_TEXT="${RANKS:-4 8 16}"
PERTURB_REPEATS="${PERTURB_REPEATS:-5}"
CLIENT_DROPOUT="${CLIENT_DROPOUT:-0.1}"
WEIGHT_JITTER="${WEIGHT_JITTER:-0.05}"
SKETCH_DIM="${SKETCH_DIM:-128}"
FORCE="${FORCE:-0}"

read -r -a SEED_ARRAY <<< "${SEEDS_TEXT}"
read -r -a ROUND_ARRAY <<< "${ROUNDS_TEXT}"
read -r -a RANK_ARRAY <<< "${RANKS_TEXT}"

export OMP_NUM_THREADS="${OMP_NUM_THREADS:-${SLURM_CPUS_PER_TASK:-8}}"
export MKL_NUM_THREADS="${MKL_NUM_THREADS:-${OMP_NUM_THREADS}}"

if [[ "${FORCE}" != "1" && -s "${OUT_DIR}/v1_verdict.json" ]]; then
  echo "V1 already complete: ${OUT_DIR}/v1_verdict.json"
  exit 0
fi

dump_dirs=()
for seed in "${SEED_ARRAY[@]}"; do
  for round_id in "${ROUND_ARRAY[@]}"; do
    printf -v round_tag '%03d' "${round_id}"
    dump_dirs+=("${V0_ROOT}/seed${seed}/v0_oracle/round_${round_tag}")
  done
done

start_time="$(date +%s)"
while true; do
  missing=()
  for directory in "${dump_dirs[@]}"; do
    if [[ ! -s "${directory}/round_state.pt" || ! -s "${directory}/metadata.json" ]]; then
      missing+=("${directory}")
    fi
  done
  if [[ "${#missing[@]}" -eq 0 ]]; then
    break
  fi
  now="$(date +%s)"
  elapsed="$((now - start_time))"
  if [[ "${elapsed}" -ge "${WAIT_TIMEOUT_SECONDS}" ]]; then
    echo "Timed out waiting for V0 dumps. Still missing:" >&2
    printf '  %s\n' "${missing[@]}" >&2
    exit 1
  fi
  echo "Waiting for ${#missing[@]} V0 dumps (${elapsed}s elapsed); next check in ${POLL_SECONDS}s"
  sleep "${POLL_SECONDS}"
done

mkdir -p "${OUT_DIR}"
echo "All ${#dump_dirs[@]} V0 dumps are ready; starting V1 offline analysis"
echo "CPU threads: ${OMP_NUM_THREADS}"

"${PYTHON_BIN}" -u scripts/run_v1_mode_stability.py \
  --dump-dirs "${dump_dirs[@]}" \
  --output-dir "${OUT_DIR}" \
  --ranks "${RANK_ARRAY[@]}" \
  --perturb-repeats "${PERTURB_REPEATS}" \
  --client-dropout "${CLIENT_DROPOUT}" \
  --weight-jitter "${WEIGHT_JITTER}" \
  --sketch-dim "${SKETCH_DIM}"

echo "V1 complete: ${OUT_DIR}/v1_verdict.json"
