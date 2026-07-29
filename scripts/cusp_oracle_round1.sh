#!/usr/bin/env bash
set -euo pipefail

# Official entry for the minimal CUSP pilot.
# DRY_RUN=1 prints commands only. DRY_RUN=0 runs the requested stage.
REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "${REPO_ROOT}"

PYTHON_BIN="${PYTHON_BIN:-python}"
DATA="${DATA:-DATA/}"
OUTPUT_ROOT="${OUTPUT_ROOT:-output/cusp_minimal_seed42}"
DRY_RUN="${DRY_RUN:-1}"
STAGE="${STAGE:-all}"

TRAIN_DIR="${OUTPUT_ROOT}/client-longtail_seed42_round10"
ORACLE_DIR="${OUTPUT_ROOT}/oracle_client-longtail_seed42_round10"
SCHEDULE_FILE="${OUTPUT_ROOT}/shared_client_schedule_seed42_round10.json"
DUMP_DIR="${TRAIN_DIR}/experiment_d/oracle_cusp/round_010"

TRAIN_CMD=(
  "${PYTHON_BIN}" federated_main.py
  --root "${DATA}"
  --model fedavg
  --trainer PromptFL
  --dataset cifar100_LT
  --seed 42
  --split_seed 42
  --client_schedule_seed 42
  --client_schedule_file "${SCHEDULE_FILE}"
  --num_users 30
  --frac 1.0
  --round 10
  --local_epochs 3
  --lr 0.001
  --gamma 1
  --n_ctx 4
  --n_general 1
  --ctx_init False
  --csc True
  --imb_type exp
  --imb_factor 0.01
  --train_batch_size 32
  --test_batch_size 64
  --global_eval_interval 999999
  --num_classes 100
  --tail_class_ratio 0.2
  --head_class_ratio 0.8
  --head_client_ratio 0.9
  --tail_client_ratio 0.1
  --specialization_lambda 0.75
  --intra_group_alpha 0.5
  --head_leakage_scale 3.0
  --isolate_local_optimizer_state True
  --federated_single_scheduler_step True
  --dataset-config-file configs/datasets/cifar100_LT.yaml
  --config-file configs/trainers/PromptFL/vit_b16.yaml
  --partition client-longtail
  --experimentD_enable False
  --experimentD_rounds 10
  --experimentD_include_normalized True
  --experimentD_log_update_norm True
  --experimentD_require_full_participation True
  --experimentD_verify_fedavg True
  --experimentD_eval_mode class_filtered
  --oracle_cusp_enable True
  --oracle_cusp_round 10
  --oracle_cusp_cache_train_features True
  --oracle_cusp_max_train_samples_per_class 0
  --output-dir "${TRAIN_DIR}"
)

ORACLE_CMD=(
  "${PYTHON_BIN}" scripts/oracle_cusp_single_round.py
  --run-dir "${TRAIN_DIR}"
  --communication-round 10
  --output-dir "${ORACLE_DIR}"
)

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

check_stage() {
  case "${STAGE}" in
    all|train|oracle) ;;
    *) echo "STAGE must be one of: all, train, oracle" >&2; exit 2 ;;
  esac
}

preflight() {
  "${PYTHON_BIN}" - <<'PY'
import importlib.util
import sys
required = ["torch", "torchvision", "tensorboard", "yacs", "cvxpy"]
missing = [name for name in required if importlib.util.find_spec(name) is None]
if missing:
    raise SystemExit("Missing required Python dependencies before training: " + ", ".join(missing))
try:
    import cvxpy as cp
    solvers = cp.installed_solvers()
except Exception as exc:
    raise SystemExit(f"Unable to inspect CVXPY solvers: {exc}")
if not solvers:
    raise SystemExit("CVXPY is installed but no solver is available")
print("Preflight OK: dependencies and CVXPY solver are available")
PY
  test -f configs/datasets/cifar100_LT.yaml
  test -f configs/trainers/PromptFL/vit_b16.yaml
  test -d "${DATA}"
}

refuse_nonempty_dir() {
  local path="$1"
  if [[ -d "${path}" && -n "$(find "${path}" -mindepth 1 -maxdepth 1 -print -quit)" ]]; then
    echo "Refusing to overwrite non-empty directory: ${path}" >&2
    exit 2
  fi
}

verify_dump() {
  "${PYTHON_BIN}" - <<PY
from pathlib import Path
from utils.oracle_cusp import load_round_dump, sha256_file
dump = Path(r"${DUMP_DIR}")
payload, metadata = load_round_dump(dump)
if not (dump / "train_feature_cache.pt").exists():
    raise SystemExit("missing train_feature_cache.pt")
if metadata.get("round_state_sha256") != sha256_file(dump / "round_state.pt"):
    raise SystemExit("round_state.pt hash mismatch")
if metadata.get("train_feature_cache_sha256") != sha256_file(dump / "train_feature_cache.pt"):
    raise SystemExit("train_feature_cache.pt hash mismatch")
print("Oracle dump verified:", dump)
PY
}

verify_outputs() {
  "${PYTHON_BIN}" - <<PY
import csv
from pathlib import Path
root = Path(r"${ORACLE_DIR}")
required = [
    "candidate_states.pt", "candidate_manifest.json", "oracle_method_summary.csv",
    "oracle_per_class.csv", "random_reweight_distribution.csv", "oracle_solver.json",
    "oracle_metadata.json", "oracle_report.md",
]
missing = [name for name in required if not (root / name).exists()]
if missing:
    raise SystemExit("missing oracle outputs: " + ", ".join(missing))
with (root / "oracle_method_summary.csv").open(newline="", encoding="utf-8") as handle:
    methods = {row["method"] for row in csv.DictReader(handle)}
if methods != {"fedavg", "random_reweight", "classwise_weighting", "oracle_cusp"}:
    raise SystemExit(f"unexpected methods: {sorted(methods)}")
with (root / "random_reweight_distribution.csv").open(newline="", encoding="utf-8") as handle:
    count = sum(1 for _ in csv.DictReader(handle))
if count != 10:
    raise SystemExit(f"expected 10 random candidates, got {count}")
print("Oracle outputs verified:", root)
PY
}

check_stage

if [[ "${DRY_RUN}" == "1" ]]; then
  echo "CUSP minimal pilot dry run: no directories are created and no Python is started."
  echo "Stage: ${STAGE}"
  echo "Training command: client-longtail, seed 42, 30 clients, frac 1.0, local epochs 3, rounds 10, Oracle round 10."
  print_cmd "${TRAIN_CMD[@]}"
  echo "Oracle command:"
  print_cmd "${ORACLE_CMD[@]}"
  exit 0
fi

preflight

if [[ "${STAGE}" == "all" || "${STAGE}" == "train" ]]; then
  refuse_nonempty_dir "${TRAIN_DIR}"
  mkdir -p "${OUTPUT_ROOT}"
  "${TRAIN_CMD[@]}"
  verify_dump
fi

if [[ "${STAGE}" == "all" || "${STAGE}" == "oracle" ]]; then
  test -f "${DUMP_DIR}/round_state.pt"
  test -f "${DUMP_DIR}/train_feature_cache.pt"
  test -f "${DUMP_DIR}/metadata.json"
  refuse_nonempty_dir "${ORACLE_DIR}"
  mkdir -p "${ORACLE_DIR}"
  verify_dump
  "${ORACLE_CMD[@]}"
  verify_outputs
fi
