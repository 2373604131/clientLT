#!/usr/bin/env bash
set -euo pipefail

bash scripts/run_v2_semantic_acquisition_full.sh
bash scripts/run_v2_topology_replay_full.sh
bash scripts/summarize_v2_joint.sh
