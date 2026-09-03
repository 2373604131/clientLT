#!/usr/bin/env bash
set -euo pipefail

python -u scripts/run_full_participation_diagnosis.py --stage all "$@"
