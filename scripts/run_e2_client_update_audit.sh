#!/usr/bin/env bash
set -euo pipefail

python scripts/run_e2_client_update_audit.py --stage all "$@"
