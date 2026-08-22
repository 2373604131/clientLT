#!/usr/bin/env bash
set -euo pipefail

python scripts/run_post_write_rewrite_audit.py --stage all "$@"
