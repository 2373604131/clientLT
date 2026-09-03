#!/usr/bin/env bash
set -euo pipefail

python -m tools.compatibility_retention.corrected --stage prepare "$@"
python -m tools.compatibility_retention.corrected --stage background-only "$@"
python -m tools.compatibility_retention.corrected --stage summarize "$@"

