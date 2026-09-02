#!/usr/bin/env bash
set -euo pipefail

python -m tools.compatibility_retention.run --stage prepare "$@"
python -m tools.compatibility_retention.run --stage background "$@"
python -m tools.compatibility_retention.run --stage bridge "$@"
python -m tools.compatibility_retention.run --stage summarize "$@"

