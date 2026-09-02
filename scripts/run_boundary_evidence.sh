#!/usr/bin/env bash
set -euo pipefail

python -m tools.boundary_evidence.run --stage prepare "$@"
python -m tools.boundary_evidence.run --stage local "$@"
python -m tools.boundary_evidence.run --stage summarize "$@"
