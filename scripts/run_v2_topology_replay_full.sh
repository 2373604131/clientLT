#!/usr/bin/env bash
set -euo pipefail
python -m tools.semantic_acquisition.runtime --stage v2_topology --mode full --manifest-dir output/v2_v3_semantic_acquisition/manifests --output-dir output/v2_v3_semantic_acquisition/v2_topology_full --smoke-summary output/v2_v3_semantic_acquisition/v2_smoke_fp32/v2_summary.json
python -m tools.semantic_acquisition.summarize --stage v2_topology --mode full --input-dir output/v2_v3_semantic_acquisition/v2_topology_full
