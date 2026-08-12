#!/usr/bin/env bash
set -euo pipefail
python -m tools.semantic_acquisition.runtime --stage v2 --mode full --manifest-dir output/v2_v3_semantic_acquisition/manifests --output-dir output/v2_v3_semantic_acquisition/v2_full --smoke-summary output/v2_v3_semantic_acquisition/v2_smoke/v2_summary.json
python -m tools.semantic_acquisition.summarize --stage v2 --mode full --input-dir output/v2_v3_semantic_acquisition/v2_full
