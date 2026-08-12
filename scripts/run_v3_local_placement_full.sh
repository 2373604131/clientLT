#!/usr/bin/env bash
set -euo pipefail
python -m tools.semantic_acquisition.runtime --stage v3 --mode full --manifest-dir output/v2_v3_semantic_acquisition/manifests --output-dir output/v2_v3_semantic_acquisition/v3_full --smoke-summary output/v2_v3_semantic_acquisition/v3_smoke/v3_summary.json --require-v2-verdict POSITIVE_SEMANTIC_TRANSFER --v2-summary output/v2_v3_semantic_acquisition/v2_full/v2_summary.json
python -m tools.semantic_acquisition.summarize --stage v3 --mode full --input-dir output/v2_v3_semantic_acquisition/v3_full --v2-summary output/v2_v3_semantic_acquisition/v2_full/v2_summary.json
