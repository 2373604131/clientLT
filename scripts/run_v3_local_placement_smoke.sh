#!/usr/bin/env bash
set -euo pipefail
python -m tools.semantic_acquisition.runtime --stage v3 --mode smoke --manifest-dir output/v2_v3_semantic_acquisition/manifests --output-dir output/v2_v3_semantic_acquisition/v3_smoke_fp32 --v2-summary output/v2_v3_semantic_acquisition/v2_smoke_fp32/v2_summary.json
python -m tools.semantic_acquisition.summarize --stage v3 --mode smoke --input-dir output/v2_v3_semantic_acquisition/v3_smoke_fp32 --v2-summary output/v2_v3_semantic_acquisition/v2_smoke_fp32/v2_summary.json
