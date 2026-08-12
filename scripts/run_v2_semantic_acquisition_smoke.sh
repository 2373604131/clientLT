#!/usr/bin/env bash
set -euo pipefail
python -m tools.semantic_acquisition.manifests --output-dir output/v2_v3_semantic_acquisition/manifests
python -m tools.semantic_acquisition.runtime --stage v2 --mode smoke --manifest-dir output/v2_v3_semantic_acquisition/manifests --output-dir output/v2_v3_semantic_acquisition/v2_smoke
python -m tools.semantic_acquisition.summarize --stage v2 --mode smoke --input-dir output/v2_v3_semantic_acquisition/v2_smoke
