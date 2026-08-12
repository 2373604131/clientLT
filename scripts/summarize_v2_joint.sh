#!/usr/bin/env bash
set -euo pipefail
python -m tools.semantic_acquisition.summarize --stage v2_joint --mode full --input-dir output/v2_v3_semantic_acquisition/v2_topology_full --topology-dir output/v2_v3_semantic_acquisition/v2_topology_full --intervention-dir output/v2_v3_semantic_acquisition/v2_full --v1-dir output/p0_v1_context_colocation_v2
