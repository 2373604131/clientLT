#!/usr/bin/env python
"""Phase-0 ERI closure on existing paired PromptFL/ClipLora round dumps.

This is strictly offline: it rebuilds a saved model state, evaluates the
frozen train-only probes, and replays aggregation coefficients.  It does not
start or resume federated training.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eri_closure.analysis import analyze_dump
from tools.eri_closure.protocol import build_protocol
from tools.eri_closure.replay import replay_dump


def main() -> None:
    args = parse_args()
    output = args.output_root.resolve()
    protocol = build_protocol(
        output / "protocol", data_root=args.data_root, samples_per_class=args.probes_per_class,
        audit_rounds=[int(args.round_id)], overwrite=args.overwrite_protocol,
    )
    runs = {"clientlt": args.clientlt_dump, "matched_dirichlet": args.dirichlet_dump}
    manifest = {"schema_version": "eri_phase0_v1", "protocol_dir": str(protocol), "runs": {}}
    for label, dump in runs.items():
        dump = dump.resolve()
        analysis_dir = output / label / "analysis"
        report = analyze_dump(
            dump, protocol_dir=protocol, output_dir=analysis_dir, data_root=args.data_root,
            quadrature_points=args.quadrature_points, device=args.device,
        )
        replay = replay_dump(
            dump, protocol_dir=protocol, output_dir=output / label / "replay", data_root=args.data_root,
            quadrature_points=args.quadrature_points, device=args.device, permutations=args.permutations,
        )
        manifest["runs"][label] = {"dump_dir": str(dump), "analysis": report, "replay": replay}
    (output / "phase0_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    print(f"Phase 0 ERI closure complete: {output}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clientlt-dump", required=True, type=Path)
    parser.add_argument("--dirichlet-dump", required=True, type=Path)
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--data-root", default=Path("DATA"), type=Path)
    parser.add_argument("--round-id", default=10, type=int)
    parser.add_argument("--probes-per-class", default=10, type=int)
    parser.add_argument("--quadrature-points", default=8, type=int)
    parser.add_argument("--permutations", default=100, type=int)
    parser.add_argument("--device", default=None)
    parser.add_argument("--overwrite-protocol", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
