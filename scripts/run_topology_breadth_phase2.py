#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace


ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from tools.topology_breadth_audit.manifests import build as build_manifests


def _data_dir(path: Path) -> Path:
    choices = [Path(path), Path(path) / "cifar-100" / "cifar-100-python"]
    for candidate in choices:
        if (candidate / "train").is_file() and (candidate / "meta").is_file():
            return candidate
    raise FileNotFoundError(f"Cannot locate CIFAR-100 train/meta; checked {choices}")


def _runtime_args(args, topology: str, data_dir: Path):
    return SimpleNamespace(
        topology=topology, manifest_dir=args.output_root / "manifests",
        output_dir=args.output_root / "client_updates" / topology,
        data_dir=data_dir, theta0_file=args.theta0_file,
        model_init_seed=42, eval_batch_size=args.eval_batch_size, device="cuda",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Run client-level topology→breadth Phase 2")
    parser.add_argument(
        "--stage", choices=["all", "manifests", "clientlt", "matched", "analyze"],
        default="all",
    )
    parser.add_argument("--output-root", type=Path, default=Path("output/topology_breadth_phase2_seed42"))
    parser.add_argument("--sca-output-root", type=Path, default=Path("output/online_sca_seed42_v2"))
    parser.add_argument("--data-root", type=Path, default=Path("DATA"))
    parser.add_argument("--theta0-file", type=Path, default=Path("output/e1_strength_breadth/protocol_v2/theta0_seed42.pt"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    data_dir = _data_dir(args.data_root)
    if args.stage in {"all", "manifests"}:
        result = build_manifests(data_dir, args.sca_output_root, args.output_root / "manifests")
        print(json.dumps({"stage": "manifests", **result}))
    if args.stage in {"clientlt", "matched", "analyze"} and not (
        args.output_root / "manifests" / "manifest_contract.json"
    ).is_file():
        raise FileNotFoundError("Phase-2 manifests are missing; run --stage manifests first")
    if args.stage in {"all", "clientlt", "matched", "analyze"}:
        visible = [value.strip() for value in os.environ.get("CUDA_VISIBLE_DEVICES", "").split(",") if value.strip()]
        requested = str(args.gpu)
        os.environ["CUDA_VISIBLE_DEVICES"] = (
            visible[int(requested)] if visible and requested.isdigit() and int(requested) < len(visible)
            else requested
        )
    if args.stage in {"all", "clientlt"}:
        from tools.topology_breadth_audit.runtime import guarded_run
        print(json.dumps(guarded_run(_runtime_args(args, "clientlt", data_dir))))
    if args.stage in {"all", "matched"}:
        from tools.topology_breadth_audit.runtime import guarded_run
        print(json.dumps(guarded_run(_runtime_args(args, "matched_dirichlet", data_dir))))
    if args.stage in {"all", "analyze"}:
        from tools.topology_breadth_audit.analyze import guarded_run
        analysis_args = SimpleNamespace(
            manifest_dir=args.output_root / "manifests",
            clientlt_dir=args.output_root / "client_updates" / "clientlt",
            matched_dir=args.output_root / "client_updates" / "matched_dirichlet",
            output_dir=args.output_root / "analysis", data_dir=data_dir,
            theta0_file=args.theta0_file, model_init_seed=42,
            eval_batch_size=args.eval_batch_size, device="cuda",
        )
        result = guarded_run(analysis_args)
        print(json.dumps({
            "stage": "analysis", "verdict": result["verdict"],
            "spatial": result["spatial_breadth_supported"],
            "temporal": result["temporal_breadth_supported"],
        }))


if __name__ == "__main__":
    main()

