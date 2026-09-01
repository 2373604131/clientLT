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

from tools.functional_breadth_feasibility.p0_audit import run as run_p0
from tools.semantic_acquisition.common import write_json


def _data_dir(path: Path) -> Path:
    path = Path(path)
    choices = [path, path / "cifar-100" / "cifar-100-python"]
    for choice in choices:
        if (choice / "train").is_file() and (choice / "meta").is_file():
            return choice
    raise FileNotFoundError(
        f"Cannot find CIFAR-100 train/meta under {path}; checked: {choices}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Run legacy frac=1 audit and no-training Functional Breadth feasibility V3"
    )
    parser.add_argument("--stage", choices=["all", "p0", "p1"], default="all")
    parser.add_argument("--legacy-output-root", type=Path, default=Path("output"))
    parser.add_argument("--output-root", type=Path, default=Path("output/functional_breadth_p0_p1_seed42_v3"))
    parser.add_argument("--data-root", type=Path, default=Path("DATA"))
    parser.add_argument("--manifest-dir", type=Path, default=Path("output/carrier_access_audit/manifests"))
    parser.add_argument("--b-dir", type=Path, default=Path("output/carrier_access_audit/experiment_b"))
    parser.add_argument("--d1-dir", type=Path, default=Path("output/post_write_rewrite_audit/d1"))
    parser.add_argument("--theta0-file", type=Path, default=Path("output/e1_strength_breadth/protocol_v2/theta0_seed42.pt"))
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--eval-batch-size", type=int, default=64)
    parser.add_argument("--model-init-seed", type=int, default=42)
    args = parser.parse_args()
    args.output_root.mkdir(parents=True, exist_ok=True)
    summaries = {}
    if args.stage in {"all", "p0"}:
        summaries["p0"] = run_p0(
            args.legacy_output_root, args.output_root / "p0_frac1_audit"
        )
        print(json.dumps({
            "stage": "P0", "status": "complete",
            "eligible_unique_runs": summaries["p0"]["eligible_unique_runs"],
        }))
    if args.stage in {"all", "p1"}:
        # Keep P0 runnable on a login node without importing CUDA/torchvision.
        os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
        from tools.functional_breadth_feasibility.runtime import guarded_run as run_p1

        p1_args = SimpleNamespace(
            output_dir=args.output_root / "p1_functional_breadth",
            data_dir=_data_dir(args.data_root), manifest_dir=args.manifest_dir,
            b_dir=args.b_dir, d1_dir=args.d1_dir, theta0_file=args.theta0_file,
            model_init_seed=args.model_init_seed, eval_batch_size=args.eval_batch_size,
            device="cuda",
        )
        summaries["p1"] = run_p1(p1_args)
        print(json.dumps({
            "stage": "P1-V3", "status": "complete", "verdict": summaries["p1"]["verdict"],
            "matched_tail_classes": summaries["p1"]["matched_tail_classes"],
        }))
    combined = {
        "schema_version": "functional_breadth_p0_p1_v3",
        "stages_requested": args.stage, "training_performed": False,
        "summaries": summaries,
    }
    write_json(args.output_root / "p0_p1_summary.json", combined)
    lines = ["# Phase 0 + Phase 1 summary", ""]
    if "p0" in summaries:
        lines += [
            f"- P0 eligible unique `frac=1.0` Client-LT runs: **{summaries['p0']['eligible_unique_runs']}**",
            "- P0 status: descriptive legacy audit only",
        ]
    if "p1" in summaries:
        lines += [
            f"- P1 V3 verdict: **{summaries['p1']['verdict']}**",
            f"- P1 matched tail classes: **{summaries['p1']['matched_tail_classes']}/20**",
            "- P1 used gradients / optimizers / test data: **no / no / no**",
        ]
    (args.output_root / "p0_p1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(json.dumps({"status": "complete", "summary": str((args.output_root / 'p0_p1_summary.json').resolve())}))


if __name__ == "__main__":
    main()
