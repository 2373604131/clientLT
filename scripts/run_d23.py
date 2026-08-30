#!/usr/bin/env python
"""Foreground D2/D3/D2b launcher using one shared seed-42 dump trajectory."""

from __future__ import annotations

import argparse
import json
import os
import shlex
import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from scripts.run_g0_d1 import CONFIGS, _append_lora, _common_command


def load_frozen(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(
            f"D2/D3 requires the G0 freeze artifact: {path}. Run G0 first."
        )
    frozen = json.loads(path.read_text(encoding="utf-8"))
    if frozen.get("verdict") != "PASS":
        raise RuntimeError("D2/D3 refuses to run because G0 did not pass")
    expected = CONFIGS["candidate_r4"]
    if frozen.get("selected_config_id") != "candidate_r4" or frozen.get("selected_config") != expected:
        raise RuntimeError(
            "This D2/D3 protocol is frozen to the G0-selected candidate_r4 "
            f"configuration {expected}; observed {frozen.get('selected_config')}"
        )
    return frozen


def build_dump_command(args, frozen: dict) -> tuple[Path, list[str]]:
    output_dir = args.output_root / "dump_seed42"
    command = _append_lora(_common_command(args, output_dir, rounds=80), frozen["selected_config"])
    command += [
        "--g0_probe_enable", "False",
        "--experimentD_enable", "False",
        "--v0_dump_enable", "True",
        "--v0_dump_rounds", "20,50,80",
        "DATALOADER.NUM_WORKERS", str(args.num_workers),
    ]
    return output_dir, command


def dump_complete(root: Path) -> bool:
    return all(
        (root / "v0_oracle" / f"round_{round_id:03d}" / name).is_file()
        for round_id in (20, 50, 80)
        for name in ("round_state.pt", "metadata.json")
    )


def _command_text(command: list[str]) -> str:
    return subprocess.list2cmdline(command) if os.name == "nt" else shlex.join(command)


def run(command: list[str], args) -> None:
    print("\n" + "=" * 78, flush=True)
    print(_command_text(command), flush=True)
    print("=" * 78, flush=True)
    if args.dry_run:
        return
    env = os.environ.copy()
    env["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)


def analyzer_command(args, diagnostic: str, dump_root: Path) -> list[str]:
    if diagnostic == "d2":
        script = "scripts/analyze_d2_conflict.py"
    elif diagnostic == "d3":
        script = "scripts/analyze_d3_boundary.py"
    elif diagnostic == "d2b":
        script = "scripts/analyze_d2b_scalar_ceiling.py"
    else:
        raise ValueError(diagnostic)
    command = [
        args.python_bin,
        "-u",
        script,
        "--dump-root", str(dump_root),
        "--output-dir", str(args.output_root / diagnostic),
        "--rounds", "20,50,80",
        "--eval-batch-size", str(args.eval_batch_size),
    ]
    if diagnostic == "d2b":
        command += [
            "--d2-utility",
            str(args.output_root / "d2" / "d2_client_class_utility.csv"),
        ]
    return command


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage", choices=["dump", "d2", "d3", "d2b", "all"], default="all"
    )
    parser.add_argument("--output-root", type=Path, default=Path("output/d23_seed42"))
    parser.add_argument("--freeze", type=Path, default=Path("output/g0_d1_seed42/lora_freeze.json"))
    parser.add_argument("--data-root", type=Path, default=Path("DATA"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--gpu", default="0")
    parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--test-batch-size", type=int, default=128)
    parser.add_argument("--eval-batch-size", type=int, default=256)
    parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--skip-completed", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    args.freeze = args.freeze.resolve()
    args.data_root = args.data_root.resolve()
    frozen = load_frozen(args.freeze)
    dump_root, dump_command = build_dump_command(args, frozen)

    if args.stage in {"dump", "all"}:
        if dump_complete(dump_root) and args.skip_completed:
            print(f"Skip completed shared D2/D3 dump: {dump_root}", flush=True)
        else:
            if dump_root.exists() and any(dump_root.iterdir()) and not args.dry_run:
                raise FileExistsError(
                    f"Refusing to overwrite non-empty dump directory: {dump_root}. "
                    "Use --skip-completed only if all three dumps are complete, or use a fresh output root."
                )
            dump_root.mkdir(parents=True, exist_ok=True)
            run(dump_command, args)
            if not args.dry_run and not dump_complete(dump_root):
                raise RuntimeError("Training exited without all round-20/50/80 dump artifacts")

    if args.stage in {"d2", "d3", "d2b"} and not dump_complete(dump_root) and not args.dry_run:
        raise FileNotFoundError(
            f"Offline stage {args.stage} requires complete shared dumps under {dump_root}"
        )

    diagnostics = (
        [args.stage]
        if args.stage in {"d2", "d3", "d2b"}
        else (["d2", "d3", "d2b"] if args.stage == "all" else [])
    )
    for diagnostic in diagnostics:
        verdict = args.output_root / diagnostic / f"{diagnostic}_verdict.json"
        if diagnostic == "d2b" and not args.dry_run:
            utility = args.output_root / "d2" / "d2_client_class_utility.csv"
            if not utility.is_file():
                raise FileNotFoundError(
                    f"D2b requires completed D2 utility output: {utility}"
                )
        if verdict.is_file() and args.skip_completed:
            print(f"Skip completed {diagnostic.upper()}: {verdict}", flush=True)
            continue
        if verdict.parent.exists() and any(verdict.parent.iterdir()) and not args.dry_run:
            print(
                f"Restart incomplete {diagnostic.upper()} offline analysis in place; "
                "deterministic artifacts will be overwritten.",
                flush=True,
            )
        run(analyzer_command(args, diagnostic, dump_root), args)


if __name__ == "__main__":
    main()
