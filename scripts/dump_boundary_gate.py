#!/usr/bin/env python
"""Create a self-contained Boundary Gate dump from a saved CUSP round dump.

The command only rebuilds deterministic audit views from the federated train
partition.  It does not iterate over the official test loader or train a
model.  It is therefore safe to run before candidate construction.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.boundary_audit import (
    BOUNDARY_SCHEMA_VERSION,
    attach_audit_cache_to_round_dump,
    load_boundary_round_dump,
    sha256_file,
)
from utils.cusp_minimal import load_cusp_minimal_dump, write_json


def build_promptfl_trainer(metadata: dict, output_dir: Path):
    from Dassl.dassl.engine import build_trainer
    from federated_main import setup_cfg

    args = SimpleNamespace(**metadata["resolved_args"])
    args.output_dir = str(output_dir)
    cfg = setup_cfg(args)
    trainer = build_trainer(cfg)
    trainer.fed_before_train(is_global=True)
    return cfg, trainer


def load_source_dump(path: Path) -> tuple[dict, dict]:
    metadata_path = path / "metadata.json"
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if metadata.get("schema_version") == BOUNDARY_SCHEMA_VERSION:
        return load_boundary_round_dump(path)
    return load_cusp_minimal_dump(path)


def write_boundary_dump(source_dir: Path, output_dir: Path, max_edges_per_class: int, batch_size: int) -> None:
    payload, metadata = load_source_dump(source_dir)
    if bool(metadata.get("test_used_before_dump", False)):
        raise RuntimeError("source dump is invalid: official test was used before the dump")
    if "fedavg_candidate_trainable" not in payload and "global_after_fedavg_trainable" in payload:
        payload["fedavg_candidate_trainable"] = payload["global_after_fedavg_trainable"]
    payload["schema_version"] = BOUNDARY_SCHEMA_VERSION
    metadata = dict(metadata)
    metadata["schema_version"] = BOUNDARY_SCHEMA_VERSION
    metadata["source_dump"] = str(source_dir.resolve())
    metadata["source_round_state_hash"] = sha256_file(source_dir / "round_state.pt")
    metadata["test_used_before_dump"] = False

    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(payload, output_dir / "round_state.pt")
    write_json(output_dir / "metadata.json", metadata)
    cfg, trainer = build_promptfl_trainer(metadata, output_dir / "model_build")
    attach_audit_cache_to_round_dump(
        cfg, trainer, output_dir, max_edges_per_class=max_edges_per_class, batch_size=batch_size
    )
    print(f"Boundary Gate dump written: {output_dir}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dump-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--max-edges-per-class", type=int, default=3)
    parser.add_argument("--batch-size", type=int, default=256)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    write_boundary_dump(args.source_dump_dir, args.output_dir, args.max_edges_per_class, args.batch_size)


if __name__ == "__main__":
    main()
