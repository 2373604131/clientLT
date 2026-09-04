"""Frozen, train-only ERI protocol construction."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Sequence

from tools.semantic_acquisition.common import deterministic_choice
from utils.functional_coverage_validation import (
    _TrainOnlyCifar100,
    _exact_lt_raw_ids,
    _locate_cifar100,
)


SCHEMA_VERSION = "eri_closure_protocol_v1"
DEFAULT_TAIL_CLASSES = tuple(range(80, 100))
DEFAULT_AUDIT_ROUNDS = (1, 2, 3, 4, 5, 10, 20, 30, 40, 50, 60, 70, 80, 90, 100)


def parse_eri_rounds(value, total_rounds: int | None = None) -> list[int]:
    """Parse one-based, unique ERI rounds and validate their trajectory range."""
    raw = value if isinstance(value, (list, tuple, set)) else str(value).split(",")
    rounds = sorted({int(item) for item in raw if str(item).strip()})
    if not rounds or rounds[0] < 1:
        raise ValueError("ERI audit rounds must contain positive one-based round numbers")
    if total_rounds is not None and rounds[-1] > int(total_rounds):
        raise ValueError(f"ERI audit rounds exceed --round={total_rounds}: {rounds}")
    return rounds


def _write_csv(path: Path, rows: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]) if rows else ["class_id"])
        writer.writeheader()
        writer.writerows(rows)


def build_protocol(
    output_dir: str | Path,
    *,
    data_root: str | Path,
    tail_classes: Sequence[int] = DEFAULT_TAIL_CLASSES,
    samples_per_class: int = 10,
    imbalance_factor: float = 0.01,
    imbalance_type: str = "exp",
    audit_rounds: Sequence[int] = DEFAULT_AUDIT_ROUNDS,
    seed: int = 20260904,
    overwrite: bool = False,
) -> Path:
    """Create an immutable train-only probe manifest shared by every run.

    Probes are selected from CIFAR-100 training records that are *not* in the
    long-tail federated pool.  This implements the no-official-test-access
    boundary needed by ERI attribution and replay.
    """
    root = Path(output_dir)
    manifest_path = root / "probe_manifest.csv"
    protocol_path = root / "eri_protocol.json"
    if protocol_path.exists() and not overwrite:
        if not manifest_path.exists():
            raise RuntimeError(f"Existing ERI protocol lacks its probe manifest: {root}")
        return root

    tail = sorted({int(item) for item in tail_classes})
    if not tail or min(tail) < 0 or max(tail) >= 100:
        raise ValueError(f"Expected non-empty CIFAR-100 tail ids, got {tail}")
    if int(samples_per_class) < 1:
        raise ValueError("samples_per_class must be positive")
    rounds = parse_eri_rounds(audit_rounds)
    store = _TrainOnlyCifar100(_locate_cifar100(Path(data_root)))
    lt_raw_ids = set(
        int(item)
        for item in _exact_lt_raw_ids(store.labels, imbalance_factor, imbalance_type).tolist()
    )
    rows: list[dict] = []
    for class_id in tail:
        available = [
            int(index)
            for index, label in enumerate(store.labels.tolist())
            if int(label) == class_id and int(index) not in lt_raw_ids
        ]
        chosen = deterministic_choice(
            available, int(samples_per_class), "eri-closure-train-only-probe", int(seed), class_id
        )
        for slot, raw_id in enumerate(chosen):
            rows.append(
                {
                    "class_id": class_id,
                    "slot": slot,
                    "raw_train_index": raw_id,
                    "excluded_from_federated_lt_pool": 1,
                }
            )
    _write_csv(manifest_path, rows)
    root.mkdir(parents=True, exist_ok=True)
    protocol = {
        "schema_version": SCHEMA_VERSION,
        "metric": "mean true-class log-odds: z_c - logsumexp(z_not_c)",
        "data_access": "CIFAR-100 train split only; probe rows excluded from LT training pool",
        "tail_class_ids": tail,
        "samples_per_class": int(samples_per_class),
        "imbalance_factor": float(imbalance_factor),
        "imbalance_type": str(imbalance_type),
        "audit_rounds": rounds,
        "probe_seed": int(seed),
        "quadrature": "Gauss-Legendre on [0, 1]; default 8 nodes",
        "eri": "R / (W + H + epsilon), where H is class-absent positive refresh",
        "primary_unit": "per tail class, per audited trajectory round",
    }
    protocol_path.write_text(json.dumps(protocol, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    return root


def load_protocol(protocol_dir: str | Path) -> tuple[dict, list[dict]]:
    root = Path(protocol_dir)
    protocol = json.loads((root / "eri_protocol.json").read_text(encoding="utf-8"))
    with (root / "probe_manifest.csv").open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    return protocol, rows
