"""Build deterministic sample and semantic manifests for Experiments B and C."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.carrier_access_audit.protocol import NON_TAIL_CLASSES, TAIL_CLASSES, frozen_protocol
from tools.client_update_audit.manifests import load_exact_lt_pool
from tools.semantic_acquisition.common import (
    deterministic_choice,
    file_sha256,
    stable_hash,
    stable_seed,
    write_csv,
    write_json,
)


def _selected_rows(
    labels: np.ndarray,
    raw_ids: np.ndarray,
    class_ids: list[int],
    budget: int,
    role: str,
) -> list[dict]:
    rows = []
    for class_id in class_ids:
        positions = np.flatnonzero(labels == class_id).tolist()
        chosen = deterministic_choice(positions, budget, "carrier-access", 42, role, class_id)
        for slot, lt_index in enumerate(chosen):
            rows.append({
                "data_seed": 42,
                "role": role,
                "class_id": class_id,
                "slot": slot,
                "lt_index": int(lt_index),
                "raw_train_index": int(raw_ids[lt_index]),
                "base_sample_id": f"train:{int(raw_ids[lt_index])}",
                "label": int(labels[lt_index]),
            })
    return rows


def build(data_dir: Path, similarity_file: Path, output_dir: Path) -> dict:
    data_dir, similarity_file, output_dir = Path(data_dir), Path(similarity_file), Path(output_dir)
    labels, raw_ids, _, class_names, _ = load_exact_lt_pool(data_dir)
    similarity = np.load(similarity_file)
    if similarity.shape != (100, 100) or not np.isfinite(similarity).all():
        raise ValueError(f"Expected a finite 100x100 semantic similarity matrix, got {similarity.shape}")
    candidate_budget = int(frozen_protocol()["experiment_b"]["candidate_train_samples_per_class"])
    tail_budget = int(frozen_protocol()["experiment_c"]["tail_train_samples_per_class"])
    counts = np.bincount(labels, minlength=100)
    if int(counts[NON_TAIL_CLASSES].min()) < candidate_budget:
        raise RuntimeError("A non-tail class cannot supply the frozen candidate budget")
    if int(counts[TAIL_CLASSES].min()) < tail_budget:
        raise RuntimeError("A tail class cannot supply the frozen tail budget")

    candidate_rows = _selected_rows(labels, raw_ids, NON_TAIL_CLASSES, candidate_budget, "candidate")
    tail_rows = _selected_rows(labels, raw_ids, TAIL_CLASSES, tail_budget, "private_tail_evidence")
    execution_rows = []
    for row in candidate_rows + tail_rows:
        for epoch in (1, 2, 3):
            execution_rows.append({
                **row,
                "epoch": epoch,
                "batch_index": 0,
                "position_in_batch": int(row["slot"]),
                "augmentation_seed": stable_seed(
                    "carrier-access-augmentation", 42, row["role"], row["class_id"], epoch, row["base_sample_id"]
                ),
            })

    semantic_rows = []
    for tail_class in TAIL_CLASSES:
        ranked = sorted(NON_TAIL_CLASSES, key=lambda other: (-float(similarity[tail_class, other]), other))
        for rank, candidate_class in enumerate(ranked, start=1):
            semantic_rows.append({
                "data_seed": 42,
                "tail_class": tail_class,
                "tail_class_name": class_names[tail_class],
                "candidate_class": candidate_class,
                "candidate_class_name": class_names[candidate_class],
                "semantic_rank": rank,
                "cosine_similarity": float(similarity[tail_class, candidate_class]),
                "related_top10": rank <= 10,
                "unrelated_bottom10": rank >= 71,
            })

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "candidate_samples.csv", candidate_rows)
    write_csv(output_dir / "private_tail_samples.csv", tail_rows)
    write_csv(output_dir / "training_execution.csv", execution_rows)
    write_csv(output_dir / "semantic_pairs.csv", semantic_rows)
    names = (
        "candidate_samples.csv", "private_tail_samples.csv",
        "training_execution.csv", "semantic_pairs.csv",
    )
    contract = {
        "protocol": frozen_protocol(),
        "data_dir": str(data_dir.resolve()),
        "similarity_file": str(similarity_file.resolve()),
        "similarity_sha256": file_sha256(similarity_file),
        "class_names_hash": stable_hash(class_names),
        "lt_pool_hash": stable_hash([(int(raw), int(label)) for raw, label in zip(raw_ids, labels)]),
        "candidate_sample_count": len(candidate_rows),
        "private_tail_sample_count": len(tail_rows),
        "semantic_pair_count": len(semantic_rows),
        "manifest_hashes": {name: file_sha256(output_dir / name) for name in names},
    }
    write_json(output_dir / "manifest_contract.json", contract)
    return {
        "output_dir": str(output_dir.resolve()),
        "candidate_rows": len(candidate_rows),
        "private_tail_rows": len(tail_rows),
        "semantic_pairs": len(semantic_rows),
        "structural_gate": "PASS",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("DATA/cifar-100/cifar-100-python"))
    parser.add_argument("--similarity-file", type=Path, default=Path("output/p0_v1_context_colocation_v2/clip_similarity.npy"))
    parser.add_argument("--output-dir", type=Path, default=Path("output/carrier_access_audit/manifests"))
    args = parser.parse_args()
    print(json.dumps(build(args.data_dir, args.similarity_file, args.output_dir)))


if __name__ == "__main__":
    main()
