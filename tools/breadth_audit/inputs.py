from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Sequence

import numpy as np

from tools.breadth_audit.protocol import MECHANISM_VALIDATION_PROTOCOL
from tools.semantic_acquisition.common import file_sha256, stable_hash


FROZEN_NEIGHBOR_PATH = Path(__file__).with_name("frozen_neighbors.json")


def load_preregistered_neighbors(
    tail_classes: Sequence[int],
    path: Path | None = None,
) -> tuple[dict[int, list[int]], dict]:
    """Load the checked-in E1 neighbor table without requiring old outputs."""
    path = FROZEN_NEIGHBOR_PATH if path is None else Path(path)
    if not path.is_file():
        raise FileNotFoundError(f"missing preregistered neighbor table: {path}")
    metadata = json.loads(path.read_text(encoding="utf-8"))
    expected_tail = sorted(int(value) for value in tail_classes)
    observed_tail = sorted(int(value) for value in metadata.get("tail_classes", []))
    if observed_tail != expected_tail or expected_tail != list(range(80, 100)):
        raise RuntimeError(
            f"neighbor-table tail classes differ: {observed_tail} != {expected_tail}"
        )
    neighbors = {
        int(class_id): [int(value) for value in values]
        for class_id, values in metadata.get("neighbors", {}).items()
    }
    expected_count = int(
        MECHANISM_VALIDATION_PROTOCOL["breadth_audit"]
        ["neighbor_discrimination"]["neighbors_per_tail_class"]
    )
    tail_set = set(expected_tail)
    if set(neighbors) != tail_set:
        raise RuntimeError("preregistered neighbor table does not cover every tail class")
    for class_id, values in neighbors.items():
        if len(values) != expected_count or len(set(values)) != expected_count:
            raise RuntimeError(f"invalid neighbor list for class {class_id}: {values}")
        if class_id in values or tail_set.intersection(values):
            raise RuntimeError(f"neighbor list for class {class_id} is not non-tail-only")
    observed_hash = stable_hash({str(key): value for key, value in neighbors.items()})
    if observed_hash != metadata.get("neighbors_hash"):
        raise RuntimeError("preregistered neighbor-table hash mismatch")
    return neighbors, {**metadata, "table_path": str(path.resolve())}


def load_frozen_neighbors(
    v1_dir: Path,
    tail_classes: Sequence[int],
) -> tuple[dict[int, list[int]], dict]:
    """Derive fixed Top-10 neighbors from the frozen V1 similarity matrix.

    The old V1 neighbor table used a count-derived tail set and consequently
    contains class 79 instead of the index-defined class 80.  The similarity
    matrix itself covers all 100 classes and is unaffected.  We therefore rank
    it once under the corrected, preregistered tail set and hash the result.
    This is independent of all E1 model predictions.
    """
    v1_dir = Path(v1_dir)
    similarity_path = v1_dir / "clip_similarity.npy"
    metadata_path = v1_dir / "clip_similarity_meta.json"
    for path in (similarity_path, metadata_path):
        if not path.is_file():
            raise FileNotFoundError(f"missing frozen V1 semantic artifact: {path}")
    similarity = np.load(similarity_path)
    if similarity.shape != (100, 100):
        raise ValueError(f"unexpected CLIP similarity shape: {similarity.shape}")
    tail_set = set(int(value) for value in tail_classes)
    if tail_set != set(range(80, 100)):
        raise ValueError(
            "E1 frozen neighbors require the index-defined CIFAR-100-LT tail 80--99"
        )
    non_tail = sorted(set(range(100)) - tail_set)
    neighbors = {}
    expected_count = int(
        MECHANISM_VALIDATION_PROTOCOL["breadth_audit"]
        ["neighbor_discrimination"]["neighbors_per_tail_class"]
    )
    for class_id in sorted(tail_set):
        neighbors[class_id] = sorted(
            non_tail,
            key=lambda other: (-float(similarity[class_id, other]), other),
        )[:expected_count]
    metadata = {
        "neighbor_derivation": "top10_from_frozen_similarity_under_tail_ids_80_to_99",
        "similarity_sha256": file_sha256(similarity_path),
        "similarity_metadata_sha256": file_sha256(metadata_path),
        "neighbors_hash": stable_hash({str(key): value for key, value in neighbors.items()}),
        "candidate_scope": "non_tail_only_primary",
        "tail_classes": sorted(tail_set),
    }
    return neighbors, metadata


def load_dino_clusters(
    artifact_path: Path,
    metadata_path: Path | None = None,
) -> tuple[dict[str, np.ndarray], dict]:
    """Load and verify the one-time DINO cluster artifact."""
    artifact_path = Path(artifact_path)
    if metadata_path is None:
        metadata_path = artifact_path.with_name("dino_tail_clusters_meta.json")
    metadata_path = Path(metadata_path)
    if not artifact_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError("DINO cluster artifact or metadata is missing")
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    observed_sha = hashlib.sha256(artifact_path.read_bytes()).hexdigest()
    if metadata.get("artifact_sha256") != observed_sha:
        raise RuntimeError("DINO cluster artifact hash differs from frozen metadata")
    required = {
        "raw_test_indices", "labels", "cluster_ids", "embeddings", "tail_class_ids"
    }
    with np.load(artifact_path) as arrays:
        if not required <= set(arrays.files):
            raise ValueError(f"DINO artifact lacks arrays: {sorted(required - set(arrays.files))}")
        result = {name: arrays[name].copy() for name in required}
    sample_count = len(result["labels"])
    if any(len(result[name]) != sample_count for name in (
        "raw_test_indices", "cluster_ids", "embeddings"
    )):
        raise ValueError("DINO artifact arrays have inconsistent sample counts")
    tail_classes = sorted(int(value) for value in result["tail_class_ids"].tolist())
    expected_tail = sorted(
        int(value) for value in
        MECHANISM_VALIDATION_PROTOCOL["dataset"]["tail_classes"]
    )
    if tail_classes != expected_tail:
        raise RuntimeError(f"DINO artifact tail classes differ: {tail_classes} != {expected_tail}")
    cluster_count = int(
        MECHANISM_VALIDATION_PROTOCOL["breadth_audit"]
        ["visual_subgroups"]["clusters_per_tail_class"]
    )
    for class_id in tail_classes:
        values = sorted(set(int(value) for value in result["cluster_ids"][result["labels"] == class_id]))
        if values != list(range(cluster_count)):
            raise RuntimeError(f"tail class {class_id} has invalid frozen clusters: {values}")
    return result, metadata
