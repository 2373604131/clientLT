"""Build the frozen CLIP-text semantic prior when old P0 output is unavailable."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from tools.analysis.run_p0_v1_context_colocation import encode_class_text, file_sha256, sha256_bytes
from tools.client_update_audit.manifests import load_exact_lt_pool
from tools.semantic_acquisition.common import stable_hash, write_json


def build(data_dir: Path, clip_checkpoint: Path, output_file: Path) -> dict:
    data_dir, clip_checkpoint, output_file = Path(data_dir), Path(clip_checkpoint), Path(output_file)
    if not clip_checkpoint.is_file():
        raise FileNotFoundError(f"CLIP checkpoint is missing: {clip_checkpoint}")
    _, _, _, class_names, _ = load_exact_lt_pool(data_dir)
    embeddings, metadata = encode_class_text(class_names, clip_checkpoint)
    similarity = embeddings @ embeddings.T
    if similarity.shape != (100, 100) or not np.isfinite(similarity).all():
        raise RuntimeError("CLIP semantic prior is not a finite 100x100 matrix")
    if not np.allclose(similarity, similarity.T, atol=1e-8):
        raise RuntimeError("CLIP semantic prior is not symmetric")
    output_file.parent.mkdir(parents=True, exist_ok=True)
    np.save(output_file, similarity)
    metadata.update({
        "artifact_role": "carrier_access_candidate_ranking_only",
        "class_names_hash": stable_hash(class_names),
        "embedding_sha256": sha256_bytes(embeddings),
        "similarity_sha256": sha256_bytes(similarity),
        "artifact_file_sha256": file_sha256(output_file),
        "shape": [100, 100],
        "training_or_image_encoding_used": False,
    })
    metadata_file = output_file.with_name(output_file.stem + "_meta.json")
    write_json(metadata_file, metadata)
    return {
        "artifact": str(output_file.resolve()),
        "metadata": str(metadata_file.resolve()),
        "sha256": metadata["artifact_file_sha256"],
        "shape": [100, 100],
        "source": "frozen_CLIP_text_encoder",
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data-dir", type=Path, default=Path("DATA/cifar-100/cifar-100-python"))
    parser.add_argument("--clip-checkpoint", type=Path, default=Path.home() / ".cache" / "clip" / "ViT-B-16.pt")
    parser.add_argument("--output-file", type=Path, default=Path("output/p0_v1_context_colocation_v2/clip_similarity.npy"))
    args = parser.parse_args()
    print(json.dumps(build(args.data_dir, args.clip_checkpoint, args.output_file)))


if __name__ == "__main__":
    main()
