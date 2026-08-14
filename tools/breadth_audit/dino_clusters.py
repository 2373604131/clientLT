from __future__ import annotations

import argparse
import hashlib
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image

from tools.breadth_audit.protocol import MECHANISM_VALIDATION_PROTOCOL, TAIL_CLASSES
from tools.semantic_acquisition.common import stable_hash, write_json


IMAGENET_MEAN = np.asarray([0.485, 0.456, 0.406], dtype=np.float32)
IMAGENET_STD = np.asarray([0.229, 0.224, 0.225], dtype=np.float32)


def _load_test_data(data_dir: Path) -> tuple[np.ndarray, np.ndarray]:
    with (Path(data_dir) / "test").open("rb") as handle:
        payload = pickle.load(handle, encoding="latin1")
    images = np.asarray(payload["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)
    labels = np.asarray(payload["fine_labels"], dtype=np.int64)
    return images, labels


def _dino_tensor(image: np.ndarray):
    import torch

    resampling = getattr(Image, "Resampling", Image)
    pil = Image.fromarray(image, mode="RGB").resize((224, 224), resampling.BICUBIC)
    value = np.asarray(pil, dtype=np.float32) / 255.0
    value = (value - IMAGENET_MEAN) / IMAGENET_STD
    return torch.from_numpy(value.transpose(2, 0, 1).copy())


def _state_hash(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        value = tensor.detach().cpu().contiguous()
        digest.update(name.encode("utf-8"))
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_dinov2(model_name: str, local_repo: Path | None = None):
    import torch

    if local_repo is None:
        return torch.hub.load("facebookresearch/dinov2", model_name)
    return torch.hub.load(str(Path(local_repo).resolve()), model_name, source="local")


def extract_embeddings(model, images: np.ndarray, *, batch_size: int, device: str) -> np.ndarray:
    import torch

    model = model.to(device).eval()
    for parameter in model.parameters():
        parameter.requires_grad_(False)
    chunks = []
    with torch.inference_mode():
        for start in range(0, len(images), int(batch_size)):
            batch = torch.stack([_dino_tensor(image) for image in images[start:start + batch_size]])
            features = model(batch.to(device))
            if isinstance(features, dict):
                features = features.get("x_norm_clstoken", features.get("x_prenorm"))
            if features is None or features.ndim != 2:
                raise RuntimeError("DINOv2 did not return one embedding vector per image")
            features = torch.nn.functional.normalize(features.float(), dim=1)
            chunks.append(features.cpu().numpy())
    return np.concatenate(chunks, axis=0).astype(np.float32)


def build_tail_clusters(
    embeddings: np.ndarray,
    labels: np.ndarray,
    tail_classes: list[int],
    *,
    clusters_per_class: int,
    seed: int,
    n_init: int,
) -> np.ndarray:
    from sklearn.cluster import KMeans

    cluster_ids = np.full(labels.shape, -1, dtype=np.int64)
    for class_id in tail_classes:
        positions = np.flatnonzero(labels == int(class_id))
        if len(positions) < int(clusters_per_class):
            raise ValueError(f"tail class {class_id} has fewer samples than clusters")
        estimator = KMeans(
            n_clusters=int(clusters_per_class), random_state=int(seed),
            n_init=int(n_init), algorithm="lloyd",
        )
        cluster_ids[positions] = estimator.fit_predict(embeddings[positions])
    return cluster_ids


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Create the one-time frozen DINOv2 tail-subgroup artifact for E1."
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("DATA/cifar-100/cifar-100-python"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("output/e1_strength_breadth/frozen_eval"),
    )
    parser.add_argument("--model-name", default="dinov2_vitb14")
    parser.add_argument("--dinov2-repo", type=Path)
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--device", default="cuda")
    args = parser.parse_args()

    frozen = MECHANISM_VALIDATION_PROTOCOL["breadth_audit"]["visual_subgroups"]
    if args.model_name != frozen["encoder"]:
        raise ValueError(
            f"mechanism validation requires {frozen['encoder']}, got {args.model_name}"
        )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    artifact = args.output_dir / "dino_tail_clusters.npz"
    metadata_path = args.output_dir / "dino_tail_clusters_meta.json"
    if artifact.exists() or metadata_path.exists():
        raise FileExistsError(
            "Frozen DINO cluster artifacts already exist; refusing to overwrite them"
        )

    images, labels = _load_test_data(args.data_dir)
    positions = np.flatnonzero(np.isin(labels, TAIL_CLASSES))
    model = load_dinov2(args.model_name, args.dinov2_repo)
    model_hash = _state_hash(model)
    embeddings = extract_embeddings(
        model, images[positions], batch_size=args.batch_size, device=args.device
    )
    tail_labels = labels[positions]
    cluster_ids = build_tail_clusters(
        embeddings, tail_labels, TAIL_CLASSES,
        clusters_per_class=int(frozen["clusters_per_tail_class"]),
        seed=int(frozen["kmeans_seed"]), n_init=int(frozen["kmeans_n_init"]),
    )
    np.savez_compressed(
        artifact,
        raw_test_indices=positions.astype(np.int64),
        labels=tail_labels.astype(np.int64),
        cluster_ids=cluster_ids.astype(np.int64),
        embeddings=embeddings,
        tail_class_ids=np.asarray(TAIL_CLASSES, dtype=np.int64),
    )
    metadata = {
        "artifact": artifact.name,
        "artifact_sha256": hashlib.sha256(artifact.read_bytes()).hexdigest(),
        "model_name": args.model_name,
        "model_state_sha256": model_hash,
        "sample_count": int(len(positions)),
        "raw_test_indices_hash": stable_hash(positions.tolist()),
        "tail_classes": TAIL_CLASSES,
        "clusters_per_tail_class": int(frozen["clusters_per_tail_class"]),
        "kmeans_seed": int(frozen["kmeans_seed"]),
        "kmeans_n_init": int(frozen["kmeans_n_init"]),
        "reuse_requirement": "same_artifact_for_all_topologies_rounds_seeds_and_methods",
    }
    write_json(metadata_path, metadata)
    print(json.dumps({"artifact": str(artifact.resolve()), "metadata": str(metadata_path.resolve())}))


if __name__ == "__main__":
    main()
