"""Create read-only representative-image sheets for frozen DINO clusters."""

from __future__ import annotations

import argparse
import csv
import json
import pickle
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def _load_cifar100(data_dir: Path) -> tuple[np.ndarray, list[str]]:
    data_dir = Path(data_dir)
    with (data_dir / "test").open("rb") as handle:
        test = pickle.load(handle, encoding="latin1")
    with (data_dir / "meta").open("rb") as handle:
        meta = pickle.load(handle, encoding="latin1")
    images = np.asarray(test["data"], dtype=np.uint8).reshape(-1, 3, 32, 32)
    images = images.transpose(0, 2, 3, 1)
    names = [str(value).replace("_", " ") for value in meta["fine_label_names"]]
    return images, names


def representative_positions(
    embeddings: np.ndarray,
    cluster_ids: np.ndarray,
    cluster_id: int,
    count: int,
) -> tuple[np.ndarray, np.ndarray]:
    """Return cluster-local rows nearest to its embedding centroid."""
    members = np.flatnonzero(cluster_ids == int(cluster_id))
    if not len(members):
        raise ValueError(f"cluster {cluster_id} is empty")
    values = np.asarray(embeddings[members], dtype=np.float64)
    centroid = values.mean(axis=0, keepdims=True)
    distances = np.square(values - centroid).sum(axis=1)
    order = np.lexsort((members, distances))[: min(int(count), len(members))]
    return members[order], distances[order]


def _make_sheet(
    images: np.ndarray,
    representatives: dict[int, list[int]],
    cluster_sizes: dict[int, int],
    *,
    title: str,
    image_scale: int,
) -> Image.Image:
    clusters = sorted(representatives)
    rows = max(len(values) for values in representatives.values())
    tile = 32 * int(image_scale)
    header = 42
    canvas = Image.new("RGB", (len(clusters) * tile, header + rows * tile), "white")
    draw = ImageDraw.Draw(canvas)
    draw.text((4, 3), title, fill="black")
    for column, cluster_id in enumerate(clusters):
        x = column * tile
        draw.text(
            (x + 4, 20),
            f"cluster {cluster_id} (n={cluster_sizes[cluster_id]})",
            fill="black",
        )
        for row, raw_index in enumerate(representatives[cluster_id]):
            resampling = getattr(Image, "Resampling", Image)
            tile_image = Image.fromarray(images[raw_index], mode="RGB").resize(
                (tile, tile), resampling.NEAREST
            )
            canvas.paste(tile_image, (x, header + row * tile))
    return canvas


def main() -> None:
    parser = argparse.ArgumentParser(
        description=(
            "Visualize nearest-to-centroid representatives of the immutable "
            "DINO tail clusters without modifying their assignments."
        )
    )
    parser.add_argument(
        "--data-dir", type=Path,
        default=Path("DATA/cifar-100/cifar-100-python"),
    )
    parser.add_argument(
        "--artifact", type=Path,
        default=Path("output/e1_strength_breadth/frozen_eval/dino_tail_clusters.npz"),
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("output/e1_strength_breadth/cluster_inspection"),
    )
    parser.add_argument("--representatives", type=int, default=8)
    parser.add_argument("--image-scale", type=int, default=4)
    args = parser.parse_args()
    if args.representatives <= 0 or args.image_scale <= 0:
        raise ValueError("representatives and image-scale must be positive")

    images, class_names = _load_cifar100(args.data_dir)
    with np.load(args.artifact) as frozen:
        required = {
            "raw_test_indices", "labels", "cluster_ids", "embeddings",
            "tail_class_ids",
        }
        if not required <= set(frozen.files):
            raise ValueError(f"cluster artifact lacks {sorted(required - set(frozen.files))}")
        raw_indices = frozen["raw_test_indices"].astype(np.int64)
        labels = frozen["labels"].astype(np.int64)
        cluster_ids = frozen["cluster_ids"].astype(np.int64)
        embeddings = frozen["embeddings"].astype(np.float32)
        tail_classes = frozen["tail_class_ids"].astype(np.int64).tolist()
    if not (len(raw_indices) == len(labels) == len(cluster_ids) == len(embeddings)):
        raise ValueError("frozen cluster arrays have inconsistent lengths")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    csv_rows = []
    written = []
    for class_id in tail_classes:
        class_rows = np.flatnonzero(labels == int(class_id))
        local_clusters = cluster_ids[class_rows]
        local_embeddings = embeddings[class_rows]
        representatives: dict[int, list[int]] = {}
        cluster_sizes: dict[int, int] = {}
        for cluster_id in sorted(set(local_clusters.tolist())):
            chosen, distances = representative_positions(
                local_embeddings, local_clusters, cluster_id, args.representatives
            )
            artifact_rows = class_rows[chosen]
            raw_rows = raw_indices[artifact_rows]
            representatives[int(cluster_id)] = raw_rows.tolist()
            cluster_sizes[int(cluster_id)] = int(np.sum(local_clusters == cluster_id))
            for rank, (artifact_row, raw_row, distance) in enumerate(
                zip(artifact_rows, raw_rows, distances), start=1
            ):
                csv_rows.append({
                    "tail_class": int(class_id),
                    "class_name": class_names[int(class_id)],
                    "cluster_id": int(cluster_id),
                    "cluster_size": cluster_sizes[int(cluster_id)],
                    "representative_rank": rank,
                    "artifact_row": int(artifact_row),
                    "raw_test_index": int(raw_row),
                    "squared_distance_to_centroid": float(distance),
                })
        sheet = _make_sheet(
            images, representatives, cluster_sizes,
            title=f"class {class_id}: {class_names[int(class_id)]}",
            image_scale=args.image_scale,
        )
        path = args.output_dir / f"tail_class_{int(class_id):03d}_clusters.png"
        sheet.save(path)
        written.append(path.name)

    csv_path = args.output_dir / "cluster_representatives.csv"
    with csv_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(csv_rows[0]))
        writer.writeheader()
        writer.writerows(csv_rows)
    manifest = {
        "source_artifact": str(args.artifact.resolve()),
        "selection": "nearest_to_per_class_per_cluster_embedding_centroid",
        "representatives_per_cluster": int(args.representatives),
        "assignment_mutated": False,
        "sheets": written,
        "representatives_csv": csv_path.name,
        "interpretation_warning": (
            "Cluster ids are arbitrary and class-local; the same id has no "
            "shared meaning across tail classes."
        ),
    }
    manifest_path = args.output_dir / "inspection_manifest.json"
    manifest_path.write_text(
        json.dumps(manifest, ensure_ascii=False, indent=2, sort_keys=True) + "\n",
        encoding="utf-8",
    )
    print(json.dumps({
        "output_dir": str(args.output_dir.resolve()),
        "sheet_count": len(written),
        "representative_rows": len(csv_rows),
        "artifact_unchanged": True,
    }))


if __name__ == "__main__":
    main()
