from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np

from tools.semantic_acquisition.common import deterministic_choice


def select_head_safety_ids(
    raw_train_labels: Sequence[int],
    lt_raw_ids: Sequence[int],
    used_sample_ids: Iterable[str],
    class_ids: Sequence[int],
    samples_per_class: int,
) -> dict[int, list[int]]:
    """Select train-only probes disjoint from every Carrier-B training sample.

    Selecting from the residual LT pool is invalid for low-frequency non-tail
    classes because Carrier-B may have consumed nearly all of their examples.
    Conversely, excluding the whole LT pool is impossible for the largest head
    classes because all 500 raw examples can belong to that pool. We therefore
    use the original train split, exclude every manifested Carrier-B sample,
    and order outside-LT examples before unused in-LT examples.
    """
    labels = np.asarray(raw_train_labels, dtype=np.int64)
    if labels.ndim != 1 or labels.size == 0:
        raise ValueError("raw_train_labels must be a non-empty vector")
    if int(samples_per_class) <= 0:
        raise ValueError("samples_per_class must be positive")
    lt_set = {int(value) for value in lt_raw_ids}
    used_raw = set()
    for sample_id in used_sample_ids:
        split, raw = str(sample_id).split(":", 1)
        if split != "train":
            raise ValueError(f"Head-safety exclusion contains a non-train ID: {sample_id}")
        used_raw.add(int(raw))
    output = {}
    for class_id in [int(value) for value in class_ids]:
        eligible = [
            int(raw_id) for raw_id in np.flatnonzero(labels == class_id).tolist()
            if int(raw_id) not in used_raw
        ]
        outside = [raw_id for raw_id in eligible if raw_id not in lt_set]
        inside = [raw_id for raw_id in eligible if raw_id in lt_set]
        outside_count = min(int(samples_per_class), len(outside))
        chosen = deterministic_choice(
            outside, outside_count,
            "functional-breadth-head-safety-outside", 42, class_id,
        )
        remainder = int(samples_per_class) - len(chosen)
        chosen.extend(deterministic_choice(
            inside, remainder,
            "functional-breadth-head-safety-unused-lt", 42, class_id,
        ))
        output[class_id] = chosen
    return output
