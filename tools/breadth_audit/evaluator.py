from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from tools.breadth_audit.metrics import (
    multiview_robustness_metrics,
    neighbor_discrimination_metrics,
    visual_subgroup_metrics,
)
from tools.breadth_audit.protocol import MECHANISM_VALIDATION_PROTOCOL
from tools.breadth_audit.views import FROZEN_VIEW_NAMES, fixed_view


def predict_fixed_views(
    model,
    images: Sequence,
    test_transform,
    *,
    batch_size: int,
    device=None,
) -> dict[str, np.ndarray]:
    """Run the current in-memory model on all preregistered deterministic views."""
    import torch

    if device is None:
        device = next(model.parameters()).device
    was_training = bool(model.training)
    model.eval()
    output = {}
    with torch.inference_mode():
        for view in FROZEN_VIEW_NAMES:
            chunks = []
            for start in range(0, len(images), int(batch_size)):
                tensors = [
                    test_transform(fixed_view(image, view))
                    for image in images[start:start + batch_size]
                ]
                logits = model(torch.stack(tensors).to(device))
                chunks.append(logits.detach().float().cpu().numpy())
            output[view] = np.concatenate(chunks, axis=0)
    model.train(was_training)
    return output


def evaluate_three_breadth_families(
    *,
    logits_by_view: Mapping[str, np.ndarray],
    labels,
    cluster_ids,
    neighbors_by_tail: Mapping[int, Sequence[int]],
    tail_classes: Sequence[int],
) -> dict[str, list[dict]]:
    """Evaluate all three families together so none can be silently omitted."""
    frozen = MECHANISM_VALIDATION_PROTOCOL["breadth_audit"]["visual_subgroups"]
    clean = logits_by_view["clean"]
    return {
        "visual_subgroup_coverage": visual_subgroup_metrics(
            clean, labels, cluster_ids, tail_classes,
            recognized_accuracy_threshold=float(
                frozen["recognized_cluster_accuracy_threshold"]
            ),
        ),
        "multi_view_robustness": multiview_robustness_metrics(
            logits_by_view, labels, tail_classes
        ),
        "neighbor_discrimination_breadth": neighbor_discrimination_metrics(
            clean, labels, neighbors_by_tail, tail_classes
        ),
    }
