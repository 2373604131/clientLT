from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np


# +1 means larger is broader/better; -1 means smaller is broader/better.
PRIMARY_DIRECTIONS = {
    "visual_subgroup_coverage": {
        "worst_cluster_accuracy": 1,
        "cluster_balanced_accuracy": 1,
        "recognized_cluster_fraction_at_50": 1,
    },
    "multi_view_robustness": {
        "worst_view_accuracy": 1,
        "prediction_consistency": 1,
        "worst_view_margin": 1,
        "clean_to_corruption_accuracy_drop": -1,
    },
    "neighbor_discrimination_breadth": {
        "target_vs_neighbor_pairwise_margin": 1,
        "worst_neighbor_margin": 1,
        "positive_margin_neighbor_coverage": 1,
    },
}


def _index_rows(rows: Sequence[Mapping], pair_keys: Sequence[str]) -> dict[tuple, Mapping]:
    output = {}
    for row in rows:
        key = tuple(row[name] for name in pair_keys)
        if key in output:
            raise ValueError(f"duplicate paired breadth row: {key}")
        output[key] = row
    return output


def preregistered_direction_gate(
    dirichlet_by_family: Mapping[str, Sequence[Mapping]],
    clientlt_by_family: Mapping[str, Sequence[Mapping]],
    *,
    pair_keys: Sequence[str] = ("seed", "round", "tail_class"),
) -> dict:
    """Apply the frozen no-cherry-picking directional family rule.

    This is a directional gate, not the inferential analysis. Confidence
    intervals and class-clustered inference must still be reported separately.
    """
    expected = set(PRIMARY_DIRECTIONS)
    if set(dirichlet_by_family) != expected or set(clientlt_by_family) != expected:
        raise ValueError("comparison must contain all three preregistered families")
    families = {}
    for family, directions in PRIMARY_DIRECTIONS.items():
        left = _index_rows(dirichlet_by_family[family], pair_keys)
        right = _index_rows(clientlt_by_family[family], pair_keys)
        if set(left) != set(right) or not left:
            raise ValueError(f"unmatched Dirichlet/Client-LT rows for {family}")
        endpoints = {}
        for metric, direction in directions.items():
            deltas = []
            for key in sorted(left):
                if metric not in left[key] or metric not in right[key]:
                    raise ValueError(f"missing primary metric {metric} for {family}")
                delta = float(direction) * (
                    float(left[key][metric]) - float(right[key][metric])
                )
                if not np.isfinite(delta):
                    raise ValueError(f"non-finite paired delta for {family}/{metric}")
                deltas.append(delta)
            endpoints[metric] = {
                "broader_dirichlet_mean_delta": float(np.mean(deltas)),
                "paired_unit_count": len(deltas),
                "direction_supports_narrower_clientlt": bool(np.mean(deltas) > 0.0),
            }
        families[family] = {
            "primary_endpoints": endpoints,
            "directionally_consistent": all(
                row["direction_supports_narrower_clientlt"]
                for row in endpoints.values()
            ),
        }
    support_count = sum(row["directionally_consistent"] for row in families.values())
    return {
        "families": families,
        "supporting_family_count": int(support_count),
        "required_supporting_family_count": 2,
        "directional_gate_pass": bool(support_count >= 2),
        "inference_required_separately": True,
    }
