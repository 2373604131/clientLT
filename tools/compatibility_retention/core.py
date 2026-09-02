"""Pure calculations for the compatibility-to-retention bridge."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np
import torch


CONDITIONS = ("hard_competitor", "matched_control")


def additive_post_state(
    theta0: Mapping[str, torch.Tensor],
    local_state: Mapping[str, torch.Tensor],
    background_state: Mapping[str, torch.Tensor],
) -> dict[str, torch.Tensor]:
    """Return theta0 + (local-theta0) + (background-theta0).

    ``background_state`` is the sample-weighted FedAvg state of the selected
    class-absent clients, all of which start at the same ``theta0``.  The tail
    and background scales are deliberately fixed at one: this bridge has no
    scale sweep or update-norm matching.
    """
    names = sorted(theta0)
    if set(local_state) != set(names) or set(background_state) != set(names):
        raise KeyError("theta0, local, and background LoRA states must have identical keys")
    result = {}
    for name in names:
        reference = theta0[name].detach().cpu()
        local = local_state[name].detach().cpu()
        background = background_state[name].detach().cpu()
        if local.shape != reference.shape or background.shape != reference.shape:
            raise ValueError(f"LoRA shape mismatch for {name}")
        value = (
            reference.float()
            + (local.float() - reference.float())
            + (background.float() - reference.float())
        )
        if not bool(torch.isfinite(value).all()):
            raise ValueError(f"Non-finite composed LoRA state for {name}")
        result[name] = value.to(dtype=reference.dtype)
    return result


def sample_weights(sample_counts: Mapping[int, int], selected_clients: Sequence[int]) -> dict[int, float]:
    """Ordinary sample-count FedAvg weights, normalized within selected clients."""
    selected = sorted(int(value) for value in selected_clients)
    if not selected or len(set(selected)) != len(selected):
        raise ValueError("selected_clients must be non-empty and unique")
    counts = {client: int(sample_counts[client]) for client in selected}
    if any(value <= 0 for value in counts.values()):
        raise ValueError("Every selected background client must have positive sample count")
    total = float(sum(counts.values()))
    return {client: counts[client] / total for client in selected}


def tail_retention_rows(rows: Sequence[Mapping]) -> list[dict]:
    """Average components within tail class, then form the sole endpoint R_c.

    Ratios are intentionally formed *after* averaging the five hard-negative
    pairs and data seeds.  This preserves the 20-tail-class inference unit and
    avoids unstable pair-level ratios with tiny denominators.
    """
    grouped: dict[tuple[int, str], dict[str, list[float]]] = {}
    for row in rows:
        condition = str(row["condition"])
        if condition not in CONDITIONS:
            raise ValueError(f"Unexpected bridge condition: {condition}")
        key = (int(row["tail_class"]), condition)
        values = grouped.setdefault(key, {"g_local": [], "g_post": []})
        values["g_local"].append(float(row["g_local"]))
        values["g_post"].append(float(row["g_post"]))
    tail_classes = sorted({key[0] for key in grouped})
    output = []
    for tail_class in tail_classes:
        condition_values = {}
        for condition in CONDITIONS:
            key = (tail_class, condition)
            if key not in grouped:
                raise ValueError(f"Tail class {tail_class} is missing condition {condition}")
            local = float(np.mean(grouped[key]["g_local"]))
            post = float(np.mean(grouped[key]["g_post"]))
            if not np.isfinite(local) or not np.isfinite(post):
                raise ValueError(f"Non-finite gain for tail class {tail_class}, {condition}")
            if local <= 0.0:
                raise ValueError(
                    f"Retention ratio is undefined for non-positive class-level local gain: "
                    f"tail={tail_class}, condition={condition}, G_local={local}"
                )
            condition_values[condition] = {
                "g_local": local,
                "g_post": post,
                "retention_ratio": post / local,
            }
        hard = condition_values["hard_competitor"]
        control = condition_values["matched_control"]
        output.append({
            "tail_class": tail_class,
            "hard_g_local": hard["g_local"],
            "hard_g_post": hard["g_post"],
            "hard_retention_ratio": hard["retention_ratio"],
            "control_g_local": control["g_local"],
            "control_g_post": control["g_post"],
            "control_retention_ratio": control["retention_ratio"],
            "hard_minus_control_retention_ratio": (
                hard["retention_ratio"] - control["retention_ratio"]
            ),
        })
    return output

