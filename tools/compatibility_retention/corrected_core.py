"""Pure calculations for the background-adjusted retention endpoint."""

from __future__ import annotations

from collections.abc import Mapping, Sequence

import numpy as np

from tools.compatibility_retention.core import CONDITIONS


def background_adjusted_components(
    *,
    theta0_m_c: float,
    local_m_c: float,
    background_only_m_c: float,
    post_m_c: float,
) -> dict[str, float]:
    """Return the local and background-adjusted post tail contributions.

    The corrected numerator is a difference-in-differences contrast in model
    state space.  It removes the direct effect of the background update:

        G_local       = M(theta0 + delta_tail) - M(theta0)
        G_post_marginal = M(theta0 + delta_tail + delta_bg)
                          - M(theta0 + delta_bg)
    """
    values = np.asarray(
        [theta0_m_c, local_m_c, background_only_m_c, post_m_c], dtype=np.float64
    )
    if not np.isfinite(values).all():
        raise ValueError("Background-adjusted retention received a non-finite margin")
    return {
        "g_local": float(local_m_c - theta0_m_c),
        "g_post_marginal": float(post_m_c - background_only_m_c),
    }


def corrected_tail_retention_rows(rows: Sequence[Mapping]) -> list[dict]:
    """Average components within tail class, then form corrected R*_c.

    Five hard-negative pairs and data seeds are averaged before division.  A
    non-positive class-level local gain invalidates rather than silently drops
    the corresponding inference unit.
    """
    grouped: dict[tuple[int, str], dict[str, list[float]]] = {}
    for row in rows:
        condition = str(row["condition"])
        if condition not in CONDITIONS:
            raise ValueError(f"Unexpected corrected-bridge condition: {condition}")
        key = (int(row["tail_class"]), condition)
        values = grouped.setdefault(key, {"g_local": [], "g_post_marginal": []})
        values["g_local"].append(float(row["g_local"]))
        values["g_post_marginal"].append(float(row["g_post_marginal"]))
    tail_classes = sorted({key[0] for key in grouped})
    output = []
    for tail_class in tail_classes:
        by_condition = {}
        for condition in CONDITIONS:
            key = (tail_class, condition)
            if key not in grouped:
                raise ValueError(f"Tail class {tail_class} is missing condition {condition}")
            local = float(np.mean(grouped[key]["g_local"]))
            post_marginal = float(np.mean(grouped[key]["g_post_marginal"]))
            if not np.isfinite(local) or not np.isfinite(post_marginal):
                raise ValueError(f"Non-finite corrected gain for tail {tail_class}, {condition}")
            if local <= 0.0:
                raise ValueError(
                    "Corrected retention is undefined for non-positive class-level local gain: "
                    f"tail={tail_class}, condition={condition}, G_local={local}"
                )
            by_condition[condition] = {
                "g_local": local,
                "g_post_marginal": post_marginal,
                "corrected_retention_ratio": post_marginal / local,
            }
        hard = by_condition["hard_competitor"]
        control = by_condition["matched_control"]
        output.append({
            "tail_class": tail_class,
            "hard_g_local": hard["g_local"],
            "hard_g_post_marginal": hard["g_post_marginal"],
            "hard_corrected_retention_ratio": hard["corrected_retention_ratio"],
            "control_g_local": control["g_local"],
            "control_g_post_marginal": control["g_post_marginal"],
            "control_corrected_retention_ratio": control["corrected_retention_ratio"],
            "hard_minus_control_corrected_retention_ratio": (
                hard["corrected_retention_ratio"]
                - control["corrected_retention_ratio"]
            ),
        })
    return output

