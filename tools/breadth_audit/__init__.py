"""E1 semantic-breadth audit for the controlled Client-LT mechanism study.

The frozen protocol in this package applies only to the mechanism-validation
comparison.  It is deliberately isolated from later method tuning, SOTA
experiments, robustness sweeps, and ablations.
"""

from .metrics import (
    multiview_robustness_metrics,
    neighbor_discrimination_metrics,
    visual_subgroup_metrics,
)
from .artifacts import append_breadth_artifacts
from .comparison import preregistered_direction_gate
from .protocol import MECHANISM_VALIDATION_PROTOCOL, write_frozen_protocol

__all__ = [
    "MECHANISM_VALIDATION_PROTOCOL",
    "append_breadth_artifacts",
    "multiview_robustness_metrics",
    "neighbor_discrimination_metrics",
    "preregistered_direction_gate",
    "visual_subgroup_metrics",
    "write_frozen_protocol",
]
