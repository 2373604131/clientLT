from __future__ import annotations

import copy
import json
from pathlib import Path

from tools.carrier_access_audit.protocol import frozen_protocol as carrier_protocol
from tools.carrier_access_audit.rewrite_protocol import frozen_rewrite_protocol
from tools.semantic_acquisition.common import stable_hash, write_json


PROTOCOL = {
    "protocol_name": "FUNCTIONAL_BREADTH_FEASIBILITY_V1",
    "scope": "seed42_private_train_only_no_training_feasibility",
    "claim_boundary": (
        "This phase asks whether already-saved Carrier-B updates can form matched "
        "broad/narrow functional-coverage pairs. It is not a federated performance "
        "experiment and cannot establish downstream accuracy or retention gains."
    ),
    "federated_deployment": {
        "server_deployable_method": False,
        "privacy_claim": False,
        "reason": (
            "This is an offline simulator-side mechanism audit with centralized access "
            "to private-train evidence; its selection rule must not be presented as a "
            "deployable FL server algorithm."
        ),
    },
    "parent_carrier_protocol_hash": carrier_protocol()["protocol_hash"],
    "parent_rewrite_protocol_hash": frozen_rewrite_protocol()["protocol_hash"],
    "training": {
        "allowed": False,
        "candidate_source": "completed_experiment_b_candidate_states_only",
        "direct_tail_source": "completed_d1_tail_writer_states_only",
        "missing_state_policy": "fail_without_retraining",
        "gradient_or_optimizer_calls": 0,
    },
    "evidence": {
        "selection_split": "CIFAR-100-LT private train only",
        "test_split_access_allowed": False,
        "tail_samples": "five frozen Carrier-B private-tail samples per class",
        "head_safety_samples_per_class": 3,
        "head_safety_source": (
            "deterministic held-out examples from the exact LT train pool, excluding "
            "all Carrier-B candidate and private-tail manifest samples"
        ),
    },
    "boundaries": {
        "definition": "frozen semantic top-10 non-tail neighbors per tail class",
        "count_per_tail_class": 10,
        "gain": "mean_private[(z_tail-z_neighbor)_updated-(z_tail-z_neighbor)_theta0]",
        "positive_strength": "sum(max(boundary_gain,0))",
        "effective_breadth": "exp(entropy(normalized_positive_boundary_gains))",
        "negative_boundary_harm": "sum(max(-boundary_gain,0))",
    },
    "candidate_updates": {
        "classes": list(range(80)),
        "saved_tensor_artifact": "candidate_update_tensors.pt",
        "required_metadata": [
            "l2_norm", "optimizer_steps_successful", "scheduler_steps",
            "amp_overflow_count", "sample_count", "sha256",
        ],
        "direct_tail_cosine": True,
    },
    "merge": {
        "donor_count": 2,
        "rule": "theta0 + 0.5*delta_i + 0.5*delta_j",
        "pair_universe": "all unordered pairs of the 80 saved candidate updates",
        "shortlist_contrasts_per_tail": 8,
        "shortlist_selection": (
            "maximize predicted breadth separation after nearest-neighbor matching "
            "on strength, update norm, head safety, and direct-tail cosine"
        ),
        "actual_evaluation": "forward pass of shortlisted merged tensors on private train only",
    },
    "matching": {
        "exact": ["donor_count", "candidate_sample_count", "optimizer_steps"],
        "actual_symmetric_relative_strength_gap_max": 0.20,
        "relative_update_norm_gap_max": 0.10,
        "absolute_head_margin_gain_gap_max": 0.10,
        "absolute_direct_tail_cosine_gap_max": 0.15,
        "minimum_actual_effective_breadth_gap": 1.0,
    },
    "feasibility_gate": {
        "minimum_tail_classes_with_matched_broad_narrow_pair": 12,
        "verdicts": ["FEASIBLE", "PARTIAL", "INFEASIBLE"],
        "partial_minimum": 1,
        "test_metrics_used_for_selection": False,
    },
}


def frozen_protocol() -> dict:
    value = copy.deepcopy(PROTOCOL)
    value["protocol_hash"] = stable_hash(value)
    return value


def write_protocol(output_dir: Path) -> Path:
    path = Path(output_dir) / "functional_breadth_protocol.json"
    value = frozen_protocol()
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise RuntimeError(f"Refusing to overwrite a different protocol: {path}")
    if not path.exists():
        write_json(path, value)
    return path
