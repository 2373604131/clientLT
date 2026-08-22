from __future__ import annotations

import copy
import json
from pathlib import Path

from tools.carrier_access_audit.protocol import TAIL_CLASSES, frozen_protocol as carrier_protocol
from tools.semantic_acquisition.common import stable_hash, write_json


REWRITE_PROTOCOL = {
    "protocol_name": "POST_WRITE_REWRITE_AND_RETENTION_V1",
    "scope": "single_seed_post_write_mechanism_validation",
    "data_seed": 42,
    "claim_boundary": (
        "D1 tests state-conditioned signed functional effects of class-absent candidate updates after "
        "tail knowledge has been written. D2 replays fixed saved updates and tests whether private "
        "rewrite risk predicts functional forgetting. It is not a dynamic client-retraining simulation."
    ),
    "parent_protocol_hash": carrier_protocol()["protocol_hash"],
    "tail_classes": TAIL_CLASSES,
    "private_split": {
        "source": "frozen_carrier_access_private_tail_samples",
        "samples_per_tail_class": 5,
        "write_slots": [0, 1, 2],
        "evidence_slots": [3, 4],
        "test_samples_per_tail_class": 100,
        "write_evidence_disjoint": True,
    },
    "tail_write": {
        "anchor": "shared_theta0_seed42",
        "local_epochs": 3,
        "optimizer_steps": 3,
        "trainable_scope": "vision_lora_only",
        "primary_validity_endpoint": "test_margin_gain",
        "minimum_valid_tail_classes": 12,
    },
    "candidate_updates": {
        "source": "completed_experiment_b_candidate_states",
        "candidate_classes": list(range(80)),
        "normalization": "common_median_l2_norm_across_80_candidate_deltas",
        "d1_alpha": 0.5,
        "reason": "remove the observed 8.23x candidate-update norm range and match C's merge dose",
    },
    "d1": {
        "name": "matched_pre_post_rewrite_matrix",
        "pairs": 1600,
        "primary_endpoint": "test_margin_effect",
        "secondary_endpoints": ["test_nll_effect", "test_worst_neighbor_margin_effect"],
        "private_endpoint": "two-sample_private_margin_effect",
        "transitions": [
            "donor_to_donor", "donor_to_rewriter",
            "rewriter_to_donor", "rewriter_to_rewriter",
        ],
        "private_detection_metrics": [
            "spearman", "sign_agreement", "donor_precision",
            "rewriter_recall", "false_safe_rate",
        ],
        "support_rules": {
            "minimum_tail_classes_with_both_post_donors_and_rewriters": 12,
            "minimum_tail_classes_with_donor_to_rewriter_transition": 12,
            "minimum_tail_classes_with_positive_private_test_spearman": 12,
            "mean_private_test_sign_agreement_above": 0.5,
        },
        "test_metrics_used_for_candidate_selection": False,
        "test_write_gain_used_only_for_preregistered_tail_eligibility": True,
    },
    "d2": {
        "name": "cumulative_fixed_update_replay",
        "requires_valid_d1": True,
        "sequence_lengths": [5, 10, 20],
        "per_update_beta": 0.05,
        "blind_draws": 5,
        "conditions": ["low_risk", "blind", "high_risk"],
        "selection": {
            "low_risk": "top-K by private post-write margin effect",
            "high_risk": "bottom-K by private post-write margin effect",
            "blind": "deterministic random subset without test access",
        },
        "predicted_risk": (
            "beta_over_d1_alpha times sum(max(-private_post_write_effect,0))"
        ),
        "primary_endpoint": "test_forgetting",
        "secondary_endpoint": "test_retention_of_write_gain",
        "risk_correlation": "blind_sequences_only_within_each_fixed_K_then_mean_across_K",
        "support_rules": {
            "minimum_tail_classes_positive_risk_forgetting_spearman": 12,
            "minimum_tail_classes_low_risk_better_than_blind": 12,
            "minimum_tail_classes_high_risk_worse_than_blind": 12,
        },
        "test_metrics_used_for_sequence_selection": False,
        "test_write_gain_used_only_for_preregistered_tail_eligibility": True,
    },
}


def frozen_rewrite_protocol() -> dict:
    value = copy.deepcopy(REWRITE_PROTOCOL)
    value["protocol_hash"] = stable_hash(value)
    return value


def write_rewrite_protocol(output_dir: Path) -> Path:
    path = Path(output_dir) / "post_write_rewrite_protocol.json"
    value = frozen_rewrite_protocol()
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise RuntimeError(f"Refusing to overwrite a different rewrite protocol: {path}")
    if not path.exists():
        write_json(path, value)
    return path
