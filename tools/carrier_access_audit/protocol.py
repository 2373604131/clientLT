from __future__ import annotations

import copy
import json
from pathlib import Path

from tools.semantic_acquisition.common import stable_hash, write_json


TAIL_CLASSES = list(range(80, 100))
NON_TAIL_CLASSES = list(range(80))


PROTOCOL = {
    "protocol_name": "CARRIER_ACCESS_MECHANISM_AUDIT_V1",
    "scope": "single_seed_client_local_mechanism_validation",
    "data_seed": 42,
    "claim_boundary": (
        "A is a descriptive natural-topology reanalysis. B estimates a controlled candidate-update "
        "transfer matrix. C isolates same-client co-adaptation and private-evidence gating under the "
        "frozen vision-LoRA substrate. None of the three experiments alone proves downstream SOTA."
    ),
    "dataset": {
        "name": "CIFAR-100-LT",
        "imbalance_factor": 0.01,
        "tail_classes": TAIL_CLASSES,
        "tail_definition": "index_defined_last_20_classes",
        "tail_sample_count": 153,
        "non_tail_classes": NON_TAIL_CLASSES,
    },
    "model": {
        "name": "FedAvg-VisualLoRA local substrate",
        "backbone": "ViT-B/16",
        "trainable_scope": "vision_lora_only",
        "lora_position": "top3",
        "lora_rank": 2,
        "lora_alpha": 1,
        "lora_parameters": ["q", "v"],
        "text_prototypes": "frozen",
        "precision": "fp32",
        "shared_anchor": "theta0_seed42",
    },
    "experiment_a": {
        "name": "natural_carrier_functional_footprint",
        "training_required": False,
        "input": "completed_E2A_local_outputs",
        "topologies": ["dirichlet", "clientlt"],
        "unit": "tail_class",
        "carrier_weighting": "tail_sample_mass_within_tail_class",
        "metrics": [
            "positive_all_class_margin_coverage",
            "normalized_positive_gain_entropy",
            "worst_neighbor_margin_gain",
            "cross_carrier_functional_cosine_diversity",
            "effective_carrier_count",
        ],
        "interpretation": "descriptive_total_topology_effect_not_single_factor_causality",
    },
    "experiment_b": {
        "name": "tail_by_candidate_functional_transfer_matrix",
        "candidate_classes": NON_TAIL_CLASSES,
        "candidate_train_samples_per_class": 12,
        "local_epochs": 3,
        "batch_size": 32,
        "optimizer_steps_per_candidate": 3,
        "candidate_sampling": "deterministic_without_replacement_from_exact_LT_pool",
        "semantic_prior": "frozen_CLIP_text_cosine_similarity",
        "private_evidence": "target_tail_train_images_only",
        "independent_endpoint": "target_tail_test_images",
        "primary_effect": "test_tail_margin_gain",
        "secondary_effects": [
            "test_tail_nll_gain",
            "test_worst_neighbor_margin_gain",
            "private_tail_margin_gain",
        ],
        "preregistered_related_group": "semantic_ranks_1_to_10",
        "preregistered_unrelated_group": "semantic_ranks_71_to_80",
        "analyses": [
            "positive_donor_rate",
            "effect_distribution",
            "best_attainable_gain_by_candidate_budget",
            "spearman_semantic_similarity_vs_functional_effect",
            "private_evidence_selection_generalization",
        ],
        "summary_rule": {
            "minimum_tail_classes_in_expected_direction": 12,
            "semantic_enrichment_requires_two_of_three": [
                "higher_related_positive_donor_rate",
                "higher_related_mean_test_margin_gain",
                "positive_semantic_effect_spearman",
            ],
        },
        "interpretation": (
            "Semantic relatedness is an enrichment prior, not a guarantee that every related class is helpful."
        ),
    },
    "experiment_c": {
        "name": "joint_separate_private_readapt",
        "tail_train_samples_per_class": 5,
        "candidate_train_samples_per_class": 12,
        "local_epochs": 3,
        "conditions": [
            "tail_only",
            "joint_related",
            "separate_merge_related",
            "separate_readapt_related",
            "joint_unrelated",
        ],
        "related_candidate_selection": (
            "best private-tail margin gain among semantic top10 from B; never uses tail test metrics"
        ),
        "unrelated_candidate_selection": (
            "best private-tail margin gain among semantic bottom10 from B; never uses tail test metrics"
        ),
        "separate_merge": "theta0 + 0.5*tail_delta + 0.5*candidate_delta",
        "private_readapt": {
            "state": "theta0 + 0.5*tail_delta + 0.5*lambda*candidate_delta",
            "lambda_grid": [0.0, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5],
            "selection": "maximize private tail-train margin; ties choose smaller lambda",
            "test_labels_used_for_selection": False,
        },
        "joint_objective": (
            "one shared optimizer step per epoch after accumulating 0.5*tail_CE and 0.5*candidate_CE"
        ),
        "fairness": {
            "same_related_image_ids_across_joint_separate_readapt": True,
            "same_tail_image_ids_across_all_conditions": True,
            "same_per_sample_augmentation_seed": True,
            "equal_tail_and_candidate_gradient_calls": True,
            "optimizer_trajectory_is_treatment": True,
        },
        "primary_endpoint": "test_tail_margin_gain",
        "contrasts": [
            "joint_related_minus_separate_merge_related",
            "separate_readapt_related_minus_separate_merge_related",
            "joint_related_minus_joint_unrelated",
        ],
        "contrast_support_rule": {
            "mean_primary_effect_positive": True,
            "minimum_tail_classes_positive": 12,
        },
    },
}


def frozen_protocol() -> dict:
    value = copy.deepcopy(PROTOCOL)
    value["protocol_hash"] = stable_hash(value)
    return value


def write_protocol(output_dir: Path) -> Path:
    path = Path(output_dir) / "carrier_access_protocol.json"
    value = frozen_protocol()
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise RuntimeError(f"Refusing to overwrite a different protocol: {path}")
    if not path.exists():
        write_json(path, value)
    return path
