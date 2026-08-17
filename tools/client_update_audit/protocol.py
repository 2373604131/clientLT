from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from tools.semantic_acquisition.common import stable_hash, write_json


TAIL_CLASSES = list(range(80, 100))
TAIL_CLIENT_IDS = [27, 28, 29]


E2_PROTOCOL = {
    "protocol_name": "E2_CLIENT_LOCAL_FUNCTIONAL_FOOTPRINT_V1",
    "scope": "client_level_mechanism_validation_only",
    "claim_boundary": (
        "E2A is a pre-aggregation local-update audit and establishes association. "
        "Only the paired E2B companion intervention may support a causal semantic-access claim."
    ),
    "dataset": {
        "name": "CIFAR-100-LT",
        "imbalance_factor": 0.01,
        "imbalance_type": "exp",
        "num_classes": 100,
        "tail_classes": TAIL_CLASSES,
        "tail_definition": "index_defined_last_20_classes",
        "tail_sample_count": 153,
    },
    "model": {
        "name": "FedAvg-VisualLoRA local substrate",
        "backbone": "ViT-B/16",
        "trainable_scope": "vision_lora_only",
        "lora_position": "top3",
        "lora_rank": 2,
        "lora_alpha": 1,
        "lora_dropout": 0.0,
        "lora_parameters": ["q", "v"],
        "text_prototypes": "frozen",
        "precision": "fp32",
    },
    "local_training": {
        "common_anchor": "shared_theta0",
        "local_epochs": 3,
        "batch_size": 32,
        "optimizer": "sgd",
        "learning_rate": 0.001,
        "weight_decay": 0.0005,
        "momentum": 0.9,
        "lr_policy": "constant",
        "optimizer_reinitialized_per_client": True,
        "server_aggregation_permitted": False,
        "evaluation_epochs": [0, 1, 2, 3],
    },
    "e2a": {
        "name": "natural_partition_local_footprint_audit",
        "topologies": {
            "dirichlet": {"partition": "noniid-labeldir-fine", "beta": 0.5},
            "clientlt": {
                "partition": "client-longtail-controlled",
                "tail_client_ids": TAIL_CLIENT_IDS,
                "tail_leakage": 0,
                "tail_client_min_purity": 0.8,
                "max_companion_samples": 38,
            },
        },
        "trained_clients": "all_clients_with_at_least_one_tail_training_sample",
        "primary_analysis_unit": "tail_class_with_tail_sample_mass_weighting_over_clients",
        "interpretation": "descriptive_preaggregation_association_not_causal",
    },
    "e2b": {
        "name": "paired_clientlt_semantic_access_intervention",
        "conditions": ["narrow_related", "broad_related", "broad_unrelated"],
        "trained_clients": TAIL_CLIENT_IDS,
        "narrow_classes_per_tail_client": 2,
        "broad_classes_per_tail_client": 8,
        "broad_minimum_coarse_superclass_coverage": 6,
        "companion_count_policy": "exactly_preserve_each_frozen_clientlt_tail_client",
        "global_pool_policy": "one_to_one_swap_preserves_every_sample_and_every_client_size",
        "relatedness_definition": "tail_mass_weighted_shared_CIFAR100_coarse_superclass",
        "selection_evaluation_independence": (
            "Companions are selected with the dataset taxonomy; the primary endpoint uses "
            "the separately frozen CLIP-text neighbor table."
        ),
        "relatedness_match_tolerance": 0.03,
        "initial_companion_difficulty_control": {
            "metric": "theta0_cross_entropy_per_sample",
            "maximum_absolute_standardized_mean_difference": 0.25,
            "policy": "audit_before_any_local_training_and_flag_as_confound_if_violated",
        },
        "primary_endpoint": "accuracy_matched_worst_neighbor_margin_gain",
        "secondary_endpoints": [
            "final_worst_neighbor_margin_gain",
            "positive_margin_neighbor_coverage_gain",
            "all_class_positive_margin_gain_fraction",
        ],
        "confirmation_rule": {
            "broad_related_minus_narrow_mean_positive": True,
            "minimum_positive_tail_classes": 12,
            "class_cluster_bootstrap_ci_low_positive": True,
            "broad_related_minus_broad_unrelated_mean_positive": True,
            "accuracy_match_tolerance": 0.02,
        },
    },
    "semantic_neighbors": {
        "source": "checked_in_frozen_E1_top10_non_tail_table",
        "neighbors_per_tail_class": 10,
        "weight_by_rank": "1/rank",
    },
    "outputs": {
        "record_full_100_class_functional_footprint_at_epoch3": True,
        "record_tail_and_neighbor_metrics_each_epoch": True,
        "record_flattened_lora_update_vectors": True,
        "record_partition_and_execution_manifests": True,
    },
}


def frozen_protocol() -> dict:
    value = copy.deepcopy(E2_PROTOCOL)
    value["protocol_hash"] = stable_hash(value)
    return value


def write_protocol(output_dir: Path) -> Path:
    path = Path(output_dir) / "e2_client_update_protocol.json"
    value = frozen_protocol()
    if path.exists():
        current = json.loads(path.read_text(encoding="utf-8"))
        if current != value:
            raise RuntimeError(f"Refusing to overwrite a different E2 protocol: {path}")
    else:
        write_json(path, value)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", type=Path, default=Path("output/e2_client_update_audit/protocol"))
    args = parser.parse_args()
    path = write_protocol(args.output_dir)
    print(json.dumps({"protocol": str(path.resolve()), "scope": E2_PROTOCOL["scope"]}))


if __name__ == "__main__":
    main()
