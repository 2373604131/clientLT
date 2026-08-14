from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path

from tools.semantic_acquisition.common import stable_hash, write_json


# CIFAR-100-LT is constructed in class-index order: the exponential sample
# budget decreases from class 0 to class 99.  The frozen bottom-20 identity is
# therefore the index-defined range 80--99.  In particular, classes 79 and 80
# both contain 12 samples after integer truncation; recounting and sorting the
# realized histogram must not be allowed to exchange them at that boundary.
TAIL_CLASSES = list(range(80, 100))


MECHANISM_VALIDATION_PROTOCOL = {
    "protocol_name": "E1_STRONG_BUT_NARROW_MECHANISM_VALIDATION_V1",
    "scope": "mechanism_validation_only",
    "scope_exclusions": [
        "method_hyperparameter_tuning",
        "sota_comparison",
        "generalization_experiments",
        "ablation_studies",
        "sensitivity_analysis",
    ],
    "scope_note": (
        "These values are frozen only for the paired mechanism-validation "
        "experiment. Later experiments must declare their own protocols and "
        "must not inherit these as mandatory fixed hyperparameters."
    ),
    "dataset": {
        "name": "CIFAR-100-LT",
        "num_classes": 100,
        "imbalance_factor": 0.01,
        "imbalance_type": "exp",
        "tail_class_ratio": 0.2,
        "tail_classes": TAIL_CLASSES,
        "tail_definition": "index_defined_last_20_classes",
        "tail_boundary_tie_policy": (
            "class_identity follows the LT generator order; equal realized "
            "counts at classes 79 and 80 do not alter membership"
        ),
        "tail_class_count": 20,
        "tail_sample_count": 153,
        "global_pool_requirement": "identical_sample_ids_across_topologies",
    },
    "paired_topologies": {
        "dirichlet": {
            "partition": "noniid-labeldir-fine",
            "beta": 0.5,
        },
        "clientlt_controlled": {
            "partition": "client-longtail-controlled",
            "num_clients": 30,
            "tail_client_ids": [27, 28, 29],
            "tail_client_min_purity": 0.8,
            "tail_leakage_to_ordinary_clients": 0,
            "tail_samples_in_tail_clients": 153,
            "max_companion_samples_in_tail_clients": 38,
            "intra_group_alpha": 0.5,
        },
    },
    "fairness": {
        "data_seeds": [42, 2026, 3407],
        "model_initialization_equal": True,
        "client_participation_schedule_equal": True,
        "optimizer_steps_equal": True,
        "augmentation_schedule_equal": True,
        "global_training_multiset_equal": True,
    },
    "training": {
        "trainer": "ClipLora",
        "model": "fedavg",
        "num_clients": 30,
        "client_fraction": 1.0,
        "communication_rounds": 100,
        "local_epochs": 3,
        "train_batch_size": 32,
        "test_batch_size": 64,
        "optimizer": "sgd",
        "learning_rate": 0.001,
        "lr_policy": "constant",
        "precision": "amp",
        "lora": {
            "backbone": "ViT-B/16",
            "encoder": "vision",
            "position": "top3",
            "rank": 2,
            "alpha": 1,
            "dropout": 0.0,
            "parameters": ["q", "v"],
        },
    },
    "breadth_audit": {
        "tail_test_samples_only": True,
        "evaluate_every_round": True,
        "visual_subgroups": {
            "encoder": "dinov2_vitb14",
            "encoder_trainable": False,
            "clusters_per_tail_class": 4,
            "clustering": "per_tail_class_kmeans",
            "kmeans_seed": 20260813,
            "kmeans_n_init": 20,
            "recognized_cluster_accuracy_threshold": 0.5,
            "primary_metrics": [
                "worst_cluster_accuracy",
                "cluster_balanced_accuracy",
                "recognized_cluster_fraction_at_50",
            ],
            "reported_secondary_metrics": ["cluster_accuracy_std"],
        },
        "multi_view": {
            "views": [
                "clean", "crop", "color_jitter", "blur", "occlusion", "resize"
            ],
            "view_parameters": {
                "crop": "center_28_of_32_then_resize_32",
                "color_jitter": "brightness_0.8_contrast_1.2_saturation_0.8",
                "blur": "gaussian_radius_1.0",
                "occlusion": "center_8x8_rgb_128",
                "resize": "downsample_20_then_upsample_32",
            },
            "primary_metrics": [
                "worst_view_accuracy",
                "prediction_consistency",
                "worst_view_margin",
                "clean_to_corruption_accuracy_drop",
            ],
        },
        "neighbor_discrimination": {
            "neighbor_source": "frozen_V1_CLIP_text_top10_non_tail_only",
            "neighbors_per_tail_class": 10,
            "neighbors_fixed_across_all_rounds_topologies_and_seeds": True,
            "primary_metrics": [
                "target_vs_neighbor_pairwise_margin",
                "worst_neighbor_margin",
                "positive_margin_neighbor_coverage",
            ],
            "reported_secondary_metrics": ["neighbor_margin_variance"],
        },
        "claim_rule": {
            "required_supporting_metric_families": 2,
            "total_metric_families": 3,
            "no_posthoc_metric_family_selection": True,
            "family_rule": (
                "A family supports semantic narrowing only when all of its "
                "directional primary endpoints favor Dirichlet after paired "
                "class/seed analysis; secondary dispersion metrics are always "
                "reported but cannot independently pass a family."
            ),
        },
    },
}


def frozen_protocol() -> dict:
    value = copy.deepcopy(MECHANISM_VALIDATION_PROTOCOL)
    value["protocol_hash"] = stable_hash(value)
    return value


def write_frozen_protocol(output_dir: Path) -> Path:
    output_dir = Path(output_dir)
    path = output_dir / "mechanism_validation_protocol.json"
    value = frozen_protocol()
    if path.exists():
        existing = json.loads(path.read_text(encoding="utf-8"))
        if existing != value:
            raise RuntimeError(
                f"Refusing to overwrite a different frozen mechanism protocol: {path}"
            )
        return path
    write_json(path, value)
    return path


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Write the E1 mechanism-validation-only frozen protocol."
    )
    parser.add_argument(
        "--output-dir", type=Path,
        default=Path("output/e1_strength_breadth/protocol"),
    )
    args = parser.parse_args()
    path = write_frozen_protocol(args.output_dir)
    print(json.dumps({"protocol": str(path.resolve()), "scope": "mechanism_validation_only"}))


if __name__ == "__main__":
    main()
