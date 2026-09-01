from __future__ import annotations

import copy
import json
from pathlib import Path

from tools.semantic_acquisition.common import stable_hash, write_json


PROTOCOL = {
    "protocol_name": "CLIENT_TOPOLOGY_FUNCTIONAL_BREADTH_V1",
    "scope": "seed42_client_level_mechanism_audit",
    "claim_boundary": (
        "Phase 2 tests whether the fixed-margin Client-LT coupling narrows the functional "
        "breadth of real client updates under full availability and a frozen frac=0.4 "
        "schedule. It is a simulator-side mechanism audit, not a deployable FL method, "
        "and a single seed does not establish generalization."
    ),
    "independence_from_phase1": (
        "Phase 2 does not use or select the Carrier-B Broad/Narrow pairs and may run before "
        "Phase 1 finishes. Phase 3 remains blocked until Phase 1 produces matched pairs."
    ),
    "dataset": {
        "name": "CIFAR-100-LT", "imbalance_factor": 0.01,
        "tail_classes": list(range(80, 100)), "num_clients": 30,
        "split_seed": 42,
    },
    "topologies": {
        "clientlt": {
            "partition": "client-longtail", "head_client_ratio": 0.9,
            "tail_client_ratio": 0.1, "head_class_ratio": 0.8,
            "tail_class_ratio": 0.2, "specialization_lambda": 0.75,
            "intra_group_alpha": 0.5, "head_leakage_scale": 3.0,
        },
        "matched_dirichlet": {
            "partition": "matched-dirichlet", "beta": 0.5,
            "row_margins": "exactly_equal_to_clientlt_nk",
            "column_margins": "exactly_equal_to_clientlt_nc",
        },
    },
    "model": {
        "name": "Carrier-B-compatible FedAvg VisualLoRA local substrate",
        "backbone": "ViT-B/16", "trainable_scope": "vision_lora_only",
        "lora_position": "top3", "lora_rank": 2, "lora_alpha": 1,
        "lora_parameters": ["q", "v"], "precision": "fp32",
        "common_anchor": "theta0_seed42",
    },
    "local_updates": {
        "clients_per_topology": 30, "local_epochs": 3, "batch_size": 32,
        "learning_rate": 0.001, "optimizer_reinitialized_per_client": True,
        "common_anchor": True, "server_aggregation_called": False,
        "all_clients_are_trained": True,
    },
    "functional_evidence": {
        "split": "CIFAR-100 train only",
        "samples_per_tail_class": 10,
        "sampling": "held out from the complete LT federated training pool",
        "hard_boundaries": "frozen semantic top-10 non-tail neighbors",
        "test_split_accessed": False,
        "boundary_gain": "mean[(z_tail-z_neighbor)_updated-(z_tail-z_neighbor)_theta0]",
    },
    "pools": {
        "evidence_supporters": {
            "primary": True,
            "clients": "available clients with N_kc > 0",
            "merge_weight": "class count N_kc normalized within available supporters",
        },
        "all_clients": {
            "primary": False,
            "clients": "all available clients including class-absent functional donors",
            "merge_weight": "client sample count normalized within available clients",
        },
    },
    "A1_spatial": {
        "participation": "all 30 clients available",
        "metrics": [
            "positive_donor_count", "mean_positive_donors_per_boundary",
            "potential_effective_breadth", "actual_effective_breadth",
            "actual_positive_boundary_count", "actual_worst_boundary_gain",
            "actual_negative_boundary_harm",
        ],
    },
    "A2_temporal": {
        "frac": 0.4, "rounds": 80, "clients_per_round": 12,
        "schedule": "actual common seed42 SCA factorial schedule",
        "low_breadth_definition": "actual breadth below 50% of same-topology A1 breadth",
        "metrics": [
            "breadth_auc", "low_breadth_round_fraction", "no_support_round_fraction",
            "breadth_cv", "maximum_absence_streak", "early_middle_late_breadth",
        ],
    },
    "support_rule": {
        "unit": "20 paired tail classes; descriptive seed42",
        "minimum_tail_classes_in_expected_direction": 12,
        "spatial_expected_direction": "matched_dirichlet_minus_clientlt_positive",
        "temporal_expected_direction": "matched_dirichlet_minus_clientlt_breadth_auc_positive",
        "verdicts": ["BOTH", "SPATIAL_ONLY", "TEMPORAL_ONLY", "NO_CONSISTENT_GAP"],
    },
    "federated_deployment": {
        "server_deployable_method": False, "privacy_claim": False,
        "reason": "Offline simulator audit evaluates client states on centrally held train-only probes.",
    },
}


def frozen_protocol() -> dict:
    value = copy.deepcopy(PROTOCOL)
    value["protocol_hash"] = stable_hash(value)
    return value


def write_protocol(output_dir: Path) -> Path:
    path = Path(output_dir) / "topology_breadth_protocol.json"
    value = frozen_protocol()
    if path.exists() and json.loads(path.read_text(encoding="utf-8")) != value:
        raise RuntimeError(f"Refusing to overwrite a different Phase-2 protocol: {path}")
    if not path.exists():
        write_json(path, value)
    return path

