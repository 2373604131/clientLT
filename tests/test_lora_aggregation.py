import csv

import pytest
import torch

from utils.lora_aggregation import (
    aggregate_lora_state,
    append_lora_aggregation_diagnostics,
    compute_lora_aggregation_weights,
    sample_weighted_client_weights,
    support_normalized_client_weights,
)


def test_sample_weighted_client_weights_match_fedavg():
    weights = sample_weighted_client_weights([0, 2], [3, 100, 1])

    assert weights == pytest.approx({0: 0.75, 2: 0.25})


def test_support_normalized_weights_average_classwise_distributions():
    # Tail class 2: support {0, 1}, weights {2/3, 1/3}.
    # Tail class 3: support {1, 2}, weights {3/4, 1/4}.
    counts = {
        0: torch.tensor([4, 2, 1, 0]),
        1: torch.tensor([1, 1, 1, 1]),
        2: torch.tensor([0, 0, 0, 1]),
    }
    weights, details = support_normalized_client_weights(
        [0, 1, 2],
        [6, 3, 1],
        counts,
        [2, 3],
    )

    assert weights == pytest.approx({
        0: 1 / 3,
        1: (1 / 3 + 3 / 4) / 2,
        2: 1 / 8,
    })
    assert sum(weights.values()) == pytest.approx(1.0)
    assert details["covered_tail_classes"] == [2, 3]
    assert details["client_supported_tail_classes"] == {0: 1, 1: 2, 2: 1}


def test_support_normalized_skips_uncovered_classes_under_partial_participation():
    counts = {
        0: torch.tensor([0, 1, 0]),
        1: torch.tensor([0, 0, 1]),
    }
    weights, details = support_normalized_client_weights(
        [0],
        [5, 5],
        counts,
        [1, 2],
    )

    assert weights == {0: 1.0}
    assert details["covered_tail_classes"] == [1]
    assert details["uncovered_tail_classes"] == [2]


def test_support_normalized_reduces_to_fedavg_when_every_client_supports_every_tail_class():
    counts = {
        0: torch.tensor([0, 1, 1]),
        1: torch.tensor([0, 2, 3]),
    }
    support_weights, _ = support_normalized_client_weights(
        [0, 1],
        [2, 8],
        counts,
        [1, 2],
    )

    assert support_weights == pytest.approx(
        sample_weighted_client_weights([0, 1], [2, 8])
    )


def test_named_weight_resolver_requires_support_metadata():
    with pytest.raises(ValueError, match="client_class_counts"):
        compute_lora_aggregation_weights(
            "support_normalized",
            [0],
            [1],
            tail_class_ids=[0],
        )


def test_aggregate_lora_state_preserves_frozen_clip_and_uses_float32_accumulator():
    global_state = {
        "image_encoder.weight": torch.tensor([7.0], dtype=torch.float16),
        "image_encoder.attn.q_lora_A": torch.tensor([0.0], dtype=torch.float16),
    }
    local_states = [
        {
            "image_encoder.weight": torch.tensor([100.0], dtype=torch.float16),
            "image_encoder.attn.q_lora_A": torch.tensor([1.0], dtype=torch.float16),
        },
        {
            "image_encoder.weight": torch.tensor([-100.0], dtype=torch.float16),
            "image_encoder.attn.q_lora_A": torch.tensor([3.0], dtype=torch.float16),
        },
    ]

    aggregated = aggregate_lora_state(
        global_state,
        local_states,
        [0, 1],
        ["image_encoder.attn.q_lora_A"],
        {0: 0.25, 1: 0.75},
    )

    assert torch.equal(aggregated["image_encoder.weight"], global_state["image_encoder.weight"])
    assert aggregated["image_encoder.attn.q_lora_A"].item() == pytest.approx(2.5)
    assert aggregated["image_encoder.attn.q_lora_A"].dtype == torch.float16


def test_aggregation_diagnostics_are_auditable(tmp_path):
    details = {
        "tail_class_count": 2,
        "covered_tail_class_count": 2,
        "uncovered_tail_classes": [],
        "client_supported_tail_classes": {0: 1, 1: 2},
    }
    append_lora_aggregation_diagnostics(
        tmp_path,
        epoch=0,
        partition="client-longtail",
        mode="support_normalized",
        selected_clients=[0, 1],
        datanumber_client=[10, 30],
        client_weights={0: 0.4, 1: 0.6},
        details=details,
    )

    with (tmp_path / "lora_aggregation_weights.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    with (tmp_path / "lora_aggregation_summary.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        summary = list(csv.DictReader(handle))

    assert [float(row["aggregation_weight"]) for row in rows] == [0.4, 0.6]
    assert summary[0]["weight_sum"] == "1.0"
    assert summary[0]["covered_tail_class_count"] == "2"
