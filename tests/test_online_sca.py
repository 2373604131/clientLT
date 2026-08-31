import csv

import torch

from utils.class_residual import ClassResidualHead, mask_class_residual_gradients
from utils.class_separable_aggregation import (
    D4ATracker,
    aggregate_class_residual_fedavg_rows,
    aggregate_class_residual_rows,
    class_supporters,
)


def _states():
    previous = {
        "class_residual.weight": torch.tensor([[9.0, 9.0], [7.0, 7.0]]),
        "class_residual.bias": torch.tensor([9.0, 7.0]),
        "frozen": torch.tensor([5.0]),
    }
    shared = {key: value.clone() for key, value in previous.items()}
    local = {
        0: {
            "class_residual.weight": torch.tensor([[1.0, 3.0], [100.0, 100.0]]),
            "class_residual.bias": torch.tensor([2.0, 100.0]),
        },
        1: {
            "class_residual.weight": torch.tensor([[5.0, 7.0], [200.0, 200.0]]),
            "class_residual.bias": torch.tensor([6.0, 200.0]),
        },
    }
    counts = {0: torch.tensor([2, 0]), 1: torch.tensor([6, 0])}
    return previous, shared, local, counts


def test_class_count_aggregation_and_unsupported_persistence():
    previous, shared, local, counts = _states()
    result, rows = aggregate_class_residual_rows(
        shared, previous, local, [0, 1], counts, [0, 1]
    )
    assert torch.allclose(result["class_residual.weight"][0], torch.tensor([4.0, 6.0]))
    assert torch.allclose(result["class_residual.bias"][0], torch.tensor(5.0))
    assert torch.equal(result["class_residual.weight"][1], previous["class_residual.weight"][1])
    assert rows[0]["supporter_count"] == 2
    assert rows[1]["retained_previous_row"] is True
    assert torch.equal(result["frozen"], previous["frozen"])


def test_residual_fedavg_uses_all_selected_clients_with_sample_weights():
    previous, shared, local, counts = _states()
    result, rows = aggregate_class_residual_fedavg_rows(
        shared,
        previous,
        local,
        [0, 1],
        counts,
        [0, 1],
        client_weights={0: 0.25, 1: 0.75},
    )
    assert torch.allclose(
        result["class_residual.weight"][0], torch.tensor([4.0, 6.0])
    )
    # Neither client supports class 1, but ordinary FedAvg still aggregates
    # their complete local copies instead of invoking SCA persistence.
    assert torch.allclose(
        result["class_residual.weight"][1], torch.tensor([175.0, 175.0])
    )
    assert torch.allclose(result["class_residual.bias"][1], torch.tensor(175.0))
    assert rows[1]["supporter_count"] == 0
    assert rows[1]["retained_previous_row"] is False


def test_support_threshold_is_training_metadata_only():
    counts = {0: torch.tensor([2, 8]), 1: torch.tensor([6, 4])}
    assert class_supporters([0, 1], counts, 0, min_fraction=0.0) == [0, 1]
    assert class_supporters([0, 1], counts, 0, min_fraction=0.5) == [1]


def test_residual_head_is_zero_initialized_and_tail_gated():
    head = ClassResidualHead(3, 2, scale=1.0, clamp=0.0, use_bias=True)
    features = torch.tensor([[1.0, 2.0]])
    head.set_active_classes([2])
    assert torch.equal(head(features), torch.zeros(1, 3))
    with torch.no_grad():
        head.weight.fill_(1.0)
        head.bias.fill_(1.0)
    output = head(features)
    assert torch.equal(output[:, :2], torch.zeros(1, 2))
    assert torch.allclose(output[:, 2], torch.tensor([4.0]))


def test_positive_row_gradient_mask():
    class Holder(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.class_residual = ClassResidualHead(4, 2)

    model = Holder()
    model.class_residual.set_active_classes([2, 3])
    model.class_residual.weight.grad = torch.ones_like(model.class_residual.weight)
    model.class_residual.bias.grad = torch.ones_like(model.class_residual.bias)
    mask_class_residual_gradients(model, torch.tensor([0, 2, 2]))
    assert torch.equal(
        model.class_residual.weight.grad.sum(dim=1), torch.tensor([0.0, 0.0, 2.0, 0.0])
    )
    assert torch.equal(model.class_residual.bias.grad, torch.tensor([0.0, 0.0, 1.0, 0.0]))


def test_d4a_absence_streak_and_diagnostic_only_flag(tmp_path):
    tracker = D4ATracker(tmp_path, [2])
    aggregation = [{
        "class_id": 2,
        "supporter_count": 0,
        "supporter_ids": "",
        "support_mass": 0,
        "retained_previous_row": True,
        "row_delta_norm": 0.0,
    }]
    tracker.record(0, aggregation, {2: {"accuracy": 20.0, "margin": -1.0}})
    tracker.record(1, aggregation, {2: {"accuracy": 15.0, "margin": -2.5}})
    with (tmp_path / "d4a" / "d4a_per_class_round.csv").open(
        newline="", encoding="utf-8"
    ) as handle:
        rows = list(csv.DictReader(handle))
    assert [int(row["absence_streak"]) for row in rows] == [1, 2]
    assert float(rows[-1]["accuracy_drop_from_historical_best"]) == 5.0
    assert float(rows[-1]["margin_drop_from_historical_best"]) == 1.5
    assert rows[-1]["used_by_aggregation"] == "False"
