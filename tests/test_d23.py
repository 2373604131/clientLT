import json
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from scripts.analyze_d2_conflict import (
    binary_auc,
    compute_geometry,
    safe_cosine,
    sign_disagreement,
    spearman,
)
from scripts.analyze_d3_boundary import (
    apply_logit_adjustment,
    centroid_logits,
    fit_balanced_ridge,
    nearest_centroid_weights,
    ridge_logits,
    select_tau,
)
from scripts.run_d23 import build_dump_command, dump_complete, load_frozen
from scripts.run_g0_d1 import CONFIGS
from utils.cusp_minimal import make_flat_spec
from utils.d23_common import class_split, stratified_fit_calibration_split


def _dump_payload():
    before = {"block.q_lora_A": torch.zeros(2)}
    local = [
        {"block.q_lora_A": torch.tensor([1.0, 0.0])},
        {"block.q_lora_A": torch.tensor([0.0, 1.0])},
        {"block.q_lora_A": torch.tensor([-1.0, 0.0])},
    ]
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    after = {"block.q_lora_A": sum(float(w) * state["block.q_lora_A"] for w, state in zip(weights, local))}
    counts = torch.tensor([
        [10, 3, 3, 3, 2],
        [10, 3, 3, 3, 3],
        [10, 3, 3, 3, 0],
    ])
    return {
        "flatten_spec": make_flat_spec(before).as_dict(),
        "global_before_trainable": before,
        "global_after_fedavg_trainable": after,
        "local_trainable_states": local,
        "selected_client_ids": [0, 1, 2],
        "fedavg_weights": weights,
        "client_class_counts": counts,
        "global_class_counts": counts.sum(0),
    }


def test_d2_geometry_uses_leave_one_client_out_support_reference():
    rows, audit = compute_geometry(_dump_payload(), {"communication_round": 20})
    by_client = {row["client_id"]: row for row in rows}

    assert audit["fedavg_reconstruction_relative_error"] < 1e-6
    # Client 0 sees client 1's [0, 1] direction, not a direction containing itself.
    assert by_client[0]["peer_support_count"] == 1
    assert by_client[0]["cosine_to_support_direction"] == pytest.approx(0.0)
    # A non-supporter sees both supporters as its independent reference.
    assert by_client[2]["peer_support_count"] == 2


def test_d2_statistics_have_expected_direction():
    assert safe_cosine(torch.tensor([1.0, 0.0]), torch.tensor([-1.0, 0.0])) == -1.0
    assert sign_disagreement(torch.tensor([1.0, -2.0]), torch.tensor([-1.0, -3.0])) == 0.5
    assert spearman([1, 2, 3], [10, 20, 30]) == pytest.approx(1.0)
    assert binary_auc([False, True], [0.0, 1.0]) == pytest.approx(1.0)


def test_d3_train_split_preserves_every_class_on_both_sides():
    labels = torch.tensor([0] * 5 + [1] * 7 + [2] * 9)
    fit, calibration = stratified_fit_calibration_split(labels, seed=42)

    assert set(labels[fit].tolist()) == {0, 1, 2}
    assert set(labels[calibration].tolist()) == {0, 1, 2}
    assert not set(fit.tolist()).intersection(calibration.tolist())


def test_d3_centroid_and_balanced_ridge_recover_separable_features():
    features = torch.tensor([[2.0, 0.0], [1.0, 0.0], [0.0, 2.0], [0.0, 1.0]])
    labels = torch.tensor([0, 0, 1, 1])
    centroids = nearest_centroid_weights(features, labels, 2)
    ridge = fit_balanced_ridge(features, labels, 2, ridge=1e-2)

    assert torch.equal(centroid_logits(features, centroids).argmax(1), labels)
    assert torch.equal(ridge_logits(features, ridge).argmax(1), labels)


def test_d3_logit_adjustment_and_tau_selection_use_calibration_only():
    logits = torch.tensor([[2.0, 1.0], [2.0, 1.5], [1.0, 2.0], [1.0, 2.0]])
    labels = torch.tensor([0, 1, 1, 1])
    priors = torch.tensor([0.9, 0.1])
    adjusted = apply_logit_adjustment(logits, priors, 1.0)
    selected, rows = select_tau(logits, labels, priors, [0.0, 1.0], [0], [1])

    assert adjusted[:, 1].sub(logits[:, 1]).mean() > adjusted[:, 0].sub(logits[:, 0]).mean()
    assert selected in {0.0, 1.0}
    assert len(rows) == 2


def test_d23_dump_command_is_one_seed42_candidate_r4_trajectory(tmp_path):
    freeze_path = tmp_path / "freeze.json"
    frozen = {
        "verdict": "PASS",
        "selected_config_id": "candidate_r4",
        "selected_config": CONFIGS["candidate_r4"],
    }
    freeze_path.write_text(json.dumps(frozen), encoding="utf-8")
    loaded = load_frozen(freeze_path)
    args = SimpleNamespace(
        output_root=tmp_path / "d23",
        data_root=Path("DATA"),
        python_bin="python",
        lr=0.001,
        test_batch_size=128,
        num_workers=4,
    )
    root, command = build_dump_command(args, loaded)

    assert command[command.index("--round") + 1] == "80"
    assert command[command.index("--v0_dump_rounds") + 1] == "20,50,80"
    assert command[command.index("--seed") + 1] == "42"
    assert command[command.index("--partition") + 1] == "client-longtail"
    assert command[command.index("--cliplora_rank") + 1] == "4"
    assert root.name == "dump_seed42"
    assert not dump_complete(root)


def test_class_split_has_only_eighty_head_and_twenty_tail_classes():
    head, tail = class_split(torch.arange(100, 0, -1))
    assert len(head) == 80
    assert len(tail) == 20
    assert set(head).isdisjoint(tail)
