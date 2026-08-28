from types import SimpleNamespace

import torch

from utils.cusp_minimal import make_flat_spec
from utils.v0_oracle import (
    class_groups_from_counts,
    gap_closure,
    metrics_from_logits,
    optimize_span_oracle,
    save_v0_round_dump,
    sphere_candidate_from_coordinates,
    support_normalized_deltas,
    weighted_disagreement_scale,
)
from scripts.run_v0_oracle import build_candidates_pooled, stratified_cap


def synthetic_payload():
    before = {"image_encoder.block.lora_A": torch.zeros(2)}
    local = [
        {"image_encoder.block.lora_A": torch.tensor([1.0, 0.0])},
        {"image_encoder.block.lora_A": torch.tensor([0.0, 1.0])},
        {"image_encoder.block.lora_A": torch.tensor([1.0, 1.0])},
    ]
    spec = make_flat_spec(before)
    weights = torch.tensor([0.2, 0.3, 0.5], dtype=torch.float64)
    after = {
        "image_encoder.block.lora_A": sum(
            float(weight) * state["image_encoder.block.lora_A"] for weight, state in zip(weights, local)
        )
    }
    return {
        "flatten_spec": spec.as_dict(),
        "global_before_trainable": before,
        "global_after_fedavg_trainable": after,
        "local_trainable_states": local,
        "fedavg_weights": weights,
        "client_class_counts": torch.tensor([[4, 0, 0], [0, 3, 0], [0, 1, 2]]),
        "global_class_counts": torch.tensor([4, 4, 2]),
        "selected_client_ids": [0, 1, 2],
        "num_classes": 3,
    }


def test_class_groups_are_disjoint_and_frequency_ranked():
    groups = class_groups_from_counts(torch.tensor([100, 80, 60, 40, 20, 10, 5, 3, 2, 1]))
    assert groups["head"] == [0, 1, 2, 3]
    assert groups["mid"] == [4, 5, 6, 7]
    assert groups["tail"] == [8, 9]
    assert set(groups["head"]).isdisjoint(groups["mid"])
    assert set(groups["mid"]).isdisjoint(groups["tail"])


def test_spherical_candidate_matches_norm_and_trust_region():
    fedavg = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    basis = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float64)
    candidate, report = sphere_candidate_from_coordinates(
        fedavg, basis, torch.tensor([1.0]), trust_radius=0.2
    )
    assert torch.isclose(candidate.norm(), fedavg.norm(), atol=1e-10)
    assert float((candidate - fedavg).norm()) <= 0.2 + 1e-10
    assert report["fedavg_alignment"] >= 0.0


def test_weighted_disagreement_uses_fedavg_center():
    clients = torch.tensor([[1.0, 1.0], [1.0, -1.0]], dtype=torch.float64)
    weights = torch.tensor([0.5, 0.5], dtype=torch.float64)
    fedavg = torch.tensor([1.0, 0.0], dtype=torch.float64)
    assert abs(weighted_disagreement_scale(clients, fedavg, weights) - 1.0) < 1e-12


def test_span_oracle_improves_synthetic_tail_objective():
    fedavg = torch.tensor([1.0, 0.0, 0.0], dtype=torch.float64)
    basis = torch.tensor([[0.0], [1.0], [0.0]], dtype=torch.float64)

    def evaluate(delta):
        tail_loss = float((delta[1] - 0.4).square().item())
        return {"tail_loss": tail_loss, "head_loss": 0.0, "mid_loss": 0.0}

    result = optimize_span_oracle(
        evaluate,
        fedavg,
        basis,
        gamma=0.6,
        disagreement_scale=1.0,
        lambda_head=1.0,
        lambda_mid=1.0,
        iterations=6,
    )
    assert result.metrics["tail_loss"] < evaluate(fedavg)["tail_loss"]
    assert torch.isclose(result.delta.norm(), fedavg.norm(), atol=1e-8)


def test_span_oracle_accepts_and_preserves_a_better_feasible_initialization():
    fedavg = torch.tensor([1.0, 0.0], dtype=torch.float64)
    basis = torch.tensor([[0.0], [1.0]], dtype=torch.float64)

    def evaluate(delta):
        return {
            "tail_loss": float((delta[1] - 0.3).square().item()),
            "head_loss": 0.0,
            "mid_loss": 0.0,
        }

    initial = torch.tensor([0.3], dtype=torch.float64)
    initial_delta, _ = sphere_candidate_from_coordinates(
        fedavg, basis, initial, trust_radius=0.6
    )
    initial_objective = evaluate(initial_delta)["tail_loss"]
    result = optimize_span_oracle(
        evaluate,
        fedavg,
        basis,
        gamma=0.6,
        disagreement_scale=1.0,
        lambda_head=0.0,
        lambda_mid=0.0,
        iterations=2,
        initial_coordinates=initial,
        initialization="support",
    )
    assert result.metrics["tail_loss"] <= initial_objective + 1e-12
    assert result.report["initialization"] == "support"


def test_support_normalized_delta_renormalizes_only_supporters():
    payload = synthetic_payload()
    deltas = support_normalized_deltas(payload)
    # Class 2 is held only by client 2, so its counterfactual equals that client update.
    assert torch.allclose(deltas[2], torch.tensor([1.0, 1.0], dtype=torch.float64))
    # Class 1 is held by clients 1 and 2 and uses their selected-client FedAvg mass.
    expected = (0.3 * torch.tensor([0.0, 1.0]) + 0.5 * torch.tensor([1.0, 1.0])) / 0.8
    assert torch.allclose(deltas[1], expected.to(torch.float64))


def test_group_metrics_and_gap_closure():
    logits = torch.tensor([[5.0, 0.0, 0.0], [0.0, 5.0, 0.0], [0.0, 2.0, 1.0]])
    labels = torch.tensor([0, 1, 2])
    groups = {"head": [0], "mid": [1], "tail": [2], "non_tail": [0, 1]}
    metrics, per_class = metrics_from_logits(logits, labels, groups)
    assert metrics["head_acc"] == 100.0
    assert metrics["mid_acc"] == 100.0
    assert metrics["tail_acc"] == 0.0
    assert len(per_class) == 3
    assert gap_closure(30.0, 10.0, 50.0) == 0.5
    assert torch.isnan(torch.tensor(gap_closure(10.0, 10.0, 10.0)))


def test_v0_dump_contains_only_lora_trainables(tmp_path):
    key = "image_encoder.block.lora_A"
    before = {key: torch.zeros(2), "image_encoder.block.weight": torch.ones(2)}
    local = [
        {key: torch.tensor([1.0, 0.0]), "image_encoder.block.weight": torch.ones(2)},
        {key: torch.tensor([0.0, 1.0]), "image_encoder.block.weight": torch.ones(2)},
    ]
    after = {key: torch.tensor([0.5, 0.5]), "image_encoder.block.weight": torch.ones(2)}
    args = SimpleNamespace(seed=42, partition="client-longtail")
    path = save_v0_round_dump(
        output_dir=tmp_path,
        args=args,
        cfg="synthetic",
        epoch=4,
        global_before=before,
        global_after=after,
        local_weights=local,
        selected_clients=[0, 1],
        client_sample_counts=[10, 10],
        client_class_counts={0: torch.tensor([5, 0, 1]), 1: torch.tensor([0, 5, 1])},
        global_class_counts=torch.tensor([5, 5, 2]),
        trainable_keys=[key],
    )
    payload = torch.load(path / "round_state.pt", weights_only=False)
    assert payload["trainable_keys"] == [key]
    assert list(payload["global_before_trainable"]) == [key]
    assert (path / "metadata.json").exists()


def test_stratified_cap_is_balanced_and_deterministic():
    items = [{"label": label, "index": index} for label in range(3) for index in range(7)]
    first = stratified_cap(items, per_class=2, seed=9)
    second = stratified_cap(items, per_class=2, seed=9)
    assert first == second
    assert len(first) == 6
    assert [sum(item["label"] == label for item in first) for label in range(3)] == [2, 2, 2]


def test_pooled_oracle_reuses_bank_and_shortlists_safe_evaluations():
    payload = synthetic_payload()

    class FakeEvaluator:
        spec = make_flat_spec(payload["global_before_trainable"])
        groups = {"head": [0], "mid": [1], "tail": [2], "non_tail": [0, 1]}

        @staticmethod
        def state_from_delta(delta):
            return {"image_encoder.block.lora_A": delta.detach().clone()}

        @staticmethod
        def evaluate_delta(delta, loader):
            tail_acc = 40.0 + 5.0 * float(delta[1])
            head_acc = 80.0 - 0.1 * abs(float(delta[1]))
            mid_acc = 60.0
            metrics = {
                "overall_acc": (head_acc + mid_acc + tail_acc) / 3.0,
                "balanced_acc": (head_acc + mid_acc + tail_acc) / 3.0,
                "head_acc": head_acc,
                "mid_acc": mid_acc,
                "tail_acc": tail_acc,
                "non_tail_acc": (head_acc + mid_acc) / 2.0,
                "h3": 3.0 / (1.0 / head_acc + 1.0 / mid_acc + 1.0 / tail_acc),
                "head_loss": 100.0 - head_acc,
                "mid_loss": 100.0 - mid_acc,
                "tail_loss": 100.0 - tail_acc,
            }
            return metrics, []

    args = SimpleNamespace(
        rank_max=2,
        gammas=[0.2],
        random_count=2,
        random_seed=7,
        convex_random_count=2,
        axis_scales=[0.5, 1.0],
        progress_every=100,
        safe_top_k=2,
        lambda_head=[0.0, 1.0],
        lambda_mid=[0.0, 1.0],
        max_head_drop=0.5,
        max_mid_drop=0.5,
        max_overall_drop=0.25,
    )
    states, rows, support_states, context = build_candidates_pooled(
        payload, {}, FakeEvaluator(), object(), object(), args
    )
    methods = {row["method"] for row in rows}
    assert {"fedavg", "random_span", "support_weighting", "oracle_span"} <= methods
    assert set(states) == {row["candidate_id"] for row in rows}
    assert support_states
    assert context["evaluation_counts"]["safe"] < context["evaluation_counts"]["opt"]
