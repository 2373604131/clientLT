import torch

from utils.cusp_minimal import METHODS, SCHEMA_VERSION, build_cusp_candidates, freeze_cusp_candidates


def synthetic_payload():
    before = {
        "prompt_learner.class_aware_ctx": torch.zeros(3, 2),
        "prompt_learner.general_ctx": torch.zeros(1, 2),
    }
    local_states = []
    for client_id in range(4):
        local_states.append({
            "prompt_learner.class_aware_ctx": torch.ones(3, 2) * (client_id + 1) * 0.01,
            "prompt_learner.general_ctx": torch.ones(1, 2) * (client_id + 1) * 0.02,
        })
    return {
        "schema_version": SCHEMA_VERSION,
        "flatten_spec": {
            "keys": ["prompt_learner.class_aware_ctx", "prompt_learner.general_ctx"],
            "shapes": [[3, 2], [1, 2]],
            "dtypes": ["torch.float32", "torch.float32"],
            "offsets": [[0, 6], [6, 8]],
            "numel": 8,
        },
        "global_before_trainable": before,
        "global_after_fedavg_trainable": before,
        "local_trainable_states": local_states,
        "selected_client_ids": [0, 1, 2, 3],
        "fedavg_weights": torch.tensor([0.25, 0.25, 0.25, 0.25], dtype=torch.float64),
        "client_sample_counts": [1, 1, 1, 1],
        "client_class_counts": torch.tensor([[4, 1, 0], [2, 2, 0], [0, 1, 3], [0, 0, 2]]),
        "global_class_counts": torch.tensor([6, 4, 5]),
        "num_classes": 3,
    }


def test_builds_thirteen_equal_norm_candidates(tmp_path):
    payload = synthetic_payload()
    metadata = {"head_class_ids": [0, 1], "tail_class_ids": [2]}
    states, rows, context = build_cusp_candidates(payload, metadata)
    assert len(rows) == 13
    assert {row["method"] for row in rows} == set(METHODS)
    budget = context["norm_budget"]
    for row in rows:
        assert abs(row["final_norm"] - budget) < 1e-8
        assert row["candidate_id"] in states
    manifest = freeze_cusp_candidates(tmp_path, states, rows, context)
    assert manifest["test_accessed"] is False
    assert (tmp_path / "candidate_states.pt").exists()
    assert (tmp_path / "candidate_manifest.csv").exists()
