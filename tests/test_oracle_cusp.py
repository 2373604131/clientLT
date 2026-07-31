import ast
import csv
import json
import subprocess
import sys
from pathlib import Path

import torch
import torch.nn as nn

from utils.oracle_cusp import (
    ROUND1_METHODS,
    classwise_weighting_delta,
    ensure_fedavg_in_subspace,
    fedavg_delta,
    finite_difference_utility,
    flatten_state,
    make_flat_spec,
    random_reweight,
    scale_to_budget,
    unflatten_state,
)


REPO_ROOT = Path(__file__).resolve().parents[1]


class _ReplayPromptLearner(nn.Module):
    def __init__(self):
        super().__init__()
        self.general_ctx = nn.Parameter(torch.tensor([[0.2, -0.1]], dtype=torch.float32))
        self.class_aware_ctx = nn.Parameter(torch.tensor([[[0.1, 0.0]], [[-0.2, 0.3]], [[0.0, -0.1]]], dtype=torch.float32))

    def forward(self):
        general = self.general_ctx.unsqueeze(0).expand(3, -1, -1)
        return torch.cat([general, self.class_aware_ctx], dim=1)


class _ReplayTextEncoder(nn.Module):
    def forward(self, prompts, tokenized_prompts):
        del tokenized_prompts
        return prompts.sum(dim=1)


class _ExplodingImageEncoder(nn.Module):
    def forward(self, image):
        raise AssertionError("cached-feature replay must not call image_encoder")


class _ReplayCustomClipLike(nn.Module):
    def __init__(self):
        super().__init__()
        self.prompt_learner = _ReplayPromptLearner()
        self.text_encoder = _ReplayTextEncoder()
        self.image_encoder = _ExplodingImageEncoder()
        self.tokenized_prompts = torch.zeros(3, 2, dtype=torch.long)
        self.logit_scale = nn.Parameter(torch.tensor(1.25))
        self.dtype = torch.float32


def _expected_cached_logits(model, features):
    prompts = model.prompt_learner()
    text_features = model.text_encoder(prompts, model.tokenized_prompts)
    text_features = text_features / text_features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    normalized = features / features.norm(dim=-1, keepdim=True).clamp_min(1e-12)
    return model.logit_scale.exp() * normalized @ text_features.t()


def _promptfl_replay_methods():
    source_path = REPO_ROOT / "trainers" / "promptfl.py"
    tree = ast.parse(source_path.read_text(encoding="utf-8"))
    custom_clip = next(node for node in tree.body if isinstance(node, ast.ClassDef) and node.name == "CustomCLIP")
    methods = [
        node for node in custom_clip.body
        if isinstance(node, ast.FunctionDef)
        and node.name in {"_logits_from_normalized_features", "logits_from_cached_features"}
    ]
    module = ast.Module(body=methods, type_ignores=[])
    ast.fix_missing_locations(module)
    namespace = {}
    exec(compile(module, str(source_path), "exec"), namespace)
    return namespace["_logits_from_normalized_features"], namespace["logits_from_cached_features"]


def test_flatten_round_trip_includes_general_and_class_aware():
    state = {
        "prompt_learner.general_ctx": torch.tensor([1.0, 2.0]),
        "prompt_learner.class_aware_ctx": torch.tensor([[3.0], [4.0]]),
    }
    spec = make_flat_spec(state, list(state))
    flat = flatten_state(state, spec)
    assert torch.equal(flatten_state(unflatten_state(flat, spec), spec), flat)
    assert set(spec.keys) == {"prompt_learner.general_ctx", "prompt_learner.class_aware_ctx"}


def test_fedavg_subspace_and_finite_difference_are_stable():
    before = torch.zeros(3, dtype=torch.float64)
    locals_ = torch.tensor([[1., 0., 0.], [0., 1., 0.]], dtype=torch.float64)
    weights = torch.tensor([.25, .75], dtype=torch.float64)
    delta = fedavg_delta(before, locals_, weights)
    deltas = (locals_ - before).T
    from utils.oracle_cusp import subspace_from_updates
    info = ensure_fedavg_in_subspace(subspace_from_updates(deltas), delta)
    a, report = finite_difference_utility(lambda x: torch.stack([x.sum(), 2 * x[0] - x[1]]), before, info["Q"])
    assert report["stable"]
    assert report["relative_difference"] <= 0.10
    assert torch.allclose(info["Q"] @ (info["Q"].T @ delta), delta, atol=1e-10)
    assert a.shape[0] == 2


def test_scale_to_budget_enforces_equal_norm_and_rejects_zero():
    final, report = scale_to_budget(torch.tensor([3.0, 4.0]), 2.0)
    assert report["valid"]
    assert torch.isclose(final.norm(), torch.tensor(2.0, dtype=torch.float64))
    final, report = scale_to_budget(torch.zeros(2), 2.0)
    assert final is None
    assert not report["valid"]


def test_random_reweight_has_ten_seeded_equal_norm_candidates():
    deltas = torch.tensor([[1., -1., .5], [0., 1., -.5]], dtype=torch.float64)
    first, second = random_reweight(deltas, 0.5), random_reweight(deltas, 0.5)
    assert len(first) == 10
    assert [x["coefficient_hash"] for x in first] == [x["coefficient_hash"] for x in second]
    assert all(x["delta"] is not None and abs(x["final_norm"] - 0.5) <= 1e-9 for x in first)


def test_classwise_weighting_uses_support_rows_and_fedavg_general():
    before = {
        "prompt_learner.general_ctx": torch.zeros(1),
        "prompt_learner.class_aware_ctx": torch.zeros(3, 1),
    }
    spec = make_flat_spec(before, ["prompt_learner.general_ctx", "prompt_learner.class_aware_ctx"])
    locals_ = [
        {"prompt_learner.general_ctx": torch.tensor([1.0]), "prompt_learner.class_aware_ctx": torch.tensor([[10.0], [0.0], [100.0]])},
        {"prompt_learner.general_ctx": torch.tensor([3.0]), "prompt_learner.class_aware_ctx": torch.tensor([[20.0], [50.0], [200.0]])},
    ]
    delta, report = classwise_weighting_delta(
        before,
        locals_,
        torch.tensor([0.25, 0.75], dtype=torch.float64),
        torch.tensor([[1, 0, 0], [1, 2, 0]]),
        spec,
        num_classes=3,
        budget=10.0,
    )
    assert delta is not None
    assert abs(delta.norm().item() - 10.0) <= 1e-6
    raw_state = {
        "prompt_learner.general_ctx": torch.tensor([2.5]),
        "prompt_learner.class_aware_ctx": torch.tensor([[17.5], [50.0], [0.0]]),
    }
    raw_delta = flatten_state(raw_state, spec) - flatten_state(before, spec)
    assert torch.allclose(delta / delta.norm(), raw_delta / raw_delta.norm(), atol=1e-6)
    assert report["fallback_class_ids"] == [2]


def test_promptfl_cached_feature_replay_matches_formula_and_restores_state():
    logits_helper, cached_helper = _promptfl_replay_methods()

    model = _ReplayCustomClipLike()
    model._logits_from_normalized_features = logits_helper.__get__(model, _ReplayCustomClipLike)
    model.logits_from_cached_features = cached_helper.__get__(model, _ReplayCustomClipLike)
    model.train()
    features = torch.tensor([[1.0, 2.0], [-1.0, 0.5]], dtype=torch.float32)
    original_mode = model.training
    original_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    candidate_state = {
        "prompt_learner.general_ctx": original_state["prompt_learner.general_ctx"] + 0.1,
        "prompt_learner.class_aware_ctx": original_state["prompt_learner.class_aware_ctx"] - 0.05,
    }

    patched = {key: value.clone() for key, value in original_state.items()}
    patched.update(candidate_state)
    model.load_state_dict(patched, strict=False)
    expected = _expected_cached_logits(model, features)
    model.load_state_dict(original_state, strict=False)

    actual = model.logits_from_cached_features(features.cpu(), candidate_state)
    assert torch.allclose(actual.cpu(), expected.cpu(), atol=1e-6, rtol=1e-6)
    assert model.training == original_mode
    for key, value in model.state_dict().items():
        assert torch.equal(value.cpu(), original_state[key].cpu())


def test_promptfl_cached_feature_replay_restores_state_after_exception():
    logits_helper, cached_helper = _promptfl_replay_methods()

    model = _ReplayCustomClipLike()
    model._logits_from_normalized_features = logits_helper.__get__(model, _ReplayCustomClipLike)
    model.logits_from_cached_features = cached_helper.__get__(model, _ReplayCustomClipLike)
    original_state = {key: value.detach().clone() for key, value in model.state_dict().items()}
    try:
        model.logits_from_cached_features(torch.ones(1, 2), {"prompt_learner.missing": torch.ones(1)})
    except KeyError:
        pass
    else:
        raise AssertionError("unknown candidate key should fail")
    for key, value in model.state_dict().items():
        assert torch.equal(value.cpu(), original_state[key].cpu())


def test_synthetic_smoke_builds_thirteen_frozen_candidates(tmp_path):
    result = subprocess.run(
        [sys.executable, "scripts/oracle_cusp_single_round.py", "--synthetic-smoke", "--output-dir", str(tmp_path)],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0, result.stderr
    with (tmp_path / "oracle_method_summary.csv").open(newline="", encoding="utf-8") as handle:
        methods = {row["method"] for row in csv.DictReader(handle)}
    assert methods == set(ROUND1_METHODS)
    with (tmp_path / "random_reweight_distribution.csv").open(newline="", encoding="utf-8") as handle:
        random_rows = list(csv.DictReader(handle))
    assert len(random_rows) == 10
    manifest = json.loads((tmp_path / "candidate_manifest.json").read_text(encoding="utf-8"))
    assert manifest["num_concrete_candidates"] == 13
    assert manifest["test_accessed"] is False
    for name in [
        "candidate_states.pt", "candidate_manifest.json", "oracle_method_summary.csv",
        "oracle_per_class.csv", "random_reweight_distribution.csv", "oracle_solver.json",
        "oracle_metadata.json", "oracle_report.md",
    ]:
        assert (tmp_path / name).exists()


def test_minimal_launcher_does_not_enable_experiment_d_or_external_schedule_creation():
    text = (REPO_ROOT / "scripts" / "cusp_oracle_round1.py").read_text(encoding="utf-8")
    assert '"--experimentD_enable", "False"' in text
    assert "--experimentD_enable True" not in text
    assert "scripts/create_client_schedule.py" not in text
    assert '"--round", "10"' in text
    assert '"--oracle_cusp_round", "10"' in text
    assert 'train_dir / "oracle_cusp" / "round_010"' in text
