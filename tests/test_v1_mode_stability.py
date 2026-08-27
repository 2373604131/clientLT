from types import SimpleNamespace
import json
import subprocess
import sys
from pathlib import Path

import torch

from utils.cusp_minimal import make_flat_spec
from utils.v1_mode_stability import (
    ModeSet,
    build_disagreement_set,
    compare_atom_modes,
    compare_degenerate_subspaces,
    joint_sketch_modes,
    layer_segments,
    optimal_assignment,
    upload_set_from_payload,
)
from utils.v0_oracle import save_v0_round_dump


def synthetic_uploads():
    before = {
        "image_encoder.transformer.resblocks.9.attn.q_proj.w_lora_A": torch.zeros(2),
        "image_encoder.transformer.resblocks.10.attn.q_proj.w_lora_A": torch.zeros(2),
    }
    local_vectors = (
        (1.0, 0.0, 0.0, 0.0),
        (0.0, 1.0, 0.0, 0.0),
        (-1.0, 0.0, 1.0, 0.0),
        (0.0, -1.0, 0.0, 1.0),
    )
    keys = sorted(before)
    local = []
    for vector in local_vectors:
        local.append({keys[0]: torch.tensor(vector[:2]), keys[1]: torch.tensor(vector[2:])})
    spec = make_flat_spec(before)
    payload = {
        "flatten_spec": spec.as_dict(),
        "global_before_trainable": before,
        "local_trainable_states": local,
        "fedavg_weights": torch.full((4,), 0.25, dtype=torch.float64),
        "selected_client_ids": [0, 1, 2, 3],
    }
    metadata = {
        "communication_round": 20,
        "resolved_args": {"seed": 42, "partition": "client-longtail-controlled"},
    }
    return upload_set_from_payload(payload, metadata, "synthetic")


def test_layer_segments_group_lora_tensors_by_transformer_block():
    uploads = synthetic_uploads()
    assert len(layer_segments(uploads.spec)) == 2
    assert any(name.endswith("resblocks.9") for name in uploads.layers)
    assert any(name.endswith("resblocks.10") for name in uploads.layers)


def test_disagreement_rows_are_orthogonal_to_fedavg_direction():
    uploads = synthetic_uploads()
    data = build_disagreement_set(
        uploads,
        row_indices=[0, 1, 2],
        weight_multipliers=torch.tensor([1.0, 1.1, 0.9]),
    )
    assert data.matrix.shape == (3, 4)
    assert data.orthogonality_error < 1e-10


def test_degenerate_subspace_survives_atom_rotation():
    root = 2.0**-0.5
    reference = ModeSet(
        directions=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float64),
        singular_values=torch.tensor([1.0, 1.0], dtype=torch.float64),
        labels=("atom_0", "atom_1"),
    )
    candidate = ModeSet(
        directions=torch.tensor([[root, root], [-root, root]], dtype=torch.float64),
        singular_values=torch.tensor([1.0, 1.0], dtype=torch.float64),
        labels=("atom_0", "atom_1"),
    )
    atom_metrics, _ = compare_atom_modes(reference, candidate)
    subspace_metrics, _ = compare_degenerate_subspaces(
        reference, candidate, relative_gap=0.05
    )
    assert atom_metrics["stability_score"] < 0.8
    assert abs(subspace_metrics["stability_score"] - 1.0) < 1e-10


def test_joint_sketch_is_deterministic_for_fixed_seed():
    data = build_disagreement_set(synthetic_uploads())
    first = joint_sketch_modes(data, 2, sketch_dim=8, seed=2026)
    second = joint_sketch_modes(data, 2, sketch_dim=8, seed=2026)
    assert torch.allclose(torch.abs(first.directions), torch.abs(second.directions))


def test_large_assignment_uses_exact_hungarian_solution():
    scores = torch.eye(13, dtype=torch.float64)
    scores[0, 0] = 1.0
    scores[0, 1] = 0.99
    scores[1, 0] = 0.98
    scores[1, 1] = 0.0
    pairs = dict(optimal_assignment(scores))
    assert pairs[0] == 1
    assert pairs[1] == 0


def _write_runner_dump(root: Path, seed: int) -> Path:
    keys = [
        f"image_encoder.transformer.resblocks.{block}.attn.q_proj.w_lora_A"
        for block in (9, 10, 11)
    ]
    before = {key: torch.zeros(4) for key in keys}
    generator = torch.Generator().manual_seed(seed)
    local = []
    for client_id in range(12):
        state = {}
        for layer_id, key in enumerate(keys):
            shared = torch.tensor([1.0, -0.5, 0.25, 0.0]) * (layer_id + 1)
            noise = 0.05 * torch.randn(4, generator=generator)
            state[key] = shared * ((client_id % 3) - 1) + noise
        local.append(state)
    after = {
        key: torch.stack([state[key] for state in local]).mean(dim=0) for key in keys
    }
    args = SimpleNamespace(seed=seed, partition="client-longtail-controlled")
    return save_v0_round_dump(
        output_dir=root / f"seed{seed}",
        args=args,
        cfg="synthetic",
        epoch=19,
        global_before=before,
        global_after=after,
        local_weights=local,
        selected_clients=list(range(12)),
        client_sample_counts=[10] * 12,
        client_class_counts={
            client_id: torch.tensor([4, 3, 2, 1]) for client_id in range(12)
        },
        global_class_counts=torch.tensor([48, 36, 24, 12]),
        trainable_keys=keys,
    )


def test_runner_writes_complete_protocol_outputs(tmp_path):
    dump_dirs = [_write_runner_dump(tmp_path / "dumps", seed) for seed in (1, 42, 2026)]
    output_dir = tmp_path / "analysis"
    command = [
        sys.executable,
        "scripts/run_v1_mode_stability.py",
        "--dump-dirs",
        *(str(path) for path in dump_dirs),
        "--output-dir",
        str(output_dir),
        "--ranks",
        "2",
        "4",
        "--perturb-repeats",
        "1",
        "--sketch-dim",
        "8",
    ]
    subprocess.run(command, cwd=Path(__file__).resolve().parents[1], check=True)
    assert (output_dir / "stability_summary.csv").is_file()
    assert (output_dir / "v1_report.md").is_file()
    verdict = json.loads((output_dir / "v1_verdict.json").read_text(encoding="utf-8"))
    assert verdict["recommended_representation"] in {
        "single_svd_atom",
        "near_degenerate_subspace",
        "layerwise_mode",
        "cross_layer_joint_sketch",
    }
