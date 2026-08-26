from pathlib import Path
from types import SimpleNamespace

import pytest
import torch
from torch import nn

from utils.stage3_runtime import Stage3FederatedRuntime
from utils.stage3_vectors import (
    build_model_lora_flat_spec,
    flatten_model,
    load_lora_vector,
)


class _Datum:
    def __init__(self, image, label):
        self.image = torch.as_tensor(image, dtype=torch.float32)
        self.label = int(label)


class _LocalDataset:
    k_tfm = 1

    def __init__(self, data_source):
        self.data_source = list(data_source)

    def __len__(self):
        return len(self.data_source)

    def __getitem__(self, index):
        datum = self.data_source[index]
        return {"img": datum.image.clone(), "label": datum.label}


class _ToyPrivateClassifier(nn.Module):
    def __init__(self):
        super().__init__()
        self.image_encoder = nn.Module()
        self.image_encoder.q_lora_A = nn.Parameter(torch.zeros(2, 2))

    def forward(self, images):
        return images @ self.image_encoder.q_lora_A


class _Trainer:
    def __init__(self, client_count=4, samples_per_client=40):
        self.model = _ToyPrivateClassifier()
        self.cfg = SimpleNamespace(DATASET=SimpleNamespace(NAME="Cifar100_LT"))
        global_train = []
        loaders = {}
        for client_id in range(client_count):
            local = []
            for offset in range(samples_per_client):
                label = offset % 2
                image = [1.0, 0.0] if label == 0 else [0.0, 1.0]
                datum = _Datum(image, label)
                local.append(datum)
                global_train.append(datum)
            loaders[client_id] = SimpleNamespace(dataset=_LocalDataset(local))
        self.dm = SimpleNamespace(
            dataset=SimpleNamespace(train_x=global_train)
        )
        self.fed_train_loader_x_dict = loaders


def _state(model):
    return {key: value.detach().clone() for key, value in model.state_dict().items()}


def _set_delta(model, spec, anchor, delta):
    load_lora_vector(
        model,
        anchor + torch.as_tensor(delta, dtype=torch.float32).reshape(-1),
        spec,
    )


def test_fedavg_runtime_upload_is_exact_ce_delta_and_saves_round_state(tmp_path):
    trainer = _Trainer()
    initial_spec = build_model_lora_flat_spec(trainer.model)
    load_lora_vector(
        trainer.model,
        torch.tensor([1.25, -0.75, 0.5, -1.5]),
        initial_spec,
    )
    runtime = Stage3FederatedRuntime(
        trainer,
        output_dir=tmp_path,
        global_seed=42,
        condition="fedavg",
        num_users=4,
    )
    incoming_state = _state(trainer.model)
    prepared = runtime.prepare_client(
        trainer.model, incoming_state, client_id=0, round_id=0
    )
    ce_delta = torch.tensor([0.25, -0.125, -0.25, 0.125])
    _set_delta(trainer.model, runtime.spec, prepared.incoming_vector, ce_delta)

    finalized = runtime.finalize_client(trainer.model, prepared)

    assert torch.equal(finalized.upload.vector, ce_delta)
    assert finalized.norm_report["fcc_active"] is False
    assert finalized.norm_report["rtc_active"] is False
    assert finalized.norm_report["norm_gate_pass"] is True
    assert set(finalized.upload.__dict__) == {
        "client_id",
        "vector",
        "spec_hash",
        "condition",
        "round_id",
    }
    assert torch.equal(
        flatten_model(trainer.model, runtime.spec).cpu(),
        prepared.incoming_vector + ce_delta,
    )

    uploads = []
    for client_id in range(4):
        uploads.append(
            finalized.upload.__class__(
                client_id=client_id,
                vector=ce_delta * float(client_id + 1),
                spec_hash=runtime.spec.spec_hash,
                condition="fedavg",
                round_id=0,
            )
        )
    bank = runtime.complete_round(0, uploads)
    assert bank.target_round == 1
    assert (Path(tmp_path) / "stage3" / "client_private_state_latest.pt").is_file()
    assert (Path(tmp_path) / "stage3" / "proposal_bank_latest.pt").is_file()


def test_previous_round_bank_is_leave_one_out_and_privately_evaluated(tmp_path):
    trainer = _Trainer()
    runtime = Stage3FederatedRuntime(
        trainer,
        output_dir=tmp_path,
        global_seed=42,
        condition="p_fcc_only",
        num_users=4,
    )
    direction = torch.tensor([1.0, -1.0, -1.0, 1.0])
    upload_type = None
    uploads = []
    for client_id in range(4):
        incoming_state = _state(trainer.model)
        prepared = runtime.prepare_client(
            trainer.model, incoming_state, client_id=client_id, round_id=0
        )
        _set_delta(
            trainer.model,
            runtime.spec,
            prepared.incoming_vector,
            direction * (client_id + 1) * 0.1,
        )
        finalized = runtime.finalize_client(trainer.model, prepared)
        upload_type = finalized.upload.__class__
        uploads.append(finalized.upload)
        load_lora_vector(trainer.model, torch.zeros(4), runtime.spec)
    runtime.complete_round(0, uploads)

    incoming_state = _state(trainer.model)
    prepared = runtime.prepare_client(
        trainer.model, incoming_state, client_id=0, round_id=1
    )

    assert upload_type is not None
    assert prepared.selection.forward_count == 2
    assert len(prepared.selection.probes) == 1
    assert prepared.selection.probes[0].source_count == 3
    assert prepared.selection.selected_proposal_ids
    assert prepared.selection.probes[0].utility > 0
    assert torch.equal(flatten_model(trainer.model, runtime.spec), torch.zeros(4))

    # With no subsequent CE delta, every fixed-norm candidate is the zero
    # upload. The post-local private gate must deterministically fall back to
    # multiplier zero and leave the model untouched.
    finalized = runtime.finalize_client(trainer.model, prepared)
    assert finalized.postlocal_fcc is not None
    assert finalized.postlocal_fcc.selected_multiplier == 0.0
    assert finalized.postlocal_fcc.forward_count == 4
    assert torch.equal(finalized.upload.vector, torch.zeros(4))
    assert (
        Path(tmp_path)
        / "stage3"
        / "client_local_research_audit"
        / "postlocal_fcc_safety.csv"
    ).is_file()


def test_d_rtc_uses_degradation_and_keeps_final_upload_at_ce_norm(tmp_path):
    trainer = _Trainer()
    runtime = Stage3FederatedRuntime(
        trainer,
        output_dir=tmp_path,
        global_seed=42,
        condition="d_rtc_only",
        num_users=4,
    )
    zero_state = _state(trainer.model)
    first = runtime.prepare_client(
        trainer.model, zero_state, client_id=0, round_id=0
    )
    _set_delta(trainer.model, runtime.spec, first.incoming_vector, [0.1, 0, 0, 0.1])
    first_upload = runtime.finalize_client(trainer.model, first).upload
    runtime.complete_round(
        0,
        [
            first_upload.__class__(
                client_id=client_id,
                vector=first_upload.vector,
                spec_hash=runtime.spec.spec_hash,
                condition="d_rtc_only",
                round_id=0,
            )
            for client_id in range(4)
        ],
    )

    # Swap the two labels in logit space so the incoming global is worse than
    # the round-0 private reference.
    bad_incoming = torch.tensor([-2.0, 2.0, 2.0, -2.0])
    load_lora_vector(trainer.model, bad_incoming, runtime.spec)
    bad_state = _state(trainer.model)
    prepared = runtime.prepare_client(
        trainer.model, bad_state, client_id=0, round_id=1
    )
    assert prepared.degradation > 0
    ce_delta = torch.tensor([0.03, -0.02, -0.01, 0.04])
    _set_delta(trainer.model, runtime.spec, prepared.incoming_vector, ce_delta)
    finalized = runtime.finalize_client(trainer.model, prepared)

    assert finalized.norm_report["rtc_active"] is True
    assert finalized.norm_report["norm_gate_pass"] is True
    assert torch.linalg.vector_norm(finalized.upload.vector).item() == pytest.approx(
        finalized.norm_report["target_ce_norm"], rel=1e-6
    )
