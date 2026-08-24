"""Federated ClipLora orchestration for the frozen Stage-3 v1 method.

This file is the simulation adapter.  Private evidence, proposal utilities,
reference logits, and D-RTC triggers stay inside ``Stage3FederatedRuntime``;
the caller receives only one ordinary ``ClientUpload`` vector per client.
"""

from __future__ import annotations

import csv
import json
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as F

from utils.stage3_methods import (
    EvidenceBatch,
    ProposalProbe,
    ProposalSelection,
    RestoreResult,
    compute_restore_direction,
    evaluate_and_select_proposals,
    evaluate_private_logits,
    evaluate_proposals_for_audit,
)
from utils.stage3_private_state import ClientPrivateStateStore
from utils.stage3_proposals import (
    ClientUpload,
    ProposalBank,
    build_proposal_bank,
)
from utils.stage3_vectors import (
    EPS_NORM,
    LoRAFlatSpec,
    build_model_lora_flat_spec,
    compose_fixed_norm_upload,
    flatten_model,
    flatten_state,
    load_lora_vector,
)


STAGE3_CONDITIONS = (
    "fedavg",
    "p_fcc_only",
    "d_rtc_only",
    "combined",
    "random_proposal",
)
LAMBDA_FCC = 0.5
LAMBDA_RTC = 0.5


def _proposal_mode(condition: str) -> str | None:
    if condition in ("p_fcc_only", "combined"):
        return "private"
    if condition == "random_proposal":
        return "random"
    return None


def _rtc_enabled(condition: str) -> bool:
    return condition in ("d_rtc_only", "combined")


@contextmanager
def _preserved_rng():
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


class ClientEvidenceRepository:
    """Maps global train indices to deterministic client-private tensors."""

    def __init__(self, trainer, state_store: ClientPrivateStateStore, num_users: int):
        dataset_name = str(trainer.cfg.DATASET.NAME)
        if dataset_name != "Cifar100_LT":
            raise ValueError(
                "Stage-3 seed-42 MVP is frozen to Cifar100_LT; "
                f"got {dataset_name!r}"
            )
        global_train = list(getattr(trainer.dm.dataset, "train_x", []) or [])
        if not global_train:
            raise ValueError("Stage-3 requires a non-empty global train index universe")
        object_to_global_id = {id(item): index for index, item in enumerate(global_train)}
        if len(object_to_global_id) != len(global_train):
            raise ValueError("Global training universe repeats Datum object identities")

        self._state_store = state_store
        self._datasets = {}
        self._positions = {}
        self._labels = {}
        for client_id in range(int(num_users)):
            local_dataset = trainer.fed_train_loader_x_dict[client_id].dataset
            if int(getattr(local_dataset, "k_tfm", 1)) != 1:
                raise ValueError("Stage-3 deterministic evidence requires K_TRANSFORMS=1")
            data_source = list(local_dataset.data_source)
            stable_ids = []
            labels = []
            positions = {}
            for local_position, item in enumerate(data_source):
                stable_id = object_to_global_id.get(id(item))
                if stable_id is None:
                    raise ValueError(
                        f"Client {client_id} contains a sample without a stable global train ID"
                    )
                stable_id = int(stable_id)
                stable_ids.append(stable_id)
                labels.append(int(item.label))
                positions[stable_id] = int(local_position)
            evidence = state_store.get_or_create_evidence(
                client_id, stable_ids, labels
            )
            if len(positions) != len(stable_ids):
                raise ValueError(f"Client {client_id} repeats a stable sample ID")
            self._datasets[client_id] = local_dataset
            self._positions[client_id] = positions
            self._labels[client_id] = {
                stable_id: label for stable_id, label in zip(stable_ids, labels)
            }
            # Touching ``evidence`` here makes checkpoint/data mismatch fail
            # before any model forward or local optimizer step.
            _ = evidence.fingerprint

    def batch(self, client_id: int, view: str) -> EvidenceBatch | None:
        client_id = int(client_id)
        state = self._state_store.get(client_id)
        if state is None:
            raise KeyError(f"No private evidence state for client {client_id}")
        if view == "memory":
            stable_ids = state.evidence.memory_sample_ids
        elif view == "audit":
            stable_ids = state.evidence.audit_sample_ids
        else:
            raise ValueError(f"Unknown private evidence view: {view!r}")
        if not stable_ids:
            return None

        images = []
        labels = []
        dataset = self._datasets[client_id]
        with _preserved_rng():
            for stable_id in stable_ids:
                local_position = self._positions[client_id][stable_id]
                output = dataset[local_position]
                image = output.get("img") if isinstance(output, Mapping) else None
                label = output.get("label") if isinstance(output, Mapping) else None
                if not torch.is_tensor(image):
                    raise ValueError("Deterministic evidence loader did not return an image tensor")
                expected_label = self._labels[client_id][stable_id]
                if int(label) != int(expected_label):
                    raise ValueError("Evidence label changed during deterministic materialization")
                images.append(image.detach().cpu().clone())
                labels.append(expected_label)
        return EvidenceBatch(
            images=torch.stack(images, dim=0),
            labels=torch.tensor(labels, dtype=torch.long),
            sample_ids=tuple(stable_ids),
        )

@dataclass(frozen=True)
class PreparedStage3Client:
    client_id: int
    round_id: int
    incoming_vector: torch.Tensor
    evidence: EvidenceBatch
    audit: EvidenceBatch | None
    degradation: float
    selection: ProposalSelection
    audit_probes: tuple[ProposalProbe, ...]
    audit_incoming_ce: float | None


@dataclass(frozen=True)
class FinalizedStage3Client:
    upload: ClientUpload
    norm_report: Mapping


def _private_ce(model, batch: EvidenceBatch | None) -> float | None:
    if batch is None:
        return None
    logits = evaluate_private_logits(model, batch)
    labels = batch.labels.to(logits.device)
    return float(F.cross_entropy(logits, labels, reduction="mean").item())


def _empty_selection(spec: LoRAFlatSpec, base_ce: float) -> ProposalSelection:
    return ProposalSelection(
        mode="none",
        base_ce=float(base_ce),
        probes=(),
        selected_proposal_ids=(),
        selected_weights=(),
        direction=torch.zeros(spec.numel, dtype=torch.float32),
        forward_count=0,
    )


def _identity_norm_report(
    ce_delta: torch.Tensor, *, degradation: float = 0.0
) -> dict:
    norm = float(torch.linalg.vector_norm(ce_delta).item())
    return {
        "target_ce_norm": norm,
        "composition_norm_before_final_scaling": norm,
        "actual_upload_norm": norm,
        "relative_norm_error": 0.0,
        "norm_gate_pass": True,
        "fallback": "no_active_correction",
        "fcc_active": False,
        "rtc_active": False,
        "degradation": float(degradation),
        "dtype_roundtrips": 0,
    }


def _append_csv(path: Path, row: Mapping, fieldnames: Sequence[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(fieldnames), extrasaction="raise")
        if not exists:
            writer.writeheader()
        writer.writerow(row)


class Stage3FederatedRuntime:
    """Condition-local client/runtime state for one formal Stage-3 run."""

    def __init__(
        self,
        trainer,
        *,
        output_dir: str | Path,
        global_seed: int,
        condition: str,
        num_users: int,
        start_round: int = 0,
        resume_dir: str | Path | None = None,
    ):
        condition = str(condition)
        if condition not in STAGE3_CONDITIONS:
            raise ValueError(
                f"Unknown Stage-3 condition {condition!r}; choose from {STAGE3_CONDITIONS}"
            )
        self.condition = condition
        self.global_seed = int(global_seed)
        self.output_dir = Path(output_dir)
        self.stage3_dir = self.output_dir / "stage3"
        self.spec = build_model_lora_flat_spec(trainer.model)

        resume_root = None if resume_dir is None else Path(resume_dir) / "stage3"
        private_path = None if resume_root is None else resume_root / "client_private_state_latest.pt"
        bank_path = None if resume_root is None else resume_root / "proposal_bank_latest.pt"
        if private_path is not None and private_path.exists():
            if not bank_path.exists():
                raise FileNotFoundError("Stage-3 resume is missing proposal bank state")
            self.state_store = ClientPrivateStateStore.load(
                private_path,
                expected_global_seed=self.global_seed,
                expected_condition=self.condition,
                expected_flatten_spec_hash=self.spec.spec_hash,
            )
            self.bank = ProposalBank.load(
                bank_path,
                expected_global_seed=self.global_seed,
                expected_condition=self.condition,
                expected_source_round=int(start_round) - 1,
                expected_spec_hash=self.spec.spec_hash,
            )
        else:
            if bank_path is not None and bank_path.exists():
                raise FileNotFoundError("Stage-3 resume is missing client-private state")
            self.state_store = ClientPrivateStateStore(
                global_seed=self.global_seed,
                condition=self.condition,
                flatten_spec_hash=self.spec.spec_hash,
            )
            self.bank = build_proposal_bank(
                [],
                spec=self.spec,
                global_seed=self.global_seed,
                source_round=int(start_round) - 1,
                condition=self.condition,
            )
        self.evidence_repository = ClientEvidenceRepository(
            trainer, self.state_store, int(num_users)
        )
        self._write_contract()

    def _write_contract(self) -> None:
        self.stage3_dir.mkdir(parents=True, exist_ok=True)
        payload = {
            "schema_version": "p_fcc_d_rtc_runtime_v1",
            "condition": self.condition,
            "global_seed": self.global_seed,
            "flatten_spec": self.spec.as_dict(),
            "lambda_fcc": LAMBDA_FCC,
            "lambda_rtc": LAMBDA_RTC,
            "proposal_probe_dose": 0.5,
            "accepted_proposals": 2,
            "restore_temperature": 2.0,
            "server_public_data": False,
            "runtime_class_information": False,
            "client_upload": "one ordinary LoRA delta plus standard protocol metadata",
        }
        path = self.stage3_dir / "runtime_contract.json"
        path.write_text(
            json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2) + "\n",
            encoding="utf-8",
        )

    def assert_model_spec(self, model) -> None:
        candidate = build_model_lora_flat_spec(model)
        if candidate.spec_hash != self.spec.spec_hash:
            raise ValueError("Global/local ClipLora flatten specs differ")

    def prepare_client(
        self,
        model,
        incoming_state: Mapping[str, torch.Tensor],
        *,
        client_id: int,
        round_id: int,
    ) -> PreparedStage3Client:
        self.assert_model_spec(model)
        incoming = flatten_state(incoming_state, self.spec, device="cpu")
        observed = flatten_model(model, self.spec, device="cpu")
        if not torch.equal(incoming, observed):
            raise RuntimeError("Stage-3 client did not start from the incoming global LoRA")
        evidence = self.evidence_repository.batch(client_id, "memory")
        if evidence is None:
            raise RuntimeError("Functional memory cannot be empty for a participating client")
        audit = self.evidence_repository.batch(client_id, "audit")

        # Frozen order: incoming-global reference first, proposals second.
        incoming_logits = evaluate_private_logits(model, evidence)
        observation = self.state_store.observe_incoming_global(
            client_id,
            logits=incoming_logits,
            labels=evidence.labels,
            round_id=round_id,
        )
        payload = self.bank.payload_for(
            client_id,
            expected_condition=self.condition,
            expected_target_round=round_id,
            expected_spec_hash=self.spec.spec_hash,
        )
        mode = _proposal_mode(self.condition)
        if mode is None:
            selection = _empty_selection(self.spec, observation.current_ce)
            audit_probes = ()
        else:
            selection = evaluate_and_select_proposals(
                model,
                self.spec,
                incoming,
                evidence,
                payload,
                mode=mode,
                global_seed=self.global_seed,
                round_id=round_id,
                client_id=client_id,
            )
            audit_probes = evaluate_proposals_for_audit(
                model, self.spec, incoming, audit, payload
            )
        audit_incoming_ce = _private_ce(model, audit)
        self._log_probes(
            client_id=client_id,
            round_id=round_id,
            selection=selection,
            audit_probes=audit_probes,
        )
        return PreparedStage3Client(
            client_id=int(client_id),
            round_id=int(round_id),
            incoming_vector=incoming,
            evidence=evidence,
            audit=audit,
            degradation=float(observation.degradation),
            selection=selection,
            audit_probes=audit_probes,
            audit_incoming_ce=audit_incoming_ce,
        )

    def finalize_client(
        self,
        model,
        prepared: PreparedStage3Client,
    ) -> FinalizedStage3Client:
        if prepared.round_id != self.bank.target_round:
            raise ValueError("Prepared client and active proposal bank rounds differ")
        local_ce_vector = flatten_model(model, self.spec, device="cpu")
        ce_delta = local_ce_vector - prepared.incoming_vector
        audit_local_ce = _private_ce(model, prepared.audit)

        restore = RestoreResult(
            direction=torch.zeros(self.spec.numel, dtype=torch.float32),
            loss=0.0,
            ce=0.0,
            kl=0.0,
            backward_count=0,
            gradient_norm=0.0,
        )
        if _rtc_enabled(self.condition):
            state = self.state_store.get(prepared.client_id)
            if state is None or state.reference is None:
                raise RuntimeError("D-RTC client has no incoming-global reference")
            restore = compute_restore_direction(
                model,
                self.spec,
                prepared.evidence,
                state.reference,
                degradation=prepared.degradation,
            )

        fcc_active = (
            _proposal_mode(self.condition) is not None
            and float(torch.linalg.vector_norm(prepared.selection.direction).item())
            > EPS_NORM
        )
        rtc_active = (
            _rtc_enabled(self.condition)
            and prepared.degradation > 0.0
            and float(torch.linalg.vector_norm(restore.direction).item()) > EPS_NORM
        )
        if not fcc_active and not rtc_active:
            # The no-correction branch is deliberately bitwise passive: the
            # ordinary local CE model remains untouched and its already
            # representable FP32 delta is uploaded verbatim.
            upload = ce_delta.detach().cpu().to(torch.float32).clone()
            norm_report = _identity_norm_report(
                upload,
                degradation=(
                    prepared.degradation if _rtc_enabled(self.condition) else 0.0
                ),
            )
        else:
            parameters = dict(model.named_parameters())
            upload, norm_report = compose_fixed_norm_upload(
                ce_delta,
                fcc_direction=prepared.selection.direction,
                rtc_direction=restore.direction,
                lambda_fcc=LAMBDA_FCC if fcc_active else 0.0,
                lambda_rtc=LAMBDA_RTC if rtc_active else 0.0,
                degradation=prepared.degradation if rtc_active else 0.0,
                spec=self.spec,
                like=parameters,
                anchor_vector=prepared.incoming_vector,
            )
            final_vector = prepared.incoming_vector + upload
            load_lora_vector(model, final_vector, self.spec)
            actual_upload = (
                flatten_model(model, self.spec, device="cpu")
                - prepared.incoming_vector
            )
            if not torch.equal(actual_upload, upload):
                raise RuntimeError(
                    "Applied Stage-3 upload differs after model-dtype loading"
                )
        audit_final_ce = _private_ce(model, prepared.audit)
        self._log_client_runtime(
            prepared,
            restore,
            norm_report,
            ce_delta=ce_delta,
            upload=upload,
            audit_local_ce=audit_local_ce,
            audit_final_ce=audit_final_ce,
        )
        return FinalizedStage3Client(
            upload=ClientUpload(
                client_id=prepared.client_id,
                vector=upload.detach().cpu().to(torch.float32).clone(),
                spec_hash=self.spec.spec_hash,
                condition=self.condition,
                round_id=prepared.round_id,
            ),
            norm_report=dict(norm_report),
        )

    def complete_round(
        self, round_id: int, uploads: Sequence[ClientUpload]
    ) -> ProposalBank:
        self.bank = build_proposal_bank(
            list(uploads),
            spec=self.spec,
            global_seed=self.global_seed,
            source_round=int(round_id),
            condition=self.condition,
        )
        self.state_store.save(self.stage3_dir / "client_private_state_latest.pt")
        self.bank.save(self.stage3_dir / "proposal_bank_latest.pt")
        diagnostics = self.bank.diagnostics()
        _append_csv(
            self.stage3_dir / "proposal_bank_rounds.csv",
            {
                "condition": self.condition,
                "source_round": int(round_id),
                "target_round": int(round_id) + 1,
                "valid_update_count": diagnostics["valid_update_count"],
                "invalid_update_count": diagnostics["invalid_update_count"],
                "median_update_norm": diagnostics["median_update_norm"],
                "initial_cluster_count": diagnostics["initial_cluster_count"],
                "final_cluster_count": diagnostics["final_cluster_count"],
                "cluster_source_counts": json.dumps(
                    diagnostics["cluster_source_counts"], sort_keys=True
                ),
            },
            (
                "condition",
                "source_round",
                "target_round",
                "valid_update_count",
                "invalid_update_count",
                "median_update_norm",
                "initial_cluster_count",
                "final_cluster_count",
                "cluster_source_counts",
            ),
        )
        return self.bank

    def _log_probes(
        self,
        *,
        client_id: int,
        round_id: int,
        selection: ProposalSelection,
        audit_probes: Sequence[ProposalProbe],
    ) -> None:
        selected = set(selection.selected_proposal_ids)
        for view, probes in (("memory", selection.probes), ("audit", audit_probes)):
            for probe in probes:
                common = {
                    "condition": self.condition,
                    "round": int(round_id),
                    "client_id": int(client_id),
                    "view": view,
                    "selection_mode": selection.mode,
                    "proposal_id": int(probe.proposal_id),
                    "source_count": int(probe.source_count),
                    "selected": int(probe.proposal_id in selected),
                }
                rows = [("", probe.utility)] + list(probe.per_class_utility)
                for class_id, utility in rows:
                    _append_csv(
                        self.stage3_dir
                        / "client_local_research_audit"
                        / "proposal_utility.csv",
                        {
                            **common,
                            "class_id": class_id,
                            "utility": float(utility),
                        },
                        (
                            "condition",
                            "round",
                            "client_id",
                            "view",
                            "selection_mode",
                            "proposal_id",
                            "source_count",
                            "selected",
                            "class_id",
                            "utility",
                        ),
                    )

    def _log_client_runtime(
        self,
        prepared: PreparedStage3Client,
        restore: RestoreResult,
        norm_report: Mapping,
        *,
        ce_delta: torch.Tensor,
        upload: torch.Tensor,
        audit_local_ce: float | None,
        audit_final_ce: float | None,
    ) -> None:
        _append_csv(
            self.stage3_dir
            / "client_local_research_audit"
            / "runtime_clients.csv",
            {
                "condition": self.condition,
                "round": prepared.round_id,
                "client_id": prepared.client_id,
                "memory_size": prepared.evidence.size,
                "audit_size": 0 if prepared.audit is None else prepared.audit.size,
                "degradation": prepared.degradation,
                "proposal_count": len(prepared.selection.probes),
                "selected_count": len(prepared.selection.selected_proposal_ids),
                "selected_ids": json.dumps(prepared.selection.selected_proposal_ids),
                "restore_backward_count": restore.backward_count,
                "restore_gradient_norm": restore.gradient_norm,
                "ce_delta_norm": float(torch.linalg.vector_norm(ce_delta).item()),
                "upload_norm": float(torch.linalg.vector_norm(upload).item()),
                "norm_relative_error": norm_report["relative_norm_error"],
                "norm_fallback": norm_report["fallback"],
                "audit_incoming_ce": prepared.audit_incoming_ce,
                "audit_local_ce": audit_local_ce,
                "audit_final_ce": audit_final_ce,
            },
            (
                "condition",
                "round",
                "client_id",
                "memory_size",
                "audit_size",
                "degradation",
                "proposal_count",
                "selected_count",
                "selected_ids",
                "restore_backward_count",
                "restore_gradient_norm",
                "ce_delta_norm",
                "upload_norm",
                "norm_relative_error",
                "norm_fallback",
                "audit_incoming_ce",
                "audit_local_ce",
                "audit_final_ce",
            ),
        )
