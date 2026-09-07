"""Client-private functional realization feedback primitives and runtime.

The server-facing caller receives ordinary model states only. Functional
memory, class scores, and targets remain inside this simulation runtime and
are checkpointed as client-local state for deterministic smoke validation.
"""

from __future__ import annotations

import csv
import json
import math
import os
import random
import tempfile
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np
import torch
import torch.nn.functional as torch_f

from utils.stage3_private_state import ClientPrivateStateStore, stable_seed
from utils.stage3_runtime import ClientEvidenceRepository
from utils.stage3_vectors import build_model_lora_flat_spec


PFRF_CONDITIONS = (
    "ordinary_ce",
    "matched_memory_ce",
    "class_reweighted_ce",
    "instantaneous_target",
    "pfrf_max",
    "pfrf_add",
)
PFRF_SCHEMA = "pfrf_client_runtime_v1"
PFRF_CHECKPOINT_SCHEMA = "pfrf_federated_checkpoint_v1"


def _finite_probability(value: float, name: str) -> float:
    value = float(value)
    if not math.isfinite(value) or not 0.0 <= value <= 1.0:
        raise ValueError(f"{name} must be finite and in [0, 1], got {value}")
    return value


def max_target(previous: float, proposal: float) -> float:
    """Persistent absolute target used by the primary PFRF-Max condition."""
    return max(
        _finite_probability(previous, "previous"),
        _finite_probability(proposal, "proposal"),
    )


def additive_target(
    previous: float,
    incoming: float,
    proposal: float,
    *,
    cap: float = 1.0,
) -> tuple[float, float, float, float]:
    """Return capped additive target, old gap, fresh gain, and discarded mass."""
    previous = _finite_probability(previous, "previous")
    incoming = _finite_probability(incoming, "incoming")
    proposal = _finite_probability(proposal, "proposal")
    cap = _finite_probability(cap, "cap")
    old_gap = max(previous - incoming, 0.0)
    fresh_gain = max(proposal - incoming, 0.0)
    raw = incoming + old_gap + fresh_gain
    target = min(raw, cap)
    return target, old_gap, fresh_gain, max(raw - cap, 0.0)


def functional_gap(target: float, observed: float) -> float:
    return max(
        _finite_probability(target, "target")
        - _finite_probability(observed, "observed"),
        0.0,
    )


@contextmanager
def _fixed_rng(seed: int):
    python_state = random.getstate()
    numpy_state = np.random.get_state()
    torch_state = torch.random.get_rng_state()
    cuda_state = torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None
    random.seed(int(seed))
    np.random.seed(int(seed) % (2**32 - 1))
    torch.manual_seed(int(seed))
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(int(seed))
    try:
        yield
    finally:
        random.setstate(python_state)
        np.random.set_state(numpy_state)
        torch.random.set_rng_state(torch_state)
        if cuda_state is not None:
            torch.cuda.set_rng_state_all(cuda_state)


def capture_rng_state() -> dict:
    return {
        "python": random.getstate(),
        "numpy": np.random.get_state(),
        "torch": torch.random.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
    }


def restore_rng_state(value: Mapping) -> None:
    random.setstate(value["python"])
    np.random.set_state(value["numpy"])
    torch.random.set_rng_state(torch.as_tensor(value["torch"], dtype=torch.uint8))
    cuda = value.get("cuda")
    if cuda is not None:
        if not torch.cuda.is_available():
            raise RuntimeError("Checkpoint contains CUDA RNG state but CUDA is unavailable")
        torch.cuda.set_rng_state_all(cuda)


def _model_device(model) -> torch.device:
    try:
        return next(model.parameters()).device
    except StopIteration as error:
        raise ValueError("PFRF model has no parameters") from error


def class_probability_scores(
    model,
    images: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Return sorted classes, class-mean correct probabilities, and counts."""
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    device = _model_device(model)
    images = images.to(device)
    labels = labels.to(device=device, dtype=torch.long)
    logits = model(images)
    return class_probability_scores_from_logits(logits, labels, temperature=temperature)


def class_probability_scores_from_logits(
    logits: torch.Tensor,
    labels: torch.Tensor,
    *,
    temperature: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Class-mean correct probabilities for already-computed logits."""
    temperature = float(temperature)
    if not math.isfinite(temperature) or temperature <= 0:
        raise ValueError("temperature must be finite and positive")
    if logits.ndim != 2 or logits.shape[0] != labels.numel():
        raise ValueError("PFRF logits and labels do not align")
    probabilities = torch_f.softmax(logits.float() / temperature, dim=1)
    if bool((labels < 0).any()) or bool((labels >= probabilities.shape[1]).any()):
        raise ValueError("PFRF labels are outside the model class range")
    correct = probabilities.gather(1, labels[:, None]).squeeze(1)
    classes = torch.unique(labels, sorted=True)
    values = torch.stack([correct[labels == class_id].mean() for class_id in classes])
    counts = torch.stack([(labels == class_id).sum() for class_id in classes])
    return classes, values, counts


def _score_dict(classes: torch.Tensor, values: torch.Tensor) -> dict[int, float]:
    detached = values.detach().cpu().to(torch.float64)
    return {
        int(class_id): _finite_probability(value, "class score")
        for class_id, value in zip(classes.detach().cpu().tolist(), detached.tolist())
    }


def evaluate_class_probabilities(model, evidence, *, temperature: float) -> dict[int, float]:
    modes = {module: module.training for module in model.modules()}
    try:
        model.eval()
        with torch.no_grad():
            classes, values, _ = class_probability_scores(
                model,
                evidence.images,
                evidence.labels,
                temperature=temperature,
            )
        return _score_dict(classes, values)
    finally:
        for module, training in modes.items():
            module.training = training


@dataclass
class ClientTargetState:
    targets: dict[int, float] = field(default_factory=dict)
    last_participation_round: int = -1

    def as_dict(self) -> dict:
        return {
            "targets": {str(key): float(self.targets[key]) for key in sorted(self.targets)},
            "last_participation_round": int(self.last_participation_round),
        }

    @classmethod
    def from_dict(cls, value: Mapping) -> "ClientTargetState":
        return cls(
            targets={
                int(key): _finite_probability(item, "checkpoint target")
                for key, item in value.get("targets", {}).items()
            },
            last_participation_round=int(value.get("last_participation_round", -1)),
        )


class PFRFTargetStore:
    def __init__(self, *, condition: str, global_seed: int, num_classes: int):
        if condition not in PFRF_CONDITIONS:
            raise ValueError(f"Unknown PFRF condition: {condition}")
        self.condition = str(condition)
        self.global_seed = int(global_seed)
        self.num_classes = int(num_classes)
        self._clients: dict[int, ClientTargetState] = {}

    def client(self, client_id: int) -> ClientTargetState:
        return self._clients.setdefault(int(client_id), ClientTargetState())

    def state_dict(self) -> dict:
        return {
            "condition": self.condition,
            "global_seed": self.global_seed,
            "num_classes": self.num_classes,
            "clients": {
                str(client_id): self._clients[client_id].as_dict()
                for client_id in sorted(self._clients)
            },
        }

    @classmethod
    def from_state_dict(
        cls,
        value: Mapping,
        *,
        expected_condition: str,
        expected_global_seed: int,
        expected_num_classes: int,
    ) -> "PFRFTargetStore":
        store = cls(
            condition=str(value["condition"]),
            global_seed=int(value["global_seed"]),
            num_classes=int(value["num_classes"]),
        )
        if store.condition != str(expected_condition):
            raise ValueError("PFRF checkpoint condition mismatch")
        if store.global_seed != int(expected_global_seed):
            raise ValueError("PFRF checkpoint seed mismatch")
        if store.num_classes != int(expected_num_classes):
            raise ValueError("PFRF checkpoint class-count mismatch")
        store._clients = {
            int(key): ClientTargetState.from_dict(item)
            for key, item in value.get("clients", {}).items()
        }
        return store


@dataclass
class PFRFClientSession:
    runtime: "PFRFRuntime"
    client_id: int
    round_id: int
    evidence: object
    incoming: dict[int, float]
    target_before: dict[int, float]
    proposal: dict[int, float] | None = None
    correction_target: dict[int, float] | None = None
    discarded: dict[int, float] = field(default_factory=dict)
    upload: dict[int, float] | None = None
    completed: bool = False
    auxiliary_calls: int = 0

    @property
    def condition(self) -> str:
        return self.runtime.condition

    @property
    def proposal_epochs(self) -> int:
        return self.runtime.proposal_epochs

    def before_local_epoch(self, model, local_epoch: int) -> None:
        if int(local_epoch) != self.proposal_epochs or self.proposal is not None:
            return
        self.proposal = evaluate_class_probabilities(
            model, self.evidence, temperature=self.runtime.temperature
        )
        if set(self.proposal) != set(self.incoming):
            raise RuntimeError("PFRF proposal classes changed within a client round")

        state = self.runtime.targets.client(self.client_id)
        target = {}
        for class_id in sorted(self.incoming):
            f0 = self.incoming[class_id]
            fhat = self.proposal[class_id]
            previous = state.targets.get(class_id, f0)
            if self.condition == "instantaneous_target":
                target[class_id] = fhat
            elif self.condition == "pfrf_max":
                updated = max_target(previous, fhat)
                state.targets[class_id] = updated
                target[class_id] = updated
            elif self.condition == "pfrf_add":
                updated, _, _, discarded = additive_target(
                    previous, f0, fhat, cap=self.runtime.add_cap
                )
                state.targets[class_id] = updated
                target[class_id] = updated
                self.discarded[class_id] = discarded
        if self.condition in {"instantaneous_target", "pfrf_max", "pfrf_add"}:
            self.correction_target = target
        state.last_participation_round = self.round_id

    def uses_auxiliary(self, local_epoch: int) -> bool:
        return int(local_epoch) >= self.proposal_epochs and self.condition != "ordinary_ce"

    def auxiliary_loss(self, model) -> tuple[torch.Tensor, dict]:
        if self.proposal is None:
            raise RuntimeError("PFRF correction began before the proposal was frozen")
        device = _model_device(model)
        images = self.evidence.images.to(device)
        labels = self.evidence.labels.to(device=device, dtype=torch.long)
        logits = model(images)

        if self.condition == "matched_memory_ce":
            raw = torch_f.cross_entropy(logits.float(), labels, reduction="mean")
            active = 1.0
        elif self.condition == "class_reweighted_ce":
            counts = self.runtime.client_class_counts[self.client_id].to(
                device=device, dtype=torch.float32
            )
            sample_weights = torch.rsqrt(torch.clamp(counts[labels], min=1.0))
            sample_weights = sample_weights / sample_weights.mean().clamp_min(1e-12)
            raw = (
                torch_f.cross_entropy(logits.float(), labels, reduction="none")
                * sample_weights
            ).mean()
            active = 1.0
        elif self.condition in {"instantaneous_target", "pfrf_max", "pfrf_add"}:
            classes, values, _ = class_probability_scores_from_logits(
                logits,
                labels,
                temperature=self.runtime.temperature,
            )
            targets = torch.tensor(
                [self.correction_target[int(item)] for item in classes.detach().cpu().tolist()],
                device=values.device,
                dtype=values.dtype,
            )
            deficits = torch.relu(targets.detach() - values)
            raw = torch.mean(deficits.square())
            active = float((deficits.detach() > 0).float().mean().item())
        else:
            raw = torch.zeros((), device=device, dtype=torch.float32)
            active = 0.0

        if not bool(torch.isfinite(raw)):
            raise FloatingPointError("PFRF auxiliary loss is NaN or Inf")
        self.auxiliary_calls += 1
        return self.runtime.aux_lambda * raw, {
            "pfrf_aux_raw": float(raw.detach().item()),
            "pfrf_active_class_fraction": active,
        }


class PFRFRuntime:
    """Condition-local PFRF orchestration for ClipLoRA federated rounds."""

    def __init__(
        self,
        trainer,
        *,
        output_dir: str | Path,
        global_seed: int,
        condition: str,
        num_users: int,
        num_classes: int,
        client_class_counts,
        temperature: float = 1.0,
        aux_lambda: float = 1.0,
        proposal_epochs: int = 2,
        add_cap: float = 1.0,
        aggregation_scaling: float,
        resume_state: Mapping | None = None,
    ):
        if condition not in PFRF_CONDITIONS:
            raise ValueError(f"Unknown PFRF condition {condition!r}")
        self.condition = str(condition)
        self.global_seed = int(global_seed)
        self.num_users = int(num_users)
        self.num_classes = int(num_classes)
        self.temperature = float(temperature)
        self.aux_lambda = float(aux_lambda)
        self.proposal_epochs = int(proposal_epochs)
        self.add_cap = float(add_cap)
        self.aggregation_scaling = float(aggregation_scaling)
        if self.proposal_epochs < 1:
            raise ValueError("PFRF proposal_epochs must be positive")
        if not math.isfinite(self.aux_lambda) or self.aux_lambda < 0:
            raise ValueError("PFRF lambda must be finite and non-negative")
        _finite_probability(self.add_cap, "add_cap")
        if not math.isfinite(self.aggregation_scaling) or self.aggregation_scaling <= 0:
            raise ValueError("PFRF aggregation scaling must be finite and positive")
        self.output_dir = Path(output_dir)
        self.root = self.output_dir / "pfrf"
        self.root.mkdir(parents=True, exist_ok=True)
        self.spec = build_model_lora_flat_spec(trainer.model)
        self.client_class_counts = {
            int(client_id): torch.as_tensor(client_class_counts[client_id]).cpu().long()
            for client_id in range(self.num_users)
        }

        if resume_state is None:
            self.evidence_store = ClientPrivateStateStore(
                global_seed=self.global_seed,
                condition="pfrf_evidence",
                flatten_spec_hash=self.spec.spec_hash,
            )
            self.targets = PFRFTargetStore(
                condition=self.condition,
                global_seed=self.global_seed,
                num_classes=self.num_classes,
            )
        else:
            self._validate_runtime_state(resume_state)
            self.evidence_store = ClientPrivateStateStore.from_state_dict(
                resume_state["evidence_store"],
                expected_global_seed=self.global_seed,
                expected_condition="pfrf_evidence",
                expected_flatten_spec_hash=self.spec.spec_hash,
            )
            self.targets = PFRFTargetStore.from_state_dict(
                resume_state["targets"],
                expected_condition=self.condition,
                expected_global_seed=self.global_seed,
                expected_num_classes=self.num_classes,
            )
        self.evidence_repository = ClientEvidenceRepository(
            trainer, self.evidence_store, self.num_users
        )
        self._write_contract()

    def _validate_runtime_state(self, value: Mapping) -> None:
        if value.get("schema_version") != PFRF_SCHEMA:
            raise ValueError("Unsupported PFRF runtime checkpoint schema")
        expected = {
            "condition": self.condition,
            "global_seed": self.global_seed,
            "num_users": self.num_users,
            "num_classes": self.num_classes,
            "temperature": self.temperature,
            "aux_lambda": self.aux_lambda,
            "proposal_epochs": self.proposal_epochs,
            "add_cap": self.add_cap,
            "aggregation_scaling": self.aggregation_scaling,
            "lora_spec_hash": self.spec.spec_hash,
        }
        for key, wanted in expected.items():
            if value.get(key) != wanted:
                raise ValueError(
                    f"PFRF runtime checkpoint mismatch for {key}: "
                    f"{value.get(key)!r} != {wanted!r}"
                )

    def assert_model_spec(self, model) -> None:
        candidate = build_model_lora_flat_spec(model)
        if candidate.spec_hash != self.spec.spec_hash:
            raise ValueError("PFRF global/local LoRA specifications differ")

    def _write_contract(self) -> None:
        payload = {
            "schema_version": PFRF_SCHEMA,
            "condition": self.condition,
            "global_seed": self.global_seed,
            "num_users": self.num_users,
            "num_classes": self.num_classes,
            "temperature": self.temperature,
            "aux_lambda": self.aux_lambda,
            "proposal_epochs": self.proposal_epochs,
            "add_cap": self.add_cap,
            "aggregation_scaling": self.aggregation_scaling,
            "lora_spec_hash": self.spec.spec_hash,
            "functional_memory_size": 32,
            "server_payload": "ordinary LoRA state only",
        }
        (self.root / "runtime_contract.json").write_text(
            json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8"
        )

    def state_dict(self) -> dict:
        return {
            "schema_version": PFRF_SCHEMA,
            "condition": self.condition,
            "global_seed": self.global_seed,
            "num_users": self.num_users,
            "num_classes": self.num_classes,
            "temperature": self.temperature,
            "aux_lambda": self.aux_lambda,
            "proposal_epochs": self.proposal_epochs,
            "add_cap": self.add_cap,
            "aggregation_scaling": self.aggregation_scaling,
            "lora_spec_hash": self.spec.spec_hash,
            "evidence_store": self.evidence_store.state_dict(),
            "targets": self.targets.state_dict(),
        }

    def _evidence(self, client_id: int):
        seed = stable_seed("pfrf-fixed-memory", self.global_seed, int(client_id))
        with _fixed_rng(seed):
            batch = self.evidence_repository.batch(int(client_id), "memory")
        if batch is None:
            raise RuntimeError(f"Client {client_id} has no PFRF functional memory")
        return batch

    def prepare_client(self, trainer, *, client_id: int, round_id: int) -> PFRFClientSession:
        evidence = self._evidence(client_id)
        incoming = evaluate_class_probabilities(
            trainer.model, evidence, temperature=self.temperature
        )
        state = self.targets.client(client_id)
        target_before = {
            class_id: state.targets.get(class_id, incoming[class_id])
            for class_id in incoming
        }
        session = PFRFClientSession(
            runtime=self,
            client_id=int(client_id),
            round_id=int(round_id),
            evidence=evidence,
            incoming=incoming,
            target_before=target_before,
        )
        trainer.set_pfrf_session(session)
        return session

    def finalize_client(self, trainer, session: PFRFClientSession) -> None:
        if session.proposal is None:
            raise RuntimeError("PFRF local training did not enter its correction epoch")
        session.upload = evaluate_class_probabilities(
            trainer.model, session.evidence, temperature=self.temperature
        )
        session.completed = True
        trainer.clear_pfrf_session(session)

    def record_client_budget(
        self,
        trainer,
        session: PFRFClientSession,
        *,
        optimizer_steps: int,
        scheduler_steps: int,
    ) -> None:
        batches = len(trainer.fed_train_loader_x_dict[session.client_id])
        expected_optimizer = (self.proposal_epochs + 1) * batches
        expected_auxiliary = 0 if self.condition == "ordinary_ce" else batches
        if int(optimizer_steps) != expected_optimizer:
            raise RuntimeError(
                f"PFRF optimizer-step mismatch: {optimizer_steps} != {expected_optimizer}"
            )
        if int(scheduler_steps) != self.proposal_epochs + 1:
            raise RuntimeError(
                f"PFRF scheduler-step mismatch: {scheduler_steps} != {self.proposal_epochs + 1}"
            )
        if int(session.auxiliary_calls) != expected_auxiliary:
            raise RuntimeError(
                f"PFRF memory-backward mismatch: {session.auxiliary_calls} != "
                f"{expected_auxiliary}"
            )
        self._append_rows(
            self.root / "budget.csv",
            [
                {
                    "round": session.round_id + 1,
                    "client_id": session.client_id,
                    "condition": self.condition,
                    "train_batches": batches,
                    "optimizer_steps": int(optimizer_steps),
                    "scheduler_steps": int(scheduler_steps),
                    "memory_backward_calls": int(session.auxiliary_calls),
                }
            ],
        )

    def complete_round(self, model, sessions: Sequence[PFRFClientSession]) -> None:
        rows = []
        for session in sessions:
            if not session.completed or session.upload is None or session.proposal is None:
                raise RuntimeError("PFRF round contains an incomplete client session")
            next_global = evaluate_class_probabilities(
                model, session.evidence, temperature=self.temperature
            )
            classes = sorted(session.incoming)
            for class_id in classes:
                target_after = (
                    session.correction_target.get(class_id)
                    if session.correction_target is not None
                    else math.nan
                )
                before = session.target_before.get(class_id, math.nan)
                row = {
                    "round": session.round_id + 1,
                    "client_id": session.client_id,
                    "class_id": class_id,
                    "condition": self.condition,
                    "memory_count": int((session.evidence.labels == class_id).sum().item()),
                    "f_incoming": session.incoming[class_id],
                    "f_proposal": session.proposal[class_id],
                    "target_before": before,
                    "target_after": target_after,
                    "f_upload": session.upload[class_id],
                    "f_next_global": next_global[class_id],
                    "gap_before": (
                        functional_gap(before, session.incoming[class_id])
                        if math.isfinite(before) else math.nan
                    ),
                    "correction_deficit": (
                        functional_gap(target_after, session.proposal[class_id])
                        if math.isfinite(target_after) else math.nan
                    ),
                    "gap_after": (
                        functional_gap(target_after, next_global[class_id])
                        if math.isfinite(target_after) else math.nan
                    ),
                    "discarded_target": session.discarded.get(class_id, 0.0),
                }
                rows.append(row)
        self._append_rows(self.root / "client_functional_state.csv", rows)

    @staticmethod
    def _append_rows(path: Path, rows: list[dict]) -> None:
        if not rows:
            return
        exists = path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            if not exists:
                writer.writeheader()
            writer.writerows(rows)


def _atomic_torch_save(value, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=str(path.parent)
    )
    os.close(descriptor)
    temporary_path = Path(temporary)
    try:
        torch.save(value, temporary_path)
        os.replace(temporary_path, path)
    finally:
        if temporary_path.exists():
            temporary_path.unlink()


def save_pfrf_checkpoint(
    output_dir: str | Path,
    *,
    completed_round: int,
    global_state: Mapping[str, torch.Tensor],
    runtime: PFRFRuntime,
    rng_state: Mapping,
    aggregation_scaling: float,
) -> Path:
    root = Path(output_dir) / "pfrf"
    if not math.isclose(
        float(aggregation_scaling),
        float(runtime.aggregation_scaling),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("Checkpoint scaling differs from PFRF runtime scaling")
    payload = {
        "schema_version": PFRF_CHECKPOINT_SCHEMA,
        "completed_round": int(completed_round),
        "global_lora_state": {
            key: global_state[key].detach().cpu().clone() for key in runtime.spec.names
        },
        "runtime": runtime.state_dict(),
        "rng_state": dict(rng_state),
        "aggregation_scaling": float(aggregation_scaling),
    }
    round_path = root / f"checkpoint_round_{int(completed_round):03d}.pt"
    latest_path = root / "checkpoint_latest.pt"
    _atomic_torch_save(payload, round_path)
    _atomic_torch_save(payload, latest_path)
    return latest_path


def load_pfrf_checkpoint(
    resume_dir: str | Path,
    *,
    expected_scaling: float,
) -> dict:
    path = Path(resume_dir) / "pfrf" / "checkpoint_latest.pt"
    if not path.is_file():
        raise FileNotFoundError(f"PFRF resume checkpoint does not exist: {path}")
    try:
        payload = torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:
        payload = torch.load(path, map_location="cpu")
    if payload.get("schema_version") != PFRF_CHECKPOINT_SCHEMA:
        raise ValueError("Unsupported PFRF federated checkpoint schema")
    if not math.isclose(
        float(payload.get("aggregation_scaling", math.nan)),
        float(expected_scaling),
        rel_tol=0.0,
        abs_tol=1e-12,
    ):
        raise ValueError("PFRF checkpoint aggregation scaling mismatch")
    return payload
