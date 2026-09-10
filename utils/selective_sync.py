"""First-version functional selective synchronization for federated ClipLoRA."""

from __future__ import annotations

import csv
import json
import math
import random
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping

import numpy as np
import torch
import torch.nn.functional as F
from scipy.optimize import minimize


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


@dataclass(frozen=True)
class FunctionalBatch:
    images: torch.Tensor
    labels: torch.Tensor


class FunctionalMemoryRepository:
    """A small class-balanced memory selected from each client's train data."""

    def __init__(self, trainer, num_users: int, memory_size: int, seed: int):
        self.seed = int(seed)
        self.positions = {}
        self.datasets = {}
        for client_id in range(int(num_users)):
            dataset = trainer.fed_train_loader_x_dict[client_id].dataset
            by_class = {}
            for position, item in enumerate(dataset.data_source):
                label = int(item["label"] if isinstance(item, Mapping) else item.label)
                by_class.setdefault(label, []).append(position)

            rng = random.Random(self.seed + 1009 * client_id)
            classes = list(by_class)
            rng.shuffle(classes)
            for values in by_class.values():
                rng.shuffle(values)

            chosen = []
            cursor = {class_id: 0 for class_id in classes}
            limit = min(int(memory_size), len(dataset))
            while len(chosen) < limit:
                changed = False
                for class_id in classes:
                    index = cursor[class_id]
                    if index >= len(by_class[class_id]):
                        continue
                    chosen.append(by_class[class_id][index])
                    cursor[class_id] += 1
                    changed = True
                    if len(chosen) == limit:
                        break
                if not changed:
                    break
            self.datasets[client_id] = dataset
            self.positions[client_id] = tuple(chosen)

    def batch(self, client_id: int) -> FunctionalBatch:
        client_id = int(client_id)
        images, labels = [], []
        with _fixed_rng(self.seed + 1000003 * client_id):
            for position in self.positions[client_id]:
                item = self.datasets[client_id][position]
                images.append(item["img"].detach().cpu())
                labels.append(int(item["label"]))
        return FunctionalBatch(
            images=torch.stack(images, dim=0),
            labels=torch.tensor(labels, dtype=torch.long),
        )


def _model_device(model) -> torch.device:
    return next(model.parameters()).device


def _class_metric_tensors(model, batch: FunctionalBatch, temperature: float):
    images = batch.images.to(_model_device(model))
    labels = batch.labels.to(device=images.device, dtype=torch.long)
    logits = model(images)
    probabilities = F.softmax(logits.float() / float(temperature), dim=1)
    correct = probabilities.gather(1, labels[:, None]).squeeze(1)
    true_logits = logits.float().gather(1, labels[:, None]).squeeze(1)
    competitors = logits.float().clone()
    competitors.scatter_(1, labels[:, None], -torch.inf)
    hardest_negative = competitors.max(dim=1).values
    logit_scale = logits.float().std(dim=1, unbiased=False).clamp_min(1e-6)
    normalized_margin = (true_logits - hardest_negative) / logit_scale
    classes = torch.unique(labels, sorted=True)
    probability_values = {
        int(class_id.item()): correct[labels == class_id].mean()
        for class_id in classes
    }
    margin_values = {
        int(class_id.item()): normalized_margin[labels == class_id].mean()
        for class_id in classes
    }
    return probability_values, margin_values


def _class_probability_tensors(model, batch: FunctionalBatch, temperature: float):
    probabilities, _ = _class_metric_tensors(model, batch, temperature)
    return probabilities


def evaluate_class_probabilities(model, batch: FunctionalBatch, temperature: float):
    probabilities, _ = evaluate_class_metrics(model, batch, temperature)
    return probabilities


def evaluate_class_metrics(model, batch: FunctionalBatch, temperature: float):
    training = model.training
    model.eval()
    try:
        with torch.no_grad():
            probabilities, margins = _class_metric_tensors(model, batch, temperature)
        return (
            {class_id: float(value.item()) for class_id, value in probabilities.items()},
            {class_id: float(value.item()) for class_id, value in margins.items()},
        )
    finally:
        model.train(training)


def _set_b_state(model, values: Mapping[str, torch.Tensor]) -> None:
    parameters = dict(model.named_parameters())
    with torch.no_grad():
        for key, value in values.items():
            parameters[key].copy_(value.to(parameters[key].device, parameters[key].dtype))


def _capture_b_state(model, keys) -> dict[str, torch.Tensor]:
    parameters = dict(model.named_parameters())
    return {key: parameters[key].detach().cpu().clone() for key in keys}


def _flatten(values: Mapping[str, torch.Tensor], keys) -> torch.Tensor:
    return torch.cat([values[key].detach().cpu().double().reshape(-1) for key in keys])


def _unflatten(vector: torch.Tensor, references: Mapping[str, torch.Tensor], keys):
    output = {}
    offset = 0
    for key in keys:
        reference = references[key]
        count = reference.numel()
        output[key] = vector[offset : offset + count].reshape(reference.shape).to(reference.dtype)
        offset += count
    return output


def _add_direction(base, direction, scale: float, keys):
    return {
        key: base[key] + float(scale) * direction[key].to(base[key].dtype)
        for key in keys
    }


def class_gradients(model, batch, class_ids, b_keys, temperature):
    training = model.training
    model.eval()
    try:
        values = _class_probability_tensors(model, batch, temperature)
        parameters = dict(model.named_parameters())
        selected_parameters = [parameters[key] for key in b_keys]
        rows = []
        for order, class_id in enumerate(class_ids):
            gradients = torch.autograd.grad(
                values[int(class_id)],
                selected_parameters,
                retain_graph=order + 1 < len(class_ids),
            )
            rows.append(
                torch.cat([item.detach().cpu().double().reshape(-1) for item in gradients])
            )
        return torch.stack(rows, dim=0)
    finally:
        model.train(training)


def project_direction(raw_direction, gradient_rows, priorities, slack_mu):
    """Solve the Top-L soft constraints through their small non-negative dual."""
    raw = raw_direction.detach().cpu().double()
    gradients = gradient_rows.detach().cpu().double()
    priority = torch.as_tensor(priorities, dtype=torch.float64).clamp_min(1e-12)
    hessian = gradients @ gradients.T
    hessian += torch.diag(1.0 / (2.0 * float(slack_mu) * priority))
    linear = gradients @ raw
    hessian_np = hessian.numpy()
    linear_np = linear.numpy()

    def objective(value):
        return 0.5 * value.dot(hessian_np @ value) + linear_np.dot(value)

    def jacobian(value):
        return hessian_np @ value + linear_np

    result = minimize(
        objective,
        np.zeros(len(priorities), dtype=np.float64),
        jac=jacobian,
        bounds=[(0.0, None)] * len(priorities),
        method="L-BFGS-B",
    )
    dual = torch.from_numpy(result.x).double()
    return raw + gradients.T @ dual


def fixed_a_gradient_capture(model, batch: FunctionalBatch, b_keys):
    """Measure how much unconstrained weight-gradient energy lies in row(A)."""
    modules = dict(model.named_modules())
    pairs = []
    for b_key in b_keys:
        module_name = b_key.rpartition(".")[0]
        module = modules[module_name]
        pairs.append((module_name, module.weight, module.w_lora_A))

    training = model.training
    model.eval()
    original_requires_grad = [weight.requires_grad for _, weight, _ in pairs]
    for _, weight, _ in pairs:
        weight.requires_grad_(True)
    try:
        images = batch.images.to(_model_device(model))
        labels = batch.labels.to(device=images.device, dtype=torch.long)
        loss = F.cross_entropy(model(images).float(), labels)
        gradients = torch.autograd.grad(loss, [weight for _, weight, _ in pairs])
        rows = []
        for (module_name, _, a), gradient in zip(pairs, gradients):
            work_gradient = gradient.detach().float()
            work_a = a.detach().float()
            gram_inverse = torch.linalg.pinv(work_a @ work_a.T)
            projected = (work_gradient @ work_a.T @ gram_inverse) @ work_a
            full_energy = float(torch.sum(work_gradient.square()).item())
            projected_energy = float(torch.sum(projected.square()).item())
            rows.append(
                {
                    "module": module_name,
                    "full_gradient_energy": full_energy,
                    "projected_gradient_energy": projected_energy,
                    "capture_ratio": projected_energy / max(full_energy, 1e-30),
                    "a_rank": int(torch.linalg.matrix_rank(work_a).item()),
                }
            )
        return rows
    finally:
        for (_, weight, _), requires_grad in zip(pairs, original_requires_grad):
            weight.requires_grad_(requires_grad)
        model.train(training)


@dataclass
class SelectiveSyncSession:
    runtime: "SelectiveSyncRuntime"
    client_id: int
    round_id: int
    evidence: FunctionalBatch
    private_scores: dict[int, float]
    private_margins: dict[int, float]
    raw_scores: dict[int, float]
    raw_margins: dict[int, float]
    projected_scores: dict[int, float]
    projected_margins: dict[int, float]
    accepted_scores: dict[int, float]
    accepted_margins: dict[int, float]
    raw_harm: dict[int, float]
    selected_classes: tuple[int, ...]
    accepted_scale: float
    backtracks: int
    raw_direction_norm: float
    safe_direction_norm: float
    received_direction_norm: float
    postlocal_scores: dict[int, float] | None = None
    postlocal_margins: dict[int, float] | None = None

    def before_local_epoch(self, model, local_epoch: int) -> None:
        return None

    def uses_auxiliary(self, local_epoch: int) -> bool:
        return (
            self.runtime.local_loss == "class_reweighted_memory_ce"
            and int(local_epoch) >= self.runtime.memory_start_epoch
        )

    def auxiliary_loss(self, model):
        images = self.evidence.images.to(_model_device(model))
        labels = self.evidence.labels.to(device=images.device, dtype=torch.long)
        logits = model(images)
        counts = self.runtime.client_class_counts[self.client_id].to(
            device=images.device, dtype=torch.float32
        )
        weights = torch.rsqrt(torch.clamp(counts[labels], min=1.0))
        weights = weights / weights.mean().clamp_min(1e-12)
        loss = (F.cross_entropy(logits.float(), labels, reduction="none") * weights).mean()
        return self.runtime.memory_lambda * loss, {
            "selective_sync_memory_loss": float(loss.detach().item())
        }


class SelectiveSyncRuntime:
    def __init__(
        self,
        trainer,
        *,
        output_dir,
        global_state,
        num_users,
        client_class_counts,
        client_sample_counts,
        seed=42,
        memory_size=32,
        temperature=1.0,
        receive_ratio=1.0,
        top_l=5,
        harm_epsilon=1e-4,
        slack_mu=100.0,
        success_kappa=0.5,
        backtrack_factor=0.5,
        max_backtracks=8,
        local_loss="ordinary_ce",
        memory_lambda=1.0,
        memory_start_epoch=2,
        capacity_rounds="1,20,50,100",
    ):
        self.root = Path(output_dir) / "selective_sync"
        self.root.mkdir(parents=True, exist_ok=True)
        self.temperature = float(temperature)
        self.receive_ratio = float(receive_ratio)
        self.top_l = int(top_l)
        self.harm_epsilon = float(harm_epsilon)
        self.slack_mu = float(slack_mu)
        self.success_kappa = float(success_kappa)
        self.backtrack_factor = float(backtrack_factor)
        self.max_backtracks = int(max_backtracks)
        self.local_loss = str(local_loss)
        self.memory_lambda = float(memory_lambda)
        self.memory_start_epoch = int(memory_start_epoch)
        self.client_class_counts = {
            int(client_id): torch.as_tensor(values).cpu().float()
            for client_id, values in client_class_counts.items()
        }
        self.client_sample_counts = {
            client_id: float(client_sample_counts[client_id])
            for client_id in range(int(num_users))
        }
        self.global_class_counts = sum(self.client_class_counts.values())
        tail_count = max(int(round(0.2 * len(self.global_class_counts))), 1)
        self.tail_classes = set(
            torch.argsort(self.global_class_counts)[:tail_count].tolist()
        )
        self.capacity_rounds = {
            int(value.strip())
            for value in str(capacity_rounds).split(",")
            if value.strip()
        }
        self.b_keys = tuple(
            sorted(name for name, _ in trainer.model.named_parameters() if name.endswith("_lora_B"))
        )
        self.private_b = {
            client_id: {
                key: global_state[key].detach().cpu().clone() for key in self.b_keys
            }
            for client_id in range(int(num_users))
        }
        self.previous_global = _flatten(
            {key: global_state[key] for key in self.b_keys}, self.b_keys
        )
        self.previous_global_step = None
        self.memory = FunctionalMemoryRepository(
            trainer, num_users=num_users, memory_size=memory_size, seed=seed
        )
        (self.root / "config.json").write_text(
            json.dumps(
                {
                    "seed": int(seed),
                    "memory_size": int(memory_size),
                    "temperature": self.temperature,
                    "receive_ratio": self.receive_ratio,
                    "top_l": self.top_l,
                    "harm_epsilon": self.harm_epsilon,
                    "slack_mu": self.slack_mu,
                    "success_kappa": self.success_kappa,
                    "backtrack_factor": self.backtrack_factor,
                    "max_backtracks": self.max_backtracks,
                    "local_loss": self.local_loss,
                    "memory_lambda": self.memory_lambda,
                    "memory_start_epoch": self.memory_start_epoch,
                    "capacity_rounds": sorted(self.capacity_rounds),
                    "normalized_margin": "(true_logit - max_other_logit) / per_sample_logit_std",
                    "server_aggregation": "sample_weighted_fedavg",
                    "trainable_lora_factor": "B_only",
                },
                indent=2,
            )
            + "\n",
            encoding="utf-8",
        )

    def prepare_client(self, trainer, global_state, *, client_id: int, round_id: int):
        client_id = int(client_id)
        model = trainer.model
        model.load_state_dict(global_state, strict=True)
        private = self.private_b[client_id]
        _set_b_state(model, private)
        evidence = self.memory.batch(client_id)
        if int(round_id) + 1 in self.capacity_rounds:
            capacity_rows = fixed_a_gradient_capture(model, evidence, self.b_keys)
            for row in capacity_rows:
                row.update({"round": int(round_id) + 1, "client_id": client_id})
            self._append(self.root / "fixed_a_capacity.csv", capacity_rows)
        private_scores, private_margins = evaluate_class_metrics(
            model, evidence, self.temperature
        )

        raw_direction = {
            key: self.receive_ratio
            * (global_state[key].detach().cpu() - private[key])
            for key in self.b_keys
        }
        raw_state = _add_direction(private, raw_direction, 1.0, self.b_keys)
        _set_b_state(model, raw_state)
        raw_scores, raw_margins = evaluate_class_metrics(
            model, evidence, self.temperature
        )
        raw_harm = {
            class_id: max(private_scores[class_id] - raw_scores[class_id], 0.0)
            for class_id in private_scores
        }
        selected = tuple(
            class_id
            for class_id, harm in sorted(raw_harm.items(), key=lambda item: (-item[1], item[0]))
            if harm > self.harm_epsilon
        )[: self.top_l]

        raw_flat = _flatten(raw_direction, self.b_keys)
        if selected:
            _set_b_state(model, private)
            gradients = class_gradients(
                model, evidence, selected, self.b_keys, self.temperature
            )
            selected_harm = torch.tensor([raw_harm[c] for c in selected], dtype=torch.float64)
            priorities = selected_harm / selected_harm.mean().clamp_min(1e-12)
            safe_flat = project_direction(raw_flat, gradients, priorities, self.slack_mu)
            safe_direction = _unflatten(safe_flat, private, self.b_keys)
            _set_b_state(model, _add_direction(private, safe_direction, 1.0, self.b_keys))
            projected_scores, projected_margins = evaluate_class_metrics(
                model, evidence, self.temperature
            )

            scale = 1.0
            backtracks = 0
            accepted_scores = projected_scores
            accepted_margins = projected_margins
            while True:
                passed = all(
                    max(private_scores[c] - accepted_scores[c], 0.0)
                    <= self.success_kappa * raw_harm[c]
                    for c in selected
                )
                if passed:
                    break
                backtracks += 1
                if backtracks > self.max_backtracks:
                    scale = 0.0
                    _set_b_state(model, private)
                    accepted_scores = dict(private_scores)
                    accepted_margins = dict(private_margins)
                    break
                scale *= self.backtrack_factor
                candidate = _add_direction(private, safe_direction, scale, self.b_keys)
                _set_b_state(model, candidate)
                accepted_scores, accepted_margins = evaluate_class_metrics(
                    model, evidence, self.temperature
                )
        else:
            safe_direction = raw_direction
            projected_scores = dict(raw_scores)
            projected_margins = dict(raw_margins)
            accepted_scores = dict(raw_scores)
            accepted_margins = dict(raw_margins)
            scale = 1.0
            backtracks = 0
            _set_b_state(model, raw_state)

        session = SelectiveSyncSession(
            runtime=self,
            client_id=client_id,
            round_id=int(round_id),
            evidence=evidence,
            private_scores=private_scores,
            private_margins=private_margins,
            raw_scores=raw_scores,
            raw_margins=raw_margins,
            projected_scores=projected_scores,
            projected_margins=projected_margins,
            accepted_scores=accepted_scores,
            accepted_margins=accepted_margins,
            raw_harm=raw_harm,
            selected_classes=selected,
            accepted_scale=float(scale),
            backtracks=int(backtracks),
            raw_direction_norm=float(torch.linalg.vector_norm(raw_flat).item()),
            safe_direction_norm=float(torch.linalg.vector_norm(_flatten(safe_direction, self.b_keys)).item()),
            received_direction_norm=float(
                torch.linalg.vector_norm(float(scale) * _flatten(safe_direction, self.b_keys)).item()
            ),
        )
        trainer.set_pfrf_session(session)
        return session

    def finalize_client(self, trainer, session: SelectiveSyncSession) -> None:
        session.postlocal_scores, session.postlocal_margins = evaluate_class_metrics(
            trainer.model, session.evidence, self.temperature
        )
        self.private_b[session.client_id] = _capture_b_state(trainer.model, self.b_keys)
        trainer.clear_pfrf_session(session)

    def complete_round(self, global_model, sessions, aggregation_weights) -> None:
        class_rows = []
        summary_rows = []
        for session in sessions:
            next_global, next_global_margins = evaluate_class_metrics(
                global_model, session.evidence, self.temperature
            )
            accepted_harms = {
                class_id: max(
                    session.private_scores[class_id] - session.accepted_scores[class_id], 0.0
                )
                for class_id in session.private_scores
            }
            for class_id in sorted(session.private_scores):
                class_rows.append(
                    {
                        "round": session.round_id + 1,
                        "client_id": session.client_id,
                        "class_id": class_id,
                        "memory_count": int((session.evidence.labels == class_id).sum().item()),
                        "selected": int(class_id in session.selected_classes),
                        "private_score": session.private_scores[class_id],
                        "private_margin": session.private_margins[class_id],
                        "raw_score": session.raw_scores[class_id],
                        "raw_margin": session.raw_margins[class_id],
                        "raw_harm": session.raw_harm[class_id],
                        "raw_margin_harm": max(
                            session.private_margins[class_id] - session.raw_margins[class_id],
                            0.0,
                        ),
                        "projected_score": session.projected_scores[class_id],
                        "projected_margin": session.projected_margins[class_id],
                        "accepted_score": session.accepted_scores[class_id],
                        "accepted_margin": session.accepted_margins[class_id],
                        "accepted_harm": accepted_harms[class_id],
                        "postlocal_score": session.postlocal_scores[class_id],
                        "postlocal_margin": session.postlocal_margins[class_id],
                        "next_global_score": next_global[class_id],
                        "next_global_margin": next_global_margins[class_id],
                    }
                )
            count = max(len(session.private_scores), 1)
            selected_success = [
                accepted_harms[c] <= self.success_kappa * session.raw_harm[c]
                for c in session.selected_classes
            ]
            summary_rows.append(
                {
                    "round": session.round_id + 1,
                    "client_id": session.client_id,
                    "local_loss": self.local_loss,
                    "memory_class_count": len(session.private_scores),
                    "raw_harmed_class_count": sum(
                        harm > self.harm_epsilon for harm in session.raw_harm.values()
                    ),
                    "selected_class_count": len(session.selected_classes),
                    "selected_success_rate": (
                        sum(selected_success) / len(selected_success) if selected_success else 1.0
                    ),
                    "mean_raw_harm": sum(session.raw_harm.values()) / count,
                    "mean_accepted_harm": sum(accepted_harms.values()) / count,
                    "private_mean_score": sum(session.private_scores.values()) / count,
                    "raw_mean_score": sum(session.raw_scores.values()) / count,
                    "accepted_mean_score": sum(session.accepted_scores.values()) / count,
                    "postlocal_mean_score": sum(session.postlocal_scores.values()) / count,
                    "next_global_mean_score": sum(next_global.values()) / count,
                    "accepted_scale": session.accepted_scale,
                    "backtracks": session.backtracks,
                    "raw_direction_norm": session.raw_direction_norm,
                    "safe_direction_norm": session.safe_direction_norm,
                    "received_direction_norm": session.received_direction_norm,
                }
            )
        self._append(self.root / "class_functional_metrics.csv", class_rows)
        self._append(self.root / "client_sync_metrics.csv", summary_rows)
        self._record_server_state(global_model, aggregation_weights, sessions[0].round_id + 1)

    def _record_server_state(self, global_model, aggregation_weights, round_id) -> None:
        global_b = _capture_b_state(global_model, self.b_keys)
        global_vector = _flatten(global_b, self.b_keys)
        private_vectors = torch.stack(
            [_flatten(self.private_b[client_id], self.b_keys) for client_id in sorted(self.private_b)]
        )
        distances = torch.linalg.vector_norm(private_vectors - global_vector[None, :], dim=1)
        pairwise = torch.pdist(private_vectors)
        normalized_private = F.normalize(private_vectors, dim=1, eps=1e-12)
        cosine = normalized_private @ normalized_private.T
        upper = cosine[torch.triu_indices(cosine.shape[0], cosine.shape[1], offset=1).unbind()]
        global_step = global_vector - self.previous_global
        step_norm = float(torch.linalg.vector_norm(global_step).item())
        previous_cosine = math.nan
        if self.previous_global_step is not None:
            denominator = float(
                torch.linalg.vector_norm(global_step).item()
                * torch.linalg.vector_norm(self.previous_global_step).item()
            )
            if denominator > 0:
                previous_cosine = float(
                    torch.dot(global_step, self.previous_global_step).item() / denominator
                )
        weights = torch.tensor(
            [float(aggregation_weights[client_id]) for client_id in sorted(self.private_b)],
            dtype=torch.float64,
        )
        disagreement = torch.sqrt(
            torch.sum(weights * torch.sum((private_vectors - global_vector[None, :]) ** 2, dim=1))
        )
        b_ranks = []
        for client_id in sorted(self.private_b):
            b_ranks.extend(
                int(torch.linalg.matrix_rank(self.private_b[client_id][key].float()).item())
                for key in self.b_keys
            )
        neff = 1.0 / float(torch.sum(weights.square()).item())
        self._append(
            self.root / "round_state_metrics.csv",
            [
                {
                    "round": int(round_id),
                    "global_b_norm": float(torch.linalg.vector_norm(global_vector).item()),
                    "global_step_norm": step_norm,
                    "global_step_cosine_previous": previous_cosine,
                    "mean_private_global_distance": float(distances.mean().item()),
                    "max_private_global_distance": float(distances.max().item()),
                    "rms_private_global_disagreement": float(disagreement.item()),
                    "mean_pairwise_private_distance": float(pairwise.mean().item()),
                    "max_pairwise_private_distance": float(pairwise.max().item()),
                    "mean_pairwise_private_cosine": float(upper.mean().item()),
                    "mean_private_b_rank": float(np.mean(b_ranks)),
                    "effective_client_count": neff,
                }
            ],
        )
        access_rows = []
        for class_id in range(len(self.global_class_counts)):
            access = sum(
                float(aggregation_weights[client_id])
                for client_id in aggregation_weights
                if float(self.client_class_counts[int(client_id)][class_id].item()) > 0
            )
            access_rows.append(
                {
                    "round": int(round_id),
                    "class_id": class_id,
                    "global_class_count": float(self.global_class_counts[class_id].item()),
                    "tail20": int(class_id in self.tail_classes),
                    "supporter_access": access,
                    "effective_client_count": neff,
                }
            )
        self._append(self.root / "server_access_metrics.csv", access_rows)
        self.previous_global = global_vector
        self.previous_global_step = global_step

    @staticmethod
    def _append(path: Path, rows) -> None:
        if not rows:
            return
        exists = path.exists()
        with path.open("a", encoding="utf-8", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            if not exists:
                writer.writeheader()
            writer.writerows(rows)
