"""Train-only functional ERI attribution for frozen round dumps."""

from __future__ import annotations

import csv
import json
import math
import sys
from pathlib import Path
from types import SimpleNamespace
from typing import Mapping, Sequence

import torch

from tools.eri_closure.attribution import (
    completeness_record,
    first_order_client_effects,
    integrated_client_effects,
    rows_from_effects,
)
from tools.eri_closure.protocol import load_protocol
from utils.cusp_minimal import FlatSpec, flatten_state, unflatten_state, write_csv, write_json
from utils.functional_coverage_validation import _TrainOnlyCifar100, _locate_cifar100


def load_round_dump(dump_dir: str | Path) -> tuple[dict, dict]:
    root = Path(dump_dir)
    payload = torch.load(root / "round_state.pt", map_location="cpu", weights_only=False)
    metadata = json.loads((root / "metadata.json").read_text(encoding="utf-8"))
    required = {
        "flatten_spec", "global_before_trainable",
        "local_trainable_states", "selected_client_ids", "client_class_counts",
    }
    missing = required - set(payload)
    if not any(key in payload for key in ("global_after_trainable", "global_after_fedavg_trainable")):
        missing.add("global_after_trainable")
    if missing:
        raise ValueError(f"Round dump lacks ERI-required fields {sorted(missing)}: {root}")
    return payload, metadata


def build_trainer_from_metadata(metadata: Mapping, output_dir: str | Path):
    """Rebuild the exact trainer without entering a training/update loop."""
    from Dassl.dassl.engine import build_trainer
    from federated_main import setup_cfg

    resolved = dict(metadata.get("resolved_args", {}))
    if not resolved:
        raise ValueError("ERI attribution requires metadata.resolved_args to rebuild the model")
    args = SimpleNamespace(**resolved)
    args.output_dir = str(output_dir)
    cfg = setup_cfg(args)
    trainer = build_trainer(cfg)
    trainer.fed_before_train(is_global=True)
    return cfg, trainer


class TrainOnlyFunctionalEvaluator:
    """Differentiable class log-odds on a frozen train-only probe manifest."""

    def __init__(
        self,
        *,
        cfg,
        trainer,
        payload: Mapping,
        protocol_dir: str | Path,
        data_root: str | Path | None = None,
        device: str | None = None,
        batch_size: int = 100,
    ):
        from Dassl.dassl.data.transforms import build_transform

        protocol, rows = load_protocol(protocol_dir)
        self.protocol = protocol
        self.class_ids = [int(item) for item in protocol["tail_class_ids"]]
        self.model = trainer.model
        self.model.eval()
        self.spec = FlatSpec.from_dict(payload["flatten_spec"])
        self.parameter_by_name = dict(self.model.named_parameters())
        missing = [key for key in self.spec.keys if key not in self.parameter_by_name]
        if missing:
            raise RuntimeError(
                "Rebuilt model parameter names do not match the frozen dump; "
                f"first missing keys={missing[:5]}"
            )
        if device:
            self.model.to(torch.device(device))
        self.device = next(self.model.parameters()).device
        self.batch_size = int(batch_size)
        data_dir = _locate_cifar100(Path(data_root or cfg.DATASET.ROOT))
        store = _TrainOnlyCifar100(data_dir)
        transform = build_transform(cfg, is_train=False)
        by_class: dict[int, list[torch.Tensor]] = {class_id: [] for class_id in self.class_ids}
        for row in rows:
            class_id = int(row["class_id"])
            if class_id not in by_class:
                continue
            by_class[class_id].append(transform(store.image(int(row["raw_train_index"]))))
        self.images = {
            class_id: torch.stack(values).to(self.device)
            for class_id, values in by_class.items()
            if values
        }
        missing_probes = sorted(set(self.class_ids) - set(self.images))
        if missing_probes:
            raise RuntimeError(f"Protocol manifest has no probes for tail classes {missing_probes}")

    def _load_vector(self, vector: torch.Tensor) -> None:
        state = unflatten_state(vector.detach().cpu(), self.spec)
        result = self.model.load_state_dict(state, strict=False)
        unexpected = [key for key in result.unexpected_keys if key in self.spec.keys]
        if unexpected:
            raise RuntimeError(f"Frozen ERI state has unexpected model keys: {unexpected}")

    def _log_odds(self, class_id: int) -> torch.Tensor:
        images = self.images[int(class_id)]
        terms = []
        for start in range(0, images.shape[0], self.batch_size):
            logits = self.model(images[start : start + self.batch_size])
            if not isinstance(logits, torch.Tensor):
                logits = logits[0]
            target = logits[:, int(class_id)]
            other = logits.clone()
            other[:, int(class_id)] = -torch.inf
            terms.append(target - torch.logsumexp(other, dim=1))
        return torch.cat(terms).mean()

    def metric(self, vector: torch.Tensor, class_id: int) -> float:
        self._load_vector(vector)
        with torch.no_grad():
            return float(self._log_odds(class_id).detach().cpu().item())

    def gradient(self, vector: torch.Tensor, class_id: int) -> torch.Tensor:
        self._load_vector(vector)
        self.model.zero_grad(set_to_none=True)
        objective = self._log_odds(class_id)
        params = [self.parameter_by_name[key] for key in self.spec.keys]
        grads = torch.autograd.grad(objective, params, allow_unused=True)
        chunks = [
            (torch.zeros_like(param) if grad is None else grad).detach().cpu().to(torch.float64).reshape(-1)
            for param, grad in zip(params, grads)
        ]
        return torch.cat(chunks) if chunks else torch.empty(0, dtype=torch.float64)


def payload_vectors(payload: Mapping) -> tuple[FlatSpec, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    spec = FlatSpec.from_dict(payload["flatten_spec"])
    before = flatten_state(payload["global_before_trainable"], spec)
    # Native ERI dumps use the aggregation-agnostic spelling. Accept the old
    # FedAvg-specific alias so existing artifacts remain analyzable.
    after_state = payload.get("global_after_trainable")
    if after_state is None:
        after_state = payload["global_after_fedavg_trainable"]
    after = flatten_state(after_state, spec)
    local = torch.stack([flatten_state(state, spec) for state in payload["local_trainable_states"]])
    deltas = local - before[None, :]
    weights = torch.as_tensor(payload.get("server_weights", payload.get("fedavg_weights")), dtype=torch.float64)
    return spec, before, after, deltas, weights


def attribute_payload(
    payload: Mapping,
    metadata: Mapping,
    evaluator: TrainOnlyFunctionalEvaluator,
    *,
    weights: torch.Tensor | None = None,
    method: str = "trained_server",
    quadrature_points: int = 8,
    epsilon: float = 1e-12,
) -> tuple[list[dict], list[dict], list[dict], dict]:
    _, before, after, deltas, default_weights = payload_vectors(payload)
    weights = default_weights if weights is None else torch.as_tensor(weights, dtype=torch.float64)
    class_ids = evaluator.class_ids
    effects, aggregate = integrated_client_effects(
        before, deltas, weights, class_ids, evaluator.gradient, quadrature_points=quadrature_points
    )
    first_effects, _ = first_order_client_effects(before, deltas, weights, class_ids, evaluator.gradient)
    client_rows, budget_rows = rows_from_effects(
        effects,
        class_ids,
        payload["selected_client_ids"],
        torch.as_tensor(payload["client_class_counts"]),
        communication_round=int(metadata["communication_round"]),
        method=method,
        epsilon=epsilon,
    )
    first_client, first_budget = rows_from_effects(
        first_effects,
        class_ids,
        payload["selected_client_ids"],
        torch.as_tensor(payload["client_class_counts"]),
        communication_round=int(metadata["communication_round"]),
        method=f"{method}_first_order",
        epsilon=epsilon,
    )
    before_scores = [evaluator.metric(before, class_id) for class_id in class_ids]
    candidate_after = before + aggregate
    after_scores = [evaluator.metric(candidate_after, class_id) for class_id in class_ids]
    validity_rows = completeness_record(
        effects, before_scores, after_scores, class_ids,
        communication_round=int(metadata["communication_round"]), method=method,
    )
    # The trained-server state may differ only by intended server numeric order.
    trained_scores = [evaluator.metric(after, class_id) for class_id in class_ids]
    for row, trained in zip(validity_rows, trained_scores):
        row["trained_server_endpoint"] = trained
        row["candidate_vs_trained_endpoint_abs_error"] = abs(
            trained - (row["direct_change"] + before_scores[class_ids.index(row["class_id"])] )
        )
    report = {
        "communication_round": int(metadata["communication_round"]),
        "method": method,
        "quadrature_points": int(quadrature_points),
        "max_absolute_completeness_error": max(
            (row["absolute_completeness_error"] for row in validity_rows), default=math.nan
        ),
        "max_candidate_vs_trained_endpoint_error": max(
            (row["candidate_vs_trained_endpoint_abs_error"] for row in validity_rows), default=math.nan
        ),
        "server_weight_sum": float(weights.sum().item()),
        "first_order_budget_rows": first_budget,
        "first_order_client_rows": first_client,
    }
    return client_rows, budget_rows, validity_rows, report


def _write_rows(path: Path, rows: list[dict]) -> None:
    if rows:
        write_csv(path, rows)


def analyze_dump(
    dump_dir: str | Path,
    *,
    protocol_dir: str | Path,
    output_dir: str | Path,
    data_root: str | Path | None = None,
    quadrature_points: int = 8,
    device: str | None = None,
    method: str = "trained_server",
) -> dict:
    payload, metadata = load_round_dump(dump_dir)
    out = Path(output_dir)
    cfg, trainer = build_trainer_from_metadata(metadata, out / "model_build")
    evaluator = TrainOnlyFunctionalEvaluator(
        cfg=cfg, trainer=trainer, payload=payload, protocol_dir=protocol_dir,
        data_root=data_root, device=device,
    )
    client_rows, budget_rows, validity_rows, report = attribute_payload(
        payload, metadata, evaluator, method=method, quadrature_points=quadrature_points
    )
    _write_rows(out / "client_effects.csv", client_rows)
    _write_rows(out / "round_signed_budgets.csv", budget_rows)
    _write_rows(out / "attribution_validity.csv", validity_rows)
    _write_rows(out / "first_order_client_effects.csv", report.pop("first_order_client_rows"))
    _write_rows(out / "first_order_signed_budgets.csv", report.pop("first_order_budget_rows"))
    write_json(out / "analysis_manifest.json", {
        "schema_version": "eri_attribution_v1",
        "dump_dir": str(Path(dump_dir).resolve()),
        "protocol_dir": str(Path(protocol_dir).resolve()),
        "data_access": "train-only probe manifest",
        "report": report,
    })
    return report


def analyze_run(
    run_dir: str | Path,
    *,
    protocol_dir: str | Path,
    data_root: str | Path | None = None,
    quadrature_points: int = 8,
    device: str | None = None,
) -> Path:
    run_dir = Path(run_dir)
    dumps = sorted((run_dir / "eri_closure" / "dumps").glob("round_*"))
    if not dumps:
        raise FileNotFoundError(f"No ERI dumps found under {run_dir}")
    root = run_dir / "eri_closure" / "analysis"
    all_client: list[dict] = []
    all_budget: list[dict] = []
    all_validity: list[dict] = []
    all_first_client: list[dict] = []
    all_first_budget: list[dict] = []
    reports = []
    first_payload, first_metadata = load_round_dump(dumps[0])
    cfg, trainer = build_trainer_from_metadata(first_metadata, root / "model_build")
    reference_keys = tuple(FlatSpec.from_dict(first_payload["flatten_spec"]).keys)
    total_dumps = len(dumps)
    print(f"ERI attribution: {run_dir} ({total_dumps} audit rounds)", flush=True)
    for dump_index, dump_dir in enumerate(dumps, start=1):
        print(f"[{dump_index}/{total_dumps}] attributing {dump_dir.name}", flush=True)
        payload, metadata = load_round_dump(dump_dir)
        if tuple(FlatSpec.from_dict(payload["flatten_spec"]).keys) != reference_keys:
            raise RuntimeError(f"ERI dump parameter keys drift across audit rounds: {dump_dir}")
        evaluator = TrainOnlyFunctionalEvaluator(
            cfg=cfg, trainer=trainer, payload=payload, protocol_dir=protocol_dir,
            data_root=data_root, device=device,
        )
        client, budget, validity, report = attribute_payload(
            payload, metadata, evaluator, quadrature_points=quadrature_points
        )
        all_client.extend(client); all_budget.extend(budget); all_validity.extend(validity)
        all_first_client.extend(report.pop("first_order_client_rows"))
        all_first_budget.extend(report.pop("first_order_budget_rows")); reports.append(report)
        print(
            f"[{dump_index}/{total_dumps}] completed {dump_dir.name}; "
            f"max_completeness_error={report['max_absolute_completeness_error']:.6g}",
            flush=True,
        )
    _write_rows(root / "client_effects.csv", all_client)
    _write_rows(root / "round_signed_budgets.csv", all_budget)
    _write_rows(root / "attribution_validity.csv", all_validity)
    _write_rows(root / "first_order_client_effects.csv", all_first_client)
    _write_rows(root / "first_order_signed_budgets.csv", all_first_budget)
    write_json(root / "analysis_manifest.json", {
        "schema_version": "eri_attribution_v1", "run_dir": str(run_dir.resolve()),
        "protocol_dir": str(Path(protocol_dir).resolve()), "reports": reports,
        "data_access": "train-only probe manifest",
    })
    return root
