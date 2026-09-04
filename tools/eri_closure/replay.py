"""Frozen aggregation replay for the ERI closure experiment.

Replay applies alternative *server coefficient vectors* to saved local states;
it never performs a further client optimizer step or accesses official test.
"""

from __future__ import annotations

import random
from pathlib import Path

import torch

from tools.eri_closure.analysis import (
    TrainOnlyFunctionalEvaluator,
    attribute_payload,
    build_trainer_from_metadata,
    load_round_dump,
    payload_vectors,
)
from utils.cusp_minimal import write_csv, write_json
from utils.lora_aggregation import support_normalized_client_weights


def _candidate_delta(deltas: torch.Tensor, weights: torch.Tensor) -> torch.Tensor:
    return (deltas * weights[:, None]).sum(dim=0)


def replay_dump(
    dump_dir: str | Path,
    *,
    protocol_dir: str | Path,
    output_dir: str | Path,
    data_root: str | Path | None = None,
    quadrature_points: int = 8,
    device: str | None = None,
    permutations: int = 100,
    permutation_seed: int = 20260904,
    cfg=None,
    trainer=None,
) -> dict:
    payload, metadata = load_round_dump(dump_dir)
    out = Path(output_dir)
    if cfg is None or trainer is None:
        cfg, trainer = build_trainer_from_metadata(metadata, out / "model_build")
    evaluator = TrainOnlyFunctionalEvaluator(
        cfg=cfg, trainer=trainer, payload=payload, protocol_dir=protocol_dir,
        data_root=data_root, device=device,
    )
    _, before, _, deltas, fedavg_weights = payload_vectors(payload)
    selected = [int(item) for item in payload["selected_client_ids"]]
    tail = evaluator.class_ids
    count_rows = torch.as_tensor(payload["client_class_counts"])
    # lora_aggregation accepts global client addressing; create the compact
    # selected-client representation deliberately to preserve saved ordering.
    compact_counts = {index: count_rows[index] for index in range(len(selected))}
    compact_samples = [float(item) for item in payload["client_sample_counts"]]
    compact_weights, details = support_normalized_client_weights(
        list(range(len(selected))), compact_samples, compact_counts, tail
    )
    sn_weights = torch.tensor([compact_weights[index] for index in range(len(selected))], dtype=torch.float64)
    if details["uncovered_tail_classes"]:
        raise RuntimeError("Full-participation ERI replay requires tail coverage: " + str(details))
    client, budgets, validity, report = attribute_payload(
        payload, metadata, evaluator, weights=sn_weights,
        method="support_normalized_replay", quadrature_points=quadrature_points,
    )
    write_csv(out / "support_normalized_client_effects.csv", client)
    write_csv(out / "support_normalized_signed_budgets.csv", budgets)
    write_csv(out / "support_normalized_attribution_validity.csv", validity)
    write_csv(out / "support_normalized_first_order_client_effects.csv", report.pop("first_order_client_rows"))
    write_csv(out / "support_normalized_first_order_signed_budgets.csv", report.pop("first_order_budget_rows"))

    before_values = {class_id: evaluator.metric(before, class_id) for class_id in tail}
    methods = [("fedavg_replay", fedavg_weights), ("support_normalized_replay", sn_weights)]
    rng = random.Random(int(permutation_seed) + int(metadata["communication_round"]))
    for permutation_id in range(int(permutations)):
        values = sn_weights.tolist()
        rng.shuffle(values)
        methods.append((f"permuted_support_normalized_{permutation_id:03d}", torch.tensor(values, dtype=torch.float64)))
    score_rows = []
    for method, weights in methods:
        candidate = before + _candidate_delta(deltas, weights)
        for class_id in tail:
            after_value = evaluator.metric(candidate, class_id)
            score_rows.append({
                "communication_round": int(metadata["communication_round"]),
                "method": method,
                "class_id": int(class_id),
                "mean_true_log_odds_before": before_values[class_id],
                "mean_true_log_odds_after": after_value,
                "immediate_margin_delta": after_value - before_values[class_id],
            })
    write_csv(out / "frozen_replay_scores.csv", score_rows)
    manifest = {
        "schema_version": "eri_frozen_replay_v1",
        "dump_dir": str(Path(dump_dir).resolve()),
        "data_access": "train-only probe manifest",
        "permutations": int(permutations),
        "permutation_seed": int(permutation_seed),
        "support_normalized_details": details,
        "support_normalized_weights_by_saved_client": {
            str(client_id): float(weight) for client_id, weight in zip(selected, sn_weights.tolist())
        },
        "attribution_report": report,
    }
    write_json(out / "frozen_replay_manifest.json", manifest)
    return manifest


def replay_run(
    run_dir: str | Path,
    **kwargs,
) -> Path:
    run_dir = Path(run_dir)
    root = run_dir / "eri_closure" / "replay"
    dumps = sorted((run_dir / "eri_closure" / "dumps").glob("round_*"))
    if not dumps:
        raise FileNotFoundError(f"No ERI dumps found under {run_dir}")
    manifests = []
    first_payload, first_metadata = load_round_dump(dumps[0])
    cfg, trainer = build_trainer_from_metadata(first_metadata, root / "model_build")
    for dump in dumps:
        manifests.append(replay_dump(dump, output_dir=root / dump.name, cfg=cfg, trainer=trainer, **kwargs))
    write_json(root / "replay_manifest.json", {"schema_version": "eri_frozen_replay_v1", "rounds": manifests})
    return root
