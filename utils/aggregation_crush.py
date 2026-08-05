#!/usr/bin/env python
"""Aggregation-crush diagnostic: does FedAvg erase tail knowledge that
individual (tail) clients demonstrably learned locally?

This is the *decisive mechanism experiment* for the client-topology framing.
For a single communication round it measures, per tail class, the gap between:

  * pre-aggregation LOCAL accuracy  -- each supporting client's own model,
    evaluated on the global test set, restricted to the classes that client
    actually holds ("did the client learn this class locally?"); and
  * post-aggregation GLOBAL accuracy -- the FedAvg model on the same class
    ("did the aggregate keep it?").

It aligns every tail class to its support-client *mass share* ``n_S / N`` so
the money figure can show tail knowledge collapsing precisely where the mass
share is small -- i.e. where FedAvg's ``n_k / N`` weighting crushes the strong
but low-mass local signal. The same curve also refutes reweighting: the local
signal it would amplify is a high-variance estimate from few samples.

The routine reuses the trainer's own ``global_test`` so that local and global
numbers come from an identical evaluation path and are directly comparable.
No test access happens outside this evaluation, and it only runs on the round
selected by the caller.
"""

from __future__ import annotations

import csv
import os
from typing import Iterable, Mapping, Sequence

import torch


def _counts_matrix(client_class_counts: Mapping[int, torch.Tensor], num_users: int, num_classes: int) -> torch.Tensor:
    """Return an ``(num_users, num_classes)`` float tensor of per-client counts."""
    counts = torch.zeros(num_users, num_classes, dtype=torch.float64)
    for client_idx in range(num_users):
        if client_idx in client_class_counts:
            counts[client_idx] = client_class_counts[client_idx].double().cpu()
    return counts


def _tail_class_ids(global_counts: torch.Tensor, tail_class_ratio: float) -> list[int]:
    """Bottom ``tail_class_ratio`` fraction of classes by global frequency."""
    num_classes = int(global_counts.numel())
    n_tail = max(1, int(round(num_classes * tail_class_ratio)))
    order = torch.argsort(global_counts, descending=True)  # head -> tail
    return sorted(int(c) for c in order[-n_tail:].tolist())


def _evaluate_per_class(trainer, weights: Mapping[str, torch.Tensor], strict: bool) -> dict[int, float]:
    """Load ``weights`` into the trainer's model and return its per-class accuracy.

    Reuses ``global_test`` whose last return element is a ``{class_id: acc%}``
    dict, so local and global evaluations share one code path and are directly
    comparable.
    """
    trainer.model.load_state_dict(weights, strict=strict)
    result = trainer.global_test(is_global=True)
    class_accuracy = result[-1]
    return {int(cls): float(acc) for cls, acc in class_accuracy.items()}


def measure_aggregation_crush(
    global_trainer,
    pre_global_weights: Mapping[str, torch.Tensor],
    local_weights,
    global_weights: Mapping[str, torch.Tensor],
    idxs_users: Sequence[int],
    datanumber_client: Mapping[int, float] | Sequence[float],
    client_class_counts: Mapping[int, torch.Tensor],
    num_users: int,
    num_classes: int,
    *,
    tail_class_ratio: float = 0.2,
    strict_load: bool = False,
    min_support_samples: int = 1,
) -> tuple[list[dict], list[dict]]:
    """Measure per-tail-class local-vs-global accuracy for one round.

    Returns ``(per_client_rows, per_class_rows)``.

    ``per_client_rows``  -- one row per (tail class, supporting selected client):
        that client's LOCAL model accuracy on that class (pre-aggregation).
    ``per_class_rows``   -- one row per tail class: the support-mass share
        ``n_S / N``, the best/mean local accuracy across supporting clients,
        the pre-aggregation GLOBAL accuracy, and the post-aggregation GLOBAL
        accuracy -- i.e. everything the money figure needs.

    The model state is restored to ``global_weights`` before returning so the
    caller's training loop is unaffected.
    """
    selected = [int(x) for x in idxs_users]

    def _datanum(idx: int) -> float:
        try:
            return float(datanumber_client[idx])
        except (KeyError, IndexError, TypeError):
            return 0.0

    counts = _counts_matrix(client_class_counts, num_users, num_classes)
    global_counts = counts.sum(dim=0)
    total_mass = float(sum(_datanum(i) for i in range(num_users))) or float(num_users)
    tail_classes = _tail_class_ids(global_counts, tail_class_ratio)
    tail_set = set(tail_classes)

    was_training = global_trainer.model.training

    # --- pre-aggregation: evaluate each selected client's LOCAL model once. ---
    # local_accuracy[idx] = {class_id: acc%} for the classes that client holds.
    local_accuracy: dict[int, dict[int, float]] = {}
    for idx in selected:
        if isinstance(local_weights, dict):
            weights = local_weights.get(idx)
        else:
            weights = local_weights[idx] if idx < len(local_weights) else None
        if not isinstance(weights, Mapping):
            continue
        local_accuracy[idx] = _evaluate_per_class(global_trainer, weights, strict_load)

    # --- global models: pre-aggregation and post-aggregation. ---
    global_pre_acc = _evaluate_per_class(global_trainer, pre_global_weights, strict_load)
    global_post_acc = _evaluate_per_class(global_trainer, global_weights, strict_load)

    # Restore the loop's expected model state.
    global_trainer.model.load_state_dict(global_weights, strict=strict_load)
    global_trainer.model.train(was_training)

    per_client_rows: list[dict] = []
    per_class_rows: list[dict] = []

    for cls in tail_classes:
        # Support = selected clients that actually hold this tail class.
        support_clients = [
            idx for idx in selected
            if counts[idx, cls].item() >= float(min_support_samples)
        ]
        support_mass = sum(_datanum(idx) for idx in support_clients)
        support_mass_share = support_mass / total_mass if total_mass > 0 else 0.0

        local_accs_here: list[float] = []
        for idx in support_clients:
            acc = local_accuracy.get(idx, {}).get(int(cls))
            if acc is None:
                continue
            local_accs_here.append(acc)
            per_client_rows.append({
                "class_id": int(cls),
                "client_id": int(idx),
                "client_local_acc": float(acc),
                "client_samples_in_class": float(counts[idx, cls].item()),
                "client_total_samples": _datanum(idx),
                "client_mass_share": _datanum(idx) / total_mass if total_mass > 0 else 0.0,
            })

        best_local = max(local_accs_here) if local_accs_here else float("nan")
        mean_local = (sum(local_accs_here) / len(local_accs_here)) if local_accs_here else float("nan")
        post = float(global_post_acc.get(int(cls), float("nan")))
        pre = float(global_pre_acc.get(int(cls), float("nan")))

        per_class_rows.append({
            "class_id": int(cls),
            "class_group": "tail",
            "global_count": float(global_counts[cls].item()),
            "num_support_clients": len(support_clients),
            "support_mass": float(support_mass),
            "support_mass_share": float(support_mass_share),
            "best_local_acc": best_local,
            "mean_local_acc": mean_local,
            "global_pre_agg_acc": pre,
            "global_post_agg_acc": post,
            # The crush: strong local signal that aggregation failed to keep.
            "crush_gap_best": (best_local - post) if local_accs_here else float("nan"),
            "crush_gap_mean": (mean_local - post) if local_accs_here else float("nan"),
        })

    return per_client_rows, per_class_rows


def should_log_aggregation_crush(args, epoch: int, is_eval_round: bool) -> bool:
    """Gate the crush diagnostic for a given 0-based ``epoch``.

    If ``--agg_crush_rounds`` is set, only those 1-based rounds run. Otherwise
    it piggybacks on the caller's global-eval rounds (``is_eval_round``) so it
    reuses the same test-loader passes the training loop already makes.
    """
    if not bool(getattr(args, "agg_crush_enable", False)):
        return False
    spec = str(getattr(args, "agg_crush_rounds", "") or "").strip()
    if not spec:
        return bool(is_eval_round)
    wanted = set()
    for value in spec.split(","):
        value = value.strip()
        if value:
            wanted.add(int(value))
    return (int(epoch) + 1) in wanted


def _append_rows(path: str, rows: Iterable[dict], extra: Mapping[str, object]) -> None:
    rows = list(rows)
    if not rows:
        return
    enriched = [{**extra, **row} for row in rows]
    fieldnames = list(enriched[0].keys())
    write_header = not os.path.exists(path)
    os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerows(enriched)


def append_aggregation_crush(
    output_dir: str,
    args,
    epoch: int,
    global_trainer,
    pre_global_weights,
    local_weights,
    global_weights,
    idxs_users,
    datanumber_client,
    client_class_counts,
    num_classes,
    tail_class_ratio: float = 0.2,
    strict_load: bool = False,
) -> None:
    """Run the crush diagnostic for one round and append CSV rows.

    Writes two files under ``output_dir``:
      * ``aggregation_crush_per_class.csv``  -- the money-figure source.
      * ``aggregation_crush_per_client.csv`` -- supporting local accuracies.

    Tagged with method/partition/frac/seed/epoch for cross-run aggregation,
    mirroring the project's other diagnostic CSVs.
    """
    per_client_rows, per_class_rows = measure_aggregation_crush(
        global_trainer,
        pre_global_weights,
        local_weights,
        global_weights,
        idxs_users,
        datanumber_client,
        client_class_counts,
        int(getattr(args, "num_users", 0) or 0),
        int(num_classes),
        tail_class_ratio=tail_class_ratio,
        strict_load=strict_load,
    )

    tag = {
        "epoch": int(epoch),
        "method": getattr(args, "trainer", ""),
        "partition": getattr(args, "partition", ""),
        "frac": getattr(args, "frac", ""),
        "seed": getattr(args, "seed", ""),
    }
    _append_rows(os.path.join(output_dir, "aggregation_crush_per_class.csv"), per_class_rows, tag)
    _append_rows(os.path.join(output_dir, "aggregation_crush_per_client.csv"), per_client_rows, tag)
    print(
        f"[aggregation-crush] epoch={epoch} tail_classes={len(per_class_rows)} "
        f"support_rows={len(per_client_rows)} written to {output_dir}"
    )
