import copy
import os
from typing import List, Optional

import torch

from util.runlog import parse_epoch_milestones


def output_dir_for_dataset(dataset: str) -> str:
    return "./output/cifar10/" if dataset == "cifar10" else "./output/cifar100/"


def epoch_snapshot_tag(global_round_1based: int, total_rounds: int) -> str:
    """Tag such as epoch_u90 for weights after completing round int(0.9 * rounds)."""
    pct = int(round(global_round_1based / total_rounds * 100))
    pct = max(1, min(100, pct))
    return f"epoch_u{pct}"


def epoch_milestone_endpoints(
    milestones: str, rounds: int, include_final: bool = True
) -> List[int]:
    """
    Round endpoints (1-based global round index) at which to snapshot weights.
    Snapshot is written at the end of round rnd when (rnd + 1) == endpoint.
    """
    endpoints = parse_epoch_milestones(milestones, rounds)
    if include_final and rounds not in endpoints:
        endpoints.append(rounds)
    return sorted(set(endpoints))


def save_fed_checkpoint(
    save_dir: str,
    run_id: str,
    tag: str,
    model,
    g_head,
    g_aux,
    l_heads,
) -> None:
    os.makedirs(save_dir, exist_ok=True)
    torch.save(model, os.path.join(save_dir, f"{run_id}model_{tag}.pth"))
    torch.save(g_head, os.path.join(save_dir, f"{run_id}g_head_{tag}.pth"))
    torch.save(g_aux, os.path.join(save_dir, f"{run_id}g_aux_{tag}.pth"))
    for i, lh in enumerate(l_heads):
        torch.save(lh, os.path.join(save_dir, f"{run_id}l_head_{i}{tag}.pth"))
    print(f"Saved epoch snapshot tag={tag} under {save_dir} (run id {run_id})")


def maybe_save_epoch_checkpoint(
    save_switch: bool,
    rnd: int,
    epoch_endpoints_set: set,
    args,
    model,
    g_head,
    g_aux,
    l_heads,
    conf_out=None,
    conf_labels=None,
    w_glob: Optional[dict] = None,
) -> str:
    """Save weights at fixed global-round milestones; return snapshot tag if saved else ''."""
    if not save_switch or (rnd + 1) not in epoch_endpoints_set:
        return ""
    if w_glob is not None:
        model.load_state_dict(copy.deepcopy(w_glob))
    save_dir = output_dir_for_dataset(args.dataset)
    tag = epoch_snapshot_tag(rnd + 1, args.rounds)
    save_fed_checkpoint(save_dir, args.id, tag, model, g_head, g_aux, l_heads)
    if getattr(args, "save_test_confusion", False) and conf_out is not None:
        from util.runlog import save_confusion_json

        save_confusion_json(
            getattr(args, "log_dir", "./output/logs/"),
            args.id,
            tag,
            rnd,
            conf_out,
            conf_labels,
        )
    return tag
