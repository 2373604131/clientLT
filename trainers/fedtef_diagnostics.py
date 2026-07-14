import torch


def assert_finite_outputs(outputs):
    for key in ("logits_base", "residual_tail", "logits", "gated_residual"):
        if key in outputs and not torch.isfinite(outputs[key]).all():
            safe = outputs[key].detach().float().nan_to_num()
            print(
                f"[FedTEF NaN DEBUG] {key}: "
                f"min={safe.min().item():.6f}, "
                f"max={safe.max().item():.6f}, "
                f"mean={safe.mean().item():.6f}"
            )
            raise FloatingPointError(f"{key} is NaN/Inf")


def summarize_gate(gate, scores=None):
    protected_ids = torch.nonzero(gate > 0, as_tuple=False).view(-1).tolist()
    summary = {
        "protected_count": len(protected_ids),
        "protected_ids": protected_ids,
        "gate_mean": float(gate.float().mean().item()) if gate.numel() else 0.0,
    }
    if scores is not None and scores.numel():
        summary.update(
            score_min=float(scores.min().item()),
            score_max=float(scores.max().item()),
            score_mean=float(scores.float().mean().item()),
        )
    return summary


def print_round_diagnostics(round_idx, gate, scores, tail_stats=None):
    summary = summarize_gate(gate, scores)
    msg = (
        f"[FedTEF diagnostics] round={round_idx} "
        f"protected={summary['protected_count']} "
        f"gate_mean={summary['gate_mean']:.4f}"
    )
    if "score_min" in summary:
        msg += (
            f" score={summary['score_min']:.4f}/"
            f"{summary['score_mean']:.4f}/"
            f"{summary['score_max']:.4f}"
        )
    if tail_stats is not None and "exposure_proxy" in tail_stats:
        exposure = tail_stats["exposure_proxy"].float()
        msg += f" exposure_mean={exposure.mean().item():.6f}"
    print(msg)
