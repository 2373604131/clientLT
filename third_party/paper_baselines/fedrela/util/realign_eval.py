"""
Global realignment evaluation (g_aux calibration).
Uses globaltest_calibra from util.update_baseline without modifying that module.
"""
import copy
from typing import Sequence, Tuple, Dict

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

from util.update_baseline import globaltest_calibra


def relabel_is_enabled(args) -> bool:
    """True when FedReLa relabeling runs during training (relabel_start < rounds)."""
    return int(getattr(args, "relabel_start", 10**9)) < int(args.rounds)


def evaluate_global_realignment(
    model,
    g_aux,
    dataset_test,
    datasetObj,
    args,
) -> Tuple[float, Dict[str, float]]:
    """Run globaltest_calibra and return overall accuracy and H/M/T shot metrics."""
    acc, shot = globaltest_calibra(
        copy.deepcopy(model).to(args.device),
        copy.deepcopy(g_aux).to(args.device),
        dataset_test,
        args,
        dataset_class=datasetObj,
    )
    shot_f = {k: float(shot[k]) for k in ("head", "middle", "tail")}
    return float(acc), shot_f


def plot_realign_metrics(
    rounds: Sequence[int],
    overall: Sequence[float],
    many: Sequence[float],
    medium: Sequence[float],
    few: Sequence[float],
    args,
    names=("overall", "head", "middle", "tail"),
):
    """Separate PDF from training curves: {id}_realign.pdf"""
    if len(rounds) == 0:
        return
    overall = np.asarray(overall)
    many = np.asarray(many)
    medium = np.asarray(medium)
    few = np.asarray(few)
    rounds = np.asarray(rounds)
    if not (len(overall) == len(many) == len(medium) == len(few) == len(rounds)):
        raise ValueError("realign plot arrays must have equal length")

    plt.figure(figsize=(12, 6))
    colors = ["#9467bd", "#8c564b", "#e377c2", "#7f7f7f"]
    h1, = plt.plot(rounds, overall, color=colors[0], label=names[0], linestyle="-")
    h2, = plt.plot(rounds, many, color=colors[1], label=names[1], linestyle="--")
    h3, = plt.plot(rounds, medium, color=colors[2], label=names[2], linestyle="--")
    h4, = plt.plot(rounds, few, color=colors[3], label=names[3], linestyle="--")

    gidx = int(np.argmax(overall))
    info = (
        f"model_id:{args.id} (realignment)\n"
        f"peak overall: round {rounds[gidx]} ({overall[gidx]:.4f})\n"
        f"head/mid/tail: {many[gidx]:.4f} / {medium[gidx]:.4f} / {few[gidx]:.4f}"
    )
    plt.axvline(x=rounds[gidx], color="black", linestyle=":", alpha=0.6)
    plt.scatter(rounds[gidx], overall[gidx], color=colors[0], zorder=5, s=50)

    plt.xlabel("Round", fontsize=12)
    plt.ylabel("Value", fontsize=12)
    plt.ylim(0, 1)
    plt.grid(True, alpha=0.3)
    plt.title("Global realignment", fontsize=11)
    plt.legend(
        handles=[h1, h2, h3, h4, Line2D([], [], color="black", linestyle=":", label="Peak overall")],
        loc="upper left",
        bbox_to_anchor=(1, 1),
        frameon=False,
        fontsize=9,
    )
    plt.text(1.05, 0.5, info, transform=plt.gca().transAxes, va="center", fontsize=9)
    plt.tight_layout()
    out = getattr(args, "id", "run") + "_realign.pdf"
    plt.savefig(out, format="pdf")
    plt.close()
