#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用仓库 ClientLT 划分函数生成模拟数据，约束泄露位置后绘图。

固定 20 个客户端、10 类；尾部类为 7/8/9，尾部客户端为 18/19。
默认：三个尾部类合计向外泄露 60 个样本，尾部客户端合计接收
20 个其他类别样本。18 只混入类别 1/3/5，19 只混入类别 2/6，
各类别混入数量使用独立随机种子分配。外泄仅到指定客户端。
所有数据均为模拟，不读取或修改真实训练数据。
"""

import argparse
import json
import logging
import sys
from pathlib import Path

import matplotlib
import numpy as np

from plot_3d_histogram import plot_distribution

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))
from utils.datasplit import partition_client_longtail


# 外泄位置：每个尾部类最多流向 3 个非尾部客户端。
TAIL_LEAKAGE_TARGETS = {7: (3, 9, 15), 8: (1, 6, 12), 9: (4, 7, 11)}
# 按客户端分别指定允许混入的类别。
INCOMING_ROUTES = {18: (1, 3, 5), 19: (2, 6)}


def constrain_leakage(counts, leakage_seed=1):
    """对模拟计数施加指定路由；守恒每类总数及双向泄露总量。

    尾部类在 18/19 上的计数保持不变。外泄计数在指定目的地间均分；
    混入总预算在 18/19 间均分，再在各自允许的类别间随机分配。
    当预算足够时，每个指定类别至少混入一个样本。
    这一步是绘图模拟的额外约束，不修改仓库的训练划分实现。
    """
    counts = counts.copy()
    incoming_budget = int(counts[18:, :7].sum())
    rng = np.random.default_rng(leakage_seed)

    for class_id, clients in TAIL_LEAKAGE_TARGETS.items():
        if not 1 <= len(clients) <= 3 or len(set(clients)) != len(clients):
            raise ValueError("Each tail class needs 1-3 distinct leakage destinations")
        if any(client < 0 or client >= 18 for client in clients):
            raise ValueError("Leakage destinations must be clients 0-17")
        budget = int(counts[:18, class_id].sum())
        counts[:18, class_id] = 0
        base, remainder = divmod(budget, len(clients))
        for index, client in enumerate(clients):
            counts[client, class_id] = base + (index < remainder)

    # 先将原先混入的样本退回该类最大的非尾部持有者，再按允许类别取样。
    for class_id in range(7):
        donor = int(np.argmax(counts[:18, class_id]))
        counts[donor, class_id] += counts[18:, class_id].sum()
        counts[18:, class_id] = 0
    base, remainder = divmod(incoming_budget, len(INCOMING_ROUTES))
    for index, (client, classes) in enumerate(INCOMING_ROUTES.items()):
        if client not in (18, 19) or not classes or len(set(classes)) != len(classes):
            raise ValueError("Incoming routes need tail clients and distinct nonempty classes")
        if any(class_id < 0 or class_id >= 7 for class_id in classes):
            raise ValueError("Incoming routes may only contain non-tail classes 0-6")
        budget = base + (index < remainder)
        minimum = np.full(len(classes), int(budget >= len(classes)), dtype=int)
        weights = rng.dirichlet(np.full(len(classes), 2.0))
        allocation = minimum + rng.multinomial(budget - int(minimum.sum()), weights)
        for class_id, quantity in zip(classes, allocation):
            if quantity > counts[:18, class_id].sum():
                raise ValueError(f"Class {class_id} has too few samples for its incoming budget")
            remaining = int(quantity)
            for donor in np.argsort(-counts[:18, class_id], kind="stable"):
                take = min(remaining, int(counts[donor, class_id]))
                counts[donor, class_id] -= take
                remaining -= take
                if remaining == 0:
                    break
            counts[client, class_id] = quantity
    return counts


def simulate_counts(head_count=5000, imb_factor=0.02, tail_leakage=60,
                    head_leakage=20, alpha=0.1, seed=1, leakage_seed=1):
    """生成 ClientLT 模拟矩阵并施加用户指定的稀疏泄露约束。

    全局类别总数按指数递减；泄露预算在各类间按类别总数分配，
    基础划分调用仓库原函数；随后按指定位置重新分配泄露计数。
    额外约束只作用于本模拟矩阵，全局类总数和泄露总预算守恒。
    """
    if head_count <= 0 or not 0 < imb_factor <= 1:
        raise ValueError("head_count > 0 and 0 < imb_factor <= 1 are required")
    if not np.isfinite(alpha) or alpha <= 0 or seed < 0 or leakage_seed < 0:
        raise ValueError("alpha must be finite and positive; seeds must be nonnegative")
    if tail_leakage < 0 or head_leakage < 0:
        raise ValueError("Leakage budgets must be nonnegative")

    totals = np.floor(head_count * imb_factor ** (np.arange(10) / 9)).astype(int)
    n_tail = int(totals[7:].sum())
    if n_tail == 0 or tail_leakage > 0.9 * n_tail:
        raise ValueError("Tail leakage must not exceed 90% of the nonzero tail total")
    incoming_classes = sorted({class_id for classes in INCOMING_ROUTES.values() for class_id in classes})
    if head_leakage > totals[incoming_classes].sum():
        raise ValueError("Head leakage exceeds the available samples of the allowed classes")
    if tail_leakage == 0 and head_leakage > 0:
        raise ValueError("The repository's ClientLT formula requires zero head leakage "
                         "when tail leakage is zero (lambda=1)")

    # 仓库公式：tail_out = N_tail * (1-q_t) * (1-lambda)。
    #           head_in = rho * N_tail * q_t * (1-lambda)。
    # q_t=2/20=0.1；从用户更易理解的“样本数预算”反推 lambda/rho。
    specialization = 1.0 - tail_leakage / (0.9 * n_tail)
    rho = 9.0 * head_leakage / tail_leakage if tail_leakage else 0.0
    labels = np.repeat(np.arange(10), totals)
    partition = partition_client_longtail(
        labels, n_parties=20, num_classes=10,
        head_client_ratio=0.9, tail_client_ratio=0.1,
        head_class_ratio=0.7, tail_class_ratio=0.3,
        specialization_lambda=specialization,
        intra_group_alpha=alpha, head_leakage_scale=rho,
        rng=np.random.RandomState(seed),
    )
    counts = np.array([
        np.bincount(labels[partition[client]], minlength=10)
        for client in range(20)
    ], dtype=np.int64)
    original_tail_counts = counts[18:, 7:].copy()
    counts = constrain_leakage(counts, leakage_seed=leakage_seed)

    # 核对样本守恒与实际泄露数量，避免只在图上看起来满足要求。
    np.testing.assert_array_equal(counts.sum(axis=0), totals)
    np.testing.assert_array_equal(counts[18:, 7:], original_tail_counts)
    assert (counts >= 0).all()
    assert int(counts[:18, 7:].sum()) == tail_leakage
    assert int(counts[18:, :7].sum()) == head_leakage
    for class_id, clients in TAIL_LEAKAGE_TARGETS.items():
        assert set(np.flatnonzero(counts[:18, class_id])).issubset(clients)
    for client, classes in INCOMING_ROUTES.items():
        assert set(np.flatnonzero(counts[client, :7])).issubset(classes)
    metadata = {
        "data_source": "Synthetic ClientLT counts with explicit sparse leakage routing",
        "base_partition_function": "utils.datasplit.partition_client_longtail",
        "routing_policy": "Post-process simulated counts, preserving every class total and both leakage budgets",
        "is_simulation": True,
        "seed": seed,
        "leakage_seed": leakage_seed,
        "head_count": head_count,
        "imb_factor": imb_factor,
        "alpha": alpha,
        "specialization_lambda": specialization,
        "head_leakage_scale": rho,
        "tail_classes": [7, 8, 9],
        "tail_clients": [18, 19],
        "class_totals": totals.tolist(),
        "tail_leakage_total": int(counts[:18, 7:].sum()),
        "tail_leakage_by_class_7_8_9": counts[:18, 7:].sum(axis=0).tolist(),
        "tail_leakage_routes": {
            str(class_id): {str(client): int(counts[client, class_id]) for client in clients}
            for class_id, clients in TAIL_LEAKAGE_TARGETS.items()
        },
        "allowed_incoming_classes": incoming_classes,
        "incoming_routes": {
            str(client): {str(class_id): int(counts[client, class_id]) for class_id in classes}
            for client, classes in INCOMING_ROUTES.items()
        },
        "non_tail_samples_on_tail_clients": int(counts[18:, :7].sum()),
        "clients": [
            {
                "client_id": client,
                "tail_samples": int(counts[client, 7:].sum()),
                "non_tail_samples": int(counts[client, :7].sum()),
                "counts_by_class": counts[client].tolist(),
            }
            for client in [18, 19]
        ],
    }
    return counts, metadata


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--head-count", type=int, default=5000)
    parser.add_argument("--imb-factor", type=float, default=0.02)
    parser.add_argument("--tail-leakage", type=int, default=60,
                        help="Total samples of classes 7/8/9 sent to clients 0-17")
    parser.add_argument("--head-leakage", type=int, default=20,
                        help="Total incoming samples: client 18 gets classes 1/3/5; 19 gets 2/6")
    parser.add_argument("--alpha", type=float, default=0.1)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--leakage-seed", type=int, default=1,
                        help="Independent seed for incoming class counts; preserves the base split")
    parser.add_argument("--elev", type=float, default=30)
    parser.add_argument("--azim", type=float, default=-45)
    parser.add_argument("--box-aspect", type=float, nargs=3, default=(1, 1, 1),
                        metavar=("X", "Y", "Z"),
                        help="Visual axis proportions; default matches the reference recreation")
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "clientlt_output")
    args = parser.parse_args()
    logging.getLogger("fontTools").setLevel(logging.WARNING)
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        counts, metadata = simulate_counts(
            args.head_count, args.imb_factor, args.tail_leakage,
            args.head_leakage, args.alpha, args.seed, args.leakage_seed,
        )
    except ValueError as exc:
        parser.error(str(exc))
    args.output.mkdir(parents=True, exist_ok=True)
    np.savetxt(args.output / "clientlt_counts.csv", counts, delimiter=",", fmt="%d",
               header="SIMULATED ClientLT data, not measured experimental data.\n"
                      "Rows=Client ID 0-19; columns=Class 0-9.\n"
                      "Tail routes: 7->3/9/15; 8->1/6/12; 9->4/7/11. Incoming: 18<-1/3/5; 19<-2/6.\n"
                      f"Incoming allocation seed={args.leakage_seed}.\n"
                      f"Tail leakage total={args.tail_leakage}; head leakage total={args.head_leakage}.")
    (args.output / "simulation_summary.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2) + "\n", encoding="utf-8",
    )

    fig, ax = plot_distribution(counts, {
        "elev": args.elev, "azim": args.azim,
        "box_aspect": tuple(args.box_aspect),
    })
    # 原始线性柱高保持不变；通过坐标刻度颜色标记指定的尾部组。
    for class_id, label in enumerate(ax.get_xticklabels()):
        if class_id in (7, 8, 9):
            label.set_color("#a63b20")
            label.set_fontweight("bold")
    for client_id, label in enumerate(ax.get_yticklabels()):
        if client_id in (18, 19):
            label.set_color("#a63b20")
            label.set_fontweight("bold")
    with matplotlib.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
        for extension in ("png", "pdf", "svg"):
            destination = args.output / f"clientlt_3d.{extension}"
            fig.savefig(destination, dpi=300, bbox_inches="tight", pad_inches=0.16,
                        facecolor="white",
                        bbox_extra_artists=[ax.xaxis.label, ax.yaxis.label, ax.zaxis.label])
            print(f"Saved: {destination}")
    print("SIMULATION: tail classes=[7,8,9], tail clients=[18,19]")
    print("Tail leakage by class:", metadata["tail_leakage_by_class_7_8_9"])
    print("Tail leakage routes:", metadata["tail_leakage_routes"])
    print("Incoming routes:", metadata["incoming_routes"])
    for record in metadata["clients"]:
        print(f"Client {record['client_id']}: tail={record['tail_samples']}, "
              f"other={record['non_tail_samples']}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
