#!/usr/bin/env python
"""visualize_partition.py

Quick sanity-check for federated long-tail splits.
Prints a full client × class table (zeros included) and
saves a stacked-bar histogram.
"""

import argparse
import os
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

# -------------------------------------------------------------------- #
#                    repo-specific import                               #
# -------------------------------------------------------------------- #
from utils.datasplit import partition_data, partition_data_LT

# -------------------------------------------------------------------- #
#                    Argparse                                           #
# -------------------------------------------------------------------- #
def get_args():
    p = argparse.ArgumentParser("Visualise FL partition")

    p.add_argument('--dataset',        default='cifar10_LT',
                   choices=['cifar10', 'cifar10_LT'])
    p.add_argument('--dataset-root',   default='./DATA/')
    p.add_argument('--partition',      default='longtail-client')
    p.add_argument('--num-users', type=int, default=10)

    # Dirichlet / long-tail params
    p.add_argument('--beta',      type=float, default=0.05,
                   help='Dirichlet alpha (group-internal)')
    p.add_argument('--imb-type',  default='exp')
    p.add_argument('--imb-factor',type=float, default=0.01)
    p.add_argument('--specialization-lambda', type=float, required=True)
    p.add_argument('--intra-group-alpha', type=float, required=True)
    p.add_argument('--head-leakage-scale', type=float, required=True)

    p.add_argument('--logdir',    default='./logs/')
    p.add_argument('--save-fig',  default='partition_hist.png')

    return p.parse_args()

# -------------------------------------------------------------------- #
#                    Pretty print                                       #
# -------------------------------------------------------------------- #
def pretty_print_counts(counts: dict[int, dict[int,int]], num_classes:int):
    header = "client " + " ".join(f"c{c:02d}" for c in range(num_classes))
    print(header)
    print("-"*len(header))
    for cid in sorted(counts):
        row = f"{cid:6d} " + " ".join(f"{counts[cid].get(cls,0):4d}"
                                      for cls in range(num_classes))
        print(row)

# -------------------------------------------------------------------- #
#                    Main                                               #
# -------------------------------------------------------------------- #
def main():
    args = get_args()
    root = Path(os.path.expanduser(args.dataset_root)); root.mkdir(exist_ok=True)

    # ---- load split --------------------------------------------------
    if args.dataset == 'cifar10_LT':
        _, _, lab2name, classnames, net_tr, net_te, cnt_tr, cnt_te, ytr = \
            partition_data_LT('cifar10_LT', str(root/'cifar-10'),
                               args.partition, args.num_users,
                               beta=args.beta, logdir=args.logdir,
                               imb_factor=args.imb_factor, imb_type=args.imb_type,
                               specialization_lambda=args.specialization_lambda,
                               intra_group_alpha=args.intra_group_alpha,
                               head_leakage_scale=args.head_leakage_scale)
    else:
        _, _, lab2name, classnames, net_tr, net_te, cnt_tr, cnt_te, ytr = \
            partition_data('cifar10', str(root/'cifar-10'),
                           args.partition, args.num_users,
                           beta=args.beta, logdir=args.logdir,
                           specialization_lambda=args.specialization_lambda,
                           intra_group_alpha=args.intra_group_alpha,
                           head_leakage_scale=args.head_leakage_scale)

    # ---- pretty table ------------------------------------------------
    num_classes = len(classnames)
    print("\n=== Training split (samples per class per client) ===")
    pretty_print_counts(cnt_tr, num_classes)

    # ---- stacked bar -------------------------------------------------
    num_clients = len(net_tr)
    counts = np.zeros((num_clients, num_classes), dtype=int)
    for cid in range(num_clients):
        for cls,cnt in cnt_tr[cid].items():
            counts[cid, cls] = cnt

    # Matplotlib ≥3.7 与旧版兼容
    try:
        cmap = plt.colormaps.get_cmap('tab10', num_classes)
    except TypeError:
        cmap = plt.get_cmap('tab10', num_classes)

    bottom = np.zeros(num_clients, dtype=int)
    ids = np.arange(num_clients)
    for cls in range(num_classes):
        heights = counts[:,cls]
        if not heights.any():           # 全 0 列直接跳过
            continue
        plt.bar(ids, heights, bottom=bottom, color=cmap(cls),
                label=classnames[cls])
        bottom += heights

    plt.xlabel("Client ID")
    plt.ylabel("#Samples (train)")
    plt.title(f"{args.dataset} — {args.partition} ({args.num_users} clients)")
    plt.legend(ncol=min(num_classes,5), fontsize='small',
               bbox_to_anchor=(1.04,1), loc='upper left')
    plt.tight_layout()
    plt.savefig(args.save_fig)
    print(f"\nHistogram saved to {args.save_fig}")

if __name__ == "__main__":
    main()
