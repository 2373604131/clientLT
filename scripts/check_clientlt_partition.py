#!/usr/bin/env python
"""Inspect the CIFAR100-LT client-longtail partition used by Experiment F."""

import argparse
import csv
import os
import pickle
import random
import sys
import types
from pathlib import Path

import numpy as np


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

def parse_args():
    parser = argparse.ArgumentParser(
        description="Print and validate a CIFAR100-LT client-longtail partition."
    )
    parser.add_argument("--data-root", default="DATA", help="Dataset root. Defaults to DATA.")
    parser.add_argument(
        "--datadir",
        default="",
        help="Explicit CIFAR-100 directory. Defaults to <data-root>/cifar-100.",
    )
    parser.add_argument("--imb-factor", type=float, default=0.01)
    parser.add_argument("--imb-type", default="exp")
    parser.add_argument("--num-clients", type=int, default=50)
    parser.add_argument("--head-client-ratio", type=float, default=0.9)
    parser.add_argument("--tail-client-ratio", type=float, default=0.1)
    parser.add_argument("--head-class-ratio", type=float, default=0.8)
    parser.add_argument("--tail-class-ratio", type=float, default=0.2)
    parser.add_argument("--specialization-lambda", type=float, default=0.75)
    parser.add_argument("--head-leakage-scale", type=float, default=3.0)
    parser.add_argument("--intra-group-alpha", type=float, default=0.5)
    parser.add_argument("--split-seed", type=int, default=1)
    parser.add_argument(
        "--csv-dir",
        default="output/partition_checks/clientlt_cifar100_lt",
        help="Directory for CSV summaries. Use an empty string to disable CSV output.",
    )
    parser.add_argument(
        "--use-repo-loader",
        action="store_true",
        help="Use utils.dataloader.load_cifar100_LT_data instead of the lightweight raw-label path.",
    )
    return parser.parse_args()


def _unpickle(path):
    with path.open("rb") as f:
        return pickle.load(f, encoding="latin1")


def _find_cifar100_python_dir(datadir):
    candidates = [
        Path(datadir) / "cifar-100-python",
        Path(datadir),
    ]
    for candidate in candidates:
        if (candidate / "train").exists() and (candidate / "meta").exists():
            return candidate
    raise FileNotFoundError(
        "Could not find CIFAR-100 raw files. Expected either "
        f"{Path(datadir) / 'cifar-100-python' / 'train'} or {Path(datadir) / 'train'}."
    )


def _load_cifar100_raw_labels(datadir):
    cifar_dir = _find_cifar100_python_dir(datadir)
    train = _unpickle(cifar_dir / "train")
    meta = _unpickle(cifar_dir / "meta")
    labels = np.asarray(train["fine_labels"], dtype=np.int64)
    classnames = [str(name) for name in meta["fine_label_names"]]
    return labels, classnames, f"raw CIFAR-100 labels from {cifar_dir}"


def _make_lt_labels_like_repo(labels, num_classes, imb_factor, imb_type):
    if imb_type != "exp":
        raise ValueError(f"Only imb_type='exp' is supported by this check script, got {imb_type}")

    random.seed(1)
    np.random.seed(1)
    label2indices = [np.where(labels == class_id)[0].tolist() for class_id in range(num_classes)]
    img_max = len(labels) / num_classes
    selected_indices = []
    for class_id in range(num_classes):
        keep = int(img_max * (float(imb_factor) ** (class_id / (num_classes - 1.0))))
        class_indices = label2indices[class_id]
        np.random.shuffle(class_indices)
        selected_indices.extend(class_indices[:keep])
    return labels[np.asarray(selected_indices, dtype=np.int64)]


def load_cifar100_lt_labels(datadir, imb_factor, imb_type, use_repo_loader):
    if use_repo_loader:
        from utils.dataloader import load_cifar100_LT_data

        (
            _x_train,
            y_train,
            _x_test,
            _y_test,
            _data_train,
            _data_test,
            _lab2cname,
            classnames,
        ) = load_cifar100_LT_data(str(datadir), imb_factor, imb_type)
        return np.asarray(y_train, dtype=np.int64), classnames, "utils.dataloader.load_cifar100_LT_data"

    raw_labels, classnames, source = _load_cifar100_raw_labels(datadir)
    y_train = _make_lt_labels_like_repo(raw_labels, 100, imb_factor, imb_type)
    return y_train, classnames, source


def import_partition_client_longtail():
    try:
        from utils.datasplit import partition_client_longtail

        return partition_client_longtail
    except ModuleNotFoundError as exc:
        if exc.name != "torchvision":
            raise

        stub_dataloader = types.ModuleType("utils.dataloader")
        for name in (
            "load_mnist_data",
            "load_fmnist_data",
            "load_fmnist_LT_data",
            "load_cifar10_data",
            "load_cifar100_data",
            "load_cifar10_LT_data",
            "load_cifar100_LT_data",
            "load_svhn_data",
            "load_celeba_data",
            "load_femnist_data",
        ):
            setattr(stub_dataloader, name, None)
        sys.modules["utils.dataloader"] = stub_dataloader

        stub_dataset = types.ModuleType("utils.dataset")
        stub_dataset.mkdirs = lambda *args, **kwargs: None
        sys.modules["utils.dataset"] = stub_dataset

        from utils.datasplit import partition_client_longtail

        return partition_client_longtail


def range_text(values):
    values = list(values)
    if not values:
        return "[]"
    if values == list(range(values[0], values[-1] + 1)):
        return f"class {values[0]}-{values[-1]}" if values[-1] <= 99 else f"{values[0]}-{values[-1]}"
    return ", ".join(str(value) for value in values)


def client_range_text(values):
    values = list(values)
    if not values:
        return "[]"
    if values == list(range(values[0], values[-1] + 1)):
        return f"client {values[0]}-{values[-1]}"
    return ", ".join(str(value) for value in values)


def build_client_class_counts(labels, net_dataidx_map, num_clients, num_classes):
    counts = np.zeros((num_clients, num_classes), dtype=np.int64)
    for client_id in range(num_clients):
        indices = np.asarray(net_dataidx_map[client_id], dtype=np.int64)
        if len(indices) == 0:
            continue
        counts[client_id] = np.bincount(labels[indices], minlength=num_classes)
    return counts


def verify_partition(labels, net_dataidx_map, client_counts, global_counts):
    merged = np.concatenate([np.asarray(v, dtype=np.int64) for v in net_dataidx_map.values()])
    coverage_ok = len(merged) == len(labels)
    unique_ok = len(np.unique(merged)) == len(labels)
    index_set_ok = np.array_equal(np.sort(merged), np.arange(len(labels), dtype=np.int64))
    class_counts_ok = np.array_equal(client_counts.sum(axis=0), global_counts)
    return {
        "coverage_ok": coverage_ok,
        "unique_ok": unique_ok,
        "index_set_ok": index_set_ok,
        "class_counts_ok": class_counts_ok,
    }


def write_csv_outputs(csv_dir, classnames, global_counts, client_counts, head_classes, tail_classes, head_clients):
    csv_dir.mkdir(parents=True, exist_ok=True)

    with (csv_dir / "global_class_counts.csv").open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=["class_id", "classname", "group", "global_count"],
        )
        writer.writeheader()
        for class_id, count in enumerate(global_counts):
            writer.writerow(
                {
                    "class_id": class_id,
                    "classname": classnames[class_id],
                    "group": "tail" if class_id in tail_classes else "head",
                    "global_count": int(count),
                }
            )

    with (csv_dir / "client_class_counts.csv").open("w", newline="", encoding="utf-8") as f:
        fieldnames = ["client_id", "client_group", "total_samples"] + [
            f"class_{class_id}" for class_id in range(client_counts.shape[1])
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for client_id, row in enumerate(client_counts):
            payload = {
                "client_id": client_id,
                "client_group": "head" if client_id in head_clients else "tail",
                "total_samples": int(row.sum()),
            }
            payload.update({f"class_{class_id}": int(row[class_id]) for class_id in range(len(row))})
            writer.writerow(payload)


def main():
    args = parse_args()
    partition_client_longtail = import_partition_client_longtail()

    datadir = Path(args.datadir) if args.datadir else Path(args.data_root) / "cifar-100"
    num_classes = 100

    print("=== Client-LT Partition Check ===")
    print(f"dataset = cifar100_LT")
    print(f"datadir = {datadir}")
    print(f"imb_factor = {args.imb_factor}  # 0.01 means approximately 100:1")
    print(f"imb_type = {args.imb_type}")
    print(f"num_clients = {args.num_clients}")
    print(f"head_client_ratio = {args.head_client_ratio}")
    print(f"tail_client_ratio = {args.tail_client_ratio}")
    print(f"head_class_ratio = {args.head_class_ratio}")
    print(f"tail_class_ratio = {args.tail_class_ratio}")
    print(f"specialization_lambda = {args.specialization_lambda}")
    print(f"head_leakage_scale = {args.head_leakage_scale}")
    print(f"intra_group_alpha = {args.intra_group_alpha}")
    print(f"split_seed = {args.split_seed}")
    print()

    y_train, classnames, data_source = load_cifar100_lt_labels(
        datadir,
        args.imb_factor,
        args.imb_type,
        args.use_repo_loader,
    )
    print(f"label_source = {data_source}")
    global_counts = np.bincount(y_train, minlength=num_classes)

    head_client_count = int(args.num_clients * args.head_client_ratio)
    head_clients = list(range(head_client_count))
    tail_clients = list(range(head_client_count, args.num_clients))
    head_class_count = int(num_classes * args.head_class_ratio)
    head_classes = list(range(head_class_count))
    tail_classes = list(range(head_class_count, num_classes))
    head_class_set = set(head_classes)
    tail_class_set = set(tail_classes)

    print("=== Expected Groups From Ratios ===")
    print(f"head classes = {range_text(head_classes)}")
    print(f"tail classes = {range_text(tail_classes)}")
    print(f"head clients = {client_range_text(head_clients)}")
    print(f"tail clients = {client_range_text(tail_clients)}")
    print(f"confirm head classes == class 0-79: {head_classes == list(range(80))}")
    print(f"confirm tail classes == class 80-99: {tail_classes == list(range(80, 100))}")
    print(f"confirm head clients == client 0-44: {head_clients == list(range(45))}")
    print(f"confirm tail clients == client 45-49: {tail_clients == list(range(45, 50))}")
    print()

    nonincreasing = bool(np.all(global_counts[:-1] >= global_counts[1:]))
    max_count = int(global_counts.max())
    min_positive_count = int(global_counts[global_counts > 0].min())
    print("=== Global CIFAR100-LT Class Counts ===")
    print(f"class IDs are non-increasing by global sample count: {nonincreasing}")
    print(f"max_count / min_positive_count = {max_count} / {min_positive_count} = {max_count / min_positive_count:.2f}")
    print("class_id, classname, group, global_count")
    for class_id, count in enumerate(global_counts):
        group = "tail" if class_id in tail_class_set else "head"
        print(f"{class_id:02d}, {classnames[class_id]}, {group}, {int(count)}")
    if not nonincreasing:
        order = np.argsort(-global_counts)
        print("WARNING: class IDs are not sorted from many samples to few samples.")
        print("descending count class order:")
        print(" ".join(str(int(class_id)) for class_id in order))
    print()

    net_dataidx_map = partition_client_longtail(
        labels=y_train,
        n_parties=args.num_clients,
        num_classes=num_classes,
        head_client_ratio=args.head_client_ratio,
        tail_client_ratio=args.tail_client_ratio,
        head_class_ratio=args.head_class_ratio,
        tail_class_ratio=args.tail_class_ratio,
        specialization_lambda=args.specialization_lambda,
        intra_group_alpha=args.intra_group_alpha,
        head_leakage_scale=args.head_leakage_scale,
        rng=np.random.RandomState(args.split_seed),
    )
    client_counts = build_client_class_counts(y_train, net_dataidx_map, args.num_clients, num_classes)
    checks = verify_partition(y_train, net_dataidx_map, client_counts, global_counts)

    tail_client_tail_samples = int(client_counts[np.ix_(tail_clients, tail_classes)].sum())
    tail_client_non_tail_samples = int(client_counts[np.ix_(tail_clients, head_classes)].sum())
    tail_client_total = tail_client_tail_samples + tail_client_non_tail_samples
    tail_client_purity = tail_client_tail_samples / tail_client_total if tail_client_total > 0 else 0.0
    n_tail = int(global_counts[tail_classes].sum())
    n_non_tail = int(global_counts[head_classes].sum())
    q_t = float(args.tail_client_ratio)
    lambda_t = float(args.specialization_lambda)
    rho = float(args.head_leakage_scale)
    tail_to_tail_budget = int(round(n_tail * (q_t + (1.0 - q_t) * lambda_t)))
    tail_to_tail_budget = min(max(tail_to_tail_budget, 0), n_tail)
    non_tail_to_tail_budget = int(round(rho * n_tail * q_t * (1.0 - lambda_t)))
    non_tail_to_tail_budget = min(max(non_tail_to_tail_budget, 0), n_non_tail)

    print("=== Partition Budget And Integrity Checks ===")
    print(f"N_tail = {n_tail}")
    print(f"N_non_tail = {n_non_tail}")
    print(f"expected tail_to_tail_budget = {tail_to_tail_budget}")
    print(f"expected non_tail_to_tail_budget = {non_tail_to_tail_budget}")
    print(f"actual_tail_client_tail_samples = {tail_client_tail_samples}")
    print(f"actual_tail_client_non_tail_samples = {tail_client_non_tail_samples}")
    print(f"actual_tail_client_purity = {tail_client_purity:.6f}")
    print(f"tail budget match: {tail_client_tail_samples == tail_to_tail_budget}")
    print(f"non-tail leakage budget match: {tail_client_non_tail_samples == non_tail_to_tail_budget}")
    for name, ok in checks.items():
        print(f"{name}: {ok}")
    print()

    print("=== Per-Client Class Counts ===")
    for client_id in range(args.num_clients):
        row = client_counts[client_id]
        group = "head" if client_id in head_clients else "tail"
        positive = [
            f"{class_id}:{int(row[class_id])}"
            for class_id in range(num_classes)
            if row[class_id] > 0
        ]
        print(
            f"client {client_id:02d} [{group}] total={int(row.sum())} "
            f"num_classes={len(positive)} classes={{" + ", ".join(positive) + "}"
        )

    if args.csv_dir:
        csv_dir = Path(args.csv_dir)
        write_csv_outputs(
            csv_dir,
            classnames,
            global_counts,
            client_counts,
            head_class_set,
            tail_class_set,
            set(head_clients),
        )
        print()
        print("=== CSV Outputs ===")
        print(csv_dir / "global_class_counts.csv")
        print(csv_dir / "client_class_counts.csv")


if __name__ == "__main__":
    main()
