import argparse
import torch
from Dassl.dassl.utils import setup_logger, set_random_seed
from Dassl.dassl.config import get_cfg_default
from Dassl.dassl.engine import build_trainer
import time
import ast
import os
import copy
import csv
import json
import hashlib
import logging
import math
from utils.fed_utils import average_weights
from utils.experiment_d import (
    append_client_update_norms,
    append_runtime_metrics,
    get_trainable_state_keys,
    run_experiment_d_round,
    should_log_experiment_d,
)
from loss.prompt_loss import PromptLoss, update_class_priors

from trainers.capt import MABScheduler

from sklearn.metrics import silhouette_score
from sklearn.cluster import KMeans
from scipy.cluster.hierarchy import dendrogram, linkage
from scipy.cluster.hierarchy import fcluster
import numpy as np
from sklearn.cluster import DBSCAN
from scipy.cluster.hierarchy import linkage, fcluster
from scipy.spatial.distance import pdist, squareform

logger = logging.getLogger(__name__)


def normalize_dataset_name(dataset_name):
    aliases = {
        "cifar10": "Cifar10",
        "cifar100": "Cifar100",
        "cifar10_LT": "Cifar10_LT",
        "cifar10-LT": "Cifar10_LT",
        "cifar100_LT": "Cifar100_LT",
        "cifar100-LT": "Cifar100_LT",
        "fmnist": "FashionMNIST",
        "fmnist_LT": "FashionMNIST_LT",
        "fmnist-LT": "FashionMNIST_LT",
        "imagenet_LT": "ImageNet_LT",
        "imagenet-LT": "ImageNet_LT",
    }
    return aliases.get(dataset_name, dataset_name)


def collect_prompt_state(prompt_learner):
    prompt_state = {}

    ctx = getattr(prompt_learner, "ctx", None)
    if ctx is not None:
        prompt_state["ctx_prompt"] = ctx.detach().cpu()

    general_ctx = getattr(prompt_learner, "general_ctx", None)
    if general_ctx is not None:
        prompt_state["general_prompt"] = general_ctx.detach().cpu()

    class_aware_ctx = getattr(prompt_learner, "class_aware_ctx", None)
    if class_aware_ctx is not None:
        prompt_state["class_aware_prompt"] = class_aware_ctx.detach().cpu()

    return prompt_state


def should_run_global_eval(epoch, max_epoch, interval):
    interval = max(int(interval or 1), 1)
    if interval <= 1:
        return True
    return epoch == 0 or epoch == max_epoch - 1 or epoch % interval == 0


def print_skip_global_eval(epoch, interval):
    print(
        f"------------global test skipped epoch: {epoch} "
        f"(global_eval_interval={interval})-------------"
    )


def print_cluster_results(clusters, idxs_users):
    cluster_dict = {}
    for i, cluster in enumerate(clusters):
        if cluster not in cluster_dict:
            cluster_dict[cluster] = []
        cluster_dict[cluster].append(idxs_users[i])

    print("Clustering Results:")
    for cluster, members in cluster_dict.items():
        print(f"Cluster {cluster}: Clients {members}")

def js_divergence(p, q):
    m = 0.5 * (p + q)
    return 0.5 * (kl_divergence(p, m) + kl_divergence(q, m))


def kl_divergence(p, q):
    epsilon = 1e-10
    p = np.clip(p, epsilon, 1)
    q = np.clip(q, epsilon, 1)
    return np.sum(p * np.log(p / q))


def similarity_clustering(client_proportions, n_clusters):

    n_clients = len(client_proportions)
    similarity_matrix = np.zeros((n_clients, n_clients))

    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            similarity_matrix[i][j] = similarity_matrix[j][i] = js_divergence(client_proportions[i],
                                                                              client_proportions[j])


    distance_matrix = 1 - similarity_matrix

    kmeans = KMeans(n_clusters=n_clusters, n_init=10)
    clusters = kmeans.fit_predict(distance_matrix)

    return clusters


def dissimilarity_clustering(client_proportions, n_clusters, head_class_threshold=0.8):
    n_clients = len(client_proportions)
    n_classes = len(client_proportions[0])

    class_frequencies = np.sum(client_proportions, axis=0)
    sorted_classes = np.argsort(class_frequencies)[::-1]
    head_classes = sorted_classes[:int(n_classes * head_class_threshold)]
    tail_classes = sorted_classes[int(n_classes * head_class_threshold):]

    complementarity_matrix = np.zeros((n_clients, n_clients))

    for i in range(n_clients):
        for j in range(i + 1, n_clients):
            head_comp = np.sum(client_proportions[i][head_classes] * (1 - client_proportions[j][head_classes]))
            tail_comp = np.sum(client_proportions[i][tail_classes] * (1 - client_proportions[j][tail_classes]))
            complementarity_matrix[i][j] = complementarity_matrix[j][i] = head_comp + tail_comp

    distance_matrix = 1 - complementarity_matrix / np.max(complementarity_matrix)

    linkage_matrix = linkage(distance_matrix[np.triu_indices(n_clients, k=1)], method='complete')
    clusters = fcluster(linkage_matrix, n_clusters, criterion='maxclust')

    return clusters


def get_client_proportions(client_proportion, idxs_users, num_classes):
    client_proportions = []
    for idx in idxs_users:
        if idx in client_proportion:
            proportions = [client_proportion[idx].get(cls, 0) for cls in range(num_classes)]
        else:
            print(f"Warning: No data for client {idx}")
            proportions = [0] * num_classes
        client_proportions.append(proportions)
    return np.array(client_proportions)



def aggregate_class_aware_prompts(client_proportions, local_weights, idxs_users, num_classes, global_prompt):
    aggregated_prompt = global_prompt.clone()  # 使用服务器的prompt参数初始化

    for class_idx in range(num_classes):
        class_weights = []
        class_prompts = []

        for i, client_idx in enumerate(idxs_users):
            if client_proportions[i][class_idx] > 0.1:
                class_weights.append(client_proportions[i][class_idx])
                class_prompts.append(local_weights[client_idx]['prompt_learner.class_aware_ctx'][class_idx].cpu())

        if class_prompts:
            class_weights = torch.tensor(class_weights)
            class_prompts = torch.stack(class_prompts)

            # # 加权平均
            # weighted_prompt = torch.sum(class_prompts * class_weights.unsqueeze(-1).unsqueeze(-1),dim=0) / class_weights.sum()
            # aggregated_prompt[class_idx] = weighted_prompt

            # 直接平均
            average_prompt = torch.mean(class_prompts, dim=0)
            aggregated_prompt[class_idx] = average_prompt

    return aggregated_prompt



def communicate_within_cluster_similarity(cluster_members, local_weights):

    class_aware_prompts = [local_weights[i]['prompt_learner.class_aware_ctx'].cpu() for i in cluster_members]
    aggregated_class_aware_prompt = torch.mean(torch.stack(class_aware_prompts), dim=0)


    for i in cluster_members:
        local_weights[i]['prompt_learner.class_aware_ctx'] = aggregated_class_aware_prompt.to(
            local_weights[i]['prompt_learner.class_aware_ctx'].device)


def communicate_within_cluster_dissimilarity(cluster_members, local_weights):
    general_prompts = [local_weights[i]['prompt_learner.general_ctx'].cpu() for i in cluster_members]
    aggregated_general_prompt = torch.mean(torch.stack(general_prompts), dim=0)


    for i in cluster_members:
        local_weights[i]['prompt_learner.general_ctx'] = aggregated_general_prompt.to(
            local_weights[i]['prompt_learner.general_ctx'].device)


def get_lt_class_splits_from_counts(global_class_counts, tail_class_ratio=0.2):
    counts = torch.as_tensor(global_class_counts, dtype=torch.float32)
    sorted_classes = torch.argsort(counts, descending=True).tolist()
    num_classes = len(sorted_classes)
    tail_count = max(1, int(round(num_classes * float(tail_class_ratio))))
    tail_count = min(tail_count, num_classes)
    tail_classes = sorted_classes[-tail_count:]
    head_classes = sorted_classes[:-tail_count]
    return {
        "head": head_classes,
        "tail": tail_classes,
        "all": sorted_classes,
    }


def _mean_class_accuracy(class_accuracy, class_ids):
    values = [float(class_accuracy.get(cls, 0.0)) for cls in class_ids]
    return float(np.mean(values)) if values else 0.0


def calculate_accuracy_tail20(class_accuracy, local_trainer, tail_class_ratio=0.2):
    cls_num_list = local_trainer.cls_num_list
    splits = get_lt_class_splits_from_counts(cls_num_list, tail_class_ratio)
    head_classes = splits["head"]
    tail_classes = splits["tail"]

    head_acc = _mean_class_accuracy(class_accuracy, head_classes)
    tail_acc = _mean_class_accuracy(class_accuracy, tail_classes)
    macro_acc = _mean_class_accuracy(class_accuracy, splits["all"])

    print(f"Macro per-class accuracy: {macro_acc:.2f}%")
    print(f"Head/non-tail accuracy (top {len(head_classes)} classes): {head_acc:.2f}%")
    print(f"Tail accuracy (bottom {len(tail_classes)} classes): {tail_acc:.2f}%")
    print(f"Tail class ids: {tail_classes}")

    return head_acc, tail_acc, macro_acc


def calculate_accuracy_tail20_compat(class_accuracy, local_trainer, tail_class_ratio=0.2):
    head_acc, tail_acc, macro_acc = calculate_accuracy_tail20(
        class_accuracy,
        local_trainer,
        tail_class_ratio,
    )
    return head_acc, 0.0, tail_acc, macro_acc


def calculate_accuracy_cumulative_legacy(class_accuracy, local_trainer):
    # 确定数据集类型
    num_classes = len(class_accuracy)

    # 使用样本数量作为排序依据
    sorted_classes = sorted(range(num_classes), key=lambda k: local_trainer.cls_num_list[k], reverse=True)
    # print(sorted_classes)

    total_samples = sum(local_trainer.cls_num_list)
    cumulative_samples = 0
    head_threshold = 0.75 * total_samples
    medium_threshold = 0.95 * total_samples

    head_acc = []
    medium_acc = []
    tail_acc = []

    for cls in sorted_classes:
        if cls in class_accuracy:
            cls_count = local_trainer.cls_num_list[cls]
            cumulative_samples += cls_count

            if cumulative_samples <= head_threshold:
                head_acc.append(class_accuracy[cls])
            elif cumulative_samples <= medium_threshold:
                medium_acc.append(class_accuracy[cls])
            else:
                tail_acc.append(class_accuracy[cls])

    head_acc_mean = np.mean(head_acc) if head_acc else 0
    medium_acc_mean = np.mean(medium_acc) if medium_acc else 0
    tail_acc_mean = np.mean(tail_acc) if tail_acc else 0
    overall_acc = np.mean(list(class_accuracy.values()))

    print(f"Overall accuracy: {overall_acc:.2f}%")
    print(f"Legacy head accuracy (cumulative top 75% samples): {head_acc_mean:.2f}%")
    print(f"Legacy medium accuracy (cumulative 75%-95% samples): {medium_acc_mean:.2f}%")
    print(f"Legacy tail accuracy (cumulative bottom 5% samples): {tail_acc_mean:.2f}%")

    return head_acc_mean, medium_acc_mean, tail_acc_mean, overall_acc


def str2bool(value):
    if isinstance(value, bool):
        return value
    if str(value).lower() in ("yes", "true", "t", "1", "y"):
        return True
    if str(value).lower() in ("no", "false", "f", "0", "n"):
        return False
    raise argparse.ArgumentTypeError(f"Invalid boolean value: {value}")


def add_expF_runtime_arguments(parser):
    parser.add_argument("--local_epochs", type=int, default=None, help="Override the number of local training epochs per selected client.")
    parser.add_argument("--split_seed", type=int, default=1, help="Random seed used only for client data partitioning.")
    parser.add_argument("--isolate_local_optimizer_state", type=str2bool, default=False, help="Rebuild local optimizer, scheduler, and AMP scaler before each selected client.")
    parser.add_argument("--federated_single_scheduler_step", type=str2bool, default=False, help="Use exactly one scheduler step per federated local epoch.")


def apply_federated_runtime_overrides(cfg, args):
    cfg.DATASET.SPLIT_SEED = int(args.split_seed)

    if args.local_epochs is None:
        return

    local_epochs = int(args.local_epochs)
    if local_epochs <= 0:
        raise ValueError(f"--local_epochs must be > 0, got {args.local_epochs}")

    cfg.OPTIM.MAX_EPOCH = local_epochs


def get_optimizer_state_entries(trainer):
    optim = getattr(trainer, "optim", None)
    if optim is None:
        return 0
    return len(getattr(optim, "state", {}))


def get_first_optimizer_lr(trainer):
    optim = getattr(trainer, "optim", None)
    param_groups = getattr(optim, "param_groups", None)
    if not param_groups:
        return None
    return float(param_groups[0]["lr"])


def get_scheduler_last_epoch(trainer):
    sched = getattr(trainer, "sched", None)
    if sched is None or not hasattr(sched, "last_epoch"):
        return None
    return int(getattr(sched, "last_epoch"))


def validate_scheduler_step_delta(client_id, local_epochs, before_last_epoch, after_last_epoch):
    if before_last_epoch is None or after_last_epoch is None:
        logger.warning(
            "Scheduler for client %s does not expose last_epoch; skipping runtime delta assertion.",
            client_id,
        )
        return None

    observed_delta = int(after_last_epoch) - int(before_last_epoch)
    expected_delta = int(local_epochs)
    if observed_delta != expected_delta:
        raise RuntimeError(
            "Federated single scheduler step violation for client "
            f"{client_id}: local_epochs={expected_delta}, "
            f"before_last_epoch={before_last_epoch}, after_last_epoch={after_last_epoch}, "
            f"observed_delta={observed_delta}"
        )
    return observed_delta


def install_scheduler_step_counter(trainer):
    schedulers = []
    sched = getattr(trainer, "sched", None)
    if sched is not None and hasattr(sched, "step"):
        schedulers.append(sched)

    for sched in getattr(trainer, "_scheds", {}).values():
        if sched is not None and hasattr(sched, "step"):
            schedulers.append(sched)

    unique_schedulers = []
    seen = set()
    for sched in schedulers:
        sched_id = id(sched)
        if sched_id in seen:
            continue
        seen.add(sched_id)
        unique_schedulers.append(sched)

    if not unique_schedulers:
        return None, lambda: None

    counter = {"steps": 0}
    originals = []

    for sched in unique_schedulers:
        original_step = sched.step

        def counted_step(*args, _original_step=original_step, **kwargs):
            counter["steps"] += 1
            return _original_step(*args, **kwargs)

        sched.step = counted_step
        originals.append((sched, original_step))

    def restore():
        for sched, original_step in originals:
            sched.step = original_step

    return counter, restore


def run_promptfl_local_train_with_scheduler_policy(local_trainer, idx, epoch, args, local_epochs):
    if not bool(args.federated_single_scheduler_step):
        optimizer_step_counter, restore_optimizer_step_counter = install_optimizer_step_counter(local_trainer)
        try:
            local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
        finally:
            restore_optimizer_step_counter()
        optimizer_steps = optimizer_step_counter["steps"] if optimizer_step_counter is not None else None
        if optimizer_steps is not None and optimizer_steps < 1:
            raise RuntimeError(
                f"Federated local optimizer did not step for client {idx}: "
                f"local_optimizer_step_count={optimizer_steps}"
            )
        return None, None, None, optimizer_steps

    before_last_epoch = get_scheduler_last_epoch(local_trainer)
    previous_value = getattr(local_trainer, "_skip_scheduler_step_at_epoch_start", False)
    step_counter, restore_step_counter = install_scheduler_step_counter(local_trainer)
    optimizer_step_counter, restore_optimizer_step_counter = install_optimizer_step_counter(local_trainer)

    try:
        try:
            local_trainer._skip_scheduler_step_at_epoch_start = True
            local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
        finally:
            restore_optimizer_step_counter()
            restore_step_counter()
    finally:
        local_trainer._skip_scheduler_step_at_epoch_start = previous_value

    after_last_epoch = get_scheduler_last_epoch(local_trainer)
    if step_counter is not None:
        observed_delta = int(step_counter["steps"])
        expected_delta = int(local_epochs)
        if observed_delta != expected_delta:
            raise RuntimeError(
                "Federated single scheduler step violation for client "
                f"{idx}: local_epochs={expected_delta}, "
                f"before_last_epoch={before_last_epoch}, after_last_epoch={after_last_epoch}, "
                f"observed_delta={observed_delta}"
            )
    else:
        observed_delta = validate_scheduler_step_delta(
            idx,
            local_epochs,
            before_last_epoch,
            after_last_epoch,
        )
    optimizer_steps = optimizer_step_counter["steps"] if optimizer_step_counter is not None else None
    if optimizer_steps is not None and optimizer_steps < 1:
        raise RuntimeError(
            f"Federated local optimizer did not step for client {idx}: "
            f"local_optimizer_step_count={optimizer_steps}"
        )
    return before_last_epoch, after_last_epoch, observed_delta, optimizer_steps


def load_or_create_client_schedule(path, num_rounds, num_users, frac, seed):
    if not path:
        return None

    m = max(int(float(frac) * int(num_users)), 1)
    if os.path.exists(path):
        with open(path, "r", encoding="utf-8") as f:
            payload = json.load(f)
        schedule = payload.get("schedule", payload) if isinstance(payload, dict) else payload
        print(f"Loaded fixed client schedule from {path}")
    else:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        rng = np.random.default_rng(int(seed))
        schedule = [
            [int(x) for x in rng.choice(int(num_users), m, replace=False).tolist()]
            for _ in range(int(num_rounds))
        ]
        payload = {
            "num_rounds": int(num_rounds),
            "num_users": int(num_users),
            "frac": float(frac),
            "clients_per_round": int(m),
            "seed": int(seed),
            "schedule": schedule,
        }
        tmp_path = f"{path}.tmp.{os.getpid()}"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(payload, f, indent=2)
        os.replace(tmp_path, path)
        print(f"Created fixed client schedule at {path}")

    if len(schedule) < int(num_rounds):
        raise ValueError(f"Client schedule has {len(schedule)} rounds, expected at least {num_rounds}")
    for round_idx, clients in enumerate(schedule[:int(num_rounds)]):
        if len(clients) != m:
            raise ValueError(f"Round {round_idx} has {len(clients)} clients, expected {m}")
        if len(set(int(x) for x in clients)) != len(clients):
            raise ValueError(f"Round {round_idx} has duplicate clients: {clients}")
        if any(int(x) < 0 or int(x) >= int(num_users) for x in clients):
            raise ValueError(f"Round {round_idx} has out-of-range clients: {clients}")
    return schedule


def validate_federated_train_loaders(local_trainer, num_users):
    print("Federated train loader diagnostics:")
    for client_id in range(int(num_users)):
        loader = local_trainer.fed_train_loader_x_dict[client_id]
        dataset = loader.dataset
        sample_count = len(dataset)
        batch_count = len(loader)
        batch_size = getattr(loader, "batch_size", None)
        drop_last = getattr(loader, "drop_last", None)
        print(
            "client_id=%d client_num_samples=%d local_batch_count=%d "
            "batch_size=%s drop_last=%s"
            % (client_id, sample_count, batch_count, batch_size, drop_last)
        )
        if batch_count < 1:
            raise RuntimeError(
                "Federated client has zero training batches: "
                f"client_id={client_id}, client_num_samples={sample_count}, "
                f"batch_size={batch_size}, drop_last={drop_last}"
            )


def install_optimizer_step_counter(trainer):
    optimizers = []
    optim = getattr(trainer, "optim", None)
    if optim is not None and hasattr(optim, "step"):
        optimizers.append(optim)
    for optim in getattr(trainer, "_optims", {}).values():
        if optim is not None and hasattr(optim, "step"):
            optimizers.append(optim)

    unique_optimizers = []
    seen = set()
    for optim in optimizers:
        optim_id = id(optim)
        if optim_id in seen:
            continue
        seen.add(optim_id)
        unique_optimizers.append(optim)

    if not unique_optimizers:
        return None, lambda: None

    counter = {"steps": 0}
    originals = []
    for optim in unique_optimizers:
        original_step = optim.step

        def counted_step(*args, _original_step=original_step, **kwargs):
            counter["steps"] += 1
            return _original_step(*args, **kwargs)

        optim.step = counted_step
        originals.append((optim, original_step))

    def restore():
        for optim, original_step in originals:
            optim.step = original_step

    return counter, restore


def select_round_clients(args, epoch, client_schedule=None):
    if client_schedule is not None:
        return np.array(client_schedule[int(epoch)], dtype=int)
    m = max(int(args.frac * args.num_users), 1)
    return np.random.choice(range(args.num_users), m, replace=False)


def get_client_class_counts(local_trainer, num_users, num_classes):
    client_class_counts = {}
    federated_train_x = getattr(local_trainer.dm.dataset, "federated_train_x", None)

    for client_idx in range(num_users):
        counts = torch.zeros(num_classes, dtype=torch.float32)
        if federated_train_x is not None and client_idx < len(federated_train_x):
            data_source = federated_train_x[client_idx]
        else:
            data_source = local_trainer.fed_train_loader_x_dict[client_idx].dataset.data_source

        for item in data_source:
            label = item["label"] if isinstance(item, dict) else item.label
            counts[int(label)] += 1
        client_class_counts[client_idx] = counts

    return client_class_counts


def _item_identity(item):
    if isinstance(item, dict):
        label = item.get("label", "")
        impath = item.get("impath", "")
    else:
        label = getattr(item, "label", "")
        impath = getattr(item, "impath", "")
    return f"{int(label)}::{str(impath)}"


def save_client_split_fingerprint(output_dir, local_trainer, num_users):
    os.makedirs(output_dir, exist_ok=True)
    federated_train_x = getattr(local_trainer.dm.dataset, "federated_train_x", None)
    payload = {"clients": {}, "global_ordered_sha256": None, "global_membership_sha256": None}
    global_ordered = hashlib.sha256()
    global_membership_items = []

    for client_idx in range(num_users):
        if federated_train_x is not None and client_idx < len(federated_train_x):
            data_source = federated_train_x[client_idx]
        else:
            data_source = local_trainer.fed_train_loader_x_dict[client_idx].dataset.data_source

        ordered = [_item_identity(item) for item in data_source]
        ordered_hash = hashlib.sha256("\n".join(ordered).encode("utf-8")).hexdigest()
        membership_hash = hashlib.sha256("\n".join(sorted(ordered)).encode("utf-8")).hexdigest()
        payload["clients"][str(client_idx)] = {
            "num_samples": len(ordered),
            "ordered_sha256": ordered_hash,
            "membership_sha256": membership_hash,
        }
        global_ordered.update(f"client:{client_idx}\n".encode("utf-8"))
        global_ordered.update("\n".join(ordered).encode("utf-8"))
        global_membership_items.extend([f"{client_idx}::{x}" for x in ordered])

    payload["global_ordered_sha256"] = global_ordered.hexdigest()
    payload["global_membership_sha256"] = hashlib.sha256(
        "\n".join(sorted(global_membership_items)).encode("utf-8")
    ).hexdigest()
    with open(os.path.join(output_dir, "client_split_fingerprint.json"), "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(f"Saved client split fingerprint to {output_dir}")


def client_counts_to_tensor(client_class_counts, num_users, num_classes):
    counts = torch.zeros(num_users, num_classes, dtype=torch.float32)
    for client_idx in range(num_users):
        if client_idx in client_class_counts:
            counts[client_idx] = client_class_counts[client_idx].float().cpu()
    return counts


def get_class_groups_from_counts(global_class_counts, tail_class_ratio=0.2):
    splits = get_lt_class_splits_from_counts(global_class_counts, tail_class_ratio)
    class_groups = {int(cls): "head" for cls in splits["head"]}
    class_groups.update({int(cls): "tail" for cls in splits["tail"]})
    return class_groups


def get_oracle_tail_mask_from_counts(global_class_counts, tail_class_ratio=0.2):
    splits = get_lt_class_splits_from_counts(global_class_counts, tail_class_ratio)
    mask = torch.zeros(len(global_class_counts), dtype=torch.bool)
    for cls in splits["tail"]:
        mask[int(cls)] = True
    if not mask.any():
        mask[int(torch.argmin(torch.as_tensor(global_class_counts)).item())] = True
    return mask


def save_partition_summary(output_dir, client_class_counts, args, num_users, num_classes):
    os.makedirs(output_dir, exist_ok=True)
    counts = client_counts_to_tensor(client_class_counts, num_users, num_classes)
    support = counts > 0
    global_counts = counts.sum(dim=0)
    class_num_clients = support.sum(dim=0)
    sorted_counts = torch.sort(counts, dim=0, descending=True).values
    top1_per_class = sorted_counts[0] if num_users > 0 else torch.zeros(num_classes)
    top2_per_class = sorted_counts[:2].sum(dim=0) if num_users > 1 else top1_per_class
    top1_client_mass = top1_per_class / torch.clamp(global_counts, min=1.0)
    top2_client_mass = top2_per_class / torch.clamp(global_counts, min=1.0)
    squared_mass_sum = torch.clamp((counts ** 2).sum(dim=0), min=1.0)
    effective_client_number = (global_counts ** 2) / squared_mass_sum
    concentration = top1_client_mass
    topology_index = 1.0 / torch.clamp(class_num_clients.float(), min=1.0)
    tail_class_ratio = getattr(args, "tail_class_ratio", 0.2)
    splits = get_lt_class_splits_from_counts(global_counts, tail_class_ratio)
    class_groups = get_class_groups_from_counts(global_counts, tail_class_ratio)
    tail_classes = [int(x) for x in splits["tail"]]
    head_classes = [int(x) for x in splits["head"]]

    def _mean_for_classes(values, class_ids):
        if not class_ids:
            return 0.0
        idx = torch.as_tensor(class_ids, dtype=torch.long)
        return float(values[idx].float().mean().item())

    client_sample_counts = counts.sum(dim=1)
    client_sample_mean = float(client_sample_counts.mean().item()) if num_users > 0 else 0.0
    client_sample_std = float(client_sample_counts.std(unbiased=False).item()) if num_users > 0 else 0.0
    client_sample_cv = client_sample_std / client_sample_mean if client_sample_mean > 0 else 0.0

    tail_client_ratio = float(getattr(args, "tail_client_ratio", 0.0))
    num_tail_clients = int(round(num_users * tail_client_ratio))
    num_tail_clients = min(max(num_tail_clients, 0), num_users)
    num_head_clients = num_users - num_tail_clients
    tail_client_ids = list(range(num_head_clients, num_users))
    head_client_ids = list(range(0, num_head_clients))

    tail_samples_in_tail_clients = (
        float(counts[tail_client_ids][:, tail_classes].sum().item())
        if tail_client_ids and tail_classes
        else 0.0
    )
    non_tail_samples_in_tail_clients = (
        float(counts[tail_client_ids][:, head_classes].sum().item())
        if tail_client_ids and head_classes
        else 0.0
    )
    tail_samples_in_head_clients = (
        float(counts[head_client_ids][:, tail_classes].sum().item())
        if head_client_ids and tail_classes
        else 0.0
    )
    non_tail_samples_in_head_clients = (
        float(counts[head_client_ids][:, head_classes].sum().item())
        if head_client_ids and head_classes
        else 0.0
    )
    tail_client_total = tail_samples_in_tail_clients + non_tail_samples_in_tail_clients
    actual_tail_client_purity = (
        tail_samples_in_tail_clients / tail_client_total if tail_client_total > 0 else 0.0
    )

    summary = {
        "partition": args.partition,
        "method": args.trainer,
        "seed": args.seed,
        "num_clients": num_users,
        "num_classes": num_classes,
        "global_class_counts": [float(x) for x in global_counts.tolist()],
        "class_num_clients": [int(x) for x in class_num_clients.tolist()],
        "num_support_clients": [int(x) for x in class_num_clients.tolist()],
        "tail_topology_index": [float(x) for x in topology_index.tolist()],
        "concentration": [float(x) for x in concentration.tolist()],
        "top1_client_mass": [float(x) for x in top1_client_mass.tolist()],
        "top2_client_mass": [float(x) for x in top2_client_mass.tolist()],
        "effective_client_number": [float(x) for x in effective_client_number.tolist()],
        "class_group": {str(k): v for k, v in class_groups.items()},
        "tail_class_ratio": float(tail_class_ratio),
        "tail_classes": tail_classes,
        "head_classes": head_classes,
        "tail_num_support_clients_mean": _mean_for_classes(class_num_clients, tail_classes),
        "tail_top1_client_mass_mean": _mean_for_classes(top1_client_mass, tail_classes),
        "tail_top2_client_mass_mean": _mean_for_classes(top2_client_mass, tail_classes),
        "tail_effective_client_number_mean": _mean_for_classes(effective_client_number, tail_classes),
        "client_sample_min": float(client_sample_counts.min().item()) if num_users > 0 else 0.0,
        "client_sample_max": float(client_sample_counts.max().item()) if num_users > 0 else 0.0,
        "client_sample_mean": client_sample_mean,
        "client_sample_std": client_sample_std,
        "client_sample_cv": client_sample_cv,
        "specialization_lambda": (
            float(args.specialization_lambda) if hasattr(args, "specialization_lambda") else None
        ),
        "intra_group_alpha": (
            float(args.intra_group_alpha) if hasattr(args, "intra_group_alpha") else None
        ),
        "head_leakage_scale": (
            float(args.head_leakage_scale) if hasattr(args, "head_leakage_scale") else None
        ),
        "num_tail_clients": num_tail_clients,
        "tail_client_ids": tail_client_ids,
        "head_client_ids": head_client_ids,
        "tail_samples_in_tail_clients": tail_samples_in_tail_clients,
        "non_tail_samples_in_tail_clients": non_tail_samples_in_tail_clients,
        "tail_samples_in_head_clients": tail_samples_in_head_clients,
        "non_tail_samples_in_head_clients": non_tail_samples_in_head_clients,
        "tail_to_tail_budget": tail_samples_in_tail_clients,
        "non_tail_to_tail_budget": non_tail_samples_in_tail_clients,
        "actual_tail_client_purity": actual_tail_client_purity,
    }
    with open(os.path.join(output_dir, "partition_summary.json"), "w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    with open(os.path.join(output_dir, "client_class_counts.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["client_id"] + [f"class_{c}" for c in range(num_classes)])
        for client_idx in range(num_users):
            writer.writerow([client_idx] + [int(x) for x in counts[client_idx].tolist()])

    with open(os.path.join(output_dir, "class_topology.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_id",
                "class_group",
                "global_count",
                "num_support_clients",
                "top1_client_mass",
                "top2_client_mass",
                "effective_client_number",
                "tail_topology_index",
                "concentration",
            ],
        )
        writer.writeheader()
        for cls in range(num_classes):
            writer.writerow({
                "class_id": cls,
                "class_group": class_groups.get(cls, "head"),
                "global_count": float(global_counts[cls].item()),
                "num_support_clients": int(class_num_clients[cls].item()),
                "top1_client_mass": float(top1_client_mass[cls].item()),
                "top2_client_mass": float(top2_client_mass[cls].item()),
                "effective_client_number": float(effective_client_number[cls].item()),
                "tail_topology_index": float(topology_index[cls].item()),
                "concentration": float(concentration[cls].item()),
            })

    print(f"Saved partition summary to {output_dir}")


def append_round_metrics(output_dir, args, epoch, result, non_tail_acc, tail_acc, macro_acc, per_class_acc):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "round_metrics.csv")
    fieldnames = [
        "epoch",
        "method",
        "partition",
        "overall_acc",
        "non_tail_acc",
        "bottom20_tail_acc",
        "macro_per_class_acc",
        "head_acc",
        "medium_acc",
        "tail_acc",
        "macro_f1",
        "seed",
        "tail_class_ratio",
    ]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "epoch": epoch,
            "method": "FedTEF" if args.trainer == "FedTEF" else args.trainer,
            "partition": args.partition,
            "overall_acc": float(result[0]),
            "non_tail_acc": float(non_tail_acc),
            "bottom20_tail_acc": float(tail_acc),
            "macro_per_class_acc": float(macro_acc),
            "head_acc": float(non_tail_acc),
            "medium_acc": 0.0,
            "tail_acc": float(tail_acc),
            "macro_f1": float(result[2]) if len(result) > 2 else "",
            "seed": args.seed,
            "tail_class_ratio": getattr(args, "tail_class_ratio", 0.2),
        })

    per_class_path = os.path.join(output_dir, f"per_class_accuracy_epoch_{epoch}.csv")
    with open(per_class_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["class_id", "per_class_acc"])
        writer.writeheader()
        for cls in range(len(per_class_acc)):
            writer.writerow({"class_id": cls, "per_class_acc": float(per_class_acc.get(cls, 0.0))})


def should_log_update_retention(args, epoch, max_epoch):
    if not getattr(args, "log_update_retention", False):
        return False
    interval = max(int(getattr(args, "update_retention_interval", 1) or 1), 1)
    return epoch == 0 or epoch == max_epoch - 1 or epoch % interval == 0


def append_update_retention(
    output_dir,
    args,
    epoch,
    global_before,
    local_weights,
    idxs_users,
    datanumber_client,
    client_class_counts,
    num_classes,
    tail_class_ratio=0.2,
    eps=1e-12,
):
    """Log how much class-wise local prompt update survives FedAvg."""
    param_key = getattr(args, "update_retention_param_key", "prompt_learner.class_aware_ctx")
    if param_key not in global_before:
        print(f"Update-retention logging skipped: missing key {param_key}")
        return

    base_value = global_before[param_key].detach().float().cpu()
    if base_value.ndim < 2 or base_value.shape[0] != num_classes:
        print(
            "Update-retention logging skipped: "
            f"{param_key} shape {tuple(base_value.shape)} is not class-wise"
        )
        return

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "update_retention_per_class.csv")
    counts = client_counts_to_tensor(client_class_counts, args.num_users, num_classes).float().cpu()
    global_counts = counts.sum(dim=0)
    class_groups = get_class_groups_from_counts(global_counts, tail_class_ratio)

    selected = [int(x) for x in idxs_users]
    total_selected_samples = sum(float(datanumber_client[idx]) for idx in selected)
    if total_selected_samples <= 0:
        total_selected_samples = float(len(selected))
    alphas = {
        idx: float(datanumber_client[idx]) / total_selected_samples
        for idx in selected
    }

    selected_deltas = {}
    for idx in selected:
        if isinstance(local_weights, dict):
            has_local = idx in local_weights
        else:
            has_local = idx < len(local_weights) and isinstance(local_weights[idx], dict)
        if not has_local or param_key not in local_weights[idx]:
            print(f"Update-retention logging skipped: missing local {param_key} for client {idx}")
            return
        selected_deltas[idx] = (
            local_weights[idx][param_key].detach().float().cpu() - base_value
        ).reshape(num_classes, -1)

    fieldnames = [
        "epoch",
        "method",
        "partition",
        "frac",
        "seed",
        "param_key",
        "class_id",
        "class_group",
        "global_count",
        "selected_clients",
        "selected_support_clients",
        "selected_support_client_rate",
        "support_weight_sum",
        "selected_support_samples",
        "sample_exposure_rate",
        "support_update_norm_mean",
        "support_update_norm_weighted",
        "non_support_update_norm_weighted",
        "all_update_norm_weighted",
        "support_delta_norm",
        "global_delta_norm",
        "retention_ratio",
        "dilution_ratio",
        "support_cancellation_rate",
        "all_cancellation_rate",
        "direction_cosine",
    ]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for cls in range(num_classes):
            support_clients = [
                idx for idx in selected
                if counts[idx, cls].item() > 0
            ]
            support_weight_sum = sum(alphas[idx] for idx in support_clients)
            selected_support_samples = sum(float(counts[idx, cls].item()) for idx in support_clients)
            sample_exposure_rate = selected_support_samples / max(float(global_counts[cls].item()), eps)

            support_delta = torch.zeros_like(selected_deltas[selected[0]][cls])
            global_delta = torch.zeros_like(support_delta)
            support_norms = []
            support_update_norm_weighted = 0.0
            non_support_update_norm_weighted = 0.0
            all_update_norm_weighted = 0.0

            for idx in selected:
                delta = selected_deltas[idx][cls]
                delta_norm = float(torch.linalg.vector_norm(delta).item())
                weighted_delta = float(alphas[idx]) * delta
                global_delta += weighted_delta
                all_update_norm_weighted += float(alphas[idx]) * delta_norm
                if idx in support_clients:
                    support_delta += weighted_delta
                    support_norms.append(delta_norm)
                    support_update_norm_weighted += float(alphas[idx]) * delta_norm
                else:
                    non_support_update_norm_weighted += float(alphas[idx]) * delta_norm

            support_delta_norm = float(torch.linalg.vector_norm(support_delta).item())
            global_delta_norm = float(torch.linalg.vector_norm(global_delta).item())
            support_update_norm_mean = float(np.mean(support_norms)) if support_norms else 0.0
            if support_update_norm_weighted > eps:
                retention_ratio = global_delta_norm / (support_update_norm_weighted + eps)
                dilution_ratio = 1.0 - retention_ratio
                support_cancellation_rate = 1.0 - support_delta_norm / (support_update_norm_weighted + eps)
            else:
                retention_ratio = ""
                dilution_ratio = ""
                support_cancellation_rate = ""
            all_cancellation_rate = (
                1.0 - global_delta_norm / (all_update_norm_weighted + eps)
                if all_update_norm_weighted > eps
                else ""
            )
            if support_delta_norm > eps and global_delta_norm > eps:
                direction_cosine = float(torch.dot(support_delta, global_delta).item() / (support_delta_norm * global_delta_norm + eps))
            else:
                direction_cosine = ""

            writer.writerow({
                "epoch": int(epoch),
                "method": args.trainer,
                "partition": args.partition,
                "frac": float(args.frac),
                "seed": int(args.seed),
                "param_key": param_key,
                "class_id": cls,
                "class_group": class_groups.get(cls, "head"),
                "global_count": float(global_counts[cls].item()),
                "selected_clients": len(selected),
                "selected_support_clients": len(support_clients),
                "selected_support_client_rate": len(support_clients) / max(len(selected), 1),
                "support_weight_sum": support_weight_sum,
                "selected_support_samples": selected_support_samples,
                "sample_exposure_rate": sample_exposure_rate,
                "support_update_norm_mean": support_update_norm_mean,
                "support_update_norm_weighted": support_update_norm_weighted,
                "non_support_update_norm_weighted": non_support_update_norm_weighted,
                "all_update_norm_weighted": all_update_norm_weighted,
                "support_delta_norm": support_delta_norm,
                "global_delta_norm": global_delta_norm,
                "retention_ratio": retention_ratio,
                "dilution_ratio": dilution_ratio,
                "support_cancellation_rate": support_cancellation_rate,
                "all_cancellation_rate": all_cancellation_rate,
                "direction_cosine": direction_cosine,
            })

    print(f"Appended update-retention diagnostics to {path}")


def _weighted_fedavg_delta(global_before, local_weights, idxs_users, datanumber_client, key):
    selected = [int(x) for x in idxs_users]
    total_weight = sum(float(datanumber_client[idx]) for idx in selected)
    if total_weight <= 0:
        total_weight = float(len(selected))
    base = global_before[key].detach().float().cpu()
    delta = torch.zeros_like(base)
    local_norm_weighted = 0.0
    for idx in selected:
        alpha = float(datanumber_client[idx]) / total_weight
        local_delta = local_weights[idx][key].detach().float().cpu() - base
        delta += alpha * local_delta
        local_norm_weighted += alpha * float(local_delta.norm().item())
    return delta, local_norm_weighted


def append_fedtef_shared_stream_diagnostics(
    output_dir,
    args,
    epoch,
    global_before,
    local_weights,
    idxs_users,
    datanumber_client,
    shared_keys,
    eps=1e-12,
):
    if not getattr(args, "fedtef_log_diagnostics", False):
        return

    def stream_name(key):
        if "lora_" in key:
            return "lora"
        if "img_adap" in key:
            return "img_adapter"
        if "prompt_learner" in key:
            return "prompt"
        return "other_shared"

    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "shared_stream_update_diagnostics.csv")
    grouped = {"shared_total": list(shared_keys)}
    for key in shared_keys:
        grouped.setdefault(stream_name(key), []).append(key)

    fieldnames = [
        "epoch",
        "seed",
        "partition",
        "stream",
        "num_tensors",
        "num_parameters",
        "local_update_norm_weighted",
        "fedavg_update_norm",
        "cancellation_rate",
    ]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for stream, keys in sorted(grouped.items()):
            local_norm = 0.0
            fedavg_norm = 0.0
            num_parameters = 0
            for key in keys:
                delta, weighted_norm = _weighted_fedavg_delta(
                    global_before,
                    local_weights,
                    idxs_users,
                    datanumber_client,
                    key,
                )
                local_norm += weighted_norm
                fedavg_norm += float(delta.norm().item())
                num_parameters += int(global_before[key].numel())
            cancellation_rate = (
                1.0 - fedavg_norm / (local_norm + eps)
                if local_norm > eps
                else 0.0
            )
            writer.writerow({
                "epoch": int(epoch),
                "seed": int(args.seed),
                "partition": args.partition,
                "stream": stream,
                "num_tensors": len(keys),
                "num_parameters": num_parameters,
                "local_update_norm_weighted": local_norm,
                "fedavg_update_norm": fedavg_norm,
                "cancellation_rate": cancellation_rate,
            })
    print(f"Appended FedTEF shared-stream diagnostics to {path}")


def build_fedavg_tail_diagnostics(
    global_before,
    global_after,
    local_weights,
    idxs_users,
    num_classes,
    eps=1e-12,
):
    tail_keys = [key for key in global_before if is_tail_stream_key(key)]
    classwise_keys = [
        key for key in tail_keys
        if global_before[key].ndim >= 1 and global_before[key].shape[0] == num_classes
    ]
    local_energy = torch.zeros(num_classes, dtype=torch.float32)
    observed_client_count = torch.zeros(num_classes, dtype=torch.float32)
    fedavg_row_energy = torch.zeros(num_classes, dtype=torch.float32)
    memory_row_norm = torch.zeros(num_classes, dtype=torch.float32)
    for idx in [int(x) for x in idxs_users]:
        client_energy = torch.zeros(num_classes, dtype=torch.float32)
        for key in classwise_keys:
            delta = (
                local_weights[idx][key].detach().float().cpu()
                - global_before[key].detach().float().cpu()
            ).reshape(num_classes, -1)
            client_energy += torch.linalg.vector_norm(delta, dim=1)
        local_energy += client_energy
        observed_client_count += (client_energy > float(eps)).float()
    for key in classwise_keys:
        delta = (
            global_after[key].detach().float().cpu()
            - global_before[key].detach().float().cpu()
        ).reshape(num_classes, -1)
        rows = global_after[key].detach().float().cpu().reshape(num_classes, -1)
        fedavg_row_energy += torch.linalg.vector_norm(delta, dim=1)
        memory_row_norm += torch.linalg.vector_norm(rows, dim=1)
    return {
        "mode": "fedavg",
        "fallback_count": 0,
        "tailagg_norm": float(fedavg_row_energy.sum().item()),
        "fedavg_norm": float(fedavg_row_energy.sum().item()),
        "local_energy_sum": local_energy,
        "local_energy_mean_observed": local_energy / torch.clamp(observed_client_count, min=1.0),
        "observed_client_count": observed_client_count,
        "fedavg_row_energy": fedavg_row_energy,
        "tailagg_row_energy": fedavg_row_energy.clone(),
        "memory_row_norm": memory_row_norm,
    }


def append_fedtef_tailagg_diagnostics(
    output_dir,
    args,
    epoch,
    diagnostics,
    global_class_counts,
    tail_class_ratio=0.2,
    eps=1e-12,
):
    if not getattr(args, "fedtef_log_diagnostics", False) or diagnostics is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "tailagg_diagnostics_per_class.csv")
    class_groups = get_class_groups_from_counts(global_class_counts, tail_class_ratio)
    fieldnames = [
        "epoch",
        "seed",
        "partition",
        "mode",
        "class_id",
        "class_group",
        "global_count",
        "observed_client_count",
        "local_energy_sum",
        "local_energy_mean_observed",
        "fedavg_row_energy",
        "tailagg_row_energy",
        "fedavg_retention_ratio",
        "tailagg_retention_ratio",
        "memory_row_norm",
    ]
    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for cls in range(len(global_class_counts)):
            local_energy = float(diagnostics["local_energy_sum"][cls].item())
            writer.writerow({
                "epoch": int(epoch),
                "seed": int(args.seed),
                "partition": args.partition,
                "mode": diagnostics["mode"],
                "class_id": cls,
                "class_group": class_groups.get(cls, "head"),
                "global_count": float(global_class_counts[cls]),
                "observed_client_count": float(diagnostics["observed_client_count"][cls].item()),
                "local_energy_sum": local_energy,
                "local_energy_mean_observed": float(diagnostics["local_energy_mean_observed"][cls].item()),
                "fedavg_row_energy": float(diagnostics["fedavg_row_energy"][cls].item()),
                "tailagg_row_energy": float(diagnostics["tailagg_row_energy"][cls].item()),
                "fedavg_retention_ratio": float(diagnostics["fedavg_row_energy"][cls].item()) / (local_energy + eps),
                "tailagg_retention_ratio": float(diagnostics["tailagg_row_energy"][cls].item()) / (local_energy + eps),
                "memory_row_norm": float(diagnostics["memory_row_norm"][cls].item()),
            })
    print(f"Appended FedTEF TailAgg diagnostics to {path}")


def compute_tail_score(exposure_count, current_round, warmup_rounds, eps=1e-12):
    exposure = exposure_count.float()
    min_exposure = torch.min(exposure)
    max_exposure = torch.max(exposure)
    if current_round < warmup_rounds or torch.isclose(max_exposure, min_exposure):
        return torch.ones_like(exposure)
    return 1.0 - (exposure - min_exposure) / (max_exposure - min_exposure + eps)


def compute_fedtef_protected_mask(
    tail_score,
    strategy="exposure",
    top_ratio=0.2,
    current_round=0,
    seed=0,
    global_class_counts=None,
    tail_class_ratio=0.2,
    round0_tie_break="bottom20_or_random",
    dataset_name="",
):
    num_classes = tail_score.numel()
    strategy = str(strategy).lower()
    if strategy == "random":
        k = max(1, int(np.ceil(num_classes * top_ratio)))
        rng = np.random.default_rng(int(seed) + int(current_round) * 1009)
        ids = rng.choice(num_classes, size=k, replace=False)
        protected = torch.zeros(num_classes, dtype=torch.bool)
        protected[torch.as_tensor(ids, dtype=torch.long)] = True
        return protected
    if strategy == "oracle_tail" and global_class_counts is not None:
        return get_oracle_tail_mask_from_counts(global_class_counts, tail_class_ratio)
    if strategy in ("all", "none"):
        return torch.ones(num_classes, dtype=torch.bool)
    if torch.isclose(tail_score.float().max(), tail_score.float().min()):
        k = max(1, int(np.ceil(num_classes * top_ratio)))
        return compute_tie_break_mask(
            num_classes,
            k,
            mode=round0_tie_break,
            seed=seed,
            current_round=current_round,
            dataset_name=dataset_name,
        )
    return compute_protected_tail_mask(tail_score, mode="top_ratio", top_ratio=top_ratio)


def compute_tie_break_mask(num_classes, k, mode="bottom20_or_random", seed=0, current_round=0, dataset_name=""):
    mode = str(mode).lower()
    protected = torch.zeros(num_classes, dtype=torch.bool)
    if mode == "none":
        return protected
    dataset_name = str(dataset_name).lower()
    if mode == "bottom20_or_random" and "cifar100" in dataset_name and num_classes == 100:
        ids = list(range(max(0, num_classes - k), num_classes))
        print("FedTEF-v2 warm-start uses bottom20 classes for debugging/controlled experiment")
        print("Using bottom20 warm-start for controlled experiment; not privacy-final.")
    else:
        rng = np.random.default_rng(int(seed) + int(current_round) * 1009)
        ids = rng.choice(num_classes, size=k, replace=False).tolist()
        if ids == list(range(k)):
            ids = rng.choice(num_classes, size=k, replace=False).tolist()
        print(f"FedTEF tie-break selected classes: {ids}")
    protected[torch.as_tensor(ids, dtype=torch.long)] = True
    return protected


def compute_protected_tail_mask(tail_score, mode="top_ratio", top_ratio=0.2, threshold=0.5):
    num_classes = tail_score.numel()
    mode = str(mode).lower()
    if mode in ("all", "soft"):
        return torch.ones(num_classes, dtype=torch.bool)
    if mode == "top_ratio":
        k = max(1, int(np.ceil(num_classes * top_ratio)))
        protected = torch.zeros(num_classes, dtype=torch.bool)
        protected[torch.topk(tail_score, k).indices.cpu()] = True
        return protected
    if mode == "threshold":
        protected = tail_score.cpu() >= float(threshold)
        if not protected.any():
            protected[int(torch.argmax(tail_score).item())] = True
        return protected

    protected = tail_score.cpu() > 0
    if not protected.any():
        protected[int(torch.argmax(tail_score).item())] = True
    return protected


def apply_fedtef_tail_context(model, tail_score, protected_tail_mask, gate=None, release_reliability=None):
    if hasattr(model, "set_tail_context"):
        try:
            model.set_tail_context(
                tail_score,
                protected_tail_mask,
                gate=gate,
                release_reliability=release_reliability,
            )
        except TypeError:
            model.set_tail_context(tail_score, protected_tail_mask)


def compute_fedtef_release_reliability(
    tracker,
    gate,
    protected_tail_mask,
    args,
    aggregated_tail_energy=None,
):
    num_classes = int(gate.numel())
    if not bool(getattr(args, "fedtef_release_reliability_enabled", False)):
        return torch.ones(num_classes, dtype=torch.float32)

    source = str(getattr(args, "fedtef_release_reliability_source", "positive_proxy_ema")).lower()
    gate = torch.as_tensor(gate, dtype=torch.float32).cpu()
    protected_tail_mask = torch.as_tensor(protected_tail_mask, dtype=torch.bool).cpu()
    active = torch.logical_and(protected_tail_mask, gate > float(args.fedtef_exposure_eps))
    if source == "ones":
        reliability = torch.where(active, torch.ones(num_classes), torch.zeros(num_classes))
        return reliability.float()

    if source == "positive_proxy_ema" and hasattr(tracker, "positive_proxy_ema"):
        evidence = tracker.positive_proxy_ema.float().clone()
    elif source == "last_positive_proxy" and hasattr(tracker, "last_positive_proxy"):
        evidence = tracker.last_positive_proxy.float().clone()
    elif source == "aggregated_energy" and aggregated_tail_energy is not None:
        evidence = torch.as_tensor(aggregated_tail_energy, dtype=torch.float32).clone()
    elif source == "observed_count" and hasattr(tracker, "observed_count"):
        evidence = tracker.observed_count.float().clone()
    else:
        evidence = torch.ones(num_classes, dtype=torch.float32)

    positive_active = torch.logical_and(active, evidence > float(args.fedtef_exposure_eps))
    if positive_active.any():
        ref = evidence[positive_active].mean()
    else:
        positive = evidence[evidence > float(args.fedtef_exposure_eps)]
        ref = positive.mean() if positive.numel() > 0 else torch.tensor(1.0)

    tau = max(float(getattr(args, "fedtef_release_reliability_tau", 1.0)), float(args.fedtef_exposure_eps))
    floor = min(max(float(getattr(args, "fedtef_release_reliability_floor", 0.0)), 0.0), 1.0)
    power = max(float(getattr(args, "fedtef_release_reliability_power", 1.0)), float(args.fedtef_exposure_eps))
    norm = torch.clamp(evidence / (ref + float(args.fedtef_exposure_eps)), min=0.0)
    reliability = norm / (norm + tau)
    if power != 1.0:
        reliability = torch.pow(torch.clamp(reliability, min=0.0), power)
    reliability = floor + (1.0 - floor) * reliability
    reliability = torch.where(active, reliability, torch.zeros_like(reliability))
    reliability = torch.clamp(reliability, min=0.0, max=1.0)

    active_values = reliability[active] if active.any() else reliability
    print(
        "FedTEF-v8 release reliability "
        f"source={source}; min/max/mean: "
        f"{active_values.min().item():.4f}/"
        f"{active_values.max().item():.4f}/"
        f"{active_values.mean().item():.4f}; "
        f"active={int(active.sum().item())}/{num_classes}; "
        f"floor/tau/power={floor:.3f}/{tau:.3f}/{power:.3f}"
    )
    return reliability.float()


def update_fedtef_exposure(exposure_count, client_class_counts, idxs_users):
    # FedTEF method signal: simulate secure aggregation of binary class support.
    # The server uses only the summed support M_c, not per-client counts.
    support_sum = torch.zeros_like(exposure_count)
    for client_idx in idxs_users:
        present_classes = client_class_counts[int(client_idx)] > 0
        support_sum += present_classes.to(dtype=exposure_count.dtype)
    exposure_count += support_sum
    return exposure_count


def classwise_tail_expert_aggregation(
    global_weights,
    local_weights,
    idxs_users,
    client_class_counts,
    aggregation="binary_support",
    eps=1e-12,
):
    tail_weight_key = "tail_expert.weight"
    tail_bias_key = "tail_expert.bias"
    tail_scale_key = "tail_expert.logit_scale"
    if tail_weight_key not in global_weights:
        return global_weights

    aggregation = str(aggregation).lower()
    if aggregation == "fedavg":
        for key in [tail_weight_key, tail_bias_key, tail_scale_key]:
            if key not in global_weights:
                continue
            temp = torch.zeros_like(global_weights[key])
            total_weight = sum([client_class_counts[int(idx)].sum().item() for idx in idxs_users])
            if total_weight <= 0:
                total_weight = len(idxs_users)
                for client_idx in idxs_users:
                    temp += local_weights[int(client_idx)][key] / total_weight
            else:
                for client_idx in idxs_users:
                    client_total = client_class_counts[int(client_idx)].sum().item()
                    temp += float(client_total / total_weight) * local_weights[int(client_idx)][key]
            global_weights[key] = temp
        return global_weights

    num_classes = global_weights[tail_weight_key].shape[0]
    for class_idx in range(num_classes):
        valid_clients = [
            int(client_idx)
            for client_idx in idxs_users
            if client_class_counts[int(client_idx)][class_idx] > 0
        ]
        if not valid_clients:
            continue

        agg_weight = torch.zeros_like(global_weights[tail_weight_key][class_idx])
        agg_bias = (
            torch.zeros_like(global_weights[tail_bias_key][class_idx])
            if tail_bias_key in global_weights
            else None
        )
        total = 0.0
        for client_idx in valid_clients:
            if aggregation == "count_oracle":
                agg_coef = max(float(client_class_counts[client_idx][class_idx].item()), 0.0)
            else:
                # Default FedTEF aggregation uses binary class support only.
                agg_coef = 1.0
            total += agg_coef
            agg_weight += float(agg_coef) * local_weights[client_idx][tail_weight_key][class_idx]
            if agg_bias is not None:
                agg_bias += float(agg_coef) * local_weights[client_idx][tail_bias_key][class_idx]

        global_weights[tail_weight_key][class_idx] = agg_weight / (total + eps)
        if agg_bias is not None:
            global_weights[tail_bias_key][class_idx] = agg_bias / (total + eps)

    if tail_scale_key in global_weights:
        temp = torch.zeros_like(global_weights[tail_scale_key])
        total_weight = sum([client_class_counts[int(idx)].sum().item() for idx in idxs_users])
        if total_weight <= 0:
            total_weight = len(idxs_users)
            for client_idx in idxs_users:
                temp += local_weights[int(client_idx)][tail_scale_key] / total_weight
        else:
            for client_idx in idxs_users:
                client_total = client_class_counts[int(client_idx)].sum().item()
                temp += float(client_total / total_weight) * local_weights[int(client_idx)][tail_scale_key]
        global_weights[tail_scale_key] = temp

    return global_weights


class ExposureTracker:
    """Server-side low-exposure gate from class-wise tail update energy."""

    def __init__(
        self,
        num_classes,
        rho=0.9,
        eps=1e-6,
        gate_mode="soft",
        temperature=1.0,
        threshold=None,
        tail_topk=20,
        round0_tie_break="bottom20_or_random",
        seed=0,
        dataset_name="",
    ):
        self.num_classes = int(num_classes)
        self.rho = float(rho)
        self.eps = float(eps)
        self.gate_mode = str(gate_mode).lower()
        self.temperature = max(float(temperature), self.eps)
        self.threshold = threshold
        self.tail_topk = max(1, min(int(tail_topk), self.num_classes))
        self.round0_tie_break = round0_tie_break
        self.seed = int(seed)
        self.dataset_name = dataset_name
        self.exposure = torch.zeros(self.num_classes, dtype=torch.float32)
        self.last_energy = torch.zeros(self.num_classes, dtype=torch.float32)
        self.round = 0

    def update_from_energy(self, energy):
        energy = torch.as_tensor(energy, dtype=torch.float32).cpu()
        self.last_energy = energy.clone()
        mean_energy = energy.mean()
        if mean_energy <= self.eps:
            energy_norm = torch.zeros_like(energy)
        else:
            energy_norm = energy / (mean_energy + self.eps)
        self.exposure = self.rho * self.exposure + (1.0 - self.rho) * energy_norm
        self.round += 1
        return self.exposure

    def compute_scores(self):
        return 1.0 / (self.exposure + self.eps)

    def compute_gate(self, current_round=0):
        scores = self.compute_scores()
        tied = torch.isclose(scores.max(), scores.min())
        if tied:
            mask = compute_tie_break_mask(
                self.num_classes,
                self.tail_topk,
                mode=self.round0_tie_break,
                seed=self.seed,
                current_round=current_round,
                dataset_name=self.dataset_name,
            )
            return mask.float(), scores, mask

        if self.gate_mode == "hard_topk":
            mask = torch.zeros(self.num_classes, dtype=torch.bool)
            mask[torch.topk(scores, self.tail_topk).indices.cpu()] = True
            return mask.float(), scores, mask

        sorted_scores = torch.sort(scores, descending=True).values
        threshold = self.threshold
        if threshold is None:
            threshold = sorted_scores[self.tail_topk - 1].item()
        score_std = scores.std(unbiased=False)
        normalized = (scores - scores.mean()) / (score_std + self.eps)
        threshold_norm = (float(threshold) - scores.mean()) / (score_std + self.eps)
        gate = torch.sigmoid((normalized - threshold_norm) / self.temperature)
        mask = torch.zeros(self.num_classes, dtype=torch.bool)
        mask[torch.topk(scores, self.tail_topk).indices.cpu()] = True
        return gate.float(), scores, mask


def is_tail_stream_key(key):
    return key.startswith("tail_stream.") or ".tail_stream." in key


def is_shared_stream_key(key, train_img_adap=False, train_lora=False):
    if key.startswith("prompt_learner.") or ".prompt_learner." in key:
        return True
    if train_img_adap and (key.startswith("img_adap.") or ".img_adap." in key):
        return True
    if train_lora and "lora_" in key:
        return True
    return False


def fedavg_keys(global_weights, local_weights, idxs_users, datanumber_client, keys):
    total_weight = sum([datanumber_client[int(idx)] for idx in idxs_users])
    for key in keys:
        temp = torch.zeros_like(global_weights[key])
        for client_idx in idxs_users:
            temp += (datanumber_client[int(client_idx)] / total_weight) * local_weights[int(client_idx)][key]
        global_weights[key] = temp
    return global_weights


def fedtef_v2_tailagg(
    global_weights,
    local_weights,
    idxs_users,
    datanumber_client,
    gate,
    num_classes,
    fallback="fedavg_or_keep",
    eps=1e-6,
):
    old_global = copy.deepcopy(global_weights)
    tail_keys = [key for key in global_weights.keys() if is_tail_stream_key(key)]
    classwise_keys = [
        key for key in tail_keys
        if global_weights[key].ndim >= 1 and global_weights[key].shape[0] == num_classes
    ]
    non_classwise_keys = [key for key in tail_keys if key not in classwise_keys]
    gate = torch.as_tensor(gate, dtype=torch.float32).cpu()
    client_energy = {
        int(client_idx): torch.zeros(num_classes, dtype=torch.float32)
        for client_idx in idxs_users
    }
    for client_idx in idxs_users:
        client_idx = int(client_idx)
        for key in classwise_keys:
            delta = (local_weights[client_idx][key].detach().cpu() - old_global[key].detach().cpu()).float()
            flat = delta.reshape(num_classes, -1)
            client_energy[client_idx] += torch.linalg.vector_norm(flat, dim=1)

    aggregated_energy = torch.zeros(num_classes, dtype=torch.float32)
    fallback_count = 0
    for key in classwise_keys:
        old_value = old_global[key]
        new_value = old_value.clone()
        for class_idx in range(num_classes):
            norms = torch.tensor(
                [client_energy[int(client_idx)][class_idx].item() for client_idx in idxs_users],
                dtype=torch.float32,
            )
            if gate[class_idx].item() <= eps:
                fallback_count += 1
                continue
            if norms.sum().item() <= eps:
                fallback_count += 1
                if str(fallback).lower() == "fedavg_or_keep":
                    temp = torch.zeros_like(old_value[class_idx])
                    total_weight = sum([datanumber_client[int(idx)] for idx in idxs_users])
                    for client_idx in idxs_users:
                        temp += (datanumber_client[int(client_idx)] / total_weight) * local_weights[int(client_idx)][key][class_idx]
                    new_value[class_idx] = temp
                continue
            weights = gate[class_idx].item() * norms + eps
            weights = weights / weights.sum()
            row_delta = torch.zeros_like(old_value[class_idx])
            for pos, client_idx in enumerate(idxs_users):
                delta = local_weights[int(client_idx)][key][class_idx] - old_value[class_idx]
                row_delta += weights[pos].to(delta.device, dtype=delta.dtype) * delta
            new_value[class_idx] = old_value[class_idx] + row_delta
            aggregated_energy[class_idx] += row_delta.detach().float().norm().cpu()
        global_weights[key] = new_value

    if non_classwise_keys:
        print(f"FedTEF-v2 TailAgg: FedAvg for non class-wise tail params: {non_classwise_keys}")
        global_weights = fedavg_keys(global_weights, local_weights, idxs_users, datanumber_client, non_classwise_keys)

    fedavg_norm = 0.0
    tailagg_norm = 0.0
    fedavg_reference = copy.deepcopy(old_global)
    fedavg_reference = fedavg_keys(fedavg_reference, local_weights, idxs_users, datanumber_client, tail_keys)
    for key in tail_keys:
        fedavg_norm += (fedavg_reference[key].detach().float().cpu() - old_global[key].detach().float().cpu()).norm().item()
        tailagg_norm += (global_weights[key].detach().float().cpu() - old_global[key].detach().float().cpu()).norm().item()

    print_fedtef_v2_tailagg_diagnostics(aggregated_energy, fallback_count, tailagg_norm, fedavg_norm)
    return global_weights, aggregated_energy


def print_fedtef_v2_tailagg_diagnostics(energy, fallback_count, tailagg_norm, fedavg_norm):
    energy = energy.float()
    topk = min(10, energy.numel())
    top_ids = torch.topk(energy, topk).indices.tolist()
    near_zero = int((energy <= 1e-8).sum().item())
    print(
        "FedTEF-v2 TailAgg row update norm "
        f"min/max/mean: {energy.min().item():.6f}/"
        f"{energy.max().item():.6f}/"
        f"{energy.mean().item():.6f}"
    )
    print(f"FedTEF-v2 TailAgg top updated classes: {top_ids}")
    print(f"FedTEF-v2 TailAgg near-zero update classes: {near_zero}")
    print(f"FedTEF-v2 TailAgg fallback rows: {fallback_count}")
    print(f"FedTEF-v2 TailAgg vs FedAvg tail update norm: {tailagg_norm:.6f}/{fedavg_norm:.6f}")


def append_fedtef_v2_gate_history(
    output_dir,
    epoch,
    tracker,
    gate,
    scores,
    protected_mask,
    local_trainer,
    tail_class_ratio=0.2,
):
    if not output_dir:
        return
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "fedtef_v2_gate_history.csv")

    gate = gate.detach().float().cpu()
    scores = scores.detach().float().cpu()
    protected_mask = protected_mask.detach().bool().cpu()
    protected_ids = torch.nonzero(protected_mask, as_tuple=False).view(-1).tolist()
    top_ids = torch.topk(gate, min(20, gate.numel())).indices.tolist()

    splits = get_lt_class_splits_from_counts(local_trainer.cls_num_list, tail_class_ratio)
    tail_set = set(int(x) for x in splits["tail"])
    head_set = set(int(x) for x in splits["head"])
    protected_set = set(int(x) for x in protected_ids)
    tail_overlap = sorted(int(x) for x in protected_set.intersection(tail_set))
    head_overlap = sorted(int(x) for x in protected_set.intersection(head_set))

    exposure = tracker.exposure.detach().float().cpu()
    opportunity = getattr(tracker, "opportunity_count", torch.zeros_like(exposure))
    opportunity = torch.as_tensor(opportunity, dtype=torch.float32).cpu()
    scarcity = torch.as_tensor(
        getattr(tracker, "scarcity_score", torch.zeros_like(exposure)),
        dtype=torch.float32,
    ).cpu()
    residual_need = torch.as_tensor(
        getattr(tracker, "residual_need_score", torch.zeros_like(exposure)),
        dtype=torch.float32,
    ).cpu()
    tail_need = torch.as_tensor(
        getattr(tracker, "tail_need_ema", exposure),
        dtype=torch.float32,
    ).cpu()
    class_prior = torch.as_tensor(
        getattr(tracker, "class_prior_ema", exposure),
        dtype=torch.float32,
    ).cpu()
    class_prior_proxy = torch.as_tensor(
        getattr(tracker, "class_prior_proxy", torch.zeros_like(exposure)),
        dtype=torch.float32,
    ).cpu()
    positive_proxy = torch.as_tensor(
        getattr(tracker, "last_positive_proxy", torch.zeros_like(exposure)),
        dtype=torch.float32,
    ).cpu()
    observed_count = torch.as_tensor(
        getattr(tracker, "observed_count", torch.zeros_like(exposure)),
        dtype=torch.float32,
    ).cpu()
    observation_ema = torch.as_tensor(
        getattr(tracker, "observation_ema", torch.zeros_like(exposure)),
        dtype=torch.float32,
    ).cpu()
    reliability = torch.as_tensor(
        getattr(tracker, "reliability", torch.zeros_like(exposure)),
        dtype=torch.float32,
    ).cpu()
    exposure_mass = torch.as_tensor(
        getattr(tracker, "exposure_mass", torch.zeros_like(exposure)),
        dtype=torch.float32,
    ).cpu()
    difficulty_signal = torch.as_tensor(
        getattr(tracker, "difficulty", torch.zeros_like(exposure)),
        dtype=torch.float32,
    ).cpu()
    survival_signal = torch.as_tensor(
        getattr(tracker, "survival", torch.ones_like(exposure)),
        dtype=torch.float32,
    ).cpu()
    age_signal = torch.as_tensor(
        getattr(tracker, "age", torch.zeros_like(exposure)),
        dtype=torch.float32,
    ).cpu()
    lock_active = int(bool(getattr(tracker, "last_lock_active", False)))
    locked_until_round = int(getattr(tracker, "locked_until_round", -1))
    lock_source_round = int(getattr(tracker, "lock_source_round", -1))
    lock_mode = getattr(tracker, "lock_mode", "")
    refine_swaps = int(getattr(tracker, "last_refine_swaps", 0))
    refine_added_ids = getattr(tracker, "last_refine_added_ids", [])
    refine_removed_ids = getattr(tracker, "last_refine_removed_ids", [])
    lifetime = torch.as_tensor(
        getattr(tracker, "protected_lifetime", torch.zeros_like(exposure)),
        dtype=torch.float32,
    ).cpu()

    def _mean_for(class_ids, values):
        if not class_ids:
            return 0.0
        idx = torch.tensor(class_ids, dtype=torch.long)
        return values[idx].float().mean().item()

    tail_ids = splits["tail"]
    head_ids = splits["head"]
    protected_lifetime = lifetime[protected_mask]
    tail_protected = torch.zeros_like(protected_mask)
    head_protected = torch.zeros_like(protected_mask)
    if tail_ids:
        tail_protected[torch.tensor(tail_ids, dtype=torch.long)] = True
    if head_ids:
        head_protected[torch.tensor(head_ids, dtype=torch.long)] = True
    tail_protected_lifetime = lifetime[torch.logical_and(protected_mask, tail_protected)]
    head_protected_lifetime = lifetime[torch.logical_and(protected_mask, head_protected)]

    write_header = not os.path.exists(path)
    fieldnames = [
        "epoch",
        "is_warmup",
        "score_mode",
        "warmup_mode",
        "gate_mode",
        "tail_topk",
        "protected_count",
        "tail_overlap_count",
        "head_overlap_count",
        "tail_overlap_ids",
        "protected_ids",
        "top_gate_ids",
        "gate_min",
        "gate_max",
        "gate_mean",
        "gate_std",
        "score_min",
        "score_max",
        "score_mean",
        "score_std",
        "exposure_min",
        "exposure_max",
        "exposure_mean",
        "opportunity_min",
        "opportunity_max",
        "opportunity_mean",
        "protected_set_jaccard_with_previous_round",
        "churn_rate",
        "average_protected_lifetime",
        "tail_protected_lifetime",
        "head_protected_lifetime",
        "head_scarcity_mean",
        "tail_scarcity_mean",
        "head_residual_need_mean",
        "tail_residual_need_mean",
        "head_tail_need_mean",
        "tail_tail_need_mean",
        "head_class_prior_mean",
        "tail_class_prior_mean",
        "head_class_prior_proxy_mean",
        "tail_class_prior_proxy_mean",
        "head_positive_proxy_mean",
        "tail_positive_proxy_mean",
        "head_observed_count_mean",
        "tail_observed_count_mean",
        "head_observation_ema_mean",
        "tail_observation_ema_mean",
        "head_reliability_mean",
        "tail_reliability_mean",
        "observer_exposure_mass_head_mean",
        "observer_exposure_mass_tail_mean",
        "observer_difficulty_head_mean",
        "observer_difficulty_tail_mean",
        "observer_survival_head_mean",
        "observer_survival_tail_mean",
        "observer_age_head_mean",
        "observer_age_tail_mean",
        "topology_exposure_budget_ids",
        "topology_survival_budget_ids",
        "gradient_prior_lock_active",
        "gradient_prior_lock_mode",
        "gradient_prior_lock_source_round",
        "gradient_prior_locked_until_round",
        "gradient_prior_refine_swaps",
        "gradient_prior_refine_added_ids",
        "gradient_prior_refine_removed_ids",
    ]
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        writer.writerow({
            "epoch": int(epoch),
            "is_warmup": int(int(epoch) < int(getattr(tracker, "warmup_rounds", 0))),
            "score_mode": getattr(tracker, "score_mode", "exposure"),
            "warmup_mode": getattr(tracker, "warmup_mode", ""),
            "gate_mode": getattr(tracker, "gate_mode", ""),
            "tail_topk": int(getattr(tracker, "tail_topk", 0)),
            "protected_count": len(protected_ids),
            "tail_overlap_count": len(tail_overlap),
            "head_overlap_count": len(head_overlap),
            "tail_overlap_ids": ";".join(str(x) for x in tail_overlap),
            "protected_ids": ";".join(str(x) for x in protected_ids),
            "top_gate_ids": ";".join(str(int(x)) for x in top_ids),
            "gate_min": gate.min().item(),
            "gate_max": gate.max().item(),
            "gate_mean": gate.mean().item(),
            "gate_std": gate.std(unbiased=False).item(),
            "score_min": scores.min().item(),
            "score_max": scores.max().item(),
            "score_mean": scores.mean().item(),
            "score_std": scores.std(unbiased=False).item(),
            "exposure_min": exposure.min().item(),
            "exposure_max": exposure.max().item(),
            "exposure_mean": exposure.mean().item(),
            "opportunity_min": opportunity.min().item(),
            "opportunity_max": opportunity.max().item(),
            "opportunity_mean": opportunity.mean().item(),
            "protected_set_jaccard_with_previous_round": float(getattr(tracker, "last_jaccard", 1.0)),
            "churn_rate": float(getattr(tracker, "last_churn_rate", 0.0)),
            "average_protected_lifetime": protected_lifetime.mean().item() if protected_lifetime.numel() else 0.0,
            "tail_protected_lifetime": tail_protected_lifetime.mean().item() if tail_protected_lifetime.numel() else 0.0,
            "head_protected_lifetime": head_protected_lifetime.mean().item() if head_protected_lifetime.numel() else 0.0,
            "head_scarcity_mean": _mean_for(head_ids, scarcity),
            "tail_scarcity_mean": _mean_for(tail_ids, scarcity),
            "head_residual_need_mean": _mean_for(head_ids, residual_need),
            "tail_residual_need_mean": _mean_for(tail_ids, residual_need),
            "head_tail_need_mean": _mean_for(head_ids, tail_need),
            "tail_tail_need_mean": _mean_for(tail_ids, tail_need),
            "head_class_prior_mean": _mean_for(head_ids, class_prior),
            "tail_class_prior_mean": _mean_for(tail_ids, class_prior),
            "head_class_prior_proxy_mean": _mean_for(head_ids, class_prior_proxy),
            "tail_class_prior_proxy_mean": _mean_for(tail_ids, class_prior_proxy),
            "head_positive_proxy_mean": _mean_for(head_ids, positive_proxy),
            "tail_positive_proxy_mean": _mean_for(tail_ids, positive_proxy),
            "head_observed_count_mean": _mean_for(head_ids, observed_count),
            "tail_observed_count_mean": _mean_for(tail_ids, observed_count),
            "head_observation_ema_mean": _mean_for(head_ids, observation_ema),
            "tail_observation_ema_mean": _mean_for(tail_ids, observation_ema),
            "head_reliability_mean": _mean_for(head_ids, reliability),
            "tail_reliability_mean": _mean_for(tail_ids, reliability),
            "observer_exposure_mass_head_mean": _mean_for(head_ids, exposure_mass),
            "observer_exposure_mass_tail_mean": _mean_for(tail_ids, exposure_mass),
            "observer_difficulty_head_mean": _mean_for(head_ids, difficulty_signal),
            "observer_difficulty_tail_mean": _mean_for(tail_ids, difficulty_signal),
            "observer_survival_head_mean": _mean_for(head_ids, survival_signal),
            "observer_survival_tail_mean": _mean_for(tail_ids, survival_signal),
            "observer_age_head_mean": _mean_for(head_ids, age_signal),
            "observer_age_tail_mean": _mean_for(tail_ids, age_signal),
            "topology_exposure_budget_ids": ";".join(
                str(int(x)) for x in getattr(tracker, "last_exposure_ids", [])
            ),
            "topology_survival_budget_ids": ";".join(
                str(int(x)) for x in getattr(tracker, "last_survival_ids", [])
            ),
            "gradient_prior_lock_active": lock_active,
            "gradient_prior_lock_mode": lock_mode,
            "gradient_prior_lock_source_round": lock_source_round,
            "gradient_prior_locked_until_round": locked_until_round,
            "gradient_prior_refine_swaps": refine_swaps,
            "gradient_prior_refine_added_ids": ";".join(str(int(x)) for x in refine_added_ids),
            "gradient_prior_refine_removed_ids": ";".join(str(int(x)) for x in refine_removed_ids),
        })


def log_fedtef_v2_gate(epoch, tracker, gate, scores, protected_mask, local_trainer, tail_class_ratio=0.2, output_dir=None):
    exposure = tracker.exposure.float()
    gate = gate.float()
    scores = scores.float()
    top_ids = torch.topk(gate, min(20, gate.numel())).indices.tolist()
    print(f"FedTEF-v2 gate round: {epoch}")
    print(
        "FedTEF-v2 exposure min/max/mean/std: "
        f"{exposure.min().item():.6f}/{exposure.max().item():.6f}/"
        f"{exposure.mean().item():.6f}/{exposure.std(unbiased=False).item():.6f}"
    )
    if hasattr(tracker, "opportunity_count"):
        opportunity = tracker.opportunity_count.float()
        print(
            "FedTEF-v2 opportunity min/max/mean/std: "
            f"{opportunity.min().item():.2f}/{opportunity.max().item():.2f}/"
            f"{opportunity.mean().item():.2f}/{opportunity.std(unbiased=False).item():.2f}"
        )
    if getattr(tracker, "score_mode", "exposure") == "tail_need":
        print(
            "FedTEF-v2 tail_need persistence "
            f"jaccard/churn/avg_life: {tracker.last_jaccard:.4f}/"
            f"{tracker.last_churn_rate:.4f}/"
            f"{tracker.last_avg_protected_lifetime:.2f}"
        )
        scarcity = tracker.scarcity_score.float()
        residual_need = tracker.residual_need_score.float()
        tail_need = tracker.tail_need_ema.float()
        print(
            "FedTEF-v2 tail_need components mean "
            f"scarcity/residual/need: {scarcity.mean().item():.4f}/"
            f"{residual_need.mean().item():.4f}/"
            f"{tail_need.mean().item():.4f}"
        )
    if getattr(tracker, "score_mode", "exposure") in ("gradient_prior", "low_exposure_router"):
        class_prior = tracker.class_prior_ema.float()
        class_prior_proxy = tracker.class_prior_proxy.float()
        positive_proxy = tracker.last_positive_proxy.float()
        observed_count = tracker.observed_count.float()
        print(
            "FedTEF-v2 gradient_prior mean "
            f"prior/proxy/positive_update/observed: "
            f"{class_prior.mean().item():.4f}/"
            f"{class_prior_proxy.mean().item():.4f}/"
            f"{positive_proxy.mean().item():.6f}/"
            f"{observed_count.mean().item():.2f}"
        )
        if int(getattr(tracker, "lock_rounds", 0)) > 0:
            print(
                "FedTEF-v2 gradient_prior lock "
                f"active/mode/source/until/length/swaps: "
                f"{int(bool(getattr(tracker, 'last_lock_active', False)))}/"
                f"{getattr(tracker, 'lock_mode', '')}/"
                f"{int(getattr(tracker, 'lock_source_round', -1))}/"
                f"{int(getattr(tracker, 'locked_until_round', -1))}/"
                f"{int(getattr(tracker, 'lock_rounds', 0))}/"
                f"{int(getattr(tracker, 'last_refine_swaps', 0))}"
            )
    if getattr(tracker, "score_mode", "exposure") == "evidence_memory":
        observation = tracker.observation_ema.float()
        reliability = tracker.reliability.float()
        scarcity = tracker.scarcity_score.float()
        print(
            "FedTEF-v4 evidence memory mean "
            f"observation/reliability/scarcity: "
            f"{observation.mean().item():.4f}/"
            f"{reliability.mean().item():.4f}/"
            f"{scarcity.mean().item():.4f}"
        )
    if getattr(tracker, "score_mode", "exposure") == "topology_observer":
        print(
            "FedTEF-v10 observer mean E/D/S/A: "
            f"{tracker.exposure_mass.float().mean().item():.6f}/"
            f"{tracker.difficulty.float().mean().item():.6f}/"
            f"{tracker.survival.float().mean().item():.4f}/"
            f"{tracker.age.float().mean().item():.2f}"
        )
        print(
            "FedTEF-v10 budget ids exposure/survival: "
            f"{getattr(tracker, 'last_exposure_ids', [])}/"
            f"{getattr(tracker, 'last_survival_ids', [])}"
        )
    print(
        "FedTEF-v2 score min/max/mean/std: "
        f"{scores.min().item():.6f}/{scores.max().item():.6f}/"
        f"{scores.mean().item():.6f}/{scores.std(unbiased=False).item():.6f}"
    )
    print(
        "FedTEF-v2 gate min/max/mean/std: "
        f"{gate.min().item():.6f}/{gate.max().item():.6f}/"
        f"{gate.mean().item():.6f}/{gate.std(unbiased=False).item():.6f}"
    )
    print(f"FedTEF-v2 top gate classes ids: {top_ids}")

    splits = get_lt_class_splits_from_counts(local_trainer.cls_num_list, tail_class_ratio)
    tail_set = set(int(x) for x in splits["tail"])
    overlap = sorted([int(x) for x in top_ids if int(x) in tail_set])
    print(f"FedTEF-v2 overlap with bottom tail classes: {len(overlap)}/{len(tail_set)} {overlap}")
    for group_name, class_ids in {"head": splits["head"], "tail": splits["tail"]}.items():
        if not class_ids:
            continue
        idx = torch.tensor(class_ids, dtype=torch.long)
        gated_count = int(protected_mask[idx].sum().item())
        print(f"FedTEF-v2 protected/gated count among {group_name}: {gated_count}/{len(class_ids)}")

    append_fedtef_v2_gate_history(
        output_dir,
        epoch,
        tracker,
        gate,
        scores,
        protected_mask,
        local_trainer,
        tail_class_ratio,
    )


from trainers.fedtef_v2_utils import (  # noqa: E402
    EvidenceMemoryTracker,
    ExposureTracker,
    GradientPriorTracker,
    LowExposureRouterTracker,
    TailNeedTracker,
    compute_tie_break_mask,
    fedtef_v2_tailagg,
)
from trainers.fedtef_aggregation import (  # noqa: E402
    compute_tail_stream_gradient_prior_proxy,
    compute_tail_stream_positive_update_stats,
    fedavg_keys,
    fedtef_v10_evidence_preserving_tailagg,
    is_shared_stream_key,
    is_tail_stream_key,
)
from trainers.fedtef_observer import TopologyExposureSurvivalObserver  # noqa: E402


def log_fedtef_exposure(epoch, exposure_count, tail_score, protected_tail_mask, local_trainer, tail_class_ratio=0.2):
    protected_ids = torch.nonzero(protected_tail_mask, as_tuple=False).view(-1).tolist()
    print(f"FedTEF exposure round: {epoch}")
    print(
        "FedTEF exposure_count "
        f"min/max/mean: {exposure_count.min().item():.2f}/"
        f"{exposure_count.max().item():.2f}/"
        f"{exposure_count.float().mean().item():.2f}"
    )
    print(
        "FedTEF tail_score "
        f"min/max/mean: {tail_score.min().item():.4f}/"
        f"{tail_score.max().item():.4f}/"
        f"{tail_score.float().mean().item():.4f}"
    )
    print(f"FedTEF protected_tail_classes count: {len(protected_ids)}")
    print(f"FedTEF protected_tail_classes ids: {protected_ids}")

    splits = get_lt_class_splits_from_counts(local_trainer.cls_num_list, tail_class_ratio)
    groups = {
        "non-tail": splits["head"],
        "bottom20 tail": splits["tail"],
    }

    for group_name, class_ids in groups.items():
        if not class_ids:
            continue
        idx = torch.tensor(class_ids, dtype=torch.long)
        group_score = tail_score[idx].float().mean().item()
        protected_count = protected_tail_mask[idx].sum().item()
        print(
            f"FedTEF {group_name} tail_score_mean: {group_score:.4f}, "
            f"protected_count among {group_name} classes: {int(protected_count)}/{len(class_ids)}"
        )


def save_fedtef_server_state(
    output_dir,
    epoch,
    exposure_count,
    tail_score,
    protected_tail_mask,
    args,
    model_state_dict=None,
    tail_gate=None,
):
    def atomic_torch_save(payload, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        tmp_path = f"{path}.tmp"
        try:
            torch.save(payload, tmp_path)
            os.replace(tmp_path, path)
        except Exception:
            if os.path.exists(tmp_path):
                try:
                    os.remove(tmp_path)
                except OSError:
                    pass
            raise

    state = {
        "epoch": epoch,
        "state_dict": model_state_dict,
        "exposure_count": exposure_count.cpu(),
        "tail_score": tail_score.cpu(),
        "protected_tail_mask": protected_tail_mask.cpu(),
        "tail_gate": tail_gate.cpu() if tail_gate is not None else None,
        "fedtef_config": {
            "method": args.method,
            "use_tail_expert": args.use_tail_expert,
            "fusion_lambda": args.fusion_lambda,
            "tail_fusion_mode": args.tail_fusion_mode,
            "tail_update_protect_mode": args.tail_update_protect_mode,
            "tail_class_ratio": args.tail_class_ratio,
            "protected_tail_ratio": args.protected_tail_ratio,
            "positive_gate": args.positive_gate,
            "classwise_tail_agg": args.classwise_tail_agg,
            "tail_expert_lr_mult": args.tail_expert_lr_mult,
            "freeze_img_adap": args.freeze_img_adap,
            "tail_expert_mode": args.tail_expert_mode,
            "tail_init_logit_scale": args.tail_init_logit_scale,
            "tail_learnable_scale": args.tail_learnable_scale,
            "tail_use_bias": args.tail_use_bias,
            "tail_logit_scale_max": args.tail_logit_scale_max,
            "fedtef_version": args.fedtef_version,
            "fedtef_gate_score_mode": args.fedtef_gate_score_mode,
            "fedtef_gradient_prior_floor": args.fedtef_gradient_prior_floor,
            "fedtef_gradient_prior_score_power": args.fedtef_gradient_prior_score_power,
            "fedtef_gradient_prior_lock_rounds": args.fedtef_gradient_prior_lock_rounds,
            "fedtef_gradient_prior_lock_mode": args.fedtef_gradient_prior_lock_mode,
            "fedtef_gradient_prior_refine_ratio": args.fedtef_gradient_prior_refine_ratio,
            "fedtef_gradient_prior_refine_max_swap": args.fedtef_gradient_prior_refine_max_swap,
            "fedtef_gradient_prior_refine_margin": args.fedtef_gradient_prior_refine_margin,
            "fedtef_gradient_prior_lock_gate_floor": args.fedtef_gradient_prior_lock_gate_floor,
            "fedtef_gradient_prior_update_all_rows": args.fedtef_gradient_prior_update_all_rows,
            "fedtef_loss_protected_base_weight": args.fedtef_loss_protected_base_weight,
            "fedtef_loss_protected_base_margin_weight": args.fedtef_loss_protected_base_margin_weight,
            "fedtef_protected_base_margin": args.fedtef_protected_base_margin,
            "fedtef_acquisition_low_exposure_weight": args.fedtef_acquisition_low_exposure_weight,
            "fedtef_acquisition_signal_source": args.fedtef_acquisition_signal_source,
            "fedtef_acquisition_signal_clamp_max": args.fedtef_acquisition_signal_clamp_max,
            "fedtef_acquisition_weight_normalize": args.fedtef_acquisition_weight_normalize,
            "fedtef_tail_stream_detach_base": args.fedtef_tail_stream_detach_base,
            "fedtef_tail_stream_mode": args.fedtef_tail_stream_mode,
            "fedtef_tailagg_mode": args.fedtef_tailagg_mode,
            "fedtef_semantic_rescue_enabled": args.fedtef_semantic_rescue_enabled,
            "fedtef_positive_residual_only": args.fedtef_positive_residual_only,
            "fedtef_residual_clamp": args.fedtef_residual_clamp,
            "fedtef_release_reliability_enabled": args.fedtef_release_reliability_enabled,
            "fedtef_release_reliability_source": args.fedtef_release_reliability_source,
            "fedtef_release_reliability_floor": args.fedtef_release_reliability_floor,
            "fedtef_release_reliability_tau": args.fedtef_release_reliability_tau,
            "fedtef_release_reliability_power": args.fedtef_release_reliability_power,
            "fedtef_train_routed_prompt": args.fedtef_train_routed_prompt,
            "fedtef_routed_prompt_lr_mult": args.fedtef_routed_prompt_lr_mult,
            "fedtef_routed_prompt_scale": args.fedtef_routed_prompt_scale,
            "fedtef_routed_prompt_update_all_rows": args.fedtef_routed_prompt_update_all_rows,
            "fedtef_evidence_memory_update_all_rows": args.fedtef_evidence_memory_update_all_rows,
            "fedtef_evidence_memory_gate_floor": args.fedtef_evidence_memory_gate_floor,
            "fedtef_evidence_memory_momentum": args.fedtef_evidence_memory_momentum,
            "fedtef_train_lora": args.fedtef_train_lora,
            "fedtef_lora_lr_mult": args.fedtef_lora_lr_mult,
            "fedtef_lora_encoder": args.fedtef_lora_encoder,
            "fedtef_lora_position": args.fedtef_lora_position,
            "fedtef_lora_rank": args.fedtef_lora_rank,
            "fedtef_lora_alpha": args.fedtef_lora_alpha,
            "fedtef_lora_params": args.fedtef_lora_params,
            "fedtef_v10_exposure_budget": args.fedtef_v10_exposure_budget,
            "fedtef_v10_survival_budget": args.fedtef_v10_survival_budget,
            "fedtef_v10_min_hold": args.fedtef_v10_min_hold,
            "fedtef_v10_hardneg_topm": args.fedtef_v10_hardneg_topm,
            "fedtef_v10_release_floor": args.fedtef_v10_release_floor,
            "fedtef_v10_agg_base_momentum": args.fedtef_v10_agg_base_momentum,
            "fedtef_v10_agg_low_survival_momentum": args.fedtef_v10_agg_low_survival_momentum,
        },
    }
    latest_path = os.path.join(output_dir, "fedtef_state_latest.pth")
    atomic_torch_save(state, latest_path)
    if getattr(args, "fedtef_save_epoch_state", False):
        epoch_path = os.path.join(output_dir, f"fedtef_state_epoch_{epoch}.pth")
        atomic_torch_save(state, epoch_path)


def evaluate_fedtef_logit_paths(trainer):
    if not (hasattr(trainer.model, "tail_expert") or hasattr(trainer.model, "tail_stream")):
        return None

    trainer.set_model_mode("eval")
    correct = {"base": 0, "tail": 0, "fused": 0}
    total = 0
    per_class = {
        "base": {},
        "tail": {},
        "fused": {},
        "total": {},
        "base_margin_sum": {},
        "base_cosine_margin_sum": {},
        "fused_margin_sum": {},
        "changed": {},
        "right_flip": {},
        "wrong_flip": {},
        "residual_abs_sum": {},
        "fused_delta_abs_sum": {},
        "semantic_rescue_gate_sum": {},
        "release_reliability_sum": {},
        "class_release_gate_sum": {},
    }
    with torch.no_grad():
        for batch in trainer.test_loader:
            inputs, labels = trainer.parse_batch_test(batch)
            outputs = trainer.model(inputs, return_dict=True)
            base_logits = outputs.get("base_logits", outputs.get("logits_base"))
            tail_logits = outputs.get("tail_logits", outputs.get("residual_tail"))
            fused_logits = outputs.get("logits", outputs.get("logits_fused"))
            preds = {
                "base": base_logits.argmax(dim=1),
                "tail": tail_logits.argmax(dim=1),
                "fused": fused_logits.argmax(dim=1),
            }
            one_hot = torch.nn.functional.one_hot(labels, num_classes=base_logits.shape[1]).bool()
            base_true = base_logits.gather(1, labels.view(-1, 1)).squeeze(1)
            fused_true = fused_logits.gather(1, labels.view(-1, 1)).squeeze(1)
            base_wrong = base_logits.masked_fill(one_hot, torch.finfo(base_logits.dtype).min).max(dim=1).values
            fused_wrong = fused_logits.masked_fill(one_hot, torch.finfo(fused_logits.dtype).min).max(dim=1).values
            base_margin = (base_true - base_wrong).float()
            fused_margin = (fused_true - fused_wrong).float()
            model = trainer.model.module if hasattr(trainer.model, "module") else trainer.model
            logit_scale = getattr(model, "logit_scale", None)
            if logit_scale is None:
                base_cosine_margin = base_margin
            else:
                base_cosine_margin = base_margin / torch.clamp(logit_scale.detach().float().exp(), min=1e-12)
            changed = preds["base"] != preds["fused"]
            right_flip = torch.logical_and(preds["base"] != labels, preds["fused"] == labels)
            wrong_flip = torch.logical_and(preds["base"] == labels, preds["fused"] != labels)
            residual_abs = tail_logits.detach().float().abs().mean(dim=1)
            fused_delta_abs = (fused_logits.detach().float() - base_logits.detach().float()).abs().mean(dim=1)
            semantic_rescue_gate = outputs.get("semantic_rescue_gate", torch.ones_like(base_logits))
            semantic_rescue_gate = semantic_rescue_gate.detach().float().mean(dim=1)
            release_reliability_vec = outputs.get("release_reliability", None)
            if release_reliability_vec is None:
                release_reliability = torch.ones_like(labels, dtype=torch.float32)
            else:
                release_reliability = release_reliability_vec.detach().float()[labels]
            class_release_gate_vec = outputs.get("class_release_gate", None)
            if class_release_gate_vec is None:
                class_release_gate = torch.ones_like(labels, dtype=torch.float32)
            else:
                class_release_gate = class_release_gate_vec.detach().float()[labels]
            total += labels.numel()
            for name, pred in preds.items():
                correct[name] += (pred == labels).sum().item()
            for label_idx in range(labels.numel()):
                cls = int(labels[label_idx].item())
                per_class["total"][cls] = per_class["total"].get(cls, 0) + 1
                for name, pred in preds.items():
                    per_class[name][cls] = per_class[name].get(cls, 0) + int(pred[label_idx].item() == cls)
                per_class["base_margin_sum"][cls] = per_class["base_margin_sum"].get(cls, 0.0) + float(base_margin[label_idx].item())
                per_class["base_cosine_margin_sum"][cls] = per_class["base_cosine_margin_sum"].get(cls, 0.0) + float(base_cosine_margin[label_idx].item())
                per_class["fused_margin_sum"][cls] = per_class["fused_margin_sum"].get(cls, 0.0) + float(fused_margin[label_idx].item())
                per_class["changed"][cls] = per_class["changed"].get(cls, 0) + int(changed[label_idx].item())
                per_class["right_flip"][cls] = per_class["right_flip"].get(cls, 0) + int(right_flip[label_idx].item())
                per_class["wrong_flip"][cls] = per_class["wrong_flip"].get(cls, 0) + int(wrong_flip[label_idx].item())
                per_class["residual_abs_sum"][cls] = per_class["residual_abs_sum"].get(cls, 0.0) + float(residual_abs[label_idx].item())
                per_class["fused_delta_abs_sum"][cls] = per_class["fused_delta_abs_sum"].get(cls, 0.0) + float(fused_delta_abs[label_idx].item())
                per_class["semantic_rescue_gate_sum"][cls] = per_class["semantic_rescue_gate_sum"].get(cls, 0.0) + float(semantic_rescue_gate[label_idx].item())
                per_class["release_reliability_sum"][cls] = per_class["release_reliability_sum"].get(cls, 0.0) + float(release_reliability[label_idx].item())
                per_class["class_release_gate_sum"][cls] = per_class["class_release_gate_sum"].get(cls, 0.0) + float(class_release_gate[label_idx].item())

    metrics = {}
    class_ids = list(per_class["total"].keys())
    for name in ["base", "tail", "fused"]:
        metrics[f"{name}_acc"] = 100.0 * correct[name] / max(total, 1)
        metrics[f"{name}_per_class_acc"] = {
            cls: 100.0 * per_class[name].get(cls, 0) / max(per_class["total"].get(cls, 0), 1)
            for cls in per_class["total"].keys()
        }
    for source, target in [
        ("base_margin_sum", "base_margin"),
        ("base_cosine_margin_sum", "base_cosine_margin"),
        ("fused_margin_sum", "fused_margin"),
        ("changed", "changed_rate"),
        ("right_flip", "right_flip_rate"),
        ("wrong_flip", "wrong_flip_rate"),
        ("residual_abs_sum", "residual_abs"),
        ("fused_delta_abs_sum", "fused_delta_abs"),
        ("semantic_rescue_gate_sum", "semantic_rescue_gate"),
        ("release_reliability_sum", "release_reliability"),
        ("class_release_gate_sum", "class_release_gate"),
    ]:
        metrics[f"{target}_per_class"] = {
            cls: float(per_class[source].get(cls, 0.0)) / max(per_class["total"].get(cls, 0), 1)
            for cls in class_ids
        }
    metrics["per_class_total"] = dict(per_class["total"])
    return metrics


def append_fedtef_branch_diagnostics(
    output_dir,
    epoch,
    path_metrics,
    class_groups,
):
    if path_metrics is None:
        return
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "fedtef_branch_diagnostics.csv")
    fieldnames = [
        "epoch",
        "scope",
        "num_classes",
        "num_samples",
        "base_acc",
        "tail_only_acc",
        "fused_acc",
        "fused_minus_base_acc",
        "base_margin",
        "base_cosine_margin",
        "fused_margin",
        "changed_rate",
        "right_flip_rate",
        "wrong_flip_rate",
        "residual_abs",
        "fused_delta_abs",
        "semantic_rescue_gate",
        "release_reliability",
        "class_release_gate",
    ]
    totals = path_metrics.get("per_class_total", {})
    scopes = {
        "all": sorted(totals),
        "head": sorted(cls for cls in totals if class_groups.get(cls, "head") == "head"),
        "tail": sorted(cls for cls in totals if class_groups.get(cls, "head") == "tail"),
    }

    def weighted(metric_name, class_ids):
        values = path_metrics.get(metric_name, {})
        denominator = sum(int(totals.get(cls, 0)) for cls in class_ids)
        if denominator <= 0:
            return 0.0
        return sum(float(values.get(cls, 0.0)) * int(totals.get(cls, 0)) for cls in class_ids) / denominator

    write_header = not os.path.exists(path)
    with open(path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()
        for scope, class_ids in scopes.items():
            base_acc = weighted("base_per_class_acc", class_ids)
            fused_acc = weighted("fused_per_class_acc", class_ids)
            writer.writerow({
                "epoch": int(epoch),
                "scope": scope,
                "num_classes": len(class_ids),
                "num_samples": sum(int(totals.get(cls, 0)) for cls in class_ids),
                "base_acc": base_acc,
                "tail_only_acc": weighted("tail_per_class_acc", class_ids),
                "fused_acc": fused_acc,
                "fused_minus_base_acc": fused_acc - base_acc,
                "base_margin": weighted("base_margin_per_class", class_ids),
                "base_cosine_margin": weighted("base_cosine_margin_per_class", class_ids),
                "fused_margin": weighted("fused_margin_per_class", class_ids),
                "changed_rate": weighted("changed_rate_per_class", class_ids),
                "right_flip_rate": weighted("right_flip_rate_per_class", class_ids),
                "wrong_flip_rate": weighted("wrong_flip_rate_per_class", class_ids),
                "residual_abs": weighted("residual_abs_per_class", class_ids),
                "fused_delta_abs": weighted("fused_delta_abs_per_class", class_ids),
                "semantic_rescue_gate": weighted("semantic_rescue_gate_per_class", class_ids),
                "release_reliability": weighted("release_reliability_per_class", class_ids),
                "class_release_gate": weighted("class_release_gate_per_class", class_ids),
            })
    print(f"Appended FedTEF branch diagnostics to {path}")


def save_fedtef_analysis(
    output_dir,
    epoch,
    exposure_count,
    tail_score,
    protected_tail_mask,
    client_class_counts,
    num_users,
    num_classes,
    per_class_acc,
    path_metrics=None,
    tail_class_ratio=0.2,
):
    os.makedirs(output_dir, exist_ok=True)
    counts = client_counts_to_tensor(client_class_counts, num_users, num_classes)
    support = counts > 0
    global_counts = counts.sum(dim=0)
    class_num_clients = support.sum(dim=0)
    class_groups = get_class_groups_from_counts(global_counts, tail_class_ratio)
    append_fedtef_branch_diagnostics(output_dir, epoch, path_metrics, class_groups)

    exposure_payload = {
        "epoch": epoch,
        "exposure_E": [float(x) for x in exposure_count.cpu().tolist()],
        "exposure_score_s": [float(x) for x in tail_score.cpu().tolist()],
        "protected_tail_mask": [bool(x) for x in protected_tail_mask.cpu().tolist()],
    }
    if path_metrics is not None:
        exposure_payload.update({
            "base_acc": path_metrics.get("base_acc"),
            "tail_only_acc": path_metrics.get("tail_acc"),
            "fused_acc": path_metrics.get("fused_acc"),
        })
    with open(os.path.join(output_dir, "fedtef_exposure.json"), "w", encoding="utf-8") as f:
        json.dump(exposure_payload, f, indent=2)

    base_pc = path_metrics.get("base_per_class_acc", {}) if path_metrics else {}
    tail_pc = path_metrics.get("tail_per_class_acc", {}) if path_metrics else {}
    fused_pc = path_metrics.get("fused_per_class_acc", {}) if path_metrics else per_class_acc
    with open(os.path.join(output_dir, "per_class_metrics.csv"), "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "class_id",
                "class_group",
                "global_count",
                "num_support_clients",
                "exposure_E",
                "exposure_score_s",
                "protected",
                "per_class_acc_base",
                "per_class_acc_tail_only",
                "per_class_acc_fedtef",
                "base_margin",
                "base_cosine_margin",
                "fused_margin",
                "changed_rate",
                "right_flip_rate",
                "wrong_flip_rate",
                "residual_abs",
                "fused_delta_abs",
                "semantic_rescue_gate",
                "release_reliability",
                "class_release_gate",
            ],
        )
        writer.writeheader()
        for cls in range(num_classes):
            writer.writerow({
                "class_id": cls,
                "class_group": class_groups.get(cls, "head"),
                "global_count": float(global_counts[cls].item()),
                "num_support_clients": int(class_num_clients[cls].item()),
                "exposure_E": float(exposure_count[cls].item()),
                "exposure_score_s": float(tail_score[cls].item()),
                "protected": bool(protected_tail_mask[cls].item()),
                "per_class_acc_base": float(base_pc.get(cls, "")) if cls in base_pc else "",
                "per_class_acc_tail_only": float(tail_pc.get(cls, "")) if cls in tail_pc else "",
                "per_class_acc_fedtef": float(fused_pc.get(cls, 0.0)),
                "base_margin": float(path_metrics.get("base_margin_per_class", {}).get(cls, "")) if path_metrics and cls in path_metrics.get("base_margin_per_class", {}) else "",
                "base_cosine_margin": float(path_metrics.get("base_cosine_margin_per_class", {}).get(cls, "")) if path_metrics and cls in path_metrics.get("base_cosine_margin_per_class", {}) else "",
                "fused_margin": float(path_metrics.get("fused_margin_per_class", {}).get(cls, "")) if path_metrics and cls in path_metrics.get("fused_margin_per_class", {}) else "",
                "changed_rate": float(path_metrics.get("changed_rate_per_class", {}).get(cls, "")) if path_metrics and cls in path_metrics.get("changed_rate_per_class", {}) else "",
                "right_flip_rate": float(path_metrics.get("right_flip_rate_per_class", {}).get(cls, "")) if path_metrics and cls in path_metrics.get("right_flip_rate_per_class", {}) else "",
                "wrong_flip_rate": float(path_metrics.get("wrong_flip_rate_per_class", {}).get(cls, "")) if path_metrics and cls in path_metrics.get("wrong_flip_rate_per_class", {}) else "",
                "residual_abs": float(path_metrics.get("residual_abs_per_class", {}).get(cls, "")) if path_metrics and cls in path_metrics.get("residual_abs_per_class", {}) else "",
                "fused_delta_abs": float(path_metrics.get("fused_delta_abs_per_class", {}).get(cls, "")) if path_metrics and cls in path_metrics.get("fused_delta_abs_per_class", {}) else "",
                "semantic_rescue_gate": float(path_metrics.get("semantic_rescue_gate_per_class", {}).get(cls, "")) if path_metrics and cls in path_metrics.get("semantic_rescue_gate_per_class", {}) else "",
                "release_reliability": float(path_metrics.get("release_reliability_per_class", {}).get(cls, "")) if path_metrics and cls in path_metrics.get("release_reliability_per_class", {}) else "",
                "class_release_gate": float(path_metrics.get("class_release_gate_per_class", {}).get(cls, "")) if path_metrics and cls in path_metrics.get("class_release_gate_per_class", {}) else "",
            })


def load_fedtef_server_state(resume_dir, num_classes):
    exposure_count = torch.zeros(num_classes, dtype=torch.float32)
    tail_score = torch.ones(num_classes, dtype=torch.float32)
    protected_tail_mask = torch.ones(num_classes, dtype=torch.bool)
    if not resume_dir:
        return exposure_count, tail_score, protected_tail_mask

    state_path = os.path.join(resume_dir, "fedtef_state_latest.pth")
    if not os.path.exists(state_path):
        print(f"FedTEF server state not found at {state_path}, start from scratch")
        return exposure_count, tail_score, protected_tail_mask

    state = torch.load(state_path, map_location="cpu")
    exposure_count = state.get("exposure_count", exposure_count).float()
    tail_score = state.get("tail_score", tail_score).float()
    protected_tail_mask = state.get("protected_tail_mask", protected_tail_mask).bool()
    print(f"Loaded FedTEF server state from {state_path}")
    return exposure_count, tail_score, protected_tail_mask


def load_fedtef_model_state(resume_dir):
    if not resume_dir:
        return None
    state_path = os.path.join(resume_dir, "fedtef_state_latest.pth")
    if not os.path.exists(state_path):
        return None
    state = torch.load(state_path, map_location="cpu")
    return state.get("state_dict", None)



def print_args(args, cfg):
    print("***************")
    print("** Arguments **")
    print("***************")
    optkeys = list(args.__dict__.keys())
    optkeys.sort()
    for key in optkeys:
        print("{}: {}".format(key, args.__dict__[key]))
    print("************")
    print("** Config **")
    print("************")
    print(cfg)


def reset_cfg(cfg, args):
    if args.root:
        cfg.DATASET.ROOT = args.root
        cfg.DATASET.imagenetROOT = args.imagenetroot

    if args.dataset:
        cfg.DATASET.NAME = normalize_dataset_name(args.dataset)

    if args.output_dir:
        cfg.OUTPUT_DIR = args.output_dir

    if args.resume:
        cfg.RESUME = args.resume

    if args.seed:
        cfg.SEED = args.seed

    if args.transforms:
        cfg.INPUT.TRANSFORMS = args.transforms

    if args.trainer:
        cfg.TRAINER.NAME = args.trainer

    if args.backbone:
        cfg.MODEL.BACKBONE.NAME = args.backbone

    if args.head:
        cfg.MODEL.HEAD.NAME = args.head


def extend_cfg(cfg, args):
    """
    Add new config variables.

    E.g.
        from yacs.config import CfgNode as CN
        cfg.TRAINER.MY_MODEL = CN()
        cfg.TRAINER.MY_MODEL.PARAM_A = 1.
        cfg.TRAINER.MY_MODEL.PARAM_B = 0.5
        cfg.TRAINER.MY_MODEL.PARAM_C = False
    """
    from yacs.config import CfgNode as CN

    cfg.TRAINER.PROMPTFL = CN()
    cfg.TRAINER.PROMPTFL.N_CTX = args.n_ctx  # number of context vectors

    try:
        cfg.TRAINER.PROMPTFL.CSC = ast.literal_eval(args.csc)  # class-specific context
    except ValueError:
        # print(f"Warning: Unable to convert '{args.csc}' to bool. Using string value.")
        cfg.TRAINER.PROMPTFL.CSC = args.csc

    try:
        cfg.TRAINER.PROMPTFL.CTX_INIT = ast.literal_eval(args.ctx_init)
    except ValueError:
        # print(f"Warning: Unable to convert '{args.ctx_init}' to bool. Using string value.")
        cfg.TRAINER.PROMPTFL.CTX_INIT = args.ctx_init
    # print(f"CSC value after setting: {cfg.TRAINER.PROMPTFL.CSC}, type: {type(cfg.TRAINER.PROMPTFL.CSC)}")
    # print(f"CSC value after setting: {cfg.TRAINER.PROMPTFL.CTX_INIT}, type: {type(cfg.TRAINER.PROMPTFL.CTX_INIT)}")
    cfg.TRAINER.PROMPTFL.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.PROMPTFL.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.PROMPTFL.n_general = args.n_general
    cfg.DATASET.USE_LMDB = True


    # ProCo
    cfg.TRAINER.PROMPTFL.TEMPERATURE = 0.1
    cfg.TRAINER.PROMPTFL.PROCO_WEIGHT = 1.0
    cfg.TRAINER.PROMPTFL.feat_dim = 512
    cfg.TRAINER.PROMPTFL.TAU = 1.0

    # New loss
    cfg.TRAINER.PROMPTFL.PCL_WEIGHT = 1.0

    # prompt_loss
    cfg.TRAINER.PROMPTFL.ALPHA = 1.0
    cfg.TRAINER.PROMPTFL.BETA = 1.0
    cfg.TRAINER.PROMPTFL.GAMMA = 0.1
    cfg.TRAINER.PROMPTFL.DELTA = 0
    cfg.TRAINER.PROMPTFL.MARGIN = 0.5

    cfg.TRAINER.PROMPTFL.PROMPT_DEPTH = args.prompt_depth

    cfg.TRAINER.FEDCLIP_TAIL = CN()
    cfg.TRAINER.FEDCLIP_TAIL.CUTOFF = args.fedclip_tail_cutoff

    cfg.TRAINER.FEDTEF = CN()
    cfg.TRAINER.FEDTEF.VERSION = args.fedtef_version
    cfg.TRAINER.FEDTEF.METHOD = args.method
    cfg.TRAINER.FEDTEF.USE_TAIL_EXPERT = args.use_tail_expert
    cfg.TRAINER.FEDTEF.TAIL_EXPERT_TYPE = args.tail_expert_type
    cfg.TRAINER.FEDTEF.TAIL_SCORE_TYPE = args.tail_score_type
    cfg.TRAINER.FEDTEF.POSITIVE_GATE = args.positive_gate
    cfg.TRAINER.FEDTEF.CLASSWISE_TAIL_AGG = args.classwise_tail_agg
    cfg.TRAINER.FEDTEF.FUSION_LAMBDA = args.fusion_lambda
    cfg.TRAINER.FEDTEF.TAIL_AGG_WEIGHT = args.tail_agg_weight
    cfg.TRAINER.FEDTEF.TAIL_EXPERT_ZERO_INIT = args.tail_expert_zero_init
    cfg.TRAINER.FEDTEF.SAVE_TAIL_SCORE = args.save_tail_score
    cfg.TRAINER.FEDTEF.WARMUP_ROUNDS_FOR_EXPOSURE = args.warmup_rounds_for_exposure
    cfg.TRAINER.FEDTEF.GATE_TAIL_UPDATE_BY_PROTECTION = args.gate_tail_update_by_protection
    cfg.TRAINER.FEDTEF.USE_EXPOSURE = args.tef_use_exposure
    cfg.TRAINER.FEDTEF.AGGREGATION = args.tef_aggregation
    cfg.TRAINER.FEDTEF.PROTECTED_STRATEGY = args.tef_protected_strategy
    cfg.TRAINER.FEDTEF.TAIL_FUSION_MODE = args.tail_fusion_mode
    cfg.TRAINER.FEDTEF.TAIL_UPDATE_PROTECT_MODE = args.tail_update_protect_mode
    cfg.TRAINER.FEDTEF.PROTECTED_TAIL_RATIO = args.protected_tail_ratio
    cfg.TRAINER.FEDTEF.TAIL_SCORE_THRESHOLD = args.tail_score_threshold
    cfg.TRAINER.FEDTEF.DEBUG = args.fedtef_debug
    cfg.TRAINER.FEDTEF.DEBUG_INTERVAL = args.fedtef_debug_interval
    cfg.TRAINER.FEDTEF.TAIL_EXPERT_LR_MULT = args.tail_expert_lr_mult
    cfg.TRAINER.FEDTEF.FREEZE_IMG_ADAP = args.freeze_img_adap
    cfg.TRAINER.FEDTEF.TAIL_EXPERT_MODE = args.tail_expert_mode
    cfg.TRAINER.FEDTEF.TAIL_INIT_LOGIT_SCALE = args.tail_init_logit_scale
    cfg.TRAINER.FEDTEF.TAIL_LEARNABLE_SCALE = args.tail_learnable_scale
    cfg.TRAINER.FEDTEF.TAIL_USE_BIAS = args.tail_use_bias
    cfg.TRAINER.FEDTEF.TAIL_LOGIT_SCALE_MAX = args.tail_logit_scale_max
    cfg.TRAINER.FEDTEF.TRAIN_PROMPT = args.fedtef_train_prompt
    cfg.TRAINER.FEDTEF.TRAIN_IMG_ADAP = args.fedtef_train_img_adap
    cfg.TRAINER.FEDTEF.TRAIN_TAIL_STREAM = args.fedtef_train_tail_stream
    cfg.TRAINER.FEDTEF.TAIL_STREAM_MODE = args.fedtef_tail_stream_mode
    cfg.TRAINER.FEDTEF.DECOUPLE_TAIL_LOSS = args.fedtef_decouple_tail_loss
    cfg.TRAINER.FEDTEF.FUSION_MODE = args.fedtef_fusion_mode
    cfg.TRAINER.FEDTEF.SCALE_CALIBRATION = args.fedtef_scale_calibration
    cfg.TRAINER.FEDTEF.SCALE_CLAMP_MAX = args.fedtef_scale_clamp_max
    cfg.TRAINER.FEDTEF.GATE_MODE = args.fedtef_gate_mode
    cfg.TRAINER.FEDTEF.GATE_SCORE_MODE = args.fedtef_gate_score_mode
    cfg.TRAINER.FEDTEF.GATE_TEMPERATURE = args.fedtef_gate_temperature
    cfg.TRAINER.FEDTEF.GATE_THRESHOLD = args.fedtef_gate_threshold
    cfg.TRAINER.FEDTEF.TAIL_TOPK = args.fedtef_tail_topk
    cfg.TRAINER.FEDTEF.EXPOSURE_EMA_RHO = args.fedtef_exposure_ema_rho
    cfg.TRAINER.FEDTEF.EXPOSURE_EPS = args.fedtef_exposure_eps
    cfg.TRAINER.FEDTEF.INIT_TAIL_MODE = args.fedtef_init_tail_mode
    cfg.TRAINER.FEDTEF.ROUND0_TIE_BREAK = args.fedtef_round0_tie_break
    cfg.TRAINER.FEDTEF.WARMUP_MODE = args.fedtef_warmup_mode
    cfg.TRAINER.FEDTEF.WARMUP_ROUNDS = args.fedtef_warmup_rounds
    cfg.TRAINER.FEDTEF.LOSS_BASE_WEIGHT = args.fedtef_loss_base_weight
    cfg.TRAINER.FEDTEF.LOSS_PROTECTED_BASE_WEIGHT = args.fedtef_loss_protected_base_weight
    cfg.TRAINER.FEDTEF.LOSS_PROTECTED_BASE_MARGIN_WEIGHT = args.fedtef_loss_protected_base_margin_weight
    cfg.TRAINER.FEDTEF.PROTECTED_BASE_MARGIN = args.fedtef_protected_base_margin
    cfg.TRAINER.FEDTEF.ACQUISITION_LOW_EXPOSURE_WEIGHT = args.fedtef_acquisition_low_exposure_weight
    cfg.TRAINER.FEDTEF.ACQUISITION_SIGNAL_SOURCE = args.fedtef_acquisition_signal_source
    cfg.TRAINER.FEDTEF.ACQUISITION_SIGNAL_CLAMP_MAX = args.fedtef_acquisition_signal_clamp_max
    cfg.TRAINER.FEDTEF.ACQUISITION_WEIGHT_NORMALIZE = args.fedtef_acquisition_weight_normalize
    cfg.TRAINER.FEDTEF.TAIL_STREAM_DETACH_BASE = args.fedtef_tail_stream_detach_base
    cfg.TRAINER.FEDTEF.V10_EXPOSURE_BUDGET = args.fedtef_v10_exposure_budget
    cfg.TRAINER.FEDTEF.V10_SURVIVAL_BUDGET = args.fedtef_v10_survival_budget
    cfg.TRAINER.FEDTEF.V10_MIN_HOLD = args.fedtef_v10_min_hold
    cfg.TRAINER.FEDTEF.V10_REPLACE_MARGIN = args.fedtef_v10_replace_margin
    cfg.TRAINER.FEDTEF.V10_DIFFICULTY_POWER = args.fedtef_v10_difficulty_power
    cfg.TRAINER.FEDTEF.V10_OBSERVER_W_EXPOSURE = args.fedtef_v10_observer_w_exposure
    cfg.TRAINER.FEDTEF.V10_OBSERVER_W_AGE = args.fedtef_v10_observer_w_age
    cfg.TRAINER.FEDTEF.V10_OBSERVER_W_SURVIVAL = args.fedtef_v10_observer_w_survival
    cfg.TRAINER.FEDTEF.V10_DIFFICULTY_MARGIN = args.fedtef_v10_difficulty_margin
    cfg.TRAINER.FEDTEF.V10_PRIOR_BASE_WEIGHT = args.fedtef_v10_prior_base_weight
    cfg.TRAINER.FEDTEF.V10_PRIOR_KAPPA = args.fedtef_v10_prior_kappa
    cfg.TRAINER.FEDTEF.V10_PRIOR_W_MAX = args.fedtef_v10_prior_w_max
    cfg.TRAINER.FEDTEF.V10_HARDNEG_TOPM = args.fedtef_v10_hardneg_topm
    cfg.TRAINER.FEDTEF.V10_HARDNEG_LAMBDA = args.fedtef_v10_hardneg_lambda
    cfg.TRAINER.FEDTEF.V10_RELEASE_FLOOR = args.fedtef_v10_release_floor
    cfg.TRAINER.FEDTEF.V10_SAMPLE_LAMBDA_MIN = args.fedtef_v10_sample_lambda_min
    cfg.TRAINER.FEDTEF.V10_SAMPLE_LAMBDA_MAX = args.fedtef_v10_sample_lambda_max
    cfg.TRAINER.FEDTEF.V10_SAMPLE_MARGIN = args.fedtef_v10_sample_margin
    cfg.TRAINER.FEDTEF.V10_SAMPLE_TEMPERATURE = args.fedtef_v10_sample_temperature
    cfg.TRAINER.FEDTEF.V10_SAFE_CONF_THRESHOLD = args.fedtef_v10_safe_conf_threshold
    cfg.TRAINER.FEDTEF.V10_SAFE_MARGIN = args.fedtef_v10_safe_margin
    cfg.TRAINER.FEDTEF.V10_EVIDENCE_THRESHOLD = args.fedtef_v10_evidence_threshold
    cfg.TRAINER.FEDTEF.V10_AGG_UPDATE_CLIP = args.fedtef_v10_agg_update_clip
    cfg.TRAINER.FEDTEF.V10_AGG_BASE_MOMENTUM = args.fedtef_v10_agg_base_momentum
    cfg.TRAINER.FEDTEF.V10_AGG_LOW_SURVIVAL_MOMENTUM = args.fedtef_v10_agg_low_survival_momentum
    cfg.TRAINER.FEDTEF.LOSS_FUSED_WEIGHT = args.fedtef_loss_fused_weight
    cfg.TRAINER.FEDTEF.LOSS_TAIL_WEIGHT = args.fedtef_loss_tail_weight
    cfg.TRAINER.FEDTEF.LOSS_KEEP_KL_WEIGHT = args.fedtef_loss_keep_kl_weight
    cfg.TRAINER.FEDTEF.LOSS_REG_WEIGHT = args.fedtef_loss_reg_weight
    cfg.TRAINER.FEDTEF.LOSS_TAIL_ONLY_WEIGHT = args.fedtef_loss_tail_only_weight
    cfg.TRAINER.FEDTEF.LOSS_TAIL_MARGIN_WEIGHT = args.fedtef_loss_tail_margin_weight
    cfg.TRAINER.FEDTEF.TAIL_MARGIN = args.fedtef_tail_margin
    cfg.TRAINER.FEDTEF.TAIL_LOSS_NORMALIZE = args.fedtef_tail_loss_normalize
    cfg.TRAINER.FEDTEF.PRIOR_BALANCED_BASE_WEIGHT = args.fedtef_prior_balanced_base_weight
    cfg.TRAINER.FEDTEF.PRIOR_BALANCE_ALPHA = args.fedtef_prior_balance_alpha
    cfg.TRAINER.FEDTEF.PRIOR_BALANCE_CLAMP_MIN = args.fedtef_prior_balance_clamp_min
    cfg.TRAINER.FEDTEF.PRIOR_BALANCE_CLAMP_MAX = args.fedtef_prior_balance_clamp_max
    cfg.TRAINER.FEDTEF.KL_TEMPERATURE = args.fedtef_kl_temperature
    cfg.TRAINER.FEDTEF.TAILAGG_ENABLED = args.fedtef_tailagg_enabled
    cfg.TRAINER.FEDTEF.TAILAGG_MODE = args.fedtef_tailagg_mode
    cfg.TRAINER.FEDTEF.TAILAGG_FALLBACK = args.fedtef_tailagg_fallback
    cfg.TRAINER.FEDTEF.TAILAGG_CONFLICT_GAMMA = args.fedtef_tailagg_conflict_gamma
    cfg.TRAINER.FEDTEF.TAILAGG_MIN_AGREEMENT = args.fedtef_tailagg_min_agreement
    cfg.TRAINER.FEDTEF.TAIL_HIDDEN_DIM = args.fedtef_tail_hidden_dim
    cfg.TRAINER.FEDTEF.IMG_ADAP_ETA = args.fedtef_img_adap_eta
    cfg.TRAINER.FEDTEF.TAIL_NEED_W_SCARCITY = args.fedtef_tail_need_w_scarcity
    cfg.TRAINER.FEDTEF.TAIL_NEED_W_RESIDUAL = args.fedtef_tail_need_w_residual
    cfg.TRAINER.FEDTEF.TAIL_NEED_W_FORGETTING = args.fedtef_tail_need_w_forgetting
    cfg.TRAINER.FEDTEF.TAIL_NEED_W_UNCERTAINTY = args.fedtef_tail_need_w_uncertainty
    cfg.TRAINER.FEDTEF.TAIL_NEED_BETA = args.fedtef_tail_need_beta
    cfg.TRAINER.FEDTEF.GATE_MIN_HOLD = args.fedtef_gate_min_hold
    cfg.TRAINER.FEDTEF.GATE_EXIT_RATIO = args.fedtef_gate_exit_ratio
    cfg.TRAINER.FEDTEF.GATE_BUDGET = args.fedtef_gate_budget
    cfg.TRAINER.FEDTEF.GRADIENT_PRIOR_FLOOR = args.fedtef_gradient_prior_floor
    cfg.TRAINER.FEDTEF.GRADIENT_PRIOR_SCORE_POWER = args.fedtef_gradient_prior_score_power
    cfg.TRAINER.FEDTEF.GRADIENT_PRIOR_LOCK_ROUNDS = args.fedtef_gradient_prior_lock_rounds
    cfg.TRAINER.FEDTEF.GRADIENT_PRIOR_LOCK_MODE = args.fedtef_gradient_prior_lock_mode
    cfg.TRAINER.FEDTEF.GRADIENT_PRIOR_REFINE_RATIO = args.fedtef_gradient_prior_refine_ratio
    cfg.TRAINER.FEDTEF.GRADIENT_PRIOR_REFINE_MAX_SWAP = args.fedtef_gradient_prior_refine_max_swap
    cfg.TRAINER.FEDTEF.GRADIENT_PRIOR_REFINE_MARGIN = args.fedtef_gradient_prior_refine_margin
    cfg.TRAINER.FEDTEF.GRADIENT_PRIOR_LOCK_GATE_FLOOR = args.fedtef_gradient_prior_lock_gate_floor
    cfg.TRAINER.FEDTEF.GRADIENT_PRIOR_UPDATE_ALL_ROWS = args.fedtef_gradient_prior_update_all_rows
    cfg.TRAINER.FEDTEF.PRIOR_LOGIT_ADJUST = args.fedtef_prior_logit_adjust
    cfg.TRAINER.FEDTEF.PRIOR_LOGIT_ADJUST_TAU = args.fedtef_prior_logit_adjust_tau
    cfg.TRAINER.FEDTEF.PRIOR_LOGIT_ADJUST_CLAMP = args.fedtef_prior_logit_adjust_clamp
    cfg.TRAINER.FEDTEF.SAMPLE_GATE_ENABLED = args.fedtef_sample_gate_enabled
    cfg.TRAINER.FEDTEF.SAMPLE_GATE_TOPM = args.fedtef_sample_gate_topm
    cfg.TRAINER.FEDTEF.SAMPLE_GATE_USE_RESIDUAL_TOPM = args.fedtef_sample_gate_use_residual_topm
    cfg.TRAINER.FEDTEF.SAMPLE_GATE_UNCERTAINTY_POWER = args.fedtef_sample_gate_uncertainty_power
    cfg.TRAINER.FEDTEF.SAMPLE_GATE_CONF_THRESHOLD = args.fedtef_sample_gate_conf_threshold
    cfg.TRAINER.FEDTEF.SAMPLE_GATE_MIN = args.fedtef_sample_gate_min
    cfg.TRAINER.FEDTEF.SAMPLE_GATE_CANDIDATE_FLOOR = args.fedtef_sample_gate_candidate_floor
    cfg.TRAINER.FEDTEF.SEMANTIC_RESCUE_ENABLED = args.fedtef_semantic_rescue_enabled
    cfg.TRAINER.FEDTEF.SEMANTIC_RESCUE_TEMPERATURE = args.fedtef_semantic_rescue_temperature
    cfg.TRAINER.FEDTEF.SEMANTIC_RESCUE_MARGIN = args.fedtef_semantic_rescue_margin
    cfg.TRAINER.FEDTEF.SEMANTIC_RESCUE_MIN = args.fedtef_semantic_rescue_min
    cfg.TRAINER.FEDTEF.POSITIVE_RESIDUAL_ONLY = args.fedtef_positive_residual_only
    cfg.TRAINER.FEDTEF.RESIDUAL_CLAMP = args.fedtef_residual_clamp
    cfg.TRAINER.FEDTEF.RELEASE_RELIABILITY_ENABLED = args.fedtef_release_reliability_enabled
    cfg.TRAINER.FEDTEF.RELEASE_RELIABILITY_SOURCE = args.fedtef_release_reliability_source
    cfg.TRAINER.FEDTEF.RELEASE_RELIABILITY_FLOOR = args.fedtef_release_reliability_floor
    cfg.TRAINER.FEDTEF.RELEASE_RELIABILITY_TAU = args.fedtef_release_reliability_tau
    cfg.TRAINER.FEDTEF.RELEASE_RELIABILITY_POWER = args.fedtef_release_reliability_power
    cfg.TRAINER.FEDTEF.TRAIN_ROUTED_PROMPT = args.fedtef_train_routed_prompt
    cfg.TRAINER.FEDTEF.ROUTED_PROMPT_LR_MULT = args.fedtef_routed_prompt_lr_mult
    cfg.TRAINER.FEDTEF.ROUTED_PROMPT_SCALE = args.fedtef_routed_prompt_scale
    cfg.TRAINER.FEDTEF.ROUTED_PROMPT_UPDATE_ALL_ROWS = args.fedtef_routed_prompt_update_all_rows
    cfg.TRAINER.FEDTEF.EVIDENCE_MEMORY_UPDATE_ALL_ROWS = args.fedtef_evidence_memory_update_all_rows
    cfg.TRAINER.FEDTEF.TRAIN_LORA = args.fedtef_train_lora
    cfg.TRAINER.FEDTEF.LORA_LR_MULT = args.fedtef_lora_lr_mult

    # LoRa

    cfg.TRAINER.CLIPLORA = CN()
    cfg.TRAINER.CLIPLORA.backbone = 'ViT-B/16'
    cfg.TRAINER.CLIPLORA.lr = 2e-4
    cfg.TRAINER.CLIPLORA.n_iters = 500
    cfg.TRAINER.CLIPLORA.CTX_INIT = "a photo of a"
    cfg.TRAINER.CLIPLORA.position = args.fedtef_lora_position
    cfg.TRAINER.CLIPLORA.encoder = (
        args.fedtef_lora_encoder if args.trainer == "FedTEF" else args.encoder
    )
    cfg.TRAINER.CLIPLORA.r = args.fedtef_lora_rank
    cfg.TRAINER.CLIPLORA.alpha = args.fedtef_lora_alpha
    cfg.TRAINER.CLIPLORA.dropout_rate = args.fedtef_lora_dropout_rate
    cfg.TRAINER.CLIPLORA.params = args.fedtef_lora_params


    cfg.TRAINER.GLP_OT = CN()
    cfg.TRAINER.GLP_OT.N_CTX = args.n_ctx  # number of context vectors
    cfg.TRAINER.GLP_OT.CSC = False  # class-specific context
    cfg.TRAINER.GLP_OT.CTX_INIT = args.ctx_init  # initialization words
    cfg.TRAINER.GLP_OT.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.GLP_OT.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'
    cfg.TRAINER.GLP_OT.N = args.num_prompt  # number of prompts

    # Config for CoOp
    cfg.TRAINER.COOP = CN()
    cfg.TRAINER.COOP.N_CTX = args.n_ctx  # number of context vectors
    cfg.TRAINER.COOP.CSC = False  # class-specific context
    cfg.TRAINER.COOP.CTX_INIT = False  # initialization words
    cfg.TRAINER.COOP.W = 1.0
    cfg.TRAINER.COOP.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.COOP.CLASS_TOKEN_POSITION = "end"  # 'middle' or 'end' or 'front'

    cfg.TRAINER.COCOOP = CN()
    cfg.TRAINER.COCOOP.N_CTX = args.n_ctx  # number of context vectors 16
    cfg.TRAINER.COCOOP.CTX_INIT = ""  # initialization words
    cfg.TRAINER.COCOOP.PREC = "fp16"  # fp16, fp32, amp

    # Config for MaPLe
    cfg.TRAINER.MAPLE = CN()
    cfg.TRAINER.MAPLE.N_CTX = args.n_ctx  # number of context vectors 2
    cfg.TRAINER.MAPLE.CTX_INIT = "a photo of a"  # initialization words
    cfg.TRAINER.MAPLE.PREC = "fp16"  # fp16, fp32, amp
    cfg.TRAINER.MAPLE.PROMPT_DEPTH = 9  # Max 12, minimum 0, for 1 it will act as shallow MaPLe (J=1)
    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new



    cfg.DATASET.SUBSAMPLE_CLASSES = "all"  # all, base or new
    cfg.DATASET.USERS = args.num_users  # number of clients
    cfg.DATASET.PARTITION = args.partition

    cfg.DATASET.BETA = args.beta
    cfg.DATASET.REPEATRATE = 0.0  # repeat rate on each client
    cfg.DATASET.IMB_FACTOR = args.imb_factor
    cfg.DATASET.IMB_TYPE = args.imb_type
    cfg.DATASET.NUM_CLASSES = args.num_classes
    cfg.DATASET.HEAD_CLIENT_RATIO = args.head_client_ratio
    cfg.DATASET.TAIL_CLIENT_RATIO = args.tail_client_ratio
    cfg.DATASET.HEAD_CLASS_RATIO = args.head_class_ratio
    cfg.DATASET.TAIL_CLASS_RATIO = args.tail_class_ratio
    cfg.DATASET.SPECIALIZATION_LAMBDA = args.specialization_lambda
    cfg.DATASET.INTRA_GROUP_ALPHA = args.intra_group_alpha
    cfg.DATASET.HEAD_LEAKAGE_SCALE = args.head_leakage_scale
    cfg.DATASET.SPLIT_SEED = int(args.split_seed)
    cfg.DATASET.LOGIT_ADJUST = args.logit_adjust
    cfg.DATASET.LOGIT_ADJUST_TAU = args.logit_adjust_tau

    cfg.OPTIM.ROUND = args.round  # global round
    cfg.OPTIM.GAMMA = args.gamma  # gamma of single-step
    cfg.OPTIM.LR = args.lr  # learning rate

    cfg.MODEL.BACKBONE.PRETRAINED = True
    cfg.DATASET.NAME = normalize_dataset_name(args.dataset)  # sync command-line dataset name to config




def setup_cfg(args):
    cfg = get_cfg_default()
    extend_cfg(cfg, args)

    # 1. From the dataset config file
    if args.dataset_config_file:
        cfg.merge_from_file(args.dataset_config_file)

    # 2. From the method config file
    if args.config_file:
        cfg.merge_from_file(args.config_file)

    cfg.DATALOADER.TRAIN_X.BATCH_SIZE = args.train_batch_size
    cfg.DATALOADER.TEST.BATCH_SIZE = args.test_batch_size

    # 3. From input arguments
    reset_cfg(cfg, args)
    # print_args(args, cfg)

    # 4. From optional input arguments
    cfg.merge_from_list(args.opts)

    apply_federated_runtime_overrides(cfg, args)

    cfg.freeze()

    return cfg


def main(args):
    cfg = setup_cfg(args)
    print(f"Resolved local epochs per selected client: {cfg.OPTIM.MAX_EPOCH}")
    if cfg.SEED >= 0:
        # print("Setting fixed seed: {}".format(cfg.SEED))
        set_random_seed(cfg.SEED)
    setup_logger(cfg.OUTPUT_DIR)

    if torch.cuda.is_available():
        torch.backends.cudnn.benchmark = True

    # print_args(args, cfg)
    # print("Collecting env info ...")
    # print("** System info **\n{}\n".format(collect_env_info()))
    if args.model != "local":
        global_trainer = build_trainer(cfg)
        global_trainer.prompt_loss = PromptLoss(
            num_classes=cfg.DATASET.NUM_CLASSES,
            temperature=cfg.TRAINER.PROMPTFL.TEMPERATURE,
            alpha=cfg.TRAINER.PROMPTFL.ALPHA,
            beta=cfg.TRAINER.PROMPTFL.BETA,
            gamma=cfg.TRAINER.PROMPTFL.GAMMA,
            delta=cfg.TRAINER.PROMPTFL.DELTA,
            margin=cfg.TRAINER.PROMPTFL.MARGIN
        )
        global_trainer.class_priors = torch.ones(cfg.DATASET.NUM_CLASSES) / cfg.DATASET.NUM_CLASSES
        # global_trainer.global_estimator = EstimatorCV(feature_num=cfg.TRAINER.PROMPTFL.feat_dim, class_num=cfg.DATASET.NUM_CLASSES)

        print("global_trainer_isbuild_type:", type(global_trainer))
        # count_parameters(global_trainer.model,"prompt_learner")
        # count_parameters(global_trainer.model, "image_encoder")
        # count_parameters(global_trainer.model, "text_encoder")
        global_trainer.fed_before_train(is_global=True)

        # copy weights
        global_weights = global_trainer.model.state_dict()
    # local_weights, local_losses = [], []

    local_weights = [[] for i in range(args.num_users)]  # different
    local_weights_0 = [[] for i in range(args.num_users)]
    local_weights_1 = [[] for i in range(args.num_users)]
    local_weights_per = [{} for i in range(args.num_users)]
    local_proj = [{} for i in range(args.num_users)]

    local_trainer = build_trainer(cfg)
    local_trainer.fed_before_train()
    validate_federated_train_loaders(local_trainer, args.num_users)

    datanumber_client = []

    if args.trainer == 'CLIP':  # different
        global_weights = copy.deepcopy(local_trainer.model.state_dict())
    else:
        for net_i in range(cfg.DATASET.USERS):
            # local_trainer = build_trainer(cfg)
            datanumber_client.append(len(local_trainer.fed_train_loader_x_dict[net_i].dataset))

    # Training
    start_epoch = 0
    max_epoch = cfg.OPTIM.ROUND
    client_schedule = load_or_create_client_schedule(
        args.client_schedule_file,
        max_epoch,
        args.num_users,
        args.frac,
        args.client_schedule_seed,
    )
    # global_trainer.before_train()

    global_test_acc_list = []
    global_test_error_list = []
    global_test_f1_list = []
    global_epoch_list = []
    global_time_list = []
    start = time.time()
    n_cls = len(local_trainer.dm.dataset.classnames)
    head_acc_list = []
    mid_acc_list = []
    tail_acc_list = []
    fedtef_enabled = (
        args.trainer == "FedTEF"
        and args.method.lower() == "fedtef"
        and args.use_tail_expert
    )
    fedtef_v2_enabled = fedtef_enabled and str(args.fedtef_version).lower() in (
        "v2", "v3", "v4", "v5", "v6", "v7", "v8", "v9", "v10"
    )
    fedtef_v10_enabled = fedtef_enabled and str(args.fedtef_version).lower() == "v10"
    if fedtef_v10_enabled:
        if not bool(args.positive_gate):
            raise ValueError("FedTEF-v10 requires --positive_gate true for positive-only tail evidence proxies.")
        if bool(args.fedtef_evidence_memory_update_all_rows):
            print(
                "FedTEF-v10 warning: EVIDENCE_MEMORY_UPDATE_ALL_ROWS=true lets tail-stream rows "
                "learn from every locally observed positive label, not only protected labels."
            )
    # Experimental diagnostics only: full per-client class counts are saved for
    # topology analysis. FedTEF itself consumes only binary support sums.
    client_class_counts = get_client_class_counts(local_trainer, args.num_users, n_cls)
    save_partition_summary(args.output_dir, client_class_counts, args, args.num_users, n_cls)
    save_client_split_fingerprint(args.output_dir, local_trainer, args.num_users)
    global_class_counts = client_counts_to_tensor(client_class_counts, args.num_users, n_cls).sum(dim=0)
    exposure_count = torch.zeros(n_cls, dtype=torch.float32)
    tail_score = torch.ones(n_cls, dtype=torch.float32)
    protected_tail_mask = torch.ones(n_cls, dtype=torch.bool)
    fedtef_v2_gate = torch.zeros(n_cls, dtype=torch.float32)
    fedtef_release_reliability = torch.ones(n_cls, dtype=torch.float32)
    exposure_tracker = None
    if fedtef_v2_enabled:
        tracker_kwargs = dict(
            num_classes=n_cls,
            rho=args.fedtef_exposure_ema_rho,
            eps=args.fedtef_exposure_eps,
            gate_mode=args.fedtef_gate_mode,
            temperature=args.fedtef_gate_temperature,
            threshold=args.fedtef_gate_threshold,
            tail_topk=args.fedtef_tail_topk,
            round0_tie_break=args.fedtef_round0_tie_break,
            warmup_mode=args.fedtef_warmup_mode,
            warmup_rounds=args.fedtef_warmup_rounds,
            seed=args.seed,
            dataset_name=args.dataset,
        )
        gate_score_mode = str(args.fedtef_gate_score_mode).lower()
        if fedtef_v10_enabled or gate_score_mode == "topology_observer":
            exposure_tracker = TopologyExposureSurvivalObserver(
                **tracker_kwargs,
                exposure_budget=args.fedtef_v10_exposure_budget,
                survival_budget=args.fedtef_v10_survival_budget,
                min_hold=args.fedtef_v10_min_hold,
                replace_margin=args.fedtef_v10_replace_margin,
                difficulty_power=args.fedtef_v10_difficulty_power,
                w_exposure=args.fedtef_v10_observer_w_exposure,
                w_age=args.fedtef_v10_observer_w_age,
                w_survival=args.fedtef_v10_observer_w_survival,
                evidence_threshold=args.fedtef_v10_evidence_threshold,
                reliability_floor=args.fedtef_v10_release_floor,
                oracle_bottom20=args.fedtef_oracle_bottom20,
                oracle_bottomk=args.fedtef_oracle_bottomk,
            )
            if args.fedtef_oracle_bottom20:
                print(
                    "FedTEF-v10 using ORACLE bottom-tail protected set: "
                    f"bottomk={args.fedtef_oracle_bottomk or '20%'}"
                )
            else:
                print(
                    "FedTEF-v10 using TopologyExposureSurvivalObserver: "
                    "E/D/S/A protected routing with exposure and survival budgets"
                )
        elif gate_score_mode == "evidence_memory":
            exposure_tracker = EvidenceMemoryTracker(
                **tracker_kwargs,
                reliability_tau=args.fedtef_evidence_memory_reliability_tau,
                gate_floor=args.fedtef_evidence_memory_gate_floor,
                residual_weight=args.fedtef_evidence_memory_residual_weight,
            )
            print("FedTEF-v4 using persistent semantic evidence-memory gate")
        elif gate_score_mode == "tail_need":
            exposure_tracker = TailNeedTracker(
                **tracker_kwargs,
                w_scarcity=args.fedtef_tail_need_w_scarcity,
                w_residual=args.fedtef_tail_need_w_residual,
                w_forgetting=args.fedtef_tail_need_w_forgetting,
                w_uncertainty=args.fedtef_tail_need_w_uncertainty,
                beta=args.fedtef_tail_need_beta,
                min_hold=args.fedtef_gate_min_hold,
                exit_ratio=args.fedtef_gate_exit_ratio,
                budget=args.fedtef_gate_budget,
            )
            print("FedTEF-v2 using TailNeedTracker persistent dynamic gate")
        elif gate_score_mode == "gradient_prior":
            exposure_tracker = GradientPriorTracker(
                **tracker_kwargs,
                prior_floor=args.fedtef_gradient_prior_floor,
                score_power=args.fedtef_gradient_prior_score_power,
                lock_rounds=args.fedtef_gradient_prior_lock_rounds,
                lock_mode=args.fedtef_gradient_prior_lock_mode,
                refine_ratio=args.fedtef_gradient_prior_refine_ratio,
                refine_max_swap=args.fedtef_gradient_prior_refine_max_swap,
                refine_margin=args.fedtef_gradient_prior_refine_margin,
                lock_gate_floor=args.fedtef_gradient_prior_lock_gate_floor,
                update_all_rows=args.fedtef_gradient_prior_update_all_rows,
            )
            print("FedTEF-v2 using GradientPriorTracker inverse class-prior gate")
            if not args.positive_gate:
                print(
                    "FedTEF-v2 gradient-prior warning: --positive_gate is false, "
                    "so row-update proxies may include negative-class softmax gradients."
                )
        elif gate_score_mode == "low_exposure_router":
            exposure_tracker = LowExposureRouterTracker(
                **tracker_kwargs,
                prior_floor=args.fedtef_gradient_prior_floor,
                score_power=args.fedtef_gradient_prior_score_power,
                lock_rounds=args.fedtef_gradient_prior_lock_rounds,
                lock_mode=args.fedtef_gradient_prior_lock_mode,
                refine_ratio=args.fedtef_gradient_prior_refine_ratio,
                refine_max_swap=args.fedtef_gradient_prior_refine_max_swap,
                refine_margin=args.fedtef_gradient_prior_refine_margin,
                lock_gate_floor=args.fedtef_gradient_prior_lock_gate_floor,
                update_all_rows=args.fedtef_gradient_prior_update_all_rows,
            )
            print(
                "FedTEF-v6 using LowExposureRouter: inverse positive-row "
                "prior for sparse protected evidence routing"
            )
            if not args.fedtef_gradient_prior_update_all_rows:
                print(
                    "FedTEF-v6 low-exposure warning: update_all_rows is false; "
                    "unprotected classes may not contribute to prior discovery."
                )
        else:
            exposure_tracker = ExposureTracker(**tracker_kwargs)
            print("FedTEF-v2 using ExposureTracker dynamic gate")
        if fedtef_v10_enabled and hasattr(exposure_tracker, "preview_gate"):
            fedtef_v2_gate, tail_score, protected_tail_mask = exposure_tracker.preview_gate(current_round=0)
        else:
            fedtef_v2_gate, tail_score, protected_tail_mask = exposure_tracker.compute_gate(current_round=0)
        if fedtef_v10_enabled and hasattr(exposure_tracker, "reliability"):
            fedtef_release_reliability = exposure_tracker.reliability.clone()
        else:
            fedtef_release_reliability = compute_fedtef_release_reliability(
                exposure_tracker,
                fedtef_v2_gate,
                protected_tail_mask,
                args,
            )
    if args.trainer == "FedTEF" and args.model != "local":
        exposure_count, tail_score, protected_tail_mask = load_fedtef_server_state(args.resume, n_cls)
        if fedtef_v2_enabled:
            exposure_tracker.exposure = exposure_count.clone()
            if hasattr(exposure_tracker, "tail_need_ema"):
                exposure_tracker.tail_need_ema = exposure_count.clone()
            if hasattr(exposure_tracker, "class_prior_ema"):
                exposure_tracker.class_prior_ema = torch.clamp(exposure_count.clone(), min=0.0)
            if fedtef_v10_enabled and hasattr(exposure_tracker, "preview_gate"):
                fedtef_v2_gate, tail_score, protected_tail_mask = exposure_tracker.preview_gate(current_round=0)
            else:
                fedtef_v2_gate, tail_score, protected_tail_mask = exposure_tracker.compute_gate(current_round=0)
            if fedtef_v10_enabled and hasattr(exposure_tracker, "reliability"):
                fedtef_release_reliability = exposure_tracker.reliability.clone()
            else:
                fedtef_release_reliability = compute_fedtef_release_reliability(
                    exposure_tracker,
                    fedtef_v2_gate,
                    protected_tail_mask,
                    args,
                )
        apply_fedtef_tail_context(
            global_trainer.model,
            tail_score,
            protected_tail_mask,
            gate=fedtef_v2_gate,
            release_reliability=fedtef_release_reliability,
        )
        apply_fedtef_tail_context(
            local_trainer.model,
            tail_score,
            protected_tail_mask,
            gate=fedtef_v2_gate,
            release_reliability=fedtef_release_reliability,
        )
        fedtef_model_state = load_fedtef_model_state(args.resume)
        if fedtef_model_state is not None:
            global_trainer.model.load_state_dict(fedtef_model_state, strict=False)
            print("Loaded FedTEF global model state from resume checkpoint")
        global_weights = global_trainer.model.state_dict()
        print("FedTEF client_class_counts are ready")

    # local_coupling_params = []
    local_coupling_params = [[] for i in range(args.num_users)]

    # 绘图
    best_acc = 0
    best_epoch = 0
    last_class_accuracy = []
    best_per_class_accuracies = None
    per_class_accuracies = []
    best_class_accuracy = []

    # Initialize MAB schedulers
    mab = MABScheduler([1, 2, 3, 5, 7, 10])

    save_dir = os.path.join(args.output_dir, 'prompt_params')
    os.makedirs(save_dir, exist_ok=True)

    for epoch in range(start_epoch, max_epoch):
        run_global_eval = should_run_global_eval(epoch, max_epoch, args.global_eval_interval)

        if args.trainer == 'CLIP':
            print("------------trainer == CLIP, global test_acpfl start-------------")
            # update global weights
            global_trainer.model.load_state_dict(global_weights)
            result = global_trainer.global_test(is_global=True, current_epoch=epoch)
            global_test_acc_list.append(result[0])
            global_test_error_list.append(result[1])
            global_test_f1_list.append(result[2])
            global_epoch_list.append(epoch)
            global_time_list.append(time.time() - start)
            last_class_accuracy = result[3]
            if result[0] > best_acc:
                best_acc = result[0]
                best_class_accuracy = last_class_accuracy
            print("------------global test_acpfl finish-------------")
            print("global_test_acc_list:", global_test_acc_list)
            print("maximum test_acpfl acc:", max(global_test_acc_list))
            print("mean of acc:", np.mean(global_test_acc_list[-5:]))
            print("std of acc:", np.std(global_test_acc_list[-5:]))

            head_acc, medium_acc, tail_acc, overall_acc = calculate_accuracy_tail20_compat(
                last_class_accuracy,
                local_trainer,
                args.tail_class_ratio,
            )
            head_acc_list.append(head_acc)
            mid_acc_list.append(medium_acc)
            tail_acc_list.append(tail_acc)
            append_round_metrics(
                args.output_dir,
                args,
                epoch,
                result,
                head_acc,
                tail_acc,
                overall_acc,
                last_class_accuracy,
            )

            print("Epoch on server :", epoch)
            break

        elif args.model == "cluster":
            if args.trainer == "CAPT":
                epoch_start_time = time.time()

                print(f"------------Epoch {epoch}: CAPT cluster training------------")

                m = max(int(args.frac * args.num_users), 1)
                idxs_users = select_round_clients(args, epoch, client_schedule)
                print(f"Selected clients for this round: {idxs_users}")

                for idx in idxs_users:


                    local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                    local_weight = local_trainer.model.state_dict()
                    local_weights[idx] = copy.deepcopy(local_weight)

                    local_coupling_param = local_trainer.model.coupling_function.state_dict()
                    local_coupling_params[idx] = copy.deepcopy(local_coupling_param)

                print("------------local train finish epoch:", epoch, "-------------")

                client_proportions = get_client_proportions(local_trainer.client_proportion, idxs_users,
                                                            cfg.DATASET.NUM_CLASSES)

                print("------------Client clustering start-------------")

                # MAB parameter selection
                intra_cluster_rounds = mab.get_value(mab.select_arm())
                similarity_iters = mab.get_value(mab.select_arm())
                dissimilarity_iters = mab.get_value(mab.select_arm())
                global_agg_freq = mab.get_value(mab.select_arm())

                print(f"MAB selected parameters: intra_cluster_rounds={intra_cluster_rounds}, "
                      f"similarity_iters={similarity_iters}, dissimilarity_iters={dissimilarity_iters}, "
                      f"global_agg_freq={global_agg_freq}")

                n_clusters_similarity = args.n_simclusters

                similarity_clusters = similarity_clustering(client_proportions, n_clusters_similarity)
                print("\nSimilarity Clustering:")
                print_cluster_results(similarity_clusters, idxs_users)

                for cluster in set(similarity_clusters):
                    cluster_members = [idxs_users[i] for i, c in enumerate(similarity_clusters) if c == cluster]
                    communicate_within_cluster_similarity(cluster_members, local_weights)

                n_clusters_dissimilarity = args.n_disclusters
                dissimilarity_clusters = dissimilarity_clustering(client_proportions, n_clusters_dissimilarity)
                print("\nDissimilarity Clustering:")
                print_cluster_results(dissimilarity_clusters, idxs_users)

                for cluster in set(dissimilarity_clusters):
                    cluster_members = [idxs_users[i] for i, c in enumerate(dissimilarity_clusters) if c == cluster]
                    communicate_within_cluster_dissimilarity(cluster_members, local_weights)

                print("------------Adaptive clustering and communication completed-------------")

                if epoch % global_agg_freq == 0:

                    global_class_aware_prompt = global_trainer.model.prompt_learner.class_aware_ctx.detach().cpu()
                    aggregated_class_aware_prompt = aggregate_class_aware_prompts(client_proportions, local_weights,
                                                                                  idxs_users, cfg.DATASET.NUM_CLASSES,
                                                                                  global_class_aware_prompt)

                    global_trainer.model.prompt_learner.class_aware_ctx.data = aggregated_class_aware_prompt.to(
                        global_trainer.model.prompt_learner.class_aware_ctx.device)
                    for idx in idxs_users:
                        local_weights[idx]['prompt_learner.class_aware_ctx'] = aggregated_class_aware_prompt.to(
                            local_weights[idx]['prompt_learner.class_aware_ctx'].device)

                    global_weights = average_weights(local_weights, idxs_users, datanumber_client)
                    global_trainer.model.load_state_dict(global_weights)

                    # update local
                    local_trainer.model.load_state_dict(global_weights, strict=False)

                    print("------------Global aggregation completed-------------")

                    if not run_global_eval:
                        print_skip_global_eval(epoch, args.global_eval_interval)
                        print("Epoch on server :", epoch)
                        continue

                    print("------------global test start-------------")
                    result = global_trainer.global_test(is_global=True, current_epoch=epoch)

                    prompt_state = {
                        'epoch': epoch,
                        'general_prompt': global_trainer.model.prompt_learner.general_ctx.detach().cpu(),
                        'class_aware_prompt': global_trainer.model.prompt_learner.class_aware_ctx.detach().cpu()
                    }
                    torch.save(prompt_state, os.path.join(save_dir, f'prompt_params_epoch_{epoch}.pth'))

                    accuracy, error, f1_score, last_class_accuracy = result[:4]
                    global_test_acc_list.append(result[0])
                    global_test_error_list.append(result[1])
                    global_test_f1_list.append(result[2])
                    global_epoch_list.append(epoch)
                    global_time_list.append(time.time() - start)

                    last_class_accuracy = result[3]
                    if result[0] > best_acc:
                        best_acc = result[0]
                        best_class_accuracy = last_class_accuracy

                        best_prompt_state = {
                            'epoch': epoch,
                            'general_prompt': global_trainer.model.prompt_learner.general_ctx.detach().cpu(),
                            'class_aware_prompt': global_trainer.model.prompt_learner.class_aware_ctx.detach().cpu(),
                            'accuracy': best_acc,
                            'class_accuracy': best_class_accuracy
                        }
                        torch.save(best_prompt_state, os.path.join(save_dir, 'best_prompt_params.pth'))


                    print("------------global test finish-------------")
                    print("global_test_acc_list:", global_test_acc_list)
                    print("maximum test_acpfl acc:", max(global_test_acc_list))
                    print("mean of acc:", np.mean(global_test_acc_list[-5:]))
                    print("std of acc:", np.std(global_test_acc_list[-5:]))
                    print(last_class_accuracy)
                    print(len(last_class_accuracy))

                    head_acc, medium_acc, tail_acc, overall_acc = calculate_accuracy_tail20_compat(
                        last_class_accuracy,
                        local_trainer,
                        args.tail_class_ratio,
                    )
                    head_acc_list.append(head_acc)
                    mid_acc_list.append(medium_acc)
                    tail_acc_list.append(tail_acc)
                    append_round_metrics(
                        args.output_dir,
                        args,
                        epoch,
                        result,
                        head_acc,
                        tail_acc,
                        overall_acc,
                        last_class_accuracy,
                    )

                    print("Epoch on server :", epoch)

                    # Calculate reward and update MAB
                    convergence_rate = mab.calculate_convergence_rate(global_test_acc_list)
                    reward = mab.calculate_reward(accuracy, f1_score, convergence_rate)

                    mab.update(mab.get_arm_from_value(intra_cluster_rounds), reward, epoch)
                    mab.update(mab.get_arm_from_value(similarity_iters), reward, epoch)
                    mab.update(mab.get_arm_from_value(dissimilarity_iters), reward, epoch)
                    mab.update(mab.get_arm_from_value(global_agg_freq), reward, epoch)

                    print(f"Epoch {epoch}: Updated MAB schedulers with reward {reward}")


                else:
                    print(f"Skipping global aggregation at epoch {epoch}")

                epoch_end_time = time.time()
                epoch_duration = epoch_end_time - epoch_start_time
                print(f"Epoch {epoch + 1} completed in {epoch_duration:.2f} seconds")

        elif args.model == "fedavg":
            if args.trainer in ("PromptFL", "PromptFLGeneralOnly"):
                print(f"use model == fedavg and trainer == {args.trainer}")
                m = max(int(args.frac * args.num_users), 1)  # different
                idxs_users = select_round_clients(args, epoch, client_schedule)
                # idxs_users = list(range(0,cfg.DATASET.USERS))
                print("idxs_users", idxs_users)
                print("------------local train start epoch:", epoch, "-------------")
                round_start_time = time.time()
                local_training_start = time.time()
                pre_global_weights = copy.deepcopy(global_weights)
                for idx in idxs_users:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                    use_promptfl_lifecycle = args.trainer == "PromptFL"
                    log_local_state = (
                        use_promptfl_lifecycle
                        and bool(args.isolate_local_optimizer_state)
                        and bool(args.federated_single_scheduler_step)
                    )

                    if use_promptfl_lifecycle and args.isolate_local_optimizer_state:
                        if not hasattr(local_trainer, "reset_optimizer_and_scheduler"):
                            raise RuntimeError(
                                "PromptFL trainer does not implement reset_optimizer_and_scheduler(), "
                                "required by --isolate_local_optimizer_state."
                            )
                        local_trainer.reset_optimizer_and_scheduler()

                    if log_local_state:
                        print(
                            f"Client {idx} local state reset:\n"
                            f"optimizer_state_entries={get_optimizer_state_entries(local_trainer)}\n"
                            f"initial_lr={get_first_optimizer_lr(local_trainer)}\n"
                            f"scheduler_last_epoch={get_scheduler_last_epoch(local_trainer)}"
                        )

                    if use_promptfl_lifecycle:
                        _, _, scheduler_step_delta, optimizer_step_count = run_promptfl_local_train_with_scheduler_policy(
                            local_trainer,
                            idx,
                            epoch,
                            args,
                            cfg.OPTIM.MAX_EPOCH,
                        )
                    else:
                        local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                        scheduler_step_delta = None
                        optimizer_step_count = None

                    local_batch_count = len(local_trainer.fed_train_loader_x_dict[idx])
                    client_num_samples = len(local_trainer.fed_train_loader_x_dict[idx].dataset)
                    print(
                        f"Client {idx} local train diagnostics: "
                        f"client_id={idx} client_num_samples={client_num_samples} "
                        f"local_batch_count={local_batch_count} "
                        f"local_optimizer_step_count={optimizer_step_count}"
                    )
                    if local_batch_count < 1:
                        raise RuntimeError(
                            f"Client {idx} has zero local batches during training"
                        )
                    if log_local_state:
                        print(
                            f"Client {idx} local training finished:\n"
                            f"final_lr={get_first_optimizer_lr(local_trainer)}\n"
                            f"scheduler_step_delta={scheduler_step_delta}\n"
                            f"local_optimizer_step_count={optimizer_step_count}"
                        )

                    local_weight = local_trainer.model.state_dict()
                    local_weights[idx] = copy.deepcopy(local_weight)
                print("------------local train finish epoch:", epoch, "-------------")
                local_training_seconds = time.time() - local_training_start
                if should_log_update_retention(args, epoch, max_epoch):
                    append_update_retention(
                        args.output_dir,
                        args,
                        epoch,
                        pre_global_weights,
                        local_weights,
                        idxs_users,
                        datanumber_client,
                        client_class_counts,
                        n_cls,
                        args.tail_class_ratio,
                    )
                # update global weights
                global_weights = average_weights(local_weights, idxs_users, datanumber_client)
                global_trainer.model.load_state_dict(global_weights)  # hsh

                experimentD_diagnostic_seconds = 0.0
                if bool(args.experimentD_enable):
                    experimentD_start = time.time()
                    if bool(args.experimentD_log_update_norm):
                        append_client_update_norms(
                            args.output_dir,
                            args,
                            epoch,
                            pre_global_weights,
                            local_weights,
                            idxs_users,
                            datanumber_client,
                            get_trainable_state_keys(global_trainer.model),
                        )
                    if should_log_experiment_d(args, epoch):
                        run_experiment_d_round(
                            args.output_dir,
                            args,
                            epoch,
                            global_trainer,
                            pre_global_weights,
                            global_weights,
                            local_weights,
                            idxs_users,
                            datanumber_client,
                            client_class_counts,
                            n_cls,
                        )
                    experimentD_diagnostic_seconds = time.time() - experimentD_start

                if not run_global_eval:
                    print_skip_global_eval(epoch, args.global_eval_interval)
                    print("Epoch on server :", epoch)
                    if bool(args.experimentD_enable):
                        append_runtime_metrics(
                            args.output_dir,
                            epoch,
                            args.local_epochs,
                            local_training_seconds,
                            experimentD_diagnostic_seconds,
                            0.0,
                            time.time() - round_start_time,
                            time.time() - start,
                        )
                    continue

                print("------------global test start-------------")
                global_eval_start = time.time()
                result = global_trainer.global_test(is_global=True, current_epoch=epoch)
                normal_global_eval_seconds = time.time() - global_eval_start

                prompt_state = {'epoch': epoch}
                prompt_state.update(collect_prompt_state(global_trainer.model.prompt_learner))
                torch.save(prompt_state, os.path.join(save_dir, f'prompt_params_epoch_{epoch}.pth'))

                global_test_acc_list.append(result[0])
                global_test_error_list.append(result[1])
                global_test_f1_list.append(result[2])
                global_epoch_list.append(epoch)
                global_time_list.append(time.time() - start)

                last_class_accuracy = result[3]  # 获取类别准确度
                # 更新最佳准确度和对应的类别准确度
                if result[0] > best_acc:
                    best_acc = result[0]
                    best_class_accuracy = last_class_accuracy

                    best_prompt_state = {
                        'epoch': epoch,
                        'accuracy': best_acc,
                        'class_accuracy': best_class_accuracy
                    }
                    best_prompt_state.update(collect_prompt_state(global_trainer.model.prompt_learner))
                    torch.save(best_prompt_state, os.path.join(save_dir, 'best_prompt_params.pth'))



                print("------------global test_acpfl finish-------------")
                print("global_test_acc_list:", global_test_acc_list)
                print("maximum test_acpfl acc:", max(global_test_acc_list))
                print("mean of acc:", np.mean(global_test_acc_list[-5:]))
                print("std of acc:", np.std(global_test_acc_list[-5:]))
                print(last_class_accuracy)
                print(len(last_class_accuracy))

                head_acc, medium_acc, tail_acc, overall_acc = calculate_accuracy_tail20_compat(
                    last_class_accuracy,
                    local_trainer,
                    args.tail_class_ratio,
                )
                head_acc_list.append(head_acc)
                mid_acc_list.append(medium_acc)
                tail_acc_list.append(tail_acc)
                append_round_metrics(
                    args.output_dir,
                    args,
                    epoch,
                    result,
                    head_acc,
                    tail_acc,
                    overall_acc,
                    last_class_accuracy,
                )
                if bool(args.experimentD_enable):
                    append_runtime_metrics(
                        args.output_dir,
                        epoch,
                        args.local_epochs,
                        local_training_seconds,
                        experimentD_diagnostic_seconds,
                        normal_global_eval_seconds,
                        time.time() - round_start_time,
                        time.time() - start,
                    )

                print("Epoch on server :", epoch)

            elif args.trainer == "CAPT":
                print(f"------------Epoch {epoch}: CAPT training, fedavg server------------")
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = select_round_clients(args, epoch, client_schedule)
                print(f"Selected clients for this round: {idxs_users}")
                print("------------local train start epoch:", epoch, "-------------")
                local_class_priors = []
                for idx in idxs_users:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                    local_trainer.prompt_loss = copy.deepcopy(global_trainer.prompt_loss)
                    local_trainer.class_priors = copy.deepcopy(global_trainer.class_priors)
                    local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                    local_weight = local_trainer.model.state_dict()
                    local_weights[idx] = copy.deepcopy(local_weight)
                    local_class_priors.append(local_trainer.class_priors)
                print("------------local train finish epoch:", epoch, "-------------")

                # update global weights
                global_weights = average_weights(local_weights, idxs_users, datanumber_client)
                global_trainer.model.load_state_dict(global_weights)

                # update global class priors
                global_trainer.class_priors = update_class_priors(global_trainer.class_priors, local_class_priors,
                                                                  idxs_users, datanumber_client)

                if not run_global_eval:
                    print_skip_global_eval(epoch, args.global_eval_interval)
                    print("Epoch on server :", epoch)
                    continue

                print("------------global test start-------------")
                result = global_trainer.global_test(is_global=True, current_epoch=epoch)
                global_test_acc_list.append(result[0])
                global_test_error_list.append(result[1])
                global_test_f1_list.append(result[2])
                global_epoch_list.append(epoch)
                global_time_list.append(time.time() - start)
                last_class_accuracy = result[3]
                if result[0] > best_acc:
                    best_acc = result[0]
                    best_class_accuracy = last_class_accuracy



                print("------------global test finish-------------")
                print("global_test_acc_list:", global_test_acc_list)
                print("maximum test acc:", max(global_test_acc_list))
                print("mean of acc:", np.mean(global_test_acc_list[-5:]))
                print("std of acc:", np.std(global_test_acc_list[-5:]))
                print(last_class_accuracy)
                print(len(last_class_accuracy))
                head_acc, mid_acc, tail_acc, overall_acc = calculate_accuracy_tail20_compat(
                    last_class_accuracy,
                    local_trainer,
                    args.tail_class_ratio,
                )
                head_acc_list.append(head_acc)
                mid_acc_list.append(mid_acc)
                tail_acc_list.append(tail_acc)
                append_round_metrics(
                    args.output_dir,
                    args,
                    epoch,
                    result,
                    head_acc,
                    tail_acc,
                    overall_acc,
                    last_class_accuracy,
                )
                print("Epoch on server :", epoch)

            elif args.trainer == "MaPLe":
                print("use model == fedavg and trainer == MaPLe")
                m = max(int(args.frac * args.num_users), 1)  # different
                idxs_users = select_round_clients(args, epoch, client_schedule)
                # idxs_users = list(range(0,cfg.DATASET.USERS))
                print("idxs_users", idxs_users)
                print("------------local train start epoch:", epoch, "-------------")
                for idx in idxs_users:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                    local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                    local_weight = local_trainer.model.state_dict()
                    local_weights[idx] = copy.deepcopy(local_weight)
                print("------------local train finish epoch:", epoch, "-------------")
                # update global weights
                global_weights = average_weights(local_weights, idxs_users, datanumber_client)
                # update global weights
                global_trainer.model.load_state_dict(global_weights)

                if not run_global_eval:
                    print_skip_global_eval(epoch, args.global_eval_interval)
                    print("Epoch on server :", epoch)
                    continue

                print("------------global test start-------------")
                result = global_trainer.global_test(is_global=True, current_epoch=epoch)
                global_test_acc_list.append(result[0])
                global_test_error_list.append(result[1])
                global_test_f1_list.append(result[2])
                global_epoch_list.append(epoch)
                global_time_list.append(time.time() - start)

                last_class_accuracy = result[3]
                if result[0] > best_acc:
                    best_acc = result[0]
                    best_class_accuracy = last_class_accuracy

                print("------------global test_acpfl finish-------------")
                print("global_test_acc_list:", global_test_acc_list)
                print("maximum test_acpfl acc:", max(global_test_acc_list))
                print("mean of acc:", np.mean(global_test_acc_list[-5:]))
                print("std of acc:", np.std(global_test_acc_list[-5:]))
                print(last_class_accuracy)
                print(len(last_class_accuracy))
                head_acc, medium_acc, tail_acc, overall_acc = calculate_accuracy_tail20_compat(
                    last_class_accuracy,
                    local_trainer,
                    args.tail_class_ratio,
                )
                head_acc_list.append(head_acc)
                mid_acc_list.append(medium_acc)
                tail_acc_list.append(tail_acc)
                print("Epoch on server :", epoch)

            elif args.trainer == "CoCoOp":
                print("use model == fedavg and trainer == CoCoOp")
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = select_round_clients(args, epoch, client_schedule)
                print("idxs_users", idxs_users)
                print("------------local train start epoch:", epoch, "-------------")
                for idx in idxs_users:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                    local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                    local_weight = local_trainer.model.state_dict()
                    local_weights[idx] = copy.deepcopy(local_weight)
                print("------------local train finish epoch:", epoch, "-------------")
                # update global weights
                global_weights = average_weights(local_weights, idxs_users, datanumber_client)
                global_trainer.model.load_state_dict(global_weights)

                if not run_global_eval:
                    print_skip_global_eval(epoch, args.global_eval_interval)
                    print("Epoch on server :", epoch)
                    continue

                print("------------global test start-------------")
                result = global_trainer.global_test(is_global=True, current_epoch=epoch)


                global_test_acc_list.append(result[0])
                global_test_error_list.append(result[1])
                global_test_f1_list.append(result[2])
                global_epoch_list.append(epoch)
                global_time_list.append(time.time() - start)

                last_class_accuracy = result[3]
                if result[0] > best_acc:
                    best_acc = result[0]
                    best_class_accuracy = last_class_accuracy

                print("------------global test_acpfl finish-------------")
                print("global_test_acc_list:", global_test_acc_list)
                print("maximum test_acpfl acc:", max(global_test_acc_list))
                print("mean of acc:", np.mean(global_test_acc_list[-5:]))
                print("std of acc:", np.std(global_test_acc_list[-5:]))
                print(last_class_accuracy)
                print(len(last_class_accuracy))

                head_acc, medium_acc, tail_acc, overall_acc = calculate_accuracy_tail20_compat(
                    last_class_accuracy,
                    local_trainer,
                    args.tail_class_ratio,
                )
                head_acc_list.append(head_acc)
                mid_acc_list.append(medium_acc)
                tail_acc_list.append(tail_acc)
                append_round_metrics(
                    args.output_dir,
                    args,
                    epoch,
                    result,
                    head_acc,
                    tail_acc,
                    overall_acc,
                    last_class_accuracy,
                )
                if args.trainer == "FedTEF" and fedtef_enabled and args.save_tail_score:
                    path_metrics = evaluate_fedtef_logit_paths(global_trainer)
                    save_fedtef_analysis(
                        args.output_dir,
                        epoch,
                        exposure_count,
                        tail_score,
                        protected_tail_mask,
                        client_class_counts,
                        args.num_users,
                        n_cls,
                        last_class_accuracy,
                        path_metrics,
                        args.tail_class_ratio,
                    )

                print("Epoch on server :", epoch)

            elif args.trainer == "ClipLora":
                print("use model == fedavg and trainer == ClipLoRa")
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = select_round_clients(args, epoch, client_schedule)
                print("idxs_users", idxs_users)
                print("------------local train start epoch:", epoch, "-------------")
                for idx in idxs_users:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                    local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                    local_weight = local_trainer.model.state_dict()
                    local_weights[idx] = copy.deepcopy(local_weight)
                print("------------local train finish epoch:", epoch, "-------------")
                # update global weights
                global_weights = average_weights(local_weights, idxs_users, datanumber_client)
                global_trainer.model.load_state_dict(global_weights)

                if not run_global_eval:
                    print_skip_global_eval(epoch, args.global_eval_interval)
                    print("Epoch on server :", epoch)
                    continue

                print("------------global test start-------------")
                result = global_trainer.global_test(is_global=True, current_epoch=epoch)


                global_test_acc_list.append(result[0])
                global_test_error_list.append(result[1])
                global_test_f1_list.append(result[2])
                global_epoch_list.append(epoch)
                global_time_list.append(time.time() - start)

                last_class_accuracy = result[3]
                if result[0] > best_acc:
                    best_acc = result[0]
                    best_class_accuracy = last_class_accuracy

                print("------------global test_acpfl finish-------------")
                print("global_test_acc_list:", global_test_acc_list)
                print("maximum test_acpfl acc:", max(global_test_acc_list))
                print("mean of acc:", np.mean(global_test_acc_list[-5:]))
                print("std of acc:", np.std(global_test_acc_list[-5:]))
                print(last_class_accuracy)
                print(len(last_class_accuracy))

                head_acc, medium_acc, tail_acc, overall_acc = calculate_accuracy_tail20_compat(
                    last_class_accuracy,
                    local_trainer,
                    args.tail_class_ratio,
                )
                head_acc_list.append(head_acc)
                mid_acc_list.append(medium_acc)
                tail_acc_list.append(tail_acc)
                append_round_metrics(
                    args.output_dir,
                    args,
                    epoch,
                    result,
                    head_acc,
                    tail_acc,
                    overall_acc,
                    last_class_accuracy,
                )
                if args.trainer == "FedTEF" and fedtef_enabled and args.save_tail_score:
                    path_metrics = evaluate_fedtef_logit_paths(global_trainer)
                    save_fedtef_analysis(
                        args.output_dir,
                        epoch,
                        exposure_count,
                        tail_score,
                        protected_tail_mask,
                        client_class_counts,
                        args.num_users,
                        n_cls,
                        last_class_accuracy,
                        path_metrics,
                        args.tail_class_ratio,
                    )

                print("Epoch on server :", epoch)

            elif args.trainer == "KgCoOp":
                print("use model == fedavg and trainer == PromptFL")
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = select_round_clients(args, epoch, client_schedule)
                print("idxs_users", idxs_users)
                print("------------local train start epoch:", epoch, "-------------")
                for idx in idxs_users:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                    local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                    local_weight = local_trainer.model.state_dict()
                    local_weights[idx] = copy.deepcopy(local_weight)
                print("------------local train finish epoch:", epoch, "-------------")
                # update global weights
                global_weights = average_weights(local_weights, idxs_users, datanumber_client)
                global_trainer.model.load_state_dict(global_weights)

                if not run_global_eval:
                    print_skip_global_eval(epoch, args.global_eval_interval)
                    print("Epoch on server :", epoch)
                    continue

                print("------------global test start-------------")
                result = global_trainer.global_test(is_global=True, current_epoch=epoch)

                prompt_state = {'epoch': epoch}
                prompt_state.update(collect_prompt_state(global_trainer.model.prompt_learner))
                torch.save(prompt_state, os.path.join(save_dir, f'prompt_params_epoch_{epoch}.pth'))

                global_test_acc_list.append(result[0])
                global_test_error_list.append(result[1])
                global_test_f1_list.append(result[2])
                global_epoch_list.append(epoch)
                global_time_list.append(time.time() - start)

                last_class_accuracy = result[3]
                if result[0] > best_acc:
                    best_acc = result[0]
                    best_class_accuracy = last_class_accuracy

                print("------------global test_acpfl finish-------------")
                print("global_test_acc_list:", global_test_acc_list)
                print("maximum test_acpfl acc:", max(global_test_acc_list))
                print("mean of acc:", np.mean(global_test_acc_list[-5:]))
                print("std of acc:", np.std(global_test_acc_list[-5:]))
                print(last_class_accuracy)
                print(len(last_class_accuracy))

                head_acc, medium_acc, tail_acc, overall_acc = calculate_accuracy_tail20_compat(
                    last_class_accuracy,
                    local_trainer,
                    args.tail_class_ratio,
                )
                head_acc_list.append(head_acc)
                mid_acc_list.append(medium_acc)
                tail_acc_list.append(tail_acc)
                append_round_metrics(
                    args.output_dir,
                    args,
                    epoch,
                    result,
                    head_acc,
                    tail_acc,
                    overall_acc,
                    last_class_accuracy,
                )
                if args.trainer == "FedTEF" and fedtef_enabled and args.save_tail_score:
                    path_metrics = evaluate_fedtef_logit_paths(global_trainer)
                    save_fedtef_analysis(
                        args.output_dir,
                        epoch,
                        exposure_count,
                        tail_score,
                        protected_tail_mask,
                        client_class_counts,
                        args.num_users,
                        n_cls,
                        last_class_accuracy,
                        path_metrics,
                        args.tail_class_ratio,
                    )

                print("Epoch on server :", epoch)

            elif args.trainer in ("FedClip", "FedTEF", "FedClipTailModule"):
                print(f"use model == fedavg and trainer == {args.trainer}")
                m = max(int(args.frac * args.num_users), 1)
                idxs_users = select_round_clients(args, epoch, client_schedule)
                print("idxs_users", idxs_users)
                print("------------local train start epoch:", epoch, "-------------")

                if args.trainer == "FedTEF" and fedtef_enabled:
                    if fedtef_v2_enabled:
                        if not fedtef_v10_enabled:
                            fedtef_v2_gate, tail_score, protected_tail_mask = exposure_tracker.compute_gate(
                                current_round=epoch
                            )
                        if fedtef_v10_enabled and hasattr(exposure_tracker, "reliability"):
                            fedtef_release_reliability = exposure_tracker.reliability.clone()
                        else:
                            fedtef_release_reliability = compute_fedtef_release_reliability(
                                exposure_tracker,
                                fedtef_v2_gate,
                                protected_tail_mask,
                                args,
                            )
                        apply_fedtef_tail_context(
                            global_trainer.model,
                            tail_score,
                            protected_tail_mask,
                            gate=fedtef_v2_gate,
                            release_reliability=fedtef_release_reliability,
                        )
                        log_fedtef_v2_gate(
                            epoch,
                            exposure_tracker,
                            fedtef_v2_gate,
                            tail_score,
                            protected_tail_mask,
                            local_trainer,
                            args.tail_class_ratio,
                            args.output_dir,
                        )
                    elif args.tef_use_exposure:
                        tail_score = compute_tail_score(
                            exposure_count,
                            epoch,
                            args.warmup_rounds_for_exposure,
                        )
                        protected_tail_mask = compute_fedtef_protected_mask(
                            tail_score,
                            strategy=args.tef_protected_strategy,
                            top_ratio=args.protected_tail_ratio,
                            current_round=epoch,
                            seed=args.seed,
                            global_class_counts=global_class_counts,
                            tail_class_ratio=args.tail_class_ratio,
                            round0_tie_break=args.fedtef_round0_tie_break,
                            dataset_name=args.dataset,
                        )
                        apply_fedtef_tail_context(global_trainer.model, tail_score, protected_tail_mask)
                        log_fedtef_exposure(
                            epoch,
                            exposure_count,
                            tail_score,
                            protected_tail_mask,
                            local_trainer,
                            args.tail_class_ratio,
                        )
                    else:
                        tail_score = torch.ones_like(exposure_count)
                        protected_tail_mask = compute_fedtef_protected_mask(
                            tail_score,
                            strategy=args.tef_protected_strategy,
                            top_ratio=args.protected_tail_ratio,
                            current_round=epoch,
                            seed=args.seed,
                            global_class_counts=global_class_counts,
                            tail_class_ratio=args.tail_class_ratio,
                            round0_tie_break=args.fedtef_round0_tie_break,
                            dataset_name=args.dataset,
                        )
                        apply_fedtef_tail_context(global_trainer.model, tail_score, protected_tail_mask)
                    global_weights = global_trainer.model.state_dict()

                local_weights = {}
                local_fedtef_v10_difficulty = {}
                local_fedtef_v10_difficulty_count = {}
                for idx in idxs_users:
                    local_trainer.model.load_state_dict(global_weights, strict=False)
                    if args.trainer == "FedTEF" and hasattr(local_trainer, "reset_optimizer_and_scheduler"):
                        local_trainer.reset_optimizer_and_scheduler()
                    lr_before_train = (
                        local_trainer.current_lr()
                        if hasattr(local_trainer, "current_lr")
                        else float(local_trainer.optim.param_groups[0]["lr"])
                    )
                    print(f"FedTEF client {idx} lr_before_train={lr_before_train:.6e}")
                    local_trainer.train(idx=idx, global_epoch=epoch, is_fed=True)
                    lr_after_train = (
                        local_trainer.current_lr()
                        if hasattr(local_trainer, "current_lr")
                        else float(local_trainer.optim.param_groups[0]["lr"])
                    )
                    local_weight = local_trainer.model.state_dict()
                    local_update_sq = 0.0
                    for key, value in local_weight.items():
                        if key in global_weights and torch.is_floating_point(value):
                            delta = value.detach().float().cpu() - global_weights[key].detach().float().cpu()
                            local_update_sq += float(delta.pow(2).sum().item())
                    local_update_norm = math.sqrt(local_update_sq)
                    print(
                        f"FedTEF client {idx} "
                        f"lr_after_train={lr_after_train:.6e} "
                        f"local_update_norm={local_update_norm:.6e}"
                    )
                    local_weights[idx] = copy.deepcopy(local_weight)
                    if fedtef_v10_enabled:
                        local_fedtef_v10_difficulty[int(idx)] = torch.as_tensor(
                            getattr(
                                local_trainer,
                                "fedtef_v10_difficulty",
                                torch.zeros(n_cls, dtype=torch.float32),
                            ),
                            dtype=torch.float32,
                        ).clone()
                        local_fedtef_v10_difficulty_count[int(idx)] = torch.as_tensor(
                            getattr(
                                local_trainer,
                                "fedtef_v10_difficulty_count",
                                torch.zeros(n_cls, dtype=torch.float32),
                            ),
                            dtype=torch.float32,
                        ).clone()

                print("------------local train finish epoch:", epoch, "-------------")

                if fedtef_v2_enabled:
                    gradient_prior_proxy = None
                    v10_positive_proxy = None
                    v10_survival_ratio = None
                    v10_difficulty_proxy = torch.zeros(n_cls, dtype=torch.float32)
                    if fedtef_v10_enabled:
                        (
                            v10_positive_proxy,
                            v10_client_observed,
                            v10_survival_ratio,
                        ) = compute_tail_stream_positive_update_stats(
                            global_weights,
                            local_weights,
                            idxs_users,
                            n_cls,
                            eps=args.fedtef_exposure_eps,
                        )
                        for client_idx in idxs_users:
                            v10_difficulty_proxy += local_fedtef_v10_difficulty.get(
                                int(client_idx),
                                torch.zeros(n_cls, dtype=torch.float32),
                            )
                        v10_difficulty_count = torch.zeros(n_cls, dtype=torch.float32)
                        for client_idx in idxs_users:
                            v10_difficulty_count += local_fedtef_v10_difficulty_count.get(
                                int(client_idx),
                                torch.zeros(n_cls, dtype=torch.float32),
                            )
                        observed_difficulty = v10_difficulty_count > 0
                        v10_difficulty_proxy = torch.where(
                            observed_difficulty,
                            v10_difficulty_proxy / torch.clamp(v10_difficulty_count, min=1.0),
                            torch.zeros_like(v10_difficulty_proxy),
                        )
                        print(
                            "FedTEF-v10 observer proxies "
                            f"E min/max/mean={v10_positive_proxy.min().item():.6f}/"
                            f"{v10_positive_proxy.max().item():.6f}/"
                            f"{v10_positive_proxy.mean().item():.6f}; "
                            f"D mean={v10_difficulty_proxy.mean().item():.6f}; "
                            f"D observed={int(observed_difficulty.sum().item())}/{n_cls}; "
                            f"S min/mean={v10_survival_ratio.min().item():.4f}/"
                            f"{v10_survival_ratio.mean().item():.4f}; "
                            f"observed rows={int((v10_client_observed > 0).sum().item())}/{n_cls}"
                        )
                    if getattr(exposure_tracker, "score_mode", "") in (
                        "gradient_prior",
                        "evidence_memory",
                        "low_exposure_router",
                    ):
                        gradient_prior_proxy, gradient_prior_client_observed = compute_tail_stream_gradient_prior_proxy(
                            global_weights,
                            local_weights,
                            idxs_users,
                            n_cls,
                            eps=args.fedtef_exposure_eps,
                        )
                        print(
                            "FedTEF positive-row evidence proxy "
                            f"min/max/mean: {gradient_prior_proxy.min().item():.6f}/"
                            f"{gradient_prior_proxy.max().item():.6f}/"
                            f"{gradient_prior_proxy.mean().item():.6f}; "
                            f"observed rows: {int((gradient_prior_client_observed > 0).sum().item())}/{n_cls}"
                        )
                    shared_keys = [
                        key for key in global_weights.keys()
                        if is_shared_stream_key(
                            key,
                            train_img_adap=args.fedtef_train_img_adap,
                            train_lora=args.fedtef_train_lora,
                        )
                    ]
                    append_fedtef_shared_stream_diagnostics(
                        args.output_dir,
                        args,
                        epoch,
                        global_weights,
                        local_weights,
                        idxs_users,
                        datanumber_client,
                        shared_keys,
                    )
                    tail_global_before = copy.deepcopy(global_weights)
                    global_weights = fedavg_keys(
                        global_weights,
                        local_weights,
                        idxs_users,
                        datanumber_client,
                        shared_keys,
                    )
                    if fedtef_v10_enabled and args.fedtef_tailagg_enabled:
                        global_weights, aggregated_tail_energy, tailagg_diagnostics = fedtef_v10_evidence_preserving_tailagg(
                            global_weights,
                            local_weights,
                            idxs_users,
                            fedtef_v2_gate,
                            n_cls,
                            survival_ratio=v10_survival_ratio,
                            evidence_threshold=args.fedtef_v10_evidence_threshold,
                            update_clip=args.fedtef_v10_agg_update_clip,
                            base_momentum=args.fedtef_v10_agg_base_momentum,
                            low_survival_momentum=args.fedtef_v10_agg_low_survival_momentum,
                            eps=args.fedtef_exposure_eps,
                            return_diagnostics=True,
                        )
                    elif args.fedtef_tailagg_enabled:
                        global_weights, aggregated_tail_energy, tailagg_diagnostics = fedtef_v2_tailagg(
                            global_weights,
                            local_weights,
                            idxs_users,
                            datanumber_client,
                            fedtef_v2_gate,
                            n_cls,
                            mode=args.fedtef_tailagg_mode,
                            fallback=args.fedtef_tailagg_fallback,
                            conflict_gamma=args.fedtef_tailagg_conflict_gamma,
                            min_agreement=args.fedtef_tailagg_min_agreement,
                            memory_momentum=args.fedtef_evidence_memory_momentum,
                            eps=args.fedtef_exposure_eps,
                            return_diagnostics=True,
                        )
                    else:
                        tail_keys = [key for key in global_weights.keys() if is_tail_stream_key(key)]
                        global_weights = fedavg_keys(
                            global_weights,
                            local_weights,
                            idxs_users,
                            datanumber_client,
                            tail_keys,
                        )
                        aggregated_tail_energy = torch.zeros(n_cls, dtype=torch.float32)
                        tailagg_diagnostics = build_fedavg_tail_diagnostics(
                            tail_global_before,
                            global_weights,
                            local_weights,
                            idxs_users,
                            n_cls,
                            eps=args.fedtef_exposure_eps,
                        )
                    append_fedtef_tailagg_diagnostics(
                        args.output_dir,
                        args,
                        epoch,
                        tailagg_diagnostics,
                        global_class_counts,
                        args.tail_class_ratio,
                        eps=args.fedtef_exposure_eps,
                    )
                    print(f"FedTEF-v2 shared FedAvg keys: {shared_keys}")
                else:
                    fedclip_key_markers = ["img_adap"]
                    # 只聚合img_adap的参数
                    fedclip_key_markers = ["img_adap"]
                    if args.trainer == "FedClipTailModule":
                        fedclip_key_markers.append("tail_prompt_residual")

                    for key in global_weights.keys():
                        if any(marker in key for marker in fedclip_key_markers):
                            temp = torch.zeros_like(global_weights[key])
                            total_weight = sum([datanumber_client[idx] for idx in idxs_users])
                            for client_idx in idxs_users:
                                temp += (datanumber_client[client_idx] / total_weight) * local_weights[client_idx][key]
                            global_weights[key] = temp

                # 更新全局模型
                if (
                    args.trainer == "FedTEF"
                    and fedtef_enabled
                    and not fedtef_v2_enabled
                    and args.tef_aggregation in ("binary_support", "count_oracle", "fedavg")
                ):
                    global_weights = classwise_tail_expert_aggregation(
                        global_weights,
                        local_weights,
                        idxs_users,
                        client_class_counts,
                        aggregation=args.tef_aggregation,
                    )

                global_trainer.model.load_state_dict(global_weights)
                if args.trainer == "FedTEF" and fedtef_enabled:
                    if fedtef_v2_enabled:
                        if fedtef_v10_enabled:
                            exposure_count = exposure_tracker.update(
                                v10_positive_proxy
                                if v10_positive_proxy is not None
                                else aggregated_tail_energy,
                                difficulty_proxy=v10_difficulty_proxy,
                                survival_ratio=v10_survival_ratio,
                                gate=fedtef_v2_gate,
                            )
                        elif getattr(exposure_tracker, "score_mode", "") in ("gradient_prior", "low_exposure_router"):
                            proxy_for_prior = (
                                gradient_prior_proxy
                                if gradient_prior_proxy is not None
                                else aggregated_tail_energy
                            )
                            exposure_count = exposure_tracker.update_from_gradient_proxy(
                                proxy_for_prior,
                                gate=fedtef_v2_gate,
                            )
                        elif getattr(exposure_tracker, "score_mode", "") == "evidence_memory":
                            proxy_for_evidence = (
                                gradient_prior_proxy
                                if gradient_prior_proxy is not None
                                else aggregated_tail_energy
                            )
                            exposure_count = exposure_tracker.update_from_evidence(
                                proxy_for_evidence,
                                aggregated_energy=aggregated_tail_energy,
                                gate=fedtef_v2_gate,
                            )
                        else:
                            exposure_count = exposure_tracker.update_from_energy(
                                aggregated_tail_energy,
                                gate=fedtef_v2_gate,
                            )
                        if fedtef_v10_enabled and hasattr(exposure_tracker, "commit_gate"):
                            fedtef_v2_gate, tail_score, protected_tail_mask = exposure_tracker.commit_gate(
                                current_round=epoch + 1
                            )
                        else:
                            fedtef_v2_gate, tail_score, protected_tail_mask = exposure_tracker.compute_gate(
                                current_round=epoch + 1
                            )
                        if fedtef_v10_enabled and hasattr(exposure_tracker, "reliability"):
                            fedtef_release_reliability = exposure_tracker.reliability.clone()
                        else:
                            fedtef_release_reliability = compute_fedtef_release_reliability(
                                exposure_tracker,
                                fedtef_v2_gate,
                                protected_tail_mask,
                                args,
                                aggregated_tail_energy=aggregated_tail_energy,
                            )
                    else:
                        exposure_count = update_fedtef_exposure(exposure_count, client_class_counts, idxs_users)
                    if fedtef_v2_enabled:
                        pass
                    elif args.tef_use_exposure:
                        tail_score = compute_tail_score(
                            exposure_count,
                            epoch + 1,
                            args.warmup_rounds_for_exposure,
                        )
                        protected_tail_mask = compute_fedtef_protected_mask(
                            tail_score,
                            strategy=args.tef_protected_strategy,
                            top_ratio=args.protected_tail_ratio,
                            current_round=epoch + 1,
                            seed=args.seed,
                            global_class_counts=global_class_counts,
                            tail_class_ratio=args.tail_class_ratio,
                            round0_tie_break=args.fedtef_round0_tie_break,
                            dataset_name=args.dataset,
                        )
                    else:
                        tail_score = torch.ones_like(exposure_count)
                        protected_tail_mask = compute_fedtef_protected_mask(
                            tail_score,
                            strategy=args.tef_protected_strategy,
                            top_ratio=args.protected_tail_ratio,
                            current_round=epoch + 1,
                            seed=args.seed,
                            global_class_counts=global_class_counts,
                            tail_class_ratio=args.tail_class_ratio,
                            round0_tie_break=args.fedtef_round0_tie_break,
                            dataset_name=args.dataset,
                        )
                    apply_fedtef_tail_context(
                        global_trainer.model,
                        tail_score,
                        protected_tail_mask,
                        gate=fedtef_v2_gate if fedtef_v2_enabled else None,
                        release_reliability=(
                            fedtef_release_reliability if fedtef_v2_enabled else None
                        ),
                    )
                    apply_fedtef_tail_context(
                        local_trainer.model,
                        tail_score,
                        protected_tail_mask,
                        gate=fedtef_v2_gate if fedtef_v2_enabled else None,
                        release_reliability=(
                            fedtef_release_reliability if fedtef_v2_enabled else None
                        ),
                    )
                if (
                    args.trainer == "FedTEF"
                    and fedtef_enabled
                    and args.save_tail_score
                    and args.fedtef_save_server_state
                ):
                    save_fedtef_server_state(
                        args.output_dir,
                        epoch,
                        exposure_count,
                        tail_score,
                        protected_tail_mask,
                        args,
                        global_trainer.model.state_dict(),
                        tail_gate=fedtef_v2_gate if fedtef_v2_enabled else None,
                    )

                if not run_global_eval:
                    print_skip_global_eval(epoch, args.global_eval_interval)
                    print("Epoch on server :", epoch)
                    continue

                print("------------global test start-------------")
                result = global_trainer.global_test(is_global=True, current_epoch=epoch)

                global_test_acc_list.append(result[0])
                global_test_error_list.append(result[1])
                global_test_f1_list.append(result[2])
                global_epoch_list.append(epoch)
                global_time_list.append(time.time() - start)

                last_class_accuracy = result[3]
                if result[0] > best_acc:
                    best_acc = result[0]
                    best_class_accuracy = last_class_accuracy

                print("------------global test finish-------------")
                print("global_test_acc_list:", global_test_acc_list)
                print("maximum test acc:", max(global_test_acc_list))
                print("mean of acc:", np.mean(global_test_acc_list[-5:]))
                print("std of acc:", np.std(global_test_acc_list[-5:]))
                print(last_class_accuracy)
                print(len(last_class_accuracy))

                head_acc, medium_acc, tail_acc, overall_acc = calculate_accuracy_tail20_compat(
                    last_class_accuracy,
                    local_trainer,
                    args.tail_class_ratio,
                )
                head_acc_list.append(head_acc)
                mid_acc_list.append(medium_acc)
                tail_acc_list.append(tail_acc)
                append_round_metrics(
                    args.output_dir,
                    args,
                    epoch,
                    result,
                    head_acc,
                    tail_acc,
                    overall_acc,
                    last_class_accuracy,
                )
                if args.trainer == "FedTEF" and fedtef_enabled and args.save_tail_score:
                    path_metrics = evaluate_fedtef_logit_paths(global_trainer)
                    save_fedtef_analysis(
                        args.output_dir,
                        epoch,
                        exposure_count,
                        tail_score,
                        protected_tail_mask,
                        client_class_counts,
                        args.num_users,
                        n_cls,
                        last_class_accuracy,
                        path_metrics,
                        args.tail_class_ratio,
                    )

                print("Epoch on server :", epoch)



    print("------------Specific info-------------")

    print("maximum test_acpfl acc:", max(global_test_acc_list))
    print("mean of acc:", np.mean(global_test_acc_list[-5:]))
    print("std of acc:", np.std(global_test_acc_list[-5:]))

    print("Global Test Accuracy List:")
    print(global_test_acc_list)
    print("\nGlobal Test Error List:")
    print(global_test_error_list)
    print("\nGlobal Test F1 Score List:")
    print(global_test_f1_list)
    print("\nGlobal Epoch List:")
    print(global_epoch_list)
    print("\nGlobal Time List:")
    print(global_time_list)



    print("test_acpfl acc:", max(global_test_acc_list))
    print(best_class_accuracy)
    calculate_accuracy_tail20(best_class_accuracy, local_trainer, args.tail_class_ratio)


    print("\nFinish!")

    if args.trainer != 'CLIP':
        for idx in idxs_users:
            local_trainer.fed_after_train()
    if args.model != 'local':
        global_trainer.fed_after_train()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, default="cluster",help="model of aggregation, choose from:cluster,FedOTP(used with CAPT), fedavg, fedprox, local(The last three are used with PromptFL)")
    parser.add_argument("--trainer", type=str, default="CAPT",help="name of trainer, choose from: CAPT,CLIP, PromptFL, PromptFLGeneralOnly, FedClip, FedTEF, FedClipTailModule, MaPLe,CoOp")
    parser.add_argument('--round', type=int, default=10, help="number of communication round")
    parser.add_argument('--num_users', type=int, default=20, help="number of users: K")
    parser.add_argument('--frac', type=float, default=0.4, help='the fraction of clients: C')
    parser.add_argument('--client_schedule_file', type=str, default="", help='optional JSON file with fixed selected clients per round')
    parser.add_argument('--client_schedule_seed', type=int, default=1, help='seed used when creating a fixed client schedule file')
    parser.add_argument('--gamma', type=float, default=1, help='gamma of single_step')
    parser.add_argument('--train_batch_size', type=int, default=32, help="number of trainer batch size")
    parser.add_argument('--test_batch_size', type=int, default=100, help="number of test_acpfl batch size")
    parser.add_argument('--global_eval_interval', type=int, default=1, help='run global test every N communication rounds; always evaluates epoch 0 and the final round')
    parser.add_argument('--log_update_retention', type=str2bool, default=False, help='log class-wise local-to-global update retention diagnostics')
    parser.add_argument('--update_retention_interval', type=int, default=1, help='write update-retention diagnostics every N rounds')
    parser.add_argument('--update_retention_param_key', type=str, default='prompt_learner.class_aware_ctx', help='class-wise parameter key used for update-retention diagnostics')
    parser.add_argument('--experimentD_enable', type=str2bool, default=False, help='enable Experiment D counterfactual diagnostics without changing FedAvg training')
    parser.add_argument('--experimentD_rounds', type=str, default='', help='comma-separated 1-based communication rounds for Experiment D diagnostics, e.g. 5,10,20,30')
    parser.add_argument('--experimentD_include_normalized', type=str2bool, default=True, help='also evaluate the auxiliary support-normalized counterfactual')
    parser.add_argument('--experimentD_log_update_norm', type=str2bool, default=True, help='log per-client trainable-parameter update norms when Experiment D is enabled')
    parser.add_argument('--experimentD_require_full_participation', type=str2bool, default=True, help='require frac=1.0 and all clients selected for Experiment D diagnostics')
    parser.add_argument('--experimentD_verify_fedavg', type=str2bool, default=True, help='verify reconstructed full FedAvg state matches average_weights output')
    parser.add_argument('--experimentD_eval_mode', type=str, default='class_filtered', choices=['class_filtered', 'full'], help='evaluation loader mode for Experiment D counterfactual diagnostics')
    parser.add_argument('--experimentD_log_classwise_agg', type=str2bool, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--experimentD_interval', type=int, default=None, help=argparse.SUPPRESS)
    parser.add_argument('--experimentD_param_key', type=str, default=None, help=argparse.SUPPRESS)
    parser.add_argument("--seed", type=int, default=1, help="only positive value enables a fixed seed")
    add_expF_runtime_arguments(parser)
    parser.add_argument('--mu', type=float, default=0.5, help='The parameter for fedprox')

    # parameters of datasets
    # cifar10, cifar100
    parser.add_argument('--partition', type=str, default='noniid-labeldir')
    parser.add_argument('--beta', type=float, default=0.05,help='The parameter for the dirichlet distribution for data partitioning')
    parser.add_argument('--imb_type', default="exp", type=str, help='imbalance type')
    parser.add_argument('--head_client_ratio', type=float, default=0.9, help='client-longtail head-client ratio')
    parser.add_argument('--tail_client_ratio', type=float, default=0.1, help='client-longtail tail-client ratio')
    parser.add_argument('--head_class_ratio', type=float, default=0.8, help='client-longtail head-class ratio')
    parser.add_argument('--tail_class_ratio', type=float, default=0.2, help='client-longtail tail-class ratio')
    parser.add_argument(
        '--specialization_lambda',
        type=float,
        default=1.0,
        help='Client-LT lambda. Controls how strongly tail classes concentrate on tail clients.',
    )
    parser.add_argument(
        '--intra_group_alpha',
        type=float,
        default=0.1,
        help='Client-LT Dirichlet concentration inside both head-client and tail-client groups.',
    )
    parser.add_argument(
        '--head_leakage_scale',
        type=float,
        default=0.0,
        help='Client-LT scale for non-tail samples entering tail clients, relative to tail sample volume.',
    )
    parser.add_argument('--client_headtail_concentration', type=float, default=None, help='deprecated compatibility option')
    parser.add_argument('--logit_adjust', action='store_true', help='compatibility flag for experiment scripts')
    parser.add_argument('--logit_adjust_tau', type=float, default=1.0, help='compatibility value for experiment scripts')
    parser.add_argument('--imb_factor', default=0.01, type=float, help='imbalance factor，IF = 100, 50 and 10')

    # parameters of learnable prompts
    parser.add_argument('--n_ctx', type=int, default=4, help="number of text encoder of text prompts")
    parser.add_argument('--num_prompt', type=int, default=1, help="number of prompts")  # 2
    parser.add_argument('--avg_prompt', type=int, default=1, help="number of prompts to aggregate")
    parser.add_argument('--ctx_init', default=False, help="is using the ctx init, set True for CLIP")
    parser.add_argument('--csc', default="True", help="is using the ctx init, set True for CLIP")
    parser.add_argument('--fedclip_tail_cutoff', type=float, default=0.95, help='cumulative sample cutoff for FedClipTailModule tail classes')
    parser.add_argument('--method', type=str, default='fedtef', choices=['fedtef', 'fedclip'], help='FedTEF method switch')
    parser.add_argument('--use_fedtef', type=str2bool, default=None, help='enable FedTEF and switch trainer to FedTEF when true')
    parser.add_argument('--fedtef_version', type=str, default='v1', choices=['v1', 'v2', 'v3', 'v4', 'v5', 'v6', 'v7', 'v8', 'v9', 'v10'], help='FedTEF implementation version')
    parser.add_argument('--use_tail_expert', type=str2bool, default=True, help='enable FedTEF TailLogitExpert')
    parser.add_argument('--tail_expert_type', type=str, default='logit_residual', help='tail expert type')
    parser.add_argument('--tail_score_type', type=str, default='exposure', help='tail score source')
    parser.add_argument('--positive_gate', type=str2bool, default=True, help='mask TailLogitExpert gradients by positive classes')
    parser.add_argument('--classwise_tail_agg', type=str2bool, default=True, help='aggregate TailLogitExpert rows class-wise')
    parser.add_argument('--fusion_lambda', type=float, default=0.5, help='FedTEF logit fusion strength')
    parser.add_argument('--tail_agg_weight', type=str, default='log_count', choices=['log_count'], help='class-wise tail aggregation weight')
    parser.add_argument('--tail_expert_zero_init', type=str2bool, default=True, help='zero initialize TailLogitExpert')
    parser.add_argument('--save_tail_score', type=str2bool, default=True, help='save FedTEF exposure_count and tail_score')
    parser.add_argument('--fedtef_save_server_state', type=str2bool, default=True, help='save resumable FedTEF server checkpoint')
    parser.add_argument('--fedtef_save_epoch_state', type=str2bool, default=False, help='also keep per-epoch FedTEF server checkpoints')
    parser.add_argument('--warmup_rounds_for_exposure', type=int, default=1, help='rounds using all-ones tail_score before exposure is reliable')
    parser.add_argument('--gate_tail_update_by_protection', type=str2bool, default=True, help='also gate TailLogitExpert rows by protected tail classes')
    parser.add_argument('--tail_fusion_mode', type=str, default='soft', choices=['soft', 'masked'], help='FedTEF inference-time tail fusion mode')
    parser.add_argument('--tail_update_protect_mode', type=str, default='top_ratio', choices=['top_ratio', 'threshold', 'all'], help='FedTEF training-time protected class selection mode')
    parser.add_argument('--protected_tail_ratio', type=float, default=None, help='protected class ratio for top_ratio update protection; defaults to --tail_class_ratio')
    parser.add_argument('--tail_score_threshold', type=float, default=0.5, help='protected score threshold for threshold mode')
    parser.add_argument('--fedtef_debug', type=str2bool, default=False, help='print FedTEF prediction-change and logit-scale diagnostics')
    parser.add_argument('--fedtef_debug_interval', type=int, default=20, help='batch interval for FedTEF debug prints')
    parser.add_argument('--fedtef_log_diagnostics', type=str2bool, default=False, help='write structured shared-stream, TailAgg, and branch diagnostics')
    parser.add_argument('--tail_expert_lr_mult', type=float, default=10.0, help='TailLogitExpert learning-rate multiplier')
    parser.add_argument('--freeze_img_adap', type=str2bool, default=False, help='freeze FedClip image adapter for tail-only diagnostic training')
    parser.add_argument('--tail_expert_mode', type=str, default='cosine', choices=['linear', 'cosine'], help='TailLogitExpert classifier mode')
    parser.add_argument('--tail_init_logit_scale', type=float, default=10.0, help='initial TailLogitExpert cosine logit scale')
    parser.add_argument('--tail_learnable_scale', type=str2bool, default=True, help='make TailLogitExpert logit scale learnable')
    parser.add_argument('--tail_use_bias', type=str2bool, default=True, help='use TailLogitExpert class-wise bias')
    parser.add_argument('--tail_logit_scale_max', type=float, default=100.0, help='maximum TailLogitExpert exp(logit_scale)')
    parser.add_argument('--tef_lambda', type=float, default=None, help='FedTEF fusion strength alias for --fusion_lambda')
    parser.add_argument('--tef_rho', type=float, default=None, help='protected low-exposure class ratio alias for --protected_tail_ratio')
    parser.add_argument('--tef_tau_init', type=float, default=None, help='Tail expert cosine scale init alias for --tail_init_logit_scale')
    parser.add_argument('--tef_use_bias', type=str2bool, default=None, help='use bias in tail expert alias for --tail_use_bias')
    parser.add_argument('--tef_positive_protected', type=str2bool, default=None, help='enable positive-protected tail row updates')
    parser.add_argument('--tef_support_aggregation', type=str2bool, default=True, help='use support-normalized class-wise aggregation')
    parser.add_argument('--tef_tail_type', type=str, default=None, choices=['linear', 'cosine'], help='tail expert type alias for --tail_expert_mode')
    parser.add_argument('--tef_use_exposure', type=str2bool, default=True, help='use exposure score in FedTEF fusion/protection')
    parser.add_argument('--tef_aggregation', type=str, default='binary_support', choices=['binary_support', 'fedavg', 'count_oracle'], help='tail expert aggregation rule')
    parser.add_argument('--tef_protected_strategy', type=str, default='exposure', choices=['exposure', 'random', 'oracle_tail', 'all'], help='protected class selection strategy')
    parser.add_argument('--fedtef_train_prompt', type=str2bool, default=True, help='FedTEF-v2 trains PromptFL-style prompt learner')
    parser.add_argument('--fedtef_train_img_adap', type=str2bool, default=False, help='FedTEF-v2 trains image adapter')
    parser.add_argument('--fedtef_train_tail_stream', type=str2bool, default=True, help='FedTEF-v2 trains residual tail stream')
    parser.add_argument('--fedtef_tail_stream_mode', type=str, default='cosine_residual', choices=['cosine_residual'], help='FedTEF clean tail stream parameterization')
    parser.add_argument('--fedtef_decouple_tail_loss', type=str2bool, default=False, help='detach base logits for FedTEF-v2 fused/tail/KL objectives')
    parser.add_argument('--fedtef_fusion_mode', type=str, default='residual_add', choices=['residual_add'], help='FedTEF-v2 fusion mode')
    parser.add_argument('--fedtef_scale_calibration', type=str2bool, default=True, help='FedTEF-v2 calibrates residual scale')
    parser.add_argument('--fedtef_scale_clamp_max', type=float, default=5.0, help='FedTEF-v2 residual scale clamp')
    parser.add_argument('--fedtef_gate_mode', type=str, default='soft', choices=['soft', 'hard_topk'], help='FedTEF-v2 gate mode')
    parser.add_argument('--fedtef_gate_score_mode', type=str, default='exposure', choices=['exposure', 'tail_need', 'gradient_prior', 'evidence_memory', 'low_exposure_router', 'topology_observer'], help='FedTEF dynamic gate score source')
    parser.add_argument('--fedtef_gate_temperature', type=float, default=1.0, help='FedTEF-v2 soft gate temperature')
    parser.add_argument('--fedtef_gate_threshold', type=float, default=None, help='FedTEF-v2 optional gate threshold')
    parser.add_argument('--fedtef_tail_topk', type=int, default=20, help='FedTEF-v2 target gated tail classes')
    parser.add_argument('--fedtef_exposure_ema_rho', type=float, default=0.9, help='FedTEF-v2 exposure EMA rho')
    parser.add_argument('--fedtef_exposure_eps', type=float, default=1e-6, help='FedTEF-v2 exposure epsilon')
    parser.add_argument('--fedtef_init_tail_mode', type=str, default='normal_residual', choices=['normal_residual'], help='FedTEF clean cosine tail initialization')
    parser.add_argument('--fedtef_round0_tie_break', type=str, default='random', choices=['random', 'none'], help='FedTEF-v2 tied dynamic-score handling after warmup')
    parser.add_argument('--fedtef_warmup_mode', type=str, default='none', choices=['round_robin', 'all_low', 'none'], help='FedTEF clean warmup gate mode')
    parser.add_argument('--fedtef_warmup_rounds', type=int, default=5, help='FedTEF-v2 warmup rounds')
    parser.add_argument('--fedtef_loss_base_weight', type=float, default=1.0, help='FedTEF-v2 base CE weight')
    parser.add_argument('--fedtef_loss_protected_base_weight', type=float, default=0.0, help='FedTEF-v8 acquisition CE weight on protected-label base logits')
    parser.add_argument('--fedtef_loss_protected_base_margin_weight', type=float, default=0.0, help='FedTEF-v8 acquisition margin loss weight on protected-label base logits')
    parser.add_argument('--fedtef_protected_base_margin', type=float, default=0.5, help='FedTEF-v8 target base-logit margin for protected labels')
    parser.add_argument('--fedtef_acquisition_low_exposure_weight', type=float, default=0.0, help='FedTEF-v9 sample-weight boost for protected/low-exposure labels in base acquisition CE')
    parser.add_argument('--fedtef_acquisition_signal_source', type=str, default='gate', choices=['gate', 'class_release_gate', 'tail_score'], help='FedTEF-v9 low-exposure signal used to weight shared acquisition')
    parser.add_argument('--fedtef_acquisition_signal_clamp_max', type=float, default=1.0, help='FedTEF-v9 clamp for low-exposure acquisition signal')
    parser.add_argument('--fedtef_acquisition_weight_normalize', type=str2bool, default=True, help='FedTEF-v9 normalize acquisition sample weights to keep base loss scale stable')
    parser.add_argument('--fedtef_tail_stream_detach_base', type=str2bool, default=False, help='FedTEF-v9 detach base features/logits/text features before tail-stream preservation')
    parser.add_argument('--fedtef_v10_exposure_budget', type=int, default=20, help='FedTEF-v10 protected budget for low exposure / high age classes')
    parser.add_argument('--fedtef_v10_survival_budget', type=int, default=10, help='FedTEF-v10 protected budget for low survival / high difficulty classes')
    parser.add_argument('--fedtef_v10_min_hold', type=int, default=5, help='FedTEF-v10 hysteresis minimum hold rounds')
    parser.add_argument('--fedtef_v10_replace_margin', type=float, default=1.2, help='FedTEF-v10 replacement margin for persistent routing')
    parser.add_argument('--fedtef_v10_difficulty_power', type=float, default=1.0, help='FedTEF-v10 difficulty exponent in protected score')
    parser.add_argument('--fedtef_v10_observer_w_exposure', type=float, default=1.0, help='FedTEF-v10 low-exposure score weight')
    parser.add_argument('--fedtef_v10_observer_w_age', type=float, default=0.5, help='FedTEF-v10 age/intermittency score weight')
    parser.add_argument('--fedtef_v10_observer_w_survival', type=float, default=1.0, help='FedTEF-v10 low-survival score weight')
    parser.add_argument('--fedtef_v10_difficulty_margin', type=float, default=1.0, help='FedTEF-v10 base margin target for difficulty proxy')
    parser.add_argument('--fedtef_v10_prior_base_weight', type=float, default=0.2, help='FedTEF-v10 mild exposure-balanced base CE weight')
    parser.add_argument('--fedtef_v10_prior_kappa', type=float, default=0.3, help='FedTEF-v10 mild exposure-balanced base CE strength')
    parser.add_argument('--fedtef_v10_prior_w_max', type=float, default=2.0, help='FedTEF-v10 max label weight for mild prior base CE')
    parser.add_argument('--fedtef_v10_hardneg_topm', type=int, default=5, help='FedTEF-v10 controlled hard negative count')
    parser.add_argument('--fedtef_v10_hardneg_lambda', type=float, default=0.5, help='FedTEF-v10 residual strength inside hard-negative loss')
    parser.add_argument('--fedtef_v10_release_floor', type=float, default=0.3, help='FedTEF-v10 class reliability floor for release')
    parser.add_argument('--fedtef_v10_sample_lambda_min', type=float, default=0.2, help='FedTEF-v10 minimum sample margin release strength')
    parser.add_argument('--fedtef_v10_sample_lambda_max', type=float, default=1.0, help='FedTEF-v10 maximum sample margin release strength')
    parser.add_argument('--fedtef_v10_sample_margin', type=float, default=1.0, help='FedTEF-v10 base margin midpoint for sample release')
    parser.add_argument('--fedtef_v10_sample_temperature', type=float, default=1.0, help='FedTEF-v10 sample release temperature')
    parser.add_argument('--fedtef_v10_safe_conf_threshold', type=float, default=0.7, help='FedTEF-v10 confidence threshold for safe KL')
    parser.add_argument('--fedtef_v10_safe_margin', type=float, default=1.0, help='FedTEF-v10 base margin threshold for safe KL')
    parser.add_argument('--fedtef_v10_evidence_threshold', type=float, default=1e-6, help='FedTEF-v10 minimum positive row evidence')
    parser.add_argument('--fedtef_v10_agg_update_clip', type=float, default=10.0, help='FedTEF-v10 clipped update-mass aggregation weight')
    parser.add_argument('--fedtef_v10_agg_base_momentum', type=float, default=0.6, help='FedTEF-v10 aggregation step for high-survival rows')
    parser.add_argument('--fedtef_v10_agg_low_survival_momentum', type=float, default=0.25, help='FedTEF-v10 aggregation step for low-survival rows')
    parser.add_argument('--fedtef_oracle_bottom20', type=str2bool, default=False, help='Ablation: protect bottom 20 percent classes instead of using the router')
    parser.add_argument('--fedtef_oracle_bottomk', type=int, default=0, help='Ablation: explicit bottom-K classes for oracle routing; 0 uses 20 percent')
    parser.add_argument('--fedtef_loss_fused_weight', type=float, default=1.0, help='FedTEF-v2 fused CE weight')
    parser.add_argument('--fedtef_loss_tail_weight', type=float, default=0.5, help='FedTEF-v2 tail CE weight')
    parser.add_argument('--fedtef_loss_keep_kl_weight', type=float, default=0.2, help='FedTEF-v2 non-tail KL keep weight')
    parser.add_argument('--fedtef_loss_reg_weight', type=float, default=1e-4, help='FedTEF-v2 gated residual regularization weight')
    parser.add_argument('--fedtef_loss_tail_only_weight', type=float, default=0.0, help='FedTEF-v2 protected-label tail-stream-only CE weight')
    parser.add_argument('--fedtef_loss_tail_margin_weight', type=float, default=0.0, help='FedTEF-v2 protected-label fused margin loss weight')
    parser.add_argument('--fedtef_tail_margin', type=float, default=1.0, help='FedTEF-v2 target margin for protected tail labels')
    parser.add_argument('--fedtef_tail_loss_normalize', type=str2bool, default=False, help='normalize protected tail losses by active protected samples instead of batch size')
    parser.add_argument('--fedtef_prior_balanced_base_weight', type=float, default=0.0, help='FedTEF-v3 gradient-prior balanced base CE weight')
    parser.add_argument('--fedtef_prior_balance_alpha', type=float, default=0.5, help='FedTEF-v3 prior-balanced CE class-weight exponent')
    parser.add_argument('--fedtef_prior_balance_clamp_min', type=float, default=0.25, help='FedTEF-v3 minimum prior-balanced CE label weight')
    parser.add_argument('--fedtef_prior_balance_clamp_max', type=float, default=3.0, help='FedTEF-v3 maximum prior-balanced CE label weight')
    parser.add_argument('--fedtef_kl_temperature', type=float, default=2.0, help='FedTEF-v2 KL temperature')
    parser.add_argument('--fedtef_tailagg_enabled', type=str2bool, default=True, help='FedTEF-v2 TailAgg enabled')
    parser.add_argument('--fedtef_tailagg_mode', type=str, default='row_update_norm', choices=['row_update_norm', 'conflict_aware', 'evidence_memory'], help='FedTEF TailAgg mode')
    parser.add_argument('--fedtef_tailagg_fallback', type=str, default='fedavg_or_keep', choices=['fedavg_or_keep', 'keep'], help='FedTEF-v2 TailAgg fallback')
    parser.add_argument('--fedtef_tailagg_conflict_gamma', type=float, default=1.0, help='FedTEF-v3 conflict-aware TailAgg agreement exponent')
    parser.add_argument('--fedtef_tailagg_min_agreement', type=float, default=-1.0, help='FedTEF-v3 keep/fallback if mean row-update agreement is below this value')
    parser.add_argument('--fedtef_tail_hidden_dim', type=int, default=512, help='FedTEF-v2 residual MLP hidden dim')
    parser.add_argument('--fedtef_img_adap_eta', type=float, default=1.0, help='FedTEF-v2 residual image adapter strength')
    parser.add_argument('--fedtef_tail_need_w_scarcity', type=float, default=0.3, help='FedTEF-v2 tail_need scarcity score weight')
    parser.add_argument('--fedtef_tail_need_w_residual', type=float, default=1.0, help='FedTEF-v2 tail_need residual update score weight')
    parser.add_argument('--fedtef_tail_need_w_forgetting', type=float, default=0.0, help='FedTEF-v2 tail_need forgetting score weight, reserved')
    parser.add_argument('--fedtef_tail_need_w_uncertainty', type=float, default=0.0, help='FedTEF-v2 tail_need uncertainty score weight, reserved')
    parser.add_argument('--fedtef_tail_need_beta', type=float, default=0.9, help='FedTEF-v2 tail_need EMA beta')
    parser.add_argument('--fedtef_gate_min_hold', type=int, default=8, help='FedTEF-v2 tail_need minimum protected-set hold rounds')
    parser.add_argument('--fedtef_gate_exit_ratio', type=float, default=0.7, help='FedTEF-v2 tail_need exit threshold ratio against top-budget mean')
    parser.add_argument('--fedtef_gate_budget', type=int, default=0, help='FedTEF-v2 tail_need gate budget; 0 uses tail_topk')
    parser.add_argument('--fedtef_gradient_prior_floor', type=float, default=1e-3, help='FedTEF-v2 gradient_prior minimum class-prior value before inversion')
    parser.add_argument('--fedtef_gradient_prior_score_power', type=float, default=0.5, help='FedTEF-v2 gradient_prior inverse-prior score power')
    parser.add_argument('--fedtef_gradient_prior_lock_rounds', type=int, default=0, help='FedTEF-v2 gradient_prior locks the first post-warmup protected set for this many rounds; 0 disables locking')
    parser.add_argument('--fedtef_gradient_prior_lock_mode', type=str, default='full_refresh', choices=['full_refresh', 'anchor_refine', 'anchor_until_end'], help='FedTEF-v2 gradient_prior lock refresh policy')
    parser.add_argument('--fedtef_gradient_prior_refine_ratio', type=float, default=0.2, help='FedTEF-v2 gradient_prior anchor_refine replacement ratio when max_swap is 0')
    parser.add_argument('--fedtef_gradient_prior_refine_max_swap', type=int, default=4, help='FedTEF-v2 gradient_prior anchor_refine maximum classes replaced at each refresh')
    parser.add_argument('--fedtef_gradient_prior_refine_margin', type=float, default=1.5, help='FedTEF-v2 gradient_prior anchor_refine requires candidate_score > protected_score * margin')
    parser.add_argument('--fedtef_gradient_prior_lock_gate_floor', type=float, default=0.0, help='FedTEF-v2 gradient_prior optional minimum soft gate value for locked protected classes')
    parser.add_argument('--fedtef_gradient_prior_update_all_rows', type=str2bool, default=False, help='FedTEF-v6 updates inverse-prior estimates from all class-wise positive row proxies instead of only protected rows')
    parser.add_argument('--fedtef_prior_logit_adjust', type=str2bool, default=False, help='add privacy-friendly gradient-prior logit calibration to protected classes')
    parser.add_argument('--fedtef_prior_logit_adjust_tau', type=float, default=0.0, help='strength of FedTEF-v2 prior logit calibration')
    parser.add_argument('--fedtef_prior_logit_adjust_clamp', type=float, default=3.0, help='absolute clamp for FedTEF-v2 prior logit calibration')
    parser.add_argument('--fedtef_sample_gate_enabled', type=str2bool, default=False, help='FedTEF-v3 enables sample-adaptive residual fusion')
    parser.add_argument('--fedtef_sample_gate_topm', type=int, default=20, help='FedTEF-v3 base/residual top-M candidate classes for sample-adaptive fusion')
    parser.add_argument('--fedtef_sample_gate_use_residual_topm', type=str2bool, default=True, help='FedTEF-v3 includes tail-residual top-M candidates in sample gate')
    parser.add_argument('--fedtef_sample_gate_uncertainty_power', type=float, default=0.5, help='FedTEF-v3 uncertainty exponent for sample gate')
    parser.add_argument('--fedtef_sample_gate_conf_threshold', type=float, default=0.7, help='FedTEF-v3 base-confidence threshold above which tail fusion is suppressed')
    parser.add_argument('--fedtef_sample_gate_min', type=float, default=0.0, help='FedTEF-v3 minimum sample-level gate strength')
    parser.add_argument('--fedtef_sample_gate_candidate_floor', type=float, default=0.25, help='FedTEF-v3 non-candidate class multiplier for sample gate')
    parser.add_argument('--fedtef_semantic_rescue_enabled', type=str2bool, default=False, help='FedTEF-v4 injects residuals only when semantic memory adds evidence')
    parser.add_argument('--fedtef_semantic_rescue_temperature', type=float, default=1.0, help='FedTEF-v4 semantic rescue sigmoid temperature')
    parser.add_argument('--fedtef_semantic_rescue_margin', type=float, default=0.0, help='FedTEF-v4 minimum semantic residual before strong injection')
    parser.add_argument('--fedtef_semantic_rescue_min', type=float, default=0.1, help='FedTEF-v4 minimum semantic rescue multiplier')
    parser.add_argument('--fedtef_positive_residual_only', type=str2bool, default=False, help='FedTEF-v6 only injects positive residual evidence into protected classes')
    parser.add_argument('--fedtef_residual_clamp', type=float, default=0.0, help='FedTEF-v4 absolute residual trust-region clamp; 0 disables')
    parser.add_argument('--fedtef_release_reliability_enabled', type=str2bool, default=False, help='FedTEF-v8 gates residual release by class-wise tail-evidence reliability')
    parser.add_argument('--fedtef_release_reliability_source', type=str, default='positive_proxy_ema', choices=['positive_proxy_ema', 'last_positive_proxy', 'aggregated_energy', 'observed_count', 'ones'], help='FedTEF-v8 source for class-wise residual release reliability')
    parser.add_argument('--fedtef_release_reliability_floor', type=float, default=0.0, help='FedTEF-v8 minimum reliability for protected classes')
    parser.add_argument('--fedtef_release_reliability_tau', type=float, default=1.0, help='FedTEF-v8 reliability saturation temperature after evidence normalization')
    parser.add_argument('--fedtef_release_reliability_power', type=float, default=1.0, help='FedTEF-v8 exponent applied to model-side reliability before fusion')
    parser.add_argument('--fedtef_train_routed_prompt', type=str2bool, default=False, help='FedTEF-v7 trains low-exposure routed class-wise prompt residuals')
    parser.add_argument('--fedtef_routed_prompt_lr_mult', type=float, default=1.0, help='FedTEF-v7 routed prompt learning-rate multiplier')
    parser.add_argument('--fedtef_routed_prompt_scale', type=float, default=1.0, help='FedTEF-v7 scale for routed prompt residual injection')
    parser.add_argument('--fedtef_routed_prompt_update_all_rows', type=str2bool, default=False, help='FedTEF-v7 updates routed prompt rows for all positive labels instead of only protected labels')
    parser.add_argument('--fedtef_evidence_memory_update_all_rows', type=str2bool, default=False, help='FedTEF-v4 learns every observed positive class row instead of only protected rows')
    parser.add_argument('--fedtef_evidence_memory_reliability_tau', type=float, default=2.0, help='FedTEF-v4 evidence observations required for reliable fusion')
    parser.add_argument('--fedtef_evidence_memory_gate_floor', type=float, default=0.05, help='Deprecated for fusion in sparse evidence-memory gate; retained for config compatibility')
    parser.add_argument('--fedtef_evidence_memory_residual_weight', type=float, default=0.25, help='FedTEF-v4 residual-energy contribution to protection need')
    parser.add_argument('--fedtef_evidence_memory_momentum', type=float, default=0.5, help='FedTEF-v4 server step size for persistent semantic memory')
    parser.add_argument('--fedtef_train_lora', type=str2bool, default=False, help='FedTEF trains a shared low-rank CLIP adaptation stream')
    parser.add_argument('--fedtef_lora_lr_mult', type=float, default=0.1, help='FedTEF shared LoRA learning-rate multiplier')
    parser.add_argument('--fedtef_lora_encoder', type=str, default='vision', choices=['text', 'vision', 'both'], help='FedTEF shared LoRA CLIP branch')
    parser.add_argument('--fedtef_lora_position', type=str, default='top3', choices=['top', 'top1', 'top2', 'top3', 'bottom', 'mid', 'up', 'half-up', 'half-bottom', 'all'], help='FedTEF shared LoRA transformer blocks')
    parser.add_argument('--fedtef_lora_rank', type=int, default=2, help='FedTEF shared LoRA rank')
    parser.add_argument('--fedtef_lora_alpha', type=int, default=1, help='FedTEF shared LoRA alpha')
    parser.add_argument('--fedtef_lora_dropout_rate', type=float, default=0.0, help='FedTEF shared LoRA dropout')
    parser.add_argument('--fedtef_lora_params', type=str, nargs='+', default=['q', 'v'], choices=['q', 'k', 'v', 'o'], help='FedTEF shared LoRA attention projections')


    # parameters of path
    parser.add_argument('--logdir', type=str, required=False, default="./logs/", help='Log directory path')
    parser.add_argument("--root", type=str, default="./DATA/", help="path to dataset")
    parser.add_argument("--imagenetroot", type=str, default="./DATA/", help="path to dataset")
    parser.add_argument("--output-dir", type=str, default="output/test/", help="output directory")
    parser.add_argument("--config-file", type=str, default="configs/trainers/CAPT/vit_b16.yaml",help="path to config file")
    parser.add_argument("--dataset-config-file", type=str, default="configs/datasets/cifar10_LT.yaml",help="path to config file for dataset setup")  #############
    parser.add_argument("--resume", type=str, default=None,help="checkpoint directory (from which the training resumes)")
    parser.add_argument("--transforms", type=str, nargs="+", help="data augmentation methods")
    parser.add_argument("--backbone", type=str, default="", help="name of CNN backbone")
    parser.add_argument("--head", type=str, default="", help="name of head")
    parser.add_argument("--eval-only", action="store_true", help="evaluation only")
    parser.add_argument("--model-dir", type=str, default="", help="load model from this directory for eval-only mode")
    parser.add_argument("--load-epoch", type=int, help="load model weights at this epoch for evaluation")
    parser.add_argument("--no-train", action="store_true", help="do not call trainer.train()")
    parser.add_argument("opts", default=None, nargs=argparse.REMAINDER,help="modify config options using the command-line")

    parser.add_argument('--lr', '--learning-rate', default=0.3, type=float, metavar='LR', help='initial learning rate',dest='lr')
    parser.add_argument('--schedule', default=[6, 10], nargs='*', type=int,help='learning rate schedule (when to drop lr by 10x)')
    parser.add_argument('--num_classes', type=int, default=100)
    parser.add_argument('--dataset', default="cifar100-LT")
    parser.add_argument('--visualize_interval', type=int, default=10, help="Interval for generating visualizations")
    parser.add_argument('--n_general', type=int, default=1, help="number of text encoder of text prompts")
    parser.add_argument('--n_disclusters', type=int, default=4, help="number of text encoder of text prompts")
    parser.add_argument('--n_simclusters', type=int, default=4, help="number of text encoder of text prompts")
    parser.add_argument('--prompt_depth', type=int, default=9)
    # LoRA arguments
    parser.add_argument('--encoder', type=str, choices=['text', 'vision', 'both'], default='both')

    args = parser.parse_args()
    if (
        args.experimentD_log_classwise_agg is not None
        or args.experimentD_interval is not None
        or args.experimentD_param_key is not None
    ):
        print(
            "WARNING: deprecated Experiment D arguments "
            "--experimentD_log_classwise_agg/--experimentD_interval/--experimentD_param_key "
            "are ignored. Use --experimentD_enable and --experimentD_rounds instead."
        )
    if args.use_fedtef is not None:
        args.use_tail_expert = args.use_fedtef
        if args.use_fedtef:
            args.trainer = "FedTEF"
            args.method = "fedtef"
        elif args.trainer == "FedTEF":
            args.trainer = "FedClip"
            args.method = "fedclip"
    if args.tef_lambda is not None:
        args.fusion_lambda = args.tef_lambda
    if args.tef_rho is not None:
        args.protected_tail_ratio = args.tef_rho
    if args.protected_tail_ratio is None:
        args.protected_tail_ratio = args.tail_class_ratio
    if args.tef_tau_init is not None:
        args.tail_init_logit_scale = args.tef_tau_init
    if args.tef_use_bias is not None:
        args.tail_use_bias = args.tef_use_bias
    if args.tef_tail_type is not None:
        args.tail_expert_mode = args.tef_tail_type
    if args.tef_positive_protected is not None:
        args.positive_gate = args.tef_positive_protected
        args.gate_tail_update_by_protection = args.tef_positive_protected
    if not args.tef_support_aggregation:
        args.tef_aggregation = "fedavg"
    args.classwise_tail_agg = args.tef_aggregation in ("binary_support", "count_oracle", "fedavg")
    main(args)








