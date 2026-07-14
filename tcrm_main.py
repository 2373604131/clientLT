import argparse
import csv
import os
import random
import time
from pathlib import Path

import numpy as np
import torch

from trainers.tcrm.client_core import train_tcrm_client
from trainers.tcrm.diagnostics import ROUND_FIELDS, TAIL_FIELDS, append_csv, tail_diagnostic_rows, write_json
from trainers.tcrm.evaluation import evaluate_tcrm
from trainers.tcrm.feature_cache import build_or_load_feature_cache
from trainers.tcrm.prompt_learner import GeneralPromptLearner, zero_shot_text_features
from trainers.tcrm.server_core import (
    aggregate_prompt_states,
    compute_pre_reliability,
    merge_sufficient_stats,
    update_core_state,
)
from trainers.tcrm.state import init_core_state
from trainers.tcrm.topology import (
    client_class_count_matrix,
    compute_tail_topology,
    split_tail_non_tail,
    write_topology_report,
)


def str2bool(value):
    if isinstance(value, bool):
        return value
    value = str(value).lower()
    if value in {"true", "1", "yes", "y"}:
        return True
    if value in {"false", "0", "no", "n"}:
        return False
    raise argparse.ArgumentTypeError(f"Expected boolean value, got {value}")


def parse_int_list(value):
    if value is None or str(value).strip() == "":
        return []
    return [int(item.strip()) for item in str(value).replace(";", ",").split(",") if item.strip()]


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def load_clip_to_cpu(backbone_name):
    from clip import clip

    url = clip._MODELS[backbone_name]
    model_path = clip._download(url)
    try:
        model = torch.jit.load(model_path, map_location="cpu").eval()
        state_dict = None
    except RuntimeError:
        state_dict = torch.load(model_path, map_location="cpu")
    design_details = {
        "trainer": "TCRM",
        "vision_depth": 0,
        "language_depth": 0,
        "vision_ctx": 0,
        "language_ctx": 0,
    }
    model = clip.build_model(state_dict or model.state_dict(), design_details)
    for param in model.parameters():
        param.requires_grad_(False)
    return model.eval()


def apply_clip_precision(clip_model, precision):
    precision = str(precision).lower()
    if precision == "fp32":
        clip_model.float()
    elif precision == "fp16":
        clip_model.half()
    else:
        raise ValueError(f"Unsupported --clip_precision {precision}")
    return clip_model.eval()


def build_tcrm_data(args):
    from utils.datasplit import partition_data_LT

    if args.dataset != "cifar100_lt":
        raise ValueError("tcrm_main.py currently supports --dataset cifar100_lt")
    data_dir = os.path.join(args.data_root, "cifar-100")
    partition_name = "homo" if args.partition == "iid" else args.partition
    outputs = partition_data_LT(
        "cifar100_LT",
        data_dir,
        partition_name,
        args.num_users,
        imb_factor=args.imb_factor,
        imb_type=args.imb_type,
        beta=args.beta,
        logdir=None,
        head_client_ratio=args.head_client_ratio,
        tail_client_ratio=args.tail_client_ratio,
        head_class_ratio=args.head_class_ratio,
        tail_class_ratio=args.tail_class_ratio,
        specialization_lambda=args.specialization_lambda,
        intra_group_alpha=args.intra_group_alpha,
        head_leakage_scale=args.head_leakage_scale,
    )
    (
        data_train,
        data_test,
        _lab2cname,
        classnames,
        net_train,
        _net_test,
        train_counts,
        _test_counts,
        y_train,
    ) = outputs
    client_indices = [list(map(int, net_train[i])) for i in range(args.num_users)]
    train_class_counts = torch.bincount(torch.as_tensor(y_train, dtype=torch.long), minlength=len(classnames)).float()
    return data_train, data_test, classnames, client_indices, train_class_counts, y_train


def build_client_indices_from_allocation_csv(path, labels, num_classes, seed=1, allow_replacement=False):
    path = Path(path)
    labels = torch.as_tensor(labels, dtype=torch.long)
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise ValueError(f"controlled allocation CSV is empty: {path}")
    class_columns = [f"class_{class_id}" for class_id in range(int(num_classes))]
    missing = [name for name in class_columns if name not in rows[0]]
    if missing:
        raise ValueError(f"controlled allocation CSV missing columns: {missing[:5]}")
    rng = random.Random(int(seed))
    client_indices = [[] for _ in rows]
    for class_id, column in enumerate(class_columns):
        pool = torch.where(labels == int(class_id))[0].cpu().tolist()
        rng.shuffle(pool)
        requested = [int(float(row.get(column, 0) or 0)) for row in rows]
        total_requested = sum(max(v, 0) for v in requested)
        if total_requested > len(pool) and not bool(allow_replacement):
            raise ValueError(
                f"controlled allocation requests {total_requested} samples for class {class_id}, "
                f"but only {len(pool)} are available. Reduce --sweep_total_per_class or set "
                "--controlled_allocation_allow_replacement true."
            )
        cursor = 0
        for client_id, count in enumerate(requested):
            count = max(int(count), 0)
            if count == 0:
                continue
            if cursor + count <= len(pool):
                chunk = pool[cursor:cursor + count]
                cursor += count
            else:
                chunk = pool[cursor:]
                cursor = len(pool)
                while len(chunk) < count:
                    if not pool:
                        raise ValueError(f"class {class_id} has no samples available")
                    chunk.append(pool[rng.randrange(len(pool))])
            client_indices[client_id].extend(int(idx) for idx in chunk)
    for indices in client_indices:
        rng.shuffle(indices)
    return client_indices


def class_counts_from_client_indices(client_indices, labels, num_classes):
    labels = torch.as_tensor(labels, dtype=torch.long)
    counts = torch.zeros(int(num_classes), dtype=torch.float32)
    for indices in client_indices:
        if not indices:
            continue
        y = labels[torch.as_tensor(indices, dtype=torch.long)]
        counts += torch.bincount(y.cpu(), minlength=int(num_classes)).float()[:int(num_classes)]
    return counts


def build_arg_parser():
    parser = argparse.ArgumentParser(description="TCRM-Core standalone federated main")
    parser.add_argument("--method", type=str, default="tcrm_core", choices=["tcrm_core", "prompt_only", "decoupled_residual_fedavg"])
    parser.add_argument("--tcrm_variant", type=str, default="", choices=["", "prompt_only", "decoupled_residual_fedavg", "tcrm_core"])
    parser.add_argument("--dataset", type=str, default="cifar100_lt")
    parser.add_argument("--partition", type=str, default="client-longtail", choices=["client-longtail", "noniid-labeldir", "iid"])
    parser.add_argument("--data_root", type=str, default="DATA")
    parser.add_argument("--output_dir", type=str, default="output")
    parser.add_argument("--run_name", type=str, default="")
    parser.add_argument("--num_users", "--num-users", dest="num_users", type=int, default=20)
    parser.add_argument("--frac", type=float, default=0.4)
    parser.add_argument("--rounds", type=int, default=20)
    parser.add_argument("--local_bs", "--batch-size", dest="local_bs", type=int, default=32)
    parser.add_argument("--test_bs", type=int, default=512)
    parser.add_argument("--seed", type=int, default=1)
    parser.add_argument("--num_workers", type=int, default=0)

    parser.add_argument("--beta", type=float, default=0.05)
    parser.add_argument("--imb_factor", type=float, default=0.01)
    parser.add_argument("--imb_type", type=str, default="exp")
    parser.add_argument("--head_client_ratio", type=float, default=0.9)
    parser.add_argument("--tail_client_ratio", type=float, default=0.1)
    parser.add_argument("--head_class_ratio", type=float, default=0.8)
    parser.add_argument("--tail_class_ratio", type=float, default=0.2)
    parser.add_argument(
        "--specialization_lambda",
        type=float,
        default=1.0,
        help="Client-LT lambda. Controls how strongly tail classes concentrate on tail clients.",
    )
    parser.add_argument(
        "--intra_group_alpha",
        type=float,
        default=0.1,
        help="Client-LT Dirichlet concentration inside client groups.",
    )
    parser.add_argument(
        "--head_leakage_scale",
        type=float,
        default=0.0,
        help="Client-LT scale for non-tail samples entering tail clients.",
    )

    parser.add_argument("--clip_backbone", type=str, default="ViT-B/16")
    parser.add_argument("--clip_precision", type=str, default="fp32", choices=["fp32", "fp16"])
    parser.add_argument("--prompt_ctx", type=str, default="a photo of a")
    parser.add_argument("--prompt_n_ctx", type=int, default=4)
    parser.add_argument("--local_prompt_epochs", type=int, default=1)
    parser.add_argument("--prompt_lr", type=float, default=0.002)
    parser.add_argument("--prompt_weight_decay", type=float, default=0.0)
    parser.add_argument("--prompt_adam_eps", type=float, default=1e-8)
    parser.add_argument("--prompt_grad_clip", type=float, default=1.0)
    parser.add_argument("--logit_adjust_tau", type=float, default=1.0)
    parser.add_argument("--lambda_hbs", type=float, default=1.0)
    parser.add_argument("--epsilon_hbs", type=float, default=0.0)

    parser.add_argument("--use_feature_cache", type=str2bool, default=True)
    parser.add_argument("--feature_cache_dtype", type=str, default="float16", choices=["float16", "float32"])
    parser.add_argument("--feature_cache_dir", type=str, default="")
    parser.add_argument("--force_rebuild_cache", type=str2bool, default=False)

    parser.add_argument("--controlled_allocation_csv", type=str, default="")
    parser.add_argument("--controlled_allocation_seed", type=int, default=1)
    parser.add_argument("--controlled_allocation_allow_replacement", type=str2bool, default=False)
    parser.add_argument("--controlled_tail_class_ids", type=str, default="")

    parser.add_argument("--local_rho_steps", type=int, default=5)
    parser.add_argument("--local_rho_lr", type=float, default=0.05)
    parser.add_argument("--rho_grad_clip", type=float, default=1.0)
    parser.add_argument("--server_rho_lr", type=float, default=1.0)
    parser.add_argument("--rho_norm_bound", type=float, default=0.20)
    parser.add_argument("--update_norm_min", type=float, default=1e-6)
    parser.add_argument("--gain_margin_scale", type=float, default=0.10)
    parser.add_argument("--gamma_decay", type=float, default=0.15)
    parser.add_argument("--stale_horizon_ratio", type=float, default=0.25)
    parser.add_argument("--corroboration_scale_nu0", type=float, default=2.0)
    parser.add_argument("--tail_holdout_ratio", type=float, default=0.20)
    parser.add_argument("--tail_holdout_min", type=int, default=1)

    parser.add_argument("--disable_width", type=str2bool, default=False)
    parser.add_argument("--disable_write", type=str2bool, default=False)
    parser.add_argument("--disable_survival", type=str2bool, default=False)
    parser.add_argument("--disable_hbs", type=str2bool, default=False)

    parser.add_argument("--write_topology_report", type=str2bool, default=True)
    parser.add_argument("--write_tcrm_diagnostics", type=str2bool, default=True)
    parser.add_argument("--eval_interval", type=int, default=5)
    parser.add_argument("--checkpoint_interval", type=int, default=10)
    return parser


def _variant(args):
    return args.tcrm_variant or args.method


def main(args):
    from clip import clip

    set_seed(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    variant = _variant(args)
    args.tcrm_variant = variant
    run_name = args.run_name or f"tcrm_{variant}_{args.partition}_seed{args.seed}_{time.strftime('%Y%m%d_%H%M%S')}"
    output_dir = Path(args.output_dir) / "tcrm" / args.dataset / args.partition / run_name
    checkpoint_dir = output_dir / "checkpoints"
    checkpoint_dir.mkdir(parents=True, exist_ok=True)
    log_path = output_dir / "train.log"

    def log(message):
        print(message, flush=True)
        with log_path.open("a", encoding="utf-8") as handle:
            handle.write(str(message) + "\n")

    log("TCRM standalone entrypoint initialized")
    log("TCRM uses frozen CLIP feature cache and does not call federated_main.py or Dassl TrainerX")
    log(f"variant={variant}, disable_width={args.disable_width}, disable_write={args.disable_write}, disable_survival={args.disable_survival}, disable_hbs={args.disable_hbs}")
    log(f"clip_precision={args.clip_precision}, prompt_grad_clip={args.prompt_grad_clip}, rho_grad_clip={args.rho_grad_clip}")

    write_json(output_dir / "config.json", {**vars(args), "resolved_variant": variant})
    clip_model = apply_clip_precision(load_clip_to_cpu(args.clip_backbone), args.clip_precision)
    preprocess = clip._transform(clip_model.visual.input_resolution)
    data_train, data_test, classnames, client_indices, train_class_counts, y_train = build_tcrm_data(args)
    num_classes = len(classnames)
    original_train_class_counts = train_class_counts.clone()
    controlled_tail_class_ids = parse_int_list(args.controlled_tail_class_ids)
    if controlled_tail_class_ids:
        tail_class_ids = sorted(controlled_tail_class_ids)
        tail_set = set(tail_class_ids)
        non_tail_class_ids = [class_id for class_id in range(num_classes) if class_id not in tail_set]
        tail_index_of_class = {int(class_id): idx for idx, class_id in enumerate(tail_class_ids)}
    else:
        tail_class_ids, non_tail_class_ids, tail_index_of_class = split_tail_non_tail(original_train_class_counts, args.tail_class_ratio)
    log(f"tail classes: {tail_class_ids}")
    if args.controlled_allocation_csv:
        client_indices = build_client_indices_from_allocation_csv(
            args.controlled_allocation_csv,
            y_train,
            num_classes,
            seed=args.controlled_allocation_seed,
            allow_replacement=bool(args.controlled_allocation_allow_replacement),
        )
        if len(client_indices) != int(args.num_users):
            raise ValueError(
                f"controlled allocation has {len(client_indices)} clients, "
                f"but --num_users is {args.num_users}"
            )
        train_class_counts = class_counts_from_client_indices(client_indices, y_train, num_classes)
        log(
            "controlled allocation enabled: "
            f"{args.controlled_allocation_csv}, total_samples={int(train_class_counts.sum().item())}"
        )

    client_counts = client_class_count_matrix(client_indices, y_train, num_classes)
    topology_tensors, topology_rows = compute_tail_topology(client_counts, tail_class_ids, train_class_counts)
    if bool(args.write_topology_report):
        summary = write_topology_report(
            topology_rows,
            topology_tensors,
            output_dir / "tcrm_topology_bootstrap.csv",
            output_dir / "tcrm_topology_bootstrap.json",
        )
        log(f"TCRM topology bootstrap: M_mean={summary['M_mean']:.4f}, C_mean={summary['C_mean']:.4f}, D_mean={summary['D_mean']:.4f}")

    cache_dir = Path(args.feature_cache_dir) if args.feature_cache_dir else output_dir / "feature_cache"
    train_cache = cache_dir / f"{args.dataset}_{args.partition}_{args.clip_backbone.replace('/', '-')}_train.pt"
    test_cache = cache_dir / f"{args.dataset}_{args.partition}_{args.clip_backbone.replace('/', '-')}_test.pt"
    train_features, train_labels, _train_meta = build_or_load_feature_cache(
        data_train,
        preprocess,
        clip_model,
        train_cache,
        args.dataset,
        "train",
        args.clip_backbone,
        batch_size=args.test_bs,
        device=device,
        dtype=args.feature_cache_dtype,
        clip_precision=args.clip_precision,
        force_rebuild=bool(args.force_rebuild_cache) or not bool(args.use_feature_cache),
        log_fn=log,
    )
    test_features, test_labels, _test_meta = build_or_load_feature_cache(
        data_test,
        preprocess,
        clip_model,
        test_cache,
        args.dataset,
        "test",
        args.clip_backbone,
        batch_size=args.test_bs,
        device=device,
        dtype=args.feature_cache_dtype,
        clip_precision=args.clip_precision,
        force_rebuild=bool(args.force_rebuild_cache) or not bool(args.use_feature_cache),
        log_fn=log,
    )

    zero_text = zero_shot_text_features(clip_model, classnames, prompt_ctx=args.prompt_ctx, device=device)
    prompt_learner = GeneralPromptLearner(clip_model, classnames, prompt_ctx=args.prompt_ctx, n_ctx=args.prompt_n_ctx).to(device)
    prompt_state = prompt_learner.trainable_state()
    class_prior = train_class_counts / train_class_counts.sum().clamp_min(1.0)
    state = init_core_state(
        prompt_state,
        zero_text,
        topology_tensors,
        tail_class_ids,
        non_tail_class_ids,
        class_prior,
        total_rounds=args.rounds,
        rho_norm_bound=args.rho_norm_bound,
        stale_horizon_ratio=args.stale_horizon_ratio,
    )
    state.r_pre = compute_pre_reliability(state.M, state.D, state.age, state.m0, state.d0, state.stale_horizon)
    state.width_gate = torch.ones_like(state.r_pre) if bool(args.disable_width) else state.r_pre.clone()
    log(f"TCRM reliability constants: m0={state.m0:.4f}, d0={state.d0:.4f}, stale_horizon={state.stale_horizon:.4f}")
    write_json(output_dir / "config_resolved.json", {**vars(args), "resolved_variant": variant, "m0": state.m0, "d0": state.d0, "stale_horizon": state.stale_horizon})

    rng = random.Random(args.seed)
    logit_scale = float(clip_model.logit_scale.detach().float().exp().cpu().item())
    best_tail = -1.0
    best_round = -1

    for round_idx in range(int(args.rounds)):
        start = time.time()
        m = max(1, int(round(args.num_users * args.frac)))
        selected = sorted(rng.sample(list(range(args.num_users)), m))
        prompt_states = []
        prompt_weights = []
        client_stats = []
        prompt_grad_norms = []
        prompt_delta_norms = []
        rho_grad_means = []
        rho_grad_maxes = []
        for client_id in selected:
            indices = client_indices[client_id]
            if not indices:
                continue
            labels = train_labels[indices]
            features = train_features[indices]
            out = train_tcrm_client(
                prompt_learner,
                state,
                features,
                labels,
                args,
                logit_scale,
                device=device,
            )
            if int(out["prompt_weight"]) > 0:
                prompt_states.append(out["prompt_state"])
                prompt_weights.append(int(out["prompt_weight"]))
            client_stats.append(out["sufficient_stats"])
            prompt_grad_norms.append(float(out.get("prompt_grad_norm", 0.0)))
            prompt_delta_norms.append(float(out.get("prompt_delta_norm", 0.0)))
            rho_grad_means.append(float(out.get("rho_grad_norm_mean", 0.0)))
            rho_grad_maxes.append(float(out.get("rho_grad_norm_max", 0.0)))

        new_prompt_state, prompt_weight_total = aggregate_prompt_states(prompt_states, prompt_weights)
        if new_prompt_state is not None:
            state.prompt_state = new_prompt_state
            prompt_learner.load_trainable_state(state.prompt_state)
        sufficient = merge_sufficient_stats(client_stats, len(tail_class_ids), train_features.shape[1])
        state = update_core_state(
            state,
            sufficient,
            variant=variant,
            disable_width=bool(args.disable_width),
            disable_write=bool(args.disable_write),
            disable_survival=bool(args.disable_survival),
            server_rho_lr=args.server_rho_lr,
            gamma_decay=args.gamma_decay,
            corroboration_scale_nu0=args.corroboration_scale_nu0,
        )

        metrics = {}
        if round_idx % max(int(args.eval_interval), 1) == 0 or round_idx == int(args.rounds) - 1:
            metrics = evaluate_tcrm(test_features, test_labels, prompt_learner, state, logit_scale, batch_size=args.test_bs, device=device)
            if metrics["tail_acc"] > best_tail:
                best_tail = metrics["tail_acc"]
                best_round = round_idx
                torch.save({"round": round_idx, "state": state, "prompt_state": state.prompt_state, "args": vars(args)}, checkpoint_dir / "best.pt")

        hbs_loss_mean = float(sufficient["hbs_loss_sum"].item() / sufficient["hbs_loss_count"].clamp_min(1.0).item())
        rho_norm = state.rho.float().norm(dim=1) if state.rho.numel() else torch.zeros(1)
        row = {
            "round": round_idx,
            **{key: metrics.get(key, 0.0) for key in [
                "overall_acc", "macro_acc", "head_acc", "tail_acc",
                "zero_shot_overall_acc", "zero_shot_head_acc", "zero_shot_tail_acc",
                "hybrid_tail_acc", "tail_gain_over_zero_shot",
                "tail_to_head_error_rate", "tail_to_tail_error_rate",
                "mean_tail_vs_head_margin", "mean_tail_vs_tail_margin",
            ]},
            "hbs_loss_mean": hbs_loss_mean,
            "prompt_grad_norm": sum(prompt_grad_norms) / max(len(prompt_grad_norms), 1),
            "prompt_delta_norm": sum(prompt_delta_norms) / max(len(prompt_delta_norms), 1),
            "number_of_prompt_contributing_clients": len(prompt_states),
            "rho_grad_norm_mean": sum(rho_grad_means) / max(len(rho_grad_means), 1),
            "rho_grad_norm_max": max(rho_grad_maxes) if rho_grad_maxes else 0.0,
            "mean_rho_norm": float(rho_norm.mean().item()),
            "max_rho_norm": float(rho_norm.max().item()),
            "mean_r_pre": float(state.r_pre.mean().item()) if state.r_pre.numel() else 0.0,
            "mean_write_weight": float(state.last_write.mean().item()) if state.last_write.numel() else 0.0,
            "mean_direction_consistency": float(state.last_direction_consistency.mean().item()) if state.last_direction_consistency.numel() else 0.0,
            "mean_local_admission_gain": float(state.last_local_gain.mean().item()) if state.last_local_gain.numel() else 0.0,
            "mean_corroboration": float(state.last_corroboration.mean().item()) if state.last_corroboration.numel() else 0.0,
            "mean_candidate_skip_count": float(state.last_candidate_skip_count.mean().item()) if state.last_candidate_skip_count.numel() else 0.0,
            "mean_effective_age": float(state.age.mean().item()) if state.age.numel() else 0.0,
            "round_time": time.time() - start,
        }
        append_csv(output_dir / "tcrm_round_metrics.csv", row, ROUND_FIELDS)
        if bool(args.write_tcrm_diagnostics):
            for tail_row in tail_diagnostic_rows(round_idx, state, metrics):
                append_csv(output_dir / "tcrm_tail_diagnostics.csv", tail_row, TAIL_FIELDS)
        if round_idx % max(int(args.checkpoint_interval), 1) == 0:
            torch.save({"round": round_idx, "state": state, "prompt_state": state.prompt_state, "args": vars(args)}, checkpoint_dir / f"round_{round_idx:03d}.pt")
        log(
            f"Round {round_idx}: overall={row['overall_acc']:.4f} tail={row['tail_acc']:.4f} "
            f"mean_W={row['mean_write_weight']:.4f} mean_rho={row['mean_rho_norm']:.4f} prompt_clients={len(prompt_states)}"
        )

    torch.save({"round": int(args.rounds) - 1, "state": state, "prompt_state": state.prompt_state, "args": vars(args)}, checkpoint_dir / "last.pt")
    write_json(
        output_dir / "final_summary.json",
        {
            "best_round": best_round,
            "best_tail": best_tail,
            "last_round": int(args.rounds) - 1,
            "tail_class_ids": state.tail_class_ids,
            "m0": state.m0,
            "d0": state.d0,
            "stale_horizon": state.stale_horizon,
        },
    )
    log(f"TCRM run complete. Outputs written to: {output_dir}")


if __name__ == "__main__":
    main(build_arg_parser().parse_args())
