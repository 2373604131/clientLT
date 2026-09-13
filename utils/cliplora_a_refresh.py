"""C0--C3 pilot: B-only FedAvg followed by an optional, separate refresh.

Gap statistics are training-set functional proxies. They stay in client-side
diagnostics; the aggregation function takes only deltas and sample weights.
"""

import copy
import csv
import hashlib
import json
import math
import random
import time
from collections import Counter
from contextlib import contextmanager
from pathlib import Path

import numpy as np
import torch
from torch.nn import functional as F

from utils.pfrf import capture_rng_state, restore_rng_state


@contextmanager
def isolated_rng(seed=None):
    state = capture_rng_state()
    if seed is not None:
        random.seed(seed)
        np.random.seed(seed % (2**32 - 1))
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    try:
        yield
    finally:
        restore_rng_state(state)


def append_rows(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    exists = path.exists()
    with path.open("a", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        if not exists:
            writer.writeheader()
        writer.writerows(rows)


def lora_keys(model, factor):
    return sorted(name for name, _ in model.named_parameters()
                  if name.endswith(f"_lora_{factor}"))


def state_hash(state, keys):
    digest = hashlib.sha256()
    for key in keys:
        digest.update(key.encode("utf-8"))
        digest.update(state[key].detach().cpu().contiguous().numpy().tobytes())
    return digest.hexdigest()


def train_only(model, factor):
    for name, parameter in model.named_parameters():
        parameter.requires_grad_(name.endswith(f"_lora_{factor}"))
        parameter.grad = None


def aggregate_refresh_deltas(middle, uploads, client_weights, keys):
    """No class IDs, gaps, or active-client renormalization enter aggregation."""
    result = copy.deepcopy(middle)
    for key in keys:
        update = torch.zeros_like(middle[key])
        for client_id, delta in uploads.items():
            update.add_(delta[key].to(update), alpha=client_weights[client_id])
        result[key] = middle[key] + update
    return result


def product_norm_squared(left, right):
    # Exact Frobenius norm without forming a 768-by-768 effective matrix.
    return float(((left.T @ left) * (right @ right.T).T).sum().clamp_min(0))


def update_diagnostics(before, after, a_keys, scaling, initial):
    a_norm2 = b_norm2 = effective2 = novel2 = drift2 = 0.0
    rotations, initial_rotations = [], []
    for a_key in a_keys:
        b_key = a_key[:-1] + "B"
        old_a, new_a = before[a_key].double().cpu(), after[a_key].double().cpu()
        old_b, new_b = before[b_key].double().cpu(), after[b_key].double().cpu()
        da, db = new_a - old_a, new_b - old_b
        a_norm2 += float(da.square().sum())
        b_norm2 += float(db.square().sum())
        # Exact change: B_new A_new - B_old A_old = B_old dA + dB A_new.
        left, right = torch.cat((old_b, db), dim=1), torch.cat((da, new_a), dim=0)
        effective2 += scaling**2 * product_norm_squared(left, right)
        q_old = torch.linalg.qr(old_a.T, mode="reduced").Q
        outside = right - (right @ q_old) @ q_old.T
        novel2 += scaling**2 * product_norm_squared(left, outside)
        q_new = torch.linalg.qr(new_a.T, mode="reduced").Q
        init_a = initial[a_key].double().cpu()
        q_init = torch.linalg.qr(init_a.T, mode="reduced").Q
        rank = old_a.shape[0]
        rotations.append(math.sqrt(max(0.0, 1 - float((q_old.T @ q_new).square().sum()) / rank)))
        initial_rotations.append(math.sqrt(max(0.0, 1 - float((q_init.T @ q_new).square().sum()) / rank)))
        drift2 += float((new_a - init_a).square().sum())
    return {
        "delta_a_norm": math.sqrt(a_norm2), "delta_b_norm": math.sqrt(b_norm2),
        "effective_delta_norm": math.sqrt(effective2),
        "new_input_direction_norm": math.sqrt(novel2),
        "new_direction_fraction": math.sqrt(novel2 / effective2) if effective2 else 0.0,
        "a_drift_from_initial": math.sqrt(drift2),
        "mean_a_subspace_change": float(np.mean(rotations)),
        "mean_a_subspace_from_initial": float(np.mean(initial_rotations)),
    }


def correlation(x, y):
    x, y = np.asarray(x, dtype=float), np.asarray(y, dtype=float)
    if x.std() == 0 or y.std() == 0:
        return None
    return float(np.corrcoef(x, y)[0, 1])


class PrivateGapProbe:
    """Current-client training data only; not an independent held-out set."""

    def __init__(self, trainer, cfg, client_id):
        from Dassl.dassl.data.data_manager import build_data_loader
        from Dassl.dassl.data.transforms import build_transform

        source = trainer.dm.dataset.federated_train_x[client_id]
        self.counts = Counter(int(item.label) for item in source)
        self.total = len(source)
        self.loader = build_data_loader(
            cfg, sampler_type="SequentialSampler", data_source=list(source),
            batch_size=cfg.DATALOADER.TEST.BATCH_SIZE,
            tfm=build_transform(cfg, is_train=False), is_train=False,
            class_names=trainer.dm.dataset.classnames, drop_last=False,
        )
        self.losses = {}
        self.probe_examples = 0

    @torch.no_grad()
    def observe(self, trainer, stage):
        was_training = trainer.model.training
        sums = Counter()
        started = time.perf_counter()
        with isolated_rng():
            trainer.model.eval()
            for batch in self.loader:
                images, labels = trainer.parse_batch_test(batch)
                losses = F.cross_entropy(trainer.model(images), labels, reduction="none")
                for label, loss in zip(labels.cpu().tolist(), losses.cpu().tolist()):
                    sums[label] += loss
            trainer.model.train(was_training)
        self.losses[stage] = {c: sums[c] / n for c, n in self.counts.items()}
        self.probe_examples += self.total
        return time.perf_counter() - started

    def weights(self, num_classes):
        gaps = {c: max(self.losses["middle"][c] - self.losses["teacher"][c], 0.0)
                for c in self.counts}
        mean_gap = sum(self.counts[c] / self.total * g for c, g in gaps.items())
        weights = torch.zeros(num_classes, dtype=torch.float32)
        for c, gap in gaps.items():
            weights[c] = gap / (mean_gap + 1e-12)
        return gaps, mean_gap, weights


class ARefreshRuntime:
    def __init__(self, trainer, cfg, args):
        self.args, self.cfg = args, cfg
        self.root = Path(args.output_dir)
        self.variant = args.a_refresh_variant
        self.rounds = list(range(args.a_refresh_interval, args.round, args.a_refresh_interval))
        self.a_keys, self.b_keys = lora_keys(trainer.model, "A"), lora_keys(trainer.model, "B")
        self.initial = {key: value.detach().cpu().clone()
                        for key, value in trainer.model.state_dict().items()
                        if key in self.a_keys + self.b_keys}
        self.scaling = args.cliplora_alpha / math.sqrt(args.cliplora_rank)
        self.num_classes = len(trainer.dm.dataset.classnames)
        self.clients = {}
        self.normal_steps = self.refresh_steps = 0
        self.normal_samples = self.refresh_samples = 0
        self.probe_samples = 0
        self.probe_seconds = 0.0
        self.config = {
            "variant": self.variant, "rank": args.cliplora_rank,
            "alpha": args.cliplora_alpha, "scaling": self.scaling,
            "refresh_rounds": self.rounds,
            "refresh_epochs": args.a_refresh_epochs,
            "refresh_lr": args.a_refresh_lr, "refresh_momentum": 0.9,
            "refresh_weight_decay": 0.0,
            "gap_definition": "positive(CE_shared_after_B_aggregation - CE_local_teacher)",
            "gap_role": "within_client_class_direction_not_client_server_weight_or_gap_amplitude",
            "probe_protocol": "training_set_functional_gap_proxy",
            "probe_excluded_from_normal_training": False,
            "probe_excluded_from_refresh_training": False,
            "probe_uses_global_test": False,
            "loss": "batch_mean(CE * gap[label] / (client_frequency_mean_gap + 1e-12))",
            "all_zero_gap": "zero_update_without_CE_fallback",
            "aggregation": "original_sample_weights_including_zero_updates",
            "normal_optimizer_steps_expected": args.round * args.local_epochs * sum(len(loader) for loader in trainer.fed_train_loader_x_dict.values()),
            "refresh_optimizer_steps_expected": (0 if self.variant == "c0" else len(self.rounds) * args.a_refresh_epochs * sum(len(loader) for loader in trainer.fed_train_loader_x_dict.values())),
            "privacy": "no_class_statistics_required_for_updates; not_a_formal_privacy_guarantee",
        }
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / "a_refresh_config.json").write_text(json.dumps(self.config, indent=2), encoding="utf-8")
        print("A-refresh pilot:", json.dumps(self.config), flush=True)

    def is_refresh(self, round_id):
        return self.variant != "c0" and round_id in self.rounds

    def observe_client(self, trainer, client_id, round_id, stage):
        client_id = int(client_id)
        if client_id not in self.clients:
            with isolated_rng():
                self.clients[client_id] = PrivateGapProbe(trainer, self.cfg, client_id)
        self.probe_seconds += self.clients[client_id].observe(trainer, stage)
        self.probe_samples += self.clients[client_id].total

    def record_normal(self, client_id, round_id, steps, scheduler_steps, samples):
        self.normal_steps += steps
        self.normal_samples += samples * self.args.local_epochs
        append_rows(self.root / "a_refresh_budget.csv", [{
            "round": round_id, "client_id": int(client_id), "phase": "normal_B",
            "optimizer_steps": steps, "scheduler_steps": scheduler_steps,
            "sample_presentations": samples * self.args.local_epochs,
            "trainable_factor": "B",
        }])

    def refresh(self, trainer, middle, selected, client_weights, round_id):
        from trainers.cliplora import cliplora_optimizer_step

        factor = "B" if self.variant == "c1" else "A"
        keys = self.b_keys if factor == "B" else self.a_keys
        uploads, client_rows = {}, []
        started = time.perf_counter()
        for client_id in map(int, selected):
            trainer.model.load_state_dict(middle, strict=True)
            start_a_hash = state_hash(trainer.model.state_dict(), self.a_keys)
            start_b_hash = state_hash(trainer.model.state_dict(), self.b_keys)
            self.observe_client(trainer, client_id, round_id, "middle")
            probe = self.clients[client_id]
            gaps, mean_gap, weights = probe.weights(self.num_classes)
            train_only(trainer.model, factor)
            optimizer = torch.optim.SGD(
                [p for p in trainer.model.parameters() if p.requires_grad],
                lr=self.args.a_refresh_lr, momentum=0.9, weight_decay=0.0,
            )
            steps = samples = 0
            loss_sum = 0.0
            client_start = time.perf_counter()
            # Identical refresh RNG streams across C1/C2/C3; normal RNG untouched.
            with isolated_rng(self.args.seed + 1000003 * round_id + 1009 * client_id):
                trainer.model.train()
                for _ in range(self.args.a_refresh_epochs):
                    for batch in trainer.fed_train_loader_x_dict[client_id]:
                        images, labels = trainer.parse_batch_train(batch)
                        sample_weights = weights.to(labels.device)[labels] if self.variant == "c3" else None
                        _, loss, _ = cliplora_optimizer_step(
                            trainer.model, optimizer, None, "fp32", images, labels,
                            loss_weight=sample_weights,
                        )
                        steps += 1
                        samples += labels.numel()
                        loss_sum += float(loss.detach()) * labels.numel()
            local_state = trainer.model.state_dict()
            uploads[client_id] = {key: (local_state[key] - middle[key]).detach().cpu().clone() for key in keys}
            metrics = update_diagnostics(middle, local_state, self.a_keys, self.scaling, self.initial)
            row = {
                "round": round_id, "client_id": client_id, "variant": self.variant,
                "probe_protocol": "training_set_functional_gap_proxy",
                "client_mean_gap": mean_gap, "nonzero_gap_classes": sum(g > 0 for g in gaps.values()),
                "num_local_classes": len(gaps), "maximum_sample_weight": float(weights.max()),
                "server_weight": client_weights[client_id], "optimizer_steps": steps,
                "trainable_factor": factor,
                "trainable_params": sum(p.numel() for p in trainer.model.parameters() if p.requires_grad),
                "start_a_sha256": start_a_hash, "start_b_sha256": start_b_hash,
                "sample_presentations": samples, "refresh_loss": loss_sum / samples,
                "refresh_seconds": time.perf_counter() - client_start, **metrics,
            }
            client_rows.append(row)
            # Private diagnostics only. No class statistics are in uploads.
            private = self.root / "private_diagnostics" / f"client_{client_id}"
            append_rows(private / "refresh.csv", [row])
            append_rows(private / "class_gap.csv", [{
                "round": round_id, "class_id": c, "num_samples": probe.counts[c],
                "ce_pre": probe.losses["pre"][c], "ce_teacher": probe.losses["teacher"][c],
                "ce_middle": probe.losses["middle"][c],
                "local_ce_gain": max(probe.losses["pre"][c] - probe.losses["teacher"][c], 0),
                "gap": gaps[c], "sample_weight": float(weights[c]),
            } for c in sorted(gaps)])
            append_rows(self.root / "a_refresh_budget.csv", [{
                "round": round_id, "client_id": client_id, "phase": "refresh",
                "optimizer_steps": steps, "scheduler_steps": 0,
                "sample_presentations": samples, "trainable_factor": factor,
            }])
            self.refresh_steps += steps
            self.refresh_samples += samples
            print(f"A-refresh round={round_id} client={client_id} factor={factor} steps={steps}", flush=True)

        result = aggregate_refresh_deltas(middle, uploads, client_weights, keys)
        means = [r["client_mean_gap"] for r in client_rows]
        a_norms = [r["delta_a_norm"] for r in client_rows]
        append_rows(self.root / "a_refresh_summary.csv", [{
            "round": round_id, "variant": self.variant, "num_clients": len(client_rows),
            "nonzero_gap_clients": sum(g > 0 for g in means),
            "mean_client_mean_gap": float(np.mean(means)),
            "max_client_mean_gap": max(means),
            "weighted_mean_client_gap": sum(client_weights[r["client_id"]] * r["client_mean_gap"] for r in client_rows),
            "gap_vs_delta_a_pearson": correlation(means, a_norms),
            "gap_vs_effective_delta_pearson": correlation(means, [r["effective_delta_norm"] for r in client_rows]),
            "mean_client_delta_a_norm": float(np.mean(a_norms)),
            "mean_client_effective_delta_norm": float(np.mean([r["effective_delta_norm"] for r in client_rows])),
            "nonzero_effective_update_clients": sum(r["effective_delta_norm"] > 0 for r in client_rows),
            "refresh_optimizer_steps": sum(r["optimizer_steps"] for r in client_rows),
            "refresh_sample_presentations": sum(r["sample_presentations"] for r in client_rows),
            "refresh_wall_seconds_including_middle_probes": time.perf_counter() - started,
            "cumulative_probe_seconds": self.probe_seconds,
            "cumulative_probe_sample_presentations": self.probe_samples,
            "before_a_sha256": state_hash(middle, self.a_keys),
            "before_b_sha256": state_hash(middle, self.b_keys),
            "after_a_sha256": state_hash(result, self.a_keys),
            "after_b_sha256": state_hash(result, self.b_keys),
            **update_diagnostics(middle, result, self.a_keys, self.scaling, self.initial),
        }])
        train_only(trainer.model, "B")
        trainer.model.load_state_dict(result, strict=True)
        return result

    @torch.no_grad()
    def record_evaluation(self, trainer, state, round_id, stage, tail_ids):
        """Outcome-only diagnostics: never returned to the refresh decision."""
        correct = torch.zeros(self.num_classes, dtype=torch.float64)
        counts = torch.zeros_like(correct)
        was_training = trainer.model.training
        with isolated_rng():
            trainer.model.load_state_dict(state, strict=True)
            trainer.model.eval()
            for batch in trainer.test_loader:
                images, labels = trainer.parse_batch_test(batch)
                predicted = trainer.model(images).argmax(dim=1)
                y, hit = labels.cpu(), (predicted == labels).double().cpu()
                counts.scatter_add_(0, y, torch.ones_like(hit))
                correct.scatter_add_(0, y, hit)
            trainer.model.train(was_training)
        per_class = correct / counts * 100
        tail_ids = list(map(int, tail_ids))
        other = [c for c in range(self.num_classes) if c not in tail_ids]
        append_rows(self.root / "a_refresh_stage_metrics.csv", [{
            "round": round_id, "stage": stage,
            "overall_acc": float(correct.sum() / counts.sum() * 100),
            "non_tail_acc": float(per_class[other].mean()),
            "bottom20_tail_acc": float(per_class[tail_ids].mean()),
        }])
        append_rows(self.root / "a_refresh_stage_per_class.csv", [
            {"round": round_id, "stage": stage, "class_id": c, "accuracy": float(per_class[c])}
            for c in range(self.num_classes)
        ])

    def save_checkpoint(self, round_id, global_state):
        torch.save({
            "completed_round": round_id, "config": self.config,
            "global_lora_state": {k: global_state[k].detach().cpu() for k in self.a_keys + self.b_keys},
            "initial_lora_state": self.initial,
            "normal_steps": self.normal_steps, "refresh_steps": self.refresh_steps,
            "normal_samples": self.normal_samples, "refresh_samples": self.refresh_samples,
            "probe_samples": self.probe_samples,
            "probe_seconds": self.probe_seconds, "rng_state": capture_rng_state(),
        }, self.root / "a_refresh_last.pth.tar")
        (self.root / "a_refresh_progress.json").write_text(json.dumps({
            "completed_round": round_id,
            "completed_refresh_rounds": [r for r in self.rounds if r <= round_id] if self.variant != "c0" else [],
            "normal_optimizer_steps": self.normal_steps,
            "refresh_optimizer_steps": self.refresh_steps,
            "normal_sample_presentations": self.normal_samples,
            "refresh_sample_presentations": self.refresh_samples,
            "probe_sample_presentations": self.probe_samples,
        }, indent=2), encoding="utf-8")

    def restore(self, checkpoint_path, global_state):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        restored = copy.deepcopy(global_state)
        for key, value in payload["global_lora_state"].items():
            restored[key] = value.to(restored[key])
        self.initial = payload["initial_lora_state"]
        self.normal_steps, self.refresh_steps = payload["normal_steps"], payload["refresh_steps"]
        self.normal_samples, self.refresh_samples = payload["normal_samples"], payload["refresh_samples"]
        self.probe_seconds = payload["probe_seconds"]
        self.probe_samples = payload["probe_samples"]
        return restored, payload["completed_round"], payload["rng_state"]
