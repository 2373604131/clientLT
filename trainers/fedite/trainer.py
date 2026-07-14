import copy
from collections import defaultdict
from typing import Dict, Iterable, Sequence

import torch

from .losses import (
    cross_entropy_loss,
    hard_negative_ce_loss,
    margin_ranking_loss,
    protected_boundary_retention_loss,
    protected_candidate_kl_loss,
    protected_logit_retention_loss,
    router_loss,
    router_utility_targets,
    semantic_safety_loss,
    weighted_cross_entropy_loss,
)
from .utils import as_cpu_state_dict, class_counts_from_labels, group_update_norm, is_in_classes


class FedITEClientTrainer:
    """Local FedITE trainer with explicit shared/gate/tail optimization steps."""

    def __init__(self, model, args, device):
        self.model = model
        self.args = args
        self.device = torch.device(device)
        self.num_classes = model.num_classes

    def _make_optimizer(self, params, lr):
        params = [p for p in params if p.requires_grad]
        if not params:
            return None
        return torch.optim.AdamW(
            params,
            lr=float(lr),
            weight_decay=float(getattr(self.args, "fedite_weight_decay", 0.0)),
        )

    def _build_optimizers(self, tail_active=True):
        groups = self.model.get_trainable_parameter_groups()
        old_flags = {name: p.requires_grad for name, p in self.model.named_parameters()}

        self.model.set_trainable_groups(shared=True, gate=False, tail=False)
        shared_opt = self._make_optimizer(groups["shared"], getattr(self.args, "fedite_shared_lr", getattr(self.args, "lr", 1e-3)))

        gate_opt = None
        tail_opt = None
        if tail_active:
            self.model.set_trainable_groups(shared=False, gate=True, tail=False)
            gate_opt = self._make_optimizer(groups["gate"], getattr(self.args, "fedite_gate_lr", getattr(self.args, "lr", 1e-3)))

            self.model.set_trainable_groups(shared=False, gate=False, tail=True)
            tail_opt = self._make_optimizer(groups["tail"], getattr(self.args, "fedite_tail_lr", getattr(self.args, "lr", 1e-3)))

        for name, param in self.model.named_parameters():
            param.requires_grad_(old_flags[name])
        return shared_opt, gate_opt, tail_opt

    def _zero_all_grads(self):
        for param in self.model.parameters():
            param.grad = None

    def _state_subset(self, keys):
        state = self.model.state_dict()
        return {key: state[key].detach().cpu().clone() for key in keys if key in state}

    def _extract_label(self, item):
        if isinstance(item, dict):
            if "label" not in item:
                raise KeyError("FedITE sample dict is missing 'label'")
            return int(item["label"])
        if hasattr(item, "label"):
            return int(item.label)
        return int(item[1])

    def _count_dataset_labels_once(self, dataset):
        counts = torch.zeros(self.num_classes, dtype=torch.float32)
        if hasattr(dataset, "data_source") and hasattr(dataset, "indices"):
            for idx in dataset.indices:
                item = dataset.data_source[int(idx)]
                label = self._extract_label(item)
                if 0 <= label < self.num_classes:
                    counts[label] += 1.0
            return counts
        labels = None
        for attr in ("target", "targets", "labels"):
            if hasattr(dataset, attr):
                labels = getattr(dataset, attr)
                break
        if labels is not None:
            labels = torch.as_tensor(labels, dtype=torch.long)
            if labels.numel() > 0:
                counts += torch.bincount(labels.cpu(), minlength=self.num_classes).float()[: self.num_classes]
            return counts
        for _image, label in dataset:
            label = int(label)
            if 0 <= label < self.num_classes:
                counts[label] += 1.0
        return counts

    def train_one_client(
        self,
        global_state,
        train_loader,
        protected_classes,
        class_evidence_state,
        round_idx=0,
        tail_active=True,
    ):
        self.model.load_state_dict(global_state, strict=False)
        self.model.to(self.device)
        self.model.train()
        self.model.set_protected_classes(protected_classes)
        self.model.set_class_evidence_state(class_evidence_state)
        self.model.set_round_evidence_strength(1.0)
        shared_keys = self.model.get_shared_parameter_keys()
        gate_keys = self.model.get_gate_parameter_keys()
        tail_keys = self.model.get_tail_parameter_keys()
        trainable_keys = shared_keys + gate_keys + tail_keys
        initial_state = self._state_subset(trainable_keys)
        shared_opt, gate_opt, tail_opt = self._build_optimizers(tail_active=tail_active)

        # Class-client evidence topology is counted once per client round, not
        # once per local epoch. This keeps M/H/N_eff independent of local_ep.
        class_support_count = self._count_dataset_labels_once(train_loader.dataset)
        protected_class_support = torch.zeros(self.num_classes, dtype=torch.float32)
        write_sum = torch.zeros(self.num_classes, dtype=torch.float32)
        write_count = torch.zeros(self.num_classes, dtype=torch.float32)

        meters = defaultdict(float)
        steps = defaultdict(int)
        module_sums = defaultdict(lambda: defaultdict(float))
        module_counts = defaultdict(lambda: defaultdict(int))
        evidence_samples = int(len(train_loader.dataset))
        train_seen_samples = 0
        protected_positive_count = 0

        local_ep = int(getattr(self.args, "local_ep", 1))
        lambda_feature_anchor = float(getattr(self.args, "fedite_lambda_feature_anchor", 0.0))
        lambda_safe = float(getattr(self.args, "fedite_lambda_safe", 0.0))
        lambda_boundary = float(getattr(self.args, "fedite_lambda_boundary", 0.0))
        lambda_candidate_kl = float(getattr(self.args, "fedite_lambda_candidate_kl", 0.0))
        lambda_router = float(getattr(self.args, "fedite_lambda_router", 0.1))
        lambda_tail = float(getattr(self.args, "fedite_lambda_tail", 1.0))
        safe_type = str(getattr(self.args, "fedite_safe_type", "cosine")).lower()
        boundary_topk = int(getattr(self.args, "fedite_boundary_topk", 5))
        boundary_tolerance = float(getattr(self.args, "fedite_boundary_tolerance", 0.0))
        candidate_kl_topk = int(getattr(self.args, "fedite_candidate_kl_topk", boundary_topk))
        candidate_kl_temperature = float(getattr(self.args, "fedite_candidate_kl_temperature", 2.0))
        shared_survival_weight = float(getattr(self.args, "fedite_shared_survival_weight", 0.0))
        tail_survival_weight = float(getattr(self.args, "fedite_tail_survival_weight", 0.0))
        tail_hardneg_weight = float(getattr(self.args, "fedite_tail_hardneg_weight", 0.0))
        tail_hardneg_topm = int(getattr(self.args, "fedite_tail_hardneg_topm", 5))
        tail_margin_weight = float(getattr(self.args, "fedite_tail_margin_weight", 0.0))
        tail_margin = float(getattr(self.args, "fedite_tail_margin", 0.5))
        use_prefix_cache = bool(getattr(self.args, "fedite_prefix_cache", True))
        diagnostics_interval = max(int(getattr(self.args, "fedite_diagnostics_interval", 10)), 1)
        sample_diagnostics = (int(round_idx) % diagnostics_interval) == 0
        module_diagnostics_enabled = bool(getattr(self.args, "fedite_module_diagnostics", False)) and sample_diagnostics
        scalar_diagnostics_enabled = bool(getattr(self.args, "fedite_scalar_diagnostics", True)) and sample_diagnostics

        for _ in range(local_ep):
            for images, labels in train_loader:
                images = images.to(self.device)
                labels = labels.to(self.device).long()
                batch_size = int(labels.numel())
                if batch_size == 0:
                    continue
                train_seen_samples += batch_size
                prefix_tokens = self.model.encode_visual_prefix(images) if use_prefix_cache else None

                # Step A: shared adaptation, all samples.
                if shared_opt is not None:
                    has_protected_context = tail_active and len(protected_classes) > 0
                    use_feature_anchor = lambda_feature_anchor > 0
                    use_shared_retention = lambda_safe > 0 and has_protected_context
                    use_boundary_retention = lambda_boundary > 0 and has_protected_context
                    use_candidate_kl = lambda_candidate_kl > 0 and has_protected_context
                    self.model.set_trainable_groups(shared=True, gate=False, tail=False)
                    self._zero_all_grads()
                    out_shared = self.model.forward_shared(
                        images,
                        return_diagnostics=module_diagnostics_enabled,
                        prefix_tokens=prefix_tokens,
                        compute_base=use_feature_anchor or use_shared_retention or use_boundary_retention or use_candidate_kl,
                    )
                    shared_weight = None
                    if shared_survival_weight > 0:
                        state = class_evidence_state.to(self.device, dtype=torch.float32)
                        shared_weight = 1.0 + shared_survival_weight * state[labels, 0].clamp(0.0, 1.0)
                    loss_shared = weighted_cross_entropy_loss(out_shared["logits_shared"], labels, shared_weight)
                    loss_feature_anchor = images.new_tensor(0.0)
                    loss_safe = images.new_tensor(0.0)
                    loss_boundary = images.new_tensor(0.0)
                    loss_candidate_kl = images.new_tensor(0.0)
                    if use_feature_anchor:
                        loss_feature_anchor = semantic_safety_loss(
                            out_shared["shared_image_features"],
                            out_shared["base_image_features"],
                        )
                    if use_shared_retention:
                        if safe_type in {"protected_logits", "logits", "tail_logits"}:
                            loss_safe = protected_logit_retention_loss(
                                out_shared["logits_shared"],
                                out_shared["logits_base"],
                                protected_classes,
                            )
                        else:
                            loss_safe = semantic_safety_loss(
                                out_shared["shared_image_features"],
                                out_shared["base_image_features"],
                            )
                    if use_boundary_retention:
                        loss_boundary = protected_boundary_retention_loss(
                            out_shared["logits_shared"],
                            out_shared["logits_base"],
                            labels,
                            protected_classes,
                            topk=boundary_topk,
                            tolerance=boundary_tolerance,
                        )
                    if use_candidate_kl:
                        loss_candidate_kl = protected_candidate_kl_loss(
                            out_shared["logits_shared"],
                            out_shared["logits_base"],
                            labels,
                            protected_classes,
                            topk=candidate_kl_topk,
                            temperature=candidate_kl_temperature,
                        )
                    loss_shared_total = (
                        loss_shared
                        + lambda_feature_anchor * loss_feature_anchor
                        + lambda_safe * loss_safe
                        + lambda_boundary * loss_boundary
                        + lambda_candidate_kl * loss_candidate_kl
                    )
                    if torch.isfinite(loss_shared_total):
                        loss_shared_total.backward()
                        torch.nn.utils.clip_grad_norm_(shared_opt.param_groups[0]["params"], 5.0)
                        shared_opt.step()
                    else:
                        raise FloatingPointError("FedITE shared loss became NaN/Inf")
                    meters["loss_shared"] += float(loss_shared.detach().item()) * batch_size
                    meters["loss_feature_anchor"] += float(loss_feature_anchor.detach().item()) * batch_size
                    meters["loss_safe"] += float(loss_safe.detach().item()) * batch_size
                    meters["loss_boundary"] += float(loss_boundary.detach().item()) * batch_size
                    meters["loss_candidate_kl"] += float(loss_candidate_kl.detach().item()) * batch_size
                    if module_diagnostics_enabled:
                        shared_diag = out_shared.get("diagnostics", {})
                        for layer_pos, (input_norm, delta_norm) in enumerate(zip(
                            shared_diag.get("shared_input_norms", []),
                            shared_diag.get("shared_delta_norms", []),
                        )):
                            layer_id = int(self.model.visual_wrapper.adapter_layers[layer_pos])
                            module_sums[layer_id]["shared_input_norm"] += float(input_norm)
                            module_counts[layer_id]["shared_input_norm"] += 1
                            module_sums[layer_id]["shared_delta_norm"] += float(delta_norm)
                            module_counts[layer_id]["shared_delta_norm"] += 1
                            module_sums[layer_id]["shared_delta_ratio"] += float(delta_norm) / (float(input_norm) + 1e-12)
                            module_counts[layer_id]["shared_delta_ratio"] += 1
                    steps["shared_optimizer_steps"] += 1

                if not tail_active:
                    continue

                # Step B: gate routing utility, all samples, no tail adapter gradients.
                if gate_opt is not None:
                    self.model.set_trainable_groups(shared=False, gate=True, tail=False)
                    self._zero_all_grads()
                    out_router = self.model.forward_router_train(
                        images,
                        labels,
                        class_evidence_state,
                        prefix_tokens=prefix_tokens,
                    )
                    protected_mask = is_in_classes(labels, protected_classes)
                    utility = router_utility_targets(
                        labels,
                        protected_mask,
                        class_evidence_state.to(self.device),
                        out_router["logits_shared"],
                    )
                    loss_gate = lambda_router * router_loss(
                        out_router["tail_gates"],
                        utility,
                        positive_weight=3.0,
                    )
                    if torch.isfinite(loss_gate):
                        loss_gate.backward()
                        torch.nn.utils.clip_grad_norm_(gate_opt.param_groups[0]["params"], 5.0)
                        gate_opt.step()
                    else:
                        raise FloatingPointError("FedITE router loss became NaN/Inf")
                    meters["loss_router"] += float(loss_gate.detach().item()) * batch_size
                    if scalar_diagnostics_enabled or module_diagnostics_enabled:
                        gate_stack = torch.stack([g.detach().float() for g in out_router["tail_gates"]], dim=0).mean(dim=0)
                        meters["gate_mean"] += float(gate_stack.mean().item()) * batch_size
                        meters["utility_target_mean"] += float(utility.detach().float().mean().item()) * batch_size
                        num_protected = int(protected_mask.sum().item())
                        num_nonprotected = int((~protected_mask).sum().item())
                        if num_protected > 0:
                            meters["gate_protected_sum"] += float(gate_stack[protected_mask].sum().item())
                            meters["gate_protected_count"] += num_protected
                        if num_nonprotected > 0:
                            meters["gate_nonprotected_sum"] += float(gate_stack[~protected_mask].sum().item())
                            meters["gate_nonprotected_count"] += num_nonprotected
                        meters["gate_low_saturation_ratio"] += float((gate_stack < 0.05).float().mean().item()) * batch_size
                        meters["gate_high_saturation_ratio"] += float((gate_stack > 0.95).float().mean().item()) * batch_size
                        meters["gate_diagnostic_samples"] += batch_size
                        if module_diagnostics_enabled:
                            for layer_pos, gate in enumerate(out_router["tail_gates"]):
                                layer_id = int(self.model.visual_wrapper.adapter_layers[layer_pos])
                                gate = gate.detach().float()
                                module_sums[layer_id]["gate_mean"] += float(gate.mean().item())
                                module_counts[layer_id]["gate_mean"] += 1
                                module_sums[layer_id]["gate_low_saturation_ratio"] += float((gate < 0.05).float().mean().item())
                                module_counts[layer_id]["gate_low_saturation_ratio"] += 1
                                module_sums[layer_id]["gate_high_saturation_ratio"] += float((gate > 0.95).float().mean().item())
                                module_counts[layer_id]["gate_high_saturation_ratio"] += 1
                                if num_protected > 0:
                                    module_sums[layer_id]["gate_protected_mean"] += float(gate[protected_mask].sum().item())
                                    module_counts[layer_id]["gate_protected_mean"] += num_protected
                                if num_nonprotected > 0:
                                    module_sums[layer_id]["gate_nonprotected_mean"] += float(gate[~protected_mask].sum().item())
                                    module_counts[layer_id]["gate_nonprotected_mean"] += num_nonprotected
                    steps["gate_optimizer_steps"] += 1

                # Step C: positive-only tail evidence writing.
                protected_mask = is_in_classes(labels, protected_classes)
                idx = torch.where(protected_mask)[0]
                if idx.numel() == 0 or tail_opt is None:
                    steps["tail_skip_count"] += 1
                    continue

                images_p = images.index_select(0, idx)
                labels_p = labels.index_select(0, idx)
                prefix_p = self.model.select_visual_prefix(prefix_tokens, idx) if prefix_tokens is not None else None
                protected_positive_count += int(labels_p.numel())
                protected_class_support += class_counts_from_labels(labels_p.detach().cpu(), self.num_classes)

                self.model.set_trainable_groups(shared=False, gate=False, tail=True)
                self._zero_all_grads()
                out_tail = self.model.forward_tail_train(
                    images_p,
                    labels_p,
                    class_state=class_evidence_state.to(self.device),
                    return_diagnostics=module_diagnostics_enabled,
                    return_write_evidence=True,
                    prefix_tokens=prefix_p,
                )
                tail_weight = None
                if tail_survival_weight > 0:
                    state = class_evidence_state.to(self.device, dtype=torch.float32)
                    tail_weight = 1.0 + tail_survival_weight * state[labels_p, 0].clamp(0.0, 1.0)
                loss_tail_ce = weighted_cross_entropy_loss(out_tail["logits"], labels_p, tail_weight)
                loss_tail_hardneg = images_p.new_tensor(0.0)
                if tail_hardneg_weight > 0:
                    with torch.no_grad():
                        out_ref = self.model.forward_shared(
                            images_p,
                            return_diagnostics=False,
                            prefix_tokens=prefix_p,
                            compute_base=False,
                        )
                    loss_tail_hardneg = hard_negative_ce_loss(
                        out_tail["logits"],
                        labels_p,
                        out_ref["logits_shared"],
                        topm=tail_hardneg_topm,
                    )
                loss_tail_margin = images_p.new_tensor(0.0)
                if tail_margin_weight > 0:
                    loss_tail_margin = margin_ranking_loss(out_tail["logits"], labels_p, margin=tail_margin)
                loss_tail = lambda_tail * (
                    loss_tail_ce
                    + tail_hardneg_weight * loss_tail_hardneg
                    + tail_margin_weight * loss_tail_margin
                )
                if torch.isfinite(loss_tail):
                    loss_tail.backward()
                    torch.nn.utils.clip_grad_norm_(tail_opt.param_groups[0]["params"], 5.0)
                    tail_opt.step()
                else:
                    raise FloatingPointError("FedITE tail loss became NaN/Inf")

                write_layers = out_tail["diagnostics"].get("tail_write_norms", [])
                if write_layers:
                    sample_write = torch.stack([w.detach().float().cpu() for w in write_layers], dim=0).mean(dim=0)
                    for value, cls in zip(sample_write.tolist(), labels_p.detach().cpu().tolist()):
                        cls = int(cls)
                        write_sum[cls] += float(value)
                        write_count[cls] += 1.0
                    meters["tail_write_norm"] += float(sample_write.mean().item()) * int(labels_p.numel())
                    if module_diagnostics_enabled:
                        for layer_pos, write in enumerate(write_layers):
                            layer_id = int(self.model.visual_wrapper.adapter_layers[layer_pos])
                            module_sums[layer_id]["tail_write_norm"] += float(write.detach().float().mean().item())
                            module_counts[layer_id]["tail_write_norm"] += 1
                        for layer_pos, basis_diag in enumerate(out_tail["diagnostics"].get("basis_diagnostics", [])):
                            layer_id = int(self.model.visual_wrapper.adapter_layers[layer_pos])
                            module_sums[layer_id]["basis_entropy"] += float(basis_diag.get("basis_entropy", 0.0))
                            module_counts[layer_id]["basis_entropy"] += 1
                            module_sums[layer_id]["basis_max_share"] += float(basis_diag.get("basis_max_share", 0.0))
                            module_counts[layer_id]["basis_max_share"] += 1
                            module_sums[layer_id]["effective_basis_num"] += float(basis_diag.get("effective_basis_num", 0.0))
                            module_counts[layer_id]["effective_basis_num"] += 1
                            module_sums[layer_id]["tail_delta_norm"] += float(basis_diag.get("tail_delta_norm", 0.0))
                            module_counts[layer_id]["tail_delta_norm"] += 1

                meters["loss_tail"] += float(loss_tail.detach().item()) * int(labels_p.numel())
                meters["loss_tail_hardneg"] += float(loss_tail_hardneg.detach().item()) * int(labels_p.numel())
                meters["loss_tail_margin"] += float(loss_tail_margin.detach().item()) * int(labels_p.numel())
                steps["tail_optimizer_steps"] += 1

        self.model.set_trainable_groups(shared=True, gate=True, tail=True)
        return_keys = list(shared_keys)
        if tail_active:
            return_keys += gate_keys + tail_keys
        final_state = self._state_subset(trainable_keys)
        client_update = {key: final_state[key] for key in return_keys if key in final_state}
        support_clip = float(getattr(self.args, "fedite_support_clip_max", 20.0))
        write_clip = float(getattr(self.args, "fedite_write_clip_max", 20.0))
        clipped_support = class_support_count.clamp(0.0, support_clip)
        clipped_write_sum = write_sum.clamp(0.0, write_clip)
        stats = {
            "tail_active": bool(tail_active),
            "num_samples": int(evidence_samples),
            "class_support_mask": (clipped_support > 0).float(),
            "class_support_count_clipped": clipped_support,
            "M": clipped_support,
            "Q": (clipped_support > 0).float(),
            "H": clipped_support.pow(2),
            "write_sum": clipped_write_sum,
            "write_count": write_count,
            "protected_positive_count": int(protected_positive_count if tail_active else 0),
            "protected_class_support": protected_class_support,
            "tail_update_norm": group_update_norm(initial_state, final_state, tail_keys),
            "shared_update_norm": group_update_norm(initial_state, final_state, shared_keys),
            "gate_update_norm": group_update_norm(initial_state, final_state, gate_keys),
            "loss_avg": 0.0,
            "loss_shared": meters["loss_shared"] / max(train_seen_samples, 1),
            "loss_tail": meters["loss_tail"] / max(protected_positive_count, 1),
            "loss_tail_hardneg": meters["loss_tail_hardneg"] / max(protected_positive_count, 1),
            "loss_tail_margin": meters["loss_tail_margin"] / max(protected_positive_count, 1),
            "loss_router": meters["loss_router"] / max(train_seen_samples, 1),
            "loss_feature_anchor": meters["loss_feature_anchor"] / max(train_seen_samples, 1),
            "loss_safe": meters["loss_safe"] / max(train_seen_samples, 1),
            "loss_boundary": meters["loss_boundary"] / max(train_seen_samples, 1),
            "loss_candidate_kl": meters["loss_candidate_kl"] / max(train_seen_samples, 1),
            "gate_mean": meters["gate_mean"] / max(meters["gate_diagnostic_samples"], 1),
            "gate_protected_mean": meters["gate_protected_sum"] / max(meters["gate_protected_count"], 1),
            "gate_nonprotected_mean": meters["gate_nonprotected_sum"] / max(meters["gate_nonprotected_count"], 1),
            "gate_low_saturation_ratio": meters["gate_low_saturation_ratio"] / max(meters["gate_diagnostic_samples"], 1),
            "gate_high_saturation_ratio": meters["gate_high_saturation_ratio"] / max(meters["gate_diagnostic_samples"], 1),
            "utility_target_mean": meters["utility_target_mean"] / max(meters["gate_diagnostic_samples"], 1),
            "tail_write_norm": meters["tail_write_norm"] / max(protected_positive_count, 1),
            **steps,
        }
        module_diagnostics = []
        for layer_id in sorted(module_sums.keys()):
            row = {"layer_id": int(layer_id)}
            for key, value in module_sums[layer_id].items():
                denominator = max(module_counts[layer_id][key], 1)
                row[key] = float(value) / denominator
            module_diagnostics.append(row)
        stats["module_diagnostics"] = module_diagnostics
        if not tail_active:
            stats["protected_positive_count"] = 0
            stats["tail_update_norm"] = 0.0
            stats["gate_update_norm"] = 0.0
            stats["loss_tail"] = 0.0
            stats["loss_tail_hardneg"] = 0.0
            stats["loss_tail_margin"] = 0.0
            stats["loss_router"] = 0.0
            stats["gate_mean"] = 0.0
            stats["tail_write_norm"] = 0.0
        stats["loss_avg"] = (
            stats["loss_shared"]
            + stats["loss_router"]
            + stats["loss_tail"]
            + stats["loss_feature_anchor"]
            + stats["loss_safe"]
            + stats["loss_boundary"]
            + stats["loss_candidate_kl"]
        )
        return client_update, stats
