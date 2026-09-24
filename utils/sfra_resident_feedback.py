"""Opt-in v2 feedback: bounded device cache and vectorized token reductions.

The source-identification path still computes every token/view's full gradient
independently. Only the correction and score-only paths use larger forwards.
"""
from collections import OrderedDict
import math
import time

import torch
from torch.nn import functional as F

from utils.cliplora_a_refresh import isolated_rng, train_only
from utils.sfra_fast_feedback import FastWitnessFeedback
from utils.sfra_math import require_finite


def execution_config_v2(forward_batch_size=64, cache_gib=4.):
    from utils.sfra_execution import FAST_EXECUTION_CONFIG
    if forward_batch_size < 8:
        raise ValueError('Feedback batch size must be at least 8 (one complete witness token)')
    if not math.isfinite(cache_gib) or not 0 <= cache_gib <= 4:
        raise ValueError('Feedback cache GiB must be finite and between 0 and 4')
    return dict(FAST_EXECUTION_CONFIG, version=2,
                feedback_forward_batch_size=forward_batch_size,
                frozen_visual_prefix_cache='bounded_cuda_with_cpu_fallback_original_dtype',
                device_cache_gib=float(cache_gib), device_cache_reserve_gib=4.,
                token_reduction='batched_matrix_sum_then_divide',
                source_finite_checks='at_evaluation_boundary_before_return',
                zero_gradient_skip='disabled_to_avoid_host_synchronization')


class ResidentWitnessFeedback(FastWitnessFeedback):
    defer_source_finite_checks = True

    def __init__(self, bank, forward_batch_size=64, cache_gib=4., reserve_gib=4.):
        execution_config_v2(forward_batch_size, cache_gib)
        super().__init__(bank, forward_batch_size)
        self.cache_limit = int(cache_gib * 1024**3)
        self.reserve_bytes = int(reserve_gib * 1024**3)
        self.device_cache_bytes = self.cpu_cache_bytes = 0
        self.plans = OrderedDict()

    def prepare(self):
        previous = self.prefix_stamp
        super().prepare()
        if previous != self.prefix_stamp:
            self.device_cache_bytes = self.cpu_cache_bytes = 0
            self.plans.clear()

    def token_prefix(self, q, view):
        key = (q, view)
        if key not in self.features:
            chunks = []
            for images in self.token_images(q).split(self.bank.batch_size):
                images = images.to(self.bank.device)
                if view:
                    images = images.flip(-1)
                chunks.append(self.prefix.forward_prefix(images.type(self.bank.core.dtype)))
            value = torch.cat(chunks, dim=0).detach()
            size = value.numel() * value.element_size()
            keep_device = (value.is_cuda and self.device_cache_bytes + size <= self.cache_limit)
            if keep_device:
                # This is a cold-cache check only. Leave room for suffix activations.
                free, _ = torch.cuda.mem_get_info(value.device)
                keep_device = free >= self.reserve_bytes
            if keep_device:
                self.device_cache_bytes += size
            else:
                value = value.cpu()
                self.cpu_cache_bytes += size
            self.features[key] = value
        return self.features[key]

    def plan(self, group):
        key = tuple(group)
        if key in self.plans:
            self.plans.move_to_end(key)
            return self.plans[key]
        b = self.bank
        sizes = [n for _, _, n, _ in group]
        # Dense segment sums are small (at most batch_size squared), avoid
        # per-token CUDA launches and preserve sum / token_size normalization.
        reduce = torch.zeros(len(group), sum(sizes))
        labels, offset = [], 0
        for row, (q, _, n, _) in enumerate(group):
            reduce[row, offset:offset+n] = 1.
            labels.extend([b.tokens[q]['class_id']] * n)
            offset += n
        to_long = lambda values: torch.tensor(values, dtype=torch.long, device=b.device)
        plan = dict(reduce=reduce.to(b.device), sizes=torch.tensor(sizes, device=b.device),
                    labels=to_long(labels), q=to_long([q for q, _, _, _ in group]),
                    view=to_long([v for _, v, _, _ in group]),
                    active=to_long([i for i, entry in enumerate(group) if entry[3]]))
        self.plans[key] = plan
        if len(self.plans) > 512:
            self.plans.popitem(last=False)
        return plan

    def evaluate(self, targets=None, all_scores=True, classification=False, classification_gradient=True):
        b = self.bank
        if torch.device(b.device).type == 'cuda':
            torch.cuda.synchronize(b.device)
        started = time.perf_counter()
        scores = torch.zeros(len(b.tokens), 2, device=b.device)
        client_gradients = torch.zeros(b.clients, b.numel, device=b.device) if targets is not None else None
        ce_gradient = classification and classification_gradient
        ce_scores = torch.zeros_like(scores) if classification else None
        ce_clients = torch.zeros(b.clients, device=b.device) if classification else None
        ce_client_gradients = torch.zeros(b.clients, b.numel, device=b.device) if ce_gradient else None
        loss = torch.zeros((), device=b.device)
        finite = torch.ones((), dtype=torch.bool, device=b.device)
        zero_coefficients = torch.zeros((), dtype=torch.long, device=b.device)
        forward_images = backward_images = ce_backward_images = batches = 0
        modes = [(m, m.training) for m in b.model.modules()]
        flags = [(p, p.requires_grad) for p in b.model.parameters()]
        if targets is not None:
            # Decisions are host metadata; transfer once if a caller supplied CUDA flags.
            group_targets = dict(targets, active=targets['active'].detach().cpu())
            target, sigma, weights = (targets[k].to(b.device) for k in ('target', 'sigma', 'weights'))
        else:
            group_targets = None
        with isolated_rng():
            try:
                train_only(b.model, 'A')
                b.model.eval()
                for group in self.groups(group_targets, all_scores, classification):
                    client = b.tokens[group[0][0]]['client_id']
                    plan = self.plan(group)
                    # CPU fallback entries must be transferred before concatenating
                    # with resident entries; a cache may intentionally be mixed.
                    encoded = torch.cat([self.token_prefix(q, view).to(b.device)
                                         for q, view, _, _ in group], dim=0)
                    need_functional = plan['active'].numel() > 0
                    with torch.set_grad_enabled(need_functional or ce_gradient):
                        features = self.prefix.forward_suffix(encoded)
                        features = features / features.norm(dim=-1, keepdim=True)
                        cosine = features @ b.text_features.T
                        labels = plan['labels']
                        wrong = cosine.clone()
                        wrong.scatter_(1, labels[:, None], -torch.inf)
                        margins = cosine.gather(1, labels[:, None]).flatten() - wrong.max(1).values
                        means = (plan['reduce'] @ margins) / plan['sizes']
                        q, view = plan['q'], plan['view']
                        scores[q, view] = means.detach()
                        if need_functional:
                            pos = plan['active']
                            aq, av = q[pos], view[pos]
                            gap = (target[aq, av] - means[pos].detach()).clamp_min(0)
                            scale = sigma[aq, av]
                            loss += (weights[aq] * .25 * (gap / scale).square()).sum()
                            coefficients = -.5 * weights[aq] * gap / scale.square()
                            finite &= torch.isfinite(coefficients).all()
                            zero_coefficients += (coefficients == 0).all().long()
                            # Even exact-zero batches stay on-device. No .item(),
                            # bool(CUDA tensor), or .cpu().tolist() in this loop.
                            parts = torch.autograd.grad((coefficients * means[pos]).sum(), b.parameters,
                                                        retain_graph=ce_gradient)
                            client_gradients[client] += torch.cat([p.detach().reshape(-1) for p in parts])
                            backward_images += len(encoded)
                        if classification:
                            logits = (b.core.logit_scale.exp().detach() * features) @ b.text_features.T
                            per_image = F.cross_entropy(logits + b.classification_logit_adjustment,
                                                        labels, reduction='none')
                            ce = (plan['reduce'] @ per_image) / plan['sizes']
                            ce_scores[q, view] = ce.detach()
                            weighted_ce = (.5 * b.classification_token_weights[q] * ce).sum()
                            ce_clients[client] += weighted_ce.detach()
                            if ce_gradient:
                                parts = torch.autograd.grad(weighted_ce, b.parameters)
                                ce_client_gradients[client] += torch.cat([p.detach().reshape(-1) for p in parts])
                                ce_backward_images += len(encoded)
                                backward_images += len(encoded)
                    forward_images += len(encoded)
                    batches += 1
                    if batches % 100 == 0:
                        print(f'SFRA resident functional batches: {batches}', flush=True)
            finally:
                for parameter, flag in flags:
                    parameter.requires_grad_(flag)
                for module, mode in modes:
                    module.training = mode
        if not finite:
            raise FloatingPointError('SFRA nonfinite functional_coefficients')
        require_finite(scores=scores, functional_loss=loss)
        if client_gradients is not None:
            require_finite(client_functional_gradients=client_gradients)
        result = dict(scores=scores.cpu(), gradient_norms=torch.zeros_like(scores).cpu(), coordinates=None,
                      gradient=None if client_gradients is None else client_gradients.sum(0), loss=float(loss),
                      forward_images=forward_images, backward_images=backward_images,
                      execution_forward_batches=batches, execution_zero_gradient_batches=0,
                      execution_zero_coefficient_batches=int(zero_coefficients),
                      execution_device_cache_bytes=self.device_cache_bytes,
                      execution_cpu_prefix_cache_bytes=self.cpu_cache_bytes)
        if classification:
            ce_loss = (ce_clients * b.classification_client_weights).sum()
            global_gradient = ((ce_client_gradients * b.classification_client_weights[:, None]).sum(0)
                               if ce_gradient else None)
            require_finite(classification_scores=ce_scores, global_classification_loss=ce_loss)
            if ce_gradient:
                require_finite(global_classification_gradient=global_gradient)
            result.update(classification_loss=float(ce_loss), classification_gradient=global_gradient,
                          classification_client_losses=ce_clients.cpu(), classification_scores=ce_scores.cpu(),
                          classification_forward_images=forward_images,
                          classification_backward_images=ce_backward_images)
        if torch.device(b.device).type == 'cuda':
            torch.cuda.synchronize(b.device)
        result['seconds'] = time.perf_counter() - started
        return result
