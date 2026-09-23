"""Client-local training witnesses; no test images, probe files, or Tail labels."""
from collections import defaultdict
import time

import numpy as np
import torch
from torch.nn import functional as F

from utils.cliplora_a_refresh import isolated_rng, train_only
from utils.sfra_math import require_finite


class WitnessBank:
    def __init__(self, trainer, cfg, a_keys, seed, batch_size=8, tokens=None):
        from Dassl.dassl.data.data_manager import DatasetCifar100
        from Dassl.dassl.data.transforms import build_transform
        self.trainer, self.keys, self.batch_size = trainer, a_keys, batch_size
        self.model = trainer.model
        self.core = self.model.module if hasattr(self.model, 'module') else self.model
        self.device = trainer.device
        self.parameters = [dict(self.model.named_parameters())[k] for k in self.keys]
        self.numel = sum(p.numel() for p in self.parameters)
        self.clients = len(trainer.dm.dataset.federated_train_x)
        self.tokens = [] if tokens is None else tokens
        self.datasets = []
        # Fixed local sampling does not advance either training or evaluation RNG.
        with isolated_rng():
            transform = build_transform(cfg, is_train=False)
            for client, source in enumerate(trainer.dm.dataset.federated_train_x):
                self.datasets.append(DatasetCifar100(cfg, source, transform, is_train=False))
                if tokens is None:
                    groups = defaultdict(list)
                    for position, item in enumerate(source):
                        groups[int(item.label)].append(position)
                    for label, positions in sorted(groups.items()):
                        rng = np.random.default_rng(np.random.SeedSequence([seed, client, label]))
                        chosen = sorted(map(int, rng.choice(positions, min(8, len(positions)), replace=False)))
                        self.tokens.append(dict(token_id=len(self.tokens), client_id=client,
                                                class_id=label, local_positions=chosen))
            # Text, prompt, and backbone weights are fixed in this vision-only pilot.
            # Cache the exact normalized text features used by CustomCLIP.forward.
            modes = [(m, m.training) for m in self.model.modules()]
            self.model.eval()
            with torch.no_grad():
                text = self.core.text_encoder(self.core.prompt_learner(), self.core.tokenized_prompts)
                self.text_features = (text / text.norm(dim=-1, keepdim=True)).detach().float()
            for module, mode in modes:
                module.training = mode

    def token_score(self, images, label, flip, gradient):
        value, grad, _, _ = self.token_feedback(images, label, flip, gradient)
        return value, grad

    def configure_classification(self, class_counts, logit_adjustment):
        """Local class frequencies and the original sample-weighted FL objective."""
        counts = torch.as_tensor(class_counts, dtype=torch.float32, device=self.device)
        sizes = counts.sum(1)
        self.classification_client_weights = sizes / sizes.sum()
        self.classification_token_weights = torch.stack([
            counts[t['client_id'], t['class_id']] / sizes[t['client_id']]
            for t in self.tokens])
        self.classification_logit_adjustment = logit_adjustment.detach().to(self.device).float()

    def token_feedback(self, images, label, flip, gradient,
                       classification=False, classification_gradient=False, encoded_prefix=None):
        value = torch.zeros((), device=self.device)
        grad = torch.zeros(self.numel, device=self.device) if gradient else None
        ce_value = torch.zeros((), device=self.device) if classification else None
        ce_grad = torch.zeros(self.numel, device=self.device) if classification_gradient else None
        for chunk_id, chunk in enumerate(images.split(self.batch_size)):
            if encoded_prefix is None:
                chunk = chunk.to(self.device)
                if flip:
                    chunk = chunk.flip(-1)
            with torch.set_grad_enabled(gradient or classification_gradient):
                if encoded_prefix is None:
                    features = self.core.image_encoder(chunk.type(self.core.dtype))
                else:
                    start = chunk_id*self.batch_size
                    features = self._fast_feedback.prefix.forward_suffix(
                        encoded_prefix[start:start+len(chunk)].to(self.device))
                features = features / features.norm(dim=-1, keepdim=True)
                scores = features @ self.text_features.T
                correct = scores[:, label]
                wrong = scores.clone()
                wrong[:, label] = -torch.inf
                # Recompute strongest wrong class among ALL task classes every time.
                margin = (correct - wrong.max(1).values).sum() / len(images)
                value += margin.detach()
                if classification:
                    # Match CustomCLIP.forward, including its logit scale and global LA.
                    logits = (self.core.logit_scale.exp().detach() * features) @ self.text_features.T
                    logits = logits + self.classification_logit_adjustment
                    labels = torch.full((len(chunk),), label, dtype=torch.long, device=self.device)
                    ce = F.cross_entropy(logits, labels, reduction='sum') / len(images)
                    ce_value += ce.detach()
                if gradient:
                    parts = torch.autograd.grad(margin, self.parameters, create_graph=False,
                                                retain_graph=classification_gradient)
                    grad += torch.cat([p.detach().reshape(-1) for p in parts])
                if classification_gradient:
                    parts = torch.autograd.grad(ce, self.parameters, create_graph=False)
                    ce_grad += torch.cat([p.detach().reshape(-1) for p in parts])
        require_finite(score=value)
        if gradient:
            require_finite(functional_gradient=grad)
        if classification:
            require_finite(classification_loss=ce_value)
        if classification_gradient:
            require_finite(classification_gradient=ce_grad)
        return value, grad, ce_value, ce_grad

    def evaluate(self, basis=None, targets=None, all_scores=True,
                 classification=False, classification_gradient=True):
        """Same model for all clients. Return a sum of already globally weighted gradients.

        Token gradients are reduced immediately: no Q x n_A tensor or retained
        multi-client graph. Microbatches use sum/|W_q|, not mean(batch_means).
        """
        fast = getattr(self, '_fast_feedback', None)
        if fast is not None:
            fast.prepare()
            if basis is None:
                return fast.evaluate(targets, all_scores, classification, classification_gradient)
        started = time.perf_counter()
        count = len(self.tokens)
        scores = torch.zeros(count, 2, device=self.device)
        norms = torch.zeros_like(scores)
        coordinates = torch.zeros(count, 2, basis.shape[1], device=self.device) if basis is not None else None
        client_gradients = torch.zeros(self.clients, self.numel, device=self.device) if targets is not None else None
        ce_gradient = classification and classification_gradient
        ce_scores = torch.zeros_like(scores) if classification else None
        ce_clients = torch.zeros(self.clients, device=self.device) if classification else None
        ce_client_gradients = torch.zeros(self.clients, self.numel, device=self.device) if ce_gradient else None
        loss = torch.zeros((), device=self.device)
        forward_images = backward_images = 0
        classification_forward_images = classification_backward_images = 0
        modes = [(m, m.training) for m in self.model.modules()]
        flags = [(p, p.requires_grad) for p in self.model.parameters()]
        if targets is not None:
            target = targets['target'].to(self.device)
            sigma = targets['sigma'].to(self.device)
            weights = targets['weights'].to(self.device)
        with isolated_rng():
            try:
                train_only(self.model, 'A')
                self.model.eval()
                for q, token in enumerate(self.tokens):
                    active = targets is not None and bool(targets['active'][q])
                    if not all_scores and not active and not classification:
                        continue
                    gradient = basis is not None or active
                    images = (fast.token_images(q) if fast is not None else
                              torch.stack([self.datasets[token['client_id']][j]['img']
                                           for j in token['local_positions']]))
                    for view in range(2):
                        prefix = fast.token_prefix(q, view) if fast is not None else None
                        value, grad, ce_value, ce_grad = self.token_feedback(
                            images, token['class_id'], view == 1, gradient, classification, ce_gradient,
                            encoded_prefix=prefix)
                        scores[q, view] = value
                        forward_images += len(images)
                        backward_images += len(images) if gradient else 0
                        if classification:
                            ce_scores[q, view] = ce_value
                            # Mean over both views; original local class frequency, not witness count.
                            ce_weight = .5 * self.classification_token_weights[q]
                            ce_clients[token['client_id']] += ce_weight * ce_value
                            classification_forward_images += len(images)  # Shared image forward.
                            if ce_gradient:
                                ce_client_gradients[token['client_id']] += ce_weight * ce_grad
                                classification_backward_images += len(images)
                                backward_images += len(images)  # Second autograd traversal.
                        if basis is not None:
                            norms[q, view] = grad.norm()  # FULL A gradient, not Q^T g.
                            coordinates[q, view] = grad @ basis
                        if active:
                            gap = (target[q, view] - value).clamp_min(0)
                            loss += weights[q] * .25 * (gap / sigma[q, view]).square()
                            coefficient = -.5 * weights[q] * gap / sigma[q, view].square()
                            client_gradients[token['client_id']] += coefficient * grad
                    if (q+1) % 100 == 0:
                        print(f'SFRA functional tokens: {q+1}/{count}', flush=True)
            finally:
                for parameter, flag in flags:
                    parameter.requires_grad_(flag)
                for module, mode in modes:
                    module.training = mode
        result = dict(scores=scores.cpu(), gradient_norms=norms.cpu(),
                      coordinates=None if coordinates is None else coordinates.cpu(),
                      gradient=None if client_gradients is None else client_gradients.sum(0),
                      loss=float(loss), forward_images=forward_images, backward_images=backward_images,
                      seconds=time.perf_counter()-started)
        if client_gradients is not None:
            require_finite(client_functional_gradients=client_gradients, functional_loss=loss)
        if classification:
            ce_loss = (ce_clients * self.classification_client_weights).sum()
            global_gradient = ((ce_client_gradients * self.classification_client_weights[:, None]).sum(0)
                               if ce_gradient else None)
            require_finite(global_classification_loss=ce_loss)
            if ce_gradient:
                require_finite(global_classification_gradient=global_gradient)
            result.update(classification_loss=float(ce_loss), classification_gradient=global_gradient,
                          classification_client_losses=ce_clients.cpu(), classification_scores=ce_scores.cpu(),
                          classification_forward_images=classification_forward_images,
                          classification_backward_images=classification_backward_images)
        return result
