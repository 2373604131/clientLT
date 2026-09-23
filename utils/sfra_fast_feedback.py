"""Cached, client-local batched execution of the existing witness objective."""
import time

import torch
from torch.nn import functional as F

from utils.cliplora_a_refresh import isolated_rng, train_only
from utils.sfra_execution import FrozenVisionPrefix, tensor_stamp
from utils.sfra_math import require_finite


class FastWitnessFeedback:
    def __init__(self, bank, forward_batch_size=32):
        self.bank, self.forward_batch_size = bank, forward_batch_size
        self.prefix = FrozenVisionPrefix(bank.core.image_encoder)
        self.images, self.features = {}, {}
        self.prefix_stamp = None

    def prepare(self):
        stamp = tensor_stamp(self.prefix.tensors)
        if stamp != self.prefix_stamp:
            self.features.clear()
            self.prefix_stamp = stamp

    def token_images(self, q):
        if q not in self.images:
            t = self.bank.tokens[q]
            self.images[q] = torch.stack([self.bank.datasets[t['client_id']][j]['img']
                                          for j in t['local_positions']])
        return self.images[q]

    def token_prefix(self, q, view):
        key = (q, view)
        if key not in self.features:
            chunks = []
            for images in self.token_images(q).split(self.bank.batch_size):
                images = images.to(self.bank.device)
                if view:
                    images = images.flip(-1)
                chunks.append(self.prefix.forward_prefix(images.type(self.bank.core.dtype)).cpu())
            self.features[key] = torch.cat(chunks, dim=0)
        return self.features[key]

    def groups(self, targets, all_scores, classification):
        """Never combine different clients or change any token's denominator."""
        group, size, client = [], 0, None
        for q, token in enumerate(self.bank.tokens):
            active = targets is not None and bool(targets['active'][q])
            if not all_scores and not active and not classification:
                continue
            n = len(token['local_positions'])
            for view in range(2):
                if group and (token['client_id'] != client or size+n > self.forward_batch_size):
                    yield group
                    group, size = [], 0
                group.append((q, view, n, active))
                size += n
                client = token['client_id']
        if group:
            yield group

    def evaluate(self, targets=None, all_scores=True, classification=False, classification_gradient=True):
        """Source identification stays in WitnessBank's individual-gradient path.

        Here only linear sums of gradients are batched. The functional and LA
        gradients stay separate; the nonlinear GLOBAL CP penalty is unchanged
        and is still applied by SFRARuntime after this method returns.
        """
        started = time.perf_counter()
        b = self.bank
        count = len(b.tokens)
        scores = torch.zeros(count, 2, device=b.device)
        client_gradients = torch.zeros(b.clients, b.numel, device=b.device) if targets is not None else None
        ce_gradient = classification and classification_gradient
        ce_scores = torch.zeros_like(scores) if classification else None
        ce_clients = torch.zeros(b.clients, device=b.device) if classification else None
        ce_client_gradients = torch.zeros(b.clients, b.numel, device=b.device) if ce_gradient else None
        loss = torch.zeros((), device=b.device)
        forward_images = backward_images = classification_forward_images = classification_backward_images = 0
        batches = zero_gradient_batches = 0
        modes = [(m, m.training) for m in b.model.modules()]
        flags = [(p, p.requires_grad) for p in b.model.parameters()]
        if targets is not None:
            target, sigma, weights = (targets[k].to(b.device) for k in ('target', 'sigma', 'weights'))
        with isolated_rng():
            try:
                train_only(b.model, 'A')
                b.model.eval()
                for group in self.groups(targets, all_scores, classification):
                    client = b.tokens[group[0][0]]['client_id']
                    encoded = torch.cat([self.token_prefix(q, view) for q, view, _, _ in group], 0).to(b.device)
                    need_functional = any(active for _, _, _, active in group)
                    functional_terms, ce_terms = [], []
                    with torch.set_grad_enabled(need_functional or ce_gradient):
                        features = self.prefix.forward_suffix(encoded)
                        features = features / features.norm(dim=-1, keepdim=True)
                        cosine = features @ b.text_features.T
                        labels = torch.cat([torch.full((n,), b.tokens[q]['class_id'],
                                                       dtype=torch.long, device=b.device)
                                            for q, _, n, _ in group])
                        wrong = cosine.clone()
                        wrong.scatter_(1, labels[:, None], -torch.inf)
                        margins = cosine.gather(1, labels[:, None]).flatten() - wrong.max(1).values
                        if classification:
                            logits = (b.core.logit_scale.exp().detach()*features) @ b.text_features.T
                            losses = F.cross_entropy(logits+b.classification_logit_adjustment, labels, reduction='none')
                        offset = 0
                        for q, view, n, active in group:
                            margin = margins[offset:offset+n].sum()/n
                            value = margin.detach()
                            scores[q, view] = value
                            if active:
                                gap = (target[q, view]-value).clamp_min(0)
                                loss += weights[q]*.25*(gap/sigma[q, view]).square()
                                coefficient = -.5*weights[q]*gap/sigma[q, view].square()
                                functional_terms.append((coefficient, margin))
                            if classification:
                                ce = losses[offset:offset+n].sum()/n
                                ce_scores[q, view] = ce.detach()
                                ce_weight = .5*b.classification_token_weights[q]
                                ce_clients[client] += ce_weight*ce.detach()
                                if ce_gradient:
                                    ce_terms.append(ce_weight*ce)
                            offset += n
                        # No tolerance or heuristic threshold: only exact zero.
                        if functional_terms:
                            coefficients = torch.stack([c for c, _ in functional_terms])
                            require_finite(functional_coefficients=coefficients)
                            nonzero = (coefficients != 0).cpu().tolist()
                            terms = [c*m for (c, m), keep in zip(functional_terms, nonzero) if keep]
                            if terms:
                                parts = torch.autograd.grad(torch.stack(terms).sum(), b.parameters,
                                                            retain_graph=ce_gradient)
                                client_gradients[client] += torch.cat([p.detach().reshape(-1) for p in parts])
                                backward_images += len(encoded)
                            else:
                                zero_gradient_batches += 1
                        if ce_gradient:
                            parts = torch.autograd.grad(torch.stack(ce_terms).sum(), b.parameters)
                            ce_client_gradients[client] += torch.cat([p.detach().reshape(-1) for p in parts])
                            classification_backward_images += len(encoded)
                            backward_images += len(encoded)
                    forward_images += len(encoded)
                    classification_forward_images += len(encoded) if classification else 0
                    batches += 1
                    if batches % 100 == 0:
                        print(f'SFRA cached functional batches: {batches}', flush=True)
            finally:
                for parameter, flag in flags:
                    parameter.requires_grad_(flag)
                for module, mode in modes:
                    module.training = mode
        require_finite(scores=scores, functional_loss=loss)
        if client_gradients is not None:
            require_finite(client_functional_gradients=client_gradients)
        result = dict(scores=scores.cpu(), gradient_norms=torch.zeros_like(scores).cpu(), coordinates=None,
                      gradient=None if client_gradients is None else client_gradients.sum(0), loss=float(loss),
                      forward_images=forward_images, backward_images=backward_images,
                      execution_forward_batches=batches, execution_zero_gradient_batches=zero_gradient_batches,
                      seconds=time.perf_counter()-started)
        if classification:
            ce_loss = (ce_clients*b.classification_client_weights).sum()
            global_gradient = ((ce_client_gradients*b.classification_client_weights[:, None]).sum(0)
                               if ce_gradient else None)
            require_finite(classification_scores=ce_scores, global_classification_loss=ce_loss)
            if ce_gradient:
                require_finite(global_classification_gradient=global_gradient)
            result.update(classification_loss=float(ce_loss), classification_gradient=global_gradient,
                          classification_client_losses=ce_clients.cpu(), classification_scores=ce_scores.cpu(),
                          classification_forward_images=classification_forward_images,
                          classification_backward_images=classification_backward_images)
        return result
