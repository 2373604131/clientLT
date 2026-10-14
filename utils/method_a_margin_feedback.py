"""One joint LA/retention backward per batch on the existing frozen-prefix cache."""
import time

import torch
from torch.nn import functional as F

from utils.cliplora_a_refresh import isolated_rng, train_only
from utils.method_a_v2_feedback import SampleFeedback
from utils.method_a_margin_math import keep_terms
from utils.sfra_math import require_finite


class MarginFeedback(SampleFeedback):
    # Reuse only witness identities and CE coefficients, never the Av2 evaluate/source path.
    def evaluate(self, version, targets=None, gradient=False, keep_weight=1., learn_weight=1., cached=True,
                 diagnostics=False):
        b = self.bank
        if gradient and targets is None:
            raise ValueError('Joint margin backward requires fixed targets')
        cache = getattr(b, '_fast_feedback', None) if cached else None
        if cached and cache is None:
            raise ValueError('Margin production feedback requires the registered prefix cache')
        if torch.device(b.device).type == 'cuda':
            torch.cuda.synchronize(b.device)
        started = time.perf_counter()
        self.last_version = version
        scores = torch.zeros(len(self.manifest), 2, device=b.device)
        group_ce = torch.zeros(3, device=b.device)
        group_fraction = scores.new_tensor([len(g) / self.counts.shape[1] for g in self.groups])
        keep_loss = scores.new_zeros(())
        class_loss = scores.new_zeros(())
        total_gradient = scores.new_zeros(b.numel) if gradient else None
        # Opt-in observation only: retain the same graph for component gradients,
        # then execute the original joint backward unchanged.
        component_gradients = {name: scores.new_zeros(b.numel) for name in ('keep', 'learn')} if diagnostics and gradient else {}
        correct = torch.zeros_like(scores, dtype=torch.bool) if diagnostics else None
        la_correct = torch.zeros_like(correct) if diagnostics else None
        local = {k: v.detach().to(b.device) for k, v in targets.items()} if targets is not None else None
        forward_images = backward_images = backward_calls = batches = 0
        flags = [(p, p.requires_grad) for p in b.model.parameters()]
        modes = [(m, m.training) for m in b.model.modules()]
        with isolated_rng():
            try:
                train_only(b.model, 'A')
                b.model.eval()
                if cache is not None:
                    cache.prepare()
                    batch_stream = cache.groups(None, True, True)
                else:
                    batch_stream = ([(q, v, len(t['local_positions']), False)]
                                    for q, t in enumerate(b.tokens) for v in range(2))
                for batch in batch_stream:
                    ids = torch.cat([torch.arange(self.offsets[q], self.offsets[q]+n, device=b.device)
                                     for q, v, n, _ in batch])
                    views = torch.cat([torch.full((n,), v, dtype=torch.long, device=b.device)
                                       for q, v, n, _ in batch])
                    labels = torch.cat([torch.full((n,), b.tokens[q]['class_id'], dtype=torch.long, device=b.device)
                                        for q, v, n, _ in batch])
                    with torch.set_grad_enabled(gradient):
                        if cache is not None:
                            encoded = torch.cat([cache.token_prefix(q, v).to(b.device) for q, v, _, _ in batch])
                            features = cache.prefix.forward_suffix(encoded)
                        else:
                            q, v, _, _ = batch[0]
                            t = b.tokens[q]
                            images = torch.stack([b.datasets[t['client_id']][j]['img']
                                                  for j in t['local_positions']]).to(b.device)
                            if v:
                                images = images.flip(-1)
                            features = b.core.image_encoder(images.type(b.core.dtype))
                        features = features / features.norm(dim=-1, keepdim=True)
                        cosine = features @ b.text_features.T
                        wrong = cosine.clone()
                        wrong.scatter_(1, labels[:, None], -torch.inf)
                        margins = cosine.gather(1, labels[:, None]).flatten() - wrong.max(1).values
                        ce = F.cross_entropy(b.core.logit_scale.exp().detach()*cosine + self.adjustment,
                                             labels, reduction='none')
                        # Global coefficients already include both views and all client/class masses.
                        ce_terms = self.ce_mass[ids] * ce
                        learning = (ce_terms * group_fraction[self.group_ids[ids]]).sum()
                        keeping = (keep_terms(margins, local['target'][ids, views], local['sigma'],
                                              local['omega'][ids]).sum() if local is not None
                                   else learning.new_zeros(()))
                        joint = keep_weight * keeping + learn_weight * learning
                        require_finite(margins=margins, joint_loss=joint)
                        scores[ids, views] = margins.detach()
                        if diagnostics:
                            correct[ids, views] = cosine.detach().argmax(1) == labels
                            la_correct[ids, views] = (b.core.logit_scale.exp().detach()*cosine.detach()
                                                     + self.adjustment).argmax(1) == labels
                        group_ce.scatter_add_(0, self.group_ids[ids], ce_terms.detach())
                        keep_loss += keeping.detach()
                        class_loss += learning.detach()
                        if gradient:
                            if diagnostics:
                                for name, loss in (('keep', keep_weight*keeping), ('learn', learn_weight*learning)):
                                    parts = torch.autograd.grad(loss, b.parameters, retain_graph=True)
                                    component_gradients[name] += torch.cat([p.detach().flatten() for p in parts])
                            # Classification includes samples with no history and batches without a gap.
                            parts = torch.autograd.grad(joint, b.parameters)
                            total_gradient += torch.cat([p.detach().flatten() for p in parts])
                            backward_calls += 1
                            backward_images += len(ids)
                    batches += 1
                    forward_images += len(ids)
            finally:
                for p, flag in flags:
                    p.requires_grad_(flag)
                for module, mode in modes:
                    module.training = mode
        require_finite(margins=scores, group_ce=group_ce, keep_loss=keep_loss, class_loss=class_loss)
        if total_gradient is not None:
            require_finite(joint_gradient=total_gradient)
        deficit_count = (int(((scores < local['target']).any(1) & local['valid']).sum())
                         if local is not None else 0)
        if torch.device(b.device).type == 'cuda':
            torch.cuda.synchronize(b.device)
        result = dict(sample_margins=scores.cpu(), group_ce=group_ce.cpu(),
                    keep_loss=float(keep_loss), class_loss=float(class_loss),
                    loss=float(keep_weight*keep_loss + learn_weight*class_loss),
                    gradient=total_gradient, deficit_count=deficit_count,
                    seconds=time.perf_counter()-started, forward_images=forward_images,
                    backward_images=backward_images, backward_calls=backward_calls,
                    batches=batches, model_version=version)
        if diagnostics:
            require_finite(**component_gradients)
            result.update(component_gradients=component_gradients, correct=correct.cpu(),
                          la_correct=la_correct.cpu(),
                          diagnostic_extra_backward_calls=2*batches if gradient else 0)
        return result
