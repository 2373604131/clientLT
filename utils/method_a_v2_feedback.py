"""Per-image Av2 feedback on the existing frozen-prefix cache and CLIP model."""
import time

import torch
from torch.nn import functional as F

from utils.cliplora_a_refresh import isolated_rng, train_only
from utils.method_a_v2_math import history_terms
from utils.sfra_math import require_finite


class SampleFeedback:
    def __init__(self, bank, counts, groups, logit_adjustment):
        self.bank = bank
        self.counts = torch.as_tensor(counts).float().cpu()
        self.groups = [list(map(int, group)) for group in groups]
        classes = [c for group in self.groups for c in group]
        if len(set(classes)) != len(classes) or sorted(classes) != list(range(self.counts.shape[1])):
            raise ValueError('Av2 classification groups must partition the training classes')
        self.adjustment = logit_adjustment.detach().to(bank.device).float()
        self.manifest, self.offsets, mass, ce_mass, group_ids = [], [], [], [], []
        n_class = self.counts.sum(0)
        for q, token in enumerate(bank.tokens):
            self.offsets.append(len(self.manifest))
            client, c = token['client_id'], token['class_id']
            positions, raw = token['local_positions'], token['raw_sample_ids']
            group = next(g for g, ids in enumerate(self.groups) if c in ids)
            if not positions or len(raw) != len(positions):
                raise ValueError('Incomplete fixed witness identity')
            for pos, sample in zip(positions, raw):
                self.manifest.append(dict(sample_id=int(sample), owner_client_id=int(client),
                    local_position=int(pos), class_id=int(c), token_id=q))
                weight = float(self.counts[client, c]) / len(positions)
                mass.append(weight)
                ce_mass.append(.5 * weight / float(n_class[c]) / len(self.groups[group]))
                group_ids.append(group)
        if len({s['sample_id'] for s in self.manifest}) != len(self.manifest):
            raise ValueError('Duplicated fixed training witness identity')
        self.base_mass = torch.tensor(mass)
        self.ce_mass = torch.tensor(ce_mass, device=bank.device)
        self.group_ids = torch.tensor(group_ids, dtype=torch.long, device=bank.device)
        self.last_version = None

    def evaluate(self, version, basis=None, targets=None, initialize_target=False,
                 gradient=False, cached=True):
        """Source gradients stay per token; correction gradients sum per-image hinges.

        The ordinary pass builds detached targets and its z0 gradient in the same
        forward. Later passes never change targets. No CE backward is needed.
        """
        b = self.bank
        if basis is not None and targets is not None:
            raise ValueError('Source feedback and repair are different model states')
        if initialize_target and targets is None:
            raise ValueError('Target initialization requires a history snapshot')
        cache = getattr(b, '_fast_feedback', None) if cached else None
        if cached and cache is None:
            raise ValueError('Av2 production feedback requires the registered prefix cache')
        if torch.device(b.device).type == 'cuda':
            torch.cuda.synchronize(b.device)
        started = time.perf_counter()
        self.last_version = version
        scores = torch.zeros(len(self.manifest), 2, device=b.device)
        group_ce = torch.zeros(3, device=b.device)
        norms = torch.zeros(len(b.tokens), 2)
        coordinates = torch.zeros(len(b.tokens), 2, basis.shape[1]) if basis is not None else None
        total_loss = torch.zeros((), device=b.device)
        total_gradient = torch.zeros(b.numel, device=b.device) if gradient else None
        forward_images = backward_images = 0
        local = {k:v.to(b.device) for k,v in targets.items()} if targets is not None else None
        flags = [(p, p.requires_grad) for p in b.model.parameters()]
        modes = [(m, m.training) for m in b.model.modules()]
        with isolated_rng():
            try:
                train_only(b.model, 'A')
                b.model.eval()
                if cache is not None:
                    cache.prepare()
                if basis is not None or cache is None:
                    batches = ([(q,v,len(t['local_positions']),False)]
                               for q,t in enumerate(b.tokens) for v in range(2))
                else:
                    batches = cache.groups(None, True, True)
                for batch in batches:
                    ids = torch.cat([torch.arange(self.offsets[q], self.offsets[q]+n, device=b.device)
                                     for q,v,n,_ in batch])
                    views = torch.cat([torch.full((n,), v, dtype=torch.long, device=b.device)
                                       for q,v,n,_ in batch])
                    labels = torch.cat([torch.full((n,), b.tokens[q]['class_id'], dtype=torch.long, device=b.device)
                                        for q,v,n,_ in batch])
                    need_gradient = gradient and bool(local['valid'][ids].any())
                    with torch.set_grad_enabled(basis is not None or need_gradient):
                        if cache is not None:
                            encoded = torch.cat([cache.token_prefix(q,v).to(b.device) for q,v,_,_ in batch])
                            features = cache.prefix.forward_suffix(encoded)
                        else:
                            q,v,_,_ = batch[0]
                            t = b.tokens[q]
                            images = torch.stack([b.datasets[t['client_id']][j]['img'] for j in t['local_positions']]).to(b.device)
                            if v:
                                images = images.flip(-1)
                            features = b.core.image_encoder(images.type(b.core.dtype))
                        features = features / features.norm(dim=-1, keepdim=True)
                        cosine = features @ b.text_features.T
                        wrong = cosine.clone()
                        wrong.scatter_(1, labels[:,None], -torch.inf)
                        margins = cosine.gather(1, labels[:,None]).flatten() - wrong.max(1).values
                        require_finite(sample_margins=margins)
                        scores[ids, views] = margins.detach()
                        with torch.no_grad():
                            logits = (b.core.logit_scale.exp().detach()*features.detach()) @ b.text_features.T
                            ce = F.cross_entropy(logits+self.adjustment, labels, reduction='none')
                            group_ce.scatter_add_(0, self.group_ids[ids], self.ce_mass[ids]*ce)
                        if basis is not None:
                            q,v,_,_ = batch[0]
                            parts = torch.autograd.grad(margins.mean(), b.parameters)
                            g = torch.cat([p.detach().flatten() for p in parts])
                            norms[q,v] = g.norm().cpu()
                            coordinates[q,v] = (g @ basis).cpu()
                            backward_images += len(ids)
                        if local is not None:
                            if initialize_target:
                                h, k = local['H'][ids], local['effective'][ids]
                                target = h-(1-k)*(h-margins.detach()).clamp_min(0)
                                local['target'][ids,views] = torch.where(local['valid'][ids], target, 0.)
                            terms = history_terms(margins, local['target'][ids,views],
                                                  local['sigma'][ids,views], local['omega'][ids])
                            loss = terms.sum()
                            total_loss += loss.detach()
                            if need_gradient and bool(loss.detach() != 0):
                                parts = torch.autograd.grad(loss, b.parameters)
                                total_gradient += torch.cat([p.detach().flatten() for p in parts])
                                backward_images += len(ids)
                        forward_images += len(ids)
            finally:
                for p, flag in flags:
                    p.requires_grad_(flag)
                for module, mode in modes:
                    module.training = mode
        require_finite(margins=scores, group_ce=group_ce, loss=total_loss, source_norms=norms)
        if total_gradient is not None:
            require_finite(gradient=total_gradient)
        if coordinates is not None:
            require_finite(source_coordinates=coordinates)
        if initialize_target:
            targets['target'] = local['target'].detach().cpu()
        if torch.device(b.device).type == 'cuda':
            torch.cuda.synchronize(b.device)
        return dict(sample_margins=scores.cpu(), group_ce=group_ce.cpu(), loss=float(total_loss),
                    gradient=total_gradient, gradient_norms=norms, coordinates=coordinates,
                    seconds=time.perf_counter()-started, forward_images=forward_images,
                    backward_images=backward_images, model_version=version)
