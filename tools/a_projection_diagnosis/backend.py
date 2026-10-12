"""Real CLIP evaluation with streaming q/v pre-hooks and isolated local replay."""
import hashlib
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from tools.a_learning_schedule.runtime import build_config, load_training_api, make_data, save_arrays, save_torch
from tools.a_projection_diagnosis.math import row_basis
from tools.a_projection_diagnosis.protocol import fit_manifest, test_manifests, digest, write_csv, write_json, read_json, read_csv
from utils.cliplora_a_refresh import isolated_rng, state_hash, train_only


def token_summary(value, batch_first=False):
    if value.ndim != 3:
        raise ValueError('q/v pre-hook must receive actual [tokens,batch,width] input')
    tokens = value if batch_first else value.transpose(0, 1)
    if tokens.shape[1] < 2:
        raise ValueError('CLS and patch tokens are required')
    return torch.stack((tokens[:, 0], tokens[:, 1:].mean(1)), 1)


class ClipBackend:
    def __init__(self, args, source, store, root, fingerprint):
        self.args, self.source, self.store, self.root = args, source, store, Path(root)
        self.fingerprint = fingerprint
        self.device = torch.device(args.device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA unavailable; run in the server clientlt environment')
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.backends.cuda.matmul.allow_tf32 = False
        torch.backends.cudnn.allow_tf32 = False
        cfg = build_config(source, args)
        build, self.step = load_training_api()
        with isolated_rng(args.seed):
            self.model = build(cfg, source['meta']['classnames']).to(self.device)
        self.model.load_state_dict(store.base, strict=True)
        self.model.requires_grad_(False)
        from utils.sfra_execution import FrozenTextCache
        self.model._sfra_text_cache = FrozenTextCache(self.model)
        self.keys, self.a_keys = store.keys, store.a_keys
        named = dict(self.model.named_modules())
        self.modules = {k: named[k.rsplit('.', 1)[0]] for k in self.a_keys}
        # PlainMultiheadAttentionLoRA transposes batch_first inputs BEFORE q/v.
        self.batch_first = {k: False for k in self.a_keys}
        self.scales = {k: float(m.scaling) for k, m in self.modules.items()}
        locations = {(int(k.split('resblocks.')[1].split('.')[0]), k.split('.')[-2]) for k in self.a_keys}
        if locations != {(b, p) for b in (9, 10, 11) for p in ('q_proj', 'v_proj')} or set(self.scales.values()) != {.5}:
            raise ValueError('Unexpected LoRA locations/scaling')
        self.block9 = [k for k in self.a_keys if '.resblocks.9.' in k]
        with isolated_rng(args.seed):
            data = make_data(cfg, source, args)
        self.data = data
        self.groups = dict(overall=list(range(100)), head20=source['config']['head_ids'],
                           middle60=source['config']['middle_ids'], tail20=source['config']['tail_ids'])
        from utils.functional_coverage_validation import _TrainOnlyCifar100, _locate_cifar100
        directory = _locate_cifar100(args.data_root.resolve())
        raw = _TrainOnlyCifar100(directory)
        self.fit_rows = fit_manifest(source['rows'])
        fit_items = [SimpleNamespace(data=raw.images[r['raw_sample_id']], label=r['class_id'], domain=0) for r in self.fit_rows]
        with (directory / 'test').open('rb') as stream:
            test = pickle.load(stream, encoding='latin1')
        y = np.asarray(test['fine_labels'], dtype=np.int64)
        images = np.asarray(test['data'], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
        if len(y) != 10000 or not np.all(np.bincount(y, minlength=100) == 100):
            raise ValueError('Expected the official balanced CIFAR-100 test set')
        test_hash = hashlib.sha256()
        for label, img in zip(y, images):
            test_hash.update(str(int(label)).encode())
            test_hash.update(img.tobytes())
        if test_hash.hexdigest() != source['meta']['test_sha256']:
            raise ValueError('Official test images/order differ from the source run')
        self.eval_rows = test_manifests(y)
        self.ids = dict(fit=np.array([r['raw_sample_id'] for r in self.fit_rows]), test=np.arange(len(y)),
                        panel=np.array([r['sample_id'] for r in self.eval_rows if r['split'] == 'panel']),
                        remaining=np.array([r['sample_id'] for r in self.eval_rows if r['split'] == 'remaining']))
        self.labels = dict(fit=np.array([r['class_id'] for r in self.fit_rows]))
        self.loaders = dict(fit=data['loader'](fit_items, False))
        for split in ('test', 'panel', 'remaining'):
            self.labels[split] = y[self.ids[split]]
            items = [SimpleNamespace(data=images[i], label=int(y[i]), domain=0) for i in self.ids[split]]
            self.loaders[split] = data['loader'](items, False)
        self.fit_hash, self.eval_hash = digest(self.fit_rows), digest(self.eval_rows)
        write_csv(self.root / 'probe_fit_manifest.csv', self.fit_rows)
        write_csv(self.root / 'audit_eval_manifest.csv', self.eval_rows)
        prior = torch.as_tensor(data['counts'], dtype=torch.float32, device=self.device)
        self.adjustment = (prior / prior.sum()).log()
        self.costs = read_csv(self.root / 'costs.csv') if (self.root / 'costs.csv').is_file() else []
        write_json(self.root / 'environment.json', dict(torch=str(torch.__version__), cuda=torch.version.cuda,
            gpu=torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else 'CPU',
            source_environment=source['meta'].get('environment'), actual_hook='q_proj/v_proj forward_pre_hook after ln_1',
            modules=self.a_keys, scales=self.scales, fit_sha256=self.fit_hash, eval_sha256=self.eval_hash,
            code_fingerprint=fingerprint))

    def restore(self, state):
        if set(state) != set(self.keys):
            raise ValueError('Restore requires complete LoRA state')
        own = self.model.state_dict()
        with torch.no_grad():
            for k in self.keys:
                own[k].copy_(state[k])

    def measure(self, state, projections, split, force=False):
        """One ordinary forward per batch; stream its summaries into all Qs.

        Cached tensors contain only 8D/module features and predictions. Projection
        candidates never replace the inputs or weights of the forward model.
        """
        identity = dict(state=state_hash(state, self.keys),
                        projections={n: state_hash(s, self.a_keys) for n, s in projections.items()},
                        split=split, fit=self.fit_hash, evaluation=self.eval_hash, code=self.fingerprint,
                        eval_batch_size=self.args.eval_batch_size, device=str(self.device),
                        torch=str(torch.__version__), cuda=torch.version.cuda)
        cache = self.root / 'cache' / digest(identity)
        metadata = cache.with_suffix('.json')
        arrays_path = cache.with_suffix('.npz')
        names = list(projections)
        if not force and metadata.is_file() and arrays_path.is_file():
            record = read_json(metadata)
            if record['identity'] != identity:
                raise ValueError('Cache identity differs')
            with np.load(arrays_path) as saved:
                arrays = dict(saved)
        else:
            self.restore(state)
            bases = {k: torch.stack([row_basis(projections[n][k])[0] for n in names]).to(self.device, torch.float32)
                     for k in self.a_keys} if names else {}
            buffers, hashes = {}, {k: hashlib.sha256() for k in self.block9}
            output = {k: [] for k in ('labels', 'prediction', 'ce', 'margin')}
            handles = []
            counts = {k: 0 for k in self.a_keys}

            def hook(k):
                def capture(module, inputs):
                    pooled = token_summary(inputs[0].detach(), self.batch_first[k])
                    counts[k] += 1
                    if k in hashes:
                        hashes[k].update(pooled.cpu().contiguous().numpy().tobytes())
                    if names:
                        z = torch.einsum('btd,nrd->nbtr', pooled, bases[k]).flatten(2).cpu().numpy()
                        for n, value in enumerate(z):
                            buffers.setdefault('f%d_%d' % (n, self.a_keys.index(k)), []).append(value)
                return capture

            modes = [(m, m.training) for m in self.model.modules()]
            started = time.perf_counter()
            try:
                for k, module in self.modules.items():
                    handles.append(module.register_forward_pre_hook(hook(k)))
                with isolated_rng(), torch.no_grad():
                    self.model.eval()
                    scale = self.model.logit_scale.exp()
                    batches = 0
                    for batch in self.loaders[split]:
                        labels = batch['label'].to(self.device)
                        logits = self.model(batch['img'].to(self.device))
                        if not torch.isfinite(logits).all():
                            raise FloatingPointError('Nonfinite CLIP logits')
                        cosine = logits / scale
                        true = cosine.gather(1, labels[:, None]).flatten()
                        wrong = cosine.clone().scatter_(1, labels[:, None], -torch.inf).max(1).values
                        values = dict(labels=labels, prediction=logits.argmax(1), margin=true - wrong,
                                      ce=F.cross_entropy(logits, labels, reduction='none'))
                        for k, v in values.items():
                            output[k].append(v.cpu().numpy())
                        batches += 1
                    if any(v != batches for v in counts.values()):
                        raise ValueError('A q/v hook was skipped or called more than once per forward')
            finally:
                for h in handles:
                    h.remove()
                for module, mode in modes:
                    module.training = mode
            arrays = {k: np.concatenate(v) for k, v in dict(output, **buffers).items()}
            arrays['sample_id'] = self.ids[split]
            if not np.array_equal(arrays['labels'], self.labels[split]):
                raise ValueError('Evaluation order changed')
            record = dict(identity=identity, block9_input_hash={k: h.hexdigest() for k, h in hashes.items()},
                          seconds=time.perf_counter() - started, forward_images=len(arrays['labels']),
                          cache_bytes=sum(v.nbytes for v in arrays.values()))
            save_arrays(arrays_path, **arrays)
            write_json(metadata, record)
            self.costs.append(dict(kind='forward', split=split, **{k: record[k] for k in ('seconds', 'forward_images', 'cache_bytes')}))
            write_csv(self.root / 'costs.csv', self.costs)
        features = {n: {k: arrays['f%d_%d' % (i, j)] for j, k in enumerate(self.a_keys)} for i, n in enumerate(names)}
        predictions = {k: arrays[k] for k in ('labels', 'sample_id', 'prediction', 'margin', 'ce')}
        return dict(features=features, predictions=predictions, record=record)

    def regenerate(self, anchor, rnd):
        path = self.root / 'regenerated' / ('r%03d.pt' % rnd)
        identity = dict(anchor=state_hash(anchor, self.keys), round=rnd, code=self.fingerprint,
                        partition=self.source['partition_sha256'])
        if path.is_file():
            payload = torch.load(path, map_location='cpu', weights_only=False)
            if payload['identity'] != identity:
                raise ValueError('Regenerated proposal cache differs')
            return payload['proposals'], payload['provenance']
        proposals, budgets = {}, []
        train_only(self.model, 'A')
        selected = list(map(int, self.source['meta']['schedule'][rnd - 1]))
        for client in selected:
            self.restore(anchor)
            optimizer = torch.optim.SGD([p for p in self.model.parameters() if p.requires_grad],
                                        lr=.001, momentum=.9, weight_decay=0.)
            started, steps, images = time.perf_counter(), 0, 0
            with isolated_rng(self.args.seed + 1000003 * rnd + 1009 * client):
                self.model.train()
                for batch in self.data['loader'](self.data['clients'][client], True):
                    x, y = batch['img'].to(self.device), batch['label'].to(self.device)
                    self.step(self.model, optimizer, None, 'fp32', x, y, logit_adjustment=self.adjustment)
                    steps, images = steps + 1, images + len(y)
            own = self.model.state_dict()
            if any(not torch.equal(own[k].cpu(), anchor[k]) for k in self.keys if k not in self.a_keys):
                raise ValueError('Regeneration changed frozen B')
            proposals[client] = {k: own[k].detach().cpu().clone() - anchor[k] for k in self.a_keys}
            budgets.append(dict(client=client, steps=steps, images=images, seconds=time.perf_counter() - started))
            print('Regenerated paired A proposals: round%d client%d/30' % (rnd, client + 1), flush=True)
        train_only(self.model, '')
        self.restore(anchor)
        provenance = dict(proposals='regenerated_paired_proposals_from_historical_anchor',
                          historical_exact_reproduction=False, optimizer='fresh SGD .001 momentum .9 wd0, one epoch',
                          seed_rule='42 + 1000003*round + 1009*client', budgets=budgets)
        save_torch(path, dict(identity=identity, proposals=proposals, provenance=provenance))
        return proposals, provenance
