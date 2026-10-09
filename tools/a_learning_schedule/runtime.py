"""GPU worker; one anchor, four paired arms, cached shared training prefixes."""
import math
import os
import pickle
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
from torch.nn import functional as F

from tools.a_learning_schedule.protocol import (
    ARMS, HORIZON, PROTOCOL, check_source, digest, file_digest, final_nodes,
    job_root, paired_probes, phase_seed, plan, read_json, source_event, write_csv, write_json,
)
from utils.cliplora_a_refresh import isolated_rng, lora_keys, product_norm_squared, state_hash, train_only


def save_torch(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    torch.save(payload, temporary)
    os.replace(temporary, path)


def save_arrays(path, **arrays):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + '.tmp')
    with temporary.open('wb') as stream:
        np.savez_compressed(stream, **arrays)
    os.replace(temporary, path)


def load_torch(path):
    return torch.load(path, map_location='cpu', weights_only=False)


def effective_norm(before, after, a_keys, scales):
    total = 0.
    for a in a_keys:
        b = a[:-1] + 'B'
        old_a, old_b = before[a].double(), before[b].double()
        da, db = after[a].double() - old_a, after[b].double() - old_b
        left = torch.cat((old_b, db), 1)
        right = torch.cat((da, after[a].double()), 0)
        total += scales[a] ** 2 * product_norm_squared(left, right)
    return math.sqrt(total)


def aggregate(before, local_states, selected, weights, active, kind):
    """Match LA-control: normal averages states; extra averages deltas, in FP32."""
    result = {k: v.clone() for k, v in before.items()}
    for key in active:
        value = torch.zeros_like(before[key])
        for client in selected:
            update = local_states[client][key]
            if kind == 'extra':
                update = update - before[key]
            value.add_(update, alpha=float(weights[client]))
        result[key] = before[key] + value if kind == 'extra' else value
    return result


def prediction_metrics(arrays, groups, anchor=None):
    labels, pred = arrays['labels'], arrays['prediction']
    correct = pred == labels
    output = {}
    for name, ids in groups.items():
        mask = np.isin(labels, ids)
        classes = [c for c in ids if np.any(labels == c)]
        if not classes:
            continue
        # All reported group measures are class means (test has 100 images/class).
        output[name] = float(np.mean([100 * correct[labels == c].mean() for c in classes]))
        for metric in ('ce', 'la', 'margin'):
            output[name + '_' + metric] = float(np.mean([arrays[metric][labels == c].mean() for c in classes]))
        if anchor is not None:
            if not np.array_equal(labels, anchor['labels']):
                raise ValueError('Sample identity/order changed between evaluations')
            old = anchor['prediction'] == labels
            output[name + '_new_correct'] = int(np.sum(mask & ~old & correct))
            output[name + '_forgotten'] = int(np.sum(mask & old & ~correct))
            output[name + '_retained_correct'] = int(np.sum(mask & old & correct))
    return output


def build_config(source, args):
    # Load the complete archived configuration, avoiding today's launcher defaults.
    from yacs.config import CfgNode
    cfg = CfgNode.load_cfg(source['meta']['resolved_config'])
    cfg.defrost()
    cfg.DATASET.ROOT = str(args.data_root.resolve())
    cfg.OUTPUT_DIR = str(args.output_root.resolve())
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.MODEL.INIT_WEIGHTS = ''
    if (cfg.OPTIM.NAME != 'sgd' or cfg.OPTIM.LR != .001 or cfg.OPTIM.MOMENTUM != .9
            or cfg.OPTIM.WEIGHT_DECAY != .0005 or cfg.OPTIM.GAMMA != 1
            or cfg.OPTIM.LR_SCHEDULER != 'single_step'
            or cfg.OPTIM.WARMUP_EPOCH != -1 or cfg.OPTIM.SGD_NESTEROV
            or cfg.OPTIM.SGD_DAMPNING != 0 or cfg.OPTIM.MAX_EPOCH != 3):
        raise ValueError('Source optimizer differs from the fixed SGD/constant-LR protocol')
    if (cfg.TRAINER.CLIPLORA.r != 4 or cfg.TRAINER.CLIPLORA.alpha != 1 or cfg.TRAINER.CLIPLORA.encoder != 'vision'
            or cfg.TRAINER.CLIPLORA.position != 'top3' or cfg.TRAINER.COOP.PREC != 'fp32'
            or list(cfg.TRAINER.CLIPLORA.params) != ['q', 'v']
            or cfg.TRAINER.CLIPLORA.dropout_rate != 0
            or cfg.TRAINER.CLIPLORA.SCA_ENABLED):
        raise ValueError('Expected vision top3 q/v, rank4, FP32, zero dropout')
    cfg.freeze()
    return cfg


def load_training_api():
    # Match federated_main.py: finish Dassl's trainer registration first.
    # Importing cliplora first re-enters it from engine.build before ClipLora exists.
    import Dassl.dassl.engine  # noqa: F401
    from trainers.cliplora import build_cliplora_model, cliplora_optimizer_step
    return build_cliplora_model, cliplora_optimizer_step


def make_data(cfg, source, args):
    from Dassl.dassl.data.data_manager import build_data_loader
    from Dassl.dassl.data.transforms import build_transform
    from utils.functional_coverage_validation import _TrainOnlyCifar100, _locate_cifar100
    directory = _locate_cifar100(args.data_root.resolve())
    store = _TrainOnlyCifar100(directory)
    train_transform = build_transform(cfg, is_train=True)
    test_transform = build_transform(cfg, is_train=False)
    clients = [[] for _ in range(30)]
    feedback_items, feedback_rows, feedback_by_client = [], [], [[] for _ in range(30)]
    selected_counts = {}
    for row in source['rows']:
        raw, client, label = (int(row[k]) for k in ('raw_sample_id', 'client_id', 'class_id'))
        if int(store.labels[raw]) != label:
            raise ValueError('Raw image labels disagree with archived partition')
        item = SimpleNamespace(data=store.images[raw], label=label, domain=0)
        clients[client].append(item)
        key = (client, label)
        if selected_counts.get(key, 0) < PROTOCOL['feedback_per_client_class']:
            index = len(feedback_items)
            feedback_items.append(item)
            feedback_by_client[client].append(index)
            feedback_rows.append(dict(index=index, client_id=client, class_id=label,
                                      raw_sample_id=raw, local_position=int(row['local_position'])))
            selected_counts[key] = selected_counts.get(key, 0) + 1

    def loader(items, training):
        return build_data_loader(cfg, sampler_type='RandomSampler' if training else 'SequentialSampler',
                                 data_source=items, batch_size=32 if training else args.eval_batch_size,
                                 tfm=train_transform if training else test_transform,
                                 is_train=training, drop_last=False)

    with (directory / 'test').open('rb') as stream:
        test = pickle.load(stream, encoding='latin1')
    images = np.asarray(test['data'], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_items = [SimpleNamespace(data=x, label=int(y), domain=0) for x, y in zip(images, test['fine_labels'])]
    counts = np.asarray(source['counts']).sum(0)
    head = sorted(range(100), key=lambda c: (-int(counts[c]), c))[:20]
    tail = sorted(range(100), key=lambda c: (int(counts[c]), -c))[:20]
    if set(tail) != set(source['config']['tail_ids']):
        raise ValueError('Tail classes differ from source protocol')
    groups = dict(overall=list(range(100)), head20=head,
                  middle60=[c for c in range(100) if c not in head + tail], tail20=tail)
    return dict(clients=clients, loader=loader, test=loader(test_items, False),
                feedback=loader(feedback_items, False), feedback_rows=feedback_rows,
                feedback_indices=feedback_by_client,
                local_feedback=[loader([feedback_items[i] for i in ids], False) for ids in feedback_by_client],
                groups=groups, counts=counts)


class Worker:
    def __init__(self, args, source, origin, anchor_round, root):
        self.args, self.source, self.origin, self.round, self.root = args, source, origin, anchor_round, Path(root)
        self.device = torch.device(args.device)
        if self.device.type == 'cuda' and not torch.cuda.is_available():
            raise RuntimeError('CUDA is unavailable; activate the GPU clientlt environment')
        if self.device.type == 'cuda':
            torch.backends.cudnn.benchmark = False
            torch.backends.cudnn.deterministic = True
            torch.backends.cuda.matmul.allow_tf32 = False
            torch.backends.cudnn.allow_tf32 = False
        cfg = build_config(source, args)
        build_cliplora_model, cliplora_optimizer_step = load_training_api()
        with isolated_rng(args.seed):
            self.model = build_cliplora_model(cfg, source['meta']['classnames']).to(self.device)
        self.step = cliplora_optimizer_step
        event_path = source_event(source['path'], origin, anchor_round)
        event = load_torch(event_path)
        expected_phase = 'extra_B' if origin == 'e2' else 'refresh_A'
        if event['round'] != anchor_round or event['phase'] != expected_phase:
            raise ValueError('Incorrect source event round/phase')
        # Crucial: BEFORE the archived extra step, not its actual_after_lora_state.
        self.anchor = {k: v.clone() for k, v in event['anchor_lora_state'].items()}
        self.a_keys, self.b_keys = lora_keys(self.model, 'A'), lora_keys(self.model, 'B')
        self.keys = sorted(self.a_keys + self.b_keys)
        if set(self.anchor) != set(self.keys) or not all(torch.count_nonzero(self.anchor[k]) for k in self.b_keys):
            raise ValueError('Incomplete anchor or zero B: A-only learning would be invalid')
        base = load_torch(Path(source['path']) / 'checkpoints/base_model.pt')
        base.update(self.anchor)
        self.model.load_state_dict(base, strict=True)
        self.frozen_hash = state_hash(base, sorted(set(base) - set(self.keys)))
        del base
        train_only(self.model, 'B')
        from utils.sfra_execution import FrozenTextCache
        self.model._sfra_text_cache = FrozenTextCache(self.model)
        modules = dict(self.model.named_modules())
        self.scales = {k: float(modules[k.rsplit('.', 1)[0]].scaling) for k in self.a_keys}
        if set(self.scales.values()) != {.5}:
            raise ValueError('Unexpected effective LoRA scaling')
        with isolated_rng(args.seed):
            self.data = make_data(cfg, source, args)
        prior = torch.as_tensor(self.data['counts'], dtype=torch.float32, device=self.device)
        self.adjustment = (prior / prior.sum()).log()
        self.weights = [n / sum(source['sizes']) for n in source['sizes']]
        self.steps_epoch = sum(math.ceil(n / 32) for n in source['sizes'])
        self.root.mkdir(parents=True, exist_ok=True)
        write_csv(self.root / 'feedback_manifest.csv', self.data['feedback_rows'])
        write_json(self.root / 'anchor_info.json', dict(origin=origin, anchor_round=anchor_round,
                   source=str(event_path), state_key='anchor_lora_state',
                   anchor_sha256=state_hash(self.anchor, self.keys), groups=self.data['groups'],
                   feedback_scope='fixed first <=8 positions per client-class; training images; observation only',
                   frozen_sha256=self.frozen_hash, steps_per_epoch=self.steps_epoch,
                   test_samples=10000, scales=self.scales, torch=torch.__version__,
                   cuda=torch.version.cuda, device=str(self.device),
                   gpu=torch.cuda.get_device_name(self.device) if self.device.type == 'cuda' else 'CPU',
                   visible_devices=os.environ.get('CUDA_VISIBLE_DEVICES'),
                   source_training_code_hashes=source['meta'].get('training_code_hashes', {})))

    def restore(self, state):
        if set(state) != set(self.keys):
            raise ValueError('Only complete A+B states can be restored')
        own = self.model.state_dict()
        with torch.no_grad():
            for key in self.keys:
                own[key].copy_(state[key])

    @torch.no_grad()
    def evaluate_loader(self, loader):
        modes = [(m, m.training) for m in self.model.modules()]
        result = {k: [] for k in ('labels', 'prediction', 'ce', 'la', 'margin')}
        with isolated_rng():
            try:
                self.model.eval()
                scale = self.model.logit_scale.exp().detach()
                for batch in loader:
                    labels = batch['label'].to(self.device)
                    logits = self.model(batch['img'].to(self.device))
                    if not torch.isfinite(logits).all():
                        raise FloatingPointError('Nonfinite evaluation logits')
                    cosine = logits / scale
                    true = cosine.gather(1, labels[:, None]).flatten()
                    wrong = cosine.clone().scatter_(1, labels[:, None], -torch.inf)
                    values = dict(labels=labels, prediction=logits.argmax(1),
                                  ce=F.cross_entropy(logits, labels, reduction='none'),
                                  la=F.cross_entropy(logits + self.adjustment, labels, reduction='none'),
                                  margin=true - wrong.max(1).values)
                    for key, value in values.items():
                        result[key].append(value.cpu().numpy())
            finally:
                for module, training in modes:
                    module.training = training
        return {key: np.concatenate(value) for key, value in result.items()}

    def evaluate(self, identity, state):
        root = self.root / 'evaluations' / identity
        complete = root / 'metrics.json'
        if complete.is_file():
            result = read_json(complete)
            if result['state_sha256'] != state_hash(state, self.keys):
                raise ValueError('Evaluation cache state mismatch')
            if all((root / (name + '.npz')).is_file() for name in ('test', 'feedback')):
                return result
        self.restore(state)
        started = time.perf_counter()
        arrays = {name: self.evaluate_loader(self.data[name]) for name in ('test', 'feedback')}
        metrics = {}
        per_class = []
        for name, value in arrays.items():
            baseline = None
            if identity != 'anchor':
                with np.load(self.root / 'evaluations/anchor' / (name + '.npz')) as stored:
                    baseline = dict(stored)
            metrics[name] = prediction_metrics(value, self.data['groups'], baseline)
            for c in sorted(set(value['labels'].tolist())):
                mask = value['labels'] == c
                row = dict(domain=name, class_id=c, samples=int(mask.sum()),
                           accuracy=float(100 * (value['prediction'][mask] == c).mean()))
                row.update({key: float(value[key][mask].mean()) for key in ('ce', 'la', 'margin')})
                if baseline is not None:
                    correct, old = value['prediction'][mask] == c, baseline['prediction'][mask] == c
                    row.update(new_correct=int(np.sum(~old & correct)), forgotten=int(np.sum(old & ~correct)))
                per_class.append(row)
            save_arrays(root / (name + '.npz'), **value)
        write_csv(root / 'per_class.csv', per_class)
        result = dict(state_sha256=state_hash(state, self.keys), metrics=metrics,
                      seconds=time.perf_counter() - started,
                      forward_images=sum(len(value['labels']) for value in arrays.values()))
        write_json(complete, result)
        print('  EVAL %s overall=%.4f tail=%.4f' % (identity, metrics['test']['overall'], metrics['test']['tail20']), flush=True)
        return result

    def train_node(self, node, before):
        factor, kind = node['factor'], node['kind']
        active = self.a_keys if factor == 'A' else self.b_keys
        epochs = 3 if kind == 'normal' else 1
        wd = .0005 if kind == 'normal' else 0.
        train_only(self.model, factor)
        parameters = [p for p in self.model.parameters() if p.requires_grad]
        # Both doses of extra updates use the anchor's fixed client order.
        schedule_round = self.round + node['h'] if kind == 'normal' else self.round
        selected = list(map(int, self.source['meta']['schedule'][schedule_round - 1]))
        local_states, budgets, local_feedback = {}, [], []
        started = time.perf_counter()
        for client in selected:
            self.restore(before)
            seed = phase_seed(self.args.seed, self.round, node, client)
            optim = torch.optim.SGD(parameters, lr=.001, momentum=.9, weight_decay=wd)
            steps = samples = 0
            start = time.perf_counter()
            if self.device.type == 'cuda':
                torch.cuda.reset_peak_memory_stats(self.device)
            with isolated_rng(seed):
                self.model.train()
                loader = self.data['loader'](self.data['clients'][client], True)
                for epoch in range(epochs):
                    for batch in loader:
                        images, labels = batch['img'].to(self.device), batch['label'].to(self.device)
                        self.step(self.model, optim, None, 'fp32', images, labels,
                                  logit_adjustment=self.adjustment)
                        steps += 1
                        samples += len(labels)
            local_states[client] = {k: self.model.state_dict()[k].detach().cpu().clone() for k in active}
            local_after = dict(before, **local_states[client])
            budgets.append(dict(client_id=client, seed=seed, epochs=epochs, optimizer_steps=steps,
                                samples=samples, seconds=time.perf_counter() - start,
                                effective_norm=effective_norm(before, local_after, self.a_keys, self.scales),
                                peak_cuda_mib=torch.cuda.max_memory_allocated(self.device) / 2**20 if self.device.type == 'cuda' else 0.))
            if kind == 'extra':
                evaluation_started = time.perf_counter()
                value = self.evaluate_loader(self.data['local_feedback'][client])
                local_feedback.append(dict(client_id=client, indices=self.data['feedback_indices'][client],
                                           seconds=time.perf_counter() - evaluation_started, arrays=value))
            print('  %s client=%02d/30 steps=%d' % (node['id'], client + 1, steps), flush=True)
        after = aggregate(before, local_states, selected, self.weights, active, kind)
        inactive = set(self.keys) - set(active)
        if any(not torch.equal(before[k], after[k]) for k in inactive):
            raise ValueError('Frozen factor changed during aggregation')
        expected_steps = epochs * self.steps_epoch
        if sum(row['optimizer_steps'] for row in budgets) != expected_steps:
            raise ValueError('Incorrect local training budget')
        if any(not torch.isfinite(v).all() for v in after.values()):
            raise FloatingPointError('Nonfinite aggregate')
        return dict(node=node, before_sha256=state_hash(before, self.keys), after=after,
                    after_sha256=state_hash(after, self.keys), budget=budgets,
                    local_feedback=local_feedback, selected=selected,
                    effective_norm=effective_norm(before, after, self.a_keys, self.scales),
                    seconds=time.perf_counter() - started,
                    upload_bytes=sum(v.numel() * v.element_size() for k in selected for v in local_states[k].values()),
                    modeled_downlink_bytes=30 * sum(v.numel() * v.element_size() for v in before.values()))

    def node(self, node, before):
        path = self.root / 'nodes' / (node['id'] + '.pt')
        if path.is_file():
            result = load_torch(path)
            if result['node'] != node or result['before_sha256'] != state_hash(before, self.keys):
                raise ValueError('Cached node no longer matches its parent')
            if result['after_sha256'] != state_hash(result['after'], self.keys):
                raise ValueError('Corrupted cached node')
            print('  REUSE ' + node['id'], flush=True)
        else:
            result = self.train_node(node, before)
            save_torch(path, result)
        self.evaluate(node['id'], result['after'])
        return result

    def normalized_probes(self):
        rows = []
        for a_node, b_node in paired_probes():
            if a_node['parent'] != b_node['parent']:
                raise ValueError('Norm probe branches must have identical parents')
            before = self.anchor if a_node['parent'] == 'anchor' else load_torch(self.root / 'nodes' / (a_node['parent'] + '.pt'))['after']
            pair = [load_torch(self.root / 'nodes' / (n['id'] + '.pt')) for n in (a_node, b_node)]
            norms = [effective_norm(before, p['after'], self.a_keys, self.scales) for p in pair]
            target = min(norms)
            for node, payload, norm in zip((a_node, b_node), pair, norms):
                if target <= 1e-12:
                    rows.append(dict(h=node['h'], factor=node['factor'], comparable=False,
                                     reason='zero/tiny paired norm', raw_norm=norm, target=target))
                    continue
                ratio = target / norm
                after = {k: before[k] + ratio * (payload['after'][k] - before[k]) for k in self.keys}
                actual = effective_norm(before, after, self.a_keys, self.scales)
                error = abs(actual - target) / target
                identity = 'matched_h%02d_%s' % (node['h'], node['factor'])
                evaluated = self.evaluate(identity, after)
                row = dict(h=node['h'], factor=node['factor'], comparable=error <= .001,
                           raw_norm=norm, target=target, actual_norm=actual, multiplier=ratio,
                           relative_error=error, parent_sha256=state_hash(before, self.keys))
                for domain, metrics in evaluated['metrics'].items():
                    row.update({domain + '_' + k: v for k, v in metrics.items()})
                rows.append(row)
        write_csv(self.root / 'norm_probes.csv', rows)

    def run(self):
        self.evaluate('anchor', self.anchor)
        curve_rows, event_rows, budgets, local_rows = [], [], [], []
        seen = set()
        for arm in ARMS:
            before = self.anchor
            final = final_nodes(arm)
            arm_steps = 0
            print('ARM %s: %s round %d' % (arm, self.origin, self.round), flush=True)
            for node in plan(arm):
                payload = self.node(node, before)
                after = payload['after']
                arm_steps += sum(r['optimizer_steps'] for r in payload['budget'])
                evaluation = read_json(self.root / 'evaluations' / node['id'] / 'metrics.json')
                event_rows.append(dict(arm=arm, **node, before_sha256=payload['before_sha256'],
                                      after_sha256=payload['after_sha256'], effective_norm=payload['effective_norm']))
                if node == final[node['h']]:
                    row = dict(arm=arm, h=node['h'], node_id=node['id'])
                    for domain, metrics in evaluation['metrics'].items():
                        row.update({domain + '_' + key: value for key, value in metrics.items()})
                    curve_rows.append(row)
                if node['id'] not in seen:
                    seen.add(node['id'])
                    budgets.extend(dict(node_id=node['id'], h=node['h'], kind=node['kind'], factor=node['factor'], **r) for r in payload['budget'])
                    if payload['local_feedback']:
                        with np.load(self.root / 'evaluations' / node['parent'] / 'feedback.npz') as f:
                            parent_feedback = dict(f)
                        with np.load(self.root / 'evaluations' / node['id'] / 'feedback.npz') as f:
                            global_feedback = dict(f)
                        for local in payload['local_feedback']:
                            indices = np.asarray(local['indices'])
                            labels = local['arrays']['labels']
                            for c in sorted(set(labels.tolist())):
                                mask = labels == c
                                row = dict(node_id=node['id'], h=node['h'], factor=node['factor'],
                                           client_id=local['client_id'], class_id=c, samples=int(mask.sum()))
                                for metric in ('ce', 'la', 'margin'):
                                    row[metric + '_before'] = float(parent_feedback[metric][indices][mask].mean())
                                    row[metric + '_local'] = float(local['arrays'][metric][mask].mean())
                                    row[metric + '_aggregated'] = float(global_feedback[metric][indices][mask].mean())
                                local_rows.append(row)
                before = after
            if arm_steps != (3 * HORIZON + 2) * self.steps_epoch:
                raise ValueError('Unequal logical budget in ' + arm)
        self.normalized_probes()
        self.restore(self.anchor)
        final_frozen = state_hash(self.model.state_dict(), sorted(set(self.model.state_dict()) - set(self.keys)))
        if final_frozen != self.frozen_hash:
            raise ValueError('Frozen backbone/prompt buffers changed')
        write_csv(self.root / 'curves.csv', curve_rows)
        write_csv(self.root / 'events.csv', event_rows)
        write_csv(self.root / 'budget.csv', budgets)
        write_csv(self.root / 'local_feedback.csv', local_rows)
        write_json(self.root / 'complete.json', dict(version=PROTOCOL['version'], arms=list(ARMS),
                   curve_rows=len(curve_rows), unique_training_nodes=len(seen),
                   actual_optimizer_steps=sum(r['optimizer_steps'] for r in budgets),
                   logical_optimizer_steps_per_arm=(3 * HORIZON + 2) * self.steps_epoch,
                   shared_prefixes=True, primary_offsets=PROTOCOL['primary_offsets'],
                   frozen_state_unchanged=True))

    def smoke(self):
        """Real model forward/backward on two images, no formal branch artifacts."""
        batch = next(iter(self.data['feedback']))
        images, labels = batch['img'][:2].to(self.device), batch['label'][:2].to(self.device)
        records = []
        for factor in ('A', 'B'):
            self.restore(self.anchor)
            train_only(self.model, factor)
            params = [p for p in self.model.parameters() if p.requires_grad]
            opt = torch.optim.SGD(params, lr=.001, momentum=.9)
            with isolated_rng(self.args.seed):
                self.model.train()
                _, loss, _ = self.step(self.model, opt, None, 'fp32', images, labels, logit_adjustment=self.adjustment)
            after = {k: self.model.state_dict()[k].detach().cpu().clone() for k in self.keys}
            frozen = self.b_keys if factor == 'A' else self.a_keys
            if any(not torch.equal(after[k], self.anchor[k]) for k in frozen):
                raise ValueError('Smoke test changed frozen factor')
            norm = effective_norm(self.anchor, after, self.a_keys, self.scales)
            if not math.isfinite(norm) or norm <= 0:
                raise ValueError('Smoke test produced zero/nonfinite update')
            records.append(dict(factor=factor, loss=float(loss.detach()), effective_norm=norm))
        self.restore(self.anchor)
        write_json(self.root / 'smoke.json', dict(passed=True, records=records, formal_training=False))
        print('GPU smoke passed; formal branches have NOT run.', flush=True)


def register_request(root, request):
    """Allow a code fix after failed startup, never across saved model/evaluation work.

    Called only while holding the anchor's worker lock. The old import failure
    left request.json before any Worker artifacts existed; preserve that record.
    """
    root = Path(root)
    path = root / 'request.json'
    if path.exists():
        previous = read_json(path)
        if previous != request:
            changed = {key for key in previous.keys() | request.keys()
                       if previous.get(key) != request.get(key)}
            artifacts = [p for p in root.iterdir()
                         if not (p.is_file() and (p.name in ('request.json', 'worker.lock')
                                 or (p.name.startswith('request_before_startup_fix_') and p.suffix == '.json')))]
            if changed != {'code_fingerprint'} or artifacts:
                raise ValueError('Existing output has a different protocol/code/source. Choose a new --output-root.')
            backup = root / ('request_before_startup_fix_%s.json' % digest(previous)[:16])
            write_json(backup, previous)
            print('Recovered startup-only request after code fix; previous request saved to ' + str(backup), flush=True)
    write_json(path, request)


def run_worker(args):
    from scripts.run_ab_validation import file_lock
    source = check_source(args.source_run, args.origin, args.seed, [args.anchor_round])
    root = job_root(args.output_root, args.seed, args.origin, args.anchor_round)
    with file_lock(root / 'worker.lock', timeout=.1):
        request = dict(protocol=PROTOCOL, seed=args.seed, origin=args.origin, anchor_round=args.anchor_round,
                       source_run=str(Path(args.source_run).resolve()), partition_sha256=source['partition_sha256'],
                       source_metadata_sha256=file_digest(Path(args.source_run) / 'bridge_metadata.json'),
                       source_base_sha256=file_digest(Path(args.source_run) / 'checkpoints/base_model.pt'),
                       event_sha256=file_digest(source_event(args.source_run, args.origin, args.anchor_round)),
                       data_root=str(args.data_root.resolve()), num_workers=args.num_workers,
                       eval_batch_size=args.eval_batch_size, code_fingerprint=args.code_fingerprint)
        register_request(root, request)
        if args.worker_mode == 'run' and (root / 'complete.json').is_file():
            print('SKIP completed anchor: ' + str(root), flush=True)
            return
        worker = Worker(args, source, args.origin, args.anchor_round, root)
        if args.worker_mode == 'smoke':
            worker.smoke()
        else:
            worker.run()
