"""One common anchor, eight local-training branches, sixteen reaggregations.

Only saved S states and partition manifests are read. No baseline training,
partition regeneration, checkpoint hashes, controller, or test-based selection.
"""
import math
import pickle
import time
from types import SimpleNamespace

import numpy as np
import torch

from tools.factor_roles.common import PROTOCOL, TOPOLOGIES, read_csv, read_json, write_csv, write_json
from utils.cliplora_a_refresh import isolated_rng, lora_keys, product_norm_squared, train_only


def load_sources(args):
    sources = {}
    for topology, path in zip(TOPOLOGIES, (args.clt_run, args.dirichlet_run)):
        path = path.resolve()
        if not (path / 'bridge_metadata.json').is_file():
            raise FileNotFoundError(
                f'Missing S run: {path}. Supply --clt-run / --dirichlet-run. '
                'The ordinary-Dirichlet S run must exist; matched-Dirichlet is not a substitute.')
        meta = read_json(path / 'bridge_metadata.json')
        cfg = read_json(path / 'control_config.json')
        assert meta['topology'] == cfg['topology'] == topology, 'Wrong partition: no matched-Dirichlet inputs'
        assert cfg['method'] == 's' and cfg['seed'] == cfg['protocol_seed'] == 42
        assert cfg['la_tau'] == cfg['a_lr_mult'] == 1
        if topology == TOPOLOGIES[1]:
            assert meta['resolved_args']['beta'] == .5
        manifest = read_csv(path / 'partition_manifest.csv')
        rows = sorted(manifest, key=lambda r: (int(r['client_id']), int(r['local_position'])))
        counts = torch.zeros(30, 100, dtype=torch.long)
        for row in rows:
            counts[int(row['client_id']), int(row['class_id'])] += 1
        assert bool((counts.sum(1) > 0).all()) and int(counts.sum()) == 10847
        sources[topology] = dict(path=path, meta=meta, config=cfg, rows=rows, counts=counts)
    clt, dr = (sources[t] for t in TOPOLOGIES)
    assert sorted(int(r['raw_sample_id']) for r in clt['rows']) == sorted(int(r['raw_sample_id']) for r in dr['rows'])
    assert torch.equal(clt['counts'].sum(0), dr['counts'].sum(0))
    assert clt['meta']['classnames'] == dr['meta']['classnames']
    return sources


def build_cfg(source, args):
    from federated_main import setup_cfg
    resolved = SimpleNamespace(**source['meta']['resolved_args'])
    resolved.root = str(args.data_root.resolve())
    resolved.output_dir = str(args.output_root.resolve())
    cfg = setup_cfg(resolved)
    cfg.defrost()
    cfg.DATALOADER.NUM_WORKERS = args.num_workers
    cfg.MODEL.INIT_WEIGHTS = ''  # Full source base state is loaded below.
    cfg.freeze()
    assert cfg.TRAINER.CLIPLORA.r == 4 and cfg.TRAINER.CLIPLORA.alpha == 1
    assert cfg.TRAINER.CLIPLORA.encoder == 'vision' and cfg.TRAINER.CLIPLORA.position == 'top3'
    assert list(cfg.TRAINER.CLIPLORA.params) == ['q', 'v']
    assert cfg.TRAINER.CLIPLORA.dropout_rate == 0 and cfg.TRAINER.COOP.PREC == 'fp32'
    return cfg


def make_data(cfg, sources, args):
    from Dassl.dassl.data.data_manager import build_data_loader
    from Dassl.dassl.data.transforms import build_transform
    from utils.functional_coverage_validation import _TrainOnlyCifar100, _locate_cifar100
    data_dir = _locate_cifar100(args.data_root.resolve())
    store = _TrainOnlyCifar100(data_dir)
    train_tfm = build_transform(cfg, is_train=True)
    test_tfm = build_transform(cfg, is_train=False)
    clients = {}
    for topology, source in sources.items():
        clients[topology] = [[] for _ in range(30)]
        for row in source['rows']:
            raw, label = int(row['raw_sample_id']), int(row['class_id'])
            assert int(store.labels[raw]) == label
            clients[topology][int(row['client_id'])].append(
                SimpleNamespace(data=store.images[raw], label=label, domain=0))

    # Use the same CIFAR wrapper as S, including its actual normalization/resize.
    def loader(items, training):
        return build_data_loader(cfg, sampler_type='RandomSampler' if training else 'SequentialSampler',
            data_source=items, batch_size=PROTOCOL['batch_size'] if training else 64,
            tfm=train_tfm if training else test_tfm, is_train=training, drop_last=False)

    with (data_dir / 'test').open('rb') as stream:
        test = pickle.load(stream, encoding='latin1')
    images = np.asarray(test['data'], dtype=np.uint8).reshape(-1, 3, 32, 32).transpose(0, 2, 3, 1)
    test_items = [SimpleNamespace(data=x, label=int(y), domain=0) for x, y in zip(images, test['fine_labels'])]
    clt = sources[TOPOLOGIES[0]]
    probe = read_csv(clt['path'] / 'protocol/probe_manifest.csv')
    other_probe = read_csv(sources[TOPOLOGIES[1]]['path'] / 'protocol/probe_manifest.csv')
    assert probe == other_probe
    train_ids = {int(r['raw_sample_id']) for r in clt['rows']}
    assert not (train_ids & {int(r['raw_train_index']) for r in probe})
    probe_items = [SimpleNamespace(data=store.images[int(r['raw_train_index'])],
                                  label=int(r['class_id']), domain=0) for r in probe]
    groups = {name: clt['config'][key] for name, key in
              [('head20', 'head_ids'), ('middle60', 'middle_ids'), ('tail20', 'tail_ids')]}
    groups['nontail80'] = sorted(groups['head20'] + groups['middle60'])
    assert len(probe) == 200 and {x.label for x in probe_items} == set(groups['tail20'])
    return clients, loader, loader(test_items, False), loader(probe_items, False), groups


def effective_norm(anchor, delta, factor, scales):
    squared = 0.
    for key, value in delta.items():
        a_key = key[:-1] + 'A'
        if factor == 'A':
            left, right = anchor[key[:-1] + 'B'].double(), value.double()
        else:
            left, right = value.double(), anchor[a_key].double()
        squared += scales[a_key] ** 2 * product_norm_squared(left, right)
    return math.sqrt(squared)


def endpoint(anchor, delta, scale=1.):
    return {key: (value + delta[key] * scale if key in delta else value) for key, value in anchor.items()}


def local_branch(model, anchor, topology, factor, loss, clients, loader, prior,
                 rnd, device, cache, scales):
    if cache.is_file():
        print(f'  reuse completed branch: {topology}/{factor}-{loss}', flush=True)
        return torch.load(cache, map_location='cpu', weights_only=False)
    from trainers.cliplora import cliplora_optimizer_step
    train_only(model, factor)
    keys = lora_keys(model, factor)
    params = [p for p in model.parameters() if p.requires_grad]
    uploads, budgets = [], []
    adjustment = (PROTOCOL['la_tau'] * prior.log()).float().to(device) if loss == 'LA' else None
    for client, items in enumerate(clients):
        started = time.perf_counter()
        # No client inherits another client's parameters or optimizer state.
        model.load_state_dict(anchor, strict=False)
        optimizer = torch.optim.SGD(params, lr=PROTOCOL['lr'], momentum=PROTOCOL['momentum'],
                                    weight_decay=PROTOCOL['weight_decay'])
        steps = samples = 0
        seed = PROTOCOL['seed'] + 1000003 * rnd + 1009 * client + 10000019 * TOPOLOGIES.index(topology)
        if str(device).startswith('cuda'):
            torch.cuda.reset_peak_memory_stats(device)
        # Factor, loss, aggregation, and anchor origin do NOT enter this seed.
        with isolated_rng(seed):
            model.train()
            for batch in loader(items, True):
                images, labels = batch['img'].to(device), batch['label'].to(device)
                cliplora_optimizer_step(model, optimizer, None, 'fp32', images, labels,
                                        logit_adjustment=adjustment)
                steps += 1
                samples += labels.numel()
        delta = {key: model.state_dict()[key].detach().cpu().clone() - anchor[key] for key in keys}
        uploads.append(delta)
        budgets.append(dict(client_id=client, optimizer_steps=steps, samples=samples,
            effective_delta_norm=effective_norm(anchor, delta, factor, scales),
            parameter_delta_norm=math.sqrt(sum(float(v.double().square().sum()) for v in delta.values())),
            seconds=time.perf_counter() - started,
            peak_cuda_mb=torch.cuda.max_memory_allocated(device) / 2**20 if str(device).startswith('cuda') else 0.))
        print(f'    {topology} {factor}-{loss}: client {client+1}/30, steps={steps}', flush=True)
    payload = dict(topology=topology, factor=factor, loss=loss, local_deltas=uploads, budget=budgets,
                   trainable_parameters=sum(p.numel() for p in params))
    cache.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, cache)
    return payload


def aggregate(anchor, branch, sizes, weighting):
    weights = sizes.double() / sizes.sum() if weighting == 'sample' else torch.full((30,), 1/30, dtype=torch.float64)
    # Aggregate in double, then use the actual stored float32 endpoint change.
    delta = {}
    for key in branch['local_deltas'][0]:
        value = sum(float(q) * client[key].double() for q, client in zip(weights, branch['local_deltas']))
        delta[key] = (anchor[key] + value.to(anchor[key])) - anchor[key]
    return delta


@torch.no_grad()
def evaluate(model, state, test_loader, probe_loader, groups, device):
    model.load_state_dict(state, strict=False)
    model.eval()
    correct, total = torch.zeros(100, dtype=torch.long), torch.zeros(100, dtype=torch.long)
    psum, pn = torch.zeros(100, dtype=torch.float64), torch.zeros(100, dtype=torch.long)
    with isolated_rng(42):
        for batch in test_loader:
            labels = batch['label']
            logits = model(batch['img'].to(device)).cpu()
            total += torch.bincount(labels, minlength=100)
            correct += torch.bincount(labels[logits.argmax(1) == labels], minlength=100)
        for batch in probe_loader:
            labels = batch['label']
            logits = model(batch['img'].to(device)).cpu()
            target = logits.gather(1, labels[:, None]).squeeze(1)
            logits.scatter_(1, labels[:, None], -torch.inf)
            logodds = target - logits.logsumexp(1)
            psum.scatter_add_(0, labels, logodds.double())
            pn += torch.bincount(labels, minlength=100)
    acc = 100 * correct.double() / total
    metrics = {'overall': 100 * float(correct.sum()) / int(total.sum())}
    metrics.update({name: float(acc[ids].mean()) for name, ids in groups.items()})
    metrics['probe_tail_logodds'] = float((psum[groups['tail20']] / pn[groups['tail20']]).mean())
    rows = [dict(class_id=c, correct=int(correct[c]), total=int(total[c]), accuracy=float(acc[c]),
                 probe_logodds=float(psum[c]/pn[c]) if pn[c] else None) for c in range(100)]
    return dict(metrics=metrics, per_class=rows)


def run_anchor(args, sources, origin, rnd):
    root = args.output_root.resolve() / origin / f'round_{rnd:03d}'
    root.mkdir(parents=True, exist_ok=True)
    source = sources[origin]
    request = dict(protocol=PROTOCOL, origin=origin, round=rnd, num_workers=args.num_workers,
                   source_runs={t: str(s['path']) for t, s in sources.items()})
    request_file = root / 'request.json'
    if request_file.exists():
        assert read_json(request_file) == request, 'Existing cache has another setup; use a new output root'
    write_json(request_file, request)
    if (root / 'complete.json').is_file():
        print(f'SKIP completed anchor: {origin}/round_{rnd:03d}', flush=True)
        return
    event = source['path'] / 'events' / f'r{rnd:03d}_c000_main_refresh_A' / 'state.pt'
    if not event.is_file() or not (source['path'] / 'checkpoints/base_model.pt').is_file():
        raise FileNotFoundError(f'Need full server-side S states, not the lightweight analysis archive: {event}')
    saved = torch.load(event, map_location='cpu', weights_only=False)
    anchor = saved['actual_after_lora_state']
    assert saved['round'] == rnd and saved['phase'] == 'refresh_A'
    assert all(float(v.double().norm()) > 0 for k, v in anchor.items() if k.endswith('_lora_B')), 'A-only needs trained nonzero B'
    cfg = build_cfg(source, args)
    from trainers.cliplora import build_cliplora_model
    model = build_cliplora_model(cfg, source['meta']['classnames']).to(args.device)
    base = torch.load(source['path'] / 'checkpoints/base_model.pt', map_location='cpu', weights_only=False)
    base.update(anchor)
    model.load_state_dict(base, strict=True)
    del base
    a_keys, b_keys = lora_keys(model, 'A'), lora_keys(model, 'B')
    assert set(a_keys + b_keys) == set(anchor)
    assert sum(anchor[k].numel() for k in a_keys) == sum(anchor[k].numel() for k in b_keys) == 18432
    modules = dict(model.named_modules())
    scales = {k: float(modules[k.rsplit('.', 1)[0]].scaling) for k in a_keys}
    clients, loader, test_loader, probe_loader, groups = make_data(cfg, sources, args)
    counts = sources[TOPOLOGIES[0]]['counts'].sum(0)
    prior = counts.double() / counts.sum()
    candidates, budgets, exposure = [], [], []
    for topology in TOPOLOGIES:
        sizes = sources[topology]['counts'].sum(1)
        for c in range(100):
            local_counts = sources[topology]['counts'][:, c].double()
            exposure.append(dict(topology=topology, class_id=c, global_prior=float(prior[c]),
                sample_weighted_class_mass=float(local_counts.sum()/sizes.sum()),
                uniform_weighted_class_mass=float((local_counts/sizes).mean())))
        for factor in ('A', 'B'):
            for loss in ('CE', 'LA'):
                branch = local_branch(model, anchor, topology, factor, loss, clients[topology], loader,
                    prior, rnd, args.device, root/'updates'/f'{topology}_{factor}_{loss}.pt', scales)
                assert sum(row['optimizer_steps'] for row in branch['budget']) == sum(math.ceil(int(n)/PROTOCOL['batch_size']) for n in sizes)
                budgets.extend(dict(topology=topology, factor=factor, loss=loss, **row) for row in branch['budget'])
                for weighting in ('sample', 'uniform'):
                    delta = aggregate(anchor, branch, sizes, weighting)
                    candidates.append(dict(topology=topology, factor=factor, loss=loss, weighting=weighting,
                        delta=delta, effective_norm=effective_norm(anchor, delta, factor, scales)))
    # Select this length BEFORE any official test evaluation; never tune on accuracy.
    positive = [c['effective_norm'] for c in candidates if c['effective_norm'] > 0]
    target = min(positive) if positive else 0.
    write_csv(root/'budget.csv', budgets)
    write_csv(root/'class_exposure.csv', exposure)
    write_json(root/'anchor_info.json', dict(origin=origin, round=rnd, source_event=str(event),
        groups=groups, class_prior=prior.tolist(), client_sample_counts={t:s['counts'].sum(1).tolist() for t,s in sources.items()},
        a_parameters=18432, b_parameters=18432, scales=scales, norm_target=target,
        steps=sum(b['optimizer_steps'] for b in budgets),
        input_transform='existing DatasetCifar100 wrapper for train, test, and probe',
        probe_scope='fixed 200 held-out train images, tail20 only; no probe-based tuning'))
    baseline_file = root/'evaluations/anchor.json'
    if not baseline_file.exists():
        write_json(baseline_file, evaluate(model, anchor, test_loader, probe_loader, groups, args.device))
    baseline = read_json(baseline_file)
    rows, per_class = [], []
    for number, candidate in enumerate(candidates, 1):
        identity = {k:candidate[k] for k in ('topology', 'factor', 'loss', 'weighting')}
        for mode in ('raw', 'norm_matched'):
            norm = candidate['effective_norm']
            multiplier = target / norm if mode == 'norm_matched' and norm > 0 else 1.
            state = endpoint(anchor, candidate['delta'], multiplier)
            actual_delta = {k:state[k]-anchor[k] for k in candidate['delta']}
            effective = effective_norm(anchor, actual_delta, candidate['factor'], scales)
            path = root/'evaluations'/('_'.join(identity.values()) + f'_{mode}.json')
            print(f'  evaluate {number}/16 {identity} {mode}', flush=True)
            if not path.exists():
                write_json(path, evaluate(model, state, test_loader, probe_loader, groups, args.device))
            result = read_json(path)
            row = dict(origin=origin, round=rnd, mode=mode, **identity, raw_effective_norm=norm,
                effective_norm=effective, norm_target=target, multiplier=multiplier,
                norm_comparable=norm > 0 and effective > 0,
                norm_relative_error=abs(effective-target)/target if mode == 'norm_matched' and target else None,
                **result['metrics'])
            row.update({f'delta_anchor_{k}':v-baseline['metrics'][k] for k,v in result['metrics'].items()})
            rows.append(row)
            for current, before in zip(result['per_class'], baseline['per_class']):
                per_class.append(dict(origin=origin, round=rnd, mode=mode, **identity, **current,
                    delta_accuracy=current['accuracy']-before['accuracy'],
                    delta_probe_logodds=current['probe_logodds']-before['probe_logodds'] if current['probe_logodds'] is not None else None))
    write_csv(root/'candidate_metrics.csv', rows)
    write_csv(root/'per_class.csv', per_class)
    write_json(root/'complete.json', dict(origin=origin, round=rnd, training_branches=8, raw_candidates=16,
        matched_candidates=16, optimizer_steps=sum(b['optimizer_steps'] for b in budgets)))
    print(f'DONE {origin} round {rnd}: {root}', flush=True)


def run(args):
    sources = load_sources(args)
    origins = TOPOLOGIES if args.anchor_origin == 'both' else (args.anchor_origin,)
    jobs = [(origin, rnd) for origin in origins for rnd in sorted(set(args.rounds))]
    for number, (origin, rnd) in enumerate(jobs, 1):
        print(f'\n=== Anchor {number}/{len(jobs)}: {origin}, completed S round {rnd} ===', flush=True)
        run_anchor(args, sources, origin, rnd)
    print(f'Run finished. Summarize with: python -u scripts/run_cliplora_factor_roles.py --stage summary --output-root {args.output_root}', flush=True)
