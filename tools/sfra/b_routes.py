"""Portable route contracts, source preflight and audited paired collection."""
import hashlib
import math
from pathlib import Path
import statistics
from collections import defaultdict

from tools.sfra.ab_validation import code_hashes as frozen_hashes
from tools.sfra.calibration_reference import protocol_signature
from tools.sfra.maintext import FROZEN, digest, load_json
from tools.sfra.summary import METRICS, read_csv, write_csv


SCHEMA = 'b_routes_experiment_v1'
ARMS = ('N', 'P', 'M', 'S', 'C', 'F', 'P-rot', 'C-rot')
EXTRA_FILES = ('utils/b_route_math.py', 'utils/b_route_sampling.py', 'utils/cliplora_b_routes.py',
               'utils/cliplora_b_route_runtime.py', 'utils/b_route_replay.py',
               'scripts/run_cliplora_b_routes.py', 'scripts/train_cliplora_b_routes.py',
               'tools/sfra/b_routes.py')


def source_hashes(repo):
    hashes = frozen_hashes(repo)
    hashes.update({name: hashlib.sha256((Path(repo)/name).read_text(encoding='utf-8').encode()).hexdigest()
                   for name in EXTRA_FILES})
    return hashes


def check_sources(spec, repo):
    if spec.get('schema_version') != SCHEMA or spec.get('code_sha256') != source_hashes(repo):
        raise ValueError('Route code/contract changed; use the original code or a new output root')


def file_hash(path):
    value = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            value.update(block)
    return value.hexdigest()


def replay_files(rounds, include_legacy=False):
    names = ['sfra_config.json', 'command.json', 'bridge_metadata.json', 'partition_manifest.csv',
             'private_witness_manifest.json', 'execution_config.json', 'checkpoints/base_model.pt',
             'protocol/full_schedule.json', 'protocol/eri_protocol.json', 'protocol/probe_manifest.csv']
    for rnd in rounds:
        names += [f'events/r{rnd:03d}_c000_main_normal_B/{n}' for n in ('state.pt', 'event.json')]
        if include_legacy:
            names += [f'b_transfer_rounds/r{rnd:03d}/{n}' for n in
                      ('commit.pt', 'matrices.npz', 'calibration_manifest.json', 'probe_metrics.csv')]
    return names


def replay_preflight(source, rounds, include_legacy=False, hash_files=False):
    source = Path(source)
    if not rounds or len(set(rounds)) != len(rounds) or any(r not in range(30, 101, 10) for r in rounds):
        raise ValueError('Replay requires unique scheduled B events')
    names = replay_files(rounds, include_legacy)
    missing = [n for n in names if not (source/n).is_file()]
    errors = []
    if (source/'sfra_config.json').is_file():
        cfg = load_json(source/'sfra_config.json')
        for key, expected in {**FROZEN, **dict(variant='full-cp', classification_weight=1.,
                                  partition='client-longtail', protocol_seed=42)}.items():
            if cfg.get(key) != expected:
                errors.append('Source setting differs: '+key)
        if cfg.get('b_aggregation', {}).get('mode', 'sample') != 'sample':
            errors.append('Source must use sample-weighted ordinary B')
        if include_legacy:
            b = cfg.get('b_transfer', {})
            if any(b.get(k) != v for k, v in dict(mode='shared', calibration_profile='coverage_tradeoff',
                    non_tail_sampling='class-cyclic', tail_weight=.35, learning_rate=.3, regularization=.001, steps=2).items()):
                errors.append('Legacy equivalence needs the original shared tradeoff AB')
        elif cfg.get('b_transfer', {}).get('calibration_profile') not in (None, 'coverage_tradeoff', 'routes'):
            errors.append('Source must be A-only, original shared tradeoff AB, or a registered route')
    if (source/'execution_config.json').is_file():
        execution = load_json(source/'execution_config.json')
        if any(execution.get(k) != v for k, v in dict(version=2, feedback_forward_batch_size=128, device_cache_gib=4.).items()):
            errors.append('Replay source must use the frozen fast-v2 f128 c4 execution mode')
    return dict(ready=not missing and not errors, missing=missing, errors=errors,
                source_sha256={n: file_hash(source/n) for n in names} if hash_files and not missing else {})


def validate_settings(settings, full=True):
    if settings['arm'] not in ARMS or settings['seed'] < 0 or settings.get('protocol_seed', 42) != 42:
        raise ValueError('Invalid route arm/seed/protocol')
    rho = settings.get('rho', 1.)
    if not math.isfinite(rho) or rho <= 0 or settings.get('null_index', 0) < 0:
        raise ValueError('Invalid rho/random-rotation index')
    if full and (rho != 1. or settings.get('null_index', 0) != 0):
        raise ValueError('Full runs freeze rho=1 and null_index=0; use replay for sensitivities')


def validate_profile(cfg, spec):
    if spec.get('schema_version') != SCHEMA or spec.get('mode') != 'full':
        raise ValueError('Wrong full-run schema/mode')
    for k, v in {**FROZEN, 'variant': 'full-cp', 'classification_weight': 1.}.items():
        if cfg.get(k) != v:
            raise ValueError('Method A setting differs: '+k)
    s = spec['settings']
    validate_settings(s)
    if cfg.get('b_route_experiment') != spec:
        raise ValueError('Runtime route contract differs')
    for k in ('seed', 'protocol_seed', 'partition'):
        if cfg.get(k) != s[k]:
            raise ValueError('Runtime route setting differs: '+k)
    b = cfg.get('b_transfer')
    if s['arm'] == 'N':
        if b:
            raise ValueError('N must not execute B transfer')
    elif not b or any(b.get(k) != v for k, v in dict(calibration_profile='routes', route_arm=s['arm'],
            route_rho=1., route_null_index=0, steps=2, regularization=0., tail_weight=.35,
            non_tail_sampling='class-cyclic').items()):
        raise ValueError('Route transfer configuration differs')


def finite(value):
    x = float(value)
    if not math.isfinite(x):
        raise ValueError('Nonfinite result')
    return x


def read_run(run, spec):
    cfg, done = load_json(run/'sfra_config.json'), load_json(run/'completion.json')
    validate_profile(cfg, spec)
    execution = load_json(run/'execution_config.json')
    if any(execution.get(k) != v for k, v in dict(version=2, feedback_forward_batch_size=128, device_cache_gib=4.).items()):
        raise ValueError('Execution mode differs from frozen fast-v2 f128 c4')
    receipt = load_json(run/'route_receipt.json')
    if receipt.get('spec_digest') != digest(spec) or not receipt.get('code_unchanged'):
        raise ValueError('Missing/mismatched route receipt')
    if done.get('completed_round') != 100 or done.get('official_test_passes') != 101:
        raise ValueError('Incomplete 100-round result')
    for actual, expected in [('normal_optimizer_steps', 'normal_steps_expected'),
                             ('extra_optimizer_steps', 'extra_steps_expected')]:
        if done.get(actual) != cfg['base_training_config'].get(expected):
            raise ValueError('Local optimizer budget differs')
    if done.get('functional_correction_steps') != 270:
        raise ValueError('A correction budget differs')
    rows = sorted(read_csv(run/'round_metrics.csv'), key=lambda r: int(r['round']))
    if [int(r['round']) for r in rows] != list(range(101)):
        raise ValueError('Missing/duplicate official rounds')
    for row in rows:
        if any(not 0 <= finite(row[m]) <= 100 for m in METRICS):
            raise ValueError('Accuracy outside 0..100')
    prior = load_json(run/'class_prior.json')
    tail = set(prior['tail_ids'])
    head = set(sorted(range(100), key=lambda c: (-prior['counts'][c], c))[:20])
    if (len(tail) != 20 or len(prior['counts']) != 100 or tail & head or
            tail != set(sorted(range(100), key=lambda c: (prior['counts'][c], -c))[:20])):
        raise ValueError('Fixed class groups differ')
    groups = dict(overall_acc=set(range(100)), head20_acc=head, bottom20_tail_acc=tail,
                  middle60_acc=set(range(100))-head-tail, non_tail_acc=set(range(100))-tail)
    for rnd in range(81, 101):
        table = read_csv(run/f'per_class_accuracy_epoch_{rnd-1}.csv')
        if len(table) != 100 or {int(r['class_id']) for r in table} != set(range(100)):
            raise ValueError('Incomplete per-class table')
        values = {int(r['class_id']): finite(r['per_class_acc']) for r in table}
        if any(not 0 <= v <= 100 for v in values.values()):
            raise ValueError('Invalid per-class accuracy')
        for name, ids in groups.items():
            if not math.isclose(statistics.mean(values[c] for c in ids), float(rows[rnd][name]), abs_tol=1e-8):
                raise ValueError('Per-class table differs from round metric')
    s = spec['settings']
    result = dict(run=str(run), arm=s['arm'], seed=s['seed'],
                  cohort='development' if s['seed'] == 42 else 'paired-replication',
                  elapsed_seconds=finite(done['elapsed_seconds']))
    for m in METRICS:
        result['last20_'+m] = statistics.mean(float(r[m]) for r in rows[81:])
        result['final_'+m] = float(rows[-1][m])
    result['tail_peak_to_final'] = max(float(r['bottom20_tail_acc']) for r in rows)-float(rows[-1]['bottom20_tail_acc'])
    if s['arm'] != 'N':
        events = read_csv(run/'b_transfer_rounds.csv')
        if sorted(int(e['round']) for e in events) != list(range(30, 101, 10)):
            raise ValueError('Missing transfer events')
        if done.get('b_transfer_events') != 8 or done.get('b_transfer_optimizer_steps') != 16:
            raise ValueError('Transfer completion budget differs')
        for e in events:
            folder = run/'b_transfer_rounds'/f'r{int(e["round"]):03d}'
            steps = read_csv(folder/'optimization_steps.csv')
            if [int(x['step']) for x in steps] != [1, 2]:
                raise ValueError('Expected two route steps')
            if e.get('route_arm') != s['arm']:
                raise ValueError('Mixed route arms')
            for x in steps:
                if finite(x['effective_transfer_norm_after']) > finite(x['radius'])*(1+2e-6)+1e-12:
                    raise ValueError('Effective norm exceeds registered radius')
        for field in ('algorithm_forward_images', 'algorithm_backward_images', 'diagnostic_forward_images',
                      'extra_upload_bytes', 'extra_downlink_bytes', 'seconds'):
            result['B_'+field] = sum(finite(e[field]) for e in events)
    signature = protocol_signature(run)
    core = frozen_hashes(Path(__file__).resolve().parents[2])
    signature.update(base_training=digest(cfg['base_training_config']), tail_ids=digest(sorted(tail)),
                     frozen_source=digest({k: spec['code_sha256'][k] for k in core}))
    return result, signature, rows


def pair_check(a, b):
    ra, sa, ca = a
    rb, sb, cb = b
    if sa != sb or sa.get('frozen_source') is None:
        raise ValueError('Initialization/data/witness/execution/source pairing differs')
    for x, y in zip(ca[:30], cb[:30]):
        if any(abs(float(x[m])-float(y[m])) > 1e-8 for m in METRICS):
            raise ValueError('Pre-transfer official trajectory differs')
    if ra['arm'] != 'N' and rb['arm'] != 'N':
        for rnd in range(30, 101, 10):
            ma = load_json(Path(ra['run'])/f'b_transfer_rounds/r{rnd:03d}/calibration_manifest.json')
            mb = load_json(Path(rb['run'])/f'b_transfer_rounds/r{rnd:03d}/calibration_manifest.json')
            if (ma['feedback_clients'] != mb['feedback_clients'] or
                {k: v['batches'] for k, v in ma['recipients'].items()} !=
                {k: v['batches'] for k, v in mb['recipients'].items()}):
                raise ValueError('Feedback sample identities differ')


def summarize(root, baselines=()):
    root = Path(root)
    out = root/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    status, runs, pairs, audits = [], {}, [], []
    for path in sorted(root.glob('runs/*/seed*/route_spec.json')):
        run = path.parent
        spec = load_json(path)
        if not (run/'completion.json').is_file():
            status.append(dict(run=str(run), status='incomplete'))
            continue
        try:
            item = read_run(run, spec)
            key = item[0]['seed'], item[0]['arm']
            if key in runs:
                raise ValueError('Duplicate seed/arm result')
            runs[key] = item
            status.append(dict(run=str(run), status='valid'))
        except (ValueError, KeyError, OSError) as error:
            status.append(dict(run=str(run), status='invalid', error=str(error)))
    from tools.sfra.b_class_guard import read_run as read_baseline
    for path in baselines:
        try:
            r, sig, _, rows, _, _ = read_baseline(Path(path), 'a')
            if load_json(Path(path)/'sfra_config.json').get('b_transfer'):
                raise ValueError('Baseline must be A-only')
            r['arm'] = 'N'
            key = r['seed'], 'N'
            if key in runs:
                raise ValueError('Duplicate A-only baseline')
            runs[key] = r, sig, rows
            status.append(dict(run=str(path), status='valid-baseline'))
        except (ValueError, KeyError, OSError) as error:
            status.append(dict(run=str(path), status='invalid', error=str(error)))
    comparisons = [('P', 'N'), ('C', 'N'), ('F', 'N'), ('P', 'F'), ('C', 'F'), ('C', 'P'),
                   ('M', 'P'), ('S', 'M'), ('C', 'S'), ('P', 'P-rot'), ('C', 'C-rot')]
    for seed in sorted({k[0] for k in runs}):
        for left, right in comparisons:
            if (seed, left) not in runs or (seed, right) not in runs:
                continue
            a, b = runs[seed, left], runs[seed, right]
            try:
                pair_check(a, b)
                pairs.append(dict(seed=seed, comparison=left+' minus '+right,
                    **{'delta_'+m: a[0]['last20_'+m]-b[0]['last20_'+m] for m in METRICS}))
                audits.append(dict(seed=seed, comparison=left+' minus '+right, valid=True))
            except (ValueError, KeyError, OSError) as error:
                audits.append(dict(seed=seed, comparison=left+' minus '+right, valid=False, error=str(error)))
    write_csv(out/'status.csv', status)
    write_csv(out/'per_run.csv', [x[0] for x in runs.values()])
    write_csv(out/'paired.csv', pairs)
    write_csv(out/'pair_audit.csv', audits)
    lines = ['# B route comparison', '', 'Primary endpoint: mean committed Tail20 ACC over rounds 81–100.', '',
             '| Seed | Comparison | Tail20 difference (pp) | Overall difference (pp) |', '|---|---|---:|---:|']
    lines += [f'| {r["seed"]} | {r["comparison"]} | {r["delta_bottom20_tail_acc"]:.4f} | {r["delta_overall_acc"]:.4f} |' for r in pairs]
    lines += ['', 'Seed42 is development evidence. Seeds0/3407 are paired replications, not independent data partitions.',
              'Small differences do not establish statistical equivalence. Missing/invalid pairs are excluded, not filled with zero.']
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    paired_summary = []
    for comparison in sorted({r['comparison'] for r in pairs}):
        for cohort, seeds in [('development', {42}), ('replication', {0, 3407})]:
            subset = [r for r in pairs if r['comparison'] == comparison and r['seed'] in seeds]
            if subset:
                row = dict(comparison=comparison, cohort=cohort, seeds=','.join(str(r['seed']) for r in subset), n=len(subset))
                for metric in METRICS:
                    values = [r['delta_'+metric] for r in subset]
                    row.update({metric+'_mean': statistics.mean(values), metric+'_min': min(values), metric+'_max': max(values)})
                paired_summary.append(row)
    write_csv(out/'paired_summary.csv', paired_summary)
    class_pairs = []
    for pair in pairs:
        left, right = pair['comparison'].split(' minus ')
        a, b = [Path(runs[pair['seed'], arm][0]['run']) for arm in (left, right)]
        values = defaultdict(list)
        for rnd in range(81, 101):
            tables = [{int(r['class_id']): finite(r['per_class_acc']) for r in
                       read_csv(run/f'per_class_accuracy_epoch_{rnd-1}.csv')} for run in (a, b)]
            for c in range(100):
                values[c].append(tables[0][c]-tables[1][c])
        class_pairs += [dict(seed=pair['seed'], comparison=pair['comparison'], class_id=c,
                             delta_last20_accuracy=statistics.mean(v)) for c, v in values.items()]
    write_csv(out/'paired_per_class.csv', class_pairs)
    write_decision(out, pairs)
    return status, audits


def write_decision(out, pairs):
    lookup = {(r['seed'], r['comparison']): r['delta_bottom20_tail_acc'] for r in pairs}
    delta = .10
    lines = ['# Route decision evidence', '', 'Practical reference: 0.10 percentage points; not a significance or equivalence test.',
             'All judgments below use the last20 Tail20 endpoint. Overall/head/cost tradeoffs remain in the result tables.', '']
    eligible = []
    for arm in ('P', 'C'):
        required = [(42, arm+' minus '+other) for other in ('N', 'F')]
        if all(k in lookup for k in required) and all(lookup[k] > 0 for k in required):
            eligible.append(arm)
    winner = None
    if len(eligible) == 1:
        winner = eligible[0]
    elif len(eligible) == 2 and (42, 'C minus P') in lookup:
        winner = 'C' if lookup[42, 'C minus P'] >= delta else 'P'
    lines.append('Seed42 candidate for the fixed rotation control: '+str(winner or 'not established')+'.')
    for arm in ('P', 'C'):
        required = [(seed, arm+' minus '+other) for seed in (0, 3407) for other in ('N', 'F')]
        if not all(k in lookup for k in required):
            lines.append(f'{arm}: incomplete paired replication; no adoption verdict.')
            continue
        improves_a = all(lookup[seed, arm+' minus N'] > 0 for seed in (0, 3407))
        free = [lookup[seed, arm+' minus F'] for seed in (0, 3407)]
        value = improves_a and min(free) > 0 and statistics.mean(free) >= delta
        lines.append(f'{arm}: improves A on both replications={improves_a}; meets the registered extra-accuracy criterion versus F={value}.')
        nulls = [(seed, arm+' minus '+arm+'-rot') for seed in (0, 3407)]
        lines.append(f'{arm}: positive real-versus-rotation differences on both replications='+
                     (str(all(lookup[k] > 0 for k in nulls)) if all(k in lookup for k in nulls) else 'unresolved')+'.')
    lines += ['', 'This report never launches follow-up runs or selects an event, radius, seed or checkpoint.',
              'Replay folds/events are mechanism diagnostics, not independent training replications.']
    (out/'decision.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')


def replay_jobs(spec):
    jobs = [(arm, rho, 0) for rho in spec['rhos'] for arm in spec['arms']]
    if spec['rotations']:
        jobs += [(arm, 1., index) for arm in ('P-rot', 'C-rot') for index in range(3)]
    if spec.get('include_legacy'):
        jobs += [('O', 0., 0)]  # Its historical optimizer has no registered new radius.
    return jobs


def branch_path(fold, arm, rho, null_index=0):
    return f'fold{fold}/rho{rho:g}/{arm}'+(f'_null{null_index}' if arm.endswith('-rot') else '')


def summarize_replay(root):
    """Audit portable scalar artifacts; resuming additionally verifies saved tensors."""
    root = Path(root)
    out = root/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    status, metrics, costs, tables = [], [], [], defaultdict(list)
    for path in sorted(root.glob('replay/seed*/r*/route_spec.json')):
        event = path.parent
        try:
            spec = load_json(path)
            done = load_json(event/'replay_completion.json')
            receipt = load_json(event/'route_receipt.json')
            if not done.get('complete') or done.get('spec_digest') != digest(spec):
                raise ValueError('Replay event completion differs')
            for name, expected in done.get('artifacts', {}).items():
                if file_hash(event/name) != expected:
                    raise ValueError('Replay auxiliary artifact changed: '+name)
            if receipt.get('spec_digest') != digest(spec) or not receipt.get('code_unchanged'):
                raise ValueError('Replay source receipt invalid')
            event_metrics, event_costs, event_tables = [], [], defaultdict(list)
            for fold in (0, 1):
                manifest = load_json(event/f'fold{fold}/sample_manifest.json')
                for arm, rho, index in replay_jobs(spec):
                    folder = event/branch_path(fold, arm, rho, index)
                    branch = load_json(folder/'branch_completion.json')
                    expected = dict(spec_digest=digest(spec), samples=digest(manifest), arm=arm, rho=rho, null_index=index)
                    if branch.get('identity') != expected:
                        raise ValueError('Branch identity/sample manifest differs')
                    for name, expected_hash in branch['artifacts'].items():
                        if Path(name).name != name:
                            raise ValueError('Invalid artifact name')
                        if Path(name).suffix != '.pt' and file_hash(folder/name) != expected_hash:
                            raise ValueError('Replay scalar artifact changed: '+name)
                    info = dict(seed=spec['seed'], round=spec['round'], fold=fold, arm=arm,
                                rho=rho, null_index=index, sample_digest=digest(manifest))
                    rows = read_csv(folder/'group_metrics.csv')
                    expected_steps = {0} if arm == 'N' else ({0, 2} if arm == 'O' else {0, 2, 8})
                    if {int(r['step']) for r in rows if r['split'] == 'diagnostic'} != expected_steps:
                        raise ValueError('Replay diagnostic steps missing')
                    for row in rows:
                        if int(row['covered_classes']):
                            for metric in ('la', 'accuracy', 'margin'):
                                finite(row[metric])
                    event_metrics += [dict(info, **r) for r in rows]
                    event_costs.append(dict(info, **branch['costs']))
                    for name in ('source_geometry', 'optimization_steps', 'objective_trace', 'coefficients'):
                        event_tables[name] += [dict(info, **r) for r in read_csv(folder/(name+'.csv'))]
                if spec['source_probe']:
                    event_tables['source_probe'] += [dict(seed=spec['seed'], fold=fold, **r) for r in
                        read_csv(event/f'fold{fold}/probe/source_probe.csv')]
                    event_costs.append(dict(load_json(event/f'fold{fold}/probe/probe_costs.json'),
                        seed=spec['seed'], round=spec['round'], fold=fold, arm='source_probe'))
                event_tables['directions'] += [dict(seed=spec['seed'], round=spec['round'], fold=fold, **r)
                    for r in read_csv(event/f'fold{fold}/directions.csv')]
            metrics += event_metrics
            costs += event_costs
            for key, values in event_tables.items():
                tables[key] += values
            status.append(dict(run=str(event), status='valid'))
        except (ValueError, KeyError, OSError) as error:
            state = 'invalid' if (event/'replay_completion.json').is_file() else 'incomplete'
            status.append(dict(run=str(event), status=state, error=str(error)))
    write_csv(out/'replay_status.csv', status)
    write_csv(out/'event_branches.csv', metrics)
    write_csv(out/'costs.csv', costs)
    for name, values in tables.items():
        write_csv(out/('solver_trace.csv' if name == 'optimization_steps' else name+'.csv'), values)
    lookup = {}
    for r in metrics:
        if r['split'] in ('fit', 'diagnostic') and float(r['beta']) == 1:
            key = (r['seed'], r['round'], r['fold'], float(r['rho']), int(r['step']), r['split'], r['group'], r['arm'], r['null_index'])
            if key in lookup:
                raise ValueError('Duplicate replay result')
            lookup[key] = r
    comparisons = []
    for key, left in lookup.items():
        seed, rnd, fold, rho, step, split, group, arm, index = key
        if arm.endswith('-rot') or step == 0:
            continue
        for other in ('P', 'M', 'S', 'C', 'F'):
            if arm == other:
                continue
            right = lookup.get((seed, rnd, fold, 1. if arm == 'O' else rho, step, split, group, other, 0))
            if right and left['sample_digest'] == right['sample_digest'] and int(left['covered_classes']):
                comparisons.append(dict(seed=seed, round=rnd, fold=fold, rho=rho, step=step, split=split, group=group,
                    comparison=arm+' minus '+other, delta_accuracy=finite(left['accuracy'])-finite(right['accuracy']),
                    delta_la=finite(left['la'])-finite(right['la']), delta_margin=finite(left['margin'])-finite(right['margin'])))
        if arm in ('P', 'C') and rho == 1:
            controls = [lookup.get((seed, rnd, fold, rho, step, split, group, arm+'-rot', i)) for i in range(3)]
            if all(controls) and int(left['covered_classes']):
                comparisons.append(dict(seed=seed, round=rnd, fold=fold, rho=rho, step=step, split=split, group=group,
                    comparison=arm+' minus mean(3 rotations)',
                    **{'delta_'+m: finite(left[m])-statistics.mean(finite(r[m]) for r in controls) for m in ('accuracy', 'la', 'margin')}))
    write_csv(out/'replay_paired.csv', comparisons)
    (out/'replay_report.md').write_text(
        '# Same-state diagnostics\n\nSee event_branches.csv and replay_paired.csv for fixed step/radius comparisons.\n'
        'The three rotation draws are reported separately and averaged, never selected.\n'
        'primary_two_steps_* counts are deployment costs; diagnostic_extra_steps_* counts are the six extra recycled-batch steps.\n'
        'Images in the diagnostic fold can have appeared in ordinary local training. No official test metric selects these branches.\n'
        'Tensor files are unnecessary for portable summaries; branch resumption verifies them too.\n', encoding='utf-8')
    return status
