"""Portable contracts and read-only collection for the shared-C guard pilot."""
import hashlib
import math
from pathlib import Path
import statistics

from tools.sfra.ab_validation import code_hashes as frozen_hashes
from tools.sfra.calibration_reference import protocol_signature, ROUNDS
from tools.sfra.maintext import FROZEN, digest, load_json
from tools.sfra.summary import METRICS, read_csv, write_csv


SCHEMA = 'b_class_guard_experiment_v1'
EXTRA_FILES = ('utils/b_class_guard_math.py','utils/cliplora_b_class_guard.py',
    'utils/cliplora_b_guard_runtime.py','scripts/train_cliplora_b_class_guard.py',
    'scripts/run_cliplora_b_class_guard.py','tools/sfra/b_class_guard.py')


def source_hashes(repo):
    values = frozen_hashes(repo)
    values.update({name:hashlib.sha256((Path(repo)/name).read_text(encoding='utf-8').encode()).hexdigest()
                   for name in EXTRA_FILES})
    return values


def check_sources(spec, repo):
    if spec.get('schema_version') != SCHEMA or spec.get('code_sha256') != source_hashes(repo):
        raise ValueError('Guard source/configuration changed; preserve the run and use its original code or a new output root')


def baseline_kind(cfg):
    if cfg.get('variant') != 'full-cp' or cfg.get('retention_weight') != 10 or cfg.get('classification_weight') != 1:
        return None
    b = cfg.get('b_transfer')
    if b is None:
        return 'A'
    if b.get('calibration_profile') == 'coverage_tradeoff' and b.get('non_tail_sampling') == 'class-cyclic' and b.get('tail_weight') == .35:
        return 'B-old'
    return None


def validate_profile(cfg, spec=None):
    for key,value in {**FROZEN,'variant':'full-cp','classification_weight':1.,'witness_batch_size':8}.items():
        if cfg.get(key) != value:
            raise ValueError('Frozen A setting differs: '+key)
    b = cfg.get('b_transfer')
    if b:
        for key,value in dict(mode='shared',non_tail_sampling='class-cyclic',tail_weight=.35,
                learning_rate=.3,probe_step=.1,regularization=.001,steps=2,rounds=list(ROUNDS)).items():
            if b.get(key) != value:
                raise ValueError('Frozen B setting differs: '+key)
    if spec is not None:
        s = spec['settings']
        if cfg.get('b_class_guard_experiment') != spec or b is None:
            raise ValueError('Runtime guard contract differs from launcher')
        if b.get('calibration_profile') != 'class_guard' or b.get('guard_variant') != s['guard']:
            raise ValueError('Wrong runtime guard variant')
        if b.get('guard_weight') != (0. if s['guard']=='off' else 1.):
            raise ValueError('Wrong fixed guard weight')
        for key in ('seed','protocol_seed','partition'):
            if cfg[key] != s[key]:
                raise ValueError('Runtime/launcher differ: '+key)


def finite(value):
    value = float(value)
    if not math.isfinite(value):
        raise ValueError('Nonfinite result')
    return value


def read_run(run, kind, spec=None):
    cfg = load_json(run/'sfra_config.json')
    validate_profile(cfg,spec)
    if spec is not None:
        receipt = load_json(run/'guard_receipt.json')
        if receipt.get('spec_digest') != digest(spec) or not receipt.get('code_unchanged'):
            raise ValueError('Missing/mismatched guard training receipt')
    done = load_json(run/'completion.json')
    if done.get('completed_round') != 100 or done.get('official_test_passes') != 101:
        raise ValueError('Expected 100 rounds and 101 evaluations')
    for key,expected in [('normal_optimizer_steps','normal_steps_expected'),('extra_optimizer_steps','extra_steps_expected')]:
        if done[key] != cfg['base_training_config'][expected]:
            raise ValueError('Local optimization budget differs')
    rows = sorted(read_csv(run/'round_metrics.csv'),key=lambda r:int(r['round']))
    if [int(r['round']) for r in rows] != list(range(101)):
        raise ValueError('Missing/duplicate rounds')
    for r in rows:
        if any(not 0 <= finite(r[m]) <= 100 for m in METRICS):
            raise ValueError('Accuracy out of range')
    prior = load_json(run/'class_prior.json')
    tail = set(prior['tail_ids'])
    if len(tail) != 20 or len(prior['tail_ids']) != 20 or len(prior['counts']) != 100:
        raise ValueError('Fixed Tail20/counts differ')
    head = set(sorted(range(100),key=lambda c:(-prior['counts'][c],c))[:20])
    groups = dict(overall_acc=set(range(100)),head20_acc=head,middle60_acc=set(range(100))-head-tail,
                  bottom20_tail_acc=tail,non_tail_acc=set(range(100))-tail)
    if head & tail:
        raise ValueError('Overlapping head/tail groups')
    per_class = {c:[] for c in range(100)}
    for rnd in range(81,101):
        values = read_csv(run/f'per_class_accuracy_epoch_{rnd-1}.csv')
        if len(values)!=100 or {int(x['class_id']) for x in values}!=set(range(100)):
            raise ValueError('Incomplete per-class evaluation')
        values = {int(x['class_id']):finite(x['per_class_acc']) for x in values}
        if any(not 0<=x<=100 for x in values.values()):
            raise ValueError('Per-class accuracy out of range')
        for metric,ids in groups.items():
            if not math.isclose(statistics.mean(values[c] for c in ids),float(rows[rnd][metric]),abs_tol=1e-8):
                raise ValueError('Per-class table and round metric differ: '+metric)
        for c,x in values.items():
            per_class[c].append(x)
    result = dict(run=str(run),guard=kind,seed=cfg['seed'],partition=cfg['partition'],protocol_seed=cfg['protocol_seed'],
                  cohort='development' if cfg['seed']==42 else 'paired-replication')
    for metric in METRICS:
        result['last20_'+metric] = statistics.mean(float(r[metric]) for r in rows[81:])
        result['final_'+metric] = float(rows[100][metric])
    result['elapsed_seconds'] = finite(done['elapsed_seconds'])
    tables = {}
    if cfg.get('b_transfer'):
        events = read_csv(run/'b_transfer_rounds.csv')
        if sorted(int(x['round']) for x in events)!=list(ROUNDS) or done.get('b_transfer_events')!=8:
            raise ValueError('Transfer event budget differs')
        for field in ('algorithm_forward_images','algorithm_backward_images','diagnostic_forward_images',
                      'guard_diagnostic_backward_images','extra_upload_bytes','extra_downlink_bytes','seconds'):
            result['B_'+field] = sum(finite(x.get(field) or 0) for x in events)
        total = 0
        tables['b_transfer_rounds'] = events
        for rnd in ROUNDS:
            folder = run/f'b_transfer_rounds/r{rnd:03d}'
            manifest = load_json(folder/'calibration_manifest.json')
            steps = read_csv(folder/'optimization_steps.csv')
            active = bool(manifest['shared_donor_ids'])
            if [int(x['step']) for x in steps] != ([1,2] if active else []):
                raise ValueError('Transfer optimizer-step budget differs')
            total += len(steps)
            names = ['optimization_steps','probe_metrics','c_loss_trace','client_feedback_steps']
            if spec is not None and kind!='off' and active:
                names += ['guard_class_trace','guard_client_trace']
                if not (folder/'guard_references.npz').is_file():
                    raise ValueError('Missing saved guard reference')
            for name in names:
                path = folder/(name+'.csv')
                if path.is_file():
                    tables.setdefault(name,[]).extend(read_csv(path))
                elif active:
                    raise ValueError('Missing transfer diagnostic: '+str(path))
            if spec is not None and kind!='off' and active:
                trace = read_csv(folder/'c_loss_trace.csv')
                if [int(r['c_step']) for r in trace] != [0,1,2]:
                    raise ValueError('Missing guard trace states')
                if any(r['guard_variant'] != kind for r in trace+steps):
                    raise ValueError('Mixed guard variants')
                if abs(finite(trace[0]['fixed_pool_guard']))>1e-10:
                    raise ValueError('Guard reference is not the zero residual')
        if done.get('b_transfer_optimizer_steps') != total:
            raise ValueError('Transfer completion budget differs')
        result['B_optimizer_steps'] = total
    sig = protocol_signature(run)
    sig.update(base_training=digest(cfg['base_training_config']),tail_ids=digest(sorted(tail)))
    source = spec['code_sha256'] if spec is not None else None
    if spec is None and (run/'ab_validation_run.json').is_file():
        old = load_json(run/'ab_validation_run.json')
        if not old.get('code_unchanged') or old.get('job',{}).get('seed') != cfg['seed']:
            raise ValueError('Invalid frozen baseline code receipt')
        source = old['job']['code_sha256']
    core = frozen_hashes(Path(__file__).resolve().parents[2])
    # Compare the unchanged legacy files, not the intentionally new guard files.
    # A baseline without a receipt can be displayed, but cannot form a matched pair.
    sig['frozen_source'] = digest({k:source[k] for k in core}) if source is not None else None
    return result,sig,{c:statistics.mean(v) for c,v in per_class.items()},rows,tables,tail


def summarize(root, compare_root=None):
    root = Path(root).resolve();out=root/'analysis';out.mkdir(parents=True,exist_ok=True)
    candidates = [(p.parent,False) for p in sorted(root.glob('seed*/*/*/*/guard_spec.json'))]
    if compare_root:
        other = Path(compare_root).resolve()
        if not other.is_dir():
            raise ValueError('Comparison root does not exist: '+str(other))
        candidates += [(p.parent,True) for p in sorted(other.rglob('sfra_config.json'))
                       if baseline_kind(load_json(p)) is not None and p.parent not in {x[0] for x in candidates}]
    status,results,curves,classes,all_tables = [],[],[],[],{}
    valid = {}
    for run,external in candidates:
        spec = None if external else load_json(run/'guard_spec.json')
        cfg = load_json(run/'sfra_config.json') if (run/'sfra_config.json').is_file() else None
        kind = baseline_kind(cfg) if external else spec['settings']['guard']
        seed = cfg['seed'] if external else spec['settings']['seed']
        progress = load_json(run/'progress.json') if (run/'progress.json').is_file() else {}
        row = dict(run=str(run),guard=kind,seed=seed,completed_round=progress.get('completed_round',0),status='pending',reason='')
        status.append(row)
        if not (run/'completion.json').is_file():
            continue
        try:
            record = read_run(run,kind,spec)
            key = (seed,kind)
            if key in valid:
                raise ValueError('Duplicate completed seed/guard; use a narrower comparison root')
            valid[key] = record
            perf,sig,pc,curve,tables,tail = record
            results.append(perf);row.update(status='complete',completed_round=100)
            info = dict(run=str(run),guard=kind,seed=seed)
            curves.extend({**x,**info} for x in curve)
            classes.extend(dict(info,class_id=c,is_tail=c in tail,last20_acc=v) for c,v in pc.items())
            for name,rows in tables.items():
                all_tables.setdefault(name,[]).extend({**x,**info} for x in rows)
        except (OSError,ValueError,KeyError,TypeError) as error:
            row.update(status='invalid',reason=str(error))
    pairs,audits,pc_pairs = [],[],[]
    for seed in sorted({x[0] for x in valid}):
        for left,right in [('class','client'),('class','B-old'),('client','B-old'),('class','A'),('client','A'),
                           ('class','off'),('client','off'),('off','B-old')]:
            if (seed,left) not in valid or (seed,right) not in valid:
                continue
            a,b=valid[seed,left],valid[seed,right]
            audit=dict(seed=seed,left=left,right=right,valid=False,reason='')
            audits.append(audit)
            try:
                if a[1] != b[1]:
                    raise ValueError('Initialization/partition/schedule/witness/execution/training/source fingerprints differ')
                if any(not math.isclose(float(x[m]),float(y[m]),abs_tol=1e-8,rel_tol=0)
                       for x,y in zip(a[3][:30],b[3][:30]) for m in METRICS):
                    raise ValueError('Pre-transfer rounds0..29 differ')
                if left!='A' and right!='A':
                    for rnd in ROUNDS:
                        ma=load_json(Path(a[0]['run'])/f'b_transfer_rounds/r{rnd:03d}/calibration_manifest.json')
                        mb=load_json(Path(b[0]['run'])/f'b_transfer_rounds/r{rnd:03d}/calibration_manifest.json')
                        if ma['feedback_clients']!=mb['feedback_clients'] or {k:v['batches'] for k,v in ma['recipients'].items()}!={k:v['batches'] for k,v in mb['recipients'].items()}:
                            raise ValueError('Feedback sample identities/budgets differ')
                audit['valid']=True
                deltas={c:a[2][c]-b[2][c] for c in a[5]}
                pairs.append(dict(seed=seed,left=left,right=right,
                    **{'delta_'+m:a[0]['last20_'+m]-b[0]['last20_'+m] for m in METRICS},
                    tail_improved=sum(x>1e-8 for x in deltas.values()),tail_harmed=sum(x < -1e-8 for x in deltas.values()),
                    tail_negative_delta_sum=sum(min(x,0) for x in deltas.values()),worst_tail_delta=min(deltas.values())))
                pc_pairs.extend(dict(seed=seed,left=left,right=right,class_id=c,is_tail=c in a[5],
                                     delta_last20_acc=a[2][c]-b[2][c]) for c in range(100))
            except (OSError,ValueError,KeyError) as error:
                audit['reason']=str(error)
    for name,rows in [('status',status),('performance',results),('curves',curves),('per_class',classes),
                      ('paired',pairs),('pair_audit',audits),('paired_per_class',pc_pairs),*all_tables.items()]:
        write_csv(out/(name+'.csv'),rows)
    lines=['# Shared-C class guard results','',
        'Primary endpoint: committed rounds81..100 fixed Tail20 mean; units are percent / percentage points.',
        'Training class-cell harm is a proxy; client-events/classes are not independent seed repetitions.','',
        '| Seed | Variant | Overall | Head20 | Middle60 | Tail20 | Non-tail80 |',
        '|---|---|---:|---:|---:|---:|---:|']
    for r in results:
        lines.append('| '+str(r['seed'])+' | '+r['guard']+' | '+' | '.join(f'{r["last20_"+m]:.4f}' for m in METRICS)+' |')
    lines += ['',f'Complete runs: {len(results)}; pending: {sum(x["status"]=="pending" for x in status)}; invalid: {sum(x["status"]=="invalid" for x in status)}.',
              'Only fingerprint- and sample-matched pairs enter paired.csv. See status.csv and pair_audit.csv for exclusions.',
              'Guard diagnostic backward traversals are additional computation, reported separately.',
              'No guard result has been declared a success automatically.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    return status,audits
