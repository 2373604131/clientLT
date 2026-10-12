"""Offline ordinary-A projection diagnosis; sequential on one GPU by default."""
import argparse
import json
from pathlib import Path
import sys
import zipfile

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from tools.a_projection_diagnosis.protocol import (
    PROTOCOL, ANCHORS, StateIndex, check_pair, fit_manifest,
    digest, file_digest, read_json, write_json, write_csv,
)
from tools.a_projection_diagnosis.discovery import DEFAULT_SOURCE_ROOT, artifact_required, discover_source


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    p.add_argument('--stage', choices=('inventory', 'preflight', 'smoke', 'run', 'server', 'summary', 'pack'), default='inventory')
    p.add_argument('--source-root', type=Path, help='Explicit suite root; omitted: search known ordinary-A locations')
    p.add_argument('--search-root', type=Path, action='append', default=[],
                   help='Additional repository/output/suite location to search; repeatable, with automatic source discovery')
    p.add_argument('--fedavg-run', type=Path)
    p.add_argument('--tailrw16-run', type=Path)
    p.add_argument('--trajectories', nargs='+', choices=('fedavg', 'tailrw16'), default=['fedavg', 'tailrw16'])
    p.add_argument('--part', choices=('all', 'e1', 'events'), default='all')
    p.add_argument('--rounds', nargs='+', type=int, choices=ANCHORS, default=list(ANCHORS))
    p.add_argument('--output-root', type=Path, default=Path('output/a_projection_diagnosis_v1'))
    p.add_argument('--data-root', type=Path, default=Path('DATA'))
    p.add_argument('--device', default='cuda:0', help='Device for the default single-process sequential run')
    p.add_argument('--gpus', nargs='+', type=int, help='Optional multi-process GPU queue; omit for single-GPU sequential execution')
    p.add_argument('--cpu-threads', type=int, default=2, help='CPU math threads per process')
    p.add_argument('--seed', type=int, choices=(42,), default=42)
    p.add_argument('--num-workers', type=int, default=0)
    p.add_argument('--eval-batch-size', type=int, default=64)
    p.add_argument('--no-plots', action='store_true')
    p.add_argument('--archive', type=Path)
    a = p.parse_args(argv)
    a.auto_source = a.source_root is None
    a.source_root = a.source_root or DEFAULT_SOURCE_ROOT
    if a.search_root and not a.auto_source:
        p.error('Use either --source-root for an exact suite or --search-root for discovery')
    if a.num_workers < 0 or a.cpu_threads < 1 or a.eval_batch_size < 1 or len(set(a.trajectories)) != len(a.trajectories) or len(set(a.rounds)) != len(a.rounds):
        p.error('Invalid workers/batch size or duplicated selection')
    if a.part == 'events' and 'fedavg' not in a.trajectories:
        p.error('Local/aggregation/intervention panels use FedAvg anchors only')
    if a.gpus is not None:
        if a.stage not in ('preflight', 'smoke', 'run', 'server'):
            p.error('--gpus is used with preflight/smoke/run/server')
        if len(a.gpus) > 4 or len(set(a.gpus)) != len(a.gpus) or min(a.gpus) < 0:
            p.error('Select one to four distinct nonnegative GPU indices')
        if a.device != 'cuda:0':
            p.error('With --gpus each worker uses its own cuda:0; omit --device')
    return a


def code_fingerprint():
    paths = [Path(__file__)] + sorted((REPO / 'tools/a_projection_diagnosis').glob('*.py'))
    paths += [REPO / p for p in ('tools/a_learning_schedule/runtime.py', 'tools/a_learning_schedule/config.py',
              'trainers/cliplora.py', 'utils/loralib/layers.py', 'utils/cliplora_a_refresh.py',
              'tools/client_aggregation/weights.py', 'Dassl/dassl/data/data_manager.py',
              'Dassl/dassl/data/transforms/transforms.py', 'utils/cliplora_loss.py', 'utils/sfra_execution.py')]
    return digest({str(p.relative_to(REPO)).replace('\\', '/'): file_digest(p) for p in paths})


def inventory(args):
    from scripts.run_ab_validation import file_lock
    with file_lock(args.output_root / 'inventory.lock'):
        protocol_path = args.output_root / 'protocol.json'
        if protocol_path.is_file() and read_json(protocol_path) != PROTOCOL:
            raise ValueError('Output protocol differs. Select a new --output-root.')
        write_json(protocol_path, PROTOCOL)
    sources, rows, errors, discovery = [], [], [], {}
    for rule in args.trajectories:
        try:
            source, found = discover_source(rule, args.source_root, getattr(args, rule + '_run'),
                auto=args.auto_source, search_roots=args.search_root, repo=REPO, part=args.part, rounds=args.rounds)
            discovery[rule] = found
            if source is None:
                errors.append(rule + ': ' + found['error'])
                continue
            # Pin the parent's exact selection into all GPU child commands.
            setattr(args, rule + '_run', Path(source['path']))
            if found.get('automatic'):
                print('SOURCE %s -> %s' % (rule, source['path']), flush=True)
            sources.append(source)
            indexed = StateIndex(source).inventory(need_events=rule == 'fedavg')
            rows.extend(dict(trajectory=rule, **r) for r in indexed)
            required = [r for r in indexed if artifact_required(r, args.part, args.rounds)]
            ready = all(r['available'] for r in required)
            original_dir = source['meta']['resolved_args'].get('output_dir')
            source['inventory_status'] = dict(files_ready=ready, selected_run=source['path'], recorded_server_run=original_dir,
                mode='historical_artifacts_available_inspect_payloads' if ready else
                     'missing_required_models_check_server_original_directory',
                proposals_may_be_regenerated=any(not r['available'] for r in indexed if r['artifact'].startswith('proposals_')))
            with file_lock(args.output_root / 'inventory.lock'):
                write_csv(args.output_root / rule / 'probe_fit_manifest.csv', fit_manifest(source['rows']))
        except (OSError, ValueError, KeyError) as e:
            errors.append('%s: %s' % (rule, e))
    try:
        pair = check_pair(sources) if not errors else dict(paired=False, reason='invalid_source')
    except (ValueError, KeyError) as error:
        errors.append('pair: ' + str(error))
        pair = dict(paired=False, reason='source_mismatch')
    report = dict(protocol=PROTOCOL, source_errors=errors, pair=pair,
                  trajectories={s['rule']: s['inventory_status'] for s in sources},
                  cuda_tested=False, training_started=False,
                  note='File presence is checked here. Tensor hashes, counts, and aggregate reconstruction are checked by smoke/run.')
    with file_lock(args.output_root / 'inventory.lock'):
        # Separate inventories allow two trajectory workers without clobbering metadata.
        label = '_'.join(args.trajectories)
        write_json(args.output_root / ('source_discovery_' + label + '.json'), discovery)
        discovery_path = args.output_root / 'source_discovery.json'
        combined = read_json(discovery_path) if discovery_path.is_file() else {}
        combined.update(discovery)
        write_json(discovery_path, combined)
        write_json(args.output_root / ('artifact_inventory_' + label + '.json'), report)
        write_csv(args.output_root / ('artifact_inventory_' + label + '.csv'), rows)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return sources, rows, errors


def require_inputs(args, sources, rows, errors):
    if errors:
        raise ValueError('\n'.join(errors))
    missing = []
    for r in rows:
        if artifact_required(r, args.part, args.rounds) and not r['available']:
            missing.append(r['trajectory'] + ':' + r['artifact'])
    if missing:
        raise FileNotFoundError('Missing models: ' + ', '.join(missing) + '. Check the recorded SERVER directory first; lightweight archives exclude .pt files.')
    from utils.functional_coverage_validation import _locate_cifar100
    _locate_cifar100(args.data_root.resolve())
    return code_fingerprint()


def register_source(args, source, fingerprint):
    from scripts.run_ab_validation import file_lock
    root = args.output_root / source['rule']
    identity = dict(protocol=PROTOCOL, fingerprint=fingerprint, source=str(source['path']),
                    metadata=digest(source['meta']), partition=source['partition_sha256'],
                    config=digest(source['config']), eval_batch_size=args.eval_batch_size,
                    data_root=str(args.data_root.resolve()), execution_layout='isolated_tasks_v2',
                    cpu_threads=args.cpu_threads)
    # Only registration holds this trajectory lock; evaluation uses task locks.
    with file_lock(root / 'registration.lock'):
        registered = root / 'run_identity.json'
        if registered.is_file() and read_json(registered) != identity:
            raise ValueError('Protocol/code/source changed. Keep prior results and select a new output root.')
        write_json(registered, identity)


def task_finished(root, job):
    common = ('completion.json', 'numerical_checks.json', 'projection_geometry.csv')
    needed = ('representation_curves.csv', 'full_model_curves.csv') if job['part'] == 'e1' else (
        'projection_probes.csv', 'full_model_candidates.csv', 'local_class_response.csv',
        'a_weight_counterfactuals.csv', 'rowspace_interventions.csv', 'provenance.json')
    return all((root / name).is_file() for name in common + needed)


def execute(args, sources, fingerprint, skip_completed_smoke=False):
    import gc
    import torch
    from scripts.run_ab_validation import file_lock
    from tools.a_projection_diagnosis.parallel import selected_jobs, task_root
    from tools.a_projection_diagnosis.states import StateStore
    from tools.a_projection_diagnosis.backend import ClipBackend
    from tools.a_projection_diagnosis.experiment import longitudinal, event_panels
    from utils.cliplora_a_refresh import state_hash
    torch.set_num_threads(args.cpu_threads)
    sources_by_rule = {s['rule']: s for s in sources}
    for job in selected_jobs(args):
        source = sources_by_rule[job['trajectory']]
        register_source(args, source, fingerprint)
        formal_root = task_root(args.output_root, job)
        root = formal_root
        if args.stage == 'smoke':
            root = root / 'smoke'
        with file_lock(formal_root / 'run.lock', timeout=.1):
            if (args.stage == 'run' or skip_completed_smoke) and task_finished(formal_root, job):
                print('SKIP completed task: ' + job['id'], flush=True)
                continue
            print('START %s stage=%s device=%s' % (job['id'], args.stage, args.device), flush=True)
            store = StateStore(source)
            backend = ClipBackend(args, source, store, root, fingerprint)
            if args.stage == 'smoke':
                for split in ('fit', 'panel'):
                    old = backend.loaders[split]
                    indices = list(range(min(4, len(old.dataset))))
                    backend.loaders[split] = torch.utils.data.DataLoader(torch.utils.data.Subset(old.dataset, indices), batch_size=4)
                    backend.ids[split] = backend.ids[split][indices]
                    backend.labels[split] = backend.labels[split][indices]
                    backend.measure(store.initial, dict(initial=store.initial), split, force=True)
                if job['part'] == 'events':
                    anchor, _ = store.anchor(job['round'])
                    store.historical_proposals(job['round'], anchor)
                write_json(root / 'smoke_passed.json', dict(real_model_forward=True, training=False,
                    formal_measurements=False, initial_sha256=state_hash(store.initial, store.keys)))
            else:
                if job['part'] == 'e1':
                    longitudinal(backend, store, root)
                else:
                    event_panels(backend, store, job['round'], root)
            del backend, store
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            print('DONE %s stage=%s' % (job['id'], args.stage), flush=True)


def main(argv=None):
    args = parse_args(argv)
    if args.stage in ('summary', 'pack'):
        from tools.a_projection_diagnosis.summary import summarize
        result = summarize(args.output_root, not args.no_plots)
        print(json.dumps(result, ensure_ascii=False, indent=2))
        if args.stage == 'pack':
            target = args.archive or args.output_root.with_name(args.output_root.name + '_results.zip')
            target.parent.mkdir(parents=True, exist_ok=True)
            with zipfile.ZipFile(target, 'w', compression=zipfile.ZIP_DEFLATED) as archive:
                for p in sorted(args.output_root.rglob('*')):
                    if (p.is_file() and p != target and not {'cache', 'regenerated', 'smoke'}.intersection(p.relative_to(args.output_root).parts)
                            and p.suffix in ('.json', '.csv', '.txt', '.npz', '.png', '.pdf')):
                        archive.write(p, str(Path(args.output_root.name) / p.relative_to(args.output_root)))
            print('Archive:', target)
        return
    sources, rows, errors = inventory(args)
    if args.stage == 'inventory':
        if errors:
            raise ValueError('\n'.join(errors))
        return
    fingerprint = require_inputs(args, sources, rows, errors)
    if args.gpus is not None:
        from tools.a_projection_diagnosis.parallel import launch, selected_jobs, gpu_tokens
        tokens = gpu_tokens(args.gpus)
        print('GPU queue: %d jobs on %s' % (len(selected_jobs(args)), ','.join(tokens)), flush=True)
        if args.stage != 'preflight':
            for source in sources:
                register_source(args, source, fingerprint)
            launch(args, REPO)
            if args.stage in ('server', 'run'):
                from tools.a_projection_diagnosis.summary import summarize
                summarize(args.output_root, not args.no_plots)
            return
    if args.stage == 'preflight':
        print('File/configuration preflight passed; GPU execution has not been tested.')
        return
    print('Sequential execution on %s: one task/model at a time.' % args.device, flush=True)
    if args.stage == 'server':
        args.stage = 'smoke'
        execute(args, sources, fingerprint, skip_completed_smoke=True)
        args.stage = 'run'
        execute(args, sources, fingerprint)
        if set(args.trajectories) == {'fedavg', 'tailrw16'}:
            from tools.a_projection_diagnosis.summary import summarize
            summarize(args.output_root, not args.no_plots)
        else:
            print('Selected trajectory finished. Run --stage pack after all selected server jobs finish.')
    else:
        execute(args, sources, fingerprint)


if __name__ == '__main__':
    main()
