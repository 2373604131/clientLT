"""Recollect one ordinary-A FedAvg trajectory with fixed Tail80..99 on one GPU.

server: verify reference -> separate one-round smoke -> 100 training rounds ->
validate retained tensors/proposals -> sequential FedAvg projection diagnosis.
Repeating the same command resumes the registered training/diagnostic outputs.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
GROUPS = dict(head_ids=list(range(20)), middle_ids=list(range(20, 80)), tail_ids=list(range(80, 100)))
# Semantic digest of the original sample/client/order/label tuples, independent
# of CSV newline convention. A different allocation is a different experiment.
PARTITION_SHA256 = '690f7c9c0d4f261fde67aad15520b864041b92df770ae48118101b51eb036e66'


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, allow_abbrev=False)
    parser.add_argument('--stage', choices=('plan', 'preflight', 'server', 'status', 'audit', 'diagnose'), default='plan')
    parser.add_argument('--gpu', type=int, default=0, help='One index within the current CUDA_VISIBLE_DEVICES allocation')
    parser.add_argument('--reference-run', type=Path, default=Path('references/full10_clientlt'))
    parser.add_argument('--data-root', type=Path, default=Path('DATA'))
    parser.add_argument('--output-root', type=Path, default=Path('output/cifar100_LT/a_projection_recollect_seed42_tail20'))
    parser.add_argument('--diagnosis-root', type=Path)
    parser.add_argument('--num-workers', type=int, default=0)
    parser.add_argument('--client-concurrency', type=int, choices=(1, 4, 6, 8), default=1)
    parser.add_argument('--train-only', action='store_true', help='Retain/validate inputs; defer diagnosis to --stage diagnose')
    args = parser.parse_args(argv)
    if args.gpu < 0 or args.num_workers < 0:
        parser.error('GPU index and worker count must be nonnegative')
    for field in ('reference_run', 'data_root', 'output_root'):
        setattr(args, field, getattr(args, field).resolve())
    args.diagnosis_root = (args.diagnosis_root or args.output_root / 'diagnosis').resolve()
    return args


def reference_contract(reference):
    from tools.client_aggregation.protocol import count_matrix, partition_signature, read_csv, load_json
    fingerprint, counts, matrix = count_matrix(reference)
    protocol = load_json(reference / 'protocol/eri_protocol.json')
    if protocol['tail_class_ids'] != GROUPS['tail_ids'] or counts['tail_ids'] != GROUPS['tail_ids']:
        raise ValueError('Tail classes must remain exactly IDs 80..99; do not replace the reference grouping')
    signature = partition_signature(read_csv(reference / 'partition_manifest.csv'))
    if signature != PARTITION_SHA256 or int(matrix.sum()) != 10847:
        raise ValueError('Training samples/client assignment changed; restore references/full10_clientlt')
    head = sorted(range(100), key=lambda c: (-counts['class_counts'][c], c))[:20]
    tail = sorted(sorted(range(100), key=lambda c: (counts['class_counts'][c], -c))[:20])
    if head != GROUPS['head_ids'] or tail != GROUPS['tail_ids']:
        raise ValueError('Reference frequencies do not match the fixed class groups')
    return dict(fingerprint=fingerprint, partition_sha256=signature, fixed_class_groups=GROUPS,
                class_counts=counts['class_counts'], clients=30, training_samples=int(matrix.sum()), seed=42)


def register(args):
    from scripts.run_ab_validation import file_lock, write_json
    from tools.client_aggregation.protocol import SCHEMA, PLAIN_A_CONTRACT, aggregation_spec, code_hashes, load_json, write_weight_tables
    reference = reference_contract(args.reference_run)
    settings = dict(seed=42, reference_run=str(args.reference_run), data_root=str(args.data_root),
                    num_workers=args.num_workers, client_concurrency=args.client_concurrency,
                    parallel_validation='off', aggregation_rule='fedavg', lambda_value=None,
                    fixed_class_groups=GROUPS)
    plan = dict(schema=SCHEMA, contract=PLAIN_A_CONTRACT, settings=settings,
                code_sha256=code_hashes(REPO), aggregation=aggregation_spec(args.reference_run, .1, 'fedavg'))
    registration = dict(version='a_projection_recollect_v1', reference=reference,
                        training_plan=plan, launcher_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                        collection='frozen base + all 100 B and 90 A event tensors; retain all local A proposals')
    root = args.output_root
    with file_lock(root / 'locks/recollection_plan.lock'):
        manifest, path = root / 'recollection_plan.json', root / 'fedavg/experiment_plan.json'
        for file, value in ((manifest, registration), (path, plan)):
            if file.is_file() and load_json(file) != value:
                raise ValueError('Registered code/reference/settings changed; preserve this run and use a new --output-root: ' + str(root))
        if not manifest.is_file():
            # Never adopt a legacy run merely because its directory name matches.
            if (root / 'fedavg').exists() and any((root / 'fedavg').iterdir()):
                raise ValueError('Unregistered FedAvg output exists; use a new --output-root')
            write_json(manifest, registration)
        write_json(path, plan)
        write_weight_tables(root / 'fedavg', plan['aggregation'])
    print('FIXED GROUPS: Head=0..19; Middle=20..79; Tail=80..99 (20 classes)', flush=True)
    print('Seed42; original 10847 samples/30 clients; B3 every round; A1 rounds1..90; rounds91..100 B only.', flush=True)
    print('Saved training source:', root / 'fedavg/runs/seed42/plain_a', flush=True)
    return plan


def audit_inputs(args):
    import torch
    from scripts.run_ab_validation import write_json
    from tools.a_projection_diagnosis.protocol import check_source, StateIndex, read_json
    from tools.a_projection_diagnosis.states import StateStore
    run = args.output_root / 'fedavg/runs/seed42/plain_a'
    if read_json(run / 'completion.json').get('completed_round') != 100:
        raise ValueError('The recollected trajectory has not completed 100 rounds')
    source = check_source(run, 'fedavg')
    rows = StateIndex(source).inventory()
    missing = [r['artifact'] for r in rows if not r['available']]
    if missing:
        raise FileNotFoundError('Recollection is missing required models/original proposals: ' + ', '.join(missing))
    store = StateStore(source)  # Frozen base and initial LoRA hashes are verified here.
    proposals = []
    for rnd in (20, 50, 80):
        store.committed(rnd)
        anchor, _ = store.anchor(rnd)
        local, provenance = store.historical_proposals(rnd, anchor)
        if local is None or set(local) != set(range(30)):
            raise ValueError('Recollection must retain all 30 historical proposals at round%d' % rnd)
        proposals.append(dict(round=rnd, clients=len(local), **provenance))
    at90, at100 = store.committed(90), store.committed(100)
    if any(not torch.equal(at90[k], at100[k]) for k in store.a_keys):
        raise ValueError('A changed in rounds91..100')
    # Verify all native events remain on disk, not only the five diagnostic endpoints.
    events = [run / 'events' / ('r%03d_c000_main_%s/state.pt' % (rnd, phase))
              for rnd in range(1, 101) for phase in (('normal_B', 'refresh_A') if rnd <= 90 else ('normal_B',))]
    if any(not file.is_file() or file.stat().st_size == 0 for file in events):
        raise FileNotFoundError('Expected all 190 nonempty native B/A event tensor files')
    files = sorted({Path(r['path']).resolve() for r in rows if r.get('path')})
    receipt = dict(source=str(run), fixed_class_groups=GROUPS, completed_round=100,
                   frozen_base_and_initial_hashes_verified=True, a90_equals_a100=True,
                   event_tensor_count=len(events), original_proposals=proposals,
                   required_files=[dict(path=str(p.relative_to(run.resolve())), bytes=p.stat().st_size) for p in files],
                   note='Keep the whole training source directory, especially checkpoints/ and events/. Lightweight result packs omit .pt files.')
    write_json(args.output_root / 'retained_inputs_verified.json', receipt)
    print('VERIFIED: base + 190 events + all 30 original A proposals at rounds20/50/80; fixed Tail80..99.', flush=True)
    return receipt


def diagnose(args):
    from scripts.run_a_projection_diagnosis import main as diagnose_main
    source = args.output_root / 'fedavg/runs/seed42/plain_a'
    diagnose_main(['--stage', 'server', '--fedavg-run', str(source), '--trajectories', 'fedavg',
                   '--data-root', str(args.data_root), '--output-root', str(args.diagnosis_root),
                   '--device', 'cuda:0', '--num-workers', str(args.num_workers)])
    # Emit all available FedAvg results; the report explicitly retains missing TailRW16 status.
    diagnose_main(['--stage', 'pack', '--output-root', str(args.diagnosis_root)])


def main(argv=None):
    args = parse_args(argv)
    if args.stage == 'status':
        run = args.output_root / 'fedavg/runs/seed42/plain_a'
        path = run / ('completion.json' if (run / 'completion.json').is_file() else 'progress.json')
        print(path.read_text(encoding='utf-8') if path.is_file() else 'FedAvg has not started: ' + str(run))
        return
    if args.stage in ('server', 'diagnose'):
        # Respect scheduler allocations; never reinterpret visible index0 as physical GPU0.
        from tools.a_projection_diagnosis.parallel import gpu_tokens
        token = gpu_tokens([args.gpu])[0]
        os.environ['CUDA_VISIBLE_DEVICES'] = token
        print('One GPU: visible index %d -> device %s' % (args.gpu, token), flush=True)
    plan = register(args)
    if args.stage == 'plan':
        return
    if args.stage in ('audit', 'diagnose'):
        audit_inputs(args)
        if args.stage == 'diagnose':
            diagnose(args)
        return
    from scripts import run_cliplora_joint_aggregation as single
    single.preflight(plan, check_cuda=args.stage == 'server')
    if args.stage == 'preflight':
        print('Reference/config/data preflight passed; no training started.', flush=True)
        return
    from scripts.run_ab_validation import file_lock
    with file_lock(args.output_root / 'locks/recollection_server.lock', timeout=.1):
        single.execute(single.make_job(plan, 'plain_a', 'smoke'), args.output_root / 'fedavg')
        single.execute(single.make_job(plan, 'plain_a'), args.output_root / 'fedavg')
        audit_inputs(args)
        if not args.train_only:
            diagnose(args)


if __name__ == '__main__':
    try:
        main()
    except (OSError, ValueError, KeyError) as error:
        raise SystemExit(str(error))
