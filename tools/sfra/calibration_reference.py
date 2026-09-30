"""Portable training-side update norms for the AB calibration control."""
import hashlib
import math
from pathlib import Path

from tools.sfra.maintext import PAIR_HASHES, digest, load_json
from tools.sfra.summary import read_csv

ROUNDS = tuple(range(30, 101, 10))
SCHEMA = 'ab_direct_calibration_reference_v1'


def file_hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def protocol_signature(run):
    run = Path(run)
    meta = load_json(run/'bridge_metadata.json')
    signature = {k: meta[k] for k in PAIR_HASHES}
    signature['partition'] = digest(read_csv(run/'partition_manifest.csv'))
    signature['witness'] = digest(load_json(run/'private_witness_manifest.json'))
    signature['execution'] = digest(load_json(run/'execution_config.json'))
    return signature


def build_reference(run):
    """Read complete AB logs only. No checkpoint, test metric or best-round choice."""
    run = Path(run)
    cfg = load_json(run/'sfra_config.json')
    transfer = cfg.get('b_transfer', {})
    expected = dict(mode='shared', calibration_profile='coverage_tradeoff', non_tail_sampling='class-cyclic',
                    tail_weight=.35, learning_rate=.3, probe_step=.1, regularization=.001, steps=2)
    if cfg.get('variant') != 'full-cp' or cfg.get('retention_weight') != 10 or cfg.get('classification_weight') != 1:
        raise ValueError('Calibration reference must be frozen Full-CP lambda10 mu1')
    if any(transfer.get(k) != v for k, v in expected.items()) or transfer.get('rounds') != list(ROUNDS):
        raise ValueError('Calibration reference must be the frozen class-cyclic w0.35 AB')
    done = load_json(run/'completion.json')
    if any(done.get(k) != v for k,v in [('completed_round',100),('b_transfer_events',8)]):
        raise ValueError('Finish the paired AB run before starting its calibration control')
    events = {}
    for rnd in ROUNDS:
        folder = run/'b_transfer_rounds'/f'r{rnd:03d}'
        cal = load_json(folder/'calibration_manifest.json')
        steps = read_csv(folder/'optimization_steps.csv')
        expected_steps = [1,2] if cal['shared_donor_ids'] else []
        if [int(x['step']) for x in steps] != expected_steps:
            raise ValueError(f'Missing/duplicate AB optimization steps at round {rnd}')
        norms = [float(x['effective_transfer_norm_after']) for x in steps]
        if not all(math.isfinite(x) and x >= 0 for x in norms):
            raise ValueError('Invalid reference effective-update norm')
        events[str(rnd)] = dict(norms=norms, feedback_clients=cal['feedback_clients'],
            batches={k:v['batches'] for k,v in cal['recipients'].items()})
    if done.get('b_transfer_optimizer_steps') != sum(len(e['norms']) for e in events.values()):
        raise ValueError('Reference completion budget differs from recorded steps')
    return dict(schema_version=SCHEMA, seed=cfg['seed'], protocol_seed=cfg['protocol_seed'],
        partition=cfg['partition'], signature=protocol_signature(run), events=events,
        scope='training-side AB effective residual norm after each step; reference-dependent diagnostic',
        source_config_sha256=digest(cfg))


def load_reference(path):
    path = Path(path)
    value = build_reference(path) if path.is_dir() else load_json(path)
    if value.get('schema_version') != SCHEMA or set(value.get('events', {})) != set(map(str, ROUNDS)):
        raise ValueError('Invalid direct-calibration reference schema/events')
    for event in value['events'].values():
        norms = event['norms']
        if len(norms) not in (0,2) or not all(math.isfinite(x) and x >= 0 for x in norms):
            raise ValueError('Invalid direct-calibration reference norms')
        clients = event['feedback_clients']
        if not clients or clients != sorted(set(clients)) or set(event['batches']) != set(map(str,clients)):
            raise ValueError('Invalid direct-calibration reference clients')
        for batches in event['batches'].values():
            if len(batches) != 2:
                raise ValueError('Reference requires two predetermined feedback batches')
            for batch in batches:
                positions = batch['tail_positions'] + batch['non_tail_positions']
                if not batch['tail_positions'] or len(set(positions)) != len(positions) or any(type(x) is not int or x < 0 for x in positions):
                    raise ValueError('Invalid calibration sample positions')
    return value


def control_config(base, reference):
    if base.get('mode') != 'shared' or base.get('calibration_profile') != 'coverage_tradeoff' or base.get('tail_weight') != .35:
        raise ValueError('Direct control requires frozen shared class-cyclic w0.35 configuration')
    if base.get('non_tail_sampling') != 'class-cyclic':
        raise ValueError('Direct control requires class-cyclic batches')
    value = dict(base)
    value.update(schema_version='ab_direct_calibration_v1', calibration_profile='direct_norm_matched',
        regularization=0., donor_rule='none', donor_pool='none', c_scope='no C; optimize shared B residual directly',
        calibration_objective='same client/group LA; project residual to paired AB step norm; no C penalty',
        norm_rule='global sqrt(sum_module ||scaling * residual_B * current_A||_F^2)',
        norm_reference=reference, norm_reference_sha256=digest(reference),
        commit_rule='fixed second projected Adam step; no test-based selection',
        empty_donor_rule='skip exactly when paired AB has no donors; no extra calibration step',
        control_limit='different parameterization and optimizer geometry; not proof of optimal donor screening')
    return value
