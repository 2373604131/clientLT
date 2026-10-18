"""Six paired cells: frozen/plain/Full-CP A crossed with shared donor-C B."""
from pathlib import Path
from types import SimpleNamespace

EXPERIMENT = 'six_ablation_method_a_includes_la_fedavg_v2'
CELLS = {
    'frozen_a': ('frozen', False),
    'plain_a': ('plain_a', False),
    'method_a': ('method_a', False),
    'frozen_a_b': ('frozen', True),
    'plain_a_b': ('plain_a', True),
    'method_a_b': ('method_a', True),
}


def phase_losses(cell):
    """A-side LA belongs to method A; plain-A controls use unadjusted CE."""
    method_a = CELLS[cell][0] == 'method_a'
    return {
        'A': dict(loss='logit_adjusted_cross_entropy' if method_a else 'cross_entropy',
                  la_tau=1. if method_a else 0., aggregation='fedavg'),
        'B': dict(loss='logit_adjusted_cross_entropy', la_tau=1., aggregation='fedavg'),
    }


def losses_for(job):
    return phase_losses(job['settings']['six_ablation_cell'])


def enabled(job):
    return job.get('settings', {}).get('six_ablation_experiment') == EXPERIMENT


def has_b(job):
    return enabled(job) and CELLS[job['settings']['six_ablation_cell']][1]


def transfer_config(mode='formal'):
    from utils.cliplora_b_shared_transfer import shared_transfer_config
    if mode not in ('formal', 'smoke'):
        raise ValueError('Expected formal or smoke mode')
    cfg = shared_transfer_config(SimpleNamespace(b_transfer_lr=.3, b_transfer_probe_step=.1,
        b_transfer_reg=.001, b_transfer_non_tail_sampling='class-cyclic', b_transfer_tail_weight=.35))
    if mode == 'smoke':
        cfg['rounds'] = [1]
    return cfg


def contract_for(cell):
    from tools.sfra.maintext import FROZEN
    arm, use_b = CELLS[cell]
    return dict(experiment=EXPERIMENT, cell=cell, seed=42, clients=30, rounds=100,
        partition='client-longtail', ordinary_phase_losses=phase_losses(cell),
        B_epochs=3, A_epochs=1, A_rounds=[] if arm == 'frozen' else list(range(1, 91)),
        A_optimizer='fresh SGD lr0.001 momentum0.9 weight_decay0; cell-specific CE or LA',
        aggregation='sample-count FedAvg for both ordinary factors in every cell',
        method_A=dict(FROZEN, classification_weight=1.) if arm == 'method_a' else None,
        method_A_components=['A-side LA tau1', 'Full-CP source-aware retention and classification preservation'] if arm == 'method_a' else [],
        method_A_classification='LA in both local A training and Full-CP classification feedback' if arm == 'method_a' else None,
        method_B=transfer_config() if use_b else None,
        phase_order='ordinary B with LA; optional shared donor-C B; local A with registered loss through90; optional Full-CP; commit',
        primary='mean committed rounds81..100 Overall/Head20/Middle60/Tail20',
        interpretation='method A contrast includes A-side LA and Full-CP together; frozen A has fewer local steps; feedback costs are additional',
        seeds_note='one training seed; rounds are not independent replicates')


def validate(job):
    s = job['settings']
    cell = s.get('six_ablation_cell')
    slots = s.get('client_concurrency')
    if (cell not in CELLS or job['arm'] != CELLS[cell][0]
            or job['mode'] not in ('smoke', 'formal') or s.get('seed') != 42
            or type(slots) is not int or not 2 <= slots <= 30
            or job['aggregation'].get('rule') != 'fedavg'
            or job['aggregation']['lambda_value'] is not None
            or job['contract'] != contract_for(cell)
            or any(s.get(k) for k in ('factorial_experiment', 'b_comparison_experiment',
                                      'a_schedule_experiment', 'partition_experiment'))):
        raise ValueError('Six-cell ablation requires CE for plain A, LA for method A and B, FedAvg, seed42 and 2..30 parallel clients')


def audit_config(run, job, cfg):
    from tools.client_aggregation.protocol import load_json
    losses = losses_for(job)
    if (cfg.get('ordinary_phase_losses') != losses
            or cfg['base_training_config'].get('ordinary_phase_losses') != losses
            or load_json(Path(run)/'ordinary_phase_losses.json') != losses):
        raise ValueError('Plain A must use CE, method A must use LA, and B must use LA; both factors use FedAvg')
    if cfg.get('b_transfer') != (transfer_config(job['mode']) if has_b(job) else None):
        raise ValueError('Six-cell B module differs from the registered shared donor-C method')


def audit_b(run, job, progress, completed):
    from tools.client_aggregation.new_a_old_b import audit_b as audit_shared
    from tools.client_aggregation.protocol import read_csv
    if has_b(job):
        audit_shared(run, job, progress, completed)
        rounds = set(job['contract']['A_rounds'])
        for row in read_csv(Path(run)/'b_transfer_rounds.csv'):
            if (str(row.get('A_updated_this_round')).lower() == 'true') != (int(row['round']) in rounds):
                raise ValueError('Shared-B record mislabels whether A was trained')
    elif (progress.get('b_transfer_events', 0) or progress.get('b_transfer_optimizer_steps', 0)
            or ((Path(run)/'b_transfer_rounds.csv').is_file() and read_csv(Path(run)/'b_transfer_rounds.csv'))):
        raise ValueError('B-disabled cell executed supplemental B')
