"""Small, dependency-free result I/O and the frozen pilot protocol."""
import csv
import json
from pathlib import Path

TOPOLOGIES = ('client-longtail', 'noniid-labeldir-fine')
METRICS = ('overall', 'head20', 'middle60', 'nontail80', 'tail20', 'probe_tail_logodds')
PROTOCOL = {
    'version': 'factor_roles_v1', 'seed': 42, 'rounds': [20, 50, 80],
    'local_epochs': 1, 'batch_size': 32, 'lr': .001, 'momentum': .9,
    'weight_decay': 0., 'la_tau': 1., 'dirichlet_beta': .5,
    'norm_matching': 'minimum_positive_effective_norm_over_all_16_candidates_per_anchor',
    'client_overall_floor_pp': -.5, 'client_head_middle_floor_pp': -1.,
    'dir_tail_floor_pp': -.1,
    'class_overall_floor_pp': -.5, 'class_head_middle_floor_pp': -1.,
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + '\n', encoding='utf-8')


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(k for row in rows for k in row))
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
