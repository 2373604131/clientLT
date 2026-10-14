"""Read existing Client-LT experiments and export the requested two-group table.

Run from any directory. No training or modification of source results.
"""
import csv
import hashlib
import json
import math
import statistics
from pathlib import Path

OUT = Path(__file__).resolve().parent
REPO = OUT.parents[1]
SOURCE_TABLE = REPO / 'output/longtail_benchmarks_review_20261009/combined_performance.csv'
OURS = REPO / 'output/ab_factorial_joint_v1_parallel8/new_a_new_b_joint/runs/seed42/method_a_margin'
SUITES = {
    'paper': 'paper_benchmarks_seed42_v3',
    'factor': 'factor_benchmarks_seed42_v3',
    'longtail': 'longtail_benchmarks_seed42_v1',
}
# Publication identity is separate from the implementation scope in this repo.
PAPERS = [
    ('fedavg-lora', 'FedAvg (LoRA implementation)', 'AISTATS 2017', 'paper',
     'https://proceedings.mlr.press/v54/mcmahan17a.html'),
    ('fedntd', 'FedNTD (CLIP-LoRA)', 'NeurIPS 2022', 'paper',
     'https://proceedings.neurips.cc/paper_files/paper/2022/hash/fadec8f2e65f181d777507d1df69b92f-Abstract-Conference.html'),
    ('fedlf', 'FedLF (CLIP-LoRA)', 'ACML 2024; proceedings 2025', 'longtail',
     'https://proceedings.mlr.press/v260/lu25a.html'),
    ('fedyoyo', 'FedYoYo (CLIP-LoRA)', 'ICCV 2025', 'longtail',
     'https://openaccess.thecvf.com/content/ICCV2025/papers/Yan_You_Are_Your_Own_Best_Teacher_Achieving_Centralized-level_Performance_in_ICCV_2025_paper.pdf'),
    ('fedrela', 'FedReLa (FedAvg-LoRA host)', 'ICML 2026', 'longtail',
     'https://github.com/guangzhengh/FedReLa'),
    ('fedpurel', 'FedPuReL-Global (shared stage only)', 'CVPR 2026', 'paper',
     'https://openaccess.thecvf.com/content/CVPR2026/html/Hou_Fine-Tuning_Impairs_the_Balancedness_of_Foundation_Models_in_Long-tailed_Personalized_CVPR_2026_paper.html'),
    ('rolora', 'RoLoRA (CLIP-LoRA)', 'NeurIPS 2025', 'factor',
     'https://proceedings.neurips.cc/paper_files/paper/2025/hash/adf17e7346b6be4c2f2bb40de572e5bc-Abstract-Conference.html'),
    ('fedsvd', 'FedSVD (CLIP-LoRA, non-DP)', 'NeurIPS 2025', 'factor',
     'https://papers.nips.cc/paper_files/paper/2025/hash/ad922aa85d4027ff3502e8e5f406e828-Abstract-Conference.html'),
    ('lora-a2', 'LoRA-A2 (CLIP-LoRA)', 'ACL 2025', 'factor',
     'https://aclanthology.org/2025.acl-long.19/'),
    ('capt', 'CAPT (fixed aggregation frequency)', 'arXiv 2025; ACM MM 2026 accepted per author page', 'paper',
     'https://arxiv.org/abs/2503.06993'),
]


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def manifest(path):
    fields = ('raw_sample_id', 'class_id', 'client_id', 'local_position')
    return sorted(tuple(int(row[k]) for k in fields) for row in read_csv(path))


inputs = set()
saved = {row['method']: row for row in read_csv(SOURCE_TABLE)}
inputs.add(SOURCE_TABLE)
our_manifest = manifest(OURS / 'partition_manifest.csv')
our_meta = read_json(OURS / 'bridge_metadata.json')
assert len(our_manifest) == 10847
inputs.update([OURS / 'partition_manifest.csv', OURS / 'bridge_metadata.json'])
protocol_checks = {}
for key, name in SUITES.items():
    root = REPO / 'output' / (name + '_analysis') / name
    meta = read_json(root / 'protocol/bridge_metadata.json')
    assert manifest(root / 'protocol/partition_manifest.csv') == our_manifest, key
    comparisons = {field: meta[field] == our_meta[field]
                   for field in ('pool_sha256', 'test_sha256', 'schedule_sha256', 'classnames')}
    assert all(comparisons.values()), (key, comparisons)
    protocol_checks[key] = dict(identical_training_assignment=True, **comparisons)
    inputs.update([root / 'protocol/partition_manifest.csv', root / 'protocol/bridge_metadata.json'])

results = []
for method, label, venue, suite, url in PAPERS:
    name = SUITES[suite]
    root = REPO / 'output' / (name + '_analysis') / name / 'runs' / method
    done = read_json(root / 'completion.json')
    inputs.add(root / 'completion.json')
    head_values, tail_values = [], []
    for rnd in range(81, 101):
        path = root / 'rounds' / f'r{rnd:03d}.json'
        record = read_json(path)
        inputs.add(path)
        assert record['round'] == rnd
        assert len(record['correct']) == len(record['total']) == 100
        assert all(n == 100 for n in record['total'])
        assert all(isinstance(c, int) and 0 <= c <= n
                   for c, n in zip(record['correct'], record['total']))
        values = [100 * c / n for c, n in zip(record['correct'], record['total'])]
        head_values.append(statistics.mean(values[:80]))
        tail_values.append(statistics.mean(values[80:]))
    head, tail = statistics.mean(head_values), statistics.mean(tail_values)
    merged = (20 * float(saved[method]['last20_head20_acc']) +
              60 * float(saved[method]['last20_middle60_acc'])) / 80
    assert math.isclose(head, merged, abs_tol=1e-9)
    assert math.isclose(head, float(saved[method]['last20_non_tail_acc']), abs_tol=1e-9)
    assert math.isclose(tail, float(saved[method]['last20_bottom20_tail_acc']), abs_tol=1e-9)
    results.append(dict(method=method, label=label, publication=venue, paper_url=url,
                        head80=head, tail20=tail, source=str(root.relative_to(REPO)),
                        local_optimizer_steps=done['optimizer_steps']))

done = read_json(OURS / 'completion.json')
assert done['completed_round'] == 100 and done['official_test_passes'] == 101
inputs.add(OURS / 'completion.json')
curve = {int(row['round']): row for row in read_csv(OURS / 'round_metrics.csv')}
assert sorted(curve) == list(range(101))
inputs.add(OURS / 'round_metrics.csv')
heads, tails = [], []
for rnd in range(81, 101):
    path = OURS / f'per_class_accuracy_epoch_{rnd - 1}.csv'
    rows = read_csv(path)
    inputs.add(path)
    classes = {int(row['class_id']): float(row['per_class_acc']) for row in rows}
    assert len(rows) == 100 and set(classes) == set(range(100))
    head = statistics.mean(classes[c] for c in range(80))
    tail = statistics.mean(classes[c] for c in range(80, 100))
    assert math.isclose(head, float(curve[rnd]['non_tail_acc']), abs_tol=1e-9)
    assert math.isclose(tail, float(curve[rnd]['bottom20_tail_acc']), abs_tol=1e-9)
    heads.append(head)
    tails.append(tail)
ours = dict(method='new_a_new_b_joint', label='New A + New B + joint', publication='Ours',
            paper_url='', head80=statistics.mean(heads), tail20=statistics.mean(tails),
            source=str(OURS.relative_to(REPO)),
            local_optimizer_steps=done['total_local_optimizer_steps'],
            A_correction_steps=done['functional_correction_steps'],
            B_correction_steps=done['b_transfer_optimizer_steps'])
results.append(ours)
results.sort(key=lambda row: row['tail20'], reverse=True)

with (OUT / 'head_tail_table.csv').open('w', encoding='utf-8-sig', newline='') as stream:
    writer = csv.writer(stream)
    writer.writerow(['method', 'publication', 'head80_acc_percent', 'tail20_acc_percent', 'paper_url'])
    for row in results:
        writer.writerow([row['label'], row['publication'], f"{row['head80']:.4f}",
                         f"{row['tail20']:.4f}", row['paper_url']])

comparisons = [dict(reference=row['method'],
                    head80_delta_pp=ours['head80'] - row['head80'],
                    tail20_delta_pp=ours['tail20'] - row['tail20'])
               for row in results if row is not ours]
audit = dict(
    checked_on='2026-10-14', protocol='CIFAR-100-LT / Client-LT / seed42',
    reporting_window=list(range(81, 101)), classes=dict(head80=list(range(80)), tail20=list(range(80, 100))),
    formula='Head80 = (20 * Head20 + 60 * Middle60) / 80',
    publication_policy='Include public arXiv papers, per explicit user reply; retain own new AB as target.',
    publication_sources_checked_on='2026-10-14',
    capt_acceptance_source='https://jasonyanglu.github.io/publications/',
    protocols=protocol_checks, results=results, comparisons=comparisons,
    excluded=dict(ffa_lora='No returned result; do not substitute another fixed-A run.',
                  internal_controls='Old A/AB, S and ablations are omitted from the requested paper-only table.',
                  fedavg_lora_la='Constructed loss-control combination, not a separate named paper method. Still relevant to efficacy evaluation.'),
    limitations=[
        'One training seed and one partition seed; last 20 rounds are not 20 independent trials.',
        'Shared sample assignment, test set and schedule verified; implementation versions and some environments differ.',
        'These are repository adaptations, not published benchmark numbers or exact original-paper reproductions.',
        'New AB uses 137280 local steps plus 270 A and 16 B correction steps; most baselines use 105600 local steps.',
        'LoRA-A2 uses 140800 local steps. Optimizer-step counts are not equal to FLOPs across methods.',
        'Best tail accuracy among tested paper baselines is supported; unrestricted Client-LT SOTA is not established.',
    ],
    source_sha256={str(path.relative_to(REPO)): hashlib.sha256(path.read_bytes()).hexdigest()
                   for path in sorted(inputs)},
)
(OUT / 'audit.json').write_text(json.dumps(audit, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
print(json.dumps(dict(results=results, comparisons=comparisons, protocols=protocol_checks),
                 ensure_ascii=False, indent=2))
