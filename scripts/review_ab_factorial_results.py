"""Audit a copied four-cell result without rewriting its registered server paths."""
import argparse
from pathlib import Path
from statistics import mean
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))

from scripts.run_ab_validation import write_json
from scripts.run_cliplora_joint_aggregation import make_job
from tools.client_aggregation.ab_factorial import CELLS, pair_check
from tools.client_aggregation.analysis import METRICS, audit_run, stage_comparisons
from tools.client_aggregation.protocol import code_hashes, load_json, read_csv, write_table

CONTRASTS = (
    ('new_A_without_B', 'new_a_joint', 'plain_joint'),
    ('new_B_given_new_A', 'new_a_new_b_joint', 'new_a_joint'),
    ('new_A_given_new_B', 'new_a_new_b_joint', 'plain_new_b_joint'),
    ('new_B_given_plain_A', 'plain_new_b_joint', 'plain_joint'),
)


def review(root, output):
    root, output = Path(root).resolve(), Path(output).resolve()
    suite = load_json(root/'factorial_plan.json')
    current_code = code_hashes(REPO)
    results, endpoints, classes, audits, performance, a_diagnostics, costs = {}, {}, {}, [], [], [], []
    for cell, (arm, _) in CELLS.items():
        if suite['cells'][cell]['reused']:
            raise ValueError('This copied-result review requires all four rerun cells in the supplied root')
        plan = load_json(root/cell/'experiment_plan.json')
        job = make_job(plan, arm)
        run = root/cell/job['run']
        result = audit_run(run, job)
        results[cell] = result
        audits.append(dict(cell=cell, status='complete', completed_round=100,
            current_source_differences=[k for k, v in job['code_sha256'].items() if current_code.get(k) != v]))
        window = result['curves'][81:101]
        endpoints[cell] = {m: mean(float(r[m]) for r in window) for m in METRICS}
        performance.append(dict(cell=cell, **endpoints[cell]))
        classes[cell] = np.mean([np.bincount(p['class_id'], weights=p['correct'], minlength=100)
            for rnd, p in result['predictions'].items() if 81 <= rnd <= 100], axis=0)
        progress = result['progress']
        a_cost = read_csv(run/'sfra_costs.csv')
        b_cost = read_csv(run/'b_transfer_rounds.csv') if (run/'b_transfer_rounds.csv').is_file() else []
        costs.append(dict(cell=cell, elapsed_hours=progress['elapsed_seconds']/3600,
            ordinary_SGD_steps=progress['normal_optimizer_steps']+progress['extra_optimizer_steps'],
            A_correction_steps=progress['functional_correction_steps'],
            A_feedback_hours=sum(float(r['seconds']) for r in a_cost)/3600,
            B_steps=progress.get('b_transfer_optimizer_steps', 0),
            B_logged_seconds=sum(float(r['seconds']) for r in b_cost)))
        if arm == 'method_a_margin':
            rows = [r for r in read_csv(run/'sfra_rounds.csv') if int(r['round']) <= 90]
            relative = []
            for r in rows:
                event = load_json(run/f'events/r{int(r["round"]):03d}_c000_main_refresh_A/event.json')
                relative.append(float(r['radius'])*float(r['correction_z_norm'])/float(event['delta_a_norm']))
            a_diagnostics.append(dict(cell=cell, correction_rounds=sum(int(r['correction_steps']) == 3 for r in rows),
                nonzero_rounds=sum(r['nonzero_correction'] == 'True' for r in rows),
                mean_fraction_of_radius=mean(float(r['correction_z_norm']) for r in rows),
                mean_fraction_of_ordinary_A_norm=mean(relative),
                mean_correction_norm=mean(float(r['radius'])*float(r['correction_z_norm']) for r in rows),
                mean_keep_loss_reduction=mean(float(r['ordinary_keep'])-float(r['final_keep']) for r in rows),
                mean_class_loss_reduction=mean(float(r['ordinary_class'])-float(r['final_class']) for r in rows),
                mean_deficits_before=mean(float(r['ordinary_deficit_count']) for r in rows),
                mean_deficits_after=mean(float(r['final_deficit_count']) for r in rows)))
        print('AUDIT PASSED', cell, endpoints[cell], flush=True)
    pairs, comparisons, per_class = [], [], []
    for name, left, right in CONTRASTS:
        pair = pair_check(results[left], results[right])
        if pair['status'] == 'mismatch':
            raise ValueError('Unpaired results: '+name+' '+str(pair['mismatches']))
        # The shared legacy audit returns a frozen-arm label and an incomplete compute flag.
        # Keep its identity checks, then explicitly label this factorial contrast and cost scope.
        pair.update(contrast=name, left=left, right=right, local_SGD_budget_matched=True,
            total_compute_budget_matched=False)
        pair.pop('budget_matched', None)
        pair['delta_pp'] = {m: endpoints[left][m]-endpoints[right][m] for m in METRICS}
        pairs.append(pair)
        tail = results[left]['groups']['bottom20_tail_acc']
        delta = classes[left]-classes[right]
        changed, helped, harmed = [], [], []
        for rnd in range(81, 101):
            a, b = (results[cell]['predictions'][rnd] for cell in (left, right))
            changed.append(int((a['prediction'] != b['prediction']).sum()))
            helped.append(int((a['correct'] & ~b['correct']).sum()))
            harmed.append(int((~a['correct'] & b['correct']).sum()))
        comparisons.append(dict(contrast=name, left=left, right=right, **pair['delta_pp'],
            mean_prediction_changes=mean(changed), mean_corrected=mean(helped), mean_damaged=mean(harmed),
            tail_classes_improved=int((delta[tail] > 1e-9).sum()),
            tail_classes_worse=int((delta[tail] < -1e-9).sum()),
            tail_classes_tied=int((abs(delta[tail]) <= 1e-9).sum())))
        per_class.extend(dict(contrast=name, class_id=c, delta_pp=float(delta[c]), is_tail=c in tail) for c in range(100))
    interaction = {m: endpoints['new_a_new_b_joint'][m]-endpoints['new_a_joint'][m]
        -endpoints['plain_new_b_joint'][m]+endpoints['plain_joint'][m] for m in METRICS}
    output.mkdir(parents=True, exist_ok=True)
    for name, rows in dict(performance=performance, contrasts=comparisons, costs=costs,
            a_diagnostics=a_diagnostics, per_class_deltas=per_class,
            stage_effects=[dict(row, cell=cell) for cell, r in results.items() for row in stage_comparisons(r)]).items():
        write_table(output/(name+'.csv'), rows)
    write_json(output/'audit.json', dict(source_root=str(root), original_results_modified=False,
        audits=audits, pairs=pairs, interaction_pp=interaction,
        independent_seeds=1, endpoint='pre-registered committed rounds81..100 mean',
        notes=['Original contrasts.json inherits an incorrect frozen-arm label; use left/right identities.',
               'Wall times across different GPUs are descriptive, not a controlled speed benchmark.',
               'B resets CUDA peak counters at events; process peak memory is not comparable across cells.']))
    print('Review artifacts:', output)


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path('output/ab_factorial_joint_v1_parallel8'))
    p.add_argument('--output', type=Path, default=Path('presentation/ab_factorial_review_20261014'))
    args = p.parse_args()
    review(args.root, args.output)
