"""CPU-only audit and derived tables for the retained FedAvg projection diagnosis.

This reads the exported predictions; it neither trains nor revalidates absent .pt
files. Bootstrap intervals are pointwise, conditional on the saved fitted models.
"""
import argparse
import csv
import hashlib
import json
import sys
import zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from tools.a_projection_diagnosis.summary import paired_interval


GROUPS = dict(overall=range(100), head20=range(20), middle60=range(20, 80), tail20=range(80, 100))
ANCHORS = (20, 50, 80)
SNAPSHOTS = (0, 20, 50, 80, 90, 100)


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8'))


def require(condition, message):
    if not condition:
        raise ValueError(message)


def class_mean(values, labels):
    return np.array([values[labels == c].mean() for c in range(100)])


def paired(a, b):
    require(np.array_equal(a['sample_id'], b['sample_id']) and
            np.array_equal(a['labels'], b['labels']), 'Unpaired predictions')


def plot_review(output, curves, primary, interventions, contrasts, modules):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.size': 10, 'axes.spines.top': False, 'axes.spines.right': False})
    fig, axes = plt.subplots(2, 2, figsize=(12, 8.8))
    ax = axes[0, 0]
    for group, color in [('head20', '#4477AA'), ('tail20', '#CC6677')]:
        rows = [r for r in curves if r['group'] == group]
        ax.plot([int(r['round']) for r in rows], [float(r['accuracy']) for r in rows],
                '-o', label=group, color=color)
    ax.axvspan(90, 100, color='gray', alpha=.12, label='B-only phase')
    ax.set(title='A. Full-model accuracy', xlabel='Committed round', ylabel='Accuracy (%)')
    ax.legend(fontsize=8)
    ax = axes[0, 1]
    for offset, combo, label, color in [(-.13, 'fixed_input', 'Fixed input: Q(A_t) X_0', '#228833'),
                                       (.13, 'live_input', 'Live input: Q(A_t) X_t', '#AA3377')]:
        rows = [next(r for r in primary if r['module'] == m and r['combination'] == combo)
                for m in modules[:6]]
        means = np.array([float(r['mean']) for r in rows])
        low = np.array([float(r['low']) for r in rows])
        high = np.array([float(r['high']) for r in rows])
        ax.errorbar(means, np.arange(6) + offset, xerr=[means-low, high-means], fmt='o',
                    capsize=3, label=label, color=color)
    ax.axvline(0, color='gray', linewidth=.8)
    ax.set(yticks=np.arange(6), yticklabels=[m.split('resblocks.')[-1].replace('.attn.', ' ').replace('_proj.w_lora_A', '') for m in modules[:6]],
           title='B. Tail probe: mean(rounds 80, 90) - round 20', xlabel='Accuracy change (percentage points)')
    ax.invert_yaxis()
    ax.legend(fontsize=8, loc='lower left')
    ax = axes[1, 0]
    for offset, candidate, label, color in [(-.23, 'space_raw_full', 'Full A update', '#4477AA'),
            (0, 'space_raw_parallel', 'Within old row space', '#EEBB55'),
            (.23, 'space_raw_outside', 'Outside old row space', '#CC6677')]:
        rows = [next(r for r in interventions if r['round'] == n and r['candidate'] == candidate and
                     r['split'] == 'remaining' and r['metric'] == 'cosine_margin') for n in ANCHORS]
        means = np.array([r['mean'] for r in rows]) * 1e6
        low = np.array([r['low'] for r in rows]) * 1e6
        high = np.array([r['high'] for r in rows]) * 1e6
        ax.bar(np.arange(3)+offset, means, .22, label=label, color=color,
               yerr=[means-low, high-means], capsize=3)
    ax.axhline(0, color='gray', linewidth=.8)
    ax.set(xticks=np.arange(3), xticklabels=ANCHORS, title='C. Single A-stage intervention vs no A update',
           xlabel='Anchor round; remaining test samples', ylabel='Tail cosine-margin change (x 1e-6)')
    ax.legend(fontsize=8)
    ax = axes[1, 1]
    rows = [next(r for r in contrasts if int(r['round']) == n and r['left'] == 'weight_matched_tailrw16' and
                 r['split'] == 'remaining' and r['metric'] == 'cosine_margin') for n in ANCHORS]
    means = np.array([float(r['mean']) for r in rows]) * 1e6
    low = np.array([float(r['low']) for r in rows]) * 1e6
    high = np.array([float(r['high']) for r in rows]) * 1e6
    ax.errorbar(ANCHORS, means, yerr=[means-low, high-means], fmt='o-', capsize=4, color='#228833')
    ax.axhline(0, color='gray', linewidth=.8)
    ax.set(xticks=ANCHORS, title='D. TailRW16 - FedAvg, matched effective norm',
           xlabel='Anchor round; remaining test samples', ylabel='Tail cosine-margin difference (x 1e-6)')
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.15)
    fig.suptitle('FedAvg trajectory, seed 42, fixed tail classes 80-99', fontsize=14)
    fig.text(.5, .015, 'Intervals: pointwise 95% paired image bootstrap, conditional on fitted models.\n'
             'Panels C and D use different vertical scales; remaining samples are part of the same official test set.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .065, 1, .965))
    fig.savefig(output / 'evidence_overview.png', dpi=180)
    fig.savefig(output / 'evidence_overview.pdf')
    plt.close(fig)


def review(args):
    root, output = args.root.resolve(), args.output.resolve()
    require(root != output and root not in output.parents, 'Output must be outside input result directory')
    receipt = read_json(args.receipt)
    require(receipt['fixed_class_groups'] == dict(head_ids=list(range(20)), middle_ids=list(range(20, 80)),
                                                tail_ids=list(range(80, 100))), 'Class groups changed')
    require(receipt['completed_round'] == 100 and receipt['a90_equals_a100'], 'Unexpected training completion')
    require(receipt['frozen_base_and_initial_hashes_verified'], 'Unverified server receipt')
    for n in ANCHORS:
        item = next(r for r in receipt['original_proposals'] if r['round'] == n)
        require(item['clients'] == 30 and sorted(item['selected']) == list(range(30)) and
                item['proposals'] == 'original_historical_uploads' and
                item['reconstruction_max_abs_error'] == 0, 'Invalid historical proposal receipt')
    tasks = [root / 'fedavg/e1'] + [root / 'fedavg' / ('round%03d' % n) for n in ANCHORS]
    for task in tasks:
        require((task / 'completion.json').is_file(), 'Incomplete task: ' + str(task))
    environments = [read_json(t / 'environment.json') for t in tasks]
    for key in ('fit_sha256', 'eval_sha256', 'code_fingerprint', 'modules'):
        require(all(e[key] == environments[0][key] for e in environments), 'Task identity mismatch: ' + key)
    modules = environments[0]['modules'] + ['combined48']
    checks = [c for t in tasks for c in read_json(t / 'numerical_checks.json')]
    require(all(c['passed'] for c in checks), 'Failed recorded numerical check')
    manifest = read_csv(tasks[0] / 'audit_eval_manifest.csv')
    fit = read_csv(tasks[0] / 'probe_fit_manifest.csv')
    require(len({int(r['raw_sample_id']) for r in fit}) == len(fit), 'Duplicated fit samples')
    require(len(manifest) == 10000 and len({int(r['sample_id']) for r in manifest}) == 10000, 'Invalid test IDs')
    expected = {}
    for split, count in [('test', 100), ('panel', 20), ('remaining', 80)]:
        rows = [r for r in manifest if split == 'test' or r['split'] == split]
        require(all(sum(int(r['class_id']) == c for r in rows) == count for c in range(100)), 'Invalid panel counts')
        expected[split] = {int(r['sample_id']): int(r['class_id']) for r in rows}
    for t in tasks:
        require(read_csv(t / 'audit_eval_manifest.csv') == manifest and
                read_csv(t / 'probe_fit_manifest.csv') == fit, 'Sampling manifests differ')
    arrays = {}
    for path in sorted((root / 'fedavg').rglob('*.npz')):
        with np.load(path, allow_pickle=False) as saved:
            a = dict(saved)
        ids, labels = a['sample_id'], a['labels']
        split = 'panel' if path.stem.endswith('_panel') else 'remaining' if path.stem.endswith('_remaining') else 'test'
        require(len(set(ids)) == len(ids) and dict(zip(ids.tolist(), labels.tolist())) == expected[split], 'Wrong prediction identities: ' + str(path))
        for key, value in a.items():
            require(value.shape == labels.shape and np.isfinite(value).all(), 'Invalid prediction array')
            if key.endswith('prediction'):
                require(value.dtype.kind in 'iu' and (value >= 0).all() and (value < 100).all(), 'Invalid predicted class')
        arrays[path] = a

    # Recompute every CSV metric, including all 100 per-class rows, from saved predictions.
    verified_rows = 0
    def verify_table(task, filename, is_probe):
        nonlocal verified_rows
        cached = {}
        for row in read_csv(task / filename):
            stem = row['candidate'] + ('_' + row['split'] if is_probe or task.name != 'e1' else '')
            path = task / ('probe_predictions' if is_probe else 'model_predictions') / (stem + '.npz')
            key = (path, row.get('module'))
            if key not in cached:
                a = arrays[path]
                y = a['labels']
                prefix = 'm%d_' % modules.index(row['module']) if is_probe else ''
                correct = a[prefix + 'prediction'] == y
                metrics = dict(accuracy=100 * class_mean(correct, y), samples=np.bincount(y, minlength=100))
                for name in (('squared_error',) if is_probe else ('margin', 'ce')):
                    metrics[name] = class_mean(a[prefix + name], y)
                if not is_probe:
                    base = arrays[task / 'model_predictions' / ('r000.npz' if task.name == 'e1' else 'no_A_' + row['split'] + '.npz')]
                    paired(a, base)
                    old = base['prediction'] == y
                    metrics.update(lost=np.bincount(y[old & ~correct], minlength=100),
                                   gained=np.bincount(y[~old & correct], minlength=100),
                                   delta_accuracy=100 * class_mean(correct.astype(float) - old, y))
                    for name in ('margin', 'ce'):
                        metrics['delta_' + name] = class_mean(a[name] - base[name], y)
                cached[key] = metrics
            indices = list(GROUPS[row['group']]) if row['group'] in GROUPS else [int(row['group'].split('_')[1])]
            for name, values in cached[key].items():
                actual = values[indices].sum() if name in ('samples', 'lost', 'gained') else values[indices].mean()
                require(np.isclose(actual, float(row[name]), atol=1e-10, rtol=1e-7), 'CSV mismatch: %s %s %s' % (filename, key, name))
            verified_rows += 1
    for task in tasks:
        verify_table(task, 'full_model_curves.csv' if task.name == 'e1' else 'full_model_candidates.csv', False)
        verify_table(task, 'representation_curves.csv' if task.name == 'e1' else 'projection_probes.csv', True)
    verify_table(tasks[0], 'shuffled_label_control.csv', True)

    e1_models = {n: arrays[tasks[0] / 'model_predictions' / ('r%03d.npz' % n)] for n in SNAPSHOTS}
    batch_size = read_json(root / 'fedavg/run_identity.json')['eval_batch_size']
    reconstruction = []
    for n, task in zip(ANCHORS, tasks[1:]):
        pieces = [arrays[task / 'model_predictions' / ('weight_raw_fedavg_' + split + '.npz')] for split in ('panel', 'remaining')]
        joined = {k: np.concatenate([p[k] for p in pieces]) for k in pieces[0]}
        order = np.argsort(joined['sample_id'])
        reference = e1_models[n]
        reforder = np.argsort(reference['sample_id'])
        def batch_sizes(p):
            sizes = np.full(len(p['sample_id']), batch_size)
            remainder = len(sizes) % batch_size
            if remainder:
                sizes[-remainder:] = remainder
            return sizes
        different_batch_shape = np.concatenate([batch_sizes(p) for p in pieces])[order] != batch_sizes(reference)[reforder]
        errors = {}
        for key in joined:
            delta = np.abs(joined[key][order].astype(float) - reference[key][reforder])
            error = float(np.max(delta))
            tolerance = {'margin': 1e-6, 'ce': 1e-4}.get(key, 0)
            require(error <= tolerance and not (delta[~different_batch_shape] != 0).any(),
                    'Committed aggregate mismatch beyond last-batch FP32 tolerance: ' + key)
            errors[key] = error
        for split in ('panel', 'remaining'):
            a = arrays[task / 'model_predictions' / ('weight_raw_fedavg_' + split + '.npz')]
            b = arrays[task / 'model_predictions' / ('space_raw_full_' + split + '.npz')]
            require(all(np.array_equal(a[k], b[k]) for k in a), 'Duplicate raw aggregate disagrees')
        reconstruction.append(dict(round=n, class_predictions_identical=True, max_prediction_array_errors=errors,
                                   floating_differences_confined_to_changed_batch_shapes=True))

    archive_count = None
    if args.archive.is_file():
        archive_count = 0
        with zipfile.ZipFile(args.archive) as z:
            for info in z.infolist():
                if info.is_dir():
                    continue
                path = root.parent / info.filename
                require(path.is_file() and hashlib.sha256(path.read_bytes()).digest() == hashlib.sha256(z.read(info)).digest(),
                        'Archive and extracted file differ: ' + info.filename)
                archive_count += 1

    primary = [r for r in read_csv(root / 'analysis/primary_representation_change.csv') if r['trajectory'] == 'fedavg' and r['group'] == 'tail20']
    for row in primary:
        values = [arrays[tasks[0] / 'probe_predictions' / ('r%03d_%s_test.npz' % (n, row['combination']))] for n in (20, 80, 90)]
        a, b, c = values
        paired(a, b); paired(a, c)
        y = a['labels']; key = 'm%d_prediction' % modules.index(row['module'])
        delta = 100 * (((b[key] == y).astype(float) + (c[key] == y)) / 2 - (a[key] == y))
        check = paired_interval(delta, y, GROUPS['tail20'])
        require(all(np.isclose(check[k], float(row[k]), atol=1e-12) for k in check), 'Primary bootstrap mismatch')

    model_changes, class_changes = [], []
    for name, later, base, scope in [('mean80_90_minus20', (80, 90), 20, 'prespecified_primary'),
                                    ('round100_minus50', (100,), 50, 'descriptive_posthoc'),
                                    ('round100_minus90', (100,), 90, 'B_only_descriptive')]:
        a = e1_models[base]; y = a['labels']
        for n in later:
            paired(a, e1_models[n])
        delta = 100 * (np.mean([(e1_models[n]['prediction'] == y).astype(float) for n in later], axis=0) - (a['prediction'] == y))
        for group, classes in GROUPS.items():
            model_changes.append(dict(contrast=name, scope=scope, group=group, metric='accuracy_pp', **paired_interval(delta, y, classes)))
        if base == 20:
            for c in GROUPS['tail20']:
                class_changes.append(dict(class_id=c, model_accuracy_change_pp=float(delta[y == c].mean())))
    for combo in ('fixed_input', 'live_input', 'fixed_A'):
        a, b, c = [arrays[tasks[0] / 'probe_predictions' / ('r%03d_%s_test.npz' % (n, combo))] for n in (20, 80, 90)]
        y = a['labels']
        delta = 100 * (((b['m6_prediction'] == y).astype(float) + (c['m6_prediction'] == y)) / 2 - (a['m6_prediction'] == y))
        for row in class_changes:
            row['aux_combined48_' + combo + '_change_pp'] = float(delta[y == row['class_id']].mean())

    interventions, decompositions = [], []
    verified_decompositions = 0
    for n, task in zip(ANCHORS, tasks[1:]):
        for split in ('panel', 'remaining'):
            base = arrays[task / 'model_predictions' / ('no_A_' + split + '.npz')]
            y = base['labels']; old = base['prediction'] == y; tail = np.isin(y, GROUPS['tail20'])
            for candidate in read_json(task / 'completion.json')['candidates']:
                if candidate == 'no_A' or candidate.startswith('local_'):
                    continue
                a = arrays[task / 'model_predictions' / (candidate + '_' + split + '.npz')]
                paired(a, base)
                correct = a['prediction'] == y
                for metric, delta in [('accuracy_pp', 100 * (correct.astype(float) - old)), ('cosine_margin', a['margin'] - base['margin'])]:
                    interventions.append(dict(round=n, split=split, candidate=candidate, reference='no_A', group='tail20', metric=metric,
                        **paired_interval(delta, y, GROUPS['tail20']), lost=int((tail & old & ~correct).sum()), gained=int((tail & ~old & correct).sum())))
        rows = read_csv(task / 'response_decomposition.csv')
        weights = read_json(task / 'weights.json')
        local_rows = read_csv(task / 'local_class_response.csv')
        holds = {(int(r['client_id']), int(r['class_id'])): int(r['class_count']) > 0 for r in local_rows}
        base = arrays[task / 'model_predictions/no_A_panel.npz']
        effects = {j: class_mean(arrays[task / 'model_predictions' / ('local_%02d_panel.npz' % j)]['margin'] - base['margin'], base['labels'])
                   for j in range(30)}
        for saved in rows:
            c, rule = int(saved['class_id']), saved['rule']
            w = {int(j): v for j, v in weights[rule].items()}
            a = arrays[task / 'model_predictions' / ('weight_raw_' + rule + '_panel.npz')]
            actual = dict(aggregate_delta_margin=float((a['margin'] - base['margin'])[base['labels'] == c].mean()),
                          weighted_local_delta_margin=sum(w[j] * float(effects[j][c]) for j in w))
            for held, prefix in ((True, 'holder'), (False, 'nonholder')):
                ids = [j for j in w if holds[j, c] == held]
                actual[prefix + '_weight'] = sum(w[j] for j in ids)
                actual[prefix + '_positive'] = sum(w[j] * float(effects[j][c]) for j in ids if effects[j][c] > 0)
                actual[prefix + '_negative'] = sum(w[j] * float(effects[j][c]) for j in ids if effects[j][c] < 0)
            actual['nonlinear_gap'] = actual['aggregate_delta_margin'] - actual['weighted_local_delta_margin']
            require(all(np.isclose(v, float(saved[k]), atol=1e-12, rtol=1e-7) for k, v in actual.items()), 'Local response decomposition mismatch')
            verified_decompositions += 1
        for rule in ('fedavg', 'tailrw16', 'joint'):
            selected = [r for r in rows if r['rule'] == rule and int(r['class_id']) >= 80]
            require(len(selected) == 20, 'Missing tail decomposition classes')
            row = dict(round=n, rule=rule)
            for key in selected[0]:
                if key not in ('round', 'rule', 'class_id'):
                    row[key] = float(np.mean([float(r[key]) for r in selected]))
            row['holder_net'] = row['holder_positive'] + row['holder_negative']
            row['nonholder_net'] = row['nonholder_positive'] + row['nonholder_negative']
            require(np.isclose(row['holder_net'] + row['nonholder_net'] + row['nonlinear_gap'], row['aggregate_delta_margin'], atol=1e-12), 'Decomposition does not sum')
            decompositions.append(row)
    contrasts = [r for r in read_csv(root / 'analysis/paired_candidate_contrasts.csv') if r['group'] == 'tail20']
    for row in contrasts:
        folder = root / 'fedavg' / ('round%03d' % int(row['round'])) / 'model_predictions'
        a, b = [arrays[folder / (row[side] + '_' + row['split'] + '.npz')] for side in ('left', 'right')]
        paired(a, b); y = a['labels']
        delta = (100 * ((a['prediction'] == y).astype(float) - (b['prediction'] == y))
                 if row['metric'] == 'accuracy_pp' else a['margin'] - b['margin'])
        check = paired_interval(delta, y, GROUPS['tail20'])
        require(all(np.isclose(check[k], float(row[k]), atol=1e-12) for k in check), 'Candidate bootstrap mismatch')

    output.mkdir(parents=True, exist_ok=True)
    curves = [r for r in read_csv(tasks[0] / 'full_model_curves.csv') if r['group'] in GROUPS]
    for name, rows in [('full_model_curves', curves), ('full_model_changes', model_changes), ('tail_probe_primary', primary),
                       ('tail_class_primary', class_changes), ('tail_interventions_vs_no_A', interventions),
                       ('tail_candidate_contrasts', contrasts), ('tail_response_decomposition', decompositions)]:
        write_csv(output / (name + '.csv'), rows)
    audit = dict(selected_fedavg_scope_complete=True, full_two_trajectory_scope_complete=False,
                 missing_scope='TailRW16 longitudinal E1 trajectory', fixed_tail_ids=list(range(80, 100)),
                 recorded_numerical_checks_passed=len(checks), prediction_npz_checked=len(arrays),
                 csv_metric_rows_recomputed=verified_rows, primary_tail_bootstraps_recomputed=len(primary),
                 candidate_tail_bootstraps_recomputed=len(contrasts), archive_files_hash_matched=archive_count,
                 local_response_decomposition_rows_recomputed=verified_decompositions,
                 fit_samples=len(fit), tail_fit_counts={str(c): sum(int(r['class_id']) == c for r in fit) for c in range(80, 100)},
                 committed_vs_reconstructed_aggregate_checks=reconstruction,
                 server_receipt_checked=True, model_tensor_hashes_recomputed_locally=False,
                 limitation='Raw .pt files absent locally; historical tensor verification relies on supplied server receipt.',
                 uncertainty='Pointwise 95% class-stratified image bootstrap; not seed or probe-fit uncertainty.',
                 independent_training_seeds=1, input_root=str(root), receipt_sha256=hashlib.sha256(args.receipt.read_bytes()).hexdigest())
    (output / 'audit.json').write_text(json.dumps(audit, indent=2) + '\n', encoding='utf-8')
    plot_review(output, curves, primary, interventions, contrasts, modules)
    print(json.dumps(audit, indent=2))
    print('Review artifacts: ' + str(output))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--root', type=Path, default=Path('output/diagnosis_results/diagnosis'))
    parser.add_argument('--receipt', type=Path, default=Path('output/retained_inputs_verified.json'))
    parser.add_argument('--archive', type=Path, default=Path('output/diagnosis_results.zip'))
    parser.add_argument('--output', type=Path, default=Path('output/a_projection_review_20261012'))
    review(parser.parse_args())
