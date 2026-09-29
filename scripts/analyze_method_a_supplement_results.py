"""Read saved seed42 results; audit controls and export descriptive analyses only."""
import csv
import json
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.sfra.supplement import (
    METHODS, REFERENCE_ROUNDS, baseline_compatibility, class_groups,
    common_short_origin, load_prediction, metrics_from_run, read_csv,
    read_json, require_same_samples, write_csv, write_json,
)


def main():
    root = REPO / 'output/method_a_supplement_seed42_fast_v2_f128_c4_results/method_a_supplement_seed42_fast_v2_f128_c4'
    out = REPO / 'output/method_a_supplement_analysis_20260929'
    out.mkdir(parents=True, exist_ok=True)
    runs = {'full-cp': root/'references/full-cp', **{m: root/m for m in METHODS},
            'flat-cp': root/'references/flat-cp',
            's': REPO/'output/la_control_js_topology_analysis/la_control/seed42/client-longtail/s/tau1_a1_protocol42'}
    prior = read_json(runs['full-cp']/'class_prior.json')
    groups = class_groups(prior['counts'], prior['tail_ids'])
    audit = {'data_root': str(root), 'checks': {}, 'comparison_issues': {}, 'missing': []}
    curves, metrics, class_metrics = [], [], []
    matrices = {}
    predictions = {}
    reference = {r: load_prediction(root/'norm-matched/reference_eval/predictions'/f'r{r:03d}.npz')
                 for r in REFERENCE_ROUNDS}
    initial = reference[0]
    for method, path in runs.items():
        if not path.is_dir():
            audit['missing'].append(method)
            continue
        completion = read_json(path/'completion.json')
        assert completion['completed_round'] == 100, (method, 'completion')
        curve = metrics_from_run(path, groups)
        curves.extend(dict(method=method, **r) for r in curve)
        matrix = np.array([[float(x['per_class_acc']) for x in sorted(
            read_csv(path/f'per_class_accuracy_epoch_{r-1}.csv'), key=lambda x: int(x['class_id']))]
            for r in range(101)])
        matrices[method] = matrix
        for group, ids in groups.items():
            values = np.array([r['accuracy'] for r in sorted(curve, key=lambda x: x['round']) if r['group'] == group])
            metrics.append(dict(method=method, group=group, mean81_100=float(values[81:].mean()),
                final100=float(values[100]), round90=float(values[90]), change90_100=float(values[100]-values[90]),
                peak_drop_to100=float(values[1:].max()-values[100])))
        for c in range(100):
            class_metrics.append(dict(method=method, class_id=c, train_count=prior['counts'][c],
                group=next(g for g in ('Many35','Medium35','Few30') if c in groups[g]),
                is_tail=c in groups['Tail20'], mean81_100=float(matrix[81:,c].mean()), final100=float(matrix[100,c])))
        audit['comparison_issues'][method] = baseline_compatibility(runs['full-cp'], path)
        if method in METHODS:
            ps = {r: load_prediction(path/'predictions'/f'r{r:03d}.npz') for r in range(101)}
            max_error = 0.
            for r, p in ps.items():
                require_same_samples(initial, p)
                assert np.array_equal(p['correct'], p['class_id'] == p['prediction'])
                assert len(p['correct']) == 10000
                assert all(np.isfinite(p[k]).all() for k in ('ce_loss', 'la_loss', 'margin'))
                acc = np.bincount(p['class_id'], weights=p['correct'], minlength=100)
                max_error = max(max_error, float(np.abs(acc-matrix[r]).max()))
            assert max_error < 1e-8, (method, max_error)
            assert np.array_equal(ps[0]['prediction'], initial['prediction']), method
            predictions[method] = ps
            audit['checks'][method] = dict(completed_round=completion['completed_round'],
                prediction_rounds=len(ps), prediction_vs_per_class_max_error=max_error,
                identical_initial_predictions=True, local_optimizer_steps=completion['total_local_optimizer_steps'],
                correction_steps=completion['functional_correction_steps'])
    reference_error = max(float(np.abs(np.bincount(p['class_id'], weights=p['correct'], minlength=100)
                                      -matrices['full-cp'][r]).max()) for r,p in reference.items())
    assert reference_error < 1e-8
    audit['checks']['reference_prediction_max_error'] = reference_error
    predictions['full-cp'] = reference
    write_csv(out/'metrics_all_rounds.csv', curves)
    write_csv(out/'summary_7_methods.csv', metrics)
    write_csv(out/'per_class_results.csv', class_metrics)

    control_rows, harm_rows, class_harm_rows, gradient_rows = [], [], [], []
    for method in ('full-cp', *METHODS):
        path = runs[method]
        if method == 'norm-matched':
            records = [read_json(path/'sfra_rounds'/f'r{r:03d}/summary.json') for r in range(1,91)]
            errors = []
            for row in records:
                target = read_json(runs['full-cp']/'sfra_rounds'/f'r{row["round"]:03d}/summary.json')['committed_effective_norm']
                assert target == row['target_effective_norm']
                err = abs(row['committed_effective_norm']-target)/target
                assert err < 2e-4
                errors.append(err)
            audit['checks']['norm_matching'] = dict(rounds=90, maximum_relative_error=max(errors),
                mean_relative_error=float(np.mean(errors)), enlarged_rounds=sum(x['scalar'] > 1 for x in records),
                scalar_min=min(x['scalar'] for x in records), scalar_max=max(x['scalar'] for x in records),
                scalar_mean=float(np.mean([x['scalar'] for x in records])))
            continue
        labels = np.array([t['class_id'] for t in read_json(path/'private_witness_manifest.json')])
        for rnd in range(1,91):
            folder = path/'sfra_rounds'/f'r{rnd:03d}'
            summary = read_json(folder/'summary.json')
            z = load_prediction(folder/'tokens.npz')
            active, valid = z['active'].astype(bool), z['history_valid_before'].astype(bool)
            if method == 'shuffle-cp':
                donor, before, after = z['weight_donor_token'], z['weights_before_shuffle'], z['weights']
                assert np.array_equal(np.sort(donor), np.arange(len(donor)))
                assert np.array_equal(labels, labels[donor])
                assert np.array_equal(active, active[donor])
                assert np.array_equal(after, before[donor])
                assert np.array_equal(np.sort(before), np.sort(after))
                assert all(np.isclose(before[labels==c].astype(float).sum(), after[labels==c].astype(float).sum(), atol=1e-12, rtol=1e-12) for c in range(100))
                changed = int(np.count_nonzero(before != after))
                assert changed == summary['changed_weight_tokens']
            if method == 'current-cp':
                assert np.array_equal(active, z['supported'])
                assert np.array_equal(z['target'][active], z['current_target'][active])
            if method == 'plain-hold':
                target = z['F_post_B'].copy()
                target[valid] = np.maximum(target[valid], z['history_before'][valid,None])
                assert np.array_equal(target, z['target'])
                assert active.all() and np.allclose(z['weights'], 1/len(active))
                assert not any(k in z for k in ('responses','supported','u','n_eff','source_cache'))
            control_rows.append(dict(method=method, round=rnd,
                changed_weight_fraction=summary.get('changed_weight_fraction'),
                active_tokens=int(active.sum()), valid_history_tokens=int(valid.sum()),
                history_raises_target_tokens=int(((z['target'] > z['current_target']).any(1) & active).sum()),
                effective_norm_ratio=summary['committed_effective_norm']/summary['proposal_effective_norm'],
                proposal_committed_cosine=summary['proposal_committed_cosine'],
                functional_loss_change=summary['committed_functional_loss']-summary['proposal_functional_loss'],
                global_classification_loss_increase=summary['committed_classification_loss_increase']))
            cp0 = z['classification_proposal_scores'].astype(float).mean(1)
            cp1 = z['classification_committed_scores'].astype(float).mean(1)
            f0, fp, f1 = [z[k].astype(float).mean(1) for k in ('F_post_B','F_proposal','F_committed')]
            cpw = z['classification_global_token_weights'].astype(float)
            for c in range(100):
                mask = labels == c
                class_harm_rows.append(dict(method=method, round=rnd, class_id=c,
                    la_harm=float((cp1-cp0)[mask].mean()),
                    ordinary_margin_gain=float((fp-f0)[mask].mean()),
                    committed_margin_gain=float((f1-f0)[mask].mean()),
                    correction_margin_change=float((f1-fp)[mask].mean()),
                    harmed_token_fraction=float((cp1[mask]>cp0[mask]).mean()),
                    cp_weight=float(cpw[mask].sum())))
            for group, ids in groups.items():
                selected = class_harm_rows[-100:]
                selected = [x for x in selected if x['class_id'] in ids]
                harm_rows.append(dict(method=method, round=rnd, group=group,
                    **{key:float(np.mean([x[key] for x in selected])) for key in
                       ('la_harm','ordinary_margin_gain','committed_margin_gain','correction_margin_change','harmed_token_fraction')},
                    cp_weight=sum(x['cp_weight'] for x in selected)))
            for step in read_csv(folder/'correction_steps.csv'):
                if step['functional_classification_gradient_cosine']:
                    gradient_rows.append(dict(method=method, round=rnd, step=int(step['step']),
                        cosine=float(step['functional_classification_gradient_cosine'])))
    write_csv(out/'control_audit_rounds.csv', control_rows)
    write_csv(out/'witness_harm_by_group.csv', harm_rows)
    write_csv(out/'witness_harm_by_class.csv', class_harm_rows)
    write_csv(out/'gradient_conflict.csv', gradient_rows)
    audit['checks']['shuffle_90_rounds'] = 'donor permutation, within-class mapping, active set, weight multiset and per-class totals verified'
    audit['checks']['current_90_rounds'] = 'history does not activate units or raise targets'
    audit['checks']['plain_hold_90_rounds'] = 'all-unit uniform weights and max(post-B, valid history) targets verified'

    cohorts, paired = [], []
    for method, ps in predictions.items():
        for rnd in (60,80,90,100):
            p = ps[rnd]
            for group, ids in groups.items():
                mask = np.isin(initial['class_id'], ids)
                old, new = mask & initial['correct'], mask & ~initial['correct']
                cohorts.append(dict(method=method, round=rnd, group=group,
                    initial_correct=int(old.sum()), retained_initial_correct=int((old & p['correct']).sum()),
                    initial_wrong=int(new.sum()), learned_from_initial_wrong=int((new & p['correct']).sum()),
                    lost_initial_correct=int((old & ~p['correct']).sum())))
                if method != 'full-cp':
                    full = reference[rnd]['correct']
                    paired.append(dict(method=method, round=rnd, group=group,
                        full_only_correct=int((mask & full & ~p['correct']).sum()),
                        control_only_correct=int((mask & ~full & p['correct']).sum()),
                        full_minus_control_initial_correct=int((old & full).sum()-(old & p['correct']).sum()),
                        full_minus_control_initial_wrong=int((new & full).sum()-(new & p['correct']).sum())))
    write_csv(out/'common_initial_cohorts.csv', cohorts)
    write_csv(out/'paired_full_control_samples.csv', paired)
    short_roots = [REPO/'output/method_a_results_a60_a80'/f'method_a_diagnostics_a{a}_fast_v2_f128_c4' for a in (60,80)]
    shorts, missing = common_short_origin(reference, short_roots, groups)
    write_csv(out/'common_short_origin.csv', shorts)
    audit['common_short_origin_rows'] = len(shorts)
    audit['common_short_origin_missing'] = missing

    aggregates = []
    for method in ('full-cp','shuffle-cp','current-cp','plain-hold'):
        for start,end in ((1,90),(61,90),(81,90)):
            for group in groups:
                rows = [x for x in harm_rows if x['method']==method and x['group']==group and start<=x['round']<=end]
                aggregates.append(dict(method=method, start=start, end=end, group=group,
                    **{k:float(np.mean([x[k] for x in rows])) for k in
                       ('la_harm','ordinary_margin_gain','committed_margin_gain','correction_margin_change','harmed_token_fraction','cp_weight')}))
    write_csv(out/'witness_harm_windows.csv', aggregates)
    from matplotlib import pyplot as plt
    colors = {'full-cp':'#b22222', 'shuffle-cp':'#bb8b00', 'current-cp':'#2a7d42',
              'plain-hold':'#8555ad', 'norm-matched':'#267db1', 'flat-cp':'#777777', 's':'#333333'}
    labels = {'full-cp':'Full-CP (standard)', 's':'Ordinary S (standard)',
              'flat-cp':'Flat-CP (fast v1)', **{m:m+' (fast v2)' for m in METHODS}}
    fig, axes = plt.subplots(1, 3, figsize=(16, 5))
    for ax, group in zip(axes[:2], ('Tail20','Medium35')):
        for method in runs:
            if method == 'flat-cp':
                continue  # Near-overlap with Plain-Hold; included in the tradeoff panel and tables.
            values = [x['accuracy'] for x in curves if x['method']==method and x['group']==group]
            ax.plot(range(40,101), values[40:], color=colors[method],
                    ls='--' if method in ('full-cp','s') else '-', lw=1.7, label=labels[method])
        ax.axvline(90, color='#aaaaaa', lw=.8, ls=':')
        ax.set(xlabel='Training round', ylabel='Accuracy (%)', title=group+' trajectory')
        ax.grid(alpha=.2)
    offsets = {'full-cp':(8,6), 'shuffle-cp':(8,4), 'current-cp':(-75,8),
               'plain-hold':(-66,17), 'norm-matched':(8,-13), 'flat-cp':(-70,-17), 's':(-12,-17)}
    for method in runs:
        lookup = {x['group']:x['mean81_100'] for x in metrics if x['method']==method}
        x,y=lookup['Medium35'],lookup['Tail20']
        axes[2].scatter(x,y,c=colors[method], s=65, marker='*' if method=='full-cp' else 'o', zorder=3)
        axes[2].annotate(method, (x,y), xytext=offsets[method], textcoords='offset points', fontsize=9)
    axes[2].set(xlabel='Medium35 mean, rounds 81-100 (%)', ylabel='Tail20 mean, rounds 81-100 (%)',
                title='Learning / retention tradeoff')
    axes[2].margins(.2)
    axes[2].grid(alpha=.2)
    fig.suptitle('Method A, seed42: descriptive comparisons across execution versions', fontsize=13)
    handles, names=axes[0].get_legend_handles_labels()
    fig.legend(handles,names,loc='lower center',ncol=3,frameon=False,fontsize=9)
    fig.tight_layout(rect=(0,.13,1,.94))
    for ext in ('png','pdf'):
        fig.savefig(out/f'method_a_results.{ext}', dpi=180, bbox_inches='tight')
    plt.close(fig)
    write_json(out/'audit.json', audit)
    print(json.dumps(audit, ensure_ascii=False, indent=2))
    print('SUMMARY')
    for method in runs:
        print(method, {x['group']:round(x['mean81_100'],6) for x in metrics if x['method']==method},
              'tail_drop', next(x['peak_drop_to100'] for x in metrics if x['method']==method and x['group']=='Tail20'))
    print('LATE CONTROLS')
    for method in ('full-cp','shuffle-cp','current-cp','plain-hold'):
        rows = [x for x in control_rows if x['method']==method and x['round']>=61]
        print(method, {k:float(np.mean([x[k] for x in rows])) for k in rows[0]
                       if k not in ('method','round','changed_weight_fraction')},
              'gradient_cosine',float(np.mean([x['cosine'] for x in gradient_rows if x['method']==method and x['round']>=61])))
    print('LATE CLASSIFICATION HARM')
    for x in aggregates:
        if x['start']==61 and x['method']=='full-cp': print(x)
    print('TAIL INITIAL COHORTS FINAL')
    for x in cohorts:
        if x['round']==100 and x['group']=='Tail20': print(x)
    print('TAIL COMMON SHORT COHORTS')
    for x in shorts:
        if x['group']=='Tail20': print({k:v for k,v in x.items() if k!='source_file'})


if __name__ == '__main__':
    main()
