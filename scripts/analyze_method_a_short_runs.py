"""Audit and analyze the exported two-anchor Method A diagnostic without training."""
import argparse
from collections import Counter
from pathlib import Path
import sys

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from tools.sfra.diagnostics import read_json, read_csv, write_json, write_csv, file_hash, class_groups

BRANCHES = ('full-cp', 'ordinary', 'norm-matched')
ANCHORS = (60, 80)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--input', type=Path, default=REPO/'output/method_a_results_a60_a80')
    parser.add_argument('--output', type=Path, default=REPO/'output/method_a_analysis_20260927')
    args = parser.parse_args()
    inputs = set()

    def js(p):
        inputs.add(p)
        return read_json(p)

    def csv(p):
        inputs.add(p)
        return read_csv(p)

    def npz(p):
        inputs.add(p)
        with np.load(p, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}

    runs = {}
    for p in args.input.rglob('diagnostic_completion.json'):
        d = js(p)
        key = (d['anchor'], d['branch'])
        assert key not in runs, f'Duplicate completed branch: {key}'
        assert d['completed_round'] == d['anchor'] + 5, str(p)
        runs[key] = p.parent
    assert set(runs) == {(a, b) for a in ANCHORS for b in BRANCHES}
    plans = [js(runs[a, 'full-cp'].parents[1]/'diagnostic_plan.json') for a in ANCHORS]
    assert plans[0] == plans[1], 'Different training plans/source/code/execution configuration'
    plan = plans[0]
    summary, retention, geometry, group_data, class_data, paired = [], [], [], [], [], []
    same_state, cp_effect, class_harm = [], [], []
    predictions, arrays = {}, {}
    max_test_error = max_witness_error = max_norm_error = max_execution_error = 0.
    common_identity = None
    for anchor in ANCHORS:
        full = runs[anchor, 'full-cp']
        prior = js(full/'class_prior.json')
        counts = np.array(prior['counts'])
        groups = class_groups(counts, prior['tail_ids'])
        metadata = js(full/'bridge_metadata.json')
        classnames = metadata['classnames']
        partition = csv(full/'partition_manifest.csv')
        assert np.array_equal(np.bincount([int(r['class_id']) for r in partition], minlength=100), counts)
        manifest = js(full/'private_witness_manifest.json')
        labels = np.array([r['class_id'] for r in manifest])
        clients = np.array([r['client_id'] for r in manifest])
        reference_norms = csv(full/'norms.csv')
        reference_anchor = js(full/'anchor_audit.json')
        assert {g:len(ids) for g, ids in groups.items()} == dict(Overall=100, Many35=35, Medium35=35, Few30=30, Tail20=20)
        for branch in BRANCHES:
            run = runs[anchor, branch]
            job = js(run/'diagnostic_job.json')
            assert all(job[k] == v for k, v in plan.items())
            assert job['anchor'] == anchor and job['branch'] == branch
            assert js(run/'anchor_audit.json') == reference_anchor
            assert js(run/'class_prior.json') == prior
            assert js(run/'private_witness_manifest.json') == manifest
            assert file_hash(run/'partition_manifest.csv') == file_hash(full/'partition_manifest.csv')
            cfg = js(run/'sfra_config.json')
            assert cfg['variant'] == 'full-cp' and cfg['classification_weight'] == 1 and cfg['retention_weight'] == 10
            assert not cfg.get('b_transfer') and not cfg.get('b_aggregation')
            for name, source_name in [('protocol/full_schedule.json', 'protocol/full_schedule.json'),
                                      ('private_witness_manifest.json', 'private_witness_manifest.json'),
                                      ('partition_manifest.csv', 'partition_manifest.csv')]:
                inputs.add(run/name)
                assert file_hash(run/name) == plan['source_sha256'][source_name]
            audit = js(run/'diagnostic_execution_audit.json')
            assert audit['continuation_execution'] == plan['execution_override']
            assert audit['switch_after_reference_anchor_and_history_audits']
            assert audit['original_historical_scores_preserved']
            max_execution_error = max(max_execution_error, audit['anchor_margin_max_error'])
            assert audit['anchor_margin_max_error'] <= 5e-6
            norms = csv(run/'norms.csv')
            assert [int(r['round']) for r in norms] == list(range(anchor+1, anchor+6))
            for k in ('post_b_lora_sha256', 'ordinary_lora_sha256'):
                assert norms[0][k] == reference_norms[0][k]
            for r, ref in zip(norms, reference_norms):
                actual, ordinary, target = (float(r['actual_effective_norm']),
                                           float(r['ordinary_effective_norm']), float(ref['actual_effective_norm']))
                error = abs(actual-target)/target
                if branch == 'norm-matched':
                    max_norm_error = max(max_norm_error, error)
                    assert error <= 2e-6
                geometry.append(dict(anchor=anchor, branch=branch, round=int(r['round']),
                                     actual_norm=actual, ordinary_norm=ordinary, actual_to_ordinary=actual/ordinary,
                                     full_cp_target=target, relative_error_to_full_cp=error))
            metrics = csv(run/'candidate_metrics.csv')
            witness_metrics = csv(run/'witness_metrics.csv')
            for rnd, candidate, kind in sorted({(int(r['round']), r['candidate'], r['kind']) for r in metrics}):
                key = (anchor, branch, rnd, candidate, kind)
                p = run/f'predictions/r{rnd:03d}_{candidate}_{kind}.csv'
                rows = csv(p)
                data = {k:np.array([float(r[k]) for r in rows]) for k in rows[0]}
                assert all(np.isfinite(v).all() for v in data.values())
                ids = data['class_id'].astype(int)
                assert np.array_equal(data['sample_id'], np.arange(len(rows)))
                assert np.array_equal(data['correct'], data['class_id'] == data['prediction'])
                identity = list(zip(data['sample_id'], data['class_id']))
                if common_identity is None:
                    common_identity = identity
                assert identity == common_identity
                n = np.bincount(ids, minlength=100)
                assert np.array_equal(n, np.full(100, 100))
                cmetrics = {('accuracy' if k == 'correct' else k):np.bincount(ids, weights=data[k], minlength=100)/n
                            * (100 if k == 'correct' else 1) for k in ('correct','ce_loss','la_loss','margin')}
                z = npz(run/f'witness_tokens/r{rnd:03d}_{candidate}_{kind}.npz')
                assert np.array_equal(z['class_ids'], labels) and np.array_equal(z['client_ids'], clients)
                wmetrics = {k:np.array([z[field].astype(float)[labels == c].mean() for c in range(100)])
                            for k, field in [('la_loss','classification_scores'), ('margin','scores')]}
                predictions[key], arrays[key] = data, (cmetrics, wmetrics)
                for domain, cm in [('test',cmetrics), ('witness',wmetrics)]:
                    for c in range(100):
                        class_data.append(dict(anchor=anchor, branch=branch, round=rnd, candidate=candidate,
                            kind=kind, domain=domain, class_id=c, class_name=classnames[c], train_count=int(counts[c]),
                            **{k:float(v[c]) for k,v in cm.items()}))
                    for group, cs in groups.items():
                        got = {k:float(v[cs].mean()) for k,v in cm.items()}
                        recorded = [r for r in (metrics if domain == 'test' else witness_metrics)
                                    if (int(r['round']),r['candidate'],r['kind'],r['group']) == (rnd,candidate,kind,group)]
                        assert len(recorded) == 1
                        err = max(abs(float(recorded[0][k])-v) for k,v in got.items())
                        if domain == 'test':
                            max_test_error = max(max_test_error, err)
                            assert err < 1e-10
                        else:
                            max_witness_error = max(max_witness_error, err)
                            assert err < 2e-6
                        group_data.append(dict(anchor=anchor,branch=branch,round=rnd,candidate=candidate,
                                               kind=kind,domain=domain,group=group,**got))
            baseline = predictions[anchor,'full-cp',anchor,'full-cp','anchor']
            own_start = predictions[anchor,branch,anchor,branch,'anchor']
            assert all(np.array_equal(baseline[k], own_start[k]) for k in baseline)
            final = predictions[anchor,branch,anchor+5,branch,'committed']
            for group, cs in groups.items():
                mask = np.isin(baseline['class_id'],cs)
                before, after = baseline['correct'][mask].astype(bool), final['correct'][mask].astype(bool)
                anchor_acc = float(before.mean()*100)
                end_acc = float(after.mean()*100)
                mean_acc = float(np.mean([arrays[anchor,branch,r,branch,'committed'][0]['accuracy'][cs].mean()
                                         for r in range(anchor+1,anchor+6)]))
                summary.append(dict(anchor=anchor,branch=branch,group=group,anchor_accuracy=anchor_acc,
                    mean_five_accuracy=mean_acc,final_accuracy=end_acc,final_minus_anchor=end_acc-anchor_acc))
                retention.append(dict(anchor=anchor,branch=branch,group=group,samples=int(mask.sum()),
                    anchor_correct=int(before.sum()),retained_correct=int((before&after).sum()),
                    correct_to_wrong=int((before&~after).sum()),wrong_to_correct=int((~before&after).sum())))
        for branch in BRANCHES:
            actual = predictions[anchor,branch,anchor+1,branch,'committed']
            counterfactual = predictions[anchor,'full-cp',anchor+1,branch,'same-state']
            assert all(np.array_equal(actual[k],counterfactual[k]) for k in actual)
        for other in ('ordinary','norm-matched'):
            for group, cs in groups.items():
                mask = np.isin(baseline['class_id'],cs)
                a = predictions[anchor,'full-cp',anchor+5,'full-cp','committed']['correct'][mask].astype(bool)
                b = predictions[anchor,other,anchor+5,other,'committed']['correct'][mask].astype(bool)
                paired.append(dict(anchor=anchor,group=group,comparison='full-cp minus '+other,
                    full_only_correct=int((a&~b).sum()),other_only_correct=int((~a&b).sum()),
                    both_correct=int((a&b).sum()),both_wrong=int((~a&~b).sum()),net_correct=int(a.sum()-b.sum())))
        for domain_index, domain in enumerate(('test','witness')):
            candidates = {c:arrays[anchor,'full-cp',anchor+1,c,'same-state'][domain_index]
                          for c in ('pre-A','ordinary','full-cp','no-cp','norm-matched')}
            for group, cs in groups.items():
                for left, right in [('ordinary','pre-A'),('full-cp','pre-A'),('full-cp','ordinary'),
                                    ('full-cp','no-cp'),('full-cp','norm-matched')]:
                    same_state.append(dict(anchor=anchor,domain=domain,group=group,comparison=left+' minus '+right,
                        **{k:float((candidates[left][k]-candidates[right][k])[cs].mean()) for k in candidates[left]}))
                losses = {k:float(v['la_loss'][cs].mean()) for k,v in candidates.items()}
                cp_effect.append(dict(anchor=anchor,domain=domain,group=group,
                    full_minus_ordinary=losses['full-cp']-losses['ordinary'],
                    no_cp_minus_ordinary=losses['no-cp']-losses['ordinary'],
                    full_minus_no_cp=losses['full-cp']-losses['no-cp'],
                    harm_removed_fraction=((losses['no-cp']-losses['full-cp'])/(losses['no-cp']-losses['ordinary'])
                                           if losses['no-cp']>losses['ordinary'] else None)))
            for c in range(100):
                class_harm.append(dict(anchor=anchor,domain=domain,class_id=c,class_name=classnames[c],
                    train_count=int(counts[c]),frequency_group='Many35' if counts[c]>100 else 'Medium35' if counts[c]>=20 else 'Few30',
                    **{left+'_minus_'+right+'_'+k:float(candidates[left][k][c]-candidates[right][k][c])
                       for left,right in [('ordinary','pre-A'),('full-cp','pre-A'),('full-cp','ordinary'),
                                          ('full-cp','no-cp'),('full-cp','norm-matched')]
                       for k in candidates[left]}))
        full_tokens = npz(full/f'sfra_rounds/r{anchor+1:03d}/tokens.npz')
        no_cp_tokens = npz(full/f'no_cp_probe/r{anchor+1:03d}/tokens.npz')
        for k in ('target','weights','sigma','responses','history_before','F_proposal','classification_proposal_scores'):
            assert np.array_equal(full_tokens[k],no_cp_tokens[k]), f'No-CP probe input differs: {k}'
        counts_by_unit = Counter((int(r['client_id']),int(r['class_id'])) for r in partition)
        expected_weights = np.array([counts_by_unit[i,c]/len(partition) for i,c in zip(clients,labels)])
        for rnd in range(anchor+1, anchor+6):
            z = npz(full/f'sfra_rounds/r{rnd:03d}/tokens.npz')
            assert np.allclose(z['classification_global_token_weights'],expected_weights,atol=1e-8,rtol=1e-5)
        for filename, calculated, keys in [('short_run_summary.csv',summary,('anchor','branch','group')),
                                          ('sample_retention.csv',retention,('anchor','branch','group'))]:
            for old in csv(full.parents[1]/filename):
                new = next(r for r in calculated if all(str(r[k]) == old[k] for k in keys))
                for k in set(new)&set(old)-set(keys):
                    assert abs(float(new[k])-float(old[k])) < 1e-10, (filename,k)
    comparisons = []
    for a in ANCHORS:
        for group in ('Overall','Many35','Medium35','Few30','Tail20'):
            rows = {r['branch']:r for r in summary if r['anchor']==a and r['group']==group}
            for other in ('ordinary','norm-matched'):
                comparisons.append(dict(anchor=a,group=group,comparison='full-cp minus '+other,
                    mean_five_difference=rows['full-cp']['mean_five_accuracy']-rows[other]['mean_five_accuracy'],
                    final_difference=rows['full-cp']['final_accuracy']-rows[other]['final_accuracy']))
    args.output.mkdir(parents=True,exist_ok=True)
    for name, rows in [('combined_summary',summary),('sample_retention',retention),('update_geometry',geometry),
                       ('group_metrics',group_data),('class_metrics',class_data),('paired_final_predictions',paired),
                       ('same_state_differences',same_state),('cp_effect',cp_effect),('class_harm',class_harm),
                       ('branch_comparisons',comparisons)]:
        write_csv(args.output/(name+'.csv'),rows)
    write_json(args.output/'audit.json',dict(complete=True,completed_branches=6,
        plans_identical=True,initial_predictions_and_first_proposals_identical=True,
        no_cp_probe_inputs_identical=True,test_samples=len(common_identity),
        max_test_metric_recompute_error=max_test_error,max_witness_metric_recompute_error=max_witness_error,
        max_relative_effective_norm_error=max_norm_error,max_v2_anchor_margin_error=max_execution_error,
        limits=['Model weights were excluded; state/norm checks use recorded audits, not new model forwards.',
                'Two anchors from one seed and one Full-CP trajectory; not independent seeds.',
                'No-CP has only two same-state one-update probes; not a five-round no-CP branch.'],
        source_files={str(p.relative_to(args.input)):file_hash(p) for p in sorted(inputs)}))
    make_figure(args.output,summary,retention)
    print('Audited 6 branches; outputs:',args.output)
    print('Max relative effective norm error:',max_norm_error)
    print('Max test/witness metric recompute error:',max_test_error,max_witness_error)


def make_figure(output,summary,retention):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    colors = {'full-cp':'#2166AC','ordinary':'#B35806','norm-matched':'#5AAE61'}
    labels = {'full-cp':'Full-CP','ordinary':'Ordinary','norm-matched':'Norm-matched'}
    fig, axes = plt.subplots(2,2,figsize=(10,7),layout='constrained')
    for row,anchor in enumerate(ANCHORS):
        ax = axes[row,0]
        for i,b in enumerate(BRANCHES):
            vals = [next(r['final_minus_anchor'] for r in summary
                         if r['anchor']==anchor and r['branch']==b and r['group']==g) for g in ('Overall','Medium35','Tail20')]
            bars=ax.bar(np.arange(3)+(i-1)*.24,vals,width=.23,color=colors[b],label=labels[b])
            ax.bar_label(bars,fmt='%+.2f',fontsize=8,padding=3)
        ax.axhline(0,color='black',lw=.7)
        ax.set(xticks=np.arange(3),xticklabels=['Overall','Medium35','Tail20'],ylim=(-.75,.7),
               ylabel='Accuracy change (percentage points)',title=f'Anchor {anchor}: change after 5 rounds')
        if row==0: ax.legend(fontsize=8,loc='upper right')
        ax=axes[row,1]
        for i,b in enumerate(BRANCHES):
            vals=[]
            for group,key in [('Medium35','wrong_to_correct'),('Medium35','correct_to_wrong'),('Tail20','correct_to_wrong')]:
                vals.append(next(r[key] for r in retention if r['anchor']==anchor and r['branch']==b and r['group']==group))
            bars=ax.bar(np.arange(3)+(i-1)*.24,vals,width=.23,color=colors[b])
            ax.bar_label(bars,fontsize=8,padding=3)
        ax.set(xticks=np.arange(3),xticklabels=['Medium: newly correct','Medium: newly wrong','Tail: newly wrong'],
               ylabel='Test samples',ylim=(0,30),title=f'Anchor {anchor}: changes vs common starting model')
        ax.tick_params(axis='x',labelsize=8)
    fig.suptitle('Method A diagnostic: one seed, two 5-round continuations',fontsize=13)
    for suffix in ('png','pdf'):
        fig.savefig(output/f'method_a_short_runs.{suffix}',dpi=180)
    plt.close(fig)


if __name__ == '__main__':
    main()
