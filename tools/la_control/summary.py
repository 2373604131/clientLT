"""Summarize E0--E5/J/S without model loading or test-set selection."""
import csv
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

from utils.cliplora_bridge_audit import write_csv, write_json

METRICS = ('overall_acc','non_tail_acc','bottom20_tail_acc','head20_acc','middle60_acc')
NORMAL_PHASES = ('normal_B','normal_AB')


def read(path):
    with Path(path).open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))


def js(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def mean(rows,key):
    return float(np.mean([float(r[key]) for r in rows]))


def summarize(root):
    root = Path(root)
    out = root/'analysis'
    out.mkdir(parents=True,exist_ok=True)
    runs, performance, costs, audit, decision_rows, mechanisms = [], [], [], {}, [], []
    for config_path in sorted(root.glob('seed*/*/*/*/control_config.json')):
        run = config_path.parent
        label = str(run.relative_to(root))
        if not (run/'completion.json').exists():
            audit[label] = {'complete':False}
            continue
        cfg = js(config_path)
        progress = js(run/'completion.json')
        rows = read(run/'round_metrics.csv')
        events = read(run/'event_manifest.csv')
        budget = read(run/'budget.csv')
        meta = js(run/'bridge_metadata.json')
        normal_phase = 'normal_AB' if cfg['method']=='j' else 'normal_B'
        normal_expected = cfg['normal_steps_expected']
        extra_expected = cfg['extra_steps_expected']
        steps_per_epoch = cfg.get('steps_per_epoch', sum((int(n)+31)//32 for n in meta['client_sample_counts']))
        extra_rounds = cfg['candidate_rounds']
        # Older E2/E3 packages have per-class files but not these two group columns.
        if 'head20_acc' not in rows[0]:
            counts = js(run/'class_prior.json')['counts']
            head = sorted(range(100),key=lambda c:(-counts[c],c))[:20]
            middle = sorted(set(range(100))-set(head)-set(cfg['tail_ids']))
            for row in rows:
                per_class = {int(r['class_id']):float(r['per_class_acc']) for r in
                             read(run/f'per_class_accuracy_epoch_{int(row["round"])-1}.csv')}
                row['head20_acc'] = float(np.mean([per_class[c] for c in head]))
                row['middle60_acc'] = float(np.mean([per_class[c] for c in middle]))
        checks = {'complete':True,'101_unique_rounds':Counter(int(r['round']) for r in rows)==Counter(range(101)),
                  'normal_steps':progress['normal_optimizer_steps']==normal_expected,
                  'extra_steps':progress['extra_optimizer_steps']==extra_expected,
                  'overhead_steps':progress['unselected_branch_optimizer_steps']==progress['attempted_decisions']*steps_per_epoch*(1+3*cfg['lookahead_rounds']),
                  'official_test_count':progress['official_test_passes']==101,
                  'event_ids_unique':len(events)==len({r['event_id'] for r in events})}
        by_round = {int(r['round']):r for r in rows}
        committed = [r for r in events if r['committed']=='True']
        train_committed = [r for r in committed if r['phase']!='restore']
        checks['normal_rounds_once'] = Counter(int(r['round']) for r in train_committed if r['phase']==normal_phase)==Counter(range(1,101))
        checks['extra_rounds_once'] = Counter(int(r['round']) for r in train_committed if r['phase'] not in NORMAL_PHASES)==Counter(extra_rounds)
        if cfg['method'] not in ('e4','e5'):
            extra_phase = 'extra_B' if cfg['method'] in ('e0','e2') else 'refresh_A'
            checks['extra_factor'] = all(r['phase']==extra_phase for r in train_committed if r['phase'] not in NORMAL_PHASES)
        last_hash = meta['initial_lora_sha256']
        chain = True
        order = {'normal_B':0,'normal_AB':0,'extra_B':1,'refresh_A':1,'restore':2}
        for r in sorted(committed,key=lambda r:(int(r['round']),order[r['phase']])):
            chain &= r['before_lora_sha256']==last_hash
            last_hash = r['after_lora_sha256']
        checks['committed_state_chain_with_restores'] = chain
        checks['all_training_events_reconstruct'] = all(r['reconstruction_passed']=='True' for r in events if r['phase']!='restore')
        checks['sample_weights_and_schedule'] = all(
            json.loads(r['selected_client_ids'])==meta['schedule'][int(r['round'])-1] and
            np.allclose(json.loads(r['server_weights']),
                np.array(meta['client_sample_counts'])[json.loads(r['selected_client_ids'])]/sum(meta['client_sample_counts']),rtol=0,atol=1e-12)
            for r in events if r['phase']!='restore')
        checks['budget_events_match'] = set(r['event_id'] for r in budget)==set(r['event_id'] for r in events if r['phase']!='restore')
        checks['actual_normal_steps'] = sum(int(r['optimizer_steps']) for r in budget if r['committed']=='True' and r['phase'] in NORMAL_PHASES)==normal_expected
        checks['actual_extra_steps'] = sum(int(r['optimizer_steps']) for r in budget if r['committed']=='True' and r['phase'] not in NORMAL_PHASES)==extra_expected
        decisions = read(run/'control_decisions.csv') if (run/'control_decisions.csv').exists() else []
        failure = 0
        stopping_ok = True
        for d in decisions:
            accepted = d['accepted_A']=='True'
            expected = (float(d['delayed_G_all'])>cfg['min_gain'] and
                        float(d['delayed_G_tail'])>=-cfg['tail_tolerance'] and
                        float(d['delayed_G_hist'])>=-cfg['history_tolerance'])
            stopping_ok &= accepted==expected
            failure = 0 if accepted else failure+1
            stopping_ok &= int(d['failure_count'])==failure
            stopping_ok &= (d['freeze_A']=='True')==(failure>=cfg['patience'])
        checks['failure_count_independent_of_anchor'] = stopping_ok
        checks['branch_training_rng_end_equal'] = all(d['A_rng_end']==d['B_rng_end'] for d in decisions)
        audit[label] = checks
        for metric in METRICS:
            last20 = mean([by_round[r] for r in range(81,101)],metric)
            final = float(by_round[100][metric])
            peak_round = max(range(1,101),key=lambda r:float(by_round[r][metric]))
            performance.append({'run':label,'seed':cfg['seed'],'topology':cfg['topology'],'method':cfg['method'],
                'metric':metric,'initial':float(by_round[0][metric]),'last20':last20,'final':final,
                'peak':float(by_round[peak_round][metric]),'peak_round':peak_round,
                'best_to_final':float(by_round[peak_round][metric])-final})
        evals = read(run/'evaluation_budget.csv')
        costs.append({'run':label,**progress,
            'committed_a_optimizer_steps':sum(int(r['optimizer_steps']) for r in budget if r['committed']=='True' and r['trainable_factor'] in ('A','AB')),
            'committed_b_optimizer_steps':sum(int(r['optimizer_steps']) for r in budget if r['committed']=='True' and r['trainable_factor'] in ('B','AB')),
            'a_refresh_events':sum(r['phase']=='refresh_A' for r in train_committed),
            'normal_ab_events':sum(r['phase']=='normal_AB' for r in train_committed),
            'committed_training_seconds':sum(float(r['seconds']) for r in train_committed),
            'feedback_seconds':sum(float(r['seconds']) for r in evals if r['kind']=='training_side_feedback'),
            'official_test_seconds':sum(float(r['seconds']) for r in evals if r['kind']=='official_test'),
            'feedback_sample_presentations':sum(int(r['sample_presentations']) for r in evals if r['kind']=='training_side_feedback'),
            'training_upload_bytes':sum(int(r['upload_bytes']) for r in budget),
            'feedback_upload_bytes':sum(int(r['upload_bytes']) for r in evals),
            'modeled_downlink_bytes':sum(int(r['modeled_downlink_bytes']) for r in budget+evals)})
        decision_rows.extend({'run':label,**d} for d in decisions)
        if (run/'analysis/phase_class_budgets.csv').exists():
            groups = defaultdict(list)
            for r in read(run/'analysis/phase_class_budgets.csv'):
                groups[r['phase'],r['committed'],r['branch']].append(r)
            for (phase,committed_flag,branch),rr in groups.items():
                mechanisms.append({'run':label,'phase':phase,'committed':committed_flag,'branch':branch,
                    'class_events':len(rr),'event_count':len({r['event_id'] for r in rr}),
                    **{k:mean(rr,k) for k in ('W','H','D','R')},
                    'support_net':mean(rr,'W')-mean(rr,'H'),
                    'non_support_net':mean(rr,'D')-mean(rr,'R')})
        runs.append({'run':label,'config':cfg,'meta':meta,'curve':by_round,'progress':progress})
    comparisons = []
    pairing_checks = []
    for proposed in runs:
        c = proposed['config']
        targets = {'e0':(), 'e1':('e0',), 'e2':('e0',), 'e3':('e2','e1'),
                   'e4':('e0','e1'), 'e5':('e2','e3','e4'),
                   'j':('e2','e3'), 's':('e2','e3','j')}[c['method']]
        for baseline in runs:
            b = baseline['config']
            if b['method'] not in targets or (b['seed'],b['topology'])!=(c['seed'],c['topology']):
                continue
            if b.get('dirichlet_beta',.5)!=c.get('dirichlet_beta',.5):
                continue
            loss_contrast = (c['method'],b['method']) in (('e2','e0'),('e3','e1'),('e5','e4'))
            if not loss_contrast and b['la_tau']!=c['la_tau']:
                continue
            if b['method'] in ('e1','e3','e4','e5','j','s') and b['a_lr_mult']!=c['a_lr_mult']:
                continue
            if b['method']=='e4' and any(b[k]!=c[k] for k in ('lookahead_rounds','tail_tolerance','history_tolerance','min_gain','patience')):
                continue
            common = ('pool_sha256','test_sha256','probe_images_sha256','probe_manifest_sha256',
                      'schedule_sha256','initial_lora_sha256','frozen_model_sha256')
            compatible = all(proposed['meta'][k]==baseline['meta'][k] for k in common)
            compatible &= proposed['meta']['client_sample_counts']==baseline['meta']['client_sample_counts']
            compatible &= c['rng_protocol']==b['rng_protocol']
            same_code = proposed['meta']['training_code_hashes']==baseline['meta']['training_code_hashes']
            equal_steps = proposed['progress']['total_optimizer_steps']==baseline['progress']['total_optimizer_steps']
            pairing_checks.append({'method_run':proposed['run'],'baseline_run':baseline['run'],
                'data_and_rng_compatible':compatible,'training_code_hashes_equal':same_code,
                'compatible':compatible and same_code,'equal_total_optimizer_steps':equal_steps})
            if not compatible:
                continue
            for metric in METRICS:
                diff = np.mean([float(proposed['curve'][r][metric])-float(baseline['curve'][r][metric]) for r in range(81,101)])
                comparisons.append({'method_run':proposed['run'],'baseline_run':baseline['run'],'metric':metric,
                    'last20_method_minus_baseline':float(diff),
                    'training_code_hashes_equal':same_code,'equal_total_optimizer_steps':equal_steps,
                    'comparison':('paired data/configuration' if same_code else 'cross-code descriptive comparison; review training changes')
                        + ('; equal optimizer steps, not necessarily equal FLOPs' if equal_steps else '; unequal training budget')
                        + '; one seed is not significance'})
    interactions = []
    for refresh_la in runs:
        c = refresh_la['config']
        if c['method']!='e3':
            continue
        fixed_la = [r for r in runs if r['config']['method']=='e2' and
                    (r['config']['seed'],r['config']['topology'],r['config']['la_tau'])==(c['seed'],c['topology'],c['la_tau'])]
        refresh_ce = [r for r in runs if r['config']['method']=='e1' and
                      (r['config']['seed'],r['config']['topology'],r['config']['a_lr_mult'])==(c['seed'],c['topology'],c['a_lr_mult'])]
        fixed_ce = [r for r in runs if r['config']['method']=='e0' and
                    (r['config']['seed'],r['config']['topology'])==(c['seed'],c['topology'])]
        for e2 in fixed_la:
            for e1 in refresh_ce:
                for e0 in fixed_ce:
                    quartet = (e0,e1,e2,refresh_la)
                    common = ('pool_sha256','test_sha256','probe_images_sha256','probe_manifest_sha256',
                              'schedule_sha256','initial_lora_sha256','frozen_model_sha256','client_sample_counts')
                    if not all(all(r['meta'][k]==e0['meta'][k] for k in common) and
                               r['config']['rng_protocol']==e0['config']['rng_protocol'] for r in quartet):
                        continue
                    same_code = all(r['meta']['training_code_hashes']==e0['meta']['training_code_hashes'] for r in quartet)
                    for metric in METRICS:
                        value = lambda r: mean([r['curve'][t] for t in range(81,101)],metric)
                        ce_gain, la_gain = value(e1)-value(e0),value(refresh_la)-value(e2)
                        interactions.append({'seed':c['seed'],'topology':c['topology'],'metric':metric,
                            'e0':e0['run'],'e1':e1['run'],'e2':e2['run'],'e3':refresh_la['run'],
                            'refresh_gain_without_la':ce_gain,'refresh_gain_with_la':la_gain,
                            'la_refresh_interaction':la_gain-ce_gain,'training_code_hashes_equal':same_code,
                            'comparison':'paired quartet' if same_code else 'cross-code descriptive interaction; review training changes'})
    topology_rows = []
    for clt in runs:
        c = clt['config']
        if c['topology']!='client-longtail':
            continue
        for directory in runs:
            d = directory['config']
            fields = ('seed','method','la_tau','a_lr_mult','lookahead_rounds','min_gain','tail_tolerance','history_tolerance','patience')
            if d['topology'] not in ('matched-dirichlet','noniid-labeldir-fine') or any(c[k]!=d[k] for k in fields):
                continue
            common = ('pool_sha256','test_sha256','probe_images_sha256','probe_manifest_sha256',
                      'schedule_sha256','initial_lora_sha256','frozen_model_sha256')
            if any(clt['meta'][k]!=directory['meta'][k] for k in common):
                continue
            equal_sizes = clt['meta']['client_sample_counts']==directory['meta']['client_sample_counts']
            if d['topology']=='matched-dirichlet' and not equal_sizes:
                continue
            same_code = clt['meta']['training_code_hashes']==directory['meta']['training_code_hashes']
            equal_steps = clt['progress']['total_optimizer_steps']==directory['progress']['total_optimizer_steps']
            for metric in METRICS:
                topology_rows.append({'CLT_run':clt['run'],'Dir_run':directory['run'],'method':c['method'],
                    'dirichlet_partition':d['topology'],'dirichlet_beta':d.get('dirichlet_beta',.5),
                    'client_sample_counts_equal':equal_sizes,'training_code_hashes_equal':same_code,
                    'total_optimizer_steps_equal':equal_steps,
                    'comparison':('fixed-marginal topology control' if d['topology']=='matched-dirichlet'
                                  else 'standard fine-class Dirichlet; client sizes and FedAvg weights may differ')
                                 + ('; same code' if same_code else '; cross-code descriptive comparison'),
                    'metric':metric,'last20_Dir_minus_CLT':float(np.mean([
                        float(directory['curve'][r][metric])-float(clt['curve'][r][metric]) for r in range(81,101)]))})
    valid = bool(runs) and all(all(v.values()) for v in audit.values())
    write_json(out/'protocol_audit.json',{'valid':valid,'runs':audit,'pairing_checks':pairing_checks})
    for filename,data in [('performance',performance),('costs',costs),('comparisons',comparisons),
                          ('la_refresh_interactions',interactions),('decisions',decision_rows),
                          ('phase_mechanisms',mechanisms),('topology_gaps',topology_rows)]:
        if data:
            write_csv(out/f'{filename}.csv',data)
    cross_code = sum(r['data_and_rng_compatible'] and not r['training_code_hashes_equal'] for r in pairing_checks)
    lines = ['# LA-control experiment summary','',f'Completed runs: {len(runs)}. Per-run protocol checks: {valid}.',
             f'Cross-code comparison pairs: {cross_code} (descriptive only; not identical-code validation).','',
             '| Run | Last20 Overall | Last20 Non-tail | Last20 Tail | Head20 | Middle60 | A refreshes | Normal AB phases | Optimizer steps | Accepted A | Freeze round |',
             '|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for run in runs:
        vals = [mean([run['curve'][r] for r in range(81,101)],k) for k in METRICS]
        cost = next(x for x in costs if x['run']==run['run'])
        lines.append(f'| {run["run"]} | {vals[0]:.4f} | {vals[1]:.4f} | {vals[2]:.4f} | {vals[3]:.4f} | {vals[4]:.4f} | {cost["a_refresh_events"]} | {cost["normal_ab_events"]} | {cost["total_optimizer_steps"]} | {cost["accepted_A_decisions"]} | {cost["freeze_round"]} |')
    lines += ['', 'Ablations: E2-E0, E3-E1, E1-E0, E3-E2; interaction=(E3-E2)-(E1-E0).',
              'J vs E3: normal AB joint training vs normal B-only; both retain nine extra A epochs.',
              'S vs E3: 90 vs 9 extra A epochs; unequal budget, not a pure frequency ablation.',
              'Head20/Middle60/Tail20 are fixed training-frequency groups; Non-tail is not Head20.',
              'A refreshes count actual phases; Accepted A counts controller decisions only.',
              'Cross-code comparisons are exported with an explicit flag, not certified as identical-code pairs.',
              'E5 comparisons: E5 vs E2, then E5 vs E3 at the SAME A learning rate.',
              'Legacy C1/C2 are historical references, not matched v2-RNG CE controls.',
              'noniid-labeldir-fine is standard class-wise Dirichlet; matched-dirichlet is historical fixed-capacity control.',
              'Standard Dirichlet preserves the global pool but not client sizes, FedAvg weights, or exact optimizer-step counts.',
              'Training-side feedback is not held-out validation. No test checkpoint is selected.',
              'Phase budgets use audited events only; unselected branches and server restore are separate.',
              'Gate thresholds, look-ahead length and seeds are reported separately, never selected by test maximum.',
              'A larger test score at one seed does not establish significance or an equal-compute benefit.']
    (out/'report.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    if runs:
        plot(out,runs,decision_rows)
    print(f'Summary written: {out / "report.md"}',flush=True)


def plot(out,runs,decisions):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    fig,axes = plt.subplots(1,len(METRICS),figsize=(25,4))
    for run in runs:
        for ax,metric in zip(axes,METRICS):
            ax.plot(range(101),[float(run['curve'][r][metric]) for r in range(101)],label=run['run'])
            ax.set(title=metric,xlabel='Logical round',ylabel='Accuracy (%)')
    axes[-1].legend(fontsize=5)
    fig.tight_layout()
    fig.savefig(out/'training_curves.png',dpi=160)
    plt.close(fig)
    if decisions:
        fig,axes = plt.subplots(1,3,figsize=(15,4))
        for run in sorted({r['run'] for r in decisions}):
            rows = [r for r in decisions if r['run']==run]
            x = [int(r['candidate_round']) for r in rows]
            for ax,field in zip(axes,('G_all','G_tail','G_hist')):
                ax.plot(x,[float(r[f'instant_{field}']) for r in rows],'--',label=run+' instant')
                ax.plot(x,[float(r[f'delayed_{field}']) for r in rows],'-o',label=run+' delayed')
                ax.axhline(0,color='gray',lw=.5)
                ax.set(title=field,xlabel='Candidate round',ylabel='Cosine-margin units')
        axes[-1].legend(fontsize=5)
        fig.tight_layout()
        fig.savefig(out/'instant_vs_delayed.png',dpi=160)
        plt.close(fig)
