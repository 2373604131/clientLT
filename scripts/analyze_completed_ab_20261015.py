"""Offline analysis of committed prediction files; no training or model selection."""
import csv
import json
from pathlib import Path
import statistics

import numpy as np

ROOT=Path(__file__).resolve().parents[1]/'output/completed_ab_20261015'
EVIDENCE=ROOT/'evidence'
MARGIN='runs/seed42/method_a_margin'
SOURCES={
    'frozenA_noB':('no_b_ablation_20261015_01','runs/seed42/frozen'),
    'frozenA_oldB':('a_schedule_old_b_20261015_01/a_frozen',MARGIN),
    'newA_noB':('selected_a_b_comparison_20261015_01/new_a_joint',MARGIN),
    'newA_oldB':('selected_a_b_comparison_20261015_01/new_a_old_b_joint',MARGIN),
    'newA_newB':('ab_parameter_search_20261014_01/S4',MARGIN),
    'A60_oldB':('a_schedule_old_b_20261015_01/a_until60',MARGIN),
    'A9_oldB':('a_schedule_old_b_20261015_01/a_every10',MARGIN)}
GROUPS={'Overall':np.arange(100),'Head20':np.arange(20),'Middle60':np.arange(20,80),'Tail20':np.arange(80,100)}
METRIC_MAP=dict(Overall='overall_acc',Head20='head20_acc',Middle60='middle60_acc',Tail20='bottom20_tail_acc')
IDENTITIES=('initial_lora_sha256','frozen_model_sha256','schedule_sha256','pool_sha256','test_sha256',
            'probe_images_sha256','probe_manifest_sha256','client_sample_counts','classnames')
INTEGRATION={'scripts/run_cliplora_joint_aggregation.py','scripts/run_cliplora_new_a_old_b.py',
    'tools/client_aggregation/analysis.py','tools/client_aggregation/new_a_old_b.py',
    'tools/client_aggregation/protocol.py','tools/client_aggregation/runtime.py',
    'scripts/run_cliplora_a_schedule.py','tools/client_aggregation/a_schedule.py','utils/cliplora_margin_a_schedule.py'}


def load(path):return json.loads(Path(path).read_text(encoding='utf-8'))
def read_csv(path):
    if not Path(path).exists():return []
    with Path(path).open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def save(name,value):
    (ROOT/name).write_text(json.dumps(value,indent=2,ensure_ascii=False,allow_nan=False)+'\n',encoding='utf-8')
def table(name,rows):
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)
def pred(path):
    with np.load(path,allow_pickle=False) as data:
        r={k:data[k].copy() for k in ('sample_id','class_id','prediction','correct','logit_margin')}
    r['correct']=r['correct'].astype(bool)
    assert np.array_equal(r['sample_id'],np.arange(10000))
    assert np.array_equal(r['correct'],r['class_id']==r['prediction'])
    assert np.array_equal(np.bincount(r['class_id'],minlength=100),np.full(100,100))
    return r
def scores(p):return {g:float(p['correct'][np.isin(p['class_id'],ids)].mean()*100) for g,ids in GROUPS.items()}


def main():
    audit=load(ROOT/'server_audit.json')
    assert {r['name'] for r in audit}==set(SOURCES) and all(r['passed'] and r['completed_round']==100 for r in audit)
    results={};curves=[];performance=[];stage_effects=[];b_events=[];mechanism=[];pair_checks=[]
    for name,(relative,suffix) in SOURCES.items():
        folder=EVIDENCE/relative;run=folder/suffix
        config=load(run/'sfra_config.json');plan=load(folder/'experiment_plan.json')
        predictions={r:pred(run/'predictions'/f'r{r:03d}.npz') for r in range(101)}
        rows=[]
        for r,p in predictions.items():
            metric=scores(p);saved=load(run/'predictions'/f'r{r:03d}.json')
            assert all(abs(metric[k]-saved[v])<1e-9 for k,v in METRIC_MAP.items())
            row=dict(method=name,round=r,**metric);rows.append(row);curves.append(row)
        completion=load(run/'completion.json')
        row=dict(method=name,**{k:statistics.mean(r[k] for r in rows[81:101]) for k in GROUPS},
            final_Overall=rows[100]['Overall'],final_Tail20=rows[100]['Tail20'],
            elapsed_hours=completion['elapsed_seconds']/3600,A_updates=len(config['refresh_rounds']),
            B_steps=completion['normal_optimizer_steps'],A_steps=completion['extra_optimizer_steps'])
        a_rows=read_csv(run/'sfra_rounds.csv');b_rows=read_csv(run/'b_transfer_rounds.csv')
        row['A_repair_minutes']=sum(float(r.get('repair_seconds') or 0) for r in a_rows)/60
        row['B_transfer_minutes']=sum(float(r['seconds']) for r in b_rows)/60
        performance.append(row)
        # Direct, same-state diagnostic effects, separate from long-term ablations.
        for stage,operation in (('after_B_transfer','extra_B'),('ordinary_A','A_correction')):
            for path in sorted((run/'stage_predictions'/stage).glob('r*.npz')):
                rnd=int(path.stem[1:])
                if operation=='extra_B':before=pred(run/'stage_predictions/ordinary_B'/path.name);after=pred(path)
                else:before=pred(path);after=predictions[rnd]
                for group,ids in GROUPS.items():
                    mask=np.isin(before['class_id'],ids)
                    gain=int(np.sum(mask & ~before['correct'] & after['correct']))
                    loss=int(np.sum(mask & before['correct'] & ~after['correct']))
                    stage_effects.append(dict(method=name,round=rnd,operation=operation,group=group,
                        wrong_to_correct=gain,correct_to_wrong=loss,delta_pp=(gain-loss)/int(mask.sum())*100))
        if name.endswith('oldB'):
            for b in b_rows:
                rnd=int(b['round'])
                b_events.append(dict(method=name,round=rnd,donor_links=int(b['donor_links']),
                    unique_donors=int(b['unique_donors']),receivers=int(b['receivers_with_donors']),
                    tail_training_LA_gain=float(b['ordinary_global_B_tail_la'])-float(b['transferred_global_B_tail_la']),
                    non_tail_training_LA_gain=float(b['ordinary_global_B_non_tail_la'])-float(b['transferred_global_B_non_tail_la']),
                    effective_update=float(b['effective_global_transfer_norm']),
                    correction_to_ordinary_B=float(b['effective_global_transfer_norm'])/float(b['ordinary_B_effective_update_norm'])))
        active=[r for r in a_rows if r.get('ordinary_keep') not in ('',None)]
        mechanism.append(dict(method=name,A_loss_observations=len(active),
            keep_decreased=sum(float(r['final_keep'])<float(r['ordinary_keep']) for r in active),
            class_decreased=sum(float(r['final_class'])<float(r['ordinary_class']) for r in active),
            both_decreased=sum(float(r['final_keep'])<float(r['ordinary_keep']) and float(r['final_class'])<float(r['ordinary_class']) for r in active),
            mean_keep_gain=statistics.mean(float(r['ordinary_keep'])-float(r['final_keep']) for r in active) if active else 0,
            mean_class_gain=statistics.mean(float(r['ordinary_class'])-float(r['final_class']) for r in active) if active else 0))
        results[name]=dict(folder=folder,run=run,plan=plan,config=config,metadata=load(run/'bridge_metadata.json'),
            predictions=predictions,curves=rows,endpoint=row)
    ref=results['newA_noB']
    for name,r in results.items():
        assert r['plan']['aggregation']==ref['plan']['aggregation']
        assert all(r['metadata'][k]==ref['metadata'][k] for k in IDENTITIES)
        assert r['config']['client_execution']==ref['config']['client_execution']
        a,b=r['config']['base_training_config'],ref['config']['base_training_config']
        assert all(a.get(k)==b.get(k) for k in set(a)|set(b) if k not in ('candidate_rounds','extra_steps_expected','extra_trainable_factor'))
        assert all(np.array_equal(r['predictions'][0][k],ref['predictions'][0][k]) for k in ('sample_id','class_id','prediction','correct'))
        codes=[x['plan']['code_sha256'] for x in (r,ref)]
        diffs=sorted(k for k in set(codes[0])|set(codes[1]) if codes[0].get(k)!=codes[1].get(k))
        assert set(diffs)<=INTEGRATION,(name,diffs)
        pair_checks.append(dict(method=name,passed=True,reference='newA_noB',source_integration_differences=diffs))
    comparisons=[];flips=[];per_class=[]
    pairs=[('frozenA_oldB','frozenA_noB'),('newA_oldB','newA_noB'),('newA_newB','newA_noB'),
        ('newA_oldB','newA_newB'),('newA_noB','frozenA_noB'),('newA_oldB','frozenA_oldB'),
        ('A60_oldB','newA_oldB'),('A9_oldB','newA_oldB')]
    for left,right in pairs:
        a,b=results[left],results[right]
        comparisons.append(dict(contrast=left+' minus '+right,**{k:a['endpoint'][k]-b['endpoint'][k] for k in GROUPS}))
        for rnd in range(81,101):
            x,y=a['predictions'][rnd],b['predictions'][rnd]
            for group,ids in GROUPS.items():
                mask=np.isin(x['class_id'],ids)
                flips.append(dict(left=left,right=right,round=rnd,group=group,
                    only_left_correct=int(np.sum(mask & x['correct'] & ~y['correct'])),
                    only_right_correct=int(np.sum(mask & ~x['correct'] & y['correct']))))
        for c in range(100):
            values=[float((a['predictions'][r]['correct'].astype(float)-b['predictions'][r]['correct'])[a['predictions'][r]['class_id']==c].mean()*100) for r in range(81,101)]
            per_class.append(dict(left=left,right=right,class_id=c,class_name=a['metadata']['classnames'][c],delta_pp=statistics.mean(values)))
    interaction={k:(results['newA_oldB']['endpoint'][k]-results['newA_noB']['endpoint'][k])-
        (results['frozenA_oldB']['endpoint'][k]-results['frozenA_noB']['endpoint'][k]) for k in GROUPS}
    peaks=[]
    for name,r in results.items():
        for metric in ('Overall','Tail20'):
            maximum=max(row[metric] for row in r['curves'])
            peaks.append(dict(method=name,metric=metric,peak=maximum,
                rounds=[row['round'] for row in r['curves'] if row[metric]==maximum],
                final=r['curves'][100][metric],peak_to_final=maximum-r['curves'][100][metric]))
    # Check recomputed endpoints against each independently audited automatic report.
    mappings={'newA_noB':('selected_a_b_comparison_20261015_01','new_a_joint'),
        'newA_oldB':('selected_a_b_comparison_20261015_01','new_a_old_b_joint'),
        'newA_newB':('selected_a_b_comparison_20261015_01','new_a_new_b_joint'),
        'A60_oldB':('a_schedule_old_b_20261015_01','a_until60'),
        'A9_oldB':('a_schedule_old_b_20261015_01','a_every10'),
        'frozenA_oldB':('a_schedule_old_b_20261015_01','a_frozen'),
        'frozenA_noB':('no_b_ablation_20261015_01','frozen_A_no_extra_B')}
    for name,(folder,key) in mappings.items():
        report=load(EVIDENCE/folder/'analysis/report.json')
        saved=next(r for r in report['performance'] if r.get('method',r.get('cell'))==key)
        assert all(abs(results[name]['endpoint'][k]-saved[v])<1e-9 for k,v in METRIC_MAP.items())
    for filename,rows in [('performance.csv',performance),('contrasts_pp.csv',comparisons),('curves.csv',curves),
        ('immediate_stage_effects.csv',stage_effects),('old_B_events.csv',b_events),('A_training_losses.csv',mechanism),
        ('paired_prediction_counts.csv',flips),('per_class_deltas.csv',per_class)]:table(filename,rows)
    stage_summary=[]
    for name in results:
        for op in ('extra_B','A_correction'):
            for group in GROUPS:
                rows=[r for r in stage_effects if r['method']==name and r['operation']==op and r['group']==group]
                if rows:stage_summary.append(dict(method=name,operation=op,group=group,observed_rounds=[r['round'] for r in rows],
                    mean_delta_pp=statistics.mean(r['delta_pp'] for r in rows),
                    total_corrected=sum(r['wrong_to_correct'] for r in rows),total_damaged=sum(r['correct_to_wrong'] for r in rows)))
    table('immediate_effects_summary.csv',stage_summary)
    save('analysis.json',dict(all_endpoints_recomputed_from_predictions=True,primary_window=[81,100],
        evaluation='Development results on repeatedly inspected CIFAR-100 official test; one seed42; rounds are not independent replicates.',
        performance=performance,contrasts=comparisons,interaction_pp=interaction,
        pair_checks=pair_checks,peaks_descriptive_only=peaks,A_losses=mechanism,immediate_effects=stage_summary))
    print(json.dumps(dict(performance=performance,contrasts=comparisons,interaction_pp=interaction,peaks=peaks,
        immediate_effects=stage_summary,A_losses=mechanism),indent=2))
    plot(results)


def plot(results):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    styles={'frozenA_noB':('Frozen A / no extra B','#999999','--'),
        'frozenA_oldB':('Frozen A / old B','#555555','-'),
        'newA_noB':('A 1-90 / no extra B','#4c78a8','-'),
        'newA_oldB':('A 1-90 / old B','#f58518','-'),
        'newA_newB':('A 1-90 / current B','#54a24b','-'),
        'A60_oldB':('A 1-60 / old B','#b279a2','-'),
        'A9_oldB':('A every 10 / old B','#e45756','-')}
    fig,axes=plt.subplots(2,2,figsize=(12,7.5),sharex=True)
    for ax,metric in zip(axes.flat,GROUPS):
        ax.axvspan(81,100,color='grey',alpha=.10)
        for name,(label,color,style) in styles.items():
            ax.plot(range(101),[r[metric] for r in results[name]['curves']],label=label,color=color,ls=style,lw=1.35)
        ax.set_title(metric);ax.set_ylabel('Accuracy (%)');ax.grid(alpha=.2);ax.set_xlim(0,100)
    for ax in axes[1]:ax.set_xlabel('Committed round')
    handles,labels=axes[0,0].get_legend_handles_labels()
    fig.legend(handles,labels,loc='lower center',ncol=4,fontsize=9,bbox_to_anchor=(.5,.005))
    fig.suptitle('A schedules and B ablations | seed42 | shaded: fixed rounds 81-100',fontsize=13)
    fig.tight_layout(rect=(0,.095,1,.96))
    fig.savefig(ROOT/'training_curves.png',dpi=180);fig.savefig(ROOT/'training_curves.pdf')
    plt.close(fig)


if __name__=='__main__':main()
