"""Mechanism and historical ordinary-A checks using already downloaded outputs."""
import json
from pathlib import Path
import statistics
import sys
sys.path.insert(0,str(Path(__file__).resolve().parent))
import analyze_completed_ab_20261015 as a
import numpy as np

def main():
    detail={};controls=[];source_audits=[]
    result=a.load(a.ROOT/'analysis.json')
    endpoints={r['method']:r for r in result['performance']}
    for old,new in (('plain_joint','newA_noB'),('plain_new_b_joint','newA_newB')):
        folder=a.EVIDENCE/'ordinary_A_controls'/old;run=folder/'runs/seed42/plain_a'
        target=a.EVIDENCE/a.SOURCES[new][0]/a.SOURCES[new][1]
        old_plan=a.load(folder/'experiment_plan.json');new_plan=a.load(target.parents[2]/'experiment_plan.json')
        assert old_plan['aggregation']==new_plan['aggregation']
        m=[a.load(x/'bridge_metadata.json') for x in (run,target)]
        assert all(m[0][k]==m[1][k] for k in a.IDENTITIES)
        c=[a.load(x/'sfra_config.json') for x in (run,target)]
        assert c[0]['client_execution']==c[1]['client_execution']
        assert c[0]['refresh_rounds']==c[1]['refresh_rounds']
        assert c[0]['base_training_config']==c[1]['base_training_config']
        assert c[0].get('b_transfer')==c[1].get('b_transfer')
        p=[a.pred(x/'predictions/r000.npz') for x in (run,target)]
        assert all(np.array_equal(p[0][k],p[1][k]) for k in ('sample_id','class_id','prediction','correct'))
        codes=[old_plan['code_sha256'],new_plan['code_sha256']]
        diffs=sorted(k for k in set(codes[0])|set(codes[1]) if codes[0].get(k)!=codes[1].get(k))
        allowed=a.INTEGRATION|{'federated_main.py','utils/method_a_margin_math.py','utils/cliplora_method_a_margin.py'}
        assert set(diffs)<=allowed,diffs
        rows=[a.scores(a.pred(run/'predictions'/f'r{r:03d}.npz')) for r in range(81,101)]
        values={k:statistics.mean(r[k] for r in rows) for k in a.GROUPS}
        controls.append(dict(method=old,**values,**{k+'_newA_minus_ordinary_pp':endpoints[new][k]-values[k] for k in a.GROUPS}))
        source_audits.append(dict(ordinary=old,newA=new,passed=True,source_differences=diffs,
            note='Historical control; original stable-code receipt verified, same data/init/ordinary training/extra B; allowed opt-in integration and A parameter argument additions.'))
    detail['ordinary_A_controls']=controls;detail['ordinary_A_pair_audits']=source_audits
    prefixes=[]
    for left,right,stop in (('newA_oldB','newA_noB',29),('frozenA_oldB','frozenA_noB',29),('A60_oldB','newA_oldB',60)):
        runs=[a.EVIDENCE/a.SOURCES[n][0]/a.SOURCES[n][1] for n in (left,right)]
        state_same=[];pred_same=[];differences=[]
        for r in range(stop+1):
            j=[a.load(run/'predictions'/f'r{r:03d}.json') for run in runs]
            p=[a.pred(run/'predictions'/f'r{r:03d}.npz') for run in runs]
            state_same.append(all(j[0][k]==j[1][k] for k in ('a_sha256','b_sha256')))
            pred_same.append(all(np.array_equal(p[0][k],p[1][k]) for k in ('prediction','correct','logit_margin')))
            if not state_same[-1] or not pred_same[-1]:
                differences.append(dict(round=r,changed_predictions=int(np.sum(p[0]['prediction']!=p[1]['prediction'])),
                    changed_correctness=int(np.sum(p[0]['correct']!=p[1]['correct'])),
                    max_margin_difference=float(np.max(np.abs(p[0]['logit_margin']-p[1]['logit_margin'])))))
        prefixes.append(dict(left=left,right=right,through_round=stop,all_parameter_hashes_identical=all(state_same),
            all_predictions_and_margins_identical=all(pred_same),differences=differences))
    detail['before_intervention_checks']=prefixes
    b_summary=[]
    for name in ('frozenA_oldB','newA_oldB','A60_oldB','A9_oldB'):
        folder=a.EVIDENCE/a.SOURCES[name][0];run=folder/a.SOURCES[name][1]
        matrix=np.asarray(a.load(folder/'experiment_plan.json')['aggregation']['matrix'])
        scores=[]
        for rnd in range(30,101,10):
            rows=a.read_csv(run/'b_transfer_rounds'/f'r{rnd:03d}'/'donor_scores.csv')
            assert all(matrix[int(r['donor']),int(r['class_id'])]==0 for r in rows)
            assert all(int(r['class_id'])>=80 for r in rows)
            scores.extend(rows)
        events=[r for r in a.read_csv(a.ROOT/'old_B_events.csv') if r['method']==name]
        b_summary.append(dict(method=name,events=len(events),probe_records=len(scores),
            all_probed_sources_lack_target_class=True,positive_probe_records=sum(r['positive']=='True' for r in scores),
            tail_LA_improved_events=sum(float(r['tail_training_LA_gain'])>0 for r in events),
            non_tail_LA_improved_events=sum(float(r['non_tail_training_LA_gain'])>0 for r in events),
            mean_tail_LA_gain=statistics.mean(float(r['tail_training_LA_gain']) for r in events),
            mean_non_tail_LA_gain=statistics.mean(float(r['non_tail_training_LA_gain']) for r in events),
            min_update_ratio=min(float(r['correction_to_ordinary_B']) for r in events),
            max_update_ratio=max(float(r['correction_to_ordinary_B']) for r in events)))
    detail['old_B_mechanism']=b_summary
    round_values=[]
    for name in a.SOURCES:
        run=a.EVIDENCE/a.SOURCES[name][0]/a.SOURCES[name][1]
        for rnd in (50,60,90,100):round_values.append(dict(method=name,round=rnd,**a.scores(a.pred(run/'predictions'/f'r{rnd:03d}.npz'))))
    detail['selected_round_values']=round_values
    a.table('ordinary_A_controls.csv',controls);a.table('old_B_mechanism_summary.csv',b_summary)
    a.save('supplement.json',detail)
    print(json.dumps(detail,indent=2))

if __name__=='__main__':main()
