"""Assemble historical A/B evidence without modifying any experiment."""
import csv
import json
from pathlib import Path
import numpy as np

BASE=Path(__file__).resolve().parents[1]
ROOT=BASE/'output/old_ab_decision_20261015'
def readcsv(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def savecsv(name,rows):
    with (ROOT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)
FIELDS={'Overall':'last20_overall_acc','Head20':'last20_head20_acc','Middle60':'last20_middle60_acc','Tail20':'last20_bottom20_tail_acc'}
raw=json.loads((ROOT/'server_evidence.json').read_text(encoding='utf-8'))
remote={r['name']:r for r in raw}
rows=[]
for name in ('oldA_oldB_joint','oldA_noB_joint','plainA_joint'):
    r=remote[name]
    rows.append(dict(method=name,protocol='joint',seed=42,**r['metrics'],source=r['server_run'],
        caveat=('4 concurrent clients' if name=='oldA_oldB_joint' else
                'code_unchanged=false historical receipt' if name=='oldA_noB_joint' else '')))
for r in readcsv(BASE/'output/completed_ab_20261015/performance.csv'):
    rows.append(dict(method=r['method'],protocol='joint',seed=42,**{k:float(r[k]) for k in FIELDS},
        source='output/completed_ab_20261015/performance.csv',caveat='8 concurrent clients'))
for r in readcsv(BASE/'output/completed_ab_20261015/ordinary_A_controls.csv'):
    if r['method']=='plain_joint':continue
    rows.append(dict(method=r['method'],protocol='joint',seed=42,**{k:float(r[k]) for k in FIELDS},
        source='output/completed_ab_20261015/ordinary_A_controls.csv',caveat='8 concurrent clients'))
tailroot=BASE/'output/plain_a_aggregation_v1_parallel8_results/plain_a_aggregation_v1_parallel8'
tailrun=tailroot/'tailrw16/runs/seed42/plain_a'
r=next(r for r in readcsv(tailroot/'analysis/performance.csv') if r['method']=='tailrw16')
tailmetrics={k:float(r[v]) for k,v in FIELDS.items()}
observed=[]
for rnd in range(81,101):
    with np.load(tailrun/'predictions'/f'r{rnd:03d}.npz',allow_pickle=False) as p:
        y=p['class_id']; correct=p['correct'].astype(bool)
        assert np.array_equal(correct,p['prediction']==y)
        observed.append(dict(Overall=correct.mean()*100,Head20=correct[y<20].mean()*100,
            Middle60=correct[(y>=20)&(y<80)].mean()*100,Tail20=correct[y>=80].mean()*100))
assert all(abs(tailmetrics[k]-np.mean([r[k] for r in observed]))<1e-9 for k in FIELDS)
rows.append(dict(method='plainA_tailrw16',protocol='TailRW16',seed=42,**tailmetrics,
    source=str(tailrun.relative_to(BASE)),caveat='local archived predictions; RTX4090 torch2.6 vs joint RTX3090 torch2.12'))
savecsv('joint_and_strong_control.csv',rows)

samples=[]
archived=readcsv(BASE/'output/ab_decision_review_20261003/per_run.csv')
for r in raw:
    if not r['name'].startswith('sample_weighted_'):continue
    suffix=r['name'][len('sample_weighted_'):];arm,seedpart=suffix.rsplit('_',1);seed=int(seedpart[4:])
    local=next(p for p in archived if p['arm']==arm and int(p['seed'])==seed)
    assert all(abs(r['metrics'][k]-float(local[v]))<1e-9 for k,v in FIELDS.items())
    samples.append(dict(method=arm,seed=seed,protocol='original_sample_weighted',**r['metrics'],source=r['server_run']))
savecsv('original_two_seed_results.csv',samples)

lookup={r['method']:r for r in rows}
pairs=[('oldA_oldB_joint','newA_oldB'),('oldA_oldB_joint','newA_newB'),
    ('oldA_oldB_joint','oldA_noB_joint'),('oldA_oldB_joint','plainA_tailrw16'),
    ('oldA_noB_joint','plainA_tailrw16'),('newA_noB','plainA_joint'),('newA_newB','plain_new_b_joint')]
contrasts=[]
for a,b in pairs:
    contrasts.append(dict(comparison=a+' minus '+b,protocol='historical_configuration_comparison',seed=42,
        **{k:lookup[a][k]-lookup[b][k] for k in FIELDS}))
for seed in (42,3407):
    d={r['method']:r for r in samples if r['seed']==seed}
    for a,b in (('a','s'),('ab','a'),('a','a-current'),('a','a-flat'),('ab','a-calibration')):
        contrasts.append(dict(comparison=a+' minus '+b,protocol='original_sample_weighted',seed=seed,
            **{k:d[a][k]-d[b][k] for k in FIELDS}))
savecsv('contrasts_pp.csv',contrasts)
meta=json.loads((BASE/'output/completed_ab_20261015/evidence/selected_a_b_comparison_20261015_01/new_a_joint/runs/seed42/method_a_margin/bridge_metadata.json').read_text())
identities={k:v==meta.get(k) for k,v in remote['oldA_oldB_joint']['identities'].items()}
assert all(identities.values())
audit=dict(server_runs_found=len(raw),server_joint_prediction_files_recomputed=60,
    local_tailrw_prediction_files_recomputed=20,old_two_seed_metrics_match_previous_audit=True,
    oldAB_vs_newA_identity_checks=identities,
    oldAB_all_8_donor_pools_size30=all(int(r['unique_donors'])==30 for r in remote['oldA_oldB_joint']['donor_pools']),
    original_protocol_seeds=[42,3407],seed0_present=False,
    warning='These checks do not repair the oldA_noB_joint code_unchanged=false receipt or remove environment/concurrency confounding.')
(ROOT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
print(json.dumps(audit,ensure_ascii=False,indent=2))
for r in contrasts[:7]:print(r)
