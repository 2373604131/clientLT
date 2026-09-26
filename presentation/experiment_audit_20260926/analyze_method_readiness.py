"""Read-only diagnostics for deciding whether A/B are ready to freeze."""
import csv,json
from pathlib import Path
import numpy as np

OUT=Path(__file__).resolve().parent
ROOT=OUT.parents[1]
data=json.loads((OUT/'classified_inventory.json').read_text(encoding='utf-8'))
runs={r['run_id']:r for r in data['runs']}
def read(p):
    with p.open(encoding='utf-8-sig',newline='') as f:return list(csv.DictReader(f))
def write(name,rows):
    with (OUT/name).open('w',encoding='utf-8-sig',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        w.writeheader();w.writerows(rows)
def mean(a):return float(np.mean(a)) if len(a) else None

labels={'R29b405139':'Full-CP μ1','Ra116c9874':'Full-CP μ3','R2026b9a85':'Flat-CP μ1',
        'R23269e0bb':'Full λ10 无CP','R3b1af80e9':'S','R16bae8362':'E3',
        'R852d06edf':'Shared B','R5e75471ac':'旧本地B lr0.3','R42b48f11a':'Uniform8+B','Rc84498c4f':'Uniform8'}
pairs=[('R29b405139','R3b1af80e9'),('R29b405139','R16bae8362'),('R29b405139','R2026b9a85'),
       ('R29b405139','R23269e0bb'),('Ra116c9874','R29b405139'),('R852d06edf','R29b405139'),
       ('R852d06edf','R3b1af80e9'),('R5e75471ac','R29b405139'),('R42b48f11a','Rc84498c4f')]
contrasts=[]
for a,b in pairs:
    row={'比较':labels[a]+' − '+labels[b],'左侧来源':runs[a]['run_path'],'右侧来源':runs[b]['run_path']}
    for key in ['last20_overall','last20_head20','last20_middle60','last20_tail','last20_many_gt100','last20_medium20to100','last20_few_lt20','tail_peak_to_last']:
        row[key]=runs[a][key]-runs[b][key]
    contrasts.append(row)
write('方法定稿_关键比较.csv',contrasts)

geometry=[];groups=[];rounds_out=[]
for rid in ['R29b405139','Ra116c9874','R2026b9a85']:
    p=ROOT/runs[rid]['run_path']
    summaries=read(p/'sfra_rounds.csv')
    late=[r for r in summaries if 61<=int(r['round'])<=90]
    witness=json.loads((p/'private_witness_manifest.json').read_text(encoding='utf-8'))
    assert [w['token_id'] for w in witness]==list(range(len(witness)))
    cls=np.array([w['class_id'] for w in witness])
    counts=np.array(json.loads((p/'partition_summary.json').read_text(encoding='utf-8'))['global_class_counts'])
    class_sets={'Head20':np.arange(20),'Middle60':np.arange(20,80),'Tail20':np.arange(80,100),
                'Many35':np.flatnonzero(counts>100),'Medium35':np.flatnonzero((counts>=20)&(counts<=100)),'Few30':np.flatnonzero(counts<20)}
    steps=[]
    for rnd in range(61,91):
        folder=p/'sfra_rounds'/f'r{rnd:03d}'
        for s in read(folder/'correction_steps.csv'):
            s['round']=rnd;steps.append(s)
        with np.load(folder/'tokens.npz') as z:
            # Values here are training-side scores, not test accuracy or independent seeds.
            post=z['F_post_B'].astype(float).mean(1)
            proposal=z['F_proposal'].astype(float).mean(1)
            committed=z['F_committed'].astype(float).mean(1)
            weights=z['weights'].astype(float)
            cpweights=z['classification_global_token_weights'].astype(float)
            cpchange=(z['classification_committed_scores']-z['classification_proposal_scores']).astype(float).mean(1)
            for name,ids in class_sets.items():
                mask=np.isin(cls,ids)
                # Equal classes; within a class, average over its client-class units.
                def balanced(v):return mean([v[cls==c].mean() for c in ids if np.any(cls==c)])
                rounds_out.append({'方法':labels[rid],'round':rnd,'类别组':name,
                    '普通提案margin变化乘1e4':balanced(proposal-post)*1e4,
                    '修正margin变化乘1e4':balanced(committed-proposal)*1e4,
                    '提交A净margin变化乘1e4':balanced(committed-post)*1e4,
                    '分类LA修正变化_类均衡':balanced(cpchange),
                    '功能权重份额':float(weights[mask].sum()),'CP显式样本权重份额':float(cpweights[mask].sum())})
    later_steps=[s for s in steps if int(s['step'])>1]
    nonzero=[s for s in later_steps if s['functional_classification_gradient_cosine']]
    geometry.append({'方法':labels[rid],'轮次':'61–90','有效更新范数比_各自普通提案':mean([float(r['committed_effective_norm'])/float(r['proposal_effective_norm']) for r in late]),
        'A参数方向余弦_各自普通提案':mean([float(r['proposal_committed_cosine']) for r in late]),
        '投影边界步数':sum(int(r['projection_boundary_steps']) for r in late),
        'CP激活步骤2和3比例':mean([s['classification_active']=='True' for s in later_steps]),
        'CP与功能项梯度余弦_均值非零步':mean([float(s['functional_classification_gradient_cosine']) for s in nonzero]),
        '提交后训练LA高于普通提案轮数':sum(float(r['committed_classification_loss_increase'])>1e-7 for r in late),
        '提交后训练LA变化均值':mean([float(r['committed_classification_loss_increase']) for r in late]),
        '来源':runs[rid]['run_path']})
    for name in class_sets:
        selected=[r for r in rounds_out if r['方法']==labels[rid] and r['类别组']==name]
        groups.append({'方法':labels[rid],'类别组':name,'轮次':'61–90',**{k:mean([r[k] for r in selected]) for k in selected[0] if k not in ['方法','round','类别组']}})
write('方法A_更新与CP诊断.csv',geometry)
write('方法A_分组功能诊断.csv',groups)
write('方法A_逐轮功能诊断.csv',rounds_out)
(OUT/'方法定稿_诊断摘要.json').write_text(json.dumps({'comparisons':contrasts,'geometry':geometry,'group_diagnostics':groups,
    'scope':'仅复算现有seed42日志；组内先按类平均，再平均轮次。训练侧功能变化不是泛化因果证明。完整运行间差值受执行版本差异限制。'},ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({'geometry':geometry,'groups':groups[:6]},ensure_ascii=False))
