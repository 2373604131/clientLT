"""Audit the seed42 CAPT global-start artifacts and export descriptive results."""
from pathlib import Path
import csv
import hashlib
import json
import re
import subprocess

import numpy as np
import torch
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
BASE = ROOT/'output/capt_global_start_seed42_results/capt_global_start_lambda1'
OUT = ROOT/'output/capt_global_start_seed42_results/analysis'
PARTS = ['client-longtail', 'noniid-labeldir-fine']


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def class_matrix(path):
    rs = sorted(read_csv(path/'client_class_counts.csv'), key=lambda r: int(r['client_id']))
    return np.array([[int(float(r[f'class_{c}'])) for c in range(100)] for r in rs])


def class_acc(path, epoch):
    rs = read_csv(path/f'per_class_accuracy_epoch_{epoch}.csv')
    by_id = {int(r['class_id']): float(r['per_class_acc']) for r in rs}
    assert set(by_id) == set(range(100))
    return np.array([by_id[i] for i in range(100)])


def run(path, schedule):
    command = json.loads((path/'command.json').read_text())
    finish = json.loads((path/'finished.json').read_text())
    assert finish == dict(exit_code=0, requested_rounds=100)
    opts = command['command']
    opts = dict(zip(opts[3::2], opts[4::2]))
    assert opts['--capt_reset_global_before_client'] == 'True'
    assert opts['--model'] == 'cluster' and opts['--trainer'] == 'CAPT'
    assert opts['--seed'] == '42' and opts['--frac'] == '1.0'
    audit_schedule = {}
    for r in read_csv(path/'selected_clients.csv'):
        audit_schedule.setdefault(int(r['epoch_index']), []).append((int(r['selection_order']), int(r['client_id'])))
    actual = [[c for _, c in sorted(audit_schedule[e])] for e in range(100)]
    assert actual == schedule
    assert all(sorted(s) == list(range(30)) for s in actual)
    schedule_hash = hashlib.sha256(json.dumps(actual, separators=(',', ':')).encode()).hexdigest()
    assert schedule_hash == command['client_schedule_sha256']
    log = (path/'run.log').read_text(encoding='utf8')
    assert 'restore the latest server model before EVERY client' in log
    training_rounds = [int(x) for x in re.findall(r'Epoch (\d+): CAPT cluster training', log)]
    assert training_rounds == list(range(100))
    skipped = [int(x) for x in re.findall(r'Skipping global aggregation at epoch (\d+)', log)]
    aggregations = sorted(set(training_rounds)-set(skipped))
    rounds = [{**{k: float(v) for k, v in r.items() if k in ['overall_acc','head_acc','tail_acc','macro_f1']},
               'epoch': int(r['epoch'])} for r in read_csv(path/'round_metrics.csv')]
    assert [r['epoch'] for r in rounds] == aggregations
    for r in rounds:
        r['hmean'] = 2*r['head_acc']*r['tail_acc']/(r['head_acc']+r['tail_acc'])
        acc = class_acc(path, r['epoch'])
        assert np.isclose(acc.mean(), r['overall_acc'])
        assert np.isclose(acc[:80].mean(), r['head_acc'])
        assert np.isclose(acc[80:].mean(), r['tail_acc'])
    n = class_matrix(path)
    p = n/n.sum(1, keepdims=True)
    eligible = (p > .1).any(0)
    final = class_acc(path, rounds[-1]['epoch'])
    first = class_acc(path, rounds[0]['epoch'])
    snaps = []
    for r in rounds:
        snap = torch.load(path/f"prompt_params/prompt_params_epoch_{r['epoch']}.pth", map_location='cpu', weights_only=True)
        assert int(snap['epoch']) == r['epoch']
        snaps.append(snap)
    row_delta = (snaps[-1]['class_aware_prompt'].float()-snaps[0]['class_aware_prompt'].float()).flatten(1).norm(dim=1).numpy()
    row_base = snaps[0]['class_aware_prompt'].float().flatten(1).norm(dim=1).numpy()
    last20eval = [r for r in rounds if r['epoch'] >= 80]
    held = []
    for epoch in range(100):
        previous = max((r for r in rounds if r['epoch'] <= epoch), key=lambda r:r['epoch'])
        held.append({**previous, 'outer_epoch': epoch})
    means = lambda rs: {k: float(np.mean([r[k] for r in rs])) for k in ['overall_acc','head_acc','tail_acc','macro_f1','hmean']}
    source = {}
    for name, expected in command['source_sha256'].items():
        raw = (ROOT/name).read_bytes()
        source[name] = dict(exact=hashlib.sha256(raw).hexdigest()==expected,
                           lf=hashlib.sha256(raw.replace(b'\r\n', b'\n')).hexdigest()==expected,
                           crlf=hashlib.sha256(raw.replace(b'\r\n', b'\n').replace(b'\n',b'\r\n')).hexdigest()==expected)
    best_overall = max(rounds, key=lambda r:r['overall_acc'])
    best_tail = max(rounds, key=lambda r:r['tail_acc'])
    best_hmean = max(rounds, key=lambda r:r['hmean'])
    stats = dict(final=rounds[-1], first_post_training=rounds[0], best_overall=best_overall,
                 best_tail=best_tail, best_hmean=best_hmean, aggregation_count=len(aggregations),
                 aggregation_epochs=aggregations, skipped_epochs=skipped,
                 last20_evaluated_count=len(last20eval), last20_evaluated_mean=means(last20eval),
                 last20_server_hold_mean=means(held[80:]),
                 last20_evaluated_tail_std=float(np.std([r['tail_acc'] for r in last20eval])),
                 late_tail_peak_to_final=max(r['tail_acc'] for r in last20eval)-rounds[-1]['tail_acc'],
                 tail_peak_to_final=best_tail['tail_acc']-rounds[-1]['tail_acc'],
                 samples=int(n.sum()), tail_samples=int(n[:,80:].sum()),
                 samples_per_client=n.sum(1).tolist(), tail_clients=n[:,80:].sum(1).tolist(),
                 estimated_steps_per_round=int(np.ceil(n.sum(1)/32).sum()*3),
                 eligible_all=int(eligible.sum()), eligible_tail=int(eligible[80:].sum()),
                 eligible_tail_ids=np.flatnonzero(eligible[80:]).__add__(80).tolist(),
                 mean_tail_support_clients=float((n[:,80:]>0).sum(0).mean()),
                 mean_tail_effective_carriers=float((n[:,80:].sum(0)**2/(n[:,80:]**2).sum(0)).mean()),
                 general_prompt_delta_norm=float((snaps[-1]['general_prompt'].float()-snaps[0]['general_prompt'].float()).norm()),
                 never_eligible_row_relative_delta_max=float((row_delta[~eligible]/row_base[~eligible]).max()),
                 source_hash_audit=source,
                 optimizer_reset=command['optimizer_reset'], test_controls_mab=command['official_test_controls_mab'],
                 source_hashes=command['source_sha256'], schedule_hash=schedule_hash)
    rows=[]
    for c in range(100):
        rows.append(dict(partition=path.name, class_id=c, group='tail' if c>=80 else 'non_tail',
                         global_samples=int(n[:,c].sum()), supporters=int((n[:,c]>0).sum()),
                         eligible_clients=int((p[:,c]>.1).sum()), max_local_proportion=float(p[:,c].max()),
                         first_post_training_acc=float(first[c]), final_acc=float(final[c]),
                         first_to_final_delta=float(final[c]-first[c]), prompt_delta_norm=float(row_delta[c]),
                         prompt_relative_delta=float(row_delta[c]/row_base[c])))
    return stats, rows, rounds, held, n


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    schedule=json.loads(next((BASE/'schedules').glob('*.json')).read_text())['schedule']
    data={p:run(BASE/'seed42'/p, schedule) for p in PARTS}
    assert data[PARTS[0]][0]['source_hashes']==data[PARTS[1]][0]['source_hashes']
    assert data[PARTS[0]][0]['aggregation_epochs']==data[PARTS[1]][0]['aggregation_epochs']
    n0,n1=[data[p][4] for p in PARTS]
    assert np.array_equal(n0.sum(0),n1.sum(0))
    payload = dict(runs={p:data[p][0] for p in PARTS},
                   class_margins_equal=True, client_margins_equal=bool(np.array_equal(n0.sum(1),n1.sum(1))))
    rows=[r for p in PARTS for r in data[p][1]]
    write_csv(OUT/'per_class_analysis.csv',rows)
    paired=[]
    for c in range(100):
        a,b=[data[p][1][c] for p in PARTS]
        paired.append(dict(class_id=c, group=a['group'],clientlt_acc=a['final_acc'],dirichlet_acc=b['final_acc'],
                           delta=a['final_acc']-b['final_acc'],clientlt_eligible=a['eligible_clients'],
                           dirichlet_eligible=b['eligible_clients']))
    write_csv(OUT/'paired_class_differences.csv',paired)
    for group in ['tail','non_tail']:
        subset=[r for r in paired if r['group']==group]
        payload[group+'_paired']=dict(improved=sum(r['delta']>0 for r in subset),
                                      tied=sum(r['delta']==0 for r in subset),
                                      worse=sum(r['delta']<0 for r in subset),mean_delta=float(np.mean([r['delta'] for r in subset])))
    for flag in [True,False]:
        subset=[r for r in paired if r['group']=='tail' and (r['clientlt_eligible']>0)==flag]
        payload['tail_eligible' if flag else 'tail_ineligible']=dict(
            count=len(subset), clientlt_mean=float(np.mean([r['clientlt_acc'] for r in subset])),
            dirichlet_mean=float(np.mean([r['dirichlet_acc'] for r in subset])),
            delta_mean=float(np.mean([r['delta'] for r in subset])),
            improved=sum(r['delta']>0 for r in subset),worse=sum(r['delta']<0 for r in subset),
            tied=sum(r['delta']==0 for r in subset))
    historical_source=subprocess.check_output(['git','show','d0f220e:federated_main.py'],cwd=ROOT)
    historical_hash=hashlib.sha256(historical_source).hexdigest()
    payload['entrypoint_source_commit_audit']=dict(commit='d0f220e',sha256=historical_hash,
        matches_recorded=all(data[p][0]['source_hashes']['federated_main.py']==historical_hash for p in PARTS))
    assert payload['entrypoint_source_commit_audit']['matches_recorded']
    # Search historical matrices; an exact matrix alone does not imply matched training.
    candidates=[]
    for path in (ROOT/'output').rglob('client_class_counts.csv'):
        if BASE.parent in path.parents:
            continue
        if not (path.parent/'round_metrics.csv').exists():
            continue
        try:
            mat=class_matrix(path.parent)
        except (KeyError, ValueError):
            continue
        matches=[p for p in PARTS if mat.shape==data[p][4].shape and np.array_equal(mat,data[p][4])]
        if matches:
            rr=read_csv(path.parent/'round_metrics.csv')
            last=max(rr,key=lambda r:int(r['epoch']))
            same_epoch=[r for r in rr if int(r['epoch'])==98]
            candidates.append(dict(path=str(path.parent.relative_to(ROOT)),matching_partition=matches,
                                   method=last.get('method'),last_epoch=int(last['epoch']),
                                   epoch98_metrics={k:float(same_epoch[0][k]) for k in ['overall_acc','head_acc','tail_acc']} if same_epoch else None))
    payload['historical_exact_matrix_candidates']=candidates
    payload['historical_capt_exact_matrix_candidates']=[r for r in candidates if r['method']=='CAPT']
    (OUT/'analysis.json').write_text(json.dumps(payload,indent=2),encoding='utf8')
    plt.rcParams.update({'font.size':10,'axes.spines.top':False,'axes.spines.right':False})
    fig,axs=plt.subplots(2,2,figsize=(11,7),sharex=True)
    colors=['#087f8c','#cc6633']
    for ax,(metric,title) in zip(axs.flat,[('overall_acc','Overall'),('head_acc','Non-tail (80 classes)'),('tail_acc','Tail (20 classes)'),('hmean','Head-tail harmonic mean')]):
        for p,color in zip(PARTS,colors):
            rs=data[p][2]; hs=data[p][3]
            label='Client-LT, lambda=1' if p==PARTS[0] else 'Fine Dirichlet, beta=0.5'
            ax.step([r['outer_epoch']+1 for r in hs],[r[metric] for r in hs],where='post',color=color,label=label)
            ax.scatter([r['epoch']+1 for r in rs],[r[metric] for r in rs],s=9,color=color)
        ax.set_title(title); ax.set_ylabel('Accuracy (%)'); ax.grid(alpha=.2)
    for ax in axs[-1]: ax.set_xlabel('Outer training round (37 server aggregations in 100 rounds)')
    axs[0,0].legend(fontsize=9)
    fig.suptitle('CAPT cluster with global parameter reset | seed 42 | MAB retained, optimizer retained')
    fig.tight_layout()
    fig.savefig(OUT/'training_curves.png',dpi=180)
    fig.savefig(OUT/'training_curves.pdf')
    plt.close(fig)
    for p in PARTS:
        print(p,json.dumps(data[p][0],ensure_ascii=False))
    print('PAIRED',json.dumps({k:v for k,v in payload.items() if k!='runs'}))


if __name__=='__main__':
    main()
