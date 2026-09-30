"""Read-only audit of packed positive/zero results; no model replay or training.

Derived CSV/JSON files are written to presentation/b_response_review_20260930.
Run from this repository: python scripts/analyze_b_response_results.py
"""
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path
from statistics import mean

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / 'presentation/b_response_review_20260930'
PACK = ROOT / 'output/sfra_b_response_analysis/sfra_b_response'
EPS = 1e-6


def js(path):
    return json.loads(path.read_text(encoding='utf-8'))


def rows(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def save(name, values):
    with (OUT / (name + '.csv')).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in values for k in r)))
        writer.writeheader()
        writer.writerows(values)


def softmax(x):
    e = np.exp(x - np.max(x, axis=-1, keepdims=True))
    return e / e.sum(axis=-1, keepdims=True)


def margins(z, y):
    return np.take_along_axis(z, np.broadcast_to(y[None, :, None], (*z.shape[:2], 1)), -1) - z


def decision(z, y):
    m = margins(z, y).copy()
    m[:, np.arange(len(y)), y] = np.inf
    return m.min(-1)


def main():
    runs = {'A_only': ROOT / 'output/sfra_cp_analysis/sfra_cp/seed42/client-longtail/full-cp/lambda10_mu1_protocol42',
            'old_shared': next((ROOT / 'output/sfra_b_shared_transfer_analysis').rglob('sfra_config.json')).parent}
    for p in (ROOT / 'output/sfra_b_shared_tradeoff_analysis').rglob('sfra_config.json'):
        runs['w' + str(js(p)['b_transfer']['tail_weight'])] = p.parent
    runs['directed'] = next((ROOT / 'output/sfra_b_directed_analysis').rglob('sfra_config.json')).parent
    new_runs = {js(p)['b_transfer']['response_variant']: p.parent for p in PACK.rglob('sfra_config.json')}
    runs.update(new_runs)
    OUT.mkdir(parents=True, exist_ok=True)
    checks = []

    def check(name, value):
        checks.append(dict(check=name, passed=bool(value)))
        if not value:
            save('checks', checks)
            raise ValueError('Audit failed: ' + name)

    def close(name, a, b, atol=2e-6, rtol=2e-5):
        check(name, np.shape(a) == np.shape(b) and np.allclose(a, b, atol=atol, rtol=rtol))

    check('positive and zero present', set(new_runs) == {'positive', 'zero'})
    performance, perclass, curves, provenance, costs = [], [], {}, [], []
    metrics = ['overall_acc', 'head20_acc', 'middle60_acc', 'bottom20_tail_acc', 'non_tail_acc']
    for label, run in runs.items():
        curve = rows(run / 'round_metrics.csv')
        curves[label] = curve
        check(label + ': exact committed rounds 0..100', [int(r['round']) for r in curve] == list(range(101)))
        done = js(run / 'completion.json')
        check(label + ': completed round 100', done['completed_round'] == 100)
        p = dict(run=label)
        for key in metrics:
            p['last20_' + key] = mean(float(r[key]) for r in curve[81:101])
            p['final_' + key] = float(curve[-1][key])
        classes = defaultdict(list)
        counts = js(run / 'class_prior.json')['counts']
        for epoch in range(80, 100):
            cr = rows(run / f'per_class_accuracy_epoch_{epoch}.csv')
            check(f'{label}: epoch {epoch} has all classes', len(cr) == 100 and {int(x['class_id']) for x in cr} == set(range(100)))
            for x in cr:
                classes[int(x['class_id'])].append(float(x['per_class_acc']))
        for c in range(100):
            perclass.append(dict(run=label, class_id=c, train_count=counts[c], last20_acc=mean(classes[c])))
        close(label + ': per-class Tail20 equals round table', p['last20_bottom20_tail_acc'], mean(mean(classes[c]) for c in range(80, 100)), atol=1e-10)
        p['tail_delta_vs_A_pp'] = p['last20_bottom20_tail_acc'] - performance[0]['last20_bottom20_tail_acc'] if performance else 0.
        performance.append(p)
        meta = js(run / 'bridge_metadata.json')
        provenance.append(dict(run=label, path=run.relative_to(ROOT).as_posix(),
            metadata_hashes={k:v for k,v in meta.items() if k.endswith('_sha256')},
            partition_sha256=digest(run/'partition_manifest.csv'), witness_sha256=digest(run/'private_witness_manifest.json')))
        if label != 'A_only':
            base = provenance[0]
            for key in ['initial_lora_sha256', 'frozen_model_sha256', 'schedule_sha256', 'pool_sha256', 'test_sha256']:
                check(label + ': same ' + key, provenance[-1]['metadata_hashes'][key] == base['metadata_hashes'][key])
            for key in ['partition_sha256', 'witness_sha256']:
                check(label + ': same ' + key, provenance[-1][key] == base[key])
            close(label + ': same pre-B test accuracies', [[float(x[k]) for k in metrics] for x in curve[:30]],
                  [[float(x[k]) for k in metrics] for x in curves['A_only'][:30]], atol=1e-9, rtol=0)
            cfg, basecfg = js(run/'sfra_config.json'), js(runs['A_only']/'sfra_config.json')
            check(label + ': same A and base training config', {k:v for k,v in cfg.items() if k != 'b_transfer'} == {k:v for k,v in basecfg.items() if k != 'b_transfer'})
            ev = rows(run/'b_transfer_rounds.csv')
            cost = dict(run=label, events=len(ev), total_elapsed_seconds=done['elapsed_seconds'])
            for key in ['seconds', 'algorithm_forward_images', 'algorithm_backward_images', 'diagnostic_forward_images', 'optimizer_steps', 'extra_downlink_bytes', 'extra_upload_bytes']:
                cost[key] = sum(float(x.get(key) or 0) for x in ev)
            costs.append(cost)
    pc = js(new_runs['positive']/'b_transfer_config.json')
    zc = js(new_runs['zero']/'b_transfer_config.json')
    check('P/Z transfer config differs only variant', {k:v for k,v in pc.items() if k != 'response_variant'} == {k:v for k,v in zc.items() if k != 'response_variant'})
    close('P/Z packed performance matches recomputation',
          [float(x['last20_bottom20_tail_acc']) for x in sorted(rows(PACK/'analysis/performance.csv'), key=lambda x:x['response_variant'])],
          [next(x['last20_bottom20_tail_acc'] for x in performance if x['run']==v) for v in ['positive','zero']], atol=1e-10)

    events, class_changes, traces, sources_all, score_stats, logit_gradients = [], [], [], [], [], []
    event_cache = {}
    for variant, run in new_runs.items():
        cfg = js(run/'b_transfer_config.json')
        manifest = js(run/'b_transfer_manifest.json')
        target = manifest['target_clients']
        totals = {int(c):n for c,n in manifest['target_class_totals'].items()}
        partition = rows(run/'partition_manifest.csv')
        counts = Counter((int(x['client_id']), int(x['class_id'])) for x in partition)
        labels_by_position = {(int(x['client_id']), int(x['local_position'])):int(x['class_id']) for x in partition}
        clients = sorted({k for k,p in labels_by_position})
        eligible = {(c,j) for c in totals for j in clients if j not in target and counts[j,c]==0}
        prior = np.asarray(js(run/'class_prior.json')['prior'], dtype=np.float64)
        tau = js(run/'control_config.json')['la_tau']
        check(variant+': protocol full target pool', target==[27,28,29] and sum(totals.values())==119 and len(totals)==20 and all(totals[c]==sum(counts[k,c] for k in target) for c in totals))
        check(variant+': 8 events and 16 C steps', all(js(run/'completion.json')[k]==n for k,n in [('b_transfer_events',8),('b_transfer_optimizer_steps',16)]))
        for rnd in cfg['rounds']:
            tag = f'{variant} r{rnd}'
            folder = run/f'b_transfer_rounds/r{rnd:03d}'
            cal, summary = js(folder/'calibration_manifest.json'), js(folder/'summary.json')
            scores, sources = rows(folder/'donor_scores.csv'), rows(folder/'source_weights.csv')
            scored = {(int(x['class_id']),int(x['donor'])):x for x in scores}
            selected = {(int(x['class_id']),int(x['donor'])):float(x['weight']) for x in sources}
            check(tag+': complete eligible candidates', set(scored)==eligible and len(scores)==len(eligible))
            expected = {}
            for c in totals:
                ranked = sorted([(j,float(x['gain'])) for (cc,j),x in scored.items() if cc==c and float(x['gain'])>cfg['min_gain']], key=lambda x:(-x[1],x[0]))[:cfg['donors_per_class']]
                denominator = sum(g for j,g in ranked)
                expected.update({(c,j):g/denominator for j,g in ranked})
            check(tag+': exact top3 and normalized weights', len(selected)==len(sources) and selected.keys()==expected.keys() and all(abs(selected[k]-expected[k])<1e-12 for k in expected))
            check(tag+': source qualification and flags', all(j not in target and counts[j,c]==0 and int(scored[c,j]['donor_class_count'])==0 for c,j in selected) and all((x['selected'].lower()=='true')==((c,j) in selected) and abs(float(x['source_weight'])-selected.get((c,j),0.))<1e-12 and abs(float(x['gain'])-min(float(x['gain_view0']),float(x['gain_view1'])))<1e-12 for (c,j),x in scored.items()))
            mixes = {(int(c),int(j)):w for c,dd in cal['source_mixtures'].items() for j,w in dd.items()}
            check(tag+': teacher class relations in manifest', mixes==selected)
            donors = cal['shared_donor_ids']
            check(tag+': selected union and source counts', donors==sorted({j for c,j in selected}) and summary['unique_donors']==len(donors) and summary['donor_links']==len(selected) and summary['candidate_pairs']==len(scores))
            with np.load(folder/'matrices.npz') as archive:
                close(tag+': NPZ class weights', archive['source_weights'], [[selected.get((c,j),0.) for j in donors] for c in sorted(totals)], atol=1e-12, rtol=0)
                check(tag+': NPZ axes', archive['donor_ids'].tolist()==donors and archive['class_ids'].tolist()==sorted(totals))
                matrices = [archive[f'module_{i:02d}'].astype(np.float64) for i in range(len(cal['module_order']))]
                check(tag+': independent finite donor C', all(a.shape==(len(donors),4,4) and np.isfinite(a).all() for a in matrices))
                penalty = cfg['regularization']*sum(float((a*a).sum()) for a in matrices)/(len(donors)*len(matrices))
            feedback = rows(folder/'client_feedback_steps.csv')
            check(tag+': identical full-pool two-view budget', len(feedback)==6 and {(int(x['step']),int(x['client_id'])) for x in feedback}=={(s,k) for s in [1,2] for k in target} and all(int(x['tail_samples'])==sum(counts[int(x['client_id']),c] for c in totals) and int(x['non_tail_samples'])==0 and int(x['views'])==2 for x in feedback) and summary['algorithm_backward_images']==476 and summary['optimizer_steps']==2)
            record = {(x['phase'],int(x['class_id'])):x for x in rows(folder/'response_metrics.csv')}
            sums = {c:defaultdict(float) for c in totals}
            gradient_parts = {phase:{kind:[] for kind in ['la','response']} for phase in ['baseline','student']}
            baseline_class_margins, donor_class_margins = defaultdict(list), defaultdict(list)
            with np.load(folder/'response_targets.npz') as ar:
                all_baseline, all_student, all_y, all_increments = [], [], [], []
                for k in target:
                    prefix = f'client_{k:03d}_'
                    y, positions = ar[prefix+'labels'], ar[prefix+'positions']
                    check(tag+f': client{k} full original tail positions', set(positions)=={p for (i,p),c in labels_by_position.items() if i==k and c in totals} and len(set(positions))==len(positions) and all(labels_by_position[k,int(p)]==int(c) for p,c in zip(positions,y)))
                    base, student = ar[prefix+'baseline_logits'], ar[prefix+'student_logits']
                    bm, sm = margins(base,y), margins(student,y)
                    inc = np.zeros_like(bm[0],dtype=np.float64)
                    for c in np.unique(y):
                        mask = y==c
                        response = np.zeros_like(bm[:,mask],dtype=np.float64)
                        baseline_class_margins[int(c)].append(decision(base[:,mask],y[mask]).astype(np.float64))
                        for (cc,j),weight in selected.items():
                            if cc!=c:
                                continue
                            teacher = ar[prefix+f'donor_{j:03d}_logits'][:,mask]
                            response += weight*(margins(teacher,y[mask])-bm[:,mask]).astype(np.float64)
                            donor_class_margins[int(c),j].append(decision(teacher,y[mask]).astype(np.float64))
                        inc[mask] = np.maximum(0,response.min(0))
                    inc[np.arange(len(y)),y]=0
                    measured = ar[prefix+'positive_increment'].astype(np.float64)
                    applied = ar[prefix+'applied_increment']
                    target_m = ar[prefix+'target_margins']
                    weights = ar[prefix+'competitor_weights']
                    close(tag+f': client{k} mixed-then-min-clipped response',measured,inc,atol=1e-8,rtol=1e-5)
                    close(tag+f': client{k} variant application',applied,measured if variant=='positive' else np.zeros_like(measured),atol=0,rtol=0)
                    close(tag+f': client{k} baseline margin',ar[prefix+'baseline_margins'],bm,atol=0,rtol=0)
                    close(tag+f': client{k} target baseline plus increment',target_m,bm+applied[None],atol=0,rtol=0)
                    competitor = base.astype(np.float64)
                    competitor[:,np.arange(len(y)),y]=-np.inf
                    close(tag+f': client{k} frozen competitor weights',weights,softmax(competitor),atol=1e-7,rtol=1e-5)
                    positive = measured>0
                    gain = (sm-bm).min(0)
                    attained = positive & (gain>=measured-1e-6)
                    # Frozen saved-logit derivatives only: these are not C-space gradients.
                    for phase,z in [('baseline',base),('student',student)]:
                        zm=margins(z,y)
                        adjusted=z.astype(np.float64)+tau*np.log(prior)[None,None,:]
                        probs=softmax(adjusted)
                        la=-np.log(np.take_along_axis(probs,np.broadcast_to(y[None,:,None],(2,len(y),1)),-1)[:,:,0])
                        resp=(weights*np.maximum(target_m-zm,0)**2).sum(-1)
                        ideal_terms=weights*np.maximum((bm+measured[None])-zm,0)**2
                        ideal=ideal_terms.sum(-1)
                        ideal_positive=(ideal_terms*positive[None]).sum(-1)
                        ideal_zero=(ideal_terms*(~positive[None])).sum(-1)
                        g_la=probs.copy();g_la[:,np.arange(len(y)),y]-=1
                        g_resp=2*weights*np.maximum(target_m-zm,0)
                        g_resp[:,np.arange(len(y)),y]=-g_resp.sum(-1)
                        sample_weights=np.array([1/(2*len(totals)*totals[int(c)]) for c in y])[None,:,None]
                        gradient_parts[phase]['la'].append((g_la*sample_weights).reshape(-1))
                        gradient_parts[phase]['response'].append((g_resp*sample_weights).reshape(-1))
                        for c in np.unique(y):
                            mask=y==c;s=sums[int(c)]
                            s[phase+'_la_sum']+=float(la[:,mask].mean(0).sum())
                            s[phase+'_response_sum']+=float(resp[:,mask].astype(np.float64).mean(0).sum())
                            s[phase+'_ideal_response_sum']+=float(ideal[:,mask].astype(np.float64).mean(0).sum())
                            s[phase+'_ideal_positive_sum']+=float(ideal_positive[:,mask].astype(np.float64).mean(0).sum())
                            s[phase+'_ideal_zero_sum']+=float(ideal_zero[:,mask].astype(np.float64).mean(0).sum())
                            s[phase+'_correct0']+=int((z[0,mask].argmax(-1)==y[mask]).sum())
                            s[phase+'_correct_two_views']+=int((z[:,mask].argmax(-1)==y[None,mask]).sum())
                    for c in np.unique(y):
                        mask=y==c;s=sums[int(c)]
                        s['samples']+=int(mask.sum());s['positive_pairs']+=int(positive[mask].sum())
                        s['attained_pairs']+=int(attained[mask].sum())
                        s['positive_gain_sum']+=float(gain[mask][positive[mask]].astype(np.float64).sum())
                        s['negative_gain_pairs']+=int((positive[mask] & (gain[mask]<-1e-6)).sum())
                    all_baseline.append(base);all_student.append(student);all_y.append(y);all_increments.append(measured)
                baseline_array=np.concatenate(all_baseline,axis=1);student_array=np.concatenate(all_student,axis=1)
                event_cache[variant,rnd]=dict(donors=donors,selected=selected,matrices=matrices,baseline=baseline_array,
                    student=student_array,labels=np.concatenate(all_y),increment=np.concatenate(all_increments))
            for (c,j),weight in selected.items():
                b=np.concatenate(baseline_class_margins[c],axis=1).mean(1)
                d=np.concatenate(donor_class_margins[c,j],axis=1).mean(1)
                close(tag+f': selected {c}/{j} probe from logits',[float(scored[c,j]['gain_view0']),float(scored[c,j]['gain_view1'])],d-b,atol=1e-10,rtol=0)
            current=[]
            for c,s in sums.items():
                n=totals[c];check(tag+f': class{c} sample count',s['samples']==n)
                for phase,name in [('baseline','ordinary_global_B'),('student','transferred_global_B')]:
                    rec=record[name,c]
                    close(tag+f': class{c} {phase} LA reconstruction',s[phase+'_la_sum']/n,float(rec['la']))
                    close(tag+f': class{c} {phase} response reconstruction',s[phase+'_response_sum']/n,float(rec['response_loss']),atol=1e-9)
                check(tag+f': class{c} positive/attainment counts',s['positive_pairs']==int(record['transferred_global_B',c]['positive_pairs']) and s['attained_pairs']==int(record['transferred_global_B',c]['attained_positive_pairs']))
                row=dict(variant=variant,round=rnd,class_id=c,samples=n,
                    la_gain=(s['baseline_la_sum']-s['student_la_sum'])/n,
                    accuracy_delta_view0_pp=100*(s['student_correct0']-s['baseline_correct0'])/n,
                    accuracy_delta_two_views_pp=50*(s['student_correct_two_views']-s['baseline_correct_two_views'])/n,
                    response_before=s['baseline_response_sum']/n,response_after=s['student_response_sum']/n,
                    ideal_response_before=s['baseline_ideal_response_sum']/n,ideal_response_after=s['student_ideal_response_sum']/n,
                    ideal_positive_loss_after=s['student_ideal_positive_sum']/n,ideal_zero_loss_after=s['student_ideal_zero_sum']/n,
                    positive_pairs=int(s['positive_pairs']),attained_pairs=int(s['attained_pairs']),
                    negative_gain_pairs=int(s['negative_gain_pairs']))
                current.append(row);class_changes.append(row)
            for phase,g in gradient_parts.items():
                a,b=np.concatenate(g['la']),np.concatenate(g['response']);na,nb=np.linalg.norm(a),np.linalg.norm(b)
                logit_gradients.append(dict(variant=variant,round=rnd,phase=phase,
                    LA_logit_gradient_norm=float(na),response_logit_gradient_norm=float(nb),
                    response_to_LA_norm=float(nb/na),cosine=float(a@b/(na*nb)) if nb else None,
                    scope='weighted saved-logit gradient; no C Jacobian'))
            trace=rows(folder/'c_loss_trace.csv')
            check(tag+': trace contains C0/C1/C2', [int(x['c_step']) for x in trace]==[0,1,2])
            for x in trace:
                close(tag+f': step{x["c_step"]} objective components',float(x['fixed_pool_objective']),float(x['fixed_pool_la'])+cfg['response_weight']*float(x['response_loss'])+float(x['regularization_penalty']),atol=1e-10)
                traces.append(dict(variant=variant,**x))
            close(tag+': saved C reconstructs regularizer',penalty,float(trace[-1]['regularization_penalty']))
            close(tag+': endpoint LA matches logits',float(trace[-1]['fixed_pool_la']),mean(s['student_la_sum']/totals[c] for c,s in sums.items()))
            close(tag+': endpoint response matches logits',float(trace[-1]['response_loss']),mean(x['response_after'] for x in current),atol=1e-9)
            totalpairs=sum(x['positive_pairs'] for x in current)
            check(tag+': measured responses counted before ablation',totalpairs==summary['positive_response_pairs'] and not summary['skipped_no_positive_response'])
            ev=dict(variant=variant,round=rnd,candidate_pairs=len(scores),positive_donor_pairs=sum(float(x['gain'])>cfg['min_gain'] for x in scores),
                selected_edges=len(selected),donors=len(donors),C_parameters=summary['learned_c_parameters'],
                response_pairs=totalpairs,attained_pairs=sum(x['attained_pairs'] for x in current),
                attainment=sum(x['attained_pairs'] for x in current)/totalpairs,
                negative_gain_pairs=sum(x['negative_gain_pairs'] for x in current),
                la_gain=mean(x['la_gain'] for x in current),la_improved_classes=sum(x['la_gain']>EPS for x in current),
                la_harmed_classes=sum(x['la_gain']<-EPS for x in current),
                accuracy_improved_classes=sum(x['accuracy_delta_view0_pp']>EPS for x in current),
                accuracy_harmed_classes=sum(x['accuracy_delta_view0_pp']<-EPS for x in current),
                response_before=float(trace[0]['response_loss']),response_after=float(trace[2]['response_loss']),
                ideal_response_before=mean(x['ideal_response_before'] for x in current),ideal_response_after=mean(x['ideal_response_after'] for x in current),
                ideal_positive_loss_after=mean(x['ideal_positive_loss_after'] for x in current),ideal_zero_loss_after=mean(x['ideal_zero_loss_after'] for x in current),
                objective_before=float(trace[0]['fixed_pool_objective']),objective_after=float(trace[2]['fixed_pool_objective']),
                transfer_norm=summary['effective_global_transfer_norm'],
                transfer_to_ordinary_norm=summary['effective_global_transfer_norm']/summary['ordinary_B_effective_update_norm'])
            events.append(ev)
            sources_all.extend(dict(variant=variant,**x) for x in sources)
            gains=[float(scored[k]['gain']) for k in selected]
            score_stats.append(dict(variant=variant,round=rnd,min_selected_gain=min(gains),median_selected_gain=float(np.median(gains)),max_selected_gain=max(gains)))
    comparisons=[]
    for rnd in range(30,101,10):
        p,z=event_cache['positive',rnd],event_cache['zero',rnd]
        common=sorted(set(p['donors']) & set(z['donors']))
        a=np.concatenate([m[p['donors'].index(j)].ravel() for m in p['matrices'] for j in common])
        b=np.concatenate([m[z['donors'].index(j)].ravel() for m in z['matrices'] for j in common])
        u=(p['student']-p['baseline']).astype(np.float64).ravel();v=(z['student']-z['baseline']).astype(np.float64).ravel()
        comparisons.append(dict(round=rnd,source_edge_overlap=len(p['selected'].keys() & z['selected'].keys()),
            positive_edges=len(p['selected']),zero_edges=len(z['selected']),same_union=p['donors']==z['donors'],
            C_common_donor_cosine=float(a@b/(np.linalg.norm(a)*np.linalg.norm(b))),
            C_common_donor_relative_difference=float(np.linalg.norm(a-b)/np.linalg.norm(b)),
            baseline_max_logit_difference=float(np.max(np.abs(p['baseline']-z['baseline']))),
            local_logit_change_cosine=float(u@v/(np.linalg.norm(u)*np.linalg.norm(v))),
            local_logit_change_relative_difference=float(np.linalg.norm(u-v)/np.linalg.norm(v))))
    lookup={(x['run'],x['class_id']):x['last20_acc'] for x in perclass}
    tail=[]
    for c in range(80,100):
        r=dict(class_id=c,**{v:lookup[v,c] for v in ['A_only','old_shared','w0.35','directed','positive','zero']})
        r.update(positive_minus_A_pp=r['positive']-r['A_only'],zero_minus_A_pp=r['zero']-r['A_only'],positive_minus_zero_pp=r['positive']-r['zero'])
        tail.append(r)
    round_diffs=[dict(round=i,positive=float(curves['positive'][i]['bottom20_tail_acc']),zero=float(curves['zero'][i]['bottom20_tail_acc']),
        A_only=float(curves['A_only'][i]['bottom20_tail_acc']),positive_minus_zero_pp=float(curves['positive'][i]['bottom20_tail_acc'])-float(curves['zero'][i]['bottom20_tail_acc'])) for i in range(101)]
    aggregate={}
    for v in ['positive','zero']:
        ev=[x for x in events if x['variant']==v];cc=[x for x in class_changes if x['variant']==v]
        aggregate[v]=dict(events=len(ev),class_events=len(cc),
            LA_improved_class_events=sum(x['la_gain']>EPS for x in cc),LA_harmed_class_events=sum(x['la_gain']<-EPS for x in cc),
            accuracy_improved_class_events=sum(x['accuracy_delta_view0_pp']>EPS for x in cc),accuracy_harmed_class_events=sum(x['accuracy_delta_view0_pp']<-EPS for x in cc),
            mean_LA_gain=mean(x['la_gain'] for x in cc),pooled_attainment=sum(x['attained_pairs'] for x in ev)/sum(x['response_pairs'] for x in ev),
            mean_response_before=mean(x['response_before'] for x in ev),mean_response_after=mean(x['response_after'] for x in ev),
            mean_ideal_response_after=mean(x['ideal_response_after'] for x in ev),
            mean_ideal_positive_loss_after=mean(x['ideal_positive_loss_after'] for x in ev),
            mean_ideal_zero_loss_after=mean(x['ideal_zero_loss_after'] for x in ev),
            pooled_positive_pairs_with_negative_gain=sum(x['negative_gain_pairs'] for x in ev)/sum(x['response_pairs'] for x in ev),
            response_worsened_events=sum(x['response_after']>x['response_before']+1e-10 for x in ev),
            objective_improved_events=sum(x['objective_after']<x['objective_before'] for x in ev),
            tail_test_improved_classes=sum(x[v]>x['A_only']+EPS for x in tail),tail_test_harmed_classes=sum(x[v]<x['A_only']-EPS for x in tail),
            checkpoint_files=len(list(new_runs[v].rglob('*.pt'))))
    audit=dict(primary_endpoint='Tail20 committed rounds81..100 mean; seed42',performance=performance,aggregate=aggregate,
        provenance=provenance,checks_passed=len(checks),costs=costs,limitations=[
        'Single training seed; repeated rounds and classes are not independent training replications.',
        'Same training pool used for screening, optimization and response diagnostics.',
        'No raw donor updates or commit.pt in archive; C-space gradient and exact residual reconstruction unavailable.',
        'Selected teachers saved; unselected teacher scores can only be checked for eligibility and internal ranking.',
        'Historical versions change multiple design choices; they are performance references, not single-factor ablations.',
        'P/Z trajectories differ after first transfer; later source/teacher comparisons do not fix model state.'])
    for name,values in [('performance',performance),('per_class',perclass),('tail_class_comparison',tail),('events',events),
        ('class_changes',class_changes),('objective_trace',traces),('source_weights',sources_all),('selected_gain_stats',score_stats),
        ('logit_gradients',logit_gradients),('positive_zero_comparison',comparisons),('tail_curves',round_diffs),('costs',costs),('checks',checks)]:
        save(name,values)
    (OUT/'audit.json').write_text(json.dumps(audit,ensure_ascii=False,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(dict(checks_passed=len(checks),aggregate=aggregate,comparison_r30=comparisons[0]),indent=2))
    for p in performance:
        print(p['run'], 'Tail20',round(p['last20_bottom20_tail_acc'],4),'final',p['final_bottom20_tail_acc'])
    print('Output:',OUT.relative_to(ROOT))


if __name__=='__main__':
    main()
