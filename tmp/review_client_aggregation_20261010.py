"""Read-only source audit and independent derived analysis of seed42 aggregation runs."""
import csv
import json
import sys
from pathlib import Path
from statistics import mean, median

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
from scripts.run_cliplora_joint_aggregation import make_job
from tools.client_aggregation.analysis import audit_run, metrics_from_prediction, sample_dynamics, stage_comparisons, METRICS
from tools.client_aggregation.protocol import PAIR_HASHES, code_hashes, partition_signature, read_csv

ROOT = REPO / 'output/client_aggregation_v2_parallel4_results/client_aggregation_v2_parallel4'
OUT = REPO / 'tmp/client_aggregation_review_20261010'
OUT.mkdir(exist_ok=True)


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def save_json(name, value):
    (OUT / name).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding='utf-8')


def save_csv(name, rows):
    rows = list(rows)
    with (OUT / name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


plan = read(ROOT / 'experiment_plan.json')
results = {}
for arm in ('frozen', 'ab'):
    job = make_job(plan, arm)
    results[arm] = audit_run(ROOT / job['run'], job)
    print('RAW_AUDIT_PASSED', arm, flush=True)

a, b = results['ab'], results['frozen']
hashes = {k: a['metadata'][k] == b['metadata'][k] for k in PAIR_HASHES}
initial_equal = all(np.array_equal(a['predictions'][0][k], b['predictions'][0][k])
                    for k in ('sample_id', 'class_id', 'prediction', 'correct', 'logit_margin'))
assert all(hashes.values()) and initial_equal
intentional = {'candidate_rounds', 'extra_trainable_factor', 'extra_steps_expected'}
base_a, base_b = a['config']['base_training_config'], b['config']['base_training_config']
base_diffs = {k: [base_a.get(k), base_b.get(k)] for k in set(base_a) | set(base_b)
              if base_a.get(k) != base_b.get(k)}
assert set(base_diffs) <= intentional
audit = dict(source=str(ROOT), independent_seeds=1, seed=42, completed_rounds={'frozen':100, 'ab':100},
             recorded_weight_and_client_phase_count=sum(len(r['budgets'])//30 for r in results.values()),
             local_budget_rows=sum(len(r['budgets']) for r in results.values()),
             official_prediction_files=sum(len(r['predictions']) for r in results.values()),
             stage_prediction_files=sum(len(r['stages']) for r in results.values()),
             paired_hash_checks=hashes, initial_predictions_and_margins_identical=initial_equal,
             base_config_differences=base_diffs, local_source_matches_registered=code_hashes(REPO)==plan['code_sha256'],
             budget_matched=False, independent_component_effect_identified=False,
             effective_update_tensor_recomputed=False,
             note='Model checkpoint tensors are not in the archive; event audits validate recorded reconstruction and frozen-factor checks.')

performance, temporal, cohorts, classes, all_stages, dynamics = [], [], [], [], [], []
for arm, r in results.items():
    pred = r['predictions']
    curve = {i: metrics_from_prediction(p, r['groups']) for i, p in pred.items()}
    perf = dict(arm=arm, **{m:mean(curve[i][m] for i in range(81,101)) for m in METRICS})
    performance.append(perf)
    for i in (0, 1, 10, 20, 29, 30, 50, 70, 80, 90, 100):
        temporal.append(dict(arm=arm, round=i, **curve[i]))
    for anchor in (0, 50, 80):
        reference = pred[anchor]['correct']
        for group, ids in r['groups'].items():
            mask = np.isin(pred[0]['class_id'], ids)
            retained = mean(int((mask & reference & pred[i]['correct']).sum()) for i in range(81,101))
            acquired = mean(int((mask & ~reference & pred[i]['correct']).sum()) for i in range(81,101))
            cohorts.append(dict(arm=arm, anchor=anchor, group=group, total=int(mask.sum()),
                                reference_correct=int((mask & reference).sum()), retained=retained,
                                acquired=acquired, lost=int((mask & reference).sum())-retained))
    for c in range(100):
        classes.append(dict(arm=arm, class_id=c, last20=mean(float(pred[i]['correct'][pred[i]['class_id']==c].sum())
                                                           for i in range(81,101))))
    all_stages.extend(stage_comparisons(r))
    dynamics.extend(sample_dynamics(r))

perf_by = {r['arm']:r for r in performance}
audit['ab_minus_frozen'] = {m:perf_by['ab'][m]-perf_by['frozen'][m] for m in METRICS}
saved_perf = {r['arm']:r for r in read_csv(ROOT/'analysis/performance.csv')}
assert all(abs(p[m]-float(saved_perf[p['arm']]['last20_'+m]))<1e-9 for p in performance for m in METRICS)
audit['saved_performance_matches_raw_predictions'] = True

saved_stages = {(r['arm'],int(r['round']),r['operation'],r['group']):r for r in read_csv(ROOT/'analysis/stage_effects.csv')}
assert len(saved_stages)==len(all_stages)
for r in all_stages:
    old = saved_stages[(r['arm'],r['round'],r['operation'],r['group'])]
    for k in ('before_accuracy','after_accuracy','delta_accuracy','corrected_examples','damaged_examples','delta_mean_logit_margin'):
        assert abs(float(r[k])-float(old[k]))<1e-8, (r,k)
audit['saved_stage_changes_match_raw_predictions'] = True

stage_summary=[]
for arm, operation, group in sorted({(r['arm'],r['operation'],r['group']) for r in all_stages}):
    rows=[r for r in all_stages if (r['arm'],r['operation'],r['group'])==(arm,operation,group)]
    stage_summary.append(dict(arm=arm, operation=operation, group=group, events=len(rows),
        mean_pp=mean(r['delta_accuracy'] for r in rows), positive=sum(r['delta_accuracy']>1e-8 for r in rows),
        negative=sum(r['delta_accuracy']<-1e-8 for r in rows),
        corrected_examples=mean(r['corrected_examples'] for r in rows),damaged_examples=mean(r['damaged_examples'] for r in rows)))
contrasts=[]
for g in METRICS:
    by={r['arm']:r for r in cohorts if r['anchor']==0 and r['group']==g}
    contrasts.append(dict(group=g, extra_acquired=by['ab']['acquired']-by['frozen']['acquired'],
        extra_lost=by['ab']['lost']-by['frozen']['lost'], net_correct=by['ab']['retained']+by['ab']['acquired']-by['frozen']['retained']-by['frozen']['acquired']))
class_diffs=[]
for c in range(100):
    by={r['arm']:r['last20'] for r in classes if r['class_id']==c}
    class_diffs.append(dict(class_id=c, frozen=by['frozen'], ab=by['ab'], difference=by['ab']-by['frozen']))
audit['class_change_counts']={g:dict(improved=sum(r['difference']>1e-9 for r in class_diffs if r['class_id'] in ids),
                                    worsened=sum(r['difference']<-1e-9 for r in class_diffs if r['class_id'] in ids),
                                    unchanged=sum(abs(r['difference'])<=1e-9 for r in class_diffs if r['class_id'] in ids))
                              for g,ids in a['groups'].items()}
for filename, rows in [('performance.csv',performance),('temporal.csv',temporal),('retention_cohorts.csv',cohorts),
                      ('cohort_contrasts.csv',contrasts),('class_differences.csv',class_diffs),
                      ('stage_by_round.csv',all_stages),('stage_summary.csv',stage_summary),('sample_dynamics.csv',dynamics)]:
    save_csv(filename,rows)

# Historical references: match metadata and local budgets; do not silently call E2 a budget-matched baseline.
old_root=REPO/'output/ab_decision_42_0_3407_analysis/ab_decision_42_0_3407/runs'
old_ab=next((old_root/'ab/seed42').rglob('bridge_metadata.json')).parent
old_a=next((old_root/'a/seed42').rglob('bridge_metadata.json')).parent
old_s=next((old_root/'s/seed42').rglob('bridge_metadata.json')).parent
old_e2=REPO/'output/la_control_analysis/la_control/seed42/client-longtail/e2/tau1_a1_protocol42'
tailrw=REPO/'output/method_a_simple_controls_v1_probe_fix_results/method_a_simple_controls_v1_probe_fix/runs/seed42/tailrw-g16'
history=[]
history_audit={}
for name, run in [('old_AB',old_ab),('old_A',old_a),('old_S',old_s),('old_E2_extraB',old_e2),('old_tailrw_g16',tailrw)]:
    if not run.is_dir():
        continue
    rows=read_csv(run/'round_metrics.csv')
    assert sorted(int(r['round']) for r in rows)==list(range(101))
    last=[r for r in rows if 81<=int(r['round'])<=100]
    class_means={}
    for rnd in range(81,101):
        pc=read_csv(run/f'per_class_accuracy_epoch_{rnd-1}.csv')
        perclass={int(r['class_id']):float(r['per_class_acc']) for r in pc}
        for metric,ids in a['groups'].items():
            class_means.setdefault(metric,[]).append(mean(perclass[c] for c in ids))
    hist=dict(name=name,**{m:mean(vals) for m,vals in class_means.items()})
    for m in METRICS:
        if m in last[0]:
            assert abs(hist[m]-mean(float(r[m]) for r in last))<1e-8
    history.append(hist)
    meta=read(run/'bridge_metadata.json')
    current=a if name!='old_E2_extraB' else b
    new_cfg=current['config']['base_training_config']; old_cfg=read(run/'control_config.json')
    env_keys=('python','torch','cuda','cudnn','gpu')
    item=dict(path=str(run), hashes={k:meta.get(k)==current['metadata'].get(k) for k in PAIR_HASHES},
        partition_matches=partition_signature(read_csv(run/'partition_manifest.csv'))==partition_signature(read_csv(current['run']/'partition_manifest.csv')),
        environment_old=meta.get('environment'),environment_new=current['metadata'].get('environment'),
        same_recorded_environment=all(meta.get('environment',{}).get(k)==current['metadata'].get('environment',{}).get(k) for k in env_keys),
        control_config_differences={k:[old_cfg.get(k),new_cfg.get(k)] for k in set(old_cfg)|set(new_cfg) if old_cfg.get(k)!=new_cfg.get(k)},
        budget_records=len(read_csv(run/'budget.csv')) if (run/'budget.csv').exists() else None)
    if name=='old_AB':
        old_sf=read(run/'sfra_config.json'); new_sf=current['config']
        item['sfra_top_level_difference_keys']=sorted(k for k in set(old_sf)|set(new_sf) if old_sf.get(k)!=new_sf.get(k))
        item['note']='Same data, initialization, schedule, training budget and recorded environment. New run also changes execution from serial to four concurrent clients and adds observational stage tests. Pilot numerical equivalence is not full-trajectory equivalence.'
    if name=='old_E2_extraB':
        item['note']='E2 freezes A but adds 9 extra B epochs (3168 optimizer steps); new frozen arm has no extra B. Not an aggregation-only comparison.'
    history_audit[name]=item
save_csv('historical_performance.csv',history)
save_json('historical_audit.json',history_audit)

weights=read_csv(ROOT/'client_weights.csv'); proxies=read_csv(ROOT/'class_support_proxy.csv')
weight_summary=dict(tail_clients=[27,28,29],
    tail_client_old_mass=sum(float(r['sample_weight']) for r in weights if int(r['client_id'])>=27),
    tail_client_new_mass=sum(float(r['weight']) for r in weights if int(r['client_id'])>=27),
    near_zero_clients=[int(r['client_id']) for r in weights if float(r['weight'])<1e-8],
    proxy_masses={g:{k:sum(float(r[k]) for r in proxies if int(r['class_id']) in ids) for k in ('sample_mixture','joint_mixture','target')}
                  for g,ids in a['groups'].items() if g in ('head20_acc','middle60_acc','bottom20_tail_acc')})
save_json('weights_summary.json',weight_summary)

corrections=read_csv(a['run']/'sfra_rounds.csv')
corr_summary=[]
for lo,hi in ((1,10),(11,30),(31,60),(61,90),(1,90)):
    rows=[r for r in corrections if lo<=int(r['round'])<=hi]
    ratios=[float(r['committed_effective_norm'])/float(r['proposal_effective_norm']) for r in rows if float(r['proposal_effective_norm'])>0]
    corr_summary.append(dict(start=lo,end=hi,mean_effective_norm_ratio=mean(ratios),median_effective_norm_ratio=median(ratios),
       mean_proposal_commit_cosine=mean(float(r['proposal_committed_cosine']) for r in rows),
       functional_loss_improved=sum(float(r['committed_functional_loss'])<float(r['proposal_functional_loss']) for r in rows),
       classification_loss_increased=sum(float(r['committed_classification_loss_increase'])>0 for r in rows)))
save_csv('correction_diagnostics.csv',corr_summary)
pilots=[]
for arm in ('frozen','ab'):
    for path in sorted((ROOT/'smoke/seed42'/arm/'parallel_benchmark').glob('factor_*.json')):
        bench=read(path)
        assert bench['passed'] and len(bench['comparisons'])==6 and bench['short_group']['clients']==2
        assert all(r['close'] for r in bench['comparisons']+bench['short_comparisons'])
        pilots.append(dict(arm=arm,factor=bench['factor'],serial_seconds=bench['serial']['seconds'],
                           parallel_seconds=bench['parallel']['seconds'],
                           speedup=bench['serial']['seconds']/bench['parallel']['seconds'],
                           max_parameter_error=max(r['max_abs'] for r in bench['comparisons']+bench['short_comparisons']),
                           short_group_clients=2,passed=True))
save_csv('parallel_pilot.csv',pilots)
costs=[]
for arm,r in results.items():
    done=r['progress']
    official=sum(float(x['seconds']) for x in read_csv(r['run']/'evaluation_budget.csv'))
    peak=max(read(p)['process_peak_allocated_bytes'] for p in (r['run']/'parallel_execution').glob('*.json'))
    costs.append(dict(arm=arm,elapsed_hours=done['elapsed_seconds']/3600,
        excluding_official_and_stage_hours=(done['elapsed_seconds']-official-done['stage_diagnostic_seconds'])/3600,
        normal_steps=done['normal_optimizer_steps'],extra_A_steps=done['extra_optimizer_steps'],
        correction_steps=done['functional_correction_steps'],transfer_steps=done.get('b_transfer_optimizer_steps',0),
        training_images=sum(int(x['sample_presentations']) for x in r['budgets']),peak_allocated_GiB=peak/1024**3))
save_csv('costs.csv',costs)
transfers=read_csv(a['run']/'b_transfer_rounds.csv')
transfer_summary=dict(events=len(transfers),unique_donors_per_event=[int(r['unique_donors']) for r in transfers],
    recipients_per_event=[int(r['recipients']) for r in transfers],
    transfer_to_ordinary_effective_norm_ratios=[float(r['effective_global_transfer_norm'])/float(r['ordinary_B_effective_update_norm']) for r in transfers])
save_json('transfer_diagnostics.json',transfer_summary)
save_json('audit.json',audit)

# An exported comparison figure includes the historical execution difference in its caption.
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
fig,axes=plt.subplots(2,2,figsize=(11,7.3),sharex=True)
old_curve=read_csv(old_ab/'round_metrics.csv')
for ax,metric,title in zip(axes.flat,METRICS[:4],('Overall','Head20','Middle60','Tail20')):
    for arm,label,color in [('frozen','New aggregation, frozen A','#2b74a8'),('ab','New aggregation, full A+B','#d45f22')]:
        rows=results[arm]['curves']
        ax.plot([int(x['round']) for x in rows],[float(x[metric]) for x in rows],label=label,color=color,lw=1.8)
    ax.plot([int(x['round']) for x in old_curve],[float(x[metric]) for x in old_curve],
            label='Historical FedAvg, full A+B',color='#5b4b8a',ls='--',lw=1.8)
    ax.axvspan(81,100,color='grey',alpha=.10)
    ax.set(title=title,ylabel='Accuracy (%)',xlim=(0,100))
    ax.grid(alpha=.2)
for ax in axes[-1]:ax.set_xlabel('Committed round')
handles,labels=axes[0,0].get_legend_handles_labels()
fig.legend(handles,labels,loc='upper center',ncol=3,frameon=False,fontsize=9)
fig.text(.5,.012,'Seed 42 only; shaded: fixed rounds 81-100. Historical run: serial clients; new runs: four concurrent clients.',
         ha='center',fontsize=9)
fig.tight_layout(rect=(0,.035,1,.955))
fig.savefig(OUT/'historical_and_new_curves.png',dpi=180)
fig.savefig(OUT/'historical_and_new_curves.svg')
plt.close(fig)

report='''**新聚合与 A＋B：seed42 结果复核（2026-10-10）**

结论：在同一新聚合下，整套 A＋B 比永久冻结 LoRA A 有明确的描述性增益，头、中、尾三组均值都提高。相比历史 FedAvg＋A＋B，新方案呈现尾类提高、头类下降、Overall 略降的取舍。当前证据支持继续学习 A 的方向，但不足以固定现有聚合、保持修正和来源 C 的所有设计。

本文中的“方法 A”指 A 因子学习加 Full-CP 保持修正；“方法 B”指来源 C 产生的共享 B 补充更新。普通 B 因子训练不是这里的“方法 B”。

**数据及执行核验**

两条正式轨迹均完成100轮。逐样本复算202份提交模型预测和52份阶段预测，共254份、每份10000张测试图，结果与存档汇总一致。核对8700条客户端训练预算、290个聚合/并行阶段记录、样本划分、初始化、训练/测试池、客户端日程、反馈样本及配置。frozen 的101个预测元数据中 A 哈希一致；两组初始预测与间隔一致。当前本地训练源文件指纹与注册指纹一致。

压缩结果没有模型权重检查点，本次没有重新构造有效参数更新张量；重构误差、冻结因子误差及范数部分依据保存的事件审计记录。独立随机种子只有42，轮次、类别和诊断事件都不能当作独立种子。

**1．整套方法在新聚合下是否有效**

主指标统一为预先指定的第81—100轮均值，不挑最好轮次。准确率单位为%，差值单位为百分点。

| 方法 | Overall | Head20 | Middle60 | Tail20 |
|---|---:|---:|---:|---:|
| 新聚合，永久冻结 A | 67.8600 | 70.7050 | 66.3308 | 69.6025 |
| 新聚合，当前 A＋B | 70.4700 | 73.9950 | 68.5742 | 72.6325 |
| 差值 | +2.6100 | +3.2900 | +2.2433 | +3.0300 |

这说明仅调整客户端聚合权重、一直冻结 A，没有达到整套方案的表现。三组均值改善不等于每类改善：100类中63类提高、33类下降、4类不变；尾部20类中13类提高、7类下降。

当前 A 学习日程是第1—90轮每轮额外训练 A 一轮，第91—100轮不训练 A；并非这次实验已经验证了“偶尔按需解冻”。frozen 没有这些额外训练，因此这组对照没有排除单纯增加训练机会的作用。

来源 C 首次在第30轮执行。在第29轮、尚未使用来源 C 时，A＋B 分支相对冻结分支已经 Overall +2.18、Tail20 +2.70。这将早期差距定位到 A 学习及其保持修正这部分，不能将全部增益记在 C 迁移上；也不能将第29轮差距作为最终贡献比例。

**2．收益主要体现为新增学习超过额外损失**

按两组相同的第0轮预测划分“起初答对”和“起初答错”的样本，再对最后20轮取平均。下表均为 A＋B 相对 frozen 的样本数差异。

| 样本组 | 起初答错、后来答对的增加 | 起初答对、后来答错的增加 | 净增加答对 |
|---|---:|---:|---:|
| 全部10000张 | +393.50 | +132.50 | +261.00 |
| 头类2000张 | +85.45 | +19.65 | +65.80 |
| 中类6000张 | +222.95 | +88.35 | +134.60 |
| 尾类2000张 | +85.10 | +24.50 | +60.60 |

这与“长期固定 A 的学习增益不足”一致；不能据此把新增学习全部归因为 A 的特殊方向，也不能描述成“相对冻结组减少了遗忘”。“学会/遗忘”在这里指固定测试样本的答对/答错变化，不是直接测量抽象知识。

两组第100轮 Tail20 分别为69.95和72.80，均达到各自观测最高值；本次没有出现最终尾类准确率回落。样本层面的遗忘仍存在。到最后20轮，AB平均仍有头类455.10、中类1633.00、尾类482.60张测试图从未在已观察的提交轮次答对，继续学习仍有空间；这本身不能诊断为模型容量不足。

**3．新聚合相对旧聚合是什么变化**

| A＋B 的聚合版本 | Overall | Head20 | Middle60 | Tail20 |
|---|---:|---:|---:|---:|
| 历史样本量 FedAvg | 70.6965 | 75.9550 | 68.7900 | 71.1575 |
| 当前新聚合 | 70.4700 | 73.9950 | 68.5742 | 72.6325 |
| 新减旧 | −0.2265 | −1.9600 | −0.2158 | +1.4750 |

两次 AB 的7项数据/初始化/日程/反馈哈希、样本级客户端划分、训练预算、Python/PyTorch/CUDA/cuDNN及GPU型号一致。主要算法配置变化为聚合；但执行也由串行变为四客户端并行，并增加观测性阶段测试。短程数值检查通过，不能由此保证100轮完全相同的数值轨迹。因此这是可比性较好的历史对照，仍不应宣称是严格只改变聚合的一次重跑。

现有结果不支持“新聚合全面改善 A＋B”。更准确的说法是：目前获得了更高的尾类准确率，同时头类和Overall下降。若目标是提升全体类别学习，当前 lambda=0.1 尚不宜直接固定为最终方案。

旧E2的68.1010/69.6500（Overall/Tail20）也不能作为本次 frozen 的完全匹配 FedAvg 基线：E2另有9次额外B训练，共3168步，新 frozen 没有，而且记录的环境不同。历史 tailrw-g16 为70.9120/72.2775，新AB相对它 Overall −0.4420、Tail20 +0.3550；两者方法与环境也不同，不能作为聚合单因素证据。

**4．聚合改变了代表权，但尚未证明它改善了每次实际更新**

三个尾类集中客户端27—29的总聚合权重由1.1985%提高到11.4629%。按客户端类别比例混合计算的尾类质量由1.4105%变为10.7835%；头类由61.3349%变为51.6852%，中类由37.2545%变为37.5313%。这些是计数代理量，不是模型知识或梯度贡献。两个客户端5、6权重接近数值下界。

本次还保存了同一起点、同一批本地更新改用原样本量权重的未提交模型。在 AB 的8个诊断轮次，新B聚合相对原权重聚合：Head20全部下降，平均−0.14375；Overall平均−0.02250；Tail20平均−0.00625，正负各3次。frozen的相同对照平均Overall −0.01125、Tail20为0。

因此，“尾类客户端获得更多权重”已实现，但“这种权重一定给出更好的单轮尾类方向”没有被观察到。单步诊断也不能否定长期轨迹带来的尾类收益。可探索的调整应是抑制客户端权重过度集中、检查计数代理与实际类别响应的关系，而非为了给A/B留空间而故意削弱聚合。任何混合强度或下界应在训练侧验证集上确定，不在这批测试结果上反复选优。

**5．A 的保持修正是当前最值得定位的环节**

下表是预先指定诊断轮次的即时准确率变化，不是各组件的长期消融效果。

| 操作 | 事件数 | Overall | Head20 | Middle60 | Tail20 |
|---|---:|---:|---:|---:|---:|
| 普通 A 更新 | 7 | +0.0243 | −0.0143 | +0.0310 | +0.0429 |
| 随后的保持修正 | 7 | −0.0271 | +0.0143 | −0.0333 | −0.0500 |
| 来源 C 的 B 补充更新 | 6 | +0.0283 | −0.0583 | +0.0250 | +0.1250 |

保持修正在这7次测试中，Tail20为4次下降、3次不变，无正增益；第90轮普通A更新使Tail20提高0.20，修正后又下降0.20。它的平均绝对影响较小，不能把7次即时变化累加成最终损失或直接宣布保持方法无效。

参数记录提供了一条机制线索：第61—90轮，修正后有效更新范数平均约为普通A提议的39.9%；A参数更新与原提议的平均余弦为−0.112。与此同时，这30轮保持损失均降低、分类参考损失均升高。即后期修正明显改变了普通学习的方向和幅度，并在训练目标之间做取舍。范数和余弦来自保存的记录，不是本次由检查点张量重新计算。

合理假设是现有保持目标/强度与新聚合下的学习需求可能不再匹配，但现在不能证明它是长期性能不足的原因。普通A即时Head20均值也不是正值，因此不能编成“普通A每次都学好头类，保持修正专门破坏头类”的故事。

**6．当前来源 B 更像尾类补充，尚未成为全类别学习模块**

六个有官方测试阶段诊断的C事件中，Tail20有5次提高、1次下降，平均+0.125；Overall平均+0.0283，Head20平均−0.0583。全部8个实际C事件的来源并集仍是30个客户端，来源池筛选没有真正缩小。补充更新的有效范数约为当轮普通B更新的1.36—2.37倍。没有同幅度直接校准对照，因此本次不能证明收益独属于来源知识迁移，也不能推翻之前F对照的结论。

“只在尾类提供平均帮助”和“提升所有类别的学习”是两个不同目标。若B的目标转向后者，当前证据说明还需要处理头类代价与来源使用方式，继续扩大C或简单增加优化步数缺少依据。

**7．加速与成本**

正式100轮耗时：frozen 4.04小时，AB 8.73小时，AB约为2.16倍。包括测试、日志、I/O与训练侧诊断，不能全部称为纯训练开销。普通本地优化均为105600步，AB另加31680步A训练、270步保持修正与16步C优化；本地图像呈现次数增加30%。

四客户端并行的六客户端同起点试跑速度比分别为：frozen B 1.274倍、AB B 1.247倍、AB A 1.217倍。数值比较均通过，最大参数绝对误差5.73e-7；最后只有2个客户端的短组也通过。正式阶段记录显示每次30个客户端完整执行。可据此确认这次并行尝试有效且尾批可运行，不能称为4倍或完整100轮的匹配加速测量。正式记录的PyTorch峰值已分配显存约4.82/9.61 GiB，不等于nvidia-smi总占用。

**下一步用于固定方法的建议**

优先保留“需要继续学习A”的研究方向，不急于增加动态开关或扩大来源C。若下一步只补一个完整实验，建议在相同新聚合、相同A训练日程、相同共享B下，只取消A后的保持修正，普通A提议直接提交，其他设置保持一致。与现有AB比较，就能直接回答保持修正的长期收益是否值得其学习代价；它不是取消A训练，也不是同时取消C。

若去掉保持修正仍能维持尾类并改善学习，应重设保持目标/强度，而非继续包装现有强保持；若后期尾类出现退化且总体取舍较差，保持部分就有了直接的长期依据。这个对照只检验保持在当前C存在时的作用。随后若需确认来源B，再补相同新聚合下的方法A单独运行。当前计数聚合的头尾取舍需要独立处理，不能先假定让A/B必然补回来。

文件索引：[复核记录](audit.json)、[性能](performance.csv)、[历史配置核对](historical_audit.json)、[历史结果](historical_performance.csv)、[逐事件变化](stage_by_round.csv)、[共同起点样本变化](cohort_contrasts.csv)、[修正参数记录汇总](correction_diagnostics.csv)、[并行试跑](parallel_pilot.csv)、[成本](costs.csv)。

![新旧聚合轨迹对照](historical_and_new_curves.png)
'''
(OUT/'report.md').write_text(report,encoding='utf-8')
print(json.dumps(dict(output=str(OUT), audit_passed=True, raw_prediction_files=254,
    performance=performance,historical_ab=history[0],costs=costs,pilots=pilots),indent=2,ensure_ascii=False))
