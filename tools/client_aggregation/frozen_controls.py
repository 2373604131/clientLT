"""Audited frozen-A comparison; the historical joint run is read-only."""
from pathlib import Path
import statistics

from scripts.run_ab_validation import write_json
from tools.client_aggregation.analysis import METRICS, sample_dynamics, stage_comparisons
from tools.client_aggregation.frozen_tailrw import load_frozen, paired_audit
from tools.client_aggregation.protocol import load_json, write_table


def summarize(root, compare_root):
    root = Path(root).resolve()
    compare_root = Path(compare_root).resolve()
    out = root/'analysis'
    out.mkdir(parents=True, exist_ok=True)
    roots = dict(tailrw16=root/'tailrw16', fedavg=root/'fedavg', joint=compare_root)
    results, statuses = {}, []
    for rule, folder in roots.items():
        run = folder/'runs/seed42/frozen'
        status = dict(method=rule, status='missing', completed_round=0, reason='', root=str(folder))
        try:
            if (run/'progress.json').is_file():
                status.update(status='incomplete', completed_round=load_json(run/'progress.json')['completed_round'])
            if (run/'completion.json').is_file():
                results[rule] = load_frozen(folder, rule)
                status.update(status='complete', completed_round=100)
        except (ValueError, OSError, KeyError, TypeError, AssertionError) as error:
            status.update(status='invalid', reason=str(error))
        statuses.append(status)
        print(rule, status['status'], 'round='+str(status['completed_round']), status['reason'])

    pairs = []
    for left, right in (('tailrw16', 'fedavg'), ('joint', 'fedavg'), ('tailrw16', 'joint')):
        pair = dict(left=left, right=right, status='unavailable', reason='Both completed runs are required')
        if left in results and right in results:
            pair = dict(paired_audit(results[left], results[right], allow_execution_difference='joint' in (left, right)),
                        left=left, right=right)
            # Newly trained controls must share exactly the same registered source.
            if 'joint' not in (left, right) and pair['source_differences']:
                pair['mismatches'].append('new_control_sources_differ')
                pair['status'] = 'mismatch'
        pairs.append(pair)

    performance, dynamics, stages, curves, costs, pilots, classes = [], [], [], [], [], [], []
    for rule, result in results.items():
        window = [row for row in result['curves'] if 81 <= int(row['round']) <= 100]
        performance.append(dict(method=rule, seed=42,
            **{'last20_'+m: statistics.mean(float(x[m]) for x in window) for m in METRICS}))
        dynamics.extend(dict(row, method=rule) for row in sample_dynamics(result))
        stages.extend(dict(row, method=rule) for row in stage_comparisons(result))
        curves.extend(dict(row, method=rule) for row in result['curves'])
        for c in range(100):
            accuracies = [100*float(p['correct'][p['class_id'] == c].mean())
                          for r, p in result['predictions'].items() if 81 <= r <= 100]
            classes.append(dict(method=rule, class_id=c, last20_accuracy=statistics.mean(accuracies)))
        done = result['progress']
        costs.append(dict(method=rule, normal_B_steps=done['normal_optimizer_steps'],
            extra_steps=done['extra_optimizer_steps'], correction_steps=done['functional_correction_steps'],
            transfer_steps=done.get('b_transfer_optimizer_steps', 0), elapsed_seconds=done['elapsed_seconds'],
            client_concurrency=result['config']['client_execution']['max_concurrent_clients']))
        for path in (roots[rule]/'smoke/seed42/frozen/parallel_benchmark').glob('factor_*.json'):
            pilot = load_json(path)
            pilots.append(dict(method=rule, factor=pilot['factor'], passed=pilot['passed'],
                slots=pilot['parallel']['slots'], clients=pilot['parallel']['clients'],
                speedup=pilot['speedup'], short_group_clients=pilot['short_group']['clients'],
                max_parameter_error=max(x['max_abs'] for x in pilot['comparisons']+pilot['short_comparisons'])))

    perf = {r['method']: r for r in performance}
    for pair in pairs:
        if pair['status'] in ('eligible', 'execution_differs'):
            pair['delta_pp'] = {m: perf[pair['left']]['last20_'+m]-perf[pair['right']]['last20_'+m] for m in METRICS}
    for name, rows in dict(status=statuses, performance=performance, sample_dynamics=dynamics,
            stage_effects=stages, curves=curves, costs=costs, parallel_pilot=pilots, per_class=classes).items():
        write_table(out/(name+'.csv'), rows)
    write_json(out/'pair_audit.json', dict(pairs=pairs, independent_seeds=1,
        primary='rounds81..100 Overall; Head20/Middle60/Tail20 reported jointly'))
    lines = ['# 冻结 A 的聚合对照（seed42）', '',
        '新增两条100轮训练：16倍加权与FedAvg。仅聚合权重不同；A全程固定，B每轮3个epoch，共105600步。',
        '无额外A/B训练、保持修正、来源C、直接校准。本地损失与原始计数LA先验不变。',
        '主指标预先固定为第81—100轮Overall均值，同时报告头、中、尾类；只有一个种子，不作跨种子显著性结论。', '',
        '| 聚合 | Overall | Head20 | Middle60 | Tail20 |', '|---|---:|---:|---:|---:|']
    for row in performance:
        lines.append('| '+row['method']+' | '+' | '.join(f'{row["last20_"+m]:.4f}' for m in METRICS[:4])+' |')
    lines += ['', '配对结果（左减右，单位：百分点）：']
    for pair in pairs:
        lines.append('- '+pair['left']+' − '+pair['right']+'：'+pair['status'])
        if 'delta_pp' in pair:
            lines.append('  '+', '.join(f'{m}={v:+.4f}' for m,v in pair['delta_pp'].items()))
        if pair.get('mismatches'):
            lines.append('  不匹配项：'+', '.join(pair['mismatches']))
        if pair['status'] == 'execution_differs':
            lines.append('  训练预算与数据配置匹配，但并行数或CUDA数值策略不同（历史4并行、本次默认6并行）；保留此执行差异，不能当作完全相同执行设置或全程加速对照。')
        if pair.get('same_recorded_environment') is False:
            lines.append('  记录的软件或GPU环境不同，小幅精度差异及耗时须结合环境解读。')
    lines += ['', '运行状态：']
    lines += ['- '+r['method']+'：'+r['status']+'，round='+str(r['completed_round'])+' '+r['reason'] for r in statuses]
    lines += ['', 'sample_dynamics.csv 区分初始能力保持、新答对、逐轮遗忘与始终未答对；per_class.csv 保存每类结果。',
        'stage_effects.csv 是同状态同客户端更新的即时对照，不代替完整训练轨迹。',
        '本版不做串行/并行数值对比或阈值拦截。parallel_pilot.csv 若有内容，仅来自历史对照；本版smoke只执行一轮正常训练。历史新聚合只读，未重新训练。',
        '历史E2额外训练过B，旧16倍实验训练过A，均不放入此表。',
        '新聚合参考目录：'+str(compare_root)]
    (out/'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Report:', out/'report.md')
    return statuses, pairs
