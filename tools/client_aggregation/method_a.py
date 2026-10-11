"""Method-A-only suite and read-only, within-rule contrasts to ordinary A."""
from pathlib import Path
from statistics import mean

import numpy as np

from scripts.run_ab_validation import write_json
from tools.client_aggregation.analysis import METRICS, sample_dynamics
from tools.client_aggregation.frozen_tailrw import paired_audit
from tools.client_aggregation.plain_a import METHODS, load_a_run, load_plain, summarize as summarize_suite
from tools.client_aggregation.protocol import write_table

# These integration files changed to add the opt-in arm. Changes to local training,
# feedback, functional objectives, data or CUDA policy are not waived here.
INTEGRATION_FILES = {
    'tools/client_aggregation/protocol.py', 'tools/client_aggregation/runtime.py',
    'tools/client_aggregation/analysis.py', 'tools/client_aggregation/plain_a.py',
    'tools/client_aggregation/method_a.py', 'scripts/run_cliplora_joint_aggregation.py',
    'scripts/run_cliplora_plain_a_aggregation.py', 'scripts/run_cliplora_method_a_aggregation.py',
}


def compare_plain(method, plain):
    if method['job']['arm'] != 'method_a' or plain['job']['arm'] != 'plain_a':
        raise ValueError('Compare Method A only against ordinary A')
    rule = method['job']['aggregation'].get('rule', 'joint')
    if plain['job']['aggregation'].get('rule', 'joint') != rule:
        raise ValueError('Within-rule comparison requires identical aggregation rules')
    pair = paired_audit(method, plain)
    intentional = {'variant', 'budget.functional_correction_steps'}
    intentional.update('code.'+name for name in INTEGRATION_FILES)
    pair['mismatches'] = [x for x in pair['mismatches'] if x not in intentional]
    # paired_audit permits a wider legacy set; be stricter for this new contrast.
    for name in pair['source_differences']:
        if name not in INTEGRATION_FILES and 'code.'+name not in pair['mismatches']:
            pair['mismatches'].append('code.'+name)
    difference = float(np.max(np.abs(np.asarray(method['job']['aggregation']['weights'])
                                    - np.asarray(plain['job']['aggregation']['weights']))))
    if difference > 1e-8:
        pair['mismatches'].append('aggregation.weights')
    pair.update(method=rule, max_weight_difference=difference,
        contrast='Full-CP minus ordinary A with '+rule, budget_matched=False,
        local_SGD_budget_matched=not any(x in pair['mismatches'] for x in
            ('budget.normal_optimizer_steps', 'budget.extra_optimizer_steps')),
        limitation='one seed; added feedback/correction cost; source changes are listed, not numerically certified equivalent')
    pair['status'] = ('mismatch' if pair['mismatches'] else
                      'execution_differs' if not pair['same_recorded_environment'] else
                      'integration_source_differs' if pair['source_differences'] else 'eligible')
    if pair['status'] != 'mismatch':
        pair['delta_pp'] = {m: mean(float(x[m]) for x in method['curves'] if 81 <= int(x['round']) <= 100)
                           - mean(float(x[m]) for x in plain['curves'] if 81 <= int(x['round']) <= 100)
                           for m in METRICS}
        values = []
        for result in (method, plain):
            tail = [float(x['bottom20_tail_acc']) for x in result['curves']]
            dynamic = [r for r in sample_dynamics(result)
                       if r['group'] == 'bottom20_tail_acc' and 81 <= r['round'] <= 100]
            values.append(dict(tail_peak_to_final=max(tail)-tail[-1],
                tail_change_90_to_100=tail[-1]-tail[90],
                **{k: mean(r[k] for r in dynamic) for k in
                   ('retained_initial', 'learned_initial_wrong', 'previously_correct_now_wrong', 'never_correct_so_far')}))
        pair['tail_dynamics'] = dict(method_a=values[0], plain_a=values[1])
        pair['costs'] = {name: {k: result['progress'].get(k) for k in
            ('normal_optimizer_steps', 'extra_optimizer_steps', 'functional_correction_steps',
             'b_transfer_optimizer_steps', 'elapsed_seconds')}
            for name, result in (('method_a', method), ('plain_a', plain))}
    return pair


def summarize(root, plain_root):
    statuses, pairs = summarize_suite(root, arm='method_a')
    root, plain_root = Path(root).resolve(), Path(plain_root).resolve()
    comparisons, table = [], []
    for rule in METHODS:
        pair = dict(method=rule, status='unavailable', plain_root=str(plain_root))
        complete = next(s['status'] for s in statuses if s['method'] == rule) == 'complete'
        if complete and (plain_root/rule/'runs/seed42/plain_a/completion.json').is_file():
            try:
                pair.update(compare_plain(load_a_run(root/rule, rule, 'method_a'), load_plain(plain_root/rule, rule)))
            except (ValueError, OSError, KeyError, TypeError, AssertionError) as error:
                pair.update(status='mismatch', reason=str(error))
        comparisons.append(pair)
        table.append(dict(method=rule, status=pair['status'],
                          **{'delta_'+k: v for k, v in pair.get('delta_pp', {}).items()}))
    out = root/'analysis'
    write_json(out/'method_a_vs_plain_a.json', dict(comparisons=comparisons, independent_seeds=1,
        endpoint='committed rounds81..100; no best-round selection'))
    write_table(out/'method_a_vs_plain_a.csv', table)
    lines = ['', '加入方法 A 的配对增量（Full-CP减普通A，百分点）：',
             '普通A对照目录：'+str(plain_root), '',
             '| 聚合 | 配对状态 | ΔOverall | ΔHead20 | ΔMiddle60 | ΔTail20 |',
             '|---|---|---:|---:|---:|---:|']
    for pair in comparisons:
        delta = pair.get('delta_pp', {})
        lines.append('| '+pair['method']+' | '+pair['status']+' | '+
                     ' | '.join(f'{delta[m]:+.4f}' if m in delta else '' for m in METRICS[:4])+' |')
    lines += ['', '本地SGD预算保持相同，但方法A额外使用见证反馈和功能修正，不能声称总计算预算相同。',
              'integration_source_differs 表示实验接入代码有变化；核心训练与反馈文件哈希仍须匹配。',
              'execution_differs 表示记录的软硬件环境不同。完整差异、遗忘统计和实际成本见 method_a_vs_plain_a.json。',
              '普通A对照不可用时仍可打包本次有效结果；不输出未经配对验证的增量。']
    with (out/'report.md').open('a', encoding='utf-8') as stream:
        stream.write('\n'.join(lines)+'\n')
    return statuses, pairs
