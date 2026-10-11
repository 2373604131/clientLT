"""Read-only audits and summaries for three ordinary-A aggregation trajectories."""
from pathlib import Path
from statistics import mean

from scripts.run_ab_validation import write_json
from tools.client_aggregation.analysis import METRICS, audit_run, sample_dynamics, stage_comparisons
from tools.client_aggregation.frozen_tailrw import paired_audit
from tools.client_aggregation.protocol import load_json, write_table

METHODS = ('fedavg', 'tailrw16', 'joint')


def load_plain(root, rule):
    return load_a_run(root, rule, 'plain_a')


def load_a_run(root, rule, arm):
    from scripts.run_cliplora_joint_aggregation import make_job
    root = Path(root)
    plan = load_json(root / 'experiment_plan.json')
    if plan['aggregation'].get('rule', 'joint') != rule:
        raise ValueError('Unexpected aggregation policy: ' + str(root))
    job = make_job(plan, arm)
    return audit_run(root / job['run'], job)


def status_rows(root, arm='plain_a'):
    """Lightweight progress only; completed results are audited by summary/pack."""
    rows = []
    for rule in METHODS:
        run = Path(root) / rule / 'runs/seed42' / arm
        path = run / ('completion.json' if (run / 'completion.json').is_file() else 'progress.json')
        state = load_json(path) if path.is_file() else {}
        rows.append(dict(method=rule, status='completed_unverified' if path.name == 'completion.json'
                         else 'incomplete' if state else 'not_started',
                         completed_round=state.get('completed_round', 0),
                         normal_B_steps=state.get('normal_optimizer_steps', 0),
                         extra_A_steps=state.get('extra_optimizer_steps', 0),
                         correction_steps=state.get('functional_correction_steps', 0)))
    return rows


def summarize(root, arm='plain_a'):
    root = Path(root).resolve()
    out = root / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    statuses = status_rows(root, arm)
    results = {}
    for status in statuses:
        rule = status['method']
        if status['status'] == 'completed_unverified':
            try:
                results[rule] = load_plain(root / rule, rule) if arm == 'plain_a' else load_a_run(root / rule, rule, arm)
                status['status'] = 'complete'
            except (ValueError, OSError, KeyError, TypeError, AssertionError) as error:
                status.update(status='invalid', reason=str(error))
        status.setdefault('reason', '')
        print(rule, status['status'], 'round=' + str(status['completed_round']), status['reason'])

    performance, curves, dynamics, stages, classes, costs = [], [], [], [], [], []
    for rule, r in results.items():
        window = [x for x in r['curves'] if 81 <= int(x['round']) <= 100]
        tail = [float(x['bottom20_tail_acc']) for x in r['curves']]
        performance.append(dict(method=rule, seed=42,
            **{'last20_'+m: mean(float(x[m]) for x in window) for m in METRICS},
            final_tail=tail[-1], peak_tail=max(tail), tail_peak_to_final=max(tail)-tail[-1]))
        curves.extend(dict(row, method=rule) for row in r['curves'])
        dynamics.extend(dict(row, method=rule) for row in sample_dynamics(r))
        stages.extend(dict(row, method=rule) for row in stage_comparisons(r))
        for c in range(100):
            classes.append(dict(method=rule, class_id=c,
                last20_accuracy=mean(float(p['correct'][p['class_id']==c].sum())
                                     for rnd, p in r['predictions'].items() if 81 <= rnd <= 100)))
        done = r['progress']
        costs.append(dict(method=rule, normal_B_steps=done['normal_optimizer_steps'],
            extra_A_steps=done['extra_optimizer_steps'], correction_steps=done['functional_correction_steps'],
            transfer_steps=done.get('b_transfer_optimizer_steps', 0), elapsed_seconds=done['elapsed_seconds'],
            local_images=sum(int(b['sample_presentations']) for b in r['budgets']),
            stage_test_seconds=done['stage_diagnostic_seconds']))

    pairs, perf = [], {p['method']:p for p in performance}
    for left, right in (('tailrw16', 'fedavg'), ('joint', 'fedavg'), ('joint', 'tailrw16')):
        pair = dict(left=left, right=right, status='unavailable')
        if left in results and right in results:
            pair.update(paired_audit(results[left], results[right]))
            pair['contrast'] = arm + ': ' + left + ' minus ' + right
            if arm == 'method_a':
                # Allowed degenerate correction skips can differ between trajectories.
                pair['mismatches'] = [x for x in pair['mismatches'] if x != 'budget.functional_correction_steps']
                pair['same_correction_steps'] = (results[left]['progress']['functional_correction_steps']
                                                == results[right]['progress']['functional_correction_steps'])
                pair['budget_matched'] = pair['same_correction_steps']
                pair['status'] = 'mismatch' if pair['mismatches'] else 'eligible'
            if pair['source_differences']:
                pair['mismatches'].append('source_differences')
                pair['status'] = 'mismatch'
            if pair['status'] == 'eligible' and not pair['same_recorded_environment']:
                pair['status'] = 'execution_differs'
            if pair['status'] in ('eligible', 'execution_differs'):
                pair['delta_pp'] = {m: perf[left]['last20_'+m]-perf[right]['last20_'+m] for m in METRICS}
        pairs.append(pair)

    dynamic_summary, stage_summary = [], []
    for rule in results:
        for group in METRICS:
            rows = [d for d in dynamics if d['method']==rule and d['group']==group and 81<=d['round']<=100]
            dynamic_summary.append(dict(method=rule, group=group,
                **{k:mean(d[k] for d in rows) for k in ('retained_initial', 'learned_initial_wrong',
                     'never_correct_so_far', 'previously_correct_now_wrong', 'correct_to_wrong', 'wrong_to_correct')}))
        for operation, group in sorted({(d['operation'], d['group']) for d in stages if d['method']==rule}):
            rows = [d for d in stages if d['method']==rule and d['group']==group and d['operation']==operation]
            stage_summary.append(dict(method=rule, operation=operation, group=group, rounds=len(rows),
                **{k:mean(d[k] for d in rows) for k in ('delta_accuracy', 'corrected_examples', 'damaged_examples')}))
    for name, rows in dict(status=statuses, performance=performance, curves=curves,
            sample_dynamics=dynamics, sample_dynamics_last20=dynamic_summary,
            stage_effects=stages, stage_summary=stage_summary, per_class=classes, costs=costs).items():
        write_table(out / (name + '.csv'), rows)
    write_json(out / 'pair_audit.json', dict(pairs=pairs, independent_seeds=1,
        arm=arm, same_local_SGD_budget=True, additional_feedback_and_correction=arm=='method_a',
        budget_matched_among_plain_A=arm=='plain_a', budget_matched_to_frozen_A=False))
    lines = ['# 普通 A 训练：三种聚合（seed42）', '',
        '第1—90轮：B训练3个epoch并聚合，再固定聚合后的B，A训练1个epoch并聚合。',
        '第91—100轮：A固定为第90轮结果，仅训练B。A固定学习率0.001、SGD动量0.9、无权重衰减。',
        '没有保持修正、功能历史、方法B共享迁移或基于测试的调度。三组只改变A/B聚合使用的静态客户端权重。',
        '每组105600个B优化步骤、31680个A优化步骤，共137280步；新聚合lambda=0.1，TailRW gamma=16。',
        '固定终点为第81—100轮均值。单种子，不作显著性结论；与冻结A比较时训练预算不同。', '',
        '| 聚合 | Overall | Head20 | Middle60 | Tail20 |', '|---|---:|---:|---:|---:|']
    if arm == 'method_a':
        lines = ['# 方法 A（Full-CP）：三种聚合（seed42）', '',
            '第1—90轮：B训练3个epoch并聚合，固定B，A训练1个epoch并聚合，再执行Full-CP修正并提交历史。',
            '第91—100轮：固定A，仅训练B。Full-CP固定lambda=10、mu=1、三步修正、步长0.1；无方法B共享迁移。',
            '三组只改变普通A/B聚合的静态权重；CP原始样本权重、来源优先级和LA先验不变。',
            '每组本地SGD为105600个B步骤加31680个A步骤，共137280步；另有反馈计算和至多270次修正。',
            '无有效目标或提案半径为零时跳过修正，按实际步骤和耗时报告成本。',
            '固定终点为第81—100轮均值。只有seed42，不作跨种子显著性结论。', '',
            '| 聚合 | Overall | Head20 | Middle60 | Tail20 |', '|---|---:|---:|---:|---:|']
    for row in performance:
        lines.append('| '+row['method']+' | '+' | '.join(f'{row["last20_"+m]:.4f}' for m in METRICS[:4])+' |')
    lines += ['', '配对差值（百分点）：']
    for pair in pairs:
        lines.append('- '+pair['left']+' − '+pair['right']+'：'+pair['status'])
        if 'delta_pp' in pair:
            lines.append('  '+', '.join(f'{m}={v:+.4f}' for m,v in pair['delta_pp'].items()))
        if pair.get('mismatches'):
            lines.append('  不匹配：'+', '.join(pair['mismatches']))
        if pair['status'] == 'execution_differs':
            lines.append('  记录的GPU或软件环境不同，不能视为完全相同执行条件。')
    lines += ['', '运行状态：']
    lines += ['- '+s['method']+'：'+s['status']+'，round='+str(s['completed_round'])+' '+s['reason'] for s in statuses]
    lines += ['', 'stage_effects.csv 保留同状态聚合对照及普通A学习的即时变化；不等同于长期组件消融。',
              'sample_dynamics_last20.csv 区分初始能力保持、新学会、曾答对后再错和始终未答对。']
    (out / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('Report:', out / 'report.md')
    return statuses, pairs
