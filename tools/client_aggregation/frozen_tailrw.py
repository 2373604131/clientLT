"""One frozen-A TailRW16 supplement and a read-only comparison to the joint frozen arm."""
from pathlib import Path
import statistics

import numpy as np

from scripts.run_ab_validation import write_json
from tools.client_aggregation.protocol import (
    SCHEMA, DIAGNOSTIC_ROUNDS, PAIR_HASHES, load_json, read_csv, write_table)

CONTRACT = dict(schema=SCHEMA, experiment='frozen_tailrw16_v1', seed=42, rounds=100,
    clients=30, partition='client-longtail', arms=['frozen'],
    aggregation='(n_j + 16*n_j_tail)/(N + 16*N_tail)', gamma=16.,
    scope='ordinary B only, rounds1..100; initial LoRA A fixed throughout',
    normal_B_epochs=3, extra_B_epochs=0, extra_A_epochs=0, functional_correction=False,
    source_C_transfer=False, local_loss='unchanged global LA, tau=1',
    optimizer_steps=105600, primary='committed rounds81..100; report Overall/Head20/Middle60/Tail20',
    diagnostics_rounds=list(DIAGNOSTIC_ROUNDS),
    diagnostics='read-only official test; no feedback, model selection, or weight optimization',
    comparison='existing joint-aggregation frozen arm only; do not substitute E2 or TailRW with A training',
    selection='gamma16 requested as a fixed control; no new parameter sweep')


def load_frozen(root, required_rule):
    """Audit a completed run without writing anything under its experiment root."""
    from scripts.run_cliplora_joint_aggregation import make_job
    from tools.client_aggregation.analysis import audit_run
    root = Path(root)
    plan = load_json(root / 'experiment_plan.json')
    if plan['aggregation'].get('rule', 'joint') != required_rule:
        raise ValueError('Wrong aggregation policy in comparison: ' + str(root))
    job = make_job(plan, 'frozen')
    return audit_run(root / job['run'], job)


def paired_audit(tailrw, joint):
    """Same data, initialization, ordinary training, budget and execution; only weights differ."""
    mismatches = [k for k in PAIR_HASHES if tailrw['metadata'][k] != joint['metadata'][k]]
    a, b = tailrw['config']['base_training_config'], joint['config']['base_training_config']
    mismatches += ['base.'+k for k in set(a) | set(b) if k != 'aggregation' and a.get(k) != b.get(k)]
    for key in ('counts', 'matrix_sha256', 'partition_sha256'):
        if tailrw['job']['aggregation'][key] != joint['job']['aggregation'][key]:
            mismatches.append('aggregation.'+key)
    for key in ('client_execution', 'variant', 'refresh_rounds'):
        if tailrw['config'].get(key) != joint['config'].get(key):
            mismatches.append(key)
    for key in ('sample_id', 'class_id', 'prediction', 'correct'):
        if not np.array_equal(tailrw['predictions'][0][key], joint['predictions'][0][key]):
            mismatches.append('initial.'+key)
    for key in ('normal_optimizer_steps', 'extra_optimizer_steps', 'functional_correction_steps'):
        if tailrw['progress'][key] != joint['progress'][key]:
            mismatches.append('budget.'+key)
    # Both arms use the same local training code; the opt-in orchestration/weight branch changed.
    codes = [x['job'].get('code_sha256', {}) for x in (tailrw, joint)]
    code_differences = sorted(k for k in set(codes[0]) | set(codes[1]) if codes[0].get(k) != codes[1].get(k))
    extension_files = {'tools/client_aggregation/protocol.py', 'tools/client_aggregation/runtime.py',
        'tools/client_aggregation/analysis.py', 'tools/client_aggregation/frozen_tailrw.py',
        'scripts/run_cliplora_joint_aggregation.py'}
    mismatches += ['code.'+k for k in code_differences if k not in extension_files]
    environment_keys = ('python', 'torch', 'cuda', 'cudnn', 'gpu')
    same_environment = all(tailrw['metadata'].get('environment', {}).get(k)
        == joint['metadata'].get('environment', {}).get(k) for k in environment_keys)
    return dict(status='mismatch' if mismatches else 'eligible', mismatches=mismatches,
        budget_matched=not any(x.startswith('budget.') for x in mismatches),
        same_recorded_environment=same_environment, source_differences=code_differences,
        independent_seeds=1, contrast='frozen TailRW16 minus frozen joint aggregation')


def summarize(root, compare_root=None):
    from tools.client_aggregation.analysis import METRICS, sample_dynamics, stage_comparisons
    root = Path(root).resolve()
    compare_root = Path(compare_root).resolve() if compare_root else root.parent / 'client_aggregation_v2_parallel4'
    out = root / 'analysis'
    out.mkdir(parents=True, exist_ok=True)
    run = root / 'runs/seed42/frozen'
    status = dict(arm='frozen', status='missing', completed_round=0, reason='')
    if (run / 'progress.json').is_file():
        status.update(status='incomplete', completed_round=load_json(run / 'progress.json')['completed_round'])
    result = None
    if (run / 'completion.json').is_file():
        try:
            result = load_frozen(root, 'tailrw16')
            status.update(status='complete', completed_round=100)
        except (ValueError, OSError, KeyError, TypeError, AssertionError) as error:
            status.update(status='invalid', reason=str(error))
    results = {'tailrw16_frozen': result} if result else {}
    pair = dict(status='unavailable', comparison_root=str(compare_root),
                reason='TailRW16 or the joint frozen reference is not complete')
    if result and (compare_root / 'runs/seed42/frozen/completion.json').is_file():
        try:
            reference = load_frozen(compare_root, 'joint')
            pair = dict(paired_audit(result, reference), comparison_root=str(compare_root))
            results['joint_frozen'] = reference
        except (ValueError, OSError, KeyError, TypeError, AssertionError) as error:
            pair.update(status='mismatch', reason=str(error))

    performance, dynamics, stages, curves, costs, pilots = [], [], [], [], [], []
    for name, r in results.items():
        window = [row for row in r['curves'] if 81 <= int(row['round']) <= 100]
        perf = dict(method=name, seed=42, **{'last20_'+m: statistics.mean(float(x[m]) for x in window) for m in METRICS})
        tail = [float(x['bottom20_tail_acc']) for x in r['curves']]
        perf.update(final_tail=tail[-1], tail_peak_to_final=max(tail)-tail[-1], tail_change_90_to_100=tail[100]-tail[90])
        performance.append(perf)
        dynamics.extend(dict(row, method=name) for row in sample_dynamics(r))
        stages.extend(dict(row, method=name) for row in stage_comparisons(r))
        curves.extend(dict(row, method=name) for row in r['curves'])
        done = r['progress']
        costs.append(dict(method=name, normal_B_steps=done['normal_optimizer_steps'],
            extra_steps=done['extra_optimizer_steps'], correction_steps=done['functional_correction_steps'],
            transfer_steps=done.get('b_transfer_optimizer_steps', 0), elapsed_seconds=done['elapsed_seconds'],
            stage_test_seconds=done['stage_diagnostic_seconds'],
            local_images=sum(int(x['sample_presentations']) for x in r['budgets'])))
    if pair['status'] == 'eligible':
        p = {row['method']: row for row in performance}
        pair['tailrw16_minus_joint'] = {m:p['tailrw16_frozen']['last20_'+m]-p['joint_frozen']['last20_'+m] for m in METRICS}
    for path in (root / 'smoke/seed42/frozen/parallel_benchmark').glob('factor_*.json'):
        pilot = load_json(path)
        pilots.append(dict(factor=pilot['factor'], passed=pilot['passed'], speedup=pilot['speedup'],
            short_group_clients=pilot['short_group']['clients'],
            max_parameter_error=max(x['max_abs'] for x in pilot['comparisons']+pilot['short_comparisons'])))
    for name, rows in dict(status=[status], performance=performance, sample_dynamics=dynamics,
            stage_effects=stages, curves=curves, costs=costs, parallel_pilot=pilots).items():
        write_table(out / (name+'.csv'), rows)
    write_json(out / 'pair_audit.json', pair)
    lines = ['# 冻结 LoRA A：16倍加权补充实验', '',
        '全程只训练 B；100轮，每轮每客户端3个epoch，105600步；无额外B训练、无A训练、无保持修正、无来源C。',
        '权重沿用历史公式 (n_j + 16*n_j_tail)/(N + 16*N_tail)。LA、划分、初始化、客户端日程不变。',
        '固定终点为第81—100轮均值，只有seed42；不作跨种子显著性结论。', '',
        '| 方案 | Overall | Head20 | Middle60 | Tail20 |', '|---|---:|---:|---:|---:|']
    for row in performance:
        lines.append('| '+row['method']+' | '+' | '.join(f'{row["last20_"+m]:.4f}' for m in METRICS[:4])+' |')
    lines += ['', '运行状态：'+status['status']+'，round='+str(status['completed_round'])+'。']
    if status['reason']:
        lines.append('审计失败：'+status['reason'])
    if pair['status'] == 'eligible':
        lines += ['', '预算、数据、初始预测和训练设置配对通过。16倍加权减新聚合：',
            ', '.join(f'{m}={v:+.4f} pp' for m,v in pair['tailrw16_minus_joint'].items())+'。']
        if not pair['same_recorded_environment']:
            lines.append('两次记录的软硬件环境不同；小差异与耗时需谨慎解释。')
    else:
        lines += ['', '暂不输出配对结论：'+str(pair.get('reason') or pair.get('mismatches')),
            '新聚合结果目录：'+str(compare_root)+'；可用 --compare-root 指定实际目录。']
    lines += ['', '旧TailRW16训练过A，旧E2额外训练过B，均不自动放入这个聚合单因素对照。',
        'stage_effects.csv 保留同状态同本地更新下与FedAvg的即时对照，它不是FedAvg完整训练轨迹。',
        'sample_dynamics.csv 区分初始能力保持、新增答对和曾答对后再次答错。',
        '耗时包含测试、诊断与I/O；局部并行试跑的速度比不等于全程加速比。']
    (out / 'report.md').write_text('\n'.join(lines)+'\n', encoding='utf-8')
    print('frozen', status['status'], 'round='+str(status['completed_round']), status['reason'])
    print('Report:', out / 'report.md')
    return [status]
