"""Frozen contracts and offline reports for the four full-horizon A controls."""
import csv
import json
import math
from pathlib import Path

import numpy as np

SCHEMA = 'method_a_supplement_seed42_v1'
METHODS = ('shuffle-cp', 'current-cp', 'plain-hold', 'norm-matched')
REFERENCE_ROUNDS = (0, 58, 59, 60, 78, 79, 80, 90, 100)
RULES = {
    'shuffle-cp': 'Full-CP targets; active within-class weights cyclically permuted in a fixed random order',
    'current-cp': 'source current targets only; history neither raises targets nor activates tokens',
    'plain-hold': 'uniform all-token weights; T=max(post-B/pre-A margin, valid history); no source gain',
    'norm-matched': 'own ordinary direction; one scalar matches Full-CP effective norm each round 1..90',
}


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False)+'\n', encoding='utf-8')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for row in rows for k in row)))
        writer.writeheader()
        writer.writerows(rows)


def reference_state_path(source, rnd):
    source = Path(source)
    if rnd == 0:
        return source/'checkpoints/base_model.pt'
    if rnd <= 90:
        return source/'sfra_rounds'/f'r{rnd:03d}'/'commit.pt'
    return source/'events'/f'r{rnd:03d}_c000_main_normal_B'/'state.pt'


def validate_source(source):
    """Read configuration and file inventory. No hashes and no forward/training trial."""
    source = Path(source)
    cfg = read_json(source/'sfra_config.json')
    expected = dict(variant='full-cp', seed=42, protocol_seed=42, partition='client-longtail',
                    retention_weight=10., classification_weight=1., correction_steps=3,
                    correction_step_size=.1, witness_per_local_class=8, witness_views=2,
                    positive_response_threshold=1e-6, current_gain_fraction=.5,
                    history_window=5, history_improvement=.001, sigma_floor=.001,
                    radius='unweighted_client_proposal_RMS', precision='fp32',
                    commit_rule='fixed_third_step', refresh_rounds=list(range(1,91)))
    for key, value in expected.items():
        if cfg.get(key) != value:
            raise ValueError(f'Full-CP reference {key} must be {value!r}, found {cfg.get(key)!r}')
    if cfg.get('b_transfer') or cfg.get('b_aggregation'):
        raise ValueError('Reference must be A-only with ordinary sample-weighted B aggregation')
    if int(read_json(source/'completion.json')['completed_round']) != 100:
        raise ValueError('Full-CP reference must have completed 100 rounds')
    for name in ('command.json', 'partition_manifest.csv', 'bridge_metadata.json', 'private_witness_manifest.json',
                 'class_prior.json', 'protocol/full_schedule.json', 'protocol/eri_protocol.json',
                 'protocol/probe_manifest.csv'):
        if not (source/name).is_file():
            raise FileNotFoundError(source/name)
    missing = [str(reference_state_path(source, r)) for r in REFERENCE_ROUNDS
               if not reference_state_path(source, r).is_file()]
    if missing:
        raise FileNotFoundError('Saved reference models needed for prediction-only origin analysis: '+', '.join(missing))
    norms = {}
    for rnd in range(1,91):
        row = read_json(source/'sfra_rounds'/f'r{rnd:03d}'/'summary.json')
        value = float(row['committed_effective_norm'])
        if int(row['round']) != rnd or not math.isfinite(value) or value < 0:
            raise ValueError(f'Invalid reference norm at round {rnd}')
        norms[str(rnd)] = value
    return cfg, norms


def class_groups(counts, tail=None):
    counts = np.asarray(counts)
    tail = np.asarray(tail if tail is not None else sorted(range(len(counts)),
                      key=lambda c:(int(counts[c]), -c))[:20], dtype=int)
    groups = dict(Overall=np.arange(len(counts)), Many35=np.flatnonzero(counts > 100),
                  Medium35=np.flatnonzero((counts >= 20) & (counts <= 100)),
                  Few30=np.flatnonzero(counts < 20), Tail20=tail)
    if [len(v) for v in groups.values()] != [100, 35, 35, 30, 20]:
        raise ValueError('Unexpected fixed frequency groups')
    return groups


def prediction_metrics(arrays, groups, rnd):
    rows = []
    for name, classes in groups.items():
        selected = np.isin(arrays['class_id'], classes)
        rows.append(dict(round=rnd, group=name, n=int(selected.sum()),
            accuracy=float(arrays['correct'][selected].mean()*100),
            **{k:float(arrays[k][selected].mean()) for k in ('ce_loss', 'la_loss', 'margin')}))
    return rows


def load_prediction(path):
    with np.load(path, allow_pickle=False) as data:
        return {k:data[k] for k in data.files}


def require_same_samples(a, b):
    if not all(np.array_equal(a[k], b[k]) for k in ('sample_id', 'class_id')):
        raise ValueError('Prediction sample identities/order differ')


def metrics_from_run(run, groups):
    """Same table for old references and new runs; original per-class accuracies are percentages."""
    run = Path(run)
    metrics = read_csv(run/'round_metrics.csv')
    if sorted(int(r['round']) for r in metrics) != list(range(101)):
        raise ValueError(f'Missing/duplicate rounds 0..100 in {run}')
    rows = []
    for r in metrics:
        rnd = int(r['round'])
        per_class = read_csv(run/f'per_class_accuracy_epoch_{rnd-1}.csv')
        by_class = {int(x['class_id']):float(x['per_class_acc']) for x in per_class}
        if sorted(by_class) != list(range(100)) or len(per_class) != 100:
            raise ValueError(f'Incomplete per-class metrics at {run}, round {rnd}')
        for group, ids in groups.items():
            acc = float(r['overall_acc']) if group == 'Overall' else float(np.mean([by_class[c] for c in ids]))
            rows.append(dict(round=rnd, group=group, accuracy=acc))
    return rows


def origin_rows(predictions, groups, method):
    """Own stable-learned cohorts are descriptive, never called common-sample causal effects."""
    initial = predictions[0]
    rows = []
    for anchor in (60, 80):
        stable = np.ones(len(initial['correct']), dtype=bool)
        for rnd in range(anchor-2, anchor+1):
            require_same_samples(initial, predictions[rnd])
            stable &= predictions[rnd]['correct'].astype(bool)
        for cohort, mask in (('initially_correct_stable', stable & initial['correct']),
                             ('newly_learned_stable', stable & ~initial['correct']),
                             ('not_stable_at_anchor', ~stable)):
            for end in (anchor+5, 100):
                if end not in predictions:
                    continue
                require_same_samples(initial, predictions[end])
                for group, ids in groups.items():
                    selected = mask & np.isin(initial['class_id'], ids)
                    n = int(selected.sum())
                    correct = int(predictions[end]['correct'][selected].sum())
                    rows.append(dict(method=method, cohort_basis='own_trajectory_descriptive', anchor=anchor,
                        end_round=end, cohort=cohort, group=group, n=n, endpoint_correct=correct,
                        endpoint_wrong=n-correct, endpoint_accuracy=correct/n*100 if n else None))
    return rows


def common_short_origin(reference, short_roots, groups):
    """Reuse completed five-round branches on reference-defined, pre-intervention cohorts."""
    rows, missing = [], []
    initial = reference[0]
    for anchor in (60,80):
        for branch in ('full-cp', 'ordinary', 'norm-matched'):
            hits = sorted({p.resolve() for root in short_roots for p in Path(root).rglob(
                f'r{anchor+5:03d}_{branch}_committed.csv')})
            if len(hits) != 1:
                missing.append(dict(anchor=anchor, branch=branch, matching_files=len(hits)))
                continue
            endpoint_file = hits[0]
            start_file = endpoint_file.parent/f'r{anchor:03d}_{branch}_anchor.csv'
            if not start_file.is_file():
                # Some archives stored the common anchor only under full-cp.
                start_file = endpoint_file.parents[2]/'full-cp'/'predictions'/f'r{anchor:03d}_full-cp_anchor.csv'
            start, end = read_csv(start_file), read_csv(endpoint_file)
            convert = lambda records: dict(sample_id=np.array([int(x['sample_id']) for x in records]),
                class_id=np.array([int(x['class_id']) for x in records]),
                correct=np.array([bool(int(x['correct'])) for x in records]))
            a, b = convert(start), convert(end)
            require_same_samples(initial, a)
            require_same_samples(initial, b)
            if not np.array_equal(a['correct'], reference[anchor]['correct']):
                missing.append(dict(anchor=anchor, branch=branch, reason='anchor predictions differ',
                    differing_samples=int(np.count_nonzero(a['correct'] != reference[anchor]['correct'])),
                    source_file=str(start_file)))
                continue
            stable = np.logical_and.reduce([reference[r]['correct'] for r in range(anchor-2, anchor+1)])
            for cohort, mask in (('initially_correct_stable', stable & initial['correct']),
                                 ('newly_learned_stable', stable & ~initial['correct']),
                                 ('not_stable_at_anchor', ~stable)):
                for group, ids in groups.items():
                    selected = mask & np.isin(initial['class_id'], ids)
                    n = int(selected.sum())
                    correct = int(b['correct'][selected].sum())
                    rows.append(dict(anchor=anchor, branch=branch, cohort=cohort, group=group, n=n,
                        endpoint_correct=correct, endpoint_wrong=n-correct,
                        endpoint_accuracy=correct/n*100 if n else None,
                        cohort_basis='common_FullCP_pre_intervention_3round_stable', source_file=str(endpoint_file)))
    return rows, missing


def baseline_compatibility(reference, baseline):
    """Compare explicit protocol/settings; a fast/standard difference is not declared equivalent."""
    issues = []
    for name in ('partition_manifest.csv',):
        if read_csv(reference/name) != read_csv(baseline/name):
            issues.append(name+' differs')
    for name in ('class_prior.json', 'protocol/full_schedule.json'):
        if read_json(reference/name) != read_json(baseline/name):
            issues.append(name+' differs')
    for name in ('execution_config.json',):
        a = read_json(reference/name) if (reference/name).is_file() else None
        b = read_json(baseline/name) if (baseline/name).is_file() else None
        if a != b:
            issues.append('execution mode differs; numerical equivalence not established by this report')
    a = read_json(reference/'sfra_config.json')
    b = (read_json(baseline/'sfra_config.json') if (baseline/'sfra_config.json').is_file()
         else dict(variant=read_json(baseline/'control_config.json')['method'],
                   base_training_config=read_json(baseline/'control_config.json')))
    for key in ('method', 'seed', 'protocol_seed', 'topology', 'la_tau', 'a_lr_mult', 'extra_a_lr',
                'normal_b_lr', 'normal_b_weight_decay', 'extra_weight_decay', 'aggregation',
                'control_enabled', 'extra_trainable_factor', 'normal_trainable_factor', 'candidate_rounds',
                'rng_protocol', 'normal_steps_expected', 'extra_steps_expected'):
        if a['base_training_config'].get(key) != b['base_training_config'].get(key):
            issues.append('training '+key+' differs')
    # In older S manifests beta was not recorded in control_config. The exact
    # partition above, plus declared model/training flags below, are the relevant contract.
    commands = [read_json(p/'command.json') for p in (reference, baseline)]
    for flag in ('--seed', '--split_seed', '--cliplora_common_init_seed', '--cliplora_rank',
                 '--cliplora_alpha', '--cliplora_precision', '--encoder', '--cliplora_position',
                 '--train_batch_size', '--local_epochs', '--round', '--cliplora_lr_policy', '--lr'):
        values = [cmd[cmd.index(flag)+1] if flag in cmd else None for cmd in commands]
        if values[0] != values[1]:
            issues.append(flag+' differs')
    if b.get('b_transfer') or b.get('b_aggregation'):
        issues.append('B configuration differs')
    if b.get('variant') == 'flat-cp':
        for key in ('retention_weight', 'classification_weight', 'witness_batch_size',
                    'correction_steps', 'correction_step_size', 'history_window', 'history_improvement'):
            if a.get(key) != b.get(key):
                issues.append(key+' differs')
        if read_json(reference/'private_witness_manifest.json') != read_json(baseline/'private_witness_manifest.json'):
            issues.append('witness identities differ')
    return issues


def discover_inputs(root, explicit_baselines, short_roots):
    """Known prior run locations only, in fixed order; never choose a run by accuracy."""
    repo = Path(__file__).resolve().parents[2]
    baselines = dict(explicit_baselines or {})
    suffix_s = Path('la_control/seed42/client-longtail/s/tau1_a1_protocol42')
    suffix_flat = Path('sfra_cp/seed42/client-longtail/flat-cp/lambda10_mu1_protocol42_fast')
    candidates = {
        's': [root/'references/s', repo/'output/cifar100_LT'/suffix_s,
              repo/'output/la_control_js_topology_analysis'/suffix_s],
        'flat-cp': [root/'references/flat-cp', repo/'output/cifar100_LT'/suffix_flat,
                    repo/'output/sfra_cp_analysis_new'/suffix_flat],
    }
    for name, paths in candidates.items():
        if name not in baselines:
            found = next((p for p in paths if (p/'completion.json').is_file()), None)
            if found is not None:
                baselines[name] = found
    if not short_roots:
        short_roots = []
        for anchor in (60,80):
            name = f'method_a_diagnostics_a{anchor}_fast_v2_f128_c4'
            found = next((p for p in (root/'short_inputs'/name, repo/'output'/name,
                                      repo/'output/method_a_results_a60_a80'/name)
                          if (p/f'anchor{anchor}/full-cp/diagnostic_completion.json').is_file()), None)
            if found is not None:
                short_roots.append(found)
    return baselines, short_roots


def witness_harm_rows(run, method, groups):
    """Macro over classes after averaging client-class tokens; not official test accuracy."""
    manifest = run/'private_witness_manifest.json'
    if not manifest.is_file():
        return []
    labels = np.array([t['class_id'] for t in read_json(manifest)])
    rows = []
    for rnd in range(1,91):
        path = run/'sfra_rounds'/f'r{rnd:03d}'/'tokens.npz'
        if not path.is_file():
            continue
        with np.load(path, allow_pickle=False) as data:
            if 'classification_proposal_scores' not in data:
                continue
            loss0 = data['classification_proposal_scores'].mean(1)
            loss1 = data['classification_committed_scores'].mean(1)
            before, proposed, after = [data[k].mean(1) for k in ('F_post_B', 'F_proposal', 'F_committed')]
            for group, ids in groups.items():
                average = lambda v: float(np.mean([v[labels==c].mean() for c in ids]))
                rows.append(dict(method=method, round=rnd, group=group,
                    proposal_la=average(loss0), committed_la=average(loss1), la_harm=average(loss1-loss0),
                    ordinary_margin_gain=average(proposed-before), committed_margin_gain=average(after-before),
                    correction_margin_change=average(after-proposed),
                    harmed_token_fraction_class_macro=average((loss1 > loss0).astype(float)),
                    averaging='equal classes, equal client-class units within class, two-view mean'))
    return rows


def summarize(root, baselines=None, short_roots=()):
    root = Path(root)
    baselines, short_roots = discover_inputs(root, baselines, short_roots)
    jobs = {m:read_json(root/m/'supplement_job.json') for m in METHODS if (root/m/'supplement_job.json').is_file()}
    if not jobs:
        raise ValueError('No registered supplement jobs under '+str(root))
    registered_source = Path(next(iter(jobs.values()))['source_run'])
    if any(Path(j['source_run']) != registered_source for j in jobs.values()):
        raise ValueError('Experiments use different Full-CP references')
    source = registered_source if registered_source.is_dir() else root/'references/full-cp'
    prior = read_json(source/'class_prior.json')
    groups = class_groups(prior['counts'], prior['tail_ids'])
    missing = [m for m in METHODS if not (root/m/'completion.json').is_file()]
    runs = {'full-cp': source, **{m:root/m for m in METHODS if m not in missing}}
    supplement_compatibility = []
    for method in METHODS:
        if method in missing:
            continue
        run = root/method
        issues = baseline_compatibility(source, run)
        supplement_compatibility.append(dict(method=method, run=str(run),
            strictly_comparable=not issues, issues=issues,
            reference_execution=jobs[method]['reference_execution'],
            actual_execution=(read_json(run/'execution_config.json')
                              if (run/'execution_config.json').is_file() else None)))
    compatibility = []
    for name, path in (baselines or {}).items():
        if name not in ('s', 'flat-cp'):
            raise ValueError('Optional baselines are named s or flat-cp')
        path = Path(path)
        declared = (read_json(path/'sfra_config.json').get('variant') if (path/'sfra_config.json').is_file()
                    else read_json(path/'control_config.json').get('method'))
        if declared != name:
            raise ValueError(f'Baseline {name} points to another method: {path}')
        issues = baseline_compatibility(source, path)
        compatibility.append(dict(method=name, run=str(path), strictly_comparable=not issues, issues=issues))
        runs[name] = path
    curves, results, differences, costs, origins, learning, witness = [], [], [], [], [], [], []
    for name, run in runs.items():
        rows = metrics_from_run(run, groups)
        witness.extend(witness_harm_rows(run, name, groups))
        curves.extend(dict(method=name, **r) for r in rows)
        for group in groups:
            series = {r['round']:r['accuracy'] for r in rows if r['group']==group}
            results.append(dict(method=name, group=group,
                mean81_100=float(np.mean([series[r] for r in range(81,101)])), final100=series[100],
                round90=series[90], change90_100=series[100]-series[90],
                peak_drop_to100=max(series[r] for r in range(1,101))-series[100]))
        progress = read_json(run/'completion.json')
        feedback = read_csv(run/'sfra_costs.csv') if (run/'sfra_costs.csv').is_file() else []
        budget = read_csv(run/'budget.csv')
        number = lambda r,k: float(r.get(k) or 0)
        costs.append(dict(method=name, elapsed_seconds=progress.get('elapsed_seconds'),
            local_optimizer_steps=sum(number(r,'optimizer_steps') for r in budget),
            functional_correction_steps=progress.get('functional_correction_steps', 0),
            feedback_forward_images=sum(number(r,'forward_images') for r in feedback),
            feedback_backward_images=sum(number(r,'backward_images') for r in feedback),
            feedback_upload_bytes=sum(number(r,'upload_bytes') for r in feedback),
            feedback_downlink_bytes=sum(number(r,'downlink_bytes') for r in feedback),
            dependency='requires Full-CP norm trajectory' if name=='norm-matched' else 'none'))
        if name in METHODS:
            predictions = {r:load_prediction(run/'predictions'/f'r{r:03d}.npz') for r in range(101)}
            origins.extend(origin_rows(predictions, groups, name))
            initial = predictions[0]
            for rnd in range(1,101):
                a, b = predictions[rnd-1], predictions[rnd]
                require_same_samples(initial, b)
                for group, ids in groups.items():
                    selected = np.isin(b['class_id'], ids)
                    lost = int((a['correct'] & ~b['correct'] & selected).sum())
                    gained = int((~a['correct'] & b['correct'] & selected).sum())
                    old = selected & initial['correct']
                    new = selected & ~initial['correct']
                    learning.append(dict(method=name, round=rnd, group=group, correct_to_wrong=lost,
                        wrong_to_correct=gained, net_correct_change=gained-lost,
                        initial_correct_n=int(old.sum()), initial_correct_retained=int((old & b['correct']).sum()),
                        initial_wrong_n=int(new.sum()), initial_wrong_now_correct=int((new & b['correct']).sum())))
    for group in groups:
        full = next(r for r in results if r['method']=='full-cp' and r['group']==group)
        for r in results:
            if r['method'] != 'full-cp' and r['group']==group:
                eligible = next((x['strictly_comparable'] for x in compatibility + supplement_compatibility
                                 if x['method']==r['method']), True)
                differences.append(dict(comparison='full-cp minus '+r['method'], group=group,
                    strictly_comparable=eligible, **{k:full[k]-r[k] for k in ('mean81_100', 'final100', 'change90_100')}))
    reference_dir = root/'norm-matched'/'reference_eval'
    reference_ready = all((reference_dir/'predictions'/f'r{r:03d}.npz').is_file() for r in REFERENCE_ROUNDS)
    short_rows, short_missing = [], []
    if reference_ready:
        reference = {r:load_prediction(reference_dir/'predictions'/f'r{r:03d}.npz') for r in REFERENCE_ROUNDS}
        origins.extend(origin_rows(reference, groups, 'full-cp'))
        if short_roots:
            short_rows, short_missing = common_short_origin(reference, short_roots, groups)
    audit = []
    for method in METHODS:
        if method in missing:
            continue
        for rnd in range(1,91):
            row = read_json(root/method/'sfra_rounds'/f'r{rnd:03d}'/'summary.json')
            audit.append(dict(method=method, **row))
    out = root/'analysis'
    for name, rows in [('metrics_100rounds', curves), ('last20_summary', results), ('full_minus_controls', differences),
                       ('costs', costs), ('origin_own_trajectory', origins), ('learning_and_forgetting', learning),
                       ('origin_common_short_windows', short_rows), ('control_diagnostics', audit),
                       ('witness_classification_harm', witness)]:
        write_csv(out/(name+'.csv'), rows)
    status = dict(four_training_runs_complete=not missing, missing_experiments=missing,
                  reference_origin_predictions_ready=reference_ready,
                  common_short_origin_complete=bool(short_rows) and not short_missing,
                  common_short_missing=short_missing, baseline_compatibility=compatibility,
                  supplement_compatibility=supplement_compatibility,
                  resolved_baselines={k:str(v) for k,v in baselines.items()},
                  resolved_short_roots=[str(v) for v in short_roots],
                  missing_optional_baselines=[m for m in ('s', 'flat-cp') if m not in runs],
                  scope='One training seed; comparisons describe these fixed controls, not significance or all alternatives')
    write_json(out/'suite_status.json', status)
    report = ['# 方法A：seed42四组补充实验', '',
        '主指标固定为81—100轮均值；以下结果不按测试峰值选模型。', '',
        '| 方法 | 指标 | 后20轮均值 | 第100轮 | 90→100变化 |',
        '|---|---|---:|---:|---:|']
    for r in results:
        if r['group'] in ('Overall', 'Medium35', 'Tail20'):
            historical = any(x['method']==r['method'] and not x['strictly_comparable'] for x in compatibility)
            label = r['method'] + ('（历史参考）' if historical else '')
            if any(x['method']==r['method'] and not x['strictly_comparable'] for x in supplement_compatibility):
                label += '（与Full配置有差异，见核对记录）'
            report.append(f"| {label} | {r['group']} | {r['mean81_100']:.3f} | {r['final100']:.3f} | {r['change90_100']:+.3f} |")
    if compatibility:
        report += ['', '旧基线配置核对：', '']
        for item in compatibility:
            report.append('- '+item['method']+'：'+('; '.join(item['issues']) if item['issues'] else '已核对的配置字段一致。'))
    if supplement_compatibility:
        report += ['', '四组与Full的执行及训练配置核对：', '']
        for item in supplement_compatibility:
            report.append('- '+item['method']+'：'+('; '.join(item['issues']) if item['issues'] else '已核对的配置字段一致。'))
        report.append('若启用v2而旧Full使用其他执行设置，算法控制定义保留，但本报告不声称已验证数值等价。')
    report += ['', '比较口径：', '',
        '- Full对Shuffle：只解释同类不同客户端之间正确权重对应的作用；先看实际改变比例。',
        '- Full对Current：历史目标的条件作用；记录影子历史不等于用它训练。',
        '- Full对Plain：来源目标和权重整体相对普通保持的增量，不等于所有蒸馏方法比较。',
        '- Full对Norm：相同有效幅度序列下方向修正的增量；Norm借用了Full的训练记录。',
        '- 同时查看Tail、Medium、Overall和学习/遗忘；接近或更差也是答案，不自动追加实验。',
        '- own_trajectory各自学会的样本不同，仅作描述；common_short_windows才是干预前固定的共同队列。',
        '- 未提供旧短程结果时，仍已保存来源分析所需模型预测，但共同短程队列表尚未生成。',
        '- 单seed不能证明稳定性；缺失或执行版本不同的Flat不能当作严格配对证据。', '',
        '运行状态见 suite_status.json；四组训练完整和所有证据完整分别报告。']
    (out/'report.md').write_text('\n'.join(report)+'\n', encoding='utf-8')
    return status
