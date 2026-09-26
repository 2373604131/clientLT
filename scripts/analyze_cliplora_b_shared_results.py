"""Offline review of the archived shared-C pilot; never imports training code."""
import json
from collections import Counter
from pathlib import Path
from statistics import mean

from analyze_cliplora_frequency_groups import (
    ROOT, DISJOINT, read_csv, write_csv, runs, make_groups, markdown_table,
)


OUT = ROOT / 'output/sfra_b_shared_transfer_analysis/review'
ARCHIVE = ROOT / 'output/sfra_b_shared_transfer_analysis/sfra_b_shared_transfer'
NEW = ARCHIVE / 'seed42/client-longtail/full-cp/lambda10_mu1_protocol42_bshared_b_lr0.3_probe0.1_reg0.001_fast_v2_f64_c4'
KEYS = ('S', 'E3', 'J', 'Full10', 'CP1', 'CP3', 'FlatCP', 'B0.3', 'Uniform', 'UniformB')


def read_json(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def figures(summary, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'svg.fonttype': 'none', 'savefig.dpi': 200})
    styles = [('S', 'S: ordinary A/B', '#777777', ':', 'D'),
              ('CP1', 'A-only: Full-CP', '#0072B2', '--', 's'),
              ('B0.3', 'A + local-C B', '#E69F00', '-.', '^'),
              ('SharedB', 'A + shared-C B', '#009E73', '-', 'o')]
    fig, axes = plt.subplots(2, 2, figsize=(11, 7))
    for col, group in enumerate(('Overall', 'Tail20')):
        ax = axes[0, col]
        for method, label, color, ls, marker in styles:
            ax.plot(range(101), [curves[method][r][group] for r in range(101)],
                    label=label, color=color, ls=ls, marker=marker, markevery=20,
                    ms=4, lw=1.5)
        ax.set(title=f'{group}: full training trajectory', ylabel='Accuracy (%)',
               xlim=(0, 100), ylim=(50, 80))
        ax.axvline(30, color='#aaaaaa', lw=.8)
        ax = axes[1, col]
        for method, label, color, ls, marker in styles[2:]:
            ax.plot(range(101), [curves[method][r][group]-curves['CP1'][r][group]
                                for r in range(101)], label=label, color=color,
                    ls=ls, marker=marker, markevery=10, ms=4, lw=1.5)
        ax.axhline(0, color='#666666', lw=.8)
        for rnd in range(30, 101, 10):
            ax.axvline(rnd, color='#aaaaaa', lw=.5, alpha=.4)
        ax.set(title=f'{group}: difference from A-only', ylabel='Accuracy difference (pp)',
               xlabel='Completed federated round', xlim=(0, 100), ylim=(-1, 1))
    for ax in axes.flat:
        ax.grid(axis='y', alpha=.16)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc='lower center', bbox_to_anchor=(.5, .045),
               ncol=4, frameon=False, fontsize=9)
    fig.text(.5, .012, 'Shared-C produces small gains over A-only. Client-LT, seed42; raw curves, no smoothing.\n'
             'Bottom panels: paired differences; vertical guides mark transfer rounds. No across-seed uncertainty is available.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .115, 1, 1))
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'01_training_and_paired_differences.{ext}')
    plt.close(fig)

    groups = ['Overall', 'Many35', 'Medium35', 'Few30', 'Middle_Few10', 'Tail20']
    labels = ['Overall', 'Many35\n>100', 'Medium35\n20-100', 'Few30\n<20', 'Middle Few10\nIDs70-79', 'Tail20\nIDs80-99']
    baselines = ['CP1', 'B0.3', 'S']
    data = np.array([[summary['SharedB'][g]-summary[b][g] for g in groups] for b in baselines])
    fig, ax = plt.subplots(figsize=(10.5, 3.7))
    bound = max(abs(data.min()), abs(data.max()))
    im = ax.imshow(data, cmap='PuOr', vmin=-bound, vmax=bound, aspect='auto')
    ax.set_xticks(range(len(groups)), labels)
    ax.set_yticks(range(3), ['Shared-C minus A-only', 'Shared-C minus local-C', 'Shared-C minus S'])
    for i in range(3):
        for j in range(len(groups)):
            ax.text(j, i, f'{data[i,j]:+.3f}', ha='center', va='center',
                    color='white' if abs(data[i,j]) > .7*bound else '#111111')
    fig.colorbar(im, ax=ax, label='Accuracy difference (pp)', pad=.02)
    fig.text(.5, .018, 'The gain is concentrated in few-shot classes; Medium35 remains the main deficit.\n'
             'Client-LT, seed42; rounds 81-100 mean. Few30 includes Middle Few10 and Tail20; columns are not all disjoint.',
             ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .15, 1, 1))
    for ext in ('png', 'svg'):
        fig.savefig(OUT / f'02_frequency_group_differences.{ext}')
    plt.close(fig)


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    source = {k: (label, p) for k, label, p in runs()}
    source = {k: source[k] for k in KEYS}
    source['SharedB'] = ('CP1 + shared-C B lr=0.3', NEW)
    manifest = read_csv(NEW / 'partition_manifest.csv')
    counts = Counter(int(r['class_id']) for r in manifest)
    groups = make_groups(counts)
    summary, curves, raw = {}, {}, {}
    checks, all_curves, per_class, retention, sources = [], [], [], [], []
    for method, (label, path) in source.items():
        assert Counter(int(r['class_id']) for r in read_csv(path/'partition_manifest.csv')) == counts
        original = {int(r['round']): r for r in read_csv(path/'round_metrics.csv')}
        assert set(original) == set(range(101))
        values, series, error = {}, {}, 0.
        for rnd in range(101):
            rows = read_csv(path/f'per_class_accuracy_epoch_{rnd-1}.csv')
            values[rnd] = {int(r['class_id']): float(r['per_class_acc']) for r in rows}
            assert len(rows) == 100 and set(values[rnd]) == set(range(100))
            series[rnd] = {g: mean(values[rnd][c] for c in ids) for g, ids in groups.items()}
            for group, field in [('Overall', 'overall_acc'), ('Head20', 'head20_acc'),
                                 ('Middle60', 'middle60_acc'), ('Tail20', 'bottom20_tail_acc')]:
                if field in original[rnd]:
                    error = max(error, abs(series[rnd][group]-float(original[rnd][field])))
            all_curves.append(dict(method=method, round=rnd, **series[rnd]))
        assert error < 1e-8
        raw[method], curves[method] = values, series
        result = {g: mean(series[r][g] for r in range(81, 101)) for g in groups}
        summary[method] = dict(method=method, **result)
        checks.append(dict(method=method, rounds=len(values), per_class_records=10100,
                           max_original_metric_error_pp=error,
                           same_partition_manifest=read_csv(path/'partition_manifest.csv') == manifest))
        sources.append(dict(method=method, label=label, path=path.relative_to(ROOT).as_posix()))
        for c in range(100):
            per_class.append(dict(method=method, class_id=c, train_count=counts[c],
                                  last20=mean(values[r][c] for r in range(81, 101))))
        for g in groups:
            peak = max(range(1, 101), key=lambda r: series[r][g])
            retention.append(dict(method=method, group=g, last20=result[g], final=series[100][g],
                                  peak=series[peak][g], peak_round=peak,
                                  peak_to_final=series[peak][g]-series[100][g]))
    contrasts, class_changes, directions = [], [], []
    for baseline in KEYS:
        diff = {g: summary['SharedB'][g]-summary[baseline][g] for g in groups}
        contributions = {f'overall_contribution_{g}': diff[g]*len(groups[g])/100 for g in DISJOINT}
        assert abs(sum(contributions.values())-diff['Overall']) < 1e-10
        contrasts.append(dict(baseline=baseline, **{f'delta_{g}': d for g, d in diff.items()}, **contributions))
        class_delta = {c: mean(raw['SharedB'][r][c]-raw[baseline][r][c] for r in range(81, 101))
                       for c in range(100)}
        class_changes.extend(dict(baseline=baseline, class_id=c, train_count=counts[c], delta_pp=d)
                             for c, d in class_delta.items())
        for g, ids in groups.items():
            directions.append(dict(baseline=baseline, group=g,
                improved=sum(class_delta[c] > 1e-9 for c in ids),
                declined=sum(class_delta[c] < -1e-9 for c in ids),
                unchanged=sum(abs(class_delta[c]) <= 1e-9 for c in ids)))
    comparability = []
    for baseline in ('CP1', 'B0.3'):
        path = source[baseline][1]
        pre = [abs(raw['SharedB'][r][c]-raw[baseline][r][c]) for r in range(30) for c in range(100)]
        row = dict(baseline=baseline, pre_transfer_class_records=3000,
                   pre_transfer_different_records=sum(d > 1e-10 for d in pre),
                   pre_transfer_max_class_difference_pp=max(pre))
        for name, filename in [('schedule', 'protocol/full_schedule.json'),
                               ('witness', 'private_witness_manifest.json')]:
            row[f'same_{name}'] = read_json(NEW/filename) == read_json(path/filename)
        row['execution_version_note'] = 'Shared: fast_v2; baseline: original; equal predictions are not bitwise tensor equality'
        comparability.append(row)
    windows = []
    for baseline in ('CP1', 'B0.3'):
        for start in (30, 40, 50, 60, 70, 80, 90, 100):
            end = min(100, start+9)
            windows.append(dict(baseline=baseline, start_round=start, end_round=end,
                **{f'delta_{g}': mean(curves['SharedB'][r][g]-curves[baseline][r][g]
                                      for r in range(start, end+1)) for g in groups}))
    events = read_csv(ARCHIVE/'analysis/b_transfer_rounds.csv')
    receivers = read_csv(ARCHIVE/'analysis/b_transfer_receivers.csv')
    assert [int(r['round']) for r in events] == list(range(30, 101, 10))
    assert len(receivers) == 176
    event_rows = []
    for r in events:
        event_rows.append(dict(round=int(r['round']), donors=int(r['unique_donors']),
            accepted_links=int(r['donor_links']), candidate_pairs=int(r['candidate_pairs']),
            effective_update_ratio=float(r['effective_global_transfer_norm'])/float(r['ordinary_B_effective_update_norm']),
            tail_la_gain=float(r['tail_la_gain_after_B_aggregation']),
            non_tail_la_gain=float(r['ordinary_global_B_non_tail_la'])-float(r['transferred_global_B_non_tail_la']),
            tail_la_change_during_A=float(r['tail_la_change_during_A']), seconds=float(r['seconds'])))
    feedback = []
    for group in ('tail', 'non_tail'):
        for metric in ('la', 'accuracy', 'margin'):
            delta = [float(r[f'after_{group}_{metric}'])-float(r[f'before_{group}_{metric}']) for r in receivers]
            feedback.append(dict(group=group, metric=metric, convention='after minus before',
                                 records=len(delta), mean=mean(delta), minimum=min(delta), maximum=max(delta),
                                 positive=sum(d > 1e-10 for d in delta), negative=sum(d < -1e-10 for d in delta),
                                 zero=sum(abs(d) <= 1e-10 for d in delta)))
    completion = read_json(NEW/'completion.json')
    cost_fields = ['seconds', 'algorithm_forward_images', 'algorithm_backward_images',
                   'diagnostic_forward_images', 'extra_downlink_bytes', 'extra_upload_bytes',
                   'optimizer_steps', 'client_backward_batches', 'feedback_synchronizations']
    cost = {f: sum(float(r[f]) for r in events) for f in cost_fields}
    cost['whole_run_elapsed_seconds'] = completion['elapsed_seconds']
    cost['B_logged_seconds_fraction'] = cost['seconds']/completion['elapsed_seconds']
    # Reconstruct explicit LA coefficients, not gradient magnitudes, from the
    # original per-client group means and saved calibration sample positions.
    label_of = {(int(r['client_id']), int(r['local_position'])): int(r['class_id']) for r in manifest}
    objective_mass, calibration_images = Counter(), Counter()
    for rnd in range(30, 101, 10):
        saved = read_json(NEW/'b_transfer_rounds'/f'r{rnd:03d}'/'calibration_manifest.json')
        for client, info in saved['recipients'].items():
            for batch in info['batches']:
                parts = [batch['tail_positions']]
                if batch['non_tail_positions']:
                    parts.append(batch['non_tail_positions'])
                for part in parts:
                    for pos in part:
                        c = label_of[int(client), pos]
                        g = next(g for g in DISJOINT if c in groups[g])
                        objective_mass[g] += 1/(len(saved['recipients'])*len(parts)*len(part)*16)
                        calibration_images[g] += 1
    assert abs(sum(objective_mass.values())-1) < 1e-10
    objective_rows = [dict(group=g, number_of_classes=len(groups[g]),
                           calibration_image_occurrences=calibration_images[g],
                           mean_explicit_loss_weight_percent=100*objective_mass[g]) for g in DISJOINT]
    for filename, rows in [('performance_last20', list(summary.values())), ('contrasts', contrasts),
                           ('round_curves', all_curves), ('per_class_last20', per_class),
                           ('per_class_changes', class_changes), ('class_direction_counts', directions),
                           ('retention', retention), ('sources', sources), ('data_checks', checks),
                           ('comparability', comparability), ('post_transfer_windows', windows),
                           ('transfer_events', event_rows), ('receiver_changes', feedback), ('costs', [cost]),
                           ('calibration_objective_composition', objective_rows)]:
        write_csv(OUT/f'{filename}.csv', rows)
    fields = ['method', 'Overall', 'Head20', 'Middle60', 'Tail20', 'Many35', 'Medium35', 'Few30']
    report = ['# Shared-C B 离线复核统计', '',
        '结论见同目录 `分析结论.md`。这里只读取归档数据，未改训练代码、未重新训练。', '',
        '## 统一口径', '',
        'Client-LT，seed42；第81–100轮平均，对应epoch80–99。每类测试100张，Overall为100类宏平均。',
        'Many>100、Medium20–100、Few<20按全局训练样本数划分，实际分别为35/35/30类。',
        '只有单种子；20轮不是20个独立随机种子，不提供跨种子显著性结论。', '',
        '## 主表（%）', '', markdown_table(list(summary.values()), fields), '',
        '## 新版减去对照（百分点）', '',
        markdown_table(contrasts, ['baseline', 'delta_Overall', 'delta_Many35', 'delta_Medium35',
                                   'delta_Few30', 'delta_Middle_Few10', 'delta_Tail20']), '',
        '## 可比性', '', markdown_table(comparability, ['baseline', 'pre_transfer_different_records',
            'pre_transfer_max_class_difference_pp', 'same_schedule', 'same_witness']), '',
        '新实验使用fast_v2，旧A-only/旧B使用原执行路径。首次迁移前预测一致不等于所有权重逐位一致。', '',
        '## B事件', '', markdown_table(event_rows, ['round', 'donors', 'accepted_links',
            'effective_update_ratio', 'tail_la_gain', 'non_tail_la_gain', 'tail_la_change_during_A']), '',
        'effective_update_ratio为scaled BA有效权重改变量的Frobenius范数之比，不是参数B自身范数；不测方向。',
        '训练侧接收客户端等权统计不等于独立测试集逐类宏平均；probe与校准训练样本允许重叠。',
        '每步使用不同校准batch，不能把step1与step2的loss差直接当作同一批样本的优化收益。', '',
        '## 反馈变化：after-minus-before', '', markdown_table(feedback,
            ['group', 'metric', 'mean', 'positive', 'negative', 'zero']), '',
        'LA降低为好、accuracy/margin升高为好。accuracy单位为百分点，margin为余弦相似度差。', '',
        '## B校准目标实际覆盖', '', markdown_table(objective_rows,
            ['group', 'number_of_classes', 'calibration_image_occurrences', 'mean_explicit_loss_weight_percent']), '',
        '依据保存的sample positions逐张映射回训练类别，按客户端等权、tail/non-tail两组等权重构。',
        '上表是数据项中每类样本loss的显式系数，不是梯度贡献、优化结果或因果损失占比；不含C正则项。', '',
        '## 计算与通信', '',
        f'- 完整运行耗时 {cost["whole_run_elapsed_seconds"]/3600:.3f} 小时；B事件日志合计 {cost["seconds"]:.3f} 秒，占 {cost["B_logged_seconds_fraction"]*100:.3f}%。',
        '- B日志计时包含探测/校准/诊断，但不等于跨服务器同硬件的端到端消融；通信bytes为代码建模量，不是实测网络流量。',
        '- 16次全局Adam更新包含352次客户端反向batch，不是16个客户端batch。', '',
        '## 数据文件', '',
        '- per_class_changes.csv、class_direction_counts.csv：逐类收益与涨跌数量。',
        '- contrasts.csv：五个互斥类别组对Overall变化的精确贡献。',
        '- post_transfer_windows.csv：迁移后分段跨运行分差，不能解释为因果保持率。',
        '- retention.csv：每组峰值、最终值及差；新B与A-only的Tail20峰终差均为1.0。',
        '- data_checks.csv、comparability.csv、sources.csv：原始指标核对与来源。', '',
        '## 复现', '', '`python scripts/analyze_cliplora_b_shared_results.py`', '']
    (OUT/'statistics.md').write_text('\n'.join(report), encoding='utf-8')
    figures(summary, curves)
    print(markdown_table(list(summary.values()), fields))
    print('\nComparability:', comparability)
    print('\nFeedback:', feedback)
    print('\nCosts:', cost)
    print(f'\nWritten: {OUT}')


if __name__ == '__main__':
    main()
