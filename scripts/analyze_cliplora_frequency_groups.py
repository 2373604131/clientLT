"""Offline frequency regrouping of archived Client-LT per-class accuracies.

Reads existing CSVs only. No training imports, data repartition, or target changes.
Round r is stored in per_class_accuracy_epoch_{r-1}.csv.
"""
import argparse
import csv
import json
from collections import Counter
from pathlib import Path
from statistics import mean


ROOT = Path(__file__).resolve().parents[1]
DISJOINT = ('Head20', 'Middle_Many15', 'Middle_Medium35', 'Middle_Few10', 'Tail20')


def read_csv(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def runs():
    base = 'output/la_control_ablation_analysis/la_control/seed42/client-longtail'
    items = []
    for key, label, folder in [
        ('E0', 'E0: no LA, fixed A', 'e0/tau0_a1_protocol42'),
        ('E1', 'E1: no LA, nine A refreshes', 'e1/tau0_a1_protocol42'),
        ('E2', 'E2: LA, fixed A', 'e2/tau1_a1_protocol42'),
        ('E3', 'E3: LA, nine A refreshes', 'e3/tau1_a1_protocol42'),
        ('E5', 'E5: functional control', 'e5/tau1_a1_h2_t0.002_hist0.002_gain0.0001_p2_protocol42'),
        ('J', 'J: joint AB', 'j/tau1_a1_protocol42'),
        ('S', 'S: dense alternating A/B', 's/tau1_a1_protocol42'),
    ]:
        items.append((key, label, ROOT / base / folder))
    base = 'output/sfra_v1_analysis/sfra_v1/seed42/client-longtail'
    for weight in (1, 3, 10):
        items.append((f'Full{weight}', f'Full lambda={weight}', ROOT / base / f'full/lambda{weight}_protocol42'))
    for key in ('current', 'flat'):
        items.append((key.title(), f'{key.title()} lambda=10 (no CP)', ROOT / base / f'{key}/lambda10_protocol42'))
    base = 'output/sfra_cp_analysis/sfra_cp/seed42/client-longtail/full-cp'
    for weight in ('0.3', '1', '3'):
        items.append((f'CP{weight}', f'Full-CP mu={weight}', ROOT / base / f'lambda10_mu{weight}_protocol42'))
    items.append(('FlatCP', 'Flat-CP mu=1', ROOT / 'output/sfra_cp_analysis_new/sfra_cp/seed42/client-longtail/flat-cp/lambda10_mu1_protocol42_fast'))
    base = 'output/sfra_b_transfer_run2_analysis/sfra_b_transfer_run2/seed42/client-longtail/full-cp'
    for lr in ('0.03', '0.1', '0.3'):
        items.append((f'B{lr}', f'CP1 + local B lr={lr}', ROOT / base / f'lambda10_mu1_protocol42_b_lr{lr}_probe0.1_reg0.001'))
    base = 'output/sfra_b_aggregation_analysis/sfra_b_aggregation/seed42/client-longtail/full-cp'
    items.append(('Uniform', 'CP1 + uniform8, no B', ROOT / base / 'lambda10_mu1_protocol42_bagg_uniform8_fast'))
    items.append(('UniformB', 'CP1 + uniform8 + local B lr=0.3', ROOT / base / 'lambda10_mu1_protocol42_b_lr0.3_probe0.1_reg0.001_bagg_uniform8_fast'))
    return items


def make_groups(counts):
    classes = sorted(counts)
    frequency = {
        'Many35': [c for c in classes if counts[c] > 100],
        'Medium35': [c for c in classes if 20 <= counts[c] <= 100],
        'Few30': [c for c in classes if counts[c] < 20],
    }
    head, middle, tail = classes[:20], classes[20:80], classes[80:]
    groups = {'Overall': classes, **frequency, 'Head20': head, 'Middle60': middle, 'Tail20': tail}
    groups.update({
        'Middle_Many15': sorted(set(middle) & set(frequency['Many35'])),
        'Middle_Medium35': sorted(set(middle) & set(frequency['Medium35'])),
        'Middle_Few10': sorted(set(middle) & set(frequency['Few30'])),
    })
    assert [len(frequency[g]) for g in frequency] == [35, 35, 30]
    assert [len(groups[g]) for g in DISJOINT] == [20, 15, 35, 10, 20]
    return groups


def markdown_table(rows, fields):
    lines = ['| ' + ' | '.join(fields) + ' |', '| ' + ' | '.join('---' for _ in fields) + ' |']
    for row in rows:
        lines.append('| ' + ' | '.join(f'{row[k]:.3f}' if isinstance(row[k], float) else str(row[k]) for k in fields) + ' |')
    return '\n'.join(lines)


def plot_results(out, groups, summary, curves):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10, 'axes.spines.top': False,
                         'axes.spines.right': False, 'svg.fonttype': 'none', 'savefig.dpi': 190})
    labels = {'Full10': 'Full, lambda=10', 'CP1': 'Full-CP, mu=1', 'CP3': 'Full-CP, mu=3',
              'FlatCP': 'Flat-CP, mu=1', 'B0.3': 'CP1 + old B, lr=0.3', 'S': 'S', 'E3': 'E3', 'J': 'J'}
    axis_labels = ['Head20\n20 classes', 'Middle: Many\n15 classes', 'Middle: Medium\n35 classes',
                   'Middle: Few\n10 classes', 'Original Tail20\n20 classes']
    colors = ['#0072B2', '#56B4E9', '#E69F00', '#CC79A7', '#009E73']
    methods = ['Full10', 'CP1', 'CP3', 'FlatCP', 'B0.3']
    fig, ax = plt.subplots(figsize=(10.5, 4.6))
    values = np.array([[summary[m][g] - summary['S'][g] for g in DISJOINT] for m in methods])
    limit = max(abs(values.min()), abs(values.max()))
    im = ax.imshow(values, cmap='PuOr', vmin=-limit, vmax=limit, aspect='auto')
    ax.set_xticks(range(5), axis_labels)
    ax.set_yticks(range(len(methods)), [labels[m] for m in methods])
    for i in range(len(methods)):
        for j in range(5):
            ax.text(j, i, f'{values[i,j]:+.3f}', ha='center', va='center',
                    color='white' if abs(values[i,j]) > .65*limit else '#111111')
    fig.colorbar(im, ax=ax, label='Accuracy change vs S (percentage points)', pad=.02)
    ax.set_title('Where does accuracy change? Split the original Middle60', pad=14)
    fig.text(.5, .015, 'Client-LT, seed42; mean of rounds 81-100. Global training-count groups; no retraining.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .055, 1, 1))
    for ext in ('png', 'svg'):
        fig.savefig(out / f'01_disjoint_group_changes.{ext}')
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10.5, 4.9))
    y = np.arange(len(methods))
    positive, negative = np.zeros(len(methods)), np.zeros(len(methods))
    for group, color, label in zip(DISJOINT, colors, axis_labels):
        v = np.array([(summary[m][group]-summary['S'][group])*len(groups[group])/100 for m in methods])
        left = np.where(v >= 0, positive, negative)
        ax.barh(y, v, left=left, color=color, edgecolor='white', label=label.replace('\n', ': '), height=.63)
        for i in range(len(methods)):
            if abs(v[i]) >= .045:
                ax.text(left[i]+v[i]/2, i, f'{v[i]:+.3f}', ha='center', va='center', fontsize=9,
                        color='white' if color in ('#0072B2', '#009E73') else '#111111')
        positive += np.maximum(v, 0)
        negative += np.minimum(v, 0)
    net = np.array([summary[m]['Overall'] - summary['S']['Overall'] for m in methods])
    # Place net markers just below each bar, leaving segment values unobscured.
    ax.scatter(net, y+.38, color='black', marker='D', s=30, zorder=5, label='Net Overall change')
    ax.set_yticks(y, [f'{labels[m]}\nnet {net[i]:+.3f}' for i, m in enumerate(methods)])
    ax.invert_yaxis()
    ax.axvline(0, color='#666666', linewidth=.8)
    ax.set_xlabel('Contribution to Overall change vs S (percentage points)')
    ax.set_title('Overall trade-off: class-count-weighted contributions', pad=13)
    ax.legend(loc='upper center', bbox_to_anchor=(.45, -.18), ncol=3, frameon=False, fontsize=8)
    fig.text(.5, .015, 'Each contribution = group accuracy change x its fraction of 100 equally tested classes. Seed42, rounds 81-100.', ha='center', fontsize=8.5)
    fig.tight_layout(rect=(0, .055, 1, 1))
    for ext in ('png', 'svg'):
        fig.savefig(out / f'02_overall_contributions.{ext}')
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(10.5, 6.7), sharex=True)
    series = [('S', '#666666', '-', 'o'), ('E3', '#0072B2', '--', 's'),
              ('CP1', '#D55E00', '-.', '^'), ('J', '#009E73', ':', 'D')]
    for ax, group, title in zip(axes.flat,
                               ['Middle_Many15', 'Middle_Medium35', 'Middle_Few10', 'Tail20'],
                               ['Middle60: Many15 (>100)', 'Middle60: Medium35 (20-100)',
                                'Middle60: Few10 (<20)', 'Original Tail20 (also <20)']):
        for method, color, ls, marker in series:
            ax.plot(range(101), [curves[method][r][group] for r in range(101)], label=labels[method],
                    color=color, linestyle=ls, marker=marker, markevery=20, markersize=4, linewidth=1.5)
        ax.set_title(title, fontsize=11)
        ax.set_ylabel('Accuracy (%)')
        ax.grid(axis='y', alpha=.18)
        ax.set_xlim(0, 100)
        ax.set_ylim(45, 85)
    for ax in axes[-1]:
        ax.set_xlabel('Completed federated round')
    handles, legend_labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, legend_labels, loc='lower center', ncol=4, bbox_to_anchor=(.5, .045), frameon=False)
    fig.text(.5, .013, 'Client-LT, one seed (42); raw per-round accuracy, no smoothing or checkpoint selection.', ha='center', fontsize=9)
    fig.tight_layout(rect=(0, .10, 1, 1))
    for ext in ('png', 'svg'):
        fig.savefig(out / f'03_middle_and_tail_trajectories.{ext}')
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=ROOT / 'output/many_medium_few_reanalysis_20260925')
    parser.add_argument('--no-plots', action='store_true')
    args = parser.parse_args()
    out = args.output_dir.resolve()
    out.mkdir(parents=True, exist_ok=True)
    sources = runs()
    reference = next(path for key, _, path in sources if key == 'CP1')
    manifest = read_csv(reference / 'partition_manifest.csv')
    counts = Counter(int(r['class_id']) for r in manifest)
    assert set(counts) == set(range(100))
    groups = make_groups(counts)
    summary, curves, by_class = {}, {}, {}
    all_curves, per_class, checks, source_rows, retention = [], [], [], [], []
    original_fields = {'Overall': 'overall_acc', 'Head20': 'head20_acc', 'Middle60': 'middle60_acc', 'Tail20': 'bottom20_tail_acc'}
    for method, label, path in sources:
        local_counts = Counter(int(r['class_id']) for r in read_csv(path / 'partition_manifest.csv'))
        assert local_counts == counts, f'Global training histogram mismatch: {path}'
        metrics = {int(row['round']): row for row in read_csv(path / 'round_metrics.csv')}
        assert set(metrics) == set(range(101)), f'Missing/extra rounds: {path}'
        local_curves, raw = {}, {}
        max_error = 0.
        for rnd in range(101):
            epoch = rnd - 1
            rows = read_csv(path / f'per_class_accuracy_epoch_{epoch}.csv')
            values = {int(r['class_id']): float(r['per_class_acc']) for r in rows}
            assert len(rows) == 100 and set(values) == set(range(100))
            assert all(0 <= value <= 100 for value in values.values())
            raw[rnd] = values
            local_curves[rnd] = {g: mean(values[c] for c in ids) for g, ids in groups.items()}
            # Older E2/E3 exports contain Overall/Non-tail/Tail, without Head/Middle.
            for group, field in original_fields.items():
                if field not in metrics[rnd]:
                    continue
                max_error = max(max_error, abs(local_curves[rnd][group] - float(metrics[rnd][field])))
            max_error = max(max_error, abs(mean(values[c] for c in range(80)) - float(metrics[rnd]['non_tail_acc'])))
            all_curves.append({'method': method, 'round': rnd, **local_curves[rnd]})
        assert max_error < 1e-8, f'Original-metric cross-check failed: {path}, {max_error}'
        last = {c: mean(raw[r][c] for r in range(81, 101)) for c in range(100)}
        result = {g: mean(last[c] for c in ids) for g, ids in groups.items()}
        residual = abs(result['Overall'] - sum(result[g]*len(groups[g])/100 for g in DISJOINT))
        assert residual < 1e-10
        summary[method] = {'method': method, 'label': label, **result}
        curves[method], by_class[method] = local_curves, last
        checks.append({'method': method, 'rounds': len(metrics), 'per_class_files': len(raw),
                       'class_records': 100*len(raw), 'same_global_train_counts': True,
                       'checked_original_fields': ','.join([f for f in original_fields.values() if f in metrics[100]] + ['non_tail_acc']),
                       'max_original_metric_error_pp': max_error, 'overall_decomposition_error_pp': residual})
        source_rows.append({'method': method, 'label': label, 'run_directory': path.relative_to(ROOT).as_posix(),
                            'seed': 42, 'partition': 'client-longtail', 'aggregation_window': 'rounds 81-100 / epochs 80-99'})
        for c in range(100):
            per_class.append({'method': method, 'class_id': c, 'global_train_count': counts[c],
                              'frequency_group': next(g for g in ('Many35', 'Medium35', 'Few30') if c in groups[g]),
                              'old_group': next(g for g in ('Head20', 'Middle60', 'Tail20') if c in groups[g]),
                              'last20_accuracy': last[c], 'initial_accuracy': raw[0][c], 'final_accuracy': raw[100][c]})
        for g in groups:
            peak_round = max(range(1, 101), key=lambda r: local_curves[r][g])
            retention.append({'method': method, 'group': g, 'initial': local_curves[0][g],
                              'last20': result[g], 'final': local_curves[100][g],
                              'peak': local_curves[peak_round][g], 'peak_round': peak_round,
                              'peak_to_final': local_curves[peak_round][g]-local_curves[100][g],
                              'round90_to100': local_curves[100][g]-local_curves[90][g]})
    group_rows = []
    for name, ids in groups.items():
        group_rows.append({'group': name, 'number_of_classes': len(ids), 'class_ids_zero_based': ','.join(map(str, ids)),
                           'training_samples': sum(counts[c] for c in ids), 'min_train_count': min(counts[c] for c in ids),
                           'max_train_count': max(counts[c] for c in ids), 'overall_class_weight': len(ids)/100})
    pairs = [(method, 'S') for method in summary if method != 'S']
    pairs += [('CP0.3', 'Full10'), ('CP1', 'Full10'), ('CP3', 'Full10'), ('CP3', 'CP1'),
              ('CP1', 'FlatCP'), ('B0.03', 'CP1'), ('B0.1', 'CP1'), ('B0.3', 'CP1'),
              ('Uniform', 'CP1'), ('UniformB', 'Uniform'), ('UniformB', 'B0.3')]
    contrasts, changes, directional_counts = [], [], []
    for method, baseline in pairs:
        row = {'method': method, 'baseline': baseline}
        row.update({f'delta_{g}': summary[method][g]-summary[baseline][g] for g in groups})
        for g in DISJOINT:
            row[f'overall_contribution_{g}'] = row[f'delta_{g}']*len(groups[g])/100
        assert abs(sum(row[f'overall_contribution_{g}'] for g in DISJOINT)-row['delta_Overall']) < 1e-10
        middle_groups = DISJOINT[1:4]
        middle_net = sum(row[f'overall_contribution_{g}'] for g in middle_groups)
        row['middle_net_overall_contribution'] = middle_net
        row['middle_few10_share_of_net_middle_loss_percent'] = (
            100*row['overall_contribution_Middle_Few10']/middle_net if middle_net < -1e-10 else '')
        gross_loss = sum(max(0, -row[f'overall_contribution_{g}']) for g in middle_groups)
        row['middle_few10_share_of_gross_middle_loss_percent'] = (
            100*max(0, -row['overall_contribution_Middle_Few10'])/gross_loss if gross_loss else '')
        contrasts.append(row)
        for group, ids in groups.items():
            deltas = [by_class[method][c]-by_class[baseline][c] for c in ids]
            directional_counts.append({'method': method, 'baseline': baseline, 'group': group,
                                       'improved_classes': sum(d > 1e-9 for d in deltas),
                                       'declined_classes': sum(d < -1e-9 for d in deltas),
                                       'unchanged_classes': sum(abs(d) <= 1e-9 for d in deltas)})
        for c in range(100):
            changes.append({'method': method, 'baseline': baseline, 'class_id': c, 'global_train_count': counts[c],
                            'old_group': next(g for g in ('Head20', 'Middle60', 'Tail20') if c in groups[g]),
                            'frequency_group': next(g for g in ('Many35', 'Medium35', 'Few30') if c in groups[g]),
                            'delta_accuracy_pp': by_class[method][c]-by_class[baseline][c],
                            'overall_contribution_pp': (by_class[method][c]-by_class[baseline][c])/100})
    for filename, rows in [('group_definitions', group_rows), ('performance_last20', list(summary.values())),
                           ('contrasts', contrasts), ('per_class_last20', per_class), ('per_class_contrasts', changes),
                           ('round_curves', all_curves), ('retention', retention), ('data_checks', checks), ('sources', source_rows),
                           ('class_direction_counts', directional_counts)]:
        write_csv(out / f'{filename}.csv', rows)
    frequency_fields = ['method', 'Overall', 'Many35', 'Medium35', 'Few30', 'Head20', 'Middle60', 'Tail20']
    split_fields = ['method', 'Middle60', 'Middle_Many15', 'Middle_Medium35', 'Middle_Few10', 'Tail20']
    report = [
        '# Many / Medium / Few 离线重分组结果', '',
        '只读取既有 Client-LT、seed42 结果；未训练、未重新划分数据、未扩大 B 目标类别。', '',
        '研究结论及建议见同目录 [分析结论.md](分析结论.md)；本文保留可复现的全部统计。', '',
        '## 口径', '',
        '- 使用 partition_manifest.csv 中全体客户端的训练样本数之和；不是测试样本数，也不是单客户端样本数。',
        '- Many: >100；Medium: 20–100（含端点）；Few: <20。',
        '- 正文主指标统一为第 81–100 轮平均，对应 epoch_80 至 epoch_99；epoch_-1 是初始模型。',
        '- 组内按类别宏平均；原测试集每类 100 张，故 Overall 是 100 类均值，不按训练样本数加权。',
        '- 只有一个训练种子。轮间平均不是独立重复实验，也不提供跨种子置信区间。',
        '- 新共享 C 的 B 尚无归档结果，不在本表；所有 B 标记均指旧本地 C 迁移。', '',
        '## 实际分组', '', markdown_table(group_rows, ['group', 'number_of_classes', 'training_samples', 'min_train_count', 'max_train_count']), '',
        'Middle60 = Many15 (ID20–34) + Medium35 (ID35–69) + Few10 (ID70–79)。',
        'Few30 = 原 Middle60 中的 Few10 + 原 Tail20 (ID80–99)。以上 ID 均从 0 开始。', '',
        '## 完整结果（准确率 %）', '', markdown_table(list(summary.values()), frequency_fields), '',
        '## Middle60 拆解（准确率 %）', '', markdown_table(list(summary.values()), split_fields), '',
        '## 相对 S 的分组差值（百分点）', '',
        markdown_table([r for r in contrasts if r['baseline'] == 'S'],
                       ['method', 'delta_Overall', 'delta_Many35', 'delta_Medium35', 'delta_Few30', 'delta_Middle_Few10', 'delta_Tail20']), '',
        '## Overall 的精确分解', '',
        'Overall = 0.20 Head20 + 0.15 Middle_Many15 + 0.35 Middle_Medium35 + 0.10 Middle_Few10 + 0.20 Tail20。',
        'Middle60 = (15 Middle_Many15 + 35 Middle_Medium35 + 10 Middle_Few10) / 60。',
        'Few30 = (10 Middle_Few10 + 20 Tail20) / 30。',
        'contrasts.csv 给出每块对 Overall 差值的贡献；不能直接把各组准确率差值相加。',
        'Few10 的 Middle 损失占比分别记录带符号的净损失占比和只累计负项的毛损失占比；净占比可为负或超过 100%。负值表示 Few10 提升、抵消其他组损失，不能说 Few10 造成了负的损失人数。', '',
        '## 数据核对', '',
        f'- {len(checks)} 组 × 101 轮 × 100 类 = {len(checks)*10100:,} 条逐类记录，所有组全局训练类别计数一致。',
        f'- 与原 round_metrics.csv 已有字段（Overall、Non-tail、Tail；新版另含 Head/Middle）逐轮核对，最大误差：{max(r["max_original_metric_error_pp"] for r in checks):.3g} pp。',
        '- data_checks.csv 保存每个实验的完整性与代数重构核对；sources.csv 保存原始归档路径。', '',
        '## 图片', '',
        '- 01_disjoint_group_changes.png/svg：五个互斥类别组的差值热图；数字为 pp，颜色以 0 对称。',
        '- 02_overall_contributions.png/svg：按类别数加权后对 Overall 的增减贡献，黑色菱形是净变化。',
        '- 03_middle_and_tail_trajectories.png/svg：S、E3、CP1、J 的原始逐轮轨迹，颜色/线型/标记同时区分。',
        '- 图片为离线统计展示，不改变训练协议。PNG 方便查看，SVG 保留可编辑矢量。', '',
        '## 复现', '',
        '```bash', 'python scripts/analyze_cliplora_frequency_groups.py', '```', '',
    ]
    (out / 'report.md').write_text('\n'.join(report), encoding='utf-8')
    (out / 'protocol.json').write_text(json.dumps({'training_modified': False, 'b_targets_modified': False,
        'counts_source': str(reference.relative_to(ROOT) / 'partition_manifest.csv'),
        'many': '>100', 'medium': '20..100 inclusive', 'few': '<20', 'rounds': [81, 100],
        'class_id_base': 0, 'class_groups': groups, 'seed': 42}, ensure_ascii=False, indent=2), encoding='utf-8')
    if not args.no_plots:
        plot_results(out, groups, summary, curves)
    print(markdown_table([summary[m] for m in ('S', 'Full10', 'CP1', 'CP3', 'FlatCP', 'B0.3', 'Uniform', 'UniformB')], frequency_fields))
    print('\nMiddle decomposition:')
    print(markdown_table([r for r in contrasts if r['baseline'] == 'S' and r['method'] in ('Full10', 'CP1', 'CP3')],
                        ['method', 'delta_Middle_Many15', 'delta_Middle_Medium35', 'delta_Middle_Few10',
                         'middle_few10_share_of_net_middle_loss_percent']))
    print(f'\nWritten: {out}')


if __name__ == '__main__':
    main()
