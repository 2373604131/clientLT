"""Paired intervention contrasts, explicit safety costs, and role-response plots."""
from collections import defaultdict
from pathlib import Path
from statistics import mean

from tools.factor_roles.common import METRICS, PROTOCOL, TOPOLOGIES, read_csv, read_json, write_csv, write_json


def collect(root):
    candidates, classes, budgets, exposure, anchors, requests = [], [], [], [], [], []
    for origin in TOPOLOGIES:
        for rnd in PROTOCOL['rounds']:
            path = root/origin/f'round_{rnd:03d}'
            if not (path/'complete.json').is_file():
                continue
            requests.append(read_json(path/'request.json'))
            anchors.append(read_json(path/'anchor_info.json'))
            for row in read_csv(path/'candidate_metrics.csv'):
                row['round'] = int(row['round'])
                for key in METRICS + ('effective_norm', 'raw_effective_norm', 'norm_target', 'multiplier'):
                    row[key] = float(row[key])
                row['norm_comparable'] = row['norm_comparable'] == 'True'
                row['norm_relative_error'] = float(row['norm_relative_error']) if row['norm_relative_error'] else 0.
                candidates.append(row)
            classes.extend(read_csv(path/'per_class.csv'))
            budgets.extend(dict(origin=origin, round=rnd, **r) for r in read_csv(path/'budget.csv'))
            exposure.extend(dict(origin=origin, round=rnd, **r) for r in read_csv(path/'class_exposure.csv'))
    return candidates, classes, budgets, exposure, anchors, requests


def contrasts(rows):
    groups = defaultdict(list)
    for row in rows:
        groups[row['origin'], row['round'], row['mode']].append(row)
    effects, scores = [], []
    for (origin, rnd, mode), candidates in sorted(groups.items()):
        index = {(r['topology'], r['factor'], r['loss'], r['weighting']): r for r in candidates}
        assert len(index) == 16, 'An anchor must contain the whole factorial comparison'
        # Record ALL CE/LA aggregation effects and ALL sample/uniform LA effects.
        for topology in TOPOLOGIES:
            for factor in ('A', 'B'):
                for loss in ('CE', 'LA'):
                    before, after = index[topology, factor, loss, 'sample'], index[topology, factor, loss, 'uniform']
                    for metric in METRICS:
                        effects.append(dict(origin=origin, round=rnd, mode=mode, topology=topology, factor=factor,
                            correction='client', condition=loss, metric=metric,
                            before=before[metric], after=after[metric], gain=after[metric]-before[metric]))
                for weighting in ('sample', 'uniform'):
                    before, after = index[topology, factor, 'CE', weighting], index[topology, factor, 'LA', weighting]
                    for metric in METRICS:
                        effects.append(dict(origin=origin, round=rnd, mode=mode, topology=topology, factor=factor,
                            correction='class', condition=weighting, metric=metric,
                            before=before[metric], after=after[metric], gain=after[metric]-before[metric]))

        def u(topology, factor, metric='tail20'):
            return index[topology, factor, 'LA', 'uniform'][metric] - index[topology, factor, 'LA', 'sample'][metric]

        def c(topology, factor, metric='tail20'):
            return index[topology, factor, 'LA', 'sample'][metric] - index[topology, factor, 'CE', 'sample'][metric]

        clt, dr = TOPOLOGIES
        ka, kb = u(clt, 'A')-u(dr, 'A'), u(clt, 'B')-u(dr, 'B')
        client_guard = all(u(t, 'A', 'overall') >= PROTOCOL['client_overall_floor_pp']
            and min(u(t, 'A', 'head20'), u(t, 'A', 'middle60')) >= PROTOCOL['client_head_middle_floor_pp'] for t in TOPOLOGIES)
        client_guard = client_guard and u(dr, 'A') >= PROTOCOL['dir_tail_floor_pp']
        class_guard = all(c(t, 'B', 'overall') >= PROTOCOL['class_overall_floor_pp']
            and min(c(t, 'B', 'head20'), c(t, 'B', 'middle60')) >= PROTOCOL['class_head_middle_floor_pp'] for t in TOPOLOGIES)
        comparable = all(r['norm_comparable'] for r in candidates)
        if mode == 'norm_matched':
            comparable = comparable and max(r['norm_relative_error'] for r in candidates) <= .001
        client_pattern = u(clt, 'A') > 0 and u(clt, 'A') > u(clt, 'B') and ka > kb and client_guard
        class_pattern = all(c(t, 'B') > 0 and c(t, 'B') > c(t, 'A') for t in TOPOLOGIES) and class_guard
        scores.append(dict(origin=origin, round=rnd, mode=mode,
            U_A_CLT=u(clt, 'A'), U_B_CLT=u(clt, 'B'), U_A_Dir=u(dr, 'A'), U_B_Dir=u(dr, 'B'),
            K_A=ka, K_B=kb, client_interaction=ka-kb,
            C_A_CLT=c(clt, 'A'), C_B_CLT=c(clt, 'B'), class_interaction_CLT=c(clt, 'B')-c(clt, 'A'),
            C_A_Dir=c(dr, 'A'), C_B_Dir=c(dr, 'B'), class_interaction_Dir=c(dr, 'B')-c(dr, 'A'),
            client_guard_pass=client_guard, class_guard_pass=class_guard,
            norm_comparable=comparable, client_pattern=client_pattern, class_pattern=class_pattern,
            joint_pattern=client_pattern and class_pattern and comparable))
    return effects, scores


def plot_results(out, scores, effects):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    import numpy as np

    modes = ('raw', 'norm_matched')
    matrices = []
    for mode in modes:
        selected = [r for r in scores if r['mode'] == mode]
        matrices.append(np.array([[mean(r['K_A'] for r in selected), mean(r['K_B'] for r in selected)],
            [mean(r['C_A_CLT'] for r in selected), mean(r['C_B_CLT'] for r in selected)]]))
    limit = max(max(float(np.abs(m).max()) for m in matrices), .001)
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
    for ax, mode, matrix in zip(axes, modes, matrices):
        im = ax.imshow(matrix, cmap='RdBu', vmin=-limit, vmax=limit)
        ax.set_xticks([0, 1], ['A-only', 'B-only'])
        ax.set_yticks([0, 1], ['Client correction\nK = U_CLT - U_Dir', 'Class correction\nLA - CE on CLT'])
        ax.set_title(mode)
        for i in range(2):
            for j in range(2):
                ax.text(j, i, f'{matrix[i,j]:+.3f}', ha='center', va='center',
                        color='white' if abs(matrix[i,j]) > .6*limit else 'black')
    fig.colorbar(im, ax=axes, label='Tail accuracy gain (percentage points)')
    fig.suptitle('Mean paired response; diagonal dominance is the hypothesis, not a constraint')
    fig.savefig(out/'role_matrix.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 3, figsize=(13, 6), constrained_layout=True)
    columns = [('client_interaction', 'Client: K_A - K_B'),
               ('class_interaction_CLT', 'Class on CLT: C_B - C_A'),
               ('class_interaction_Dir', 'Class on Dir: C_B - C_A')]
    for i, mode in enumerate(modes):
        for ax, (column, title) in zip(axes[i], columns):
            for origin in TOPOLOGIES:
                selected = sorted((r for r in scores if r['mode'] == mode and r['origin'] == origin), key=lambda r:r['round'])
                if selected:
                    ax.plot([r['round'] for r in selected], [r[column] for r in selected], 'o-', label=origin)
            ax.axhline(0, color='black', linewidth=.7)
            ax.set_title(f'{mode}\n{title}')
            ax.set_xticks([20, 50, 80])
            ax.set_xlabel('Anchor round (NOT independent seeds)')
            ax.set_ylabel('Tail gain difference (pp)')
    axes[0, 0].legend(fontsize=7)
    fig.savefig(out/'paired_interactions.png', dpi=180)
    plt.close(fig)

    fig, axes = plt.subplots(2, 2, figsize=(12, 7), constrained_layout=True)
    metrics = ['overall', 'head20', 'middle60', 'tail20']
    for i, mode in enumerate(modes):
        for j, topology in enumerate(TOPOLOGIES):
            ax = axes[i, j]
            for offset, factor in zip([-.18, .18], ['A', 'B']):
                values = [mean(r['gain'] for r in effects if r['mode'] == mode and r['topology'] == topology
                    and r['factor'] == factor and r['correction'] == 'client' and r['condition'] == 'LA' and r['metric'] == m) for m in metrics]
                ax.bar(np.arange(4)+offset, values, width=.36, label=f'{factor}: uniform - sample')
            ax.axhline(0, color='black', linewidth=.7)
            ax.set_xticks(range(4), metrics)
            ax.set_title(f'{mode}, {topology}')
            ax.set_ylabel('Accuracy change (pp)')
            ax.legend(fontsize=8)
    fig.savefig(out/'client_correction_costs.png', dpi=180)
    plt.close(fig)


def summarize(root):
    root = Path(root)
    out = root/'analysis'
    candidates, classes, budgets, exposure, anchors, requests = collect(root)
    if not candidates:
        raise FileNotFoundError(f'No completed factor-role anchors under {root}')
    assert all(r['protocol'] == PROTOCOL for r in requests), 'Cannot pool different protocols'
    effects, scores = contrasts(candidates)
    for filename, rows in [('candidates', candidates), ('per_class', classes), ('budget', budgets),
                           ('class_exposure', exposure), ('intervention_effects', effects), ('paired_role_scores', scores)]:
        write_csv(out/f'{filename}.csv', rows)
    write_json(out/'anchor_metadata.json', anchors)
    seen = {(a['origin'], a['round']) for a in anchors}
    missing = [f'{t}/round_{rnd:03d}' for t in TOPOLOGIES for rnd in PROTOCOL['rounds'] if (t, rnd) not in seen]
    complete = not missing
    pattern_count = {mode: sum(r['joint_pattern'] for r in scores if r['mode'] == mode) for mode in ('raw', 'norm_matched')}
    # This is a descriptive pre-specified crossover, NOT a significance test.
    full_pattern = complete and all(n == 6 for n in pattern_count.values())
    write_json(out/'evidence_status.json', dict(protocol=PROTOCOL, completed_anchors=len(anchors),
        missing_anchors=missing, observed_joint_pattern_counts=pattern_count,
        full_preregistered_pattern_observed=full_pattern, independent_training_seeds=1,
        inference='functional intervention roles at the tested S anchors; not exclusive knowledge storage or long-term retention'))
    lines = ['# A/B 客户端—类别双层适配诊断', '',
        f'已完成 {len(anchors)}/6 个共同起点；{len(anchors)*8}/48 个全客户端单 epoch 分支。',
        f'实际 optimizer steps：{sum(int(r["optimizer_steps"]) for r in budgets)}。', '',
        '只使用 Client-LT 与普通 noniid-labeldir-fine（beta=0.5）。所有数值为百分点，probe log-odds 除外。',
        '六个起点是两条 seed42 轨迹的三个时点，不是六个独立训练种子。', '']
    if missing:
        lines += ['尚缺：' + '；'.join(missing) + '。以下为部分结果，不作完整分工结论。', '']
    lines += ['## 1. 直接检验交叉分工', '',
        '- 客户端修正：U = Tail_uniform − Tail_sample（LA 固定）；K = U_CLT − U_Dir。',
        '- 类别修正：C = Tail_LA − Tail_CE（样本量聚合固定）。',
        '- 主张要求 K_A > K_B，而且 CLT 上 U_A > U_B、U_A > 0；两种划分分别要求 C_B > C_A、C_B > 0。', '',
        '| 模式 | K_A | K_B | K_A−K_B | CLT C_A | CLT C_B | Dir C_A | Dir C_B | 联合模式点数 |',
        '|---|---:|---:|---:|---:|---:|---:|---:|---:|']
    for mode in ('raw', 'norm_matched'):
        selected = [r for r in scores if r['mode'] == mode]
        cells = [f'{mean(r[k] for r in selected):+.4f}' for k in
                 ('K_A', 'K_B', 'client_interaction', 'C_A_CLT', 'C_B_CLT', 'C_A_Dir', 'C_B_Dir')]
        lines.append('| ' + ' | '.join([mode, *cells, f'{pattern_count[mode]}/{len(selected)}']) + ' |')
    lines += ['', '上表是配对收益的描述性均值，不是只挑最好的时点，也不代表统计显著。', '',
        '## 2. 全部起点：不得由均值掩盖反向结果', '',
        '| 起点来源 | 轮次 | 模式 | 客户端交互 | 类别交互 CLT | 类别交互 Dir | 代价限制 | 幅度可比 | 联合模式 |',
        '|---|---:|---|---:|---:|---:|---|---|---|']
    for r in scores:
        lines.append(f'| {r["origin"]} | {r["round"]} | {r["mode"]} | {r["client_interaction"]:+.4f} | '
            f'{r["class_interaction_CLT"]:+.4f} | {r["class_interaction_Dir"]:+.4f} | '
            f'{r["client_guard_pass"] and r["class_guard_pass"]} | {r["norm_comparable"]} | {r["joint_pattern"]} |')
    lines += ['', '## 3. 实际收益与代价（不是只看拓扑差距）', '',
        '| 模式 | 修正 | 划分 | 因子 | Overall Δ | Head20 Δ | Middle60 Δ | Tail20 Δ |',
        '|---|---|---|---|---:|---:|---:|---:|']
    for mode in ('raw', 'norm_matched'):
        for correction, condition in [('client', 'LA'), ('class', 'sample')]:
            for topology in TOPOLOGIES:
                for factor in ('A', 'B'):
                    selected = [r for r in effects if r['mode'] == mode and r['correction'] == correction
                        and r['condition'] == condition and r['topology'] == topology and r['factor'] == factor]
                    values = [f'{mean(r["gain"] for r in selected if r["metric"] == m):+.4f}' for m in ('overall', 'head20', 'middle60', 'tail20')]
                    lines.append('| ' + ' | '.join([mode, correction, topology, factor, *values]) + ' |')
    lines += ['', '## 4. 预先固定的判读', '',
        '代价限制（诊断阈值，不是显著性阈值）：客户端 A 修正在两种划分 Overall 不低于 −0.5 pp、'
        'Head/Middle 不低于 −1 pp；Dir Tail 不低于 −0.1 pp。类别 B 修正两种划分 Overall 不低于 −0.5 pp、'
        'Head/Middle 不低于 −1 pp。原始收益全部保留，不能靠阈值隐藏代价。',
        '幅度匹配要求全部候选非零，实际 float32 有效改动相对目标误差不超过 0.1%。'
        '若更新过小使准确率几乎不变，结论是诊断缺少分辨力，不选择另一个测试集最优步长。', '',
        ('六个起点的原始和幅度匹配结果全部出现预设交叉模式并满足代价限制。'
         '支持在本 CLIP-LoRA/S 设置中将 A 定义为客户端级适配主导因子、B 定义为类别平衡适配主导因子。'
         if full_pattern else
         '尚未完整出现预设的跨起点、跨幅度控制交叉模式。先按上表区分是哪项响应、哪个时点或代价限制未成立；'
         '不能仅凭某一个正差值宣布完整 A/B 分工。'), '',
        '两行都由 A（或都由 B）占优，表明该因子更有总体适配能力；仅第一行成立，只支持优先修正 A 的客户端聚合。',
        '等权同时改变名义类别权重，class_exposure.csv 记录 sum_k q_k n_kc/n_k；这是数据组成代理，不是实际梯度/功能贡献。'
        '本实验建立功能干预角色，不宣称两类知识互斥储存。',
        '这里没有长轨迹，因此不能直接声称已减少后期遗忘。完整保持实验待这一步结果支持后再做，本入口不自动启动。', '',
        '## 5. 文件与图', '',
        '- candidates.csv：每个起点的全部绝对分数、相对起点变化、原始/匹配幅度。',
        '- intervention_effects.csv：全部逐指标配对收益，包括 CE 下的聚合和 uniform 下的 LA 交互检查。',
        '- paired_role_scores.csv：全部主交互及逐起点判读。',
        '- per_class.csv、budget.csv、class_exposure.csv：逐类结果、实际预算、等权导致的类别贡献变化。',
        '- anchor_metadata.json：类别组、全局先验、两种划分的客户端样本量、起点来源及匹配目标。',
        '- role_matrix.png：2×2 响应矩阵；paired_interactions.png：全部时点；client_correction_costs.png：实际收益与代价。', '']
    out.mkdir(parents=True, exist_ok=True)
    (out/'report.md').write_text('\n'.join(lines), encoding='utf-8')
    plot_results(out, scores, effects)
    print(f'Summary written: {out / "report.md"}', flush=True)
