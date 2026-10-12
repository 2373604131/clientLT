"""Summaries preserve negative and ambiguous outcomes; no automatic method choice."""
from pathlib import Path

import numpy as np

from tools.a_projection_diagnosis.protocol import SNAPSHOTS, ANCHORS, read_csv, read_json, write_csv, write_json


def paired_interval(difference, labels, classes, draws=1000):
    """Class-stratified image bootstrap conditional on the fitted models/probes."""
    rng = np.random.default_rng(20261012)
    samples = np.zeros(draws)
    present = [c for c in classes if np.any(labels == c)]
    for c in present:
        values = difference[labels == c]
        samples += values[rng.integers(0, len(values), size=(draws, len(values)))].mean(1) / len(present)
    return dict(mean=float(np.mean([difference[labels == c].mean() for c in present])),
                low=float(np.quantile(samples, .025)), high=float(np.quantile(samples, .975)))


def summarize(root, plots=True):
    root = Path(root)
    groups = dict(head20=list(range(20)), middle60=list(range(20, 80)), tail20=list(range(80, 100)))
    status, primary, interpretations, contrasts = [], [], [], []
    output = root / 'analysis'
    output.mkdir(parents=True, exist_ok=True)
    module_names = None
    for trajectory in ('fedavg', 'tailrw16'):
        folder = root / trajectory / 'e1'
        complete = (folder / 'completion.json').is_file()
        status.append(dict(trajectory=trajectory, experiment='E1', complete=complete))
        if not complete:
            continue
        rows = read_csv(folder / 'representation_curves.csv')
        module_names = list(dict.fromkeys(r['module'] for r in rows))
        for combination in ('fixed_input', 'live_input', 'fixed_A'):
            arrays = {}
            for rnd in (20, 80, 90):
                with np.load(folder / 'probe_predictions' / ('r%03d_%s_test.npz' % (rnd, combination))) as saved:
                    arrays[rnd] = dict(saved)
            y = arrays[20]['labels']
            if any(not np.array_equal(y, v['labels']) or not np.array_equal(arrays[20]['sample_id'], v['sample_id']) for v in arrays.values()):
                raise ValueError('E1 bootstrap samples are not paired')
            for i, module in enumerate(module_names):
                delta = 100 * (((arrays[80]['m%d_prediction' % i] == y).astype(float)
                                + (arrays[90]['m%d_prediction' % i] == y)) / 2
                               - (arrays[20]['m%d_prediction' % i] == y))
                for group, classes in groups.items():
                    primary.append(dict(trajectory=trajectory, module=module, combination=combination,
                                        group=group, contrast='mean80_90_minus20_pp', **paired_interval(delta, y, classes)))
        for module in module_names:
            lookup = {r['combination']: r for r in primary if r['trajectory'] == trajectory and r['module'] == module and r['group'] == 'tail20'}
            a, live = lookup['fixed_input']['mean'], lookup['live_input']['mean']
            interpretation = ('fixed_and_live_decline_check_model_and_intervention' if a < 0 and live < 0 else
                              'old_input_decline_live_not_declining_possible_rematching' if a < 0 else
                              'live_only_decline_check_upstream_interaction' if live < 0 else
                              'no_tail_readability_decline_in_this_measure')
            interpretations.append(dict(trajectory=trajectory, module=module, fixed_delta=a, live_delta=live,
                                        descriptive_sign_pattern=interpretation,
                                        caution='Signs alone do not establish a mechanism; inspect intervals and model interventions'))
    for rnd in ANCHORS:
        folder = root / 'fedavg' / ('round%03d' % rnd)
        complete = (folder / 'completion.json').is_file()
        status.append(dict(trajectory='fedavg', experiment='E2_E3_rowspace', round=rnd, complete=complete))
        if not complete:
            continue
        pairs = [('space_%s_%s' % (scale, n), 'space_%s_full' % scale)
                 for scale in ('raw', 'matched') for n in ('parallel', 'outside')]
        pairs += [('weight_%s_%s' % (scale, n), 'weight_%s_fedavg' % scale)
                  for scale in ('raw', 'matched') for n in ('tailrw16', 'joint')]
        for split in ('panel', 'remaining'):
            for left, right in pairs:
                with np.load(folder / 'model_predictions' / (left + '_' + split + '.npz')) as saved:
                    a = dict(saved)
                with np.load(folder / 'model_predictions' / (right + '_' + split + '.npz')) as saved:
                    b = dict(saved)
                if not np.array_equal(a['sample_id'], b['sample_id']) or not np.array_equal(a['labels'], b['labels']):
                    raise ValueError('Intervention bootstrap samples are not paired')
                for group, classes in groups.items():
                    for metric in ('accuracy_pp', 'cosine_margin'):
                        delta = (100 * ((a['prediction'] == a['labels']).astype(float) - (b['prediction'] == b['labels']))
                                 if metric == 'accuracy_pp' else a['margin'] - b['margin'])
                        contrasts.append(dict(round=rnd, split=split, left=left, right=right, group=group,
                                              metric=metric, **paired_interval(delta, a['labels'], classes)))
    write_csv(output / 'status.csv', status)
    write_csv(output / 'primary_representation_change.csv', primary)
    write_csv(output / 'interpretation_branches.csv', interpretations)
    write_csv(output / 'paired_candidate_contrasts.csv', contrasts)
    finished = all(r['complete'] for r in status)
    write_json(output / 'summary.json', dict(complete=finished, completed_tasks=sum(r['complete'] for r in status),
        required_tasks=len(status), bootstrap_draws=1000, independent_training_seeds=1,
        uncertainty='paired class-stratified test-image bootstrap, conditional on trained model and fitted probe',
        remaining_scope='remaining official samples, not an independent new confirmation set',
        method_A_decided=False))
    lines = ['A 投影学习诊断', '', '两轨迹与三个事件均完成。' if finished else '实验尚未完整：查看 status.csv 中缺失的任务。', '',
             'primary_representation_change.csv：六模块及辅助拼接探针的 (80+90)/2−20 变化。',
             'interpretation_branches.csv：固定输入与真实输入的描述性分支，不是自动机制裁决。',
             'paired_candidate_contrasts.csv：预定候选在固定面板和剩余样本上的配对差值。', '',
             '判读时同时查看 local_class_response、response_decomposition、行空间干预的 lost/gained 和 norm_matching。',
             '固定输入下降但真实输入稳定/改善时，优先考虑 A 与上游重新匹配。',
             '空间内同秩探针应不变；完整模型仍可能变化，不能直接推断 B 容量不足。',
             '空间外更新也改变 Gram/尺度；等总有效范数不等于等激活或等层敏感度。',
             '即使空间外更新损害尾类，也需检查新正确样本，不能据此永久冻结行空间。',
             'A 阶段的 B 共同固定，不能以 A/B 分别聚合的乘积误差解释本阶段结果。',
             '全部区间只反映给定模型/探针的测试样本不确定性，不覆盖训练种子和 probe 拟合。',
             '本轮保留单种子、预设池化与固定 lambda；不比较测试表现后再选超参。']
    (output / '结论与判读.txt').write_text('\n'.join(lines) + '\n', encoding='utf-8')
    if plots and primary:
        plot_results(root, output)
    return dict(complete=finished, tasks=status, primary_rows=len(primary), contrasts=len(contrasts))


def plot_results(root, output):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    for trajectory in ('fedavg', 'tailrw16'):
        path = root / trajectory / 'e1/representation_curves.csv'
        if not path.is_file():
            continue
        rows = [r for r in read_csv(path) if r['group'] == 'tail20' and r['module'] != 'combined48']
        modules = list(dict.fromkeys(r['module'] for r in rows))
        fig, axes = plt.subplots(3, 2, figsize=(10, 9), sharex=True)
        for ax, module in zip(axes.flat, modules):
            for combination, label in [('fixed_input', 'Q(A_t) X_0'), ('live_input', 'Q(A_t) X_t'), ('fixed_A', 'Q(A_0) X_t')]:
                selected = sorted((r for r in rows if r['module'] == module and r['combination'] == combination), key=lambda r: int(r['round']))
                ax.plot([int(r['round']) for r in selected], [float(r['accuracy']) for r in selected], marker='o', label=label)
            ax.set_title(module.split('resblocks.')[-1].replace('.w_lora_A', ''))
            ax.set_ylabel('Tail20 probe accuracy (%)')
            ax.set_xlabel('Committed round')
            ax.grid(alpha=.2)
        axes.flat[0].legend(fontsize=8)
        fig.suptitle(trajectory + ': fixed sample budget, fixed ridge readout')
        fig.tight_layout()
        fig.savefig(output / (trajectory + '_tail_readability.png'), dpi=180)
        fig.savefig(output / (trajectory + '_tail_readability.pdf'))
        plt.close(fig)
