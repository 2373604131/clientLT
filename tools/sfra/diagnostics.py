"""Read-only Method A diagnostics and a bounded experiment manifest (no torch)."""
import csv
import hashlib
import json
from pathlib import Path

import numpy as np

SCHEMA = 'method_a_diagnostics_v1'
ANCHORS = (60, 80)
HORIZON = 5
BRANCHES = ('full-cp', 'ordinary', 'norm-matched')
WINDOWS = ((1, 30), (31, 60), (61, 90))
TOLERANCE = 1e-7


def read_json(path):
    return json.loads(Path(path).read_text(encoding='utf-8-sig'))


def read_csv(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False), encoding='utf-8')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def file_hash(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def validate_reference(source):
    cfg = read_json(Path(source) / 'sfra_config.json')
    for key, value in dict(variant='full-cp', retention_weight=10., classification_weight=1.,
                           seed=42, protocol_seed=42, partition='client-longtail',
                           correction_steps=3, correction_step_size=.1).items():
        if cfg.get(key) != value:
            raise ValueError(f'Reference {key}: {cfg.get(key)!r} != {value!r}')
    if cfg.get('b_transfer') or cfg.get('b_aggregation'):
        raise ValueError('Use the original A-only, sample-weighted B reference')
    return cfg


def class_groups(counts, tail=None):
    counts = np.asarray(counts)
    tail = np.asarray(tail if tail is not None else sorted(range(len(counts)),
                      key=lambda c: (int(counts[c]), -c))[:20], dtype=int)
    return {'Overall': np.arange(len(counts)), 'Many35': np.flatnonzero(counts > 100),
            'Medium35': np.flatnonzero((counts >= 20) & (counts <= 100)),
            'Few30': np.flatnonzero(counts < 20), 'Tail20': tail}


def preflight(source, output):
    source, output = Path(source).resolve(), Path(output).resolve()
    validate_reference(source)
    names = ['command.json', 'bridge_metadata.json', 'partition_manifest.csv',
             'private_witness_manifest.json', 'checkpoints/base_model.pt',
             'protocol/full_schedule.json', 'protocol/eri_protocol.json', 'protocol/probe_manifest.csv']
    for rnd in range(1, max(ANCHORS) + 1):
        names.append(f'sfra_rounds/r{rnd:03d}/tokens.npz')
    for anchor in ANCHORS:
        names.extend([f'sfra_rounds/r{anchor:03d}/commit.pt',
                      f'events/r{anchor+1:03d}_c000_main_normal_B/event.json'])
    rows = [dict(file=n, exists=(source / n).is_file(),
                 bytes=(source / n).stat().st_size if (source / n).is_file() else 0) for n in names]
    missing = [r['file'] for r in rows if not r['exists']]
    result = dict(schema_version=SCHEMA, source=str(source), anchors=list(ANCHORS), horizon=HORIZON,
                  branches=list(BRANCHES), no_cp_probe='first A event only at each anchor',
                  ready_files=not missing, missing=missing,
                  nominal_branch_rounds=len(ANCHORS)*len(BRANCHES)*HORIZON,
                  rng_protocol='new paired per-phase stream; not exact original trajectory continuation',
                  scope='one seed, Full-CP trajectory anchors; source/replay/weights validated at runtime')
    write_csv(output / 'required_files.csv', rows)
    write_json(output / 'preflight.json', result)
    return result


def analyze(source, output, make_plot=True):
    source, output = Path(source).resolve(), Path(output).resolve()
    validate_reference(source)
    output.mkdir(parents=True, exist_ok=True)
    witnesses = read_json(source / 'private_witness_manifest.json')
    if [w['token_id'] for w in witnesses] != list(range(len(witnesses))):
        raise ValueError('Witness order is not the stored token order')
    labels = np.array([w['class_id'] for w in witnesses])
    clients = np.array([w['client_id'] for w in witnesses])
    partition = read_csv(source / 'partition_manifest.csv')
    counts = np.bincount([int(r['class_id']) for r in partition], minlength=100)
    local_counts = {}
    for row in partition:
        key = int(row['client_id']), int(row['class_id'])
        local_counts[key] = local_counts.get(key, 0) + 1
    expected_weights = np.array([local_counts[(w['client_id'], w['class_id'])] / len(partition)
                                 for w in witnesses])
    recorded_counts = read_json(source / 'partition_summary.json')['global_class_counts']
    if not np.array_equal(counts, recorded_counts):
        raise ValueError('Actual partition counts differ from recorded counts')
    groups = class_groups(counts)
    classnames = read_json(source/'bridge_metadata.json')['classnames']
    summary = {int(r['round']): r for r in read_csv(source / 'sfra_rounds.csv')}
    class_rows, group_rows, geometry = [], [], []
    hashes = {n: file_hash(source/n) for n in ('sfra_config.json', 'partition_manifest.csv',
              'private_witness_manifest.json', 'partition_summary.json', 'sfra_rounds.csv')}
    max_loss_error = 0.
    for rnd in range(1, 91):
        name = f'sfra_rounds/r{rnd:03d}/tokens.npz'
        hashes[name] = file_hash(source/name)
        with np.load(source/name, allow_pickle=False) as archive:
            z = {k: archive[k] for k in archive.files}
        weights = z['classification_global_token_weights'].astype(float)
        if not np.allclose(weights, expected_weights, atol=1e-8, rtol=1e-5):
            raise ValueError(f'Round {rnd}: classification weights differ from actual sample counts')
        post, ordinary, commit = [z[k].astype(float).mean(1) for k in
                                 ('F_post_B', 'F_proposal', 'F_committed')]
        # Subtract in float64 rather than after rounding a float32 difference.
        delta_loss = (z['classification_committed_scores'].astype(float)
                      - z['classification_proposal_scores'].astype(float)).mean(1)
        vectors = dict(la_increase=delta_loss, ordinary_margin_gain=ordinary-post,
                       correction_margin_gain=commit-ordinary, net_margin_gain=commit-post)
        if not all(np.isfinite(v).all() for v in vectors.values()):
            raise ValueError(f'Round {rnd}: nonfinite measurements')
        error = abs(float(np.dot(weights, delta_loss))
                    - float(summary[rnd]['committed_classification_loss_increase']))
        max_loss_error = max(max_loss_error, error)
        if error > 2e-6:
            raise ValueError(f'Round {rnd}: token loss does not reconstruct global loss ({error})')
        response = z['responses'].astype(float).min(1)
        positive = np.where(response > 1e-6, response, 0.)
        p = z['classification_client_weights'].astype(float)
        source_metrics = dict(positive_total=positive.sum(1), weighted_positive=positive @ p,
                              weighted_negative=np.minimum(response, 0.) @ p)
        per_class = []
        for c in range(len(counts)):
            mask = labels == c
            mass = weights[mask].sum()
            row = dict(round=rnd, class_id=c, class_name=classnames[c], train_count=int(counts[c]),
                       frequency_group='Many35' if counts[c] > 100 else 'Medium35' if counts[c] >= 20 else 'Few30',
                       is_tail=c in groups['Tail20'], token_count=int(mask.sum()))
            for key, vector in {**vectors, **source_metrics}.items():
                row[key+'_unit_mean'] = float(vector[mask].mean())
                row[key+'_sample_weighted'] = float(np.dot(vector[mask], weights[mask])/mass)
            support = mask & z['supported'].astype(bool)
            row['n_eff_supported_mean'] = float(z['n_eff'][support].mean()) if support.any() else None
            row['hurt_unit_fraction'] = float((delta_loss[mask] > TOLERANCE).mean())
            row['functional_weight_mass'] = float(z['weights'][mask].sum())
            row['cp_sample_mass'] = float(mass)
            per_class.append(row)
        class_rows.extend(per_class)
        for group, ids in groups.items():
            mask = np.isin(labels, ids)
            row = dict(round=rnd, group=group, class_count=len(ids))
            for key, vector in vectors.items():
                row[key+'_class_balanced'] = float(np.mean([per_class[c][key+'_unit_mean'] for c in ids]))
                row[key+'_sample_weighted'] = float(np.dot(vector[mask], weights[mask])/weights[mask].sum())
            row['hurt_class_count'] = sum(per_class[c]['la_increase_unit_mean'] > TOLERANCE for c in ids)
            row['hurt_unit_fraction'] = float((delta_loss[mask] > TOLERANCE).mean())
            row['cp_sample_mass'] = float(weights[mask].sum())
            group_rows.append(row)
        step_name = f'sfra_rounds/r{rnd:03d}/correction_steps.csv'
        steps = read_csv(source/step_name) if (source/step_name).exists() else []
        if steps:
            hashes[step_name] = file_hash(source/step_name)
        later = [r for r in steps if int(r['step']) in (2, 3)]
        cosines = [float(r['functional_classification_gradient_cosine']) for r in later
                   if r.get('functional_classification_gradient_cosine')]
        s = summary[rnd]
        norm = float(s['proposal_effective_norm'])
        geometry.append(dict(round=rnd, effective_norm_ratio=float(s['committed_effective_norm'])/norm if norm else None,
                             a_direction_cosine=float(s['proposal_committed_cosine']),
                             global_la_increase=float(s['committed_classification_loss_increase']),
                             cp_active_later_fraction=sum(r['classification_active']=='True' for r in later)/len(later) if later else None,
                             gradient_cosine=float(np.mean(cosines)) if cosines else None,
                             global_loss_reconstruction_error=error))
    windows, class_windows = [], []
    for first, last in WINDOWS:
        label = f'{first}-{last}'
        for group, ids in groups.items():
            selected = [r for r in group_rows if first <= r['round'] <= last and r['group'] == group]
            row = dict(window=label, group=group, rounds=len(selected), class_count=len(ids))
            for key in selected[0]:
                if key not in ('round', 'group', 'class_count'):
                    row[key] = float(np.mean([r[key] for r in selected]))
            means = [np.mean([r['la_increase_unit_mean'] for r in class_rows
                             if r['class_id']==c and first<=r['round']<=last]) for c in ids]
            row['classes_hurt_on_window_mean'] = sum(v > TOLERANCE for v in means)
            windows.append(row)
        for c in range(len(counts)):
            selected = [r for r in class_rows if r['class_id']==c and first<=r['round']<=last]
            row = {k: selected[0][k] for k in ('class_id', 'class_name', 'train_count', 'frequency_group', 'is_tail')}
            row['window'] = label
            for key in ('la_increase_unit_mean', 'la_increase_sample_weighted', 'ordinary_margin_gain_unit_mean',
                        'correction_margin_gain_unit_mean', 'net_margin_gain_unit_mean', 'hurt_unit_fraction'):
                row[key] = float(np.mean([r[key] for r in selected]))
            row['hurt_round_fraction'] = float(np.mean([r['la_increase_unit_mean'] > TOLERANCE for r in selected]))
            class_windows.append(row)
    write_csv(output/'q1_class_rounds.csv', class_rows)
    write_csv(output/'q1_group_rounds.csv', group_rows)
    write_csv(output/'q1_group_windows.csv', windows)
    write_csv(output/'q1_class_windows.csv', class_windows)
    write_csv(output/'q1_update_geometry.csv', geometry)
    late_medium = sorted((r for r in class_windows if r['window']=='61-90' and r['frequency_group']=='Medium35'),
                         key=lambda r: r['la_increase_unit_mean'], reverse=True)
    write_csv(output/'q1_medium_late_ranking.csv', late_medium)
    preflight_result = preflight(source, output)
    report = ['# 方法A问题1：已有日志分析', '', '仅分析Full-CP μ1、Client-LT、seed42。没有重新训练。', '',
              '下表是训练见证上的LA损失变化：正数表示修正后更差，负数表示更好；不是准确率百分点。', '',
              '| 轮次 | 类别组 | 类均衡LA变化 | 样本加权LA变化 | 窗口平均受损类别 |',
              '|---|---|---:|---:|---:|']
    for r in windows:
        report.append(f"| {r['window']} | {r['group']} | {r['la_increase_class_balanced']:+.6f} | "
                      f"{r['la_increase_sample_weighted']:+.6f} | {r['classes_hurt_on_window_mean']}/{r['class_count']} |")
    report += ['', '类均衡：同类客户端—类别单位等权，再按类别等权。样本加权：使用真实训练计数，组内重新归一化。',
               '两种平均服务于不同问题；不把总体平均恢复等同于每一类都受到保护。', '',
               '## 后期中频类损害最大的十类', '', '| 类别ID | 训练张数 | LA变化 | 受损轮次比例 |', '|---|---:|---:|---:|']
    for r in late_medium[:10]:
        report.append(f"| {r['class_id']} ({r['class_name']}) | {r['train_count']} | {r['la_increase_unit_mean']:+.6f} | {r['hurt_round_fraction']:.1%} |")
    report += ['', '## 普通学习收益是否被撤销', '',
               '下表为后期中频损害前三类，数值是同轮平均margin变化。先平均客户端—类别单位，再平均第61–90轮。', '',
               '| 类别 | 普通A更新带来的变化 | 功能修正带来的变化 | 整个A阶段净变化 |',
               '|---|---:|---:|---:|']
    for r in late_medium[:3]:
        report.append(f"| {r['class_id']} ({r['class_name']}) | {r['ordinary_margin_gain_unit_mean']:+.7f} | "
                      f"{r['correction_margin_gain_unit_mean']:+.7f} | {r['net_margin_gain_unit_mean']:+.7f} |")
    report += ['', '## 证据边界与下一步', '',
               '- 已定位到同轮普通提案→正式修正的训练侧变化；不能把它直接解释为Medium测试下降的全部原因。',
               '- 原始记录没有每张见证图片的预测对错，不能由平均margin恢复准确率或“学会又忘记”的样本队列。',
               '- CP贡献和等步幅方向比较仍需相同状态下的候选实验，本报告没有这些反事实结果。',
               '- 第91–100轮只有普通B训练，不计入本表的A修正统计。',
               f'- 逐单位加权损失与原逐轮汇总的最大重算误差：{max_loss_error:.3g}。',
               f"- 短程所需文件检查：{'文件齐全，仍须运行时核验' if preflight_result['ready_files'] else '缺少模型/协议文件，详见required_files.csv'}。", '']
    (output/'report.md').write_text('\n'.join(report), encoding='utf-8')
    write_json(output/'analysis_provenance.json', dict(schema_version=SCHEMA, source=str(source),
               tolerance=TOLERANCE, input_sha256=hashes, rounds=90, tokens=len(witnesses),
               max_global_loss_reconstruction_error=max_loss_error))
    if make_plot:
        import matplotlib
        matplotlib.use('Agg')
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(8, 4.3))
        names = ['Many35', 'Medium35', 'Few30', 'Tail20']
        x = np.arange(len(names))
        for i, (first, last) in enumerate(WINDOWS):
            label = f'{first}-{last}'
            values = [next(r['la_increase_class_balanced'] for r in windows if r['window']==label and r['group']==g) for g in names]
            ax.bar(x+(i-1)*.25, values, width=.24, label=f'Rounds {label}')
        ax.axhline(0, color='black', linewidth=.7)
        ax.set_xticks(x, names)
        ax.set_ylabel('Witness LA loss: corrected - ordinary')
        ax.legend(frameon=False)
        ax.set_title('Full-CP: classification cost by class group (seed42)')
        fig.tight_layout()
        fig.savefig(output/'q1_group_loss.png', dpi=180)
        fig.savefig(output/'q1_group_loss.pdf')
        plt.close(fig)
    return dict(rounds=90, tokens=len(witnesses), output=str(output),
                max_loss_error=max_loss_error, late_medium_hurt=sum(r['la_increase_unit_mean']>TOLERANCE for r in late_medium))
