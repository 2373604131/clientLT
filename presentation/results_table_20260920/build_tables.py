"""Recompute the current LA/SFRA comparison tables from raw results; PNG only."""
from pathlib import Path
import csv
import json
import html
from decimal import Decimal, ROUND_HALF_UP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Rectangle

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
CLT = ROOT / 'output/la_control_ablation_analysis/la_control/seed42/client-longtail'
DIR_OLD = ROOT / 'output/la_control_standard_dirichlet_e3_j_s_analysis/la_control/seed42/noniid-labeldir-fine'
DIR_NEW = ROOT / 'output/la_control_dirichlet_supplement_analysis/la_control/seed42/noniid-labeldir-fine'
SFRA = ROOT / 'output/sfra_v1_analysis/sfra_v1/seed42/client-longtail'
CAPT = ROOT / 'output/v2_capt_analysis/output/cifar100_LT/v2_matched/seed42/capt'
METHODS = ['E0', 'E1', 'E2', 'E3', 'E5', 'J', 'S']
EXTRA = ['Full-1', 'Full-3', 'Full-10', 'Current-10', 'Flat-10']
ORDER = ['CAPT'] + METHODS + EXTRA
DESC = {
    'CAPT': '原生提示学习（历史参照）',
    'E0': '无LA / 固定A + 9次额外B',
    'E1': '无LA / B日常 + 9次额外A',
    'E2': 'LA / 固定A + 9次额外B',
    'E3': 'LA / B日常 + 9次额外A',
    'E5': 'LA / 功能控制A或B更新',
    'J': 'LA / AB联训 + 9次额外A',
    'S': 'LA / B日常 + 90次额外A',
    'Full-1': 'S + 来源与历史修正 / λ=1',
    'Full-3': 'S + 来源与历史修正 / λ=3',
    'Full-10': 'S + 来源与历史修正 / λ=10',
    'Current-10': 'S + 仅当前目标修正 / λ=10',
    'Flat-10': 'S + 统一功能权重 / λ=10',
}
METRICS = ['overall', 'head', 'middle', 'tail', 'nontail']
COLUMNS = dict(overall='overall_acc', head='head20_acc', middle='middle60_acc',
               tail='bottom20_tail_acc', nontail='non_tail_acc')
PARTITIONS = ['client-longtail', 'noniid-labeldir-fine']
DISPLAY_PART = {'client-longtail': 'Client-LT', 'noniid-labeldir-fine': '标准Dirichlet'}
D = Decimal


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as f:
        return list(csv.DictReader(f))


def js(path):
    return json.loads(path.read_text(encoding='utf-8-sig'))


def write(name, rows):
    with (HERE / name).open('w', encoding='utf-8-sig', newline='') as f:
        writer = csv.DictWriter(f, fieldnames=list(dict.fromkeys(k for r in rows for k in r)))
        writer.writeheader()
        writer.writerows(rows)


def mean(values):
    return sum(values, D(0)) / len(values)


def fmt(value, digits=3, signed=False):
    if value is None:
        return '—'
    value = D(str(value)).quantize(D(10) ** -digits, rounding=ROUND_HALF_UP)
    return format(value, ('+' if signed else '') + f'.{digits}f')


def one_run(root, method):
    files = list((root / method.lower()).glob('*/round_metrics.csv'))
    assert len(files) == 1, (root, method, files)
    return files[0].parent


RUNS = [(m, PARTITIONS[0], one_run(CLT, m)) for m in METHODS]
RUNS += [(m, PARTITIONS[1], one_run(DIR_NEW if m in ['E0', 'E1', 'E2', 'E5'] else DIR_OLD, m)) for m in METHODS]
RUNS += [(f'{method.capitalize()}-{lam}', PARTITIONS[0], SFRA / method / f'lambda{lam}_protocol42')
         for method, lam in [('full', 1), ('full', 3), ('full', 10), ('current', 10), ('flat', 10)]]
RUNS.append(('CAPT', PARTITIONS[0], CAPT))
stats, curves, validations, perclass = [], [], [], []

for method, partition, run in RUNS:
    rows = read(run / 'round_metrics.csv')
    if method == 'CAPT':
        for row in rows:
            row['round'] = int(row['epoch']) + 1
    rows.sort(key=lambda r: int(r['round']))
    expected = list(range(1 if method == 'CAPT' else 0, 101))
    assert [int(r['round']) for r in rows] == expected, run
    assert {r['partition'] for r in rows} == {partition}, run
    assert {int(r['seed']) for r in rows} == {42}, run
    completion = js(run / 'completion.json') if method != 'CAPT' else {}
    if completion:
        assert completion['completed_round'] == 100, run
    cfg = js(run / 'control_config.json') if method in METHODS else {}
    if partition == PARTITIONS[1]:
        assert cfg['dirichlet_beta'] == .5 and cfg['topology'] == partition
    acc = []
    for epoch in range(80, 100):
        cc = sorted(read(run / f'per_class_accuracy_epoch_{epoch}.csv'), key=lambda r: int(r['class_id']))
        assert [int(r['class_id']) for r in cc] == list(range(100)), (run, epoch)
        # CIFAR100 test has 100 examples per class; remove only CSV floating-point dust.
        aa = [D(r['per_class_acc']).quantize(D('0.00000001')) for r in cc]
        acc.append(aa)
        r = next(r for r in rows if int(r['round']) == epoch + 1)
        checks = {'overall': mean(aa), 'head': mean(aa[:20]), 'middle': mean(aa[20:80]),
                  'tail': mean(aa[80:]), 'nontail': mean(aa[:80])}
        for key, val in checks.items():
            if COLUMNS[key] in r:
                assert abs(val - D(r[COLUMNS[key]])) < D('0.00000001'), (run, epoch, key)
    class_avg = [mean([a[c] for a in acc]) for c in range(100)]
    values = dict(overall=mean(class_avg), head=mean(class_avg[:20]), middle=mean(class_avg[20:80]),
                  tail=mean(class_avg[80:]), nontail=mean(class_avg[:80]))
    trained = [r for r in rows if int(r['round']) > 0]
    peak = max(trained, key=lambda r: D(r['bottom20_tail_acc']))
    tail100 = D(rows[-1]['bottom20_tail_acc'])
    record = dict(method=method, partition=partition, seed=42, description=DESC[method], **values,
                  tail_peak=D(peak['bottom20_tail_acc']), peak_round=int(peak['round']), tail_final=tail100,
                  drop=D(peak['bottom20_tail_acc']) - tail100,
                  local_steps=completion.get('total_local_optimizer_steps', completion.get('total_optimizer_steps', '')),
                  a_refreshes=completion.get('a_refresh_events', 90 if method in EXTRA else ''),
                  normal_ab_events=completion.get('normal_ab_events', ''),
                  correction_steps=completion.get('functional_correction_steps', 0),
                  source=run.relative_to(ROOT).as_posix())
    stats.append(record)
    for r in rows:
        curves.append(dict(method=method, partition=partition, round=int(r['round']),
                           overall=r['overall_acc'], tail=r['bottom20_tail_acc']))
    perclass += [dict(method=method, partition=partition, class_id=c, last20_accuracy=v) for c, v in enumerate(class_avg)]
    validations.append(dict(method=method, partition=partition, rounds=len(rows), last20_perclass_matches_metrics=True,
                            completion_marker=bool(completion), complete_100_rounds=True))

P = {(r['method'], r['partition']): r for r in stats}
protocol = []
for partition in PARTITIONS:
    selected = [(m, r) for m, p, r in RUNS if p == partition and m in METHODS]
    ref_path = next(r for m, r in selected if m == 'S')
    ref_manifest = read(ref_path / 'partition_manifest.csv')
    ref_schedule = js(ref_path / 'protocol/full_schedule.json')['schedule']
    for method, run in selected:
        protocol.append(dict(method=method, partition=partition,
                             same_partition_rows_as_S=read(run / 'partition_manifest.csv') == ref_manifest,
                             same_schedule_as_S=js(run / 'protocol/full_schedule.json')['schedule'] == ref_schedule))

wide, differences = [], []
for method in ORDER:
    record = dict(method=method, description=DESC[method])
    for partition, prefix in zip(PARTITIONS, ['CLT', 'Dir']):
        r = P.get((method, partition))
        record.update({f'{prefix}_{metric}': r[metric] if r else '' for metric in METRICS + ['drop']})
    if (method, PARTITIONS[1]) in P:
        delta = {k: P[method, PARTITIONS[1]][k] - P[method, PARTITIONS[0]][k] for k in METRICS + ['drop']}
        record['Tail_Dir_minus_CLT'] = delta['tail']
        differences.append(dict(method=method, **delta))
    else:
        record['Tail_Dir_minus_CLT'] = ''
    wide.append(record)

contrasts = []
for name, a, b in [('固定A时加入LA', 'E2', 'E0'), ('定期A时加入LA', 'E3', 'E1'),
                    ('无LA时开放A', 'E1', 'E0'), ('有LA时开放A', 'E3', 'E2'),
                    ('改为日常AB联训', 'J', 'E3'), ('改为密集A交替', 'S', 'E3'),
                    ('加入功能控制', 'E5', 'E3')]:
    r = dict(contrast=name, comparison=f'{a}-{b}')
    for metric in ['overall', 'tail', 'drop']:
        clt = P[a, PARTITIONS[0]][metric] - P[b, PARTITIONS[0]][metric]
        dir_ = P[a, PARTITIONS[1]][metric] - P[b, PARTITIONS[1]][metric]
        r.update({f'CLT_{metric}': clt, f'Dir_{metric}': dir_, f'Dir_minus_CLT_{metric}': dir_ - clt})
    contrasts.append(r)

write('完整结果_长表.csv', stats)
write('完整结果_横向大表.csv', wide)
write('两种划分差值.csv', differences)
write('策略变化对照.csv', contrasts)
write('逐类末20轮均值.csv', perclass)
write('原始训练曲线.csv', curves)
write('协议检查.csv', protocol)
(HERE / '校验记录.json').write_text(json.dumps(validations, ensure_ascii=False, indent=2), encoding='utf-8')

plt.rcParams.update({'font.family': 'Microsoft YaHei', 'axes.unicode_minus': False,
                     'figure.facecolor': 'white', 'savefig.facecolor': 'white'})
NAVY, BLUE, INK, MUTED, LINE = '#17324D', '#0072B2', '#26374A', '#596B7C', '#DAE2EA'


def add_text(ax, x, y, txt, size=15, color=INK, bold=False, ha='left'):
    ax.text(x, y, txt, transform=ax.transAxes, fontsize=size, color=color,
            fontweight='bold' if bold else 'normal', ha=ha, va='center')


def draw_table(methods, filename, title, subtitle, expanded=False):
    height = 15.7 if expanded else 10.8
    fig = plt.figure(figsize=(24, height))
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set_axis_off()
    add_text(ax, .035, .948, title, 29, NAVY, True)
    add_text(ax, .035, .901, subtitle, 15, MUTED)
    edges = [.03, .128, .348, .407, .466, .525, .584, .633, .692, .751, .810, .869, .918, .976]
    group_top, group_h, header_h = .854, .043, .041
    for a, b, label, color in [(edges[0], edges[2], '实验设置', NAVY),
                               (edges[2], edges[7], 'Client-LT', BLUE),
                               (edges[7], edges[12], '标准Dirichlet · β=0.5', NAVY),
                               (edges[12], edges[13], '拓扑差', '#576D7D')]:
        ax.add_patch(Rectangle((a, group_top-group_h), b-a, group_h, color=color, transform=ax.transAxes))
        add_text(ax, (a+b)/2, group_top-group_h/2, label, 17, 'white', True, 'center')
    headers = ['实验', '训练特点', 'Overall', 'Head20', 'Middle60', 'Tail20', '回落↓',
               'Overall', 'Head20', 'Middle60', 'Tail20', '回落↓', 'ΔTail']
    top = group_top-group_h
    ax.add_patch(Rectangle((edges[0], top-header_h), edges[-1]-edges[0], header_h, color='#EDF2F6', transform=ax.transAxes))
    for i, h in enumerate(headers):
        add_text(ax, (edges[i]+edges[i+1])/2, top-header_h/2, h, 15, NAVY, True, 'center')
    top -= header_h
    bottom = .207 if expanded else .247
    row_h = (top-bottom)/len(methods)
    for idx, method in enumerate(methods):
        y = top-idx*row_h
        bg = '#F5F8FA' if idx % 2 else 'white'
        ax.add_patch(Rectangle((edges[0], y-row_h), edges[-1]-edges[0], row_h, color=bg, transform=ax.transAxes))
        if method == 'Full-1' and expanded:
            ax.plot([edges[0], edges[-1]], [y,y], transform=ax.transAxes, color=NAVY, lw=1.4)
        elif method == 'E0' and expanded:
            ax.plot([edges[0], edges[-1]], [y,y], transform=ax.transAxes, color=LINE, lw=1)
        cy = y-row_h/2
        add_text(ax, edges[0]+.008, cy, method, 16, INK, True)
        add_text(ax, edges[1]+.008, cy, DESC[method], 14.5)
        for partition, start in zip(PARTITIONS, [2, 7]):
            r = P.get((method, partition))
            for j, metric in enumerate(['overall', 'head', 'middle', 'tail', 'drop']):
                i = start+j
                value = fmt(r[metric], 2 if metric == 'drop' else 3) if r else '—'
                add_text(ax, (edges[i]+edges[i+1])/2, cy, value, 16 if r else 15,
                         INK if r else '#93A1AE', False, 'center')
        both = (method, PARTITIONS[1]) in P
        delta = P[method, PARTITIONS[1]]['tail']-P[method, PARTITIONS[0]]['tail'] if both else None
        add_text(ax, (edges[-2]+edges[-1])/2, cy, fmt(delta, signed=True), 16,
                 BLUE if both else '#93A1AE', False, 'center')
        ax.plot([edges[0],edges[-1]], [y-row_h,y-row_h], color=LINE, lw=.5, transform=ax.transAxes)
    for x in [edges[2], edges[7], edges[12]]:
        ax.plot([x,x], [top,bottom], color='#B7C5D1', lw=1, transform=ax.transAxes)
    if expanded:
        note_y, spacing = .159, .033
        notes = ['口径：seed42；准确率为第81–100轮均值（%）；回落=Tail训练峰值−第100轮（pp）。ΔTail=Dirichlet−Client-LT（pp）。',
                 '范围：七组LA-control均已有两种划分结果；“—”表示本次未收集到对应结果，不用matched-Dirichlet或其他方法代填。',
                 '可比性：CAPT为不同结构的历史参照；SFRA含额外功能计算；S比E3多81次A阶段，E5含前瞻分支。没有统一计算预算。',
                 '新增标准Dirichlet：E0/E1/E2/E5；其余复用已有结果。Head/Middle/Tail为0–19/20–79/80–99号固定类别。']
    else:
        note_y, spacing = .190, .042
        notes = ['读表：固定A的E0/E2，两种划分Tail相差约0.2 pp；持续开放A后，J和S的Tail差距扩大到11.090和3.635 pp。',
                 '统计：seed42；准确率=第81–100轮均值（%）；回落=Tail训练峰值−第100轮（pp）；ΔTail=Dirichlet−Client-LT（pp）。',
                 '设置：普通类别Dirichlet β=0.5，不匹配客户端容量；不用matched-Dirichlet。LA τ=1；LoRA rank=4；30客户端全参与。',
                 '预算：S额外A为90次，E3为9次；E5含前瞻与分支计算。单种子描述性比较，不据此宣称统计显著性。']
    for i, note in enumerate(notes):
        add_text(ax, .035, note_y-i*spacing, note, 14, MUTED)
    fig.savefig(HERE / filename, dpi=200)
    plt.close(fig)


draw_table(METHODS, '01_LA七组_双划分大表.png', '固定A时拓扑差小；开放A后，Client-LT尾类代价明显扩大',
           'LA-control 七组配对结果  |  本次补齐 E0、E1、E2、E5 的标准Dirichlet结果')
draw_table(ORDER, '02_CAPT_LA_SFRA_完整大表.png', '基线、A/B消融与SFRA：统一口径结果总表',
           'CAPT历史参照 + LA-control七组 + SFRA五组  |  20条已完成训练记录；13种设置', expanded=True)

def md_table(methods):
    lines = ['| 实验 | 训练特点 | CLT Overall | CLT Head | CLT Middle | CLT Tail | CLT回落 | Dir Overall | Dir Head | Dir Middle | Dir Tail | Dir回落 | ΔTail |',
             '|---|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|']
    for m in methods:
        cells = [m, DESC[m]]
        for p in PARTITIONS:
            r = P.get((m,p))
            cells += [fmt(r[k], 2 if k == 'drop' else 3) if r else '—' for k in ['overall','head','middle','tail','drop']]
        delta = P[m,PARTITIONS[1]]['tail']-P[m,PARTITIONS[0]]['tail'] if (m,PARTITIONS[1]) in P else None
        cells.append(fmt(delta, signed=True))
        lines.append('| '+' | '.join(cells)+' |')
    return '\n'.join(lines)


def contrast_md():
    out = ['| 策略变化 | 对比 | CLT Tail变化 | Dir Tail变化 | 两者差：Dir−CLT |', '|---|---|---:|---:|---:|']
    for r in contrasts:
        out.append(f"| {r['contrast']} | {r['comparison']} | {fmt(r['CLT_tail'],signed=True)} | {fmt(r['Dir_tail'],signed=True)} | {fmt(r['Dir_minus_CLT_tail'],signed=True)} |")
    return '\n'.join(out)


report = f'''# 更新大表：Client-LT与标准Dirichlet

日期：2026-09-20。新增包实际包含4组完成结果：E0、E1、E2、E5；结合既有标准Dirichlet E3/J/S后，七组LA-control配对齐全。不是SFRA的Dirichlet结果。没有重新训练或修改原始输出。

## 1. 统计口径

- 准确率统一为第81–100轮均值，逐类文件对应epoch80–99；从原始逐类CSV重算并逐轮核对round_metrics。逐类输入仅清除1e-8以内浮点写出误差，展示采用half-up三位小数。
- Head20：类别0–19；Middle60：20–79；Tail20：80–99。Non-tail80另保留在CSV中，不把它称为Head。
- 回落：第1–100轮Tail最高值减去第100轮Tail，单位pp；不与峰值减末20轮均值混用。
- 标准Dirichlet只使用`noniid-labeldir-fine`、β=0.5。旧`matched-dirichlet`不纳入本表。
- seed42、100轮；单次运行，不画虚构方差。E5前瞻/S额外训练/SFRA功能修正等预算差异保留。
- 同一种划分内部，七组的样本划分记录及客户端参与日程直接比对，见`协议检查.csv`；没有做哈希审查。
- CAPT没有本协议completion标记，已检查100轮曲线及末20轮逐类文件完整。其他19组有100轮完成标记。

## 2. 七组配对大表

{md_table(METHODS)}

![七组配对大表](01_LA七组_双划分大表.png)

## 3. 包含CAPT与SFRA的扩展表

{md_table(ORDER)}

空白表示本次收集范围内没有相应结果，并不替用户断言服务器没有做过。CAPT是不同结构的历史参照；SFRA是额外功能计算的方法版本，不是LA-control的另一种代号。

![完整大表](02_CAPT_LA_SFRA_完整大表.png)

## 4. 这批新增结果补齐了什么

**固定A时，Tail拓扑差很小；开放A以后，Client-LT出现更大的尾类代价。LA改善了两种划分下的类别学习，但没有消除Client-LT在较自由适配下的额外损失。**

E0的标准Dirichlet−CLT Tail差为0.2025 pp，E2为0.1675 pp。E1增至3.2375 pp；J为11.0900 pp，S为3.6350 pp。这是具体实验设置上的观察，不是A只编码客户端知识、或者所有尾类损失只来自A的证明。

{contrast_md()}

相对E3，J的额外Tail拓扑差为10.3275 pp，S为2.8725 pp。这里比较的是策略改变带来的两种划分收益差，不是仅拿两个最终分数作结论；S训练预算同时增加，不能称为纯频率消融。

无LA时，E1相对E0在两种划分都造成明显Tail损失（CLT −10.0425，Dir −7.0075 pp）。因此不能说“标准Dirichlet从不遗忘”。LA后，E3相对E2在两种划分均改善Tail；改为J时CLT损失显著更大。标准Dirichlet的J也并非Tail最优：它的Overall最高，但Tail低于该划分的E3和S。

E5相比E3的Tail变化为CLT −0.105、Dir −0.030 pp，当前表没有显示功能门控优于简单九次A刷新。SFRA只已有本地收集到的CLT五组，不能用本次LA-control Dir结果替代SFRA跨拓扑验证。

## 5. 实验特点与预算说明

- E0/E2：A全程固定，日常B训练3 epoch；10、20、…、90轮各加1个B epoch。区别是CE/LA。
- E1/E3：日常B训练3 epoch；同九个时点，各冻结共同B训练1个A epoch。区别是CE/LA。
- E5：LA、A/B候选、2轮前瞻和功能控制。本次两种划分均接受7次A，第92轮永久冻结，含未选择分支成本。
- J：LA、日常A/B同时训练3 epoch，同时保留九次额外A；不能称为没有额外A的纯联合训练。
- S：LA、日常B训练3 epoch，第1–90轮各加1个A epoch，第91–100轮B-only。
- SFRA：沿S节奏对共享A提案增加3步功能修正；Full有来源优先级和历史目标，Current不使用历史目标，Flat统一功能权重。λ是保持强度，不是划分参数。
- LA只用于训练，tau=1；两种划分都按样本量聚合普通提案。Dir不强制匹配客户端容量，实际steps略有变化。详细步数在`完整结果_长表.csv`。

## 6. 设计与技能检查

使用figure-designer的实验结果设计与完整性审查；此处为精确数值比较，采用分组数值矩阵（大表），而非拥挤的多组柱状图。

1. 类型：实验结果支持图；两种划分并排，方法与训练特点在左，最后一列给出Tail差。
2. 布局：配对表24×10.8英寸、扩展表24×15.7英寸，200dpi；深蓝标题、蓝/深蓝分区、交替浅灰行。
3. 标注：所有单元格为实测重算值；回落和准确率单位分开；不存在的对照标记“—”；不把Full整行加粗为赢家。
4. 工具：Matplotlib可复现渲染，CSV和HTML可编辑；不使用图像生成模型绘制数值。
5. 格式：遵循用户此前PNG、不输出PDF的偏好，覆盖技能默认矢量导出；纸面投稿时可由同一脚本增导矢量。
6. 色盲可读：颜色只作分组辅助，组名、完整数值和符号独立传达信息；无坐标轴截断、无3D、无装饰渐变。
7. 完整性：未选择测试最佳checkpoint；20轮不是20次独立重复；新4组与旧3组来源明确区分。

已逐图检查两张PNG的文字大小、边距、列对齐、缺项与脚注；可见文字无重叠、裁切。通用审查：字体/配色/自包含说明/完整性通过，坐标轴不适用；矢量项按用户PNG偏好豁免。0项CRITICAL、0项MAJOR；投稿时需按版式重新导出矢量。用于大表阅读/汇报，不建议把整张13行宽表缩成论文单栏。

## 7. 文件与复现

- `01_LA七组_双划分大表.png`：七组严格配对，适合本次新增结果汇报。
- `02_CAPT_LA_SFRA_完整大表.png`：加CAPT/SFRA，总计13行设置。
- `完整结果_横向大表.csv`：可用Excel打开；`完整结果_长表.csv`保留精度、预算和每行原始目录。
- `完整大表.html`：浏览器查看、复制表格；`策略变化对照.csv`保留策略差分。
- `校验记录.json`、`协议检查.csv`：数据完整性与分区日程比对。

复现命令：`python presentation/results_table_20260920/build_tables.py`。
'''
(HERE / '大表与说明.md').write_text(report, encoding='utf-8')
headers = ['实验', '训练特点', 'CLT Overall', 'CLT Head20', 'CLT Middle60', 'CLT Tail20', 'CLT回落',
           'Dir Overall', 'Dir Head20', 'Dir Middle60', 'Dir Tail20', 'Dir回落', 'ΔTail']
body = []
for line in md_table(ORDER).splitlines()[2:]:
    cells = [html.escape(v.strip()) for v in line.strip('|').split('|')]
    body.append('<tr>' + ''.join(f'<td>{c}</td>' for c in cells) + '</tr>')
page = '<!doctype html><html lang="zh-CN"><meta charset="utf-8"><title>实验完整大表</title><style>body{font-family:Microsoft YaHei,sans-serif;margin:28px;color:#26374a}table{border-collapse:collapse;white-space:nowrap;font-variant-numeric:tabular-nums}td,th{padding:12px 14px;border-bottom:1px solid #dae2ea;text-align:right}td:nth-child(-n+2),th:nth-child(-n+2){text-align:left}th{background:#17324d;color:white;position:sticky;top:0}tr:nth-child(even){background:#f5f8fa}</style><h1>Client-LT / 标准Dirichlet 实验完整大表</h1><p>准确率：第81–100轮均值（%）；回落：峰值减第100轮（pp）；ΔTail = Dir−CLT。seed42。</p><p>标准Dirichlet β=0.5；不含matched-Dirichlet。“—”为未收集到的对应结果，不是零。</p><table><thead><tr>' + ''.join(f'<th>{h}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join(body) + '</tbody></table><p>CAPT不同结构；SFRA额外功能成本、S额外训练与E5前瞻成本未对齐。方法说明及来源见同目录Markdown/CSV。</p></html>'
(HERE / '完整大表.html').write_text(page, encoding='utf-8')
print(md_table(METHODS))
print('\nProtocol partition/schedule equality:', all(r['same_partition_rows_as_S'] and r['same_schedule_as_S'] for r in protocol))
print(f'Wrote {len(stats)} runs, {len(ORDER)} setting rows, and 2 PNG tables to {HERE}')
