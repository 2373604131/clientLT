"""Meeting update: existing CSV results plus explicitly labelled proposal diagrams.

No training, model inference, checkpoint audit, or invented method results.
Run from the repository root: python output/la_control_ablation_analysis/presentation/make_closure_figures.py
"""
import csv
import json
from pathlib import Path
import runpy

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
STANDARD = REPO / 'output/la_control_standard_dirichlet_e3_j_s_analysis/review'
ROLES = REPO / 'output/factor_roles_analysis/review'

# Regenerate the six existing figures; figure 05 now uses ordinary Dirichlet.
runpy.run_path(str(HERE / 'make_figures.py'), run_name='__main__')
plt.rcParams.update({'font.family': 'Microsoft YaHei', 'font.size': 15,
    'axes.unicode_minus': False, 'axes.spines.top': False, 'axes.spines.right': False,
    'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
CLT, DIR = 'client-longtail', 'noniid-labeldir-fine'
BLUE, GREEN, ORANGE, RED, GREY = '#2378B8', '#38947C', '#E49429', '#D34B4B', '#657483'


def read(path):
    with path.open(encoding='utf-8-sig', newline='') as stream:
        return list(csv.DictReader(stream))


def write(name, rows):
    with (HERE/name).open('w', encoding='utf-8-sig', newline='') as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def save(fig, name, footer):
    fig.text(.045, .025, footer, fontsize=11, color=GREY, va='bottom')
    fig.savefig(HERE/f'{name}.png', dpi=200)
    fig.savefig(HERE/f'{name}.pdf')
    plt.close(fig)


partitions = {r['topology']: r for r in read(STANDARD/'partition_characteristics.csv')}
performance = read(STANDARD/'performance_retention.csv')
roles = read(ROLES/'effect_summary.csv')
write('source_standard_dirichlet.csv', performance)
write('source_partition_structure.csv', list(partitions.values()))
write('source_factor_roles.csv', [r for r in roles if r['metric'] in ('tail20', 'probe_tail_logodds')])

# 07: Show the actual client distribution, not a stylized Dirichlet cartoon.
fig, axes = plt.subplots(2, 2, figsize=(16, 9), sharex=True, sharey='row')
fig.subplots_adjust(left=.07, right=.97, top=.82, bottom=.23, hspace=.27, wspace=.16)
fig.suptitle('相同尾类样本总量，不同客户端证据分布', x=.045, ha='left', y=.96, fontsize=25, fontweight='bold')
for j, (top, title) in enumerate([(CLT, 'Client-LT'), (DIR, '普通 Dirichlet，β=0.5')]):
    r = partitions[top]
    sizes, tails = json.loads(r['client_sizes']), json.loads(r['tail_samples_per_client'])
    colors = [ORANGE if top == CLT and i >= 27 else BLUE for i in range(30)]
    axes[0, j].bar(range(30), sizes, color=colors)
    axes[1, j].bar(range(30), tails, color=colors)
    axes[0, j].set_title(title, pad=16)
    axes[0, j].set_ylim(0, 690)
    axes[1, j].set_ylim(0, 48)
    axes[1, j].set(xlabel='客户端编号', xticks=[0, 5, 10, 15, 20, 25, 29])
    for ax in axes[:, j]:
        ax.grid(axis='y', alpha=.15)
        ax.set_axisbelow(True)
axes[0, 0].set_ylabel('全部训练样本数')
axes[1, 0].set_ylabel('尾类训练样本数')
fig.text(.07, .095, 'CLT：77.78%的尾类样本集中在3个小客户端；其样本量聚合权重合计1.20%。', fontsize=18)
save(fig, '07_client_evidence_distribution', '两种划分均10847张训练样本，其中Tail20共153张；平均每尾类覆盖客户端数：3.65 vs 6.05。权重比例不是梯度贡献比例。')

# 08: Disclose the negative role result; this is not a replacement success endpoint.
fig, axes = plt.subplots(1, 2, figsize=(16, 9), sharey=True)
fig.subplots_adjust(left=.08, right=.97, top=.78, bottom=.22, wspace=.18)
fig.suptitle('短分支诊断：没有证实 A/B 硬分工，不否定 A 作为方法入口', x=.045, ha='left', y=.96, fontsize=23, fontweight='bold')
for ax, mode, title in zip(axes, ('raw', 'norm_matched'), ('原始更新', '有效权重改动幅度匹配')):
    for offset, factor, color in [(-.18, 'A', BLUE), (.18, 'B', ORANGE)]:
        vals = [float(next(r['mean'] for r in roles if r['mode'] == mode and r['metric'] == 'probe_tail_logodds'
                         and r['topology'] == CLT and r['factor'] == factor and r['correction'] == kind))*1000
                for kind in ('client', 'class')]
        bars = ax.bar(np.arange(2)+offset, vals, width=.34, color=color, label=factor)
        ax.bar_label(bars, fmt='%+.3f', padding=6, fontsize=14)
    ax.set_title(title)
    ax.set_xticks([0, 1], ['等权 − 样本量聚合\nLA固定', 'LA − CE\n样本量聚合固定'])
    ax.axhline(0, color=GREY, lw=1)
    ax.set_ylim(-.8, 6.7)
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
    ax.legend(frameon=False, loc='upper left')
axes[0].set_ylabel('尾类 probe log-odds 变化 × 1000')
fig.text(.08, .115, '准确率联合模式：原始0/6，等幅0/6；Client-LT等权修正的Tail准确率变化均为0。', fontsize=17)
save(fig, '08_factor_roles_boundary', '六个相关起点的均值，非独立种子；200张训练外probe。连续指标不是准确率，不替代主判据；这里只展示CLT分支。')


def box(ax, xy, wh, title, body, color, status=None, size=16):
    x, y = xy
    w, h = wh
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle='round,pad=0.012',
        linewidth=1.5, edgecolor=color, facecolor=color+'0E', transform=ax.transAxes))
    ax.text(x+.018, y+h-.035, title, ha='left', va='top', fontsize=19, fontweight='bold', color=color, transform=ax.transAxes)
    ax.text(x+.018, y+h-.105, body, ha='left', va='top', fontsize=size, linespacing=1.65, transform=ax.transAxes)
    if status:
        ax.text(x+.018, y+.025, status, va='bottom', fontsize=12, color=color, transform=ax.transAxes)


def arrow(ax, start, end, color=GREY):
    ax.add_patch(FancyArrowPatch(start, end, transform=ax.transAxes,
        arrowstyle='-|>', mutation_scale=20, linewidth=2, color=color))


# 09: A code-native vector schematic, explicitly separating evidence and proposal.
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_axis_off()
fig.subplots_adjust(left=.035, right=.98, bottom=.08, top=.86)
fig.suptitle('核心方法切入点：让 A 继续学，同时保护稀缺客户端的有效贡献', x=.045, ha='left', y=.96, fontsize=24, fontweight='bold')
box(ax, (.025, .50), (.265, .43), '已观察：稳定—适配矛盾',
    '固定A：Tail基本不遗忘\n持续开放A：适配更充分\nClient-LT：尾类代价更大', BLUE, 'E2 / E3 / S / J + 普通Dir对照')
box(ax, (.36, .50), (.265, .43), '待验证：功能贡献失衡',
    '本地获得的有益能力\n在聚合或后续轮次受损\n不是预先认定正贡献减少', ORANGE, '正贡献减少 / 负贡献增加，分别测量')
box(ax, (.695, .50), (.265, .43), '拟方法：修正 A 聚合方向',
    '本地功能缺口 → 保护方向\n普通A候选 → 冲突修正\nB训练与LA保持原样', GREEN, '候选设计，尚未实现、尚无方法成绩')
arrow(ax, (.294, .72), (.352, .72))
arrow(ax, (.63, .72), (.687, .72))
box(ax, (.13, .06), (.34, .30), '结果闭环', '学得比固定A充分\n遗忘比常规持续A更新少', BLUE, size=17)
box(ax, (.54, .06), (.34, .30), '机制闭环', '独立诊断上：有益功能保留更多\n优于同等更新幅度的普通方向', GREEN, size=17)
arrow(ax, (.825, .48), (.71, .38))
arrow(ax, (.70, .48), (.31, .38))
save(fig, '09_method_closure', '不宣称A储存客户端知识；不把冻结或投影本身当作已验证的新颖贡献。最终方法需要实际训练与机制结果支持。')

# 10: The paired continuation design includes a necessary fresh RNG baseline.
fig, ax = plt.subplots(figsize=(16, 9))
ax.set_axis_off()
fig.subplots_adjust(left=.04, right=.98, bottom=.08, top=.86)
fig.suptitle('下一步只围绕方法：同起点、多轮续训、同步记录机制', x=.045, ha='left', y=.96, fontsize=25, fontweight='bold')
box(ax, (.025, .65), (.925, .27), '共同起点：已有 S 第20轮结束状态',
    '保留原样本池 / LA / B训练 / A日程；继续第21–100轮，不再做单次聚合即结束的诊断。', BLUE, size=16)
for x, title, body, color in [(.025, 'R：常规续训', 'A仍按样本量聚合\n同新分支随机流\n不是重跑完整S', GREY),
                             (.345, 'P：保护式 A 聚合', '相同本地A更新\n按功能保护方向修正\n只改变服务器A更新', GREEN),
                             (.665, 'K：同状态缩幅对照', '自身起点计算P候选幅度\n只缩放普通A方向\n区分保护方向与少更新', ORANGE)]:
    box(ax, (x, .20), (.285, .34), title, body, color, size=16)
    arrow(ax, (x+.14, .635), (x+.14, .56))
ax.text(.03, .08, '先Client-LT三支；有信号后补普通Dir三支及功能信号消融；最后扩独立种子。', fontsize=18, transform=ax.transAxes)
save(fig, '10_next_experiments', 'R/P/K均为待实现的续训分支；旧事件没有中途RNG，不能直接把旧S后半段冒充同随机流对照。新方法不读官方测试或离线probe。')

# 11: Difference of strategy effects using only ordinary Dirichlet results.
perf = {(r['topology'], r['method']): r for r in performance}
fig, axes = plt.subplots(1, 2, figsize=(16, 9))
fig.subplots_adjust(left=.08, right=.97, top=.79, bottom=.20, wspace=.25)
fig.suptitle('相同策略改变：两边都更会学习，尾类保持代价却不对称', x=.045, ha='left', y=.96, fontsize=24, fontweight='bold')
interaction_rows = []
for ax, metric, title in zip(axes, ('overall', 'tail'), ('Overall变化', 'Tail变化')):
    for offset, topology, color, label in [(-.18, CLT, BLUE, 'Client-LT'), (.18, DIR, GREEN, '普通Dirichlet')]:
        vals = [float(perf[topology, m][metric])-float(perf[topology, 'e3'][metric]) for m in ('j', 's')]
        bars = ax.bar(np.arange(2)+offset, vals, width=.34, color=color, label=label)
        ax.bar_label(bars, labels=[f'{v:+.4f}' for v in vals], padding=5, fontsize=15)
        for method, value in zip(('j', 's'), vals):
            interaction_rows.append(dict(topology=topology, comparison=f'{method}-e3', metric=metric, difference_pp=value))
    ax.set_xticks([0, 1], ['E3 → J', 'E3 → S'])
    ax.set_title(title)
    ax.set_ylabel('末20轮准确率变化（百分点）')
    ax.axhline(0, color=GREY, lw=1)
    ax.grid(axis='y', alpha=.15)
    ax.set_axisbelow(True)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc='upper center', bbox_to_anchor=(.52, .895), ncol=2, frameon=False, fontsize=14)
axes[0].set_ylim(-.45, 3.65)
axes[1].set_ylim(-13.8, 2.5)
fig.text(.08, .105, '额外尾类拓扑代价：J +10.3275 pp；S +2.8725 pp。', fontsize=19)
save(fig, '11_strategy_topology_interaction', 'seed42，第81–100轮均值；同全局样本池。S同时增加训练预算，此图识别完整策略差异，不是纯刷新频率效应。')
write('source_strategy_interactions.csv', interaction_rows)
print('Updated 11 PNG + 11 PDF figures in:', HERE)
