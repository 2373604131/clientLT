"""Render meeting figures from exported metrics only; no model inference."""
import csv
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path
from statistics import mean

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
RUNS = HERE.parent / "la_control/seed42"
REVIEW = HERE.parent / "review"
CAPT = REPO / "output/v2_capt_analysis/output/cifar100_LT/v2_matched/seed42/capt"
plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
                     "font.size": 15, "axes.titlesize": 20, "axes.labelsize": 17,
                     "xtick.labelsize": 13, "ytick.labelsize": 13,
                     "axes.spines.top": False, "axes.spines.right": False,
                     "pdf.fonttype": 42, "savefig.facecolor": "white"})
COLORS = {"e0": "#8D939B", "e1": "#82533C", "e2": "#52A69C", "e3": "#2378B8",
          "e5": "#8C67A8", "j": "#D34B4B", "s": "#E49429", "capt": "#232B35"}
LABELS = {"e0": "E0", "e1": "E1", "e2": "E2", "e3": "E3", "e5": "E5", "j": "J", "s": "S", "capt": "CAPT"}


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def rounded(value, signed=False):
    value = Decimal(str(round(value, 9))).quantize(Decimal("0.001"), rounding=ROUND_HALF_UP)
    return f"{value:+.3f}" if signed else f"{value:.3f}"


def save(fig, name, footer):
    fig.text(.055, .027, footer, fontsize=11, color="#626A73", va="bottom")
    fig.savefig(HERE / f"{name}.png", dpi=200)
    fig.savefig(HERE / f"{name}.pdf")
    plt.close(fig)


perf = {}
for row in read(REVIEW / "verified_performance.csv"):
    topology, method = row["run"].split("/")
    if topology == "client-longtail":
        perf.setdefault(method, {})[row["metric"]] = float(row["last20"])
pc = {e: {int(r["class_id"]): float(r["per_class_acc"]) for r in read(CAPT / f"per_class_accuracy_epoch_{e}.csv")} for e in range(80, 100)}
perf["capt"] = {metric: mean(pc[e][c] for e in range(80, 100) for c in ids) for metric, ids in {
    "overall_acc": range(100), "non_tail_acc": range(80), "head20_acc": range(20),
    "middle60_acc": range(20, 80), "bottom20_tail_acc": range(80, 100)}.items()}
common_footer = "Client-LT，seed42；第81–100轮均值；复用已完成运行；S与其他组预算不相等。"

# 1: Expose the trade-off rather than compressing all results into one rank.
fig, ax = plt.subplots(figsize=(13.33, 7.5))
fig.subplots_adjust(left=.09, right=.96, top=.86, bottom=.14)
ax.set_title("总体提升与尾类保持，不是同一个目标", loc="left", pad=22, fontweight="bold")
cap = perf["capt"]
ax.axvline(cap["overall_acc"], color=COLORS["capt"], ls="--", lw=1.2, alpha=.4)
ax.axhline(cap["bottom20_tail_acc"], color=COLORS["capt"], ls="--", lw=1.2, alpha=.4)
offsets = {"e0": (9, -20), "e1": (9, 9), "e2": (-32, 10), "e3": (-33, 17),
           "e5": (-65, -28), "j": (10, -20), "s": (10, 10), "capt": (10, 10)}
for method in ("e0", "e1", "e2", "e3", "e5", "capt", "j", "s"):
    x, y = perf[method]["overall_acc"], perf[method]["bottom20_tail_acc"]
    ax.scatter(x, y, s=145 if method != "capt" else 190, marker="D" if method == "capt" else "o",
               color=COLORS[method], edgecolors="white", linewidths=1.3, zorder=3)
    ax.annotate(LABELS[method], (x, y), xytext=offsets[method], textcoords="offset points", fontsize=16,
                color=COLORS[method], fontweight="bold", arrowprops={"arrowstyle": "-", "color": COLORS[method], "lw": .7})
ax.text(.035, .9, "E3：尾类保持较好\nJ：总体更高，尾类退化\nS：总体与尾类均高于CAPT参照", transform=ax.transAxes,
        fontsize=16, va="top", bbox={"facecolor": "#F3F5F7", "edgecolor": "none", "pad": 12})
ax.set(xlabel="Overall accuracy (%)", ylabel="Tail accuracy (%)", xlim=(65.3, 71.9), ylim=(55.8, 74.5))
ax.grid(alpha=.15)
save(fig, "01_overall_tail", common_footer + " CAPT为固定每轮聚合的对齐配置，非等计算预算比较。")

# 2: Two lines make the interaction clear; no invented confidence intervals.
fig, axes = plt.subplots(1, 2, figsize=(13.33, 7.5))
fig.subplots_adjust(left=.08, right=.96, top=.78, bottom=.19, wspace=.28)
fig.suptitle("开放 A 的效果，会随训练损失改变", x=.055, ha="left", y=.95, fontsize=24, fontweight="bold")
for ax, metric, title in zip(axes, ("overall_acc", "bottom20_tail_acc"), ("Overall", "Tail")):
    for methods, color, label in ((["e0", "e1"], COLORS["e1"], "普通 CE"), (["e2", "e3"], COLORS["e3"], "LA，τ=1")):
        vals = [perf[m][metric] for m in methods]
        ax.plot([0, 1], vals, marker="o", ms=10, lw=3, color=color, label=f"{label}：变化 {rounded(vals[1]-vals[0], True)} 点")
        for x, method, value in zip([0, 1], methods, vals):
            ax.annotate(f"{LABELS[method]}  {rounded(value)}", (x, value), xytext=(-3 if x else 0, 13 if methods[0] == "e2" else -27),
                        textcoords="offset points", ha="right" if x else "center", color=color, fontsize=15)
    ax.set_xticks([0, 1], ["固定 A\n九次额外 B", "周期更新 A\n九次额外 A"])
    ax.set_xlim(-.33, 1.33)
    ax.set_title(title)
    ax.set_ylabel("Accuracy (%)")
    ax.set_ylim((64.5, 71) if metric == "overall_acc" else (54, 74))
    ax.grid(axis="y", alpha=.15)
    ax.legend(loc="lower left", fontsize=12, frameon=False)
save(fig, "02_la_refresh_interaction", "日常均为B-only；九次额外B或A；总优化步骤相同。复用历史消融，跨线为描述性比较；seed42。")

# 3: Full raw trajectories, not peaks masquerading as final results.
fig, axes = plt.subplots(1, 3, figsize=(16, 7.2))
fig.subplots_adjust(left=.06, right=.98, top=.78, bottom=.17, wspace=.29)
fig.suptitle("较自由的训练曾学会尾类，但后续没有保住", x=.055, ha="left", y=.95, fontsize=24, fontweight="bold")
for method in ("e2", "e3", "s", "j"):
    rows = read(next((RUNS / "client-longtail" / method).glob("*/round_metrics.csv")))
    for ax, metric in zip(axes, ("overall_acc", "non_tail_acc", "bottom20_tail_acc")):
        ax.plot([int(r["round"]) for r in rows], [float(r[metric]) for r in rows],
                label=LABELS[method], color=COLORS[method], lw=2.7)
    if method in ("j", "s"):
        peak = max(rows, key=lambda r: float(r["bottom20_tail_acc"]))
        xy = int(peak["round"]), float(peak["bottom20_tail_acc"])
        axes[-1].scatter(*xy, color=COLORS[method], s=50, zorder=5)
        axes[-1].annotate(f"{LABELS[method]}：{xy[1]:.2f}@{xy[0]}", xy, xytext=(-26, 30 if method == "j" else 10),
                          textcoords="offset points", fontsize=12, color=COLORS[method],
                          arrowprops={"arrowstyle": "-", "color": COLORS[method]})
for ax, title in zip(axes, ("Overall", "Non-tail", "Tail")):
    ax.set(title=title, xlabel="全局轮次", ylabel="Accuracy (%)", xlim=(0, 104))
    ax.grid(alpha=.16)
handles, labels = axes[0].get_legend_handles_labels()
fig.legend(handles, labels, loc="upper center", bbox_to_anchor=(.52, .89), ncol=4, frameon=False, fontsize=14)
axes[-1].axhline(66.35, color="#8D939B", lw=1, ls="--")
axes[-1].text(3, 65.7, "初始 Tail = 66.35", fontsize=11, color="#757B82")
axes[-1].set_ylim(57, 75)
save(fig, "03_training_trajectories", "E2固定A，Tail峰到最终仅降0.05点；曲线未平滑。峰值只用于事后诊断；各策略A活跃量、预算不同。")

# 4: Grouped deltas separate head, middle and tail; bars always start at zero.
fig, ax = plt.subplots(figsize=(13.33, 7.5))
fig.subplots_adjust(left=.09, right=.97, top=.81, bottom=.17)
fig.suptitle("相对 E3：新增适配收益，伴随不同程度的尾类损失", x=.055, ha="left", y=.95, fontsize=23, fontweight="bold")
metrics = ["head20_acc", "middle60_acc", "bottom20_tail_acc"]
x = np.arange(3)
for shift, method in ((-.19, "j"), (.19, "s")):
    vals = [perf[method][metric]-perf["e3"][metric] for metric in metrics]
    bars = ax.bar(x+shift, vals, width=.34, color=COLORS[method], label=f"{LABELS[method]} − E3")
    ax.bar_label(bars, labels=[f"{v:+.3f}" for v in vals], padding=5, fontsize=17)
ax.axhline(0, color="#333A42", lw=1)
ax.set_xticks(x, ["Head20\n最高频20类", "Middle60\n中间60类", "Tail20\n最低频20类"])
ax.set(ylabel="准确率差值（百分点）", ylim=(-13, 7))
ax.legend(frameon=False, fontsize=17, loc="lower left")
ax.grid(axis="y", alpha=.15)
ax.set_axisbelow(True)
save(fig, "04_frequency_group_changes", common_footer + " Non-tail包括Head20与Middle60，不能把80个非尾类全部称为最高频头类。")

# 5: Replace the obsolete matched-Dirichlet / missing-JS slide in place.
standard = REPO / "output/la_control_standard_dirichlet_e3_j_s_analysis/review"
curves = read(standard / "all_round_curves.csv")
retention = {(r['topology'], r['method']): r for r in read(standard / "performance_retention.csv")}
fig, axes = plt.subplots(1, 3, figsize=(16, 9), sharey=True)
fig.subplots_adjust(left=.065, right=.98, top=.79, bottom=.19, wspace=.16)
fig.suptitle("普通 Dirichlet 对照已完成：Client-LT 放大尾类保持代价", x=.045, ha="left", y=.95, fontsize=23, fontweight="bold")
for ax, method in zip(axes, ('e3', 's', 'j')):
    for top, style, label in [('client-longtail', '-', 'Client-LT'), ('noniid-labeldir-fine', '--', '普通 Dirichlet')]:
        rows = sorted((r for r in curves if r['topology'] == top and r['method'] == method), key=lambda r: int(r['round']))
        ax.plot([int(r['round']) for r in rows], [float(r['tail']) for r in rows],
                color=COLORS[method], ls=style, lw=2.7, label=label)
    clt, dr = (retention[t, method] for t in ('client-longtail', 'noniid-labeldir-fine'))
    ax.set(title=LABELS[method], xlabel='全局轮次', xlim=(0, 102), ylim=(57, 76))
    ax.text(.04, .06, f"峰→最终下降\nCLT：{float(clt['peak_to_final']):.2f} 点\nDir：{float(dr['peak_to_final']):.2f} 点",
            transform=ax.transAxes, fontsize=14, bbox=dict(facecolor='white', edgecolor='#DEE3E8', alpha=.9))
    ax.grid(alpha=.15)
    ax.legend(frameon=False, fontsize=12, loc='upper left')
axes[0].set_ylabel('Tail accuracy (%)')
fig.text(.065, .105, '相对 E3，末20轮 Tail 的额外拓扑差距：J +10.3275 点；S +2.8725 点。', fontsize=17)
save(fig, "05_topology_evidence_boundary", "普通 noniid-labeldir-fine，β=0.5；同全局样本池，不匹配客户端容量；seed42。Dir也有回落，不能写“完全不遗忘”。")

# 6: Every tail class, in class-id order; not a hand-picked failure gallery.
diff = read(REVIEW / "per_class_differences.csv")
fig, axes = plt.subplots(1, 2, figsize=(16, 8.5))
fig.subplots_adjust(left=.16, right=.92, top=.84, bottom=.11, wspace=.13)
fig.suptitle("逐类复核：尾类下降并非个别异常", x=.055, ha="left", y=.95, fontsize=24, fontweight="bold")
for ax, method in zip(axes, ("j", "s")):
    rows = sorted([r for r in diff if r["comparison"] == f"{method}-e3" and int(r["class_id"]) >= 80], key=lambda r: int(r["class_id"]))
    vals = np.array([float(r["last20_difference"]) for r in rows]).reshape(20, 1)
    im = ax.imshow(vals, cmap="RdBu", vmin=-45, vmax=45, aspect="auto")
    ax.set_title(f"{LABELS[method]} − E3：{sum(vals[:,0]<0)}/20 类下降", pad=12)
    ax.set_xticks([])
    ax.set_yticks(np.arange(20), [f"{r['class_id']}  {r['class_name']}" for r in rows] if method == "j" else [""]*20)
    ax.tick_params(axis="y", length=0, labelsize=11)
    for i, value in enumerate(vals[:, 0]):
        ax.text(0, i, f"{value:+.2f}", ha="center", va="center", fontsize=12, color="white" if abs(value)>25 else "#202733")
    for spine in ax.spines.values():
        spine.set_visible(False)
cb = fig.colorbar(im, ax=axes, fraction=.025, pad=.025)
cb.set_label("准确率差值（百分点）")
save(fig, "06_all_tail_classes", "末20轮逐类均值差；按类别编号展示全部20个尾类，没有按受损程度筛选。差值不是独立随机种子，不能据此声称统计显著。")

rows = [{"method": LABELS[m], **perf[m]} for m in ("capt", "e0", "e1", "e2", "e3", "e5", "j", "s")]
with (HERE / "slide_source_metrics.csv").open("w", encoding="utf-8-sig", newline="") as stream:
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
print("Saved 6 PNG figures, 6 vector PDFs, and slide_source_metrics.csv to", HERE)
