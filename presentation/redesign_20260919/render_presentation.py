"""Redraw the advisor presentation from existing CSVs; PNG output only.

No model loading, training, test-set selection, or PDF generation.
Diagrams/layout: Pillow primitives. Quantitative panels: Matplotlib.
Run at repo root: python presentation/redesign_20260919/render_presentation.py
"""
from __future__ import annotations

import csv
import io
import json
from decimal import Decimal, ROUND_HALF_UP
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
SOURCE = HERE.parent
STANDARD = REPO / "output/la_control_standard_dirichlet_e3_j_s_analysis/review"
ABLATION = REPO / "output/la_control_ablation_analysis"
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
W, H = 2560, 1440
NAVY, BLUE, ORANGE = "#10294F", "#2262B8", "#C87318"
RED, TEAL, GREY = "#B84447", "#237F75", "#65758A"
PALE, BORDER = "#F7F9FD", "#CBD7E7"
COLORS = dict(E0=GREY, E1=GREY, E2=TEAL, E3=BLUE,
              E5=GREY, J=RED, S=ORANGE, CAPT=NAVY)
STYLES = dict(E2=("-", "o"), E3=("--", "s"), S=("-.", "^"), J=(":", "D"))
FONT = "C:/Windows/Fonts/msyh.ttc"
BOLD = "C:/Windows/Fonts/msyhbd.ttc"
plt.rcParams.update({"font.family": "Microsoft YaHei", "axes.unicode_minus": False,
    "font.size": 12, "axes.labelsize": 13, "axes.titlesize": 15,
    "xtick.labelsize": 11, "ytick.labelsize": 11, "legend.fontsize": 11,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#7A899A", "axes.labelcolor": NAVY, "text.color": NAVY,
    "xtick.color": GREY, "ytick.color": GREY, "axes.linewidth": .8,
    "savefig.facecolor": "white"})
LAYOUT_WARNINGS = []
GENERATED = []


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def snapshot(name, rows):
    with (DATA / name).open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def fmt(x, signed=False, digits=3):
    value = Decimal(str(round(float(x), 10))).quantize(Decimal(10) ** -digits,
                                                      rounding=ROUND_HALF_UP)
    return f"{value:+.{digits}f}" if signed else f"{value:.{digits}f}"


PERF_ROWS = read(SOURCE / "slide_source_metrics.csv")
PERF = {r["method"]: {k: float(v) for k, v in r.items() if k != "method"} for r in PERF_ROWS}
RET_ROWS = read(STANDARD / "performance_retention.csv")
RET = {(r["topology"], r["method"].upper()): r for r in RET_ROWS}
PART_ROWS = read(STANDARD / "partition_characteristics.csv")
PART = {r["topology"]: r for r in PART_ROWS}
ROLE = read(SOURCE / "source_factor_roles.csv")
DIFF = [r for r in read(ABLATION / "review/per_class_differences.csv")
        if r["comparison"] in ("j-e3", "s-e3") and int(r["class_id"]) >= 80]
CURVE = read(STANDARD / "all_round_curves.csv")
CLT, DIR = "client-longtail", "noniid-labeldir-fine"
e2_path = next((ABLATION / "la_control/seed42/client-longtail/e2").glob("*/round_metrics.csv"))
for r in read(e2_path):
    CURVE.append(dict(topology=CLT, method="e2", round=r["round"], overall=r["overall_acc"],
        head20="", middle60="", tail=r["bottom20_tail_acc"],
        nontail=r["non_tail_acc"]))


def curve(top, method):
    return sorted((r for r in CURVE if r["topology"] == top and r["method"] == method.lower()),
                  key=lambda r: int(r["round"]))


def series(top, method, key):
    rows = curve(top, method)
    return np.array([int(r["round"]) for r in rows]), np.array([float(r[key]) for r in rows])


CHECKS = []
for top in (CLT, DIR):
    for method in ("E3", "J", "S"):
        rows = curve(top, method)
        assert [int(r["round"]) for r in rows] == list(range(101))
        for metric in ("overall", "head20", "middle60", "tail", "nontail"):
            value = np.mean([float(r[metric]) for r in rows if int(r["round"]) >= 81])
            assert np.isclose(value, float(RET[top, method][metric]), atol=1e-8)
CHECKS.append("Six standard-topology curves contain rounds 0..100; last20 means match summaries.")
for method in ("E2", "E3", "J", "S"):
    _, tail = series(CLT, method, "tail")
    assert np.isclose(np.mean(tail[-20:]), PERF[method]["bottom20_tail_acc"], atol=1e-8)
for method in ("J", "S"):
    rows = [r for r in DIFF if r["comparison"] == method.lower() + "-e3"]
    assert len(rows) == 20
    assert np.isclose(np.mean([float(r["last20_difference"]) for r in rows]),
        PERF[method]["bottom20_tail_acc"] - PERF["E3"]["bottom20_tail_acc"], atol=1e-8)
CHECKS.append("Client-LT last20 values and all 20 per-class tail deltas match the main table.")
for p in PART_ROWS:
    assert sum(json.loads(p["client_sizes"])) == int(p["total_samples"]) == 10847
    assert sum(json.loads(p["tail_samples_per_client"])) == int(p["tail_samples"]) == 153
CHECKS.append("Both partitions contain 10847 training samples and 153 tail samples.")
INTERACTIONS = []
for method in ("J", "S"):
    for metric in ("overall", "tail"):
        for top in (CLT, DIR):
            INTERACTIONS.append(dict(topology=top, method=method, metric=metric,
                delta_vs_e3=float(RET[top, method][metric])-float(RET[top, "E3"][metric])))
for filename, rows in [("performance_clientlt.csv", PERF_ROWS), ("retention.csv", RET_ROWS),
    ("partition_structure.csv", PART_ROWS), ("factor_effects.csv", ROLE),
    ("tail_class_deltas.csv", DIFF), ("training_curves.csv", CURVE),
    ("strategy_interactions.csv", INTERACTIONS)]:
    snapshot(filename, rows)


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def wrapped(text, ft, width):
    lines = []
    for paragraph in str(text).split("\n"):
        line = ""
        for char in paragraph:
            if ft.getlength(line + char) > width and line:
                lines.append(line.rstrip())
                line = char.lstrip()
            else:
                line += char
        lines.append(line)
    return lines


class Slide:
    def __init__(self, name, chapter, title, subtitle, takeaway, source):
        self.name = name
        self.im = Image.new("RGB", (W, H), "white")
        self.d = ImageDraw.Draw(self.im)
        self.text(100, 42, chapter, 25, BLUE, bold=True)
        self.text(100, 92, title, 61, NAVY, bold=True, width=2360)
        self.text(104, 178, subtitle, 29, GREY, width=2310)
        self.d.line((100, 235, 2460, 235), fill=BLUE, width=2)
        self.d.line((1180, 235, 1380, 235), fill=BLUE, width=8)
        self.d.rounded_rectangle((100, 1215, 2460, 1320), radius=18,
                                 fill="#FFF7F5", outline="#E9BBB5", width=2)
        self.icon("target", 133, 1245, RED, 42)
        self.text(210, 1235, takeaway, 35, RED, bold=True, width=2180, max_y=1310)
        self.text(100, 1350, source, 22, GREY, width=2240, max_y=1428)
        self.text(2370, 1360, name[:2].upper(), 29, BLUE, bold=True)

    def text(self, x, y, text, size=32, color=NAVY, bold=False, width=None,
             max_y=None, anchor=None, spacing=1.42):
        ft = font(size, bold)
        if anchor:
            self.d.text((x, y), str(text), font=ft, fill=color, anchor=anchor)
            return y + size * spacing
        lines = wrapped(text, ft, width or (W-x-80))
        bottom = y + len(lines) * size * spacing
        if bottom > (max_y or H):
            LAYOUT_WARNINGS.append(f"{self.name}: text overflow {text[:35]} ({bottom:.0f})")
        for line in lines:
            self.d.text((x, y), line, font=ft, fill=color)
            y += size * spacing
        return y

    def panel(self, x, y, w, h, title=None, color=BLUE, fill=PALE, number=None):
        self.d.rounded_rectangle((x, y, x+w, y+h), radius=20,
                                 fill=fill, outline=BORDER, width=2)
        if title:
            shift = 34
            if number is not None:
                self.d.ellipse((x+25, y+23, x+77, y+75), fill=color)
                self.text(x+51, y+46, str(number), 29, "white", True, anchor="mm")
                shift = 97
            self.text(x+shift, y+22, title, 35, color, bold=True, width=w-shift-25)
            self.d.line((x+30, y+91, x+w-30, y+91), fill=color, width=2)

    def bullets(self, x, y, w, items, color=BLUE, size=31, gap=35, max_y=1180):
        for heading, body in items:
            self.d.ellipse((x, y+13, x+12, y+25), fill=color)
            y = self.text(x+30, y, heading, size, color, bold=True, width=w-30, max_y=max_y)
            if body:
                y = self.text(x+30, y+7, body, size-3, NAVY, width=w-30, max_y=max_y)
            y += gap

    def stat(self, x, y, value, label, color=BLUE, sub=None):
        self.text(x, y, value, 69, color, bold=True)
        self.text(x, y+97, label, 29, NAVY, width=630)
        if sub:
            self.text(x, y+145, sub, 25, GREY, width=630)

    def arrow(self, start, end, color=BLUE, width=5):
        x1, y1 = start
        x2, y2 = end
        self.d.line((x1, y1, x2, y2), fill=color, width=width)
        angle = np.arctan2(y2-y1, x2-x1)
        a, b = angle+2.65, angle-2.65
        self.d.polygon([(x2, y2), (x2+22*np.cos(a), y2+22*np.sin(a)),
                        (x2+22*np.cos(b), y2+22*np.sin(b))], fill=color)

    def icon(self, kind, x, y, color=BLUE, s=74):
        d = self.d
        if kind == "server":
            for k in range(3):
                yy = y+k*s*.31
                d.rounded_rectangle((x, yy, x+s, yy+s*.23), radius=5, outline=color, width=3)
                d.ellipse((x+9, yy+6, x+17, yy+14), fill=color)
        elif kind == "target":
            for r in (s*.47, s*.29, s*.10):
                d.ellipse((x+s/2-r, y+s/2-r, x+s/2+r, y+s/2+r), outline=color, width=3)
            d.line((x+s*.6, y+s*.4, x+s, y), fill=color, width=4)
        elif kind == "shield":
            pts = [(x+s*.5, y), (x+s, y+s*.2), (x+s*.87, y+s*.72),
                   (x+s*.5, y+s), (x+s*.13, y+s*.72), (x, y+s*.2)]
            d.line(pts+[pts[0]], fill=color, width=4)
            d.line((x+s*.27, y+s*.5, x+s*.44, y+s*.67, x+s*.75, y+s*.32), fill=color, width=5)
        elif kind == "chart":
            d.line((x, y, x, y+s, x+s, y+s), fill=color, width=3)
            d.line((x+8, y+s*.78, x+s*.35, y+s*.50, x+s*.63, y+s*.6, x+s*.94, y+s*.1), fill=color, width=5)
        elif kind == "lock":
            d.arc((x+s*.2, y, x+s*.8, y+s*.75), 180, 360, fill=color, width=5)
            d.rounded_rectangle((x+s*.08, y+s*.37, x+s*.92, y+s), radius=8, outline=color, width=4)
            d.line((x+s*.5, y+s*.60, x+s*.5, y+s*.82), fill=color, width=5)

    def plot(self, box, painter):
        x, y, w, h = box
        fig = plt.figure(figsize=(w/160, h/160), dpi=160)
        painter(fig)
        buffer = io.BytesIO()
        fig.savefig(buffer, format="png", dpi=160)
        plt.close(fig)
        buffer.seek(0)
        panel = Image.open(buffer).convert("RGB")
        assert abs(panel.width-w) <= 1 and abs(panel.height-h) <= 1
        self.im.paste(panel, (x, y))

    def save(self):
        self.im.save(HERE / (self.name+".png"), optimize=True)
        GENERATED.append(self.name)
        print("Rendered", self.name, flush=True)


def axis(ax, ylabel=None):
    ax.grid(axis="y", color="#D8E0EA", alpha=.7, lw=.7)
    ax.set_axisbelow(True)
    if ylabel:
        ax.set_ylabel(ylabel)


def method_line(ax, top, method, metric="tail", label=None, lw=2.1):
    x, y = series(top, method, metric)
    style, marker = STYLES[method]
    ax.plot(x, y, label=label or method, color=COLORS[method], ls=style,
            marker=marker, markevery=12, ms=4, lw=lw)


# 01: a narrative bridge, not a fabricated new-method performance teaser.
s = Slide("01_story_bridge", "研究主线 / 从旧问题走向新方法", "从“证据在哪里”，走向“学到后如何留住”",
          "保留客户端拓扑主线；在 LoRA A/B 框架中重新建立证据，随后设计核心方法。",
          "下一步的重点：让 A 继续学习，同时减少对稀缺客户端有效贡献的损伤。",
          "状态说明：左栏为历史研究动机，中栏为当前已完成结果，右栏为待验证的方法目标。")
for x, title, color, num, ico in [(100, "原始问题：证据位置", BLUE, 1, "server"),
    (905, "当前证据：保持差异", ORANGE, 2, "chart"), (1710, "方法目标：学习且保持", TEAL, 3, "shield")]:
    s.panel(x, 285, 750, 780, title, color, number=num)
    s.icon(ico, x+325, 425, color, 95)
s.bullets(145, 565, 650, [("类别频率之外，还有客户端分布", "同一类的证据由哪些客户端持有？"),
    ("旧研究给出机制线索", "Prompt 阶段的归因是动机来源；不能直接充当当前 LoRA 的归因结果。")], size=32)
s.bullets(950, 565, 650, [("固定 A：稳定但适配有限", "E2 尾类峰到最终仅回落 0.05 pp。"),
    ("持续适配：Client-LT 代价更大", "J：12.15 pp 回落；普通 Dir：1.80 pp。")], color=ORANGE, size=32)
s.bullets(1755, 565, 650, [("干预 A 的更新与聚合", "保留充分适配能力，避免共享更新反复损伤已有尾类功能。"),
    ("先做出有效方法，再补齐闭环", "机制记录随方法训练一起采集，不先扩大角色验证实验。")], color=TEAL, size=32)
s.arrow((855, 690), (898, 690), GREY)
s.arrow((1660, 690), (1703, 690), GREY)
s.text(125, 1110, "已有现象", 30, BLUE, True)
s.text(365, 1110, "≠ 已排他确认根因", 30, GREY)
s.text(1050, 1110, "方法切入点已明确", 30, TEAL, True)
s.text(1450, 1110, "≠ 方法已验证成功", 30, GREY)
s.save()

# 02: real client counts, not an idealized partition cartoon.
s = Slide("02_client_evidence", "问题结构 / 同样数量，不同位置", "尾类样本一样多，进入共享模型的条件却不同",
          "Client-LT 与普通 Dirichlet 使用相同的全局训练池；柱形展示真实客户端清单。",
          "长尾不只是“总共有多少”，还包括“分布在哪里、由多少客户端支撑”。",
          "数据：partition_structure.csv；普通 Dir = noniid-labeldir-fine，β=0.5。权重占比不是实际梯度贡献占比。")
s.panel(100, 285, 1605, 880, "实际客户端分布", BLUE)
def partition_plot(fig):
    axs = fig.subplots(2, 2)
    fig.subplots_adjust(left=.065, right=.985, bottom=.105, top=.91, hspace=.37, wspace=.18)
    for j, (top, name) in enumerate([(CLT, "Client-LT"), (DIR, "普通 Dirichlet")]):
        p = PART[top]
        for i, (key, label, lim) in enumerate([("client_sizes", "全部样本数", 700),
            ("tail_samples_per_client", "尾类样本数", 50)]):
            a = axs[i, j]
            vals = json.loads(p[key])
            a.bar(np.arange(30), vals, color=[ORANGE if top == CLT and k>=27 else BLUE for k in range(30)], width=.75)
            if top == CLT:
                a.axvspan(26.5, 29.6, facecolor="#FFF4DF", alpha=.6, zorder=0)
            a.set_ylim(0, lim)
            a.set_xticks([0, 10, 20, 29])
            if i == 0:
                a.set_title(name)
            else:
                a.set_xlabel("客户端编号")
            axis(a, label if j == 0 else None)
s.plot((125, 395, 1550, 740), partition_plot)
s.panel(1750, 285, 710, 880, "证据集中，但名义权重很小", ORANGE)
s.stat(1800, 410, "77.78%", "尾类图片位于 CLT 的 3 个小客户端", ORANGE, "119 / 153 张；客户端编号 27–29")
s.stat(1800, 650, "1.20%", "这 3 个客户端的样本量聚合权重", RED, "130 / 10847；不是全部尾类支持者")
s.text(1800, 887, "平均每个尾类的客户端覆盖数", 29, NAVY, True, width=620)
s.text(1800, 950, "CLT  3.65    /    Dir  6.05", 38, BLUE, True, width=620)
s.text(1800, 1030, "两边均为 10847 张训练图，Tail20 共 153 张。", 28, GREY, width=620)
s.save()

# 03: clarify what A/B separation means, and how J differs from S.
s = Slide("03_ab_protocol", "实验框架 / 先看清楚我们改变了什么", "比较的不是 A 放在哪，而是 A 如何参与更新",
          "服务器维护共享 A/B；客户端从共同状态开始本地训练，再上传更新。LA 加在本地损失中。",
          "E2 → E3 → S/J 比较训练策略；不能把所有差异解释为纯粹的 A 刷新频率效应。",
          "依据：run_cliplora_la_control.py 与现有实验协议；rank=4，30/30 客户端，100 轮。J 与 S 不等计算预算。")
s.panel(100, 285, 2360, 295, "日常路径：B-only 阶段", BLUE)
steps = [(145, "共同全局状态", "共享 A、B"), (725, "客户端本地训练", "固定 A；训练 B 3 epochs"),
         (1310, "聚合本地 B", "按客户端样本量平均"), (1900, "额外更新阶段", "由实验策略决定")]
for x, title, body in steps:
    s.text(x, 418, title, 35, BLUE, True, width=480)
    s.text(x, 480, body, 29, NAVY, width=490)
for start, end in [(610, 690), (1200, 1275), (1790, 1865)]:
    s.arrow((start, 466), (end, 466))
rows = [("E2", "B-only", "第 10,20,…,90 轮额外训练 B", "A 始终固定"),
        ("E3", "B-only", "第 10,20,…,90 轮额外训练 A", "9 次 A 刷新"),
        ("S", "B-only", "第 1–90 轮每轮额外训练 A", "90 次 A 刷新"),
        ("J", "A/B 同时训练", "另保留 9 次只训练 A", "日常联训 + 稀疏刷新")]
for i, (method, daily, extra, tag) in enumerate(rows):
    y = 620+i*130
    s.panel(100, y, 2360, 110, fill="white" if i%2 else PALE)
    s.text(140, y+26, method, 39, COLORS[method], True)
    s.text(340, y+32, daily, 32, NAVY, True, width=450)
    s.text(890, y+32, extra, 32, NAVY, width=940)
    s.text(1900, y+33, tag, 29, COLORS[method], True, width=515)
s.text(140, 1160, "上述四组均有 LA（τ=1）；额外阶段每次 1 epoch。E0/E1 分别是 E2/E3 的无 LA 对照。", 27, GREY, width=2300)
s.save()

# 04: retain learning curves and inspect the whole trajectory.
s = Slide("04_learning_retention", "核心现象 / 学会之后为什么又退回去", "总体继续提高，不代表尾类能力被持续保留",
          "Client-LT：同一训练过程同时观察 Overall、Non-tail 与 Tail；所有曲线保留原始逐轮值。",
          "固定 A 保持稳定；较自由更新带来更多适配，也可能付出明显的尾类保持代价。",
          "数据：training_curves.csv；seed42。回落 = Tail 峰值 − 第100轮。峰值仅作诊断，不用于挑选检查点；曲线单位为 %。")
def trajectory_plot(fig):
    axs = fig.subplots(1, 3)
    fig.subplots_adjust(left=.05, right=.988, bottom=.15, top=.81, wspace=.27)
    for a, metric, title, limits in zip(axs, ["overall", "nontail", "tail"],
        ["Overall / 全部类别", "Non-tail / 前 80 类", "Tail / 后 20 类"], [(60, 75), (60, 78), (55, 76)]):
        for method in ("E2", "E3", "S", "J"):
            method_line(a, CLT, method, metric)
        a.set(xlim=(0, 100), ylim=limits, xlabel="全局轮次", title=title)
        axis(a, "准确率 (%)")
    axs[2].axhline(66.35, ls="--", lw=.9, color=GREY, alpha=.7)
    axs[2].text(3, 65.1, "初始 66.35", fontsize=10, color=GREY)
    axs[2].annotate("J：71.70 @ 22", (22, 71.7), (2, 74.2), fontsize=11, color=RED,
                    arrowprops=dict(arrowstyle="-", color=RED))
    axs[2].annotate("S：71.80 @ 46", (46, 71.8), (47, 74.2), fontsize=11, color=ORANGE,
                    arrowprops=dict(arrowstyle="-", color=ORANGE))
    fig.legend(*axs[0].get_legend_handles_labels(), loc="upper center", ncol=4, frameon=False)
s.plot((105, 300, 2340, 760), trajectory_plot)
for x, method in zip((175, 775, 1380, 1980), ("E2", "E3", "S", "J")):
    _, vals = series(CLT, method, "tail")
    s.text(x, 1100, f"{method} 回落  {fmt(max(vals)-vals[-1], digits=2)} pp", 34, COLORS[method], True)
s.save()

# 05: show the LA x refresh interaction and an explicit scope boundary.
s = Slide("05_la_interaction", "组件证据 / 类别平衡不等于跨轮保持", "LA 改善 A 刷新的结果，但没有消除持续更新的遗忘",
          "同为 B-only 日常训练；只比较九次额外训练 B 与九次额外刷新 A，分别使用 CE 和 LA。",
          "LA 是有效底座；接下来的核心方法应解决“继续学习时如何保持”，而不是重新调 LA。",
          "数据：performance_clientlt.csv；第 81–100 轮均值，seed42。CE/LA 来自历史消融，不报告虚假的多种子误差条。")
s.panel(100, 285, 1605, 880, "LA 与 A 刷新的交互", BLUE)
def la_plot(fig):
    axs = fig.subplots(1, 2)
    fig.subplots_adjust(left=.075, right=.975, top=.89, bottom=.2, wspace=.30)
    for a, key, title, limits in zip(axs, ["overall_acc", "bottom20_tail_acc"], ["Overall", "Tail"], [(63, 72), (53, 76)]):
        for methods, color, name, sty, marker in [(('E0','E1'), ORANGE, 'CE', '-', 'o'),
                                                   (('E2','E3'), BLUE, 'LA (τ=1)', '--', 's')]:
            ys = [PERF[m][key] for m in methods]
            a.plot([0,1], ys, color=color, ls=sty, marker=marker, ms=7, lw=2.2,
                   label=f"{name}：{fmt(ys[1]-ys[0], True)} pp")
            for x, method, value in zip([0,1], methods, ys):
                a.annotate(f"{method} {fmt(value)}", (x,value), xytext=(0, 14 if name != 'CE' else -24),
                    textcoords='offset points', ha='center', color=color, fontsize=11)
        a.set(xlim=(-.35,1.35), ylim=limits, title=title)
        a.set_xticks([0,1], ["固定 A\n额外 B", "九次 A\n刷新"])
        a.legend(loc="lower left", frameon=False, fontsize=10)
        axis(a, "准确率 (%)")
s.plot((125, 405, 1550, 735), la_plot)
s.panel(1750, 285, 710, 880, "三个直接结论", ORANGE)
s.bullets(1795, 415, 615, [("无 LA：刷新 A 后 Tail 下降", "E1 − E0 = −10.043 pp。"),
    ("有 LA：刷新 A 后 Tail 上升", "E3 − E2 = +1.605 pp。"),
    ("但 LA 不是保持保证", "J/S 同样使用 LA，后期仍出现尾类回落。")], color=ORANGE, size=34, gap=42)
s.save()

# 06: common axes; ordinary Dirichlet, no matched partitions.
s = Slide("06_standard_topology", "关键对照 / 换成普通 Dirichlet 后还成立吗", "Client-LT 放大了较自由适配的尾类保持代价",
          "对每个训练策略，比较 Client-LT 与普通 Dirichlet；同一面板中只有数据划分不同。",
          "问题不是“Dirichlet 完全不遗忘”，而是 Client-LT 中的尾类回落明显更大。",
          "数据：training_curves.csv、retention.csv；β=0.5，seed42。普通 Dir 同时改变容量、覆盖与共现，不是单变量结构干预。")
def topology_plot(fig):
    axs = fig.subplots(1, 3, sharey=True)
    fig.subplots_adjust(left=.05, right=.988, top=.88, bottom=.15, wspace=.16)
    for a, method in zip(axs, ("E3", "S", "J")):
        for top, color, style, mark, label in [(CLT, ORANGE, '-', 'o', 'Client-LT'),
                                              (DIR, BLUE, '--', 's', '普通 Dir')]:
            x,y = series(top, method, "tail")
            a.plot(x,y,color=color,ls=style,marker=mark,markevery=12,ms=4,lw=2,label=label)
        a.set(xlim=(0,100), ylim=(55,76), title=method, xlabel="全局轮次")
        axis(a)
        a.legend(loc="lower left", frameon=False)
    axs[0].set_ylabel("Tail 准确率 (%)")
s.plot((115, 295, 2320, 750), topology_plot)
for x, method in zip((165, 950, 1730), ("E3", "S", "J")):
    clt, dr = float(RET[CLT,method]['peak_to_final']), float(RET[DIR,method]['peak_to_final'])
    s.text(x, 1060, "峰值 → 第 100 轮回落", 28, GREY)
    s.text(x, 1120, f"CLT {clt:.2f}   /   Dir {dr:.2f} pp", 37, NAVY, True)
s.save()

# 07: difference of strategy effects, not simply endpoint rankings.
s = Slide("07_topology_interaction", "关键量化 / 比较策略改变的额外代价", "改用更自由的训练，Client-LT 多付出多少代价？",
          "先分别计算两种划分内的策略收益，再比较差异，避免只看最终分数高低。",
          "额外尾类拓扑代价：E3 → J 为 +10.3275 pp；E3 → S 为 +2.8725 pp。",
          "定义：D_m = Tail_Dir,m − Tail_CLT,m；比较 D_m − D_E3。第 81–100 轮均值；完整策略效应，非纯刷新次数效应。")
s.panel(100, 285, 1605, 880, "相对 E3 的准确率变化", BLUE)
def effect_plot(fig):
    axs = fig.subplots(1,2)
    fig.subplots_adjust(left=.075, right=.98, bottom=.15, top=.84, wspace=.30)
    for a, metric, title, ylim in zip(axs, ("overall", "tail"), ("Overall", "Tail"), ((-.5,3.7),(-13,3))):
        for shift,top,color,marker,label in [(-.16,CLT,ORANGE,'o','Client-LT'),(.16,DIR,BLUE,'s','普通 Dir')]:
            ys = [float(RET[top,m][metric])-float(RET[top,'E3'][metric]) for m in ('J','S')]
            pos = np.arange(2)+shift
            a.vlines(pos, 0, ys, color=color, lw=3)
            a.scatter(pos,ys,c=color,marker=marker,s=65,label=label,zorder=4)
            for xx,yy in zip(pos,ys):
                a.annotate(fmt(yy,True), (xx,yy), (0,13 if yy>=0 else -22),
                    textcoords='offset points',ha='center',color=color,fontsize=11)
        a.axhline(0,color=GREY,lw=1)
        a.set(ylim=ylim,xlim=(-.5,1.5),title=title)
        a.set_xticks([0,1],["E3 → J","E3 → S"])
        axis(a,"准确率变化 (pp)")
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',ncol=2,frameon=False)
s.plot((125, 405, 1550, 725), effect_plot)
s.panel(1750, 285, 710, 880, "为什么这比端点比较更有力？", ORANGE)
s.bullets(1795, 415, 610, [("两边的 Overall 都提高", "更自由的训练确实带来适配收益。"),
    ("Tail 的方向与幅度不同", "E3 → S：Dir +0.280 pp；CLT −2.593 pp。"),
    ("方法目标不是压低 Dir", "要改善 CLT 自身保持，并缩小额外损失。")], color=ORANGE, size=32, gap=40)
s.save()

# 08: all methods shown, no fictitious Ours star or Pareto target point.
s = Slide("08_performance_tradeoff", "结果全貌 / 不能只看 Overall 排名", "目前没有一个策略同时拿到最高 Overall 和最高 Tail",
          "CAPT 保留为既有对齐配置参照；E3 保持较好，J 适配更充分，S 位于两者之间。",
          "要突破的是适配与保持的折中；不能把不同轮次、不同模型的最优分数拼成一个“理想点”。",
          "数据：performance_clientlt.csv；第 81–100 轮均值，seed42。CAPT 与 S 不等计算预算；坐标范围已明确标出。")
s.panel(100, 285, 1605, 880, "Overall–Tail 二维表现", BLUE)
def scatter_plot(fig):
    a = fig.add_subplot()
    fig.subplots_adjust(left=.10,right=.97,bottom=.14,top=.96)
    cap=PERF['CAPT']
    a.axvline(cap['overall_acc'],ls='--',color=GREY,lw=1,alpha=.65)
    a.axhline(cap['bottom20_tail_acc'],ls='--',color=GREY,lw=1,alpha=.65)
    offsets=dict(E0=(8,-20),E1=(9,7),E2=(-27,12),E3=(-23,19),E5=(-74,-25),J=(10,-18),S=(10,9),CAPT=(10,9))
    markers=dict(E0='o',E1='x',E2='o',E3='s',E5='P',J='D',S='^',CAPT='X')
    for method,p in PERF.items():
        x,y=p['overall_acc'],p['bottom20_tail_acc']
        a.scatter(x,y,color=COLORS[method],marker=markers[method],s=95,zorder=4)
        a.annotate(method,(x,y),offsets[method],textcoords='offset points',color=COLORS[method],fontsize=14,
                   arrowprops=dict(arrowstyle='-',color=COLORS[method],lw=.6))
    a.set(xlim=(64,73),ylim=(50,80),xlabel='Overall 准确率 (%)',ylabel='Tail 准确率 (%)')
    axis(a)
s.plot((130, 400, 1540, 740), scatter_plot)
s.panel(1750, 285, 710, 880, "三个参照，三个意义", BLUE)
s.bullets(1795, 410, 610, [("E3：稳定适配参照", "Overall 69.646 / Tail 71.255。"),
    ("J：充分适配的收益与代价", "Overall 71.196 / Tail 60.270。"),
    ("S：适合方法干预的底座", "A/B 阶段分离。\nOverall 70.910 / Tail 68.663。")], size=31, gap=40)
s.save()

# 09: show both beneficial and harmful class-group changes.
s = Slide("09_group_gains", "方法必须保留什么 / 不靠少学来换稳定", "需要保住的不只是尾类分数，还有头中部的适配收益",
          "J/S 相对 E3 的收益并非只发生在最高频 20 类；Middle60 同样得到改善。",
          "理想干预是减少尾类损伤，同时保住充分训练带来的 Head / Middle 收益。",
          "数据：performance_clientlt.csv；Δ = 方法 − E3，第 81–100 轮均值。Non-tail = Head20 + Middle60，不等于最高频头类。")
s.panel(100, 285, 1605, 880, "按类别频率分组的变化", BLUE)
def group_plot(fig):
    a=fig.add_subplot()
    fig.subplots_adjust(left=.10,right=.98,top=.9,bottom=.17)
    keys=['head20_acc','middle60_acc','bottom20_tail_acc']
    for shift,m,marker in [(-.17,'J','D'),(.17,'S','^')]:
        ys=[PERF[m][k]-PERF['E3'][k] for k in keys]
        bars=a.bar(np.arange(3)+shift,ys,width=.29,color=COLORS[m],alpha=.85,label=f'{m} − E3')
        a.scatter(np.arange(3)+shift,ys,marker=marker,color=COLORS[m],s=45,zorder=4)
        a.bar_label(bars,labels=[fmt(v,True) for v in ys],padding=8,fontsize=13)
    a.axhline(0,color=GREY,lw=1)
    a.set(ylim=(-13,7),ylabel='准确率变化 (pp)')
    a.set_xticks(np.arange(3),['Head20\n最高频 20 类','Middle60\n中间 60 类','Tail20\n最低频 20 类'])
    a.legend(loc='lower left',frameon=False)
    axis(a)
s.plot((130, 395, 1540, 745), group_plot)
s.panel(1750, 285, 710, 880, "新方法的双重要求", TEAL)
s.icon('shield',1805,425,TEAL,70)
s.text(1920,430,'减少尾类损伤',37,TEAL,True,width=480)
s.text(1810,545,'不能只靠降低学习率或让 A 接近冻结，制造一条平稳曲线。',31,NAVY,width=595)
s.icon('chart',1805,740,BLUE,70)
s.text(1920,745,'保留适配能力',37,BLUE,True,width=480)
s.text(1810,865,'同时报告 Overall、Head、Middle、Tail，以及实际 A 更新幅度。',31,NAVY,width=595)
s.save()

# 10: candidate architecture; visual label never implies implemented success.
s = Slide("10_method_candidate", "核心方法 / 候选设计，尚无新方法成绩", "让普通更新负责学习，让功能反馈约束损伤",
          "候选：功能保护式 A 聚合。B 的训练与 LA 不变；本地 A 仍用 LA，只修正服务器提交的 A 更新。",
          "创新要落在“保护什么、如何修正且不妨碍学习”；不是把冻结、等权或投影本身当作创新。",
          "状态：待选定、待实现、待验证；图为算法信息流，不是已完成方法结果。功能向量不等于隐私保证。")
s.panel(100, 285, 1015, 700, "客户端：发现需要保护的功能", BLUE, number=1)
s.panel(1250, 285, 1210, 700, "服务器：修正 A 的聚合方向", TEAL, number=2)
s.panel(145,415,425,165,fill='white')
s.panel(640,415,425,165,fill='white')
s.text(180,437,'本地 B 模型',33,BLUE,True,width=360)
s.text(180,498,'我在本地能学到多少？',27,NAVY,width=370)
s.text(675,437,'B 聚合后的共享模型',29,BLUE,True,width=360)
s.text(675,498,'共享模型保留了多少？',27,NAVY,width=370)
s.arrow((350,591),(535,650),BLUE)
s.arrow((852,591),(670,650),BLUE)
s.text(260,670,'正功能缺口 → 保护方向',37,ORANGE,True,width=780)
s.text(190,744,'训练集代理；逐类信号保留在客户端',29,GREY,width=850)
s.d.line((150, 810, 1060, 810), fill=BORDER, width=2)
s.text(150, 840, '固定共享 B，用原 LA 训练 A',31,NAVY,True,width=910)
s.text(150, 905, '上传本地 A 更新 + 功能保护方向',31,BLUE,True,width=910)
s.arrow((1125, 630),(1238,630),BLUE)
s.text(1300,418,'按样本量形成普通候选 d0',34,TEAL,True,width=1100)
s.text(1300,480,'保护约束：尽量保留学习方向，减少冲突',30,NAVY,width=1100)
s.d.rounded_rectangle((1360,555,2345,838),radius=18,fill='white',outline=BORDER,width=2)
s.d.rectangle((1475,590,2250,690),fill='#EDF7F3')
s.d.line((1475,690,2270,690),fill=BORDER,width=2)
s.arrow((1520,690),(1520,577),TEAL,4)
s.text(1555,567,'保护方向 u',28,TEAL,True,width=420)
s.arrow((1520,690),(2140,690),TEAL,6)
s.arrow((1520,690),(2140,797),RED,5)
s.text(1920,631,'修正后 dP',30,TEAL,True,width=360)
s.text(2160,767,'普通 d0',28,RED,True,width=185)
s.text(1380,850,'方向示意，非实测向量；多客户端约束需共同求解。',26,GREY,width=980)
s.text(1300,914,'提交新 A；B 不变 → 下一轮继续适配',31,TEAL,True,width=1090)
s.panel(100, 1030, 2360, 140, fill='#FFF9F0')
s.text(140, 1050,'两个关键风险',31,ORANGE,True)
s.text(520,1050,'训练集缺口可能混入局部拟合优势；过强约束可能让 A 退回近冻结。',31,NAVY,width=1860)
s.text(520,1105,'因此：需要训练外诊断，也需要同状态缩幅对照。',29,GREY,width=1860)
s.save()

# 11: minimal pilot, with the same-state magnitude-control interpretation.
s = Slide("11_minimal_pilot", "下一步 / 先做出有效方法，再扩完整闭环", "先跑同起点的 R / P / K，回答核心方法是否有效",
          "复用已有 S 第 20 轮模型；继续第 21–100 轮。先 Client-LT，暂不新增角色全排列或 rank / LA 网格。",
          "现象证据已经足够支持启动方法；新增对照服务于方法效果，不重复证明旧问题。",
          "待实现方案：同一续训随机协议；21–90 轮交替 B/A，91–100 轮只训练 B。旧 S 后半段不是严格配对的新 R。")
s.panel(100,285,2360,180,'共同起点：S 第 20 轮结束的共享 A/B',BLUE)
s.text(145,397,'同数据、同 LA、同本地预算、同新随机流；只改变提交的 A 更新。',31,NAVY,width=2250)
for x,title,color,items in [(100,'R：常规续训',GREY,[('普通聚合方向','按样本量聚合本地 A 更新。'),('基准问题','正常继续学，会遗忘多少？')]),
    (905,'P：保护式聚合',TEAL,[('保护修正方向','用客户端功能反馈修正普通候选。'),('核心问题','能否少遗忘，同时保持适配？')]),
    (1710,'K：同状态缩幅',ORANGE,[('只改幅度，不改方向','在自身状态计算 P 候选幅度，再缩放普通方向。'),('排除解释','是否只是因为 A 更新更小？')])]:
    s.arrow((x+375,475),(x+375,535),color)
    s.panel(x,550,750,555,title,color)
    s.bullets(x+42,685,665,items,color,size=31,gap=37,max_y=1090)
s.text(135,1140,'K 的“等幅”是各自当前状态下的匹配；分支分开后，并不保证 P 与 K 的累计路径相同。',27,GREY,width=2270)
s.save()

# 12: a staged checklist, not another immediate experiment expansion.
s = Slide("12_closure_roadmap", "收束 / 把下一阶段控制在一个方法问题上", "先验证有效，再补齐机制与稳健性",
          "方法训练顺手保留关键中间状态；不把所有闭环证据都设置为设计方法前的门槛。",
          "汇报的落点：问题仍在、方法入口明确；现在需要一个能同时改善学习与保持的核心干预。",
          "所有右侧任务均为后续计划，不代表已经完成。新方法不得使用官方测试集或离线 Tail probe 决定更新方向。")
for x,title,color,num,items in [(100,'现在：方法是否有效',BLUE,1,[('Client-LT 的 R / P / K','看 Tail 绝对水平与后期回落；\n同时报告 Overall / Non-tail。'),('不靠近冻结换稳定','报告 A 更新范数、相对 K 的收益，\n以及额外计算开销。')]),
    (905,'有信号后：为何有效',TEAL,2,[('聚合前后 + 后续轮次','本地有益功能被保留多少？下一轮 B 是否又损伤它？'),('独立诊断与信号消融','训练外 probe；必要时补 P-flat，判断 gap 定向是否重要。')]),
    (1710,'规则固定后：能否复现',ORANGE,3,[('普通 Dir 与新种子','CLT 要有实际收益，\n不能靠压低 Dir 缩小差距。'),('完整训练与第二数据集','确认方法可复现，并报告性能、保持与成本。')])]:
    s.panel(x,300,750,820,title,color,number=num)
    s.bullets(x+42,445,655,items,color,size=32,gap=52,max_y=1100)
    s.icon('target' if num==1 else ('shield' if num==2 else 'chart'),x+320,970,color,80)
s.save()

# A1: full table with row definitions, not a selective four-method ranking.
s = Slide("a1_full_results", "附录 A / 统一主结果与实验定义", "完整主表：所有已完成策略都保留",
          "主口径统一为第 81–100 轮均值；加粗列最大值，不把单个指标最高等同于方法全面领先。",
          "E3 / E5 的强项是尾类保持；J 的强项是总体适配；S 在两者之间提供可干预底座。",
          "数据：performance_clientlt.csv、retention.csv；seed42。CAPT 是对齐实现参照，非等算力 SOTA 结论；所有值为准确率 (%)。")
cols=[140,350,1440,1680,1940,2220]
headers=['方法','训练方式','Overall','Head20','Middle60','Tail20']
for x,t in zip(cols,headers): s.text(x,295,t,31,BLUE,True)
s.d.line((120,360,2440,360),fill=BLUE,width=3)
desc={'CAPT':'CAPT 对齐配置','E0':'无 LA；固定 A；九次额外 B','E1':'无 LA；B-only；九次 A 刷新',
      'E2':'有 LA；固定 A；九次额外 B','E3':'有 LA；B-only；九次 A 刷新','E5':'有 LA；功能控制',
      'J':'有 LA；日常 AB 联训；另有九次 A 刷新','S':'有 LA；B-only；前 90 轮逐轮刷新 A'}
keys=['overall_acc','head20_acc','middle60_acc','bottom20_tail_acc']
maxs={k:max(p[k] for p in PERF.values()) for k in keys}
for i,(m,p) in enumerate(PERF.items()):
    y=380+i*77
    if i%2==0: s.d.rectangle((120,y-7,2440,y+67),fill=PALE)
    s.text(cols[0],y,m,32,COLORS[m],True)
    s.text(cols[1],y+3,desc[m],29,NAVY,width=1020)
    for x,k in zip(cols[2:],keys):
        best=np.isclose(p[k],maxs[k])
        s.text(x,y+2,fmt(p[k]),32,BLUE if best else NAVY,best)
s.panel(100,1050,2360,130,fill=PALE)
s.text(135,1065,'普通 Dir 对照（Overall / Tail）',29,BLUE,True,width=820)
s.text(970,1065,'E3   69.604 / 72.018      J   72.293 / 71.360      S   71.014 / 72.298',29,NAVY,width=1420)
s.text(135,1120,'普通 Dir 的 E2 不在本批数据中；不借用旧 matched-Dirichlet 填补。',27,GREY,width=2200)
s.save()

# A2: all twenty classes retained; shared symmetric scale and direct values.
s = Slide("a2_all_tail_classes", "附录 B / 不只挑受损最大的几个类", "尾类退化并非由一两个异常类别造成",
          "展示全部 20 个尾类，按原类别编号排序；每个单元格是方法相对 E3 的末 20 轮准确率差。",
          "J 有 16 / 20 个尾类下降；S 有 13 / 20 个尾类下降。类别覆盖面不等于多种子显著性。",
          "数据：tail_class_deltas.csv；单位 pp。相同的对称色标；数字与正负号提供非颜色编码；不按受损大小筛选类别。")
def class_plot(fig):
    axs=fig.subplots(1,2)
    fig.subplots_adjust(left=.15,right=.89,top=.90,bottom=.09,wspace=.62)
    rows=sorted([r for r in DIFF if r['comparison']=='j-e3'],key=lambda r:int(r['class_id']))
    classnames=[r['class_name'].replace('_',' ') for r in rows]
    classids=[int(r['class_id']) for r in rows]
    vmax=45
    for panel,indices in zip(axs,[list(range(10)),list(range(10,20))]):
        vals=np.array([[float(next(r['last20_difference'] for r in DIFF if int(r['class_id'])==classids[i]
                    and r['comparison']==m)) for m in ['j-e3','s-e3']] for i in indices])
        im=panel.imshow(vals,cmap='RdBu',vmin=-vmax,vmax=vmax,aspect='auto')
        panel.set_xticks([0,1],['J − E3','S − E3'])
        panel.xaxis.tick_top()
        panel.set_yticks(np.arange(10),[f'{classids[i]}  {classnames[i]}' for i in indices],fontsize=12)
        for (i,j),value in np.ndenumerate(vals):
            panel.text(j,i,f'{value:+.2f}',ha='center',va='center',color='white' if abs(value)>25 else NAVY,fontsize=14)
        panel.tick_params(length=0,pad=10)
        for sp in panel.spines.values(): sp.set_visible(False)
    cax=fig.add_axes([.935,.14,.016,.70])
    fig.colorbar(im,cax=cax,label='准确率差值 (pp)')
s.plot((125,300,2320,865),class_plot)
s.save()

# A3: the null hard-role diagnosis belongs in an appendix, not hidden.
s = Slide("a3_factor_diagnostic", "附录 C / A/B 角色诊断给出的边界", "一步等权没有带来 Tail 准确率改善，但它不是保持方法实验",
          "6 个相关模型起点，48 个全客户端单 epoch 分支；只观察一次聚合后的即时响应。",
          "没有证实 A/B 硬分工；因此不靠这个定义立论，也不靠增加 epoch 追求预期模式。",
          "数据：factor_effects.csv；原始 / 等幅联合判据均为 0/6。连续 log-odds 不是准确率，相关起点不是独立种子。")
s.panel(100,285,1605,880,'Client-LT：原始更新与有效权重幅度匹配',BLUE)
def roles_plot(fig):
    axs=fig.subplots(1,2,sharey=True)
    fig.subplots_adjust(left=.09,right=.975,top=.84,bottom=.22,wspace=.17)
    for a,mode,title in zip(axs,['raw','norm_matched'],['原始更新','有效幅度匹配']):
        for shift,factor,color,marker in [(-.15,'A',BLUE,'s'),(.15,'B',ORANGE,'o')]:
            vals=[float(next(r['mean'] for r in ROLE if r['mode']==mode and r['topology']==CLT
                    and r['metric']=='probe_tail_logodds' and r['factor']==factor and r['correction']==c))*1000
                    for c in ['client','class']]
            xx=np.arange(2)+shift
            a.vlines(xx,0,vals,color=color,lw=3)
            a.scatter(xx,vals,color=color,marker=marker,s=70,label=factor,zorder=4)
            for x,y in zip(xx,vals):
                a.annotate(fmt(y,True),(x,y),(0,13 if y>=0 else -17),textcoords='offset points',ha='center',color=color,fontsize=10)
        a.set(ylim=(-1.5,7),xlim=(-.5,1.5),title=title)
        a.set_xticks([0,1],['等权 − 样本量\nLA 固定','LA − CE\n样本量聚合固定'])
        a.axhline(0,color=GREY,lw=1)
        axis(a)
    axs[0].set_ylabel('Tail probe log-odds 变化 × 1000')
    fig.legend(*axs[0].get_legend_handles_labels(),loc='upper center',frameon=False,ncol=2)
s.plot((125,410,1550,725),roles_plot)
s.panel(1750,285,710,880,'怎样解读这个负结果？',ORANGE)
s.bullets(1795,410,610,[('直接结果','CLT 等权修正的 Tail 准确率变化均为 0。'),
    ('提示，而非角色证明','连续指标里 B 对 LA 有更大平均响应，不能替代准确率交叉判据。'),
    ('下一步不纠缠硬分工','多轮 A 保护是否减少保持损失，需要真正的方法干预。')],color=ORANGE,size=31,gap=32)
s.save()

# Contact sheet for review; not a replacement for full-size inspection.
contact=Image.new('RGB',(2560,1440),'#EAF0F7')
for i,name in enumerate(GENERATED):
    with Image.open(HERE/(name+'.png')) as im:
        thumb=im.resize((640,360),Image.Resampling.LANCZOS)
        contact.paste(thumb,((i%4)*640,(i//4)*360))
contact.save(HERE/'00_contact_sheet.png',optimize=True)
with (HERE/'validation.json').open('w',encoding='utf-8') as stream:
    json.dump(dict(images=GENERATED,count=len(GENERATED),size=[W,H],format='PNG only',
        evidence_checks=CHECKS,layout_warnings=LAYOUT_WARNINGS,
        status='Rendered; visual audit recorded separately in 论文逻辑骨架与设计审查.md'),stream,ensure_ascii=False,indent=2)
assert len(GENERATED)==15
assert not LAYOUT_WARNINGS, LAYOUT_WARNINGS
assert not list(HERE.glob('*.pdf'))
print('Complete: 15 slides + 1 contact sheet, PNG only; no training.',flush=True)
