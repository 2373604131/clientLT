"""Render 9 advisor slides with one concise, implemented SFRA A overview.

PNG only. All charts read existing results; schematic examples are labelled.
No training/model imports. Run from repo root:
    python presentation/method_ab_20260920/render_slides.py
"""
from __future__ import annotations

import csv
import io
import json
import math
import shutil
import zipfile
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.mathtext import math_to_image
import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = HERE.parent / "redesign_20260919/data"
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
W, H = 2560, 1440
NAVY, BLUE, ORANGE = "#142E50", "#2363AD", "#B86817"
TEAL, RED, GREY = "#187D76", "#B4414A", "#617187"
LIGHT, LINE = "#F5F8FC", "#CDD8E5"
LIGHT_BLUE, LIGHT_ORANGE, LIGHT_TEAL = "#EDF4FC", "#FFF6E9", "#EDF7F4"
FONT = "C:/Windows/Fonts/msyh.ttc"
BOLD = "C:/Windows/Fonts/msyhbd.ttc"
COLORS = {"e2": TEAL, "e3": BLUE, "j": RED, "s": ORANGE}
STYLES = {"e2": ("-", "o"), "e3": ("--", "s"), "j": (":", "D"), "s": ("-.", "^")}
CLT, DIR = "client-longtail", "noniid-labeldir-fine"
GENERATED, WARNINGS, CHECKS = [], [], []
plt.rcParams.update({
    "font.family": "Microsoft YaHei", "axes.unicode_minus": False,
    "font.size": 13, "axes.labelsize": 14, "axes.titlesize": 17,
    "xtick.labelsize": 12, "ytick.labelsize": 12, "legend.fontsize": 12,
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": GREY, "axes.labelcolor": NAVY, "text.color": NAVY,
    "xtick.color": GREY, "ytick.color": GREY, "axes.linewidth": .8,
    "savefig.facecolor": "white",
})


def read(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


SOURCES = {
    "partition_structure.csv": OLD / "partition_structure.csv",
    "training_curves.csv": OLD / "training_curves.csv",
    "retention.csv": OLD / "retention.csv",
    "performance_clientlt.csv": OLD / "performance_clientlt.csv",
    "b_phase_attribution.csv": ROOT / "output/a_refresh_topology_bridge_analysis/review/phase_means_recomputed.csv",
}
for name, path in SOURCES.items():
    shutil.copy2(path, DATA / name)
PART = {r["topology"]: r for r in read(DATA / "partition_structure.csv")}
CURVES = read(DATA / "training_curves.csv")
RET = {(r["topology"], r["method"]): r for r in read(DATA / "retention.csv")}
PERF = {r["method"]: r for r in read(DATA / "performance_clientlt.csv")}
ATTR = {r["method"]: r for r in read(DATA / "b_phase_attribution.csv")
        if r["topology"] == CLT and r["phase"] == "normal_B"}


def series(top, method, metric="tail"):
    rows = sorted((r for r in CURVES if r["topology"] == top and r["method"] == method),
                  key=lambda r: int(r["round"]))
    return np.array([int(r["round"]) for r in rows]), np.array([float(r[metric]) for r in rows])


for top in (CLT, DIR):
    for method in ("e3", "j", "s"):
        x, y = series(top, method)
        assert list(x) == list(range(101))
        assert np.isclose(y[-20:].mean(), float(RET[top, method]["tail"]))
        assert np.isclose(y.max() - y[-1], float(RET[top, method]["peak_to_final"]))
for top, row in PART.items():
    assert sum(json.loads(row["client_sizes"])) == 10847
    assert sum(json.loads(row["tail_samples_per_client"])) == 153
CHECKS += ["Six topology curves: rounds 0..100, last20 and peak-to-final verified.",
           "Both partitions: 10847 total images, 153 Tail20 images."]
assert np.isclose(series(CLT, "e2")[1][-20:].mean(), float(PERF["E2"]["bottom20_tail_acc"]))
for row in ATTR.values():
    assert int(row["events"]) == 7
    assert np.isclose(float(row["W"]) + float(row["D"]) - float(row["H"]) - float(row["R"]), float(row["net"]))
CHECKS += ["E2 curve agrees with Client-LT result table.",
           "B attribution is seven audited events; W + D - H - R agrees with net."]


def font(size, bold=False):
    return ImageFont.truetype(BOLD if bold else FONT, size)


def wrap(text, ft, width):
    lines = []
    for para in str(text).split("\n"):
        line = ""
        for ch in para:
            if line and ft.getlength(line + ch) > width:
                lines.append(line)
                line = ch
            else:
                line += ch
        lines.append(line)
    return lines


class Slide:
    def __init__(self, name, chapter, title, subtitle, takeaway, source, state="已有结果"):
        self.name = name
        self.im = Image.new("RGB", (W, H), "white")
        self.d = ImageDraw.Draw(self.im)
        self.text(100, 39, chapter, 26, BLUE, True, width=1810)
        state_color = ORANGE if "设想" in state else TEAL if "实现" in state else BLUE
        self.box(2175, 34, 285, 51, fill=LIGHT_ORANGE if "设想" in state else LIGHT_BLUE, edge=state_color)
        self.text(2317, 59, state, 25, state_color, True, anchor="mm")
        self.text(100, 96, title, 60, NAVY, True, width=2360, max_y=184)
        self.text(104, 185, subtitle, 29, GREY, width=2350, max_y=232)
        self.d.line((100, 245, 2460, 245), fill=BLUE, width=2)
        self.d.line((1190, 245, 1370, 245), fill=BLUE, width=7)
        self.box(100, 1220, 2360, 103, fill="#FFF5F3", edge="#E5BDB7")
        self.text(135, 1250, "→", 42, RED, True)
        self.text(210, 1242, takeaway, 34, RED, True, width=2195, max_y=1316, spacing=1.28)
        self.text(100, 1352, source, 23, GREY, width=2225, max_y=1435, spacing=1.28)
        self.text(2395, 1365, f"{len(GENERATED)+1:02d}", 29, BLUE, True)

    def text(self, x, y, value, size=34, color=NAVY, bold=False, width=None, max_y=None,
             spacing=1.37, anchor=None):
        ft = font(size, bold)
        if anchor:
            self.d.text((x, y), str(value), font=ft, fill=color, anchor=anchor)
            return y + size * spacing
        lines = wrap(value, ft, width or W-x-90)
        bottom = y + len(lines)*size*spacing
        if bottom > (max_y or H):
            WARNINGS.append(f"{self.name}: text overflow: {str(value)[:32]} -> {bottom:.0f}")
        for line in lines:
            self.d.text((x, y), line, font=ft, fill=color)
            y += size*spacing
        return y

    def box(self, x, y, w, h, fill=LIGHT, edge=LINE, radius=17, stroke=2):
        self.d.rounded_rectangle((x, y, x+w, y+h), radius=radius, fill=fill, outline=edge, width=stroke)

    def panel(self, x, y, w, h, title, color=BLUE, fill=LIGHT):
        self.box(x, y, w, h, fill)
        self.text(x+30, y+23, title, 36, color, True, width=w-60, max_y=y+89)
        self.d.line((x+30, y+91, x+w-30, y+91), fill=color, width=2)

    def card(self, x, y, w, h, title, body, color=BLUE, fill=LIGHT, num=None, size=32):
        self.box(x, y, w, h, fill)
        shift = 30
        if num is not None:
            self.d.ellipse((x+28, y+27, x+80, y+79), fill=color)
            self.text(x+54, y+52, num, 28, "white", True, anchor="mm")
            shift = 97
        end = self.text(x+shift, y+25, title, 36, color, True, width=w-shift-24, max_y=y+h-20)
        self.text(x+30, max(y+105, end+14), body, size, NAVY, width=w-60, max_y=y+h-20)

    def arrow(self, start, end, color=BLUE, width=5, dashed=False):
        x1, y1 = start
        x2, y2 = end
        if dashed:
            length = math.hypot(x2-x1, y2-y1)
            for pos in np.arange(0, length, 27):
                a, b = pos/length, min(pos+15, length)/length
                self.d.line((x1+(x2-x1)*a, y1+(y2-y1)*a, x1+(x2-x1)*b, y1+(y2-y1)*b), fill=color, width=width)
        else:
            self.d.line((x1, y1, x2, y2), fill=color, width=width)
        angle = math.atan2(y2-y1, x2-x1)
        self.d.polygon([(x2, y2), (x2+21*math.cos(angle+2.65), y2+21*math.sin(angle+2.65)),
                        (x2+21*math.cos(angle-2.65), y2+21*math.sin(angle-2.65))], fill=color)

    def client(self, x, y, label, color=BLUE, w=128):
        self.box(x, y, w, 72, "white", color, radius=7, stroke=3)
        self.d.line((x+w*.5, y+73, x+w*.5, y+89), fill=color, width=3)
        self.d.line((x+w*.27, y+90, x+w*.73, y+90), fill=color, width=3)
        self.text(x+w/2, y+35, label, 25, color, True, anchor="mm")

    def server(self, x, y, label="共享模型", color=BLUE):
        for i in range(3):
            self.box(x, y+i*39, 154, 31, LIGHT_BLUE, color, 5, 3)
            self.d.ellipse((x+12, y+11+i*39, x+22, y+21+i*39), fill=color)
        self.text(x+77, y+142, label, 29, color, True, anchor="mm")

    def plot(self, box, painter):
        x, y, w, h = box
        fig = plt.figure(figsize=(w/160, h/160), dpi=160)
        painter(fig)
        buff = io.BytesIO()
        fig.savefig(buff, format="png", dpi=160)
        plt.close(fig)
        buff.seek(0)
        self.im.paste(Image.open(buff).convert("RGB"), (x, y))

    def save(self):
        self.im.save(HERE / f"{self.name}.png", optimize=True)
        GENERATED.append(self.name)
        print("Rendered", self.name, flush=True)

    def formula(self, box, expression, color=NAVY):
        """Render editable math source into a white formula band, not an AI image."""
        x, y, width, height = box
        buff = io.BytesIO()
        math_to_image(expression, buff, dpi=220, format="png", color=color)
        buff.seek(0)
        im = Image.open(buff).convert("RGBA")
        scale = min(width/im.width, height/im.height)
        im = im.resize((round(im.width*scale), round(im.height*scale)), Image.Resampling.LANCZOS)
        self.im.paste(im, (x+(width-im.width)//2, y+(height-im.height)//2), im)


def grid(ax):
    ax.grid(axis="y", color=LINE, lw=.7, alpha=.7)
    ax.set_axisbelow(True)


# 01: Actual topology, not an invented head/tail-client cartoon.
s = Slide("01_client_level_problem", "问题发现 / 客户端层面的不均衡", "尾类证据集中在少数客户端，共享更新却由多数数据推动",
          "同一 CIFAR100-LT 全局训练池；Client-LT 与普通 Dirichlet 改变的是证据在客户端之间的分布。",
          "我们要处理的是客户端之间的有效支持不均衡，而不仅是全局类别样本数不均衡。",
          "来源：data/partition_structure.csv。普通 Dir：noniid-labeldir-fine，β=0.5；名义聚合权重不等于实际功能贡献。")
s.panel(100, 285, 1590, 870, "真实划分：同样 153 张尾类图片，落在不同客户端")


def topology_plot(fig):
    axs = fig.subplots(2, 2)
    fig.subplots_adjust(left=.10, right=.98, top=.91, bottom=.105, hspace=.27, wspace=.15)
    for j, (top, title) in enumerate(((CLT, "Client-LT"), (DIR, "普通 Dirichlet"))):
        sizes = json.loads(PART[top]["client_sizes"])
        tails = json.loads(PART[top]["tail_samples_per_client"])
        colors = [ORANGE if top == CLT and i >= 27 else BLUE for i in range(30)]
        for i, values in enumerate((sizes, tails)):
            a = axs[i, j]
            a.bar(range(30), values, color=colors, width=.8)
            a.set_ylim(0, 700 if i == 0 else 50)
            a.set_xticks([0, 10, 20, 29])
            grid(a)
            if top == CLT:
                a.axvspan(26.5, 29.5, color=LIGHT_ORANGE, zorder=0)
            if j == 0:
                a.set_ylabel("全部样本数" if i == 0 else "尾类样本数")
            if i == 1:
                a.set_xlabel("客户端编号")
        axs[0, j].set_title(title, pad=15)


s.plot((125, 410, 1540, 715), topology_plot)
s.panel(1730, 285, 730, 870, "Client-LT 的客户端结构", ORANGE)
for yy, val, desc, detail, color in [
    (421, "77.78%", "尾类图片集中在 3 个小客户端", "119 / 153 张；客户端 27–29", ORANGE),
    (656, "1.20%", "这 3 个客户端的聚合权重合计", "130 / 10847；不是全部尾类支持者", RED),
    (890, "3.65  vs  6.05", "每个尾类平均覆盖的客户端数", "Client-LT                 普通 Dir", BLUE),
]:
    s.text(1770, yy, val, 63 if yy < 800 else 49, color, True, width=650)
    s.text(1770, yy+90, desc, 31, NAVY, True, width=650)
    s.text(1770, yy+145, detail, 26, GREY, width=650)
s.save()

# 02: Measured curves, no fabricated method performance.
s = Slide("02_tail_retention_evidence", "问题发现 / 持续适配与客户端分布发生交互", "持续更新 A 后，Client-LT 付出更大的尾类保持代价",
          "以下策略都使用 LA，τ=1。固定 A 的 E2 较稳定；J/S 的尾类回落在普通 Dirichlet 下明显较小。",
          "切入点：让 A 继续学习，同时针对 Client-LT 中额外出现的尾类保持损失进行干预。",
          "来源：data/training_curves.csv、retention.csv。seed42；回落=峰值−第100轮值；J/S 不是等预算对照。")
s.panel(100, 285, 2360, 700, "真实逐轮曲线：Tail20 准确率（%）")


def curves_plot(fig):
    axs = fig.subplots(1, 2, sharey=True)
    fig.subplots_adjust(left=.062, right=.98, top=.83, bottom=.14, wspace=.12)
    for a, top, methods, title in zip(axs, (CLT, DIR), (("e2", "e3", "s", "j"), ("e3", "s", "j")), ("Client-LT", "普通 Dirichlet")):
        for method in methods:
            x, y = series(top, method)
            ls, marker = STYLES[method]
            a.plot(x, y, color=COLORS[method], ls=ls, marker=marker, markevery=12, ms=4,
                   lw=2.1, label={"e2":"E2：固定 A", "e3":"E3：9 次 A", "s":"S：90 次 A", "j":"J：AB 联训"}[method])
        a.set(xlim=(0, 100), ylim=(55, 76), xlabel="联邦轮次", title=title)
        a.legend(loc="lower left", frameon=False, ncol=2, fontsize=11)
        grid(a)
    axs[0].set_ylabel("Tail20 准确率（%）")


s.plot((128, 388, 2305, 565), curves_plot)
for i, (method, label) in enumerate((("e3", "E3：定期 A"), ("s", "S：密集 A"), ("j", "J：AB 联训"))):
    x = 100+i*802
    s.box(x, 1017, 754, 151, LIGHT)
    s.text(x+28, 1034, label, 31, COLORS[method], True, width=698)
    a, b = float(RET[CLT, method]["peak_to_final"]), float(RET[DIR, method]["peak_to_final"])
    s.text(x+28, 1087, f"尾类回落  CLT {a:.2f}  /  Dir {b:.2f} pp", 31, NAVY, width=700)
s.save()

# 03: Separate observation, explanatory hypothesis, and intervention.
s = Slide("03_why_intervene_on_A", "方法动机 / 从客户端结构到共享 A 的更新", "冻结 A 稳住了适配条件，但没有消除客户端不均衡",
          "我们的干预目标不是重新冻结 A，而是让少数客户端支持的学习需求充分参与共享 A 的持续更新。",
          "A 的设计问题：如何让共享适配既吸收多数客户端的新知识，又照顾少数来源支持的有效需求？",
          "左栏为实测描述；中栏为方法针对的机制假设，非排他因果结论。固定 A 仍使用同样的 B 样本量聚合。", state="结果 → 设计")
s.panel(100, 290, 720, 860, "① 已观察到什么？")
s.card(132, 410, 656, 240, "固定 A：尾类保持稳定", "E2 尾类峰值 69.85 → 最终 69.80\n但末 20 轮 Overall 为 68.101。", TEAL, "white", size=30)
s.card(132, 680, 656, 275, "持续适配：学得更多", "S 的 Overall 为 70.910，\n但 Tail 比 E3 低 2.593 pp。\nClient-LT 的保持代价更突出。", ORANGE, "white", size=30)
s.text(145, 1001, "说明需要改更新规则，\n而不是简单取消 A 的学习。", 32, NAVY, True, width=620)
s.arrow((835, 720), (891, 720))
s.panel(910, 290, 735, 860, "② 客户端层面的解释", ORANGE)
for i in range(4):
    s.client(952+(i%2)*174, 422+(i//2)*126, f"客户端 {i+1}")
s.client(1340, 485, "少数来源", ORANGE, 160)
s.arrow((1160, 716), (1235, 789), BLUE, 7)
s.arrow((1420, 716), (1318, 789), ORANGE, 4)
s.box(1130, 797, 292, 85, LIGHT_BLUE, BLUE)
s.text(1276, 839, "变化中的共享 A", 32, BLUE, True, anchor="mm")
s.text(953, 919, "当有效支持集中在少数客户端，\n共享 A 的改变可能未充分兼顾\n这些客户端的学习需求。", 31, NAVY, width=650)
s.text(953, 1080, "结构示意，不是实测梯度比例。", 25, GREY, width=650)
s.arrow((1660, 720), (1717, 720))
s.panel(1735, 290, 725, 860, "③ 我们怎样介入？", TEAL)
s.card(1770, 415, 654, 245, "识别谁在提供有效帮助", "不只看客户端有多少数据；\n看各客户端的 A 更新对哪些\n本地判别能力产生正向响应。", TEAL, "white", size=29)
s.card(1770, 690, 654, 275, "让支持稀缺的需求进入更新", "支持越集中，功能修正优先级越高；\n结合已学能力的历史记录，\n共同影响最终提交的 A。", TEAL, "white", size=30)
s.text(1782, 1010, "校正客户端支持不均衡，\n以减少尾类遗忘为目标。", 32, TEAL, True, width=620)
s.save()

# 04: One method-only page replaces the four verbose A-detail slides.
s = Slide("04_A_method_overview", "A 方法 / 客户端支持感知的功能保持", "共享 A 怎么更新：先正常学习，再照顾少数来源支持的能力",
          "每轮先照常训练并聚合 B；以下过程固定 B。功能 q = 某客户端上的某个类别，F 表示其判别能力。",
          "来源决定优先照顾谁，当前与历史决定照顾到什么程度；这些信息直接参与共享 A 的更新。",
          "公式为方法级简写：省略视图下标、有效目标筛选与尺度归一化；实现保留这些操作及修正幅度约束。", state="A 已实现")

s.panel(100, 290, 720, 505, "① 形成普通 A 提案", BLUE, LIGHT_BLUE)
s.text(135, 415, "各客户端从共同模型出发，\n用本地 LA 训练 A；服务器按\n原样本量权重聚合。", 33, NAVY, width=650)
s.box(133, 571, 654, 103, "white", LINE, 12)
s.formula((159, 579, 602, 90), r"$\bar A=A_t+\sum_j p_j\,\Delta A_j$", BLUE)
s.text(135, 697, "ΔA_j：客户端 j 的 A 更新\np_j：原样本量聚合权重", 29, GREY, width=650)

s.panel(870, 290, 750, 505, "② 识别支持，分配优先级", ORANGE, LIGHT_ORANGE)
s.text(905, 415, "共同起点：功能对 A 的梯度，\n与各客户端 ΔA 做内积，取正向 r。\n据此衡量支持来源的集中程度。", 33, NAVY, width=680)
s.box(903, 571, 684, 103, "white", LINE, 12)
s.formula((929, 579, 632, 90), r"$\omega_q\propto(\sum_j r_{qj}^{\,2})\, /\, (\sum_j r_{qj})^2$", ORANGE)
s.text(905, 697, "正向帮助越集中于少数客户端，\n该功能的修正优先级 ω 就越高。", 29, ORANGE, True, width=680)

s.panel(1670, 290, 790, 505, "③ 确定需要达到的功能目标", TEAL, LIGHT_TEAL)
s.text(1705, 415, "当前：争取来源提示的正向改善。\n历史：保持已经稳定学到的能力。\n两者取较高的有效目标。", 33, NAVY, width=720)
s.box(1703, 571, 724, 103, "white", LINE, 12)
s.formula((1729, 588, 672, 68), r"$T_q=\max\{F_q(A_t)+\frac{1}{2}u_q,\ H_q\}$", TEAL)
s.text(1705, 697, "u：正向帮助的均值\nH：从正式提交模型记录的稳定水平", 29, GREY, width=720)
s.arrow((830, 540), (858, 540), BLUE, width=4)
s.arrow((1630, 540), (1658, 540), ORANGE, width=4)
for x, color in [(460, BLUE), (1245, ORANGE), (2065, TEAL)]:
    s.arrow((x, 801), (x, 828), color, width=4)

s.box(100, 835, 2360, 343, LIGHT, LINE)
s.text(135, 858, "④ 从普通提案出发，修正并提交 A", 36, BLUE, True, width=1220)
s.text(1595, 865, "让来源权重 ω 真正影响最终更新", 30, TEAL, True, width=815)
s.box(133, 921, 1230, 110, "white", LINE, 12)
s.formula((158, 930, 1180, 90), r"$\min_A\quad d(A,\bar A)+\lambda\sum_q\omega_q\,[T_q-F_q(A)]_+^2$", NAVY)
s.text(1410, 933, "第一项：保留普通提案的学习收益\n第二项：优先补足来源稀缺功能的缺口", 29, NAVY, width=1005)
for x, width, title, color, fill in [
    (135, 390, "普通 A 提案", BLUE, LIGHT_BLUE),
    (605, 650, "客户端：测缺口，算 A 梯度", TEAL, LIGHT_TEAL),
    (1335, 650, "服务器：汇总梯度，修正 A", TEAL, LIGHT_TEAL),
    (2065, 360, "提交新的 A", BLUE, LIGHT_BLUE),
]:
    s.box(x, 1055, width, 74, fill, color, 12)
    s.text(x+width/2, 1091, title, 29, color, True, anchor="mm")
for start, end in [(525, 591), (1255, 1321), (1985, 2051)]:
    s.arrow((start+9, 1092), (end, 1092), TEAL, width=4)
s.text(622, 1138, "候选 A 下发后重新计算反馈；重复修正三步，提交后更新历史。", 25, GREY, width=1500)
s.save()

# 08: Existing experimental design, no new runs silently mandated.
s = Slide("08_A_experiments_and_readout", "A 方法 / 当前实验如何回答客户端层面的问题", "Full 与 Flat 的差别，直接检验“客户端来源信息有没有用”",
          "S 是持续 A 更新底座；Current / Full / Flat 保留相同基本训练。A 的方法效果尚待服务器结果回传。",
          "先看来源感知能否改善 Client-LT，再追踪少数来源支持的功能是否在共享更新后保留得更多。",
          "协议：docs/cliplora_sfra_v1.md。Full λ=1/3/10；Current/Flat 当前 λ=10，正式比较须使用相同 λ。", state="A 已实现")
s.panel(100, 290, 1490, 870, "四组设置：每个对照都对应一个明确问题")
xs = [142, 435, 785, 1150]
for x, label in zip(xs, ("实验", "来源优先级", "当前目标", "历史目标")):
    s.text(x, 420, label, 31, GREY, True, width=325)
rows = [("S", "无修正", "—", "—"), ("Current", "有", "有", "无"), ("Full", "有", "有", "有"), ("Flat", "全部相同", "有", "有")]
for i, row in enumerate(rows):
    yy = 501+i*122
    s.box(126, yy-9, 1438, 96, LIGHT_TEAL if row[0] == "Full" else "white", LINE, 9, 1)
    for j, (x, value) in enumerate(zip(xs, row)):
        s.text(x, yy+11, value, 34, TEAL if row[0] == "Full" else NAVY, j == 0, width=320)
s.text(145, 1020, "Full − Flat：来源优先级是否有价值？\nFull − Current：跨轮持续保持是否有价值？", 32, BLUE, True, width=1360)
s.panel(1640, 290, 820, 870, "同时看结果与客户端机制", TEAL)
s.card(1675, 420, 750, 258, "结果：学得充分，也留得更好", "Overall / Head / Middle / Tail\n末 20 轮均值 + Tail 峰后回落\n对照 S、固定 A 的 E2 和定期 E3。", TEAL, "white", size=30)
s.card(1675, 711, 750, 304, "机制：稀缺来源功能的去向", "按有效支持来源多少分组；\n对比普通 A 提案、正式提交 A、\n下一轮 B 训练后的功能。\n不只看所有功能的平均值。", TEAL, "white", size=30)
s.text(1690, 1050, "rank=4；LA τ=1；30/30 客户端；\n历史五轮；修正 3 步 × 0.1。", 28, GREY, width=730)
s.save()

# 09: Historical B evidence explicitly separated from SFRA and LA results.
s = Slide("09_B_evidence_and_hypothesis", "B 方向 / 从已有外部帮助到可用迁移", "没有该尾类样本的客户端，也能提供有益的 B 更新",
          "旧桥接实验已观察到正向外部贡献；同时也存在负向外部贡献。因此，下一步必须识别“哪些帮助值得迁移”。",
          "B 的机会不是把其他客户端全部混进来，而是提取其中真正有利于接收端的学习增量。",
          "来源：data/b_phase_attribution.csv。旧无 LA 的 CLT C1/C2；7 个 B 事件 × 20 尾类均值；不是新方法效果。")
s.panel(100, 290, 1540, 868, "真实归因：正帮助与负影响并存")


def attribution_plot(fig):
    ax = fig.subplots()
    fig.subplots_adjust(left=.095, right=.98, bottom=.23, top=.84)
    labels = ["有该类样本\n正贡献 W", "有该类样本\n负贡献 −H", "无该类样本\n正贡献 D", "无该类样本\n负贡献 −R"]
    for offset, method, color, label, hatch in [(-.19, "c1", BLUE, "C1：固定 A", ""), (.19, "c2", ORANGE, "C2：定期刷新 A", "//")]:
        row = ATTR[method]
        vals = [float(row[k])*sign*1000 for k, sign in (("W",1),("H",-1),("D",1),("R",-1))]
        ax.bar(np.arange(4)+offset, vals, width=.36, color=color, label=label, hatch=hatch, edgecolor="white")
        for x, val in zip(np.arange(4)+offset, vals):
            ax.text(x, val + (.35 if val >= 0 else -.38), f"{val:+.2f}", ha="center", va="bottom" if val >= 0 else "top", fontsize=12, color=color)
    ax.set_xticks(range(4), labels)
    ax.set(ylim=(-13, 12), ylabel="功能归因贡献（log-odds × 1000）")
    ax.axhline(0, color=GREY, lw=1)
    ax.legend(loc="upper left", ncol=2, frameon=False)
    grid(ax)


s.plot((130, 422, 1480, 682), attribution_plot)
s.panel(1685, 290, 775, 868, "下一步要验证的猜想", ORANGE)
s.card(1720, 423, 704, 238, "外部有益知识可以被吸收", "其他客户端学到的共享判别信息，\n可能补充尾部客户端有限的监督。", ORANGE, "white", size=30)
s.card(1720, 694, 704, 254, "只取有益的那部分", "同一外部更新可能有利于一个\n分类关系，却损害另一个关系；\n迁移应由接收端判断。", ORANGE, LIGHT_ORANGE, size=30)
s.text(1730, 991, "旧归因说明“存在帮助”，\n不等于某个外部模型已是好教师；\n当前 LA 框架中仍需直接验证。", 29, GREY, width=680)
s.save()

# 10: Candidate B transfer, specific incremental learning, not parameter mixing.
s = Slide("10_B_transfer_direction", "B 方向 / 接收端验证的有益更新迁移", "不搬走整个 B，只吸收对接收端有用的判别增量",
          "拟保留 LA 与共同的 A；比较外部客户端这轮新学到的东西，是否能改善接收端、且尚未被普通共享 B 充分吸收。",
          "B 模块的核心是“谁来教、教什么、接收端是否真正受益”，而不只是换一种参数平均公式。",
          "以下为设计方向，尚未实现。箭头表示拟议的信息流；判别关系示例不代表实际类别实验或已测收益。", state="B 待验证设想")
cards = [
    (100, "识别新学到的知识", "从共同 A / B 起点出发，\n对比每个外部 B 更新前后\n带来的判别变化。\n不把 CLIP 原有能力算成贡献。"),
    (710, "由接收客户端验证", "在接收端本地图片上，\n检查外部更新是否真有帮助；\n再与普通聚合 B 比较，\n筛掉已经吸收的重复知识。"),
    (1320, "只传递有益关系", "识别对哪些样本、哪些\n易混类别的区分有帮助；\n只迁移这些判别关系，\n不照单全收外部预测。"),
    (1930, "通过 B 学习吸收", "接收端保留真实标签与 LA；\n用额外迁移目标训练 B；\n共享模型也要吸收改善，\n不能只停在本地个性化。"),
]
for i, (x, title, body) in enumerate(cards):
    s.card(x, 334, 530, 470, title, body, ORANGE, LIGHT_ORANGE, i+1, size=30)
    if i < 3:
        s.arrow((x+548, 565), (x+593, 565), ORANGE, dashed=True)
s.panel(100, 867, 2360, 288, "一个接收端例子：同一外部 B，不同判别关系可能一利一弊", ORANGE, "white")
s.box(140, 987, 622, 107, LIGHT_TEAL, TEAL)
s.text(173, 1013, "尾类 vs 易混类别甲：有帮助 → 学", 30, TEAL, True, width=560)
s.box(829, 987, 622, 107, "#FCEFF0", RED)
s.text(862, 1013, "尾类 vs 易混类别乙：有损害 → 不学", 28, RED, True, width=560)
s.text(1520, 984, "迁移的是经过接收端检验的关系，\n不是把外部类别标签或整个 B 复制过来。", 32, NAVY, True, width=855)
s.save()

# 11: Plain FL/long-tail vocabulary; B and the joint training remain proposed.
s = Slide("11_AB_joint_story", "整体方法 / 让尾部客户端学得更多，学到的知识留得住", "A 缓解客户端长尾，B 帮助尾部客户端学习更多知识",
          "尾部客户端既面临本地数据不足，也面临学到的知识难以在全局模型中保留：两个模块分别处理这两个问题。",
          "共同目标：让尾部客户端学得更充分，让全局模型更少遗忘尾类知识。",
          "A 在普通聚合后修正全局参数，不直接改变样本量聚合权重；B 和 A+B 联合训练仍是待验证的设计。", state="A 已实现 / B 设想")
s.panel(100, 295, 1110, 700, "A：让尾部客户端更好地影响全局模型", BLUE, LIGHT_BLUE)
s.text(145, 418, "要解决的问题", 31, BLUE, True, width=1010)
s.text(145, 478, "尾类数据集中在少数小客户端，\n全局更新容易忽略它们学到的知识。", 36, NAVY, width=1020)
s.text(145, 609, "我们怎么做", 31, BLUE, True, width=1010)
s.text(145, 668, "① 找到能帮助本地类别的客户端更新。\n② 只有少数客户端能帮助的分类能力，优先保护。\n③ 修正聚合后的 A，保留已经学到的知识。", 33, NAVY,
       width=1020, max_y=833, spacing=1.55)
s.box(140, 865, 1030, 95, "white", BLUE)
s.text(171, 891, "缓解客户端之间的不平衡，减少尾类遗忘。", 33, BLUE, True, width=967)
s.panel(1350, 295, 1110, 700, "B：用其他客户端的知识补充尾部学习", ORANGE, LIGHT_ORANGE)
s.text(1395, 418, "要解决的问题", 31, ORANGE, True, width=1010)
s.text(1395, 478, "尾部客户端样本少，单靠本地训练学不充分；\n其他客户端也可能提供对尾类有用的知识。", 36, NAVY, width=1020)
s.text(1395, 609, "计划怎么做", 31, ORANGE, True, width=1010)
s.text(1395, 668, "① 找出其他客户端对尾类有用的知识。\n② 用尾部客户端的数据检查是否真的有帮助。\n③ 保留 LA，再用这些知识帮助尾部客户端训练 B。", 33, NAVY,
       width=1020, max_y=833, spacing=1.55)
s.box(1390, 865, 1030, 95, "white", ORANGE)
s.text(1421, 891, "通过跨客户端知识迁移，补充尾类训练信息。", 33, ORANGE, True, width=967)
s.text(105, 1034, "计划中的组合训练", 30, GREY, True, width=700)
for x, width, title, color, fill in [
    (100, 780, "训练 B：本地 LA + 有益知识迁移", ORANGE, LIGHT_ORANGE),
    (965, 840, "更新 A：聚合后保护已经学到的知识", BLUE, LIGHT_BLUE),
    (1890, 570, "进入下一轮联邦训练", TEAL, LIGHT_TEAL),
]:
    s.box(x, 1090, width, 86, fill, color, 12)
    s.text(x+width/2, 1132, title, 31, color, True, anchor="mm")
s.arrow((898, 1133), (947, 1133), ORANGE, dashed=True)
s.arrow((1823, 1133), (1872, 1133), BLUE, dashed=True)
s.save()

# 12: Minimal decision-oriented next steps, not expansion of role experiments.
s = Slide("12_validation_and_next_steps", "下一步 / 先验证当前 A，再决定 B 如何落地", "闭环要同时看到：客户端贡献被兼顾，尾类收益被兑现",
          "已有结果支撑从客户端分布出发干预 A；接下来围绕方法结果推进，不再重新铺开 A/B 知识角色定义实验。",
          "最终目标：充分适配时，少数客户端的有效需求不被掩盖，外部客户端的有益知识也能被尾部学习利用。",
          "这是后续验证顺序，不是已完成结论。标准 Dir、多种子与跨数据集验证，在核心方法出现有效信号后推进。", state="验证计划")
items = [
    (100, BLUE, "先读当前 A 实验", "Full λ=1 / 3 / 10\nCurrent、Flat 与同 λ Full 对比\n已有同协议 S 作为持续更新参照。", "要回答：来源感知是否真有用？"),
    (914, ORANGE, "再做最小 B 迁移验证", "在同一个 A 方案上比较：\n原 LA-B vs LA-B + 有益迁移。\n记录接收端受益及共享模型是否吸收。", "要回答：外部帮助能否真正转化？"),
    (1728, TEAL, "最后检验组合与机制", "检查支持稀缺功能的聚合后保持；\n同时报告 Head / Middle / Tail；\n核对适配幅度、计算和通信成本。", "要回答：如何改善，而不只是分数涨了？"),
]
for i, (x, color, title, body, question) in enumerate(items):
    s.card(x, 341, 732, 487, title, body, color, LIGHT_BLUE if i == 0 else LIGHT_ORANGE if i == 1 else LIGHT_TEAL, i+1, size=33)
    s.box(x, 865, 732, 132, "white", color)
    s.text(x+28, 893, question, 31, color, True, width=676, max_y=984)
    if i < 2:
        s.arrow((x+748, 571), (x+797, 571), color)
s.box(100, 1042, 2360, 132, LIGHT, LINE)
s.text(135, 1065, "汇报收尾", 31, BLUE, True, width=220)
s.text(390, 1065, "我们不是给 A 加一个泛化的“防遗忘模块”，而是根据客户端之间谁在有效支持谁，\n改变共享 A 的更新；再通过 B，让尾部客户端主动吸收其他客户端的有益知识。", 32, NAVY, True, width=2020)
s.save()

# Index image and a PNG-only convenience archive.
contact = Image.new("RGB", (2560, 30+math.ceil(len(GENERATED)/3)*490), "#E9EFF7")
for i, name in enumerate(GENERATED):
    with Image.open(HERE / f"{name}.png") as im:
        contact.paste(im.resize((800, 450), Image.Resampling.LANCZOS), (40+(i%3)*840, 30+(i//3)*490))
contact.save(HERE / "00_contact_sheet.png", optimize=True)
validation = dict(
    count=len(GENERATED), slide_size=[W, H], format="PNG only", images=GENERATED,
    layout_warnings=WARNINGS, evidence_checks=CHECKS,
    data_sources={name: str(path.relative_to(ROOT)) for name, path in SOURCES.items()},
    no_new_experimental_results=True, A_status="implemented; this edit condenses the method only",
    B_status="conceptual direction, not implemented", matched_dirichlet_in_current_comparison=False,
    schematic_pages=[3, 4, 7, 8, 9],
    audit="See 设计审查与数据口径.md for visual review and evidence boundaries.",
)
(HERE / "validation.json").write_text(json.dumps(validation, ensure_ascii=False, indent=2), encoding="utf-8")
assert len(GENERATED) == 9
assert not WARNINGS, WARNINGS
assert not list(HERE.glob("*.pdf"))
with zipfile.ZipFile(HERE / "A方法与B方向_精简9页PNG.zip", "w", zipfile.ZIP_DEFLATED) as z:
    for page, name in enumerate(GENERATED, 1):
        z.write(HERE / f"{name}.png", arcname=f"{page:02d}_{name[3:]}.png")
print("Complete: 9 full-size PNG slides with one concise A overview. No training.", flush=True)
