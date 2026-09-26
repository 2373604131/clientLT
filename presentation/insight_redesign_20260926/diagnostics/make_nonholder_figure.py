"""Plot a bounded descriptive finding from audited, real Full-CP responses."""
from pathlib import Path
import csv
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch
import numpy as np


HERE = Path(__file__).resolve().parent


def read_csv(name):
    with (HERE / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def main():
    summary = {r["group"]: r for r in read_csv("nonholder_sources_summary.csv") if r["averaging"] == "class_equal"}
    rounds = read_csv("nonholder_sources_rounds.csv")
    plt.rcParams.update({"font.family": "Arial", "font.size": 9, "axes.labelsize": 9, "xtick.labelsize": 8.5, "ytick.labelsize": 9, "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "axes.linewidth": .7})
    fig = plt.figure(figsize=(17.5 / 2.54, 8.3 / 2.54), facecolor="white")
    dark, gray, pale = "#172B3A", "#73808C", "#D9E0E5"
    fig.text(.045, .955, "Potential positive sources extend beyond label ownership", fontsize=11.3, fontweight="bold", color=dark, va="top")
    fig.text(.045, .884, "Protected Full-CP trajectory  ·  descriptive", fontsize=9.2, color="#5D6973")
    fig.text(.045, .793, "(a) Distinct measurements", fontsize=9.5, fontweight="bold", color=dark)
    fig.text(.465, .793, "(b) Non-holder response and candidate shares", fontsize=9.5, fontweight="bold", color=dark)

    schematic = fig.add_axes([.045, .23, .345, .48])
    schematic.set(xlim=(0, 1), ylim=(0, 1)); schematic.axis("off")
    schematic.text(.015, .99, "SCHEMATIC", fontsize=8.5, color=gray, va="top")
    def box(x, y, w, h, label, fc, edge):
        schematic.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.01,rounding_size=0.025", linewidth=.9, facecolor=fc, edgecolor=edge))
        schematic.text(x + w / 2, y + h / 2, label, ha="center", va="center", fontsize=9, color=dark)
    box(.015, .58, .39, .24, "Client holds\nclass c", "#EDF2F5", "#92A4B0")
    box(.015, .16, .39, .24, "Client has\nno class c", "white", "#92A4B0")
    box(.68, .37, .305, .25, "Margin for\nclass c", "#E8F2F7", "#0072B2")
    for y in [.70, .28]:
        schematic.add_patch(FancyArrowPatch((.415, y), (.67, .495), arrowstyle="-|>", mutation_scale=10, linewidth=1.15, linestyle="--", color="#0072B2", connectionstyle="arc3,rad=0"))
    schematic.text(.50, .07, "Both may have a\npositive first-order response", ha="center", va="top", color="#425461", fontsize=8.5)
    fig.add_artist(Line2D([.413, .413], [.20, .74], transform=fig.transFigure, color=pale, linewidth=.8))

    ax = fig.add_axes([.545, .275, .405, .36])
    colors = {"tail20": "#C36A28", "non_tail80": "#0072B2"}
    groups = [("tail20", 1, "Tail20"), ("non_tail80", 0, "Non-tail80")]
    for group, y, label in groups:
        data = summary[group]
        response = float(data["nonholder_positive_share_given_supported_mean_over_rounds"]) * 100
        candidate = float(data["nonholder_candidate_share_given_supported_mean_over_rounds"]) * 100
        round_data = [r for r in rounds if r["group"] == group and r["averaging"] == "class_equal"]
        values = np.array([float(r["nonholder_positive_share_given_supported"]) * 100 for r in round_data])
        jitter = (np.arange(len(values)) % 9 - 4) * .015
        ax.scatter(values, y + .12 + jitter, s=7, color=colors[group], alpha=.19, edgecolors="none", zorder=2)
        ax.plot([response, candidate], [y + .12, y - .16], color="#B5BEC6", lw=.8, zorder=1)
        ax.scatter([response], [y + .12], s=48, color=colors[group], marker="o", edgecolors="white", linewidths=.8, zorder=4)
        ax.scatter([candidate], [y - .16], s=40, facecolors="white", edgecolors=gray, marker="D", linewidths=1.1, zorder=3)
        ax.text(response, y + .34, f"{response:.2f}%", ha="center", fontsize=9.5, color=colors[group], fontweight="bold")
        ax.text(candidate, y - .43, f"{candidate:.2f}%", ha="center", fontsize=8.5, color=gray)
    ax.set(xlim=(0, 100), ylim=(-.62, 1.64), xticks=[0, 25, 50, 75, 100], yticks=[1, 0], yticklabels=["Tail20", "Non-tail80"], xlabel="Non-holder share (%)")
    ax.grid(axis="x", color="#E7ECF0", lw=.6); ax.set_axisbelow(True)
    ax.tick_params(axis="y", length=0, pad=9)
    ax.spines[["top", "right", "left"]].set_visible(False)
    ax.spines["bottom"].set_color("#AAB5BD")
    legend = [Line2D([], [], marker="o", ls="none", markersize=5, color="#3C5667", label="Positive first-order response"), Line2D([], [], marker="D", ls="none", markersize=5, markerfacecolor="white", markeredgecolor=gray, label="Candidate clients (count reference)")]
    fig.legend(handles=legend, loc="upper left", bbox_to_anchor=(.463, .75), frameon=False, fontsize=8.5, handletextpad=.55, borderaxespad=0, labelspacing=.5)
    fig.text(.465, .11, "Pale points: 90 rounds, not independent seeds.", fontsize=8.5, color=gray)
    fig.text(.045, .045, "A non-holder positive source exists for 98.65% of supported Tail units and 95.85% of supported Non-tail units.*", fontsize=8.5, color="#425461")
    fig.canvas.draw()
    stem = HERE / "insight2_functional_sources_candidate"
    fig.savefig(stem.with_suffix(".png"), dpi=300)
    fig.savefig(stem.with_suffix(".pdf"))
    fig.savefig(stem.with_suffix(".svg"))
    caption = """# Insight 2 descriptive figure candidate

**Potential positive update sources are not restricted to clients holding the target class.** (a) Conceptual schematic; the arrows represent possible positive first-order responses and do not identify measured individual clients. (b) Full-CP, Client-LT, seed42, all A-update rounds 1–90. Filled markers show the non-holder fraction of positive first-order response magnitude; hollow diamonds show the corresponding fraction of candidate clients lacking the target class. The latter is a candidate-count composition reference, not a statistical null model. Ownership comes from the complete actual partition manifest. For each client–class witness unit, take the minimum response across the two views and retain it only when greater than 10⁻⁶, exactly as in the implementation. Divide the sum of retained non-holder responses by the sum over all clients. Exclude units with no positive response; average supported units within each class, then classes equally, then all 90 rounds equally. Apply identical conditional averaging to candidate-count shares. Pale points are individual round-level class-balanced response shares; they are not independent runs or confidence intervals. Tail20 includes 73 client–class units; Non-tail80 includes 1,443. Of 6,570 Tail unit–round records, 488 lack positive support and are excluded from conditional metrics; the corresponding Non-tail counts are 129,870 and 6,604. Classes without a supported unit in a round are excluded from that round's conditional class mean. *The existence percentages use the same class-balanced, then round-balanced conditional averaging.

Non-holder response shares are 86.46% for Tail20 and 33.52% for Non-tail80; their candidate-count shares are 87.81% and 39.81%, respectively. Thus the large tail response share does not demonstrate greater per-client effectiveness or excess response beyond source availability. These responses are unweighted gradient–update inner products on an already protected Full-CP trajectory, not FedAvg-weighted realized gains, finite-update improvements, actual B-transfer gains, unprotected-baseline behavior, or proof of causality/generalization. Witnesses are training samples.

中文解读：在已有 Full-CP 轨迹中，有目标类别数据的客户端并不囊括所有一阶正向响应来源。尾类非持有者的正向响应份额为86.46%，但它们在候选客户端中的占比也达到87.81%；不能由前一个大数推断它们更高效、更强或贡献了同等比例的真实知识。图支持“不能直接用标签持有关系代替功能支持关系”这一有限判断。

Design audit: supporting experimental figure, paired composition points plus explicitly labelled concept schematic; 17.5 cm × 8.3 cm; smallest font 8.5 pt at that width; PDF/SVG vectors and 300-dpi PNG. Full 0–100% axis, direct numeric labels, filled circle versus hollow diamond encoding. No confidence intervals, smoothing, simulated measurements, or method-effect claims. Round dots describe trajectory variation only. This figure is designed for 17.5 cm width; a 9 cm layout requires reflow rather than scaling.

Reproduce after the audit script: `python presentation/insight_redesign_20260926/diagnostics/make_nonholder_figure.py`.
"""
    (HERE / "insight2_functional_sources_candidate_caption.md").write_text(caption, encoding="utf-8")
    print(json.dumps({"figure": str(stem), "width_cm": 17.5, "height_cm": 8.3, "minimum_font_pt": 8.5}, ensure_ascii=False))


if __name__ == "__main__":
    main()
