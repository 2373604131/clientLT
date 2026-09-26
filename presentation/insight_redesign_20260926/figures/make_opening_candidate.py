"""Create an evidence-based opening-figure candidate from existing CSV exports.

Run from anywhere: python make_opening_candidate.py
No synthetic values, smoothing, aggregation across seeds, or inferred error bars.
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
DATA = ROOT / "presentation" / "insights_20260925" / "data"
performance = pd.read_csv(DATA / "performance_retention.csv")
curves = pd.read_csv(DATA / "training_curves.csv")
rows = performance[performance["method"].eq("S")].set_index("partition")
CLT, DIR = "client-longtail", "noniid-labeldir-fine"
clt, direct = rows.loc[CLT], rows.loc[DIR]
assert clt["seed"] == direct["seed"] == 42
for part in [CLT, DIR]:
    trace = curves[curves["method"].eq("S") & curves["partition"].eq(part)].sort_values("round")
    for metric in ["overall", "nontail", "tail"]:
        assert np.isclose(trace.tail(20)[metric].mean(), rows.loc[part, metric])
    assert np.isclose(trace["tail"].max(), rows.loc[part, "tail_peak"])
    assert np.isclose(trace.iloc[-1]["tail"], rows.loc[part, "tail_final"])
colors = {CLT: "#C76B32", DIR: "#327DA7"}
ink, gray = "#253746", "#65717C"
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8, "axes.titlesize": 9,
    "xtick.labelsize": 8, "ytick.labelsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42,
    "svg.fonttype": "none", "axes.spines.top": False,
    "axes.spines.right": False, "axes.edgecolor": "#A7B1B8",
    "axes.labelcolor": ink, "text.color": ink,
    "xtick.color": gray, "ytick.color": gray,
})
fig = plt.figure(figsize=(17.5 / 2.54, 9 / 2.54), facecolor="white")
fig.text(.027, .965, "Same global data, different tail retention", fontsize=11,
         weight="bold", va="top")

# Left: deliberately schematic protocol, with no invented allocation statistics.
protocol = fig.add_axes([.024, .135, .259, .717])
protocol.set(xlim=(0, 1), ylim=(0, 1))
protocol.axis("off")
protocol.text(.0, 1.02, "Fixed global data", fontsize=9, weight="bold")

def box(x, y, w, h, text, edge="#BDC6CC", face="#F5F7F8", color=ink):
    protocol.add_patch(FancyBboxPatch((x, y), w, h,
        boxstyle="round,pad=0.012,rounding_size=0.026",
        linewidth=.9, edgecolor=edge, facecolor=face))
    protocol.text(x+w/2, y+h/2, text, ha="center", va="center",
                  fontsize=8, color=color, linespacing=1.35)

def arrow(start, end):
    protocol.add_patch(FancyArrowPatch(start, end, arrowstyle="-|>",
        mutation_scale=9, linewidth=.8, color="#8D99A2"))

box(.05, .70, .90, .22, "Same image pool\nSame class counts")
arrow((.33, .69), (.25, .58))
arrow((.67, .69), (.75, .58))
box(.015, .43, .45, .15, "Client-LT", colors[CLT], "#FBF0E8", colors[CLT])
box(.535, .43, .45, .15, "Dirichlet", colors[DIR], "#EAF3F8", colors[DIR])
arrow((.25, .42), (.33, .30))
arrow((.75, .42), (.67, .30))
box(.05, .08, .90, .22, "Same LA objective\nSame staged update")
protocol.text(.5, -.035, "Two client allocations", ha="center", color=gray, fontsize=8)

# Upper right: encode the comparison itself, rather than juxtaposing raw scores.
fig.text(.372, .86, "Overall hides the tail gap", fontsize=9, weight="bold")
effect = fig.add_axes([.48, .665, .485, .16])
groups = ["Overall", "Non-tail", "Tail"]
keys = ["overall", "nontail", "tail"]
deltas = [float(clt[k] - direct[k]) for k in keys]
ys = [2, 1, 0]
effect.axvline(0, color="#8E989F", lw=.8, zorder=0)
for y, g, delta in zip(ys, groups, deltas):
    c = colors[CLT] if g == "Tail" else "#657789"
    effect.plot([0, delta], [y, y], color=c, lw=2.1, solid_capstyle="round")
    effect.scatter(delta, y, s=25, color=c, zorder=3)
    # Labels stay away from the category text and the zero reference.
    if delta < -1:
        effect.text(delta + .18, y + .24, f"{delta:+.3f}", color=c, va="center", fontsize=8)
    elif delta < 0:
        effect.text(.18, y, f"{delta:+.3f}", color=c, va="center", fontsize=8)
    else:
        effect.text(delta + .17, y, f"{delta:+.3f}", color=c, va="center", fontsize=8)
effect.set(xlim=(-4.15, 1.6), ylim=(-.5, 2.5), yticks=ys, yticklabels=groups,
           xticks=[-4, -2, 0, 1], xticklabels=["−4", "−2", "0", "+1"])
effect.tick_params(axis="y", length=0, pad=7)
effect.tick_params(axis="x", length=2.5, pad=3)
effect.spines["left"].set_visible(False)
effect.spines["bottom"].set_visible(False)
effect.set_xlabel("Client-LT − Dirichlet (pp)", labelpad=2)

# Lower right: unsmoothed round-level observations from the same two runs.
fig.text(.372, .495, "The gap develops during training", fontsize=9, weight="bold")
ax = fig.add_axes([.395, .145, .570, .305])
ax.axvspan(90, 100, color="#65717C", alpha=.065, linewidth=0)
ax.axvline(90, color="#AAB2B9", lw=.65, ls=(0, (2, 3)))
for part, label, linestyle in [(DIR, "Dirichlet", "-"), (CLT, "Client-LT", (0, (4, 2)))]:
    d = curves[curves["method"].eq("S") & curves["partition"].eq(part)].sort_values("round")
    assert np.array_equal(d["round"].to_numpy(), np.arange(101))
    ax.plot(d["round"], d["tail"], color=colors[part], lw=1.6,
            label=label, ls=linestyle)
    r = rows.loc[part]
    ax.scatter([r["peak_round"], 100], [r["tail_peak"], r["tail_final"]],
               s=17, facecolor="white", edgecolor=colors[part], lw=1, zorder=4)
    ax.plot([r["peak_round"], 103], [r["tail_peak"], r["tail_peak"]],
             color=colors[part], lw=.7, ls=(0, (2, 2)), alpha=.75)
    ax.plot([100, 103], [r["tail_final"], r["tail_final"]],
             color=colors[part], lw=.7)
    ax.plot([103, 103], [r["tail_final"], r["tail_peak"]],
             color=colors[part], lw=.9)
    mid = (r["tail_peak"] + r["tail_final"]) / 2
    ax.text(99, mid, f"{r['drop']:.2f} pp drop", ha="right", va="center",
             fontsize=8, color=colors[part],
             bbox=dict(facecolor="white", edgecolor="none", alpha=.85, pad=.4))
ax.text(94.7, 66.7, "B-only", ha="center", color=gray, fontsize=8)
ax.set(xlim=(0, 105), ylim=(65.8, 74.8), xticks=[0, 25, 50, 75, 100],
       yticks=[66, 68, 70, 72, 74], xlabel="Communication round", ylabel="Tail accuracy (%)")
ax.tick_params(length=2.5, pad=3)
ax.set_xlabel("Communication round", labelpad=2)
ax.set_ylabel("Tail accuracy (%)", labelpad=5)
ax.legend(loc="upper left", frameon=False, ncol=2, handlelength=1.7,
          borderaxespad=.25, columnspacing=1.2, handletextpad=.45, fontsize=8)
ax.grid(axis="y", color="#E9EDF0", lw=.5)
ax.set_axisbelow(True)

basename = HERE / "opening_same_pool_retention_candidate"
for ext in ["pdf", "svg", "png"]:
    fig.savefig(basename.with_suffix("." + ext), dpi=300, facecolor="white")
plt.close(fig)

caption = """DESIGN CANDIDATE — supplied evidence only; not a final paper figure.

Same global data, different tail retention. Two client allocations use the same
global image pool and global class counts, the same logit-adjusted (LA) objective,
and the same staged update recipe (S). Client-LT is compared with label-Dirichlet
allocation (beta=0.5). The upper panel reports Client-LT minus Dirichlet in macro
accuracy, averaged over the last 20 logged rounds (81–100): overall −0.104 pp,
non-tail +0.77875 pp, and tail −3.635 pp. Non-tail combines the fixed head and
middle class sets with equal weight per class; it is not an unweighted average
of two group means. The lower panel plots all 101 raw tail-accuracy observations
without smoothing. Peak-to-final drop is the run's maximum tail accuracy over
rounds 0–100 minus its accuracy at round 100: Client-LT 71.80% at round 46 to
68.35%, drop 3.45 pp; Dirichlet 72.90% at round 57 to 72.00%, drop 0.90 pp.
Peak circles and brackets are descriptive, retrospective summaries, not causal
estimates or model-selection recommendations. Gray shading marks the B-only
stage after round 90.

Evidence limitations: these are single-seed (42), macro class-accuracy results.
There are no replicate error bars or significance claims. Allocation changes
several aspects of local data composition; this figure does not isolate a causal
effect of class concentration or identify effective knowledge-source clients.
Identical training recipes do not imply identical realized compute: logged
optimizer steps are 137,280 (Client-LT) and 138,060 (Dirichlet). Thus the left
schematic deliberately does not claim that allocation is the only changing
quantity. The upper panel's last-20-round mean and the lower panel's round-100
endpoint are different summaries and must not be conflated. All numbers come
from the supplied CSV exports; global-pool matching is a protocol condition
documented in the source audit, not inferred from these two aggregate CSVs.

Source CSVs:
presentation/insights_20260925/data/performance_retention.csv
presentation/insights_20260925/data/training_curves.csv
"""
basename.with_suffix(".caption.txt").write_text(caption, encoding="utf-8")
audit = {
    "status": "design candidate", "seed": 42,
    "comparison": "Client-LT minus Dirichlet, S with LA",
    "last20_delta_pp": dict(zip(keys, deltas)),
    "peak_to_final_drop_pp": {p: float(rows.loc[p, "drop"]) for p in [CLT, DIR]},
    "optimizer_steps": {p: int(rows.loc[p, "optimizer_steps"]) for p in [CLT, DIR]},
    "source_csvs": [str(DATA / "performance_retention.csv"), str(DATA / "training_curves.csv")],
}
basename.with_suffix(".audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(json.dumps(audit, ensure_ascii=False, indent=2))
print("Saved:", basename)
