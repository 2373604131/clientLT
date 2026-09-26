"""Historical no-CP v1 ablation, from checked per-class observations.

This design candidate is a method ablation, not a new causal discovery or the
newer current-cp experiment. Run: python make_retention_ablation_candidate.py
"""
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
BASE = ROOT / "output" / "sfra_v1_analysis" / "sfra_v1"
summary = pd.read_csv(BASE / "analysis" / "performance.csv")
curves = pd.read_csv(BASE / "analysis" / "curves.csv")
summary = summary[summary["retention_weight"].eq(10) & summary["method"].isin(["full", "current"])].set_index("method")
selected = {}
stats = {}
for method in ["full", "current"]:
    run = BASE / "seed42" / "client-longtail" / method / "lambda10_protocol42"
    prior = json.loads((run / "class_prior.json").read_text(encoding="utf-8"))
    order = np.argsort(-np.asarray(prior["counts"]), kind="stable")
    tail_ids = order[-20:]
    trace = curves[curves["method"].eq("sfra_" + method) & curves["retention_weight"].eq(10)].sort_values("round")
    assert len(trace) == 101 and np.array_equal(trace["round"], np.arange(101))
    # Verify every exported point against its original per-class evaluation.
    for _, record in trace.iterrows():
        raw = pd.read_csv(run / f"per_class_accuracy_epoch_{int(record['epoch'])}.csv").set_index("class_id")
        assert np.isclose(raw.loc[tail_ids, "per_class_acc"].mean(), record["bottom20_tail_acc"])
        assert np.isclose(raw["per_class_acc"].mean(), record["overall_acc"])
    tail_mean = float(trace.tail(20)["bottom20_tail_acc"].mean())
    overall_mean = float(trace.tail(20)["overall_acc"].mean())
    peak_idx = trace["bottom20_tail_acc"].idxmax()
    tail_peak = float(trace.loc[peak_idx, "bottom20_tail_acc"])
    peak_round = int(trace.loc[peak_idx, "round"])
    tail_final = float(trace.iloc[-1]["bottom20_tail_acc"])
    drop = tail_peak - tail_final
    assert np.isclose(tail_mean, summary.loc[method, "last20_bottom20_tail_acc"])
    assert np.isclose(overall_mean, summary.loc[method, "last20_overall_acc"])
    assert np.isclose(drop, summary.loc[method, "tail_peak_to_final"])
    selected[method] = trace
    stats[method] = {"last20_tail": tail_mean, "last20_overall": overall_mean,
                     "tail_peak": tail_peak, "tail_peak_round": peak_round,
                     "tail_final": tail_final, "peak_to_final_drop": drop}

delta_overall = stats["full"]["last20_overall"] - stats["current"]["last20_overall"]
delta_tail = stats["full"]["last20_tail"] - stats["current"]["last20_tail"]
drop_reduction = stats["current"]["peak_to_final_drop"] - stats["full"]["peak_to_final_drop"]
assert np.isclose(delta_overall, -.296)
assert np.isclose(delta_tail, 2.110)
assert np.isclose(drop_reduction, 2.55)

ink, muted = "#253746", "#65717C"
colors = {"full": "#318775", "current": "#C76B32"}
plt.rcParams.update({
    "font.family": "DejaVu Sans", "font.size": 8,
    "axes.labelsize": 8, "xtick.labelsize": 8, "ytick.labelsize": 8,
    "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none",
    "axes.spines.top": False, "axes.spines.right": False,
    "axes.edgecolor": "#A7B1B8", "axes.labelcolor": ink,
    "text.color": ink, "xtick.color": muted, "ytick.color": muted,
})
fig = plt.figure(figsize=(17.5/2.54, 8/2.54), facecolor="white")
fig.text(.027, .96, "Similar early peaks, different late retention", fontsize=11,
         weight="bold", va="top")
fig.text(.027, .865, "Historical ablation · Without CP · λ = 10 · seed 42",
         fontsize=8.5, color=muted)

ax = fig.add_axes([.085, .185, .618, .605])
ax.axvspan(81, 100, facecolor="#EEF1F3", linewidth=0)
ax.axvline(81, color="#B8C1C7", lw=.65, ls=(0,(2,3)))
for method, label, linestyle in [
    ("full", "Full (current + history)", "-"),
    ("current", "Current (current only)", (0,(4,2)))]:
    trace = selected[method]
    ax.plot(trace["round"], trace["bottom20_tail_acc"], color=colors[method],
             lw=1.65, ls=linestyle, label=label)
    mean = stats[method]["last20_tail"]
    # A separate horizontal segment denotes the 81–100 mean, not the endpoint.
    ax.plot([81, 100], [mean, mean], color=colors[method], lw=2.5, alpha=.9)
    ax.plot([100, 103], [mean, mean], color=colors[method], lw=.8)
    ax.text(104.5, mean, f"{mean:.4f}%", va="center", color=colors[method], fontsize=8)
ax.text(112, 72.7, "Last 20\nmean", va="center", ha="center", fontsize=8, color=muted)
ax.text(90.5, 66.48, "81–100", ha="center", fontsize=8, color=muted)
ax.set(xlim=(0,126), ylim=(65.8,73.7), xticks=[0,25,50,75,100],
       yticks=[66,68,70,72], xlabel="Communication round", ylabel="Tail accuracy (%)")
ax.set_xlabel("Communication round", labelpad=5)
ax.set_ylabel("Tail accuracy (%)", labelpad=6)
ax.tick_params(length=2.5, pad=3)
ax.spines["bottom"].set_bounds(0,100)
ax.grid(axis="y", color="#E9EDF0", lw=.5)
ax.set_axisbelow(True)
ax.legend(loc="upper left", frameon=False, fontsize=8,
          handlelength=2.0, labelspacing=.45, handletextpad=.6, borderaxespad=.4)

# A small textual comparison makes the trade-off explicit without another chart.
side = fig.add_axes([.764, .175, .213, .63])
side.axis("off")
side.set(xlim=(0,1), ylim=(0,1))
side.text(0, .99, "Full − Current", fontsize=9, weight="bold", va="top")
side.text(0, .80, "Late overall", fontsize=8, color=muted)
side.text(0, .67, f"{delta_overall:+.3f} pp", fontsize=12, color=ink)
side.text(0, .50, "Late tail", fontsize=8, color=muted)
side.text(0, .37, f"{delta_tail:+.3f} pp", fontsize=12, color=colors["full"], weight="bold")
side.plot([0,1], [.29,.29], color="#DCE3E7", lw=.8)
side.text(0, .19, "Peak-to-final drop", fontsize=8, color=muted)
side.text(0, .07, "3.25 → 0.70 pp", fontsize=10, color=ink)
side.text(0, -.04, f"{drop_reduction:.2f} pp smaller", fontsize=8, color=colors["full"])

stem = HERE / "insight3_history_retention_candidate"
for ext in ["png", "pdf", "svg"]:
    fig.savefig(stem.with_suffix("."+ext), dpi=300, facecolor="white")
plt.close(fig)

caption = """DESIGN CANDIDATE — historical method ablation, not a new causal discovery.

Similar early peaks, different late retention. This comparison uses the original
SFRA v1 Full and Current variants on the Client-LT allocation, without CP, at the
same retention weight lambda=10 and seed=42. Full uses current and historical
retention information; Current uses current information only. The original
analysis explicitly states that the variants use the same rules and budget,
but their evolving targets are not numerically identical. Therefore the curves
are a method-level historical ablation and must not be described as the newer
current-cp experiment or as an isolated causal intervention on target history.

The main panel shows all raw tail macro-accuracy observations from rounds 0–100.
The shaded window is rounds 81–100, not uncertainty. Horizontal segments show
the mean over that window: Full 71.0725%, Current 68.9625%. These are not the
round-100 endpoints, which are 70.85% and 68.55%, respectively. Full minus Current
is +2.110 percentage points (pp) for late tail accuracy and −0.296 pp for late
overall accuracy (70.565% versus 70.861%). Thus the comparison includes a trade-
off and should not be presented as a gain on every group or as universal Pareto
improvement. Peak-to-final tail drop, defined as max over rounds 0–100 minus the
round-100 value, changes from 3.25 pp for Current (71.80%, round 48) to 0.70 pp
for Full (71.55%, round 64), a reduction of 2.55 pp. This descriptive metric is
different from the last-20-round mean. No significance claim, error band, or
best-checkpoint model selection is included.

Evidence scope: single seed, one original v1 configuration, retrospective
method ablation. This supports examining persistent retention in addition to
current information; it does not prove that every current-only method must
forget, nor does it identify effective client sources or establish a mechanism.

All 202 curve points were checked against the original per-class accuracy CSVs.
The last-20 means and peak-to-final summaries were independently recomputed and
matched against analysis/performance.csv. Tail is the fixed bottom 20 classes
by the saved global class counts, with equal weight per class.

Sources under output/sfra_v1_analysis/sfra_v1/:
analysis/curves.csv; analysis/performance.csv; analysis/report.md
seed42/client-longtail/full/lambda10_protocol42/per_class_accuracy_epoch_*.csv
seed42/client-longtail/current/lambda10_protocol42/per_class_accuracy_epoch_*.csv
"""
stem.with_suffix(".caption.txt").write_text(caption, encoding="utf-8")
audit = {"status": "design candidate; historical no-CP v1 method ablation",
         "lambda": 10, "seed": 42, "partition": "client-longtail",
         "variants": stats, "full_minus_current_last20_overall_pp": delta_overall,
         "full_minus_current_last20_tail_pp": delta_tail,
         "peak_to_final_drop_reduction_pp": drop_reduction,
         "verified_raw_curve_points": 202}
stem.with_suffix(".audit.json").write_text(json.dumps(audit, indent=2), encoding="utf-8")
print(json.dumps(audit, indent=2))
print("Saved:", stem)
