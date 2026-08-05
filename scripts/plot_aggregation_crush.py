#!/usr/bin/env python
"""Plot the aggregation-crush money figure.

Reads ``aggregation_crush_per_class.csv`` (and optionally the per-client file)
emitted by ``utils.aggregation_crush`` across one or more runs, and produces the
decisive mechanism figure:

  * x-axis: a tail class's support-client MASS share ``n_S / N``;
  * y-axis: accuracy on that class;
  * one series for pre-aggregation LOCAL (best-of-supporting-clients) accuracy;
  * one series for post-aggregation GLOBAL (FedAvg) accuracy.

The gap between the two, growing as mass share shrinks, is the crush: strong
local tail knowledge that data-mass-weighted FedAvg discards. It simultaneously
argues against reweighting -- the local signal it would amplify is a
high-variance estimate from few samples (small ``n_S``).

Aggregates over rounds/seeds by default; use ``--epoch`` to pin one round.
"""

from __future__ import annotations

import argparse
import csv
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--per-class-csv",
        nargs="+",
        required=True,
        help="One or more aggregation_crush_per_class.csv files (globs allowed by the shell).",
    )
    parser.add_argument("--output", default="output/aggregation_crush/figure_crush.png")
    parser.add_argument("--epoch", type=int, default=None, help="Pin one 0-based epoch; default aggregates all logged rounds.")
    parser.add_argument("--nbins", type=int, default=8, help="Number of mass-share bins for the binned trend lines.")
    parser.add_argument("--title", default="Aggregation crushes concentrated tail knowledge")
    return parser.parse_args()


def read_rows(paths: list[str]) -> list[dict]:
    rows: list[dict] = []
    for pattern in paths:
        for path in sorted(Path().glob(pattern)) or [Path(pattern)]:
            if not Path(path).exists():
                print(f"warning: no such file: {path}")
                continue
            with open(path, newline="", encoding="utf-8") as f:
                rows.extend(csv.DictReader(f))
    return rows


def _f(value: str) -> float:
    try:
        v = float(value)
        return v if math.isfinite(v) else math.nan
    except (TypeError, ValueError):
        return math.nan


def binned_trend(xs: np.ndarray, ys: np.ndarray, nbins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Mean +/- sem of ``ys`` within equal-width bins of ``xs`` over [0, 1]."""
    edges = np.linspace(0.0, min(1.0, float(np.nanmax(xs)) if xs.size else 1.0), nbins + 1)
    centers, means, sems = [], [], []
    for lo, hi in zip(edges[:-1], edges[1:]):
        mask = (xs >= lo) & (xs < hi) if hi < edges[-1] else (xs >= lo) & (xs <= hi)
        vals = ys[mask]
        vals = vals[np.isfinite(vals)]
        if vals.size == 0:
            continue
        centers.append(0.5 * (lo + hi))
        means.append(float(vals.mean()))
        sems.append(float(vals.std(ddof=1) / math.sqrt(vals.size)) if vals.size > 1 else 0.0)
    return np.asarray(centers), np.asarray(means), np.asarray(sems)


def main() -> None:
    args = parse_args()
    rows = read_rows(args.per_class_csv)
    if args.epoch is not None:
        rows = [r for r in rows if int(_f(r.get("epoch", "nan"))) == args.epoch]
    if not rows:
        raise SystemExit("No rows to plot (check paths / --epoch).")

    mass = np.asarray([_f(r.get("support_mass_share")) for r in rows])
    local_best = np.asarray([_f(r.get("best_local_acc")) for r in rows])
    global_post = np.asarray([_f(r.get("global_post_agg_acc")) for r in rows])

    fig, (ax_scatter, ax_gap) = plt.subplots(1, 2, figsize=(12.5, 5.0))

    # Left: raw scatter + binned trends.
    ax_scatter.scatter(mass, local_best, s=30, alpha=0.5, color="#4C78A8", label="Pre-agg local (best client)")
    ax_scatter.scatter(mass, global_post, s=30, alpha=0.5, color="#E45756", label="Post-agg global (FedAvg)")
    for ys, color in ((local_best, "#4C78A8"), (global_post, "#E45756")):
        cx, cy, cs = binned_trend(mass, ys, args.nbins)
        if cx.size:
            ax_scatter.plot(cx, cy, color=color, linewidth=2.2)
            ax_scatter.fill_between(cx, cy - cs, cy + cs, color=color, alpha=0.15)
    ax_scatter.set_xlabel("Tail-class support-client mass share  $n_S / N$")
    ax_scatter.set_ylabel("Per-class accuracy (%)")
    ax_scatter.set_title("Local learns it; aggregation keeps it only at high mass share")
    ax_scatter.legend(frameon=False)
    ax_scatter.grid(alpha=0.2)

    # Right: the crush gap vs mass share.
    gap = local_best - global_post
    ax_gap.scatter(mass, gap, s=30, alpha=0.55, color="#54A24B")
    cx, cy, cs = binned_trend(mass, gap, args.nbins)
    if cx.size:
        ax_gap.plot(cx, cy, color="#2F6B27", linewidth=2.4, label="Mean crush gap")
        ax_gap.fill_between(cx, cy - cs, cy + cs, color="#54A24B", alpha=0.18)
    ax_gap.axhline(0.0, color="black", linewidth=0.8, alpha=0.6)
    ax_gap.set_xlabel("Tail-class support-client mass share  $n_S / N$")
    ax_gap.set_ylabel("Crush gap: local best - global post (%)")
    ax_gap.set_title("The crush grows as tail knowledge is locked on low-mass clients")
    ax_gap.legend(frameon=False)
    ax_gap.grid(alpha=0.2)

    fig.suptitle(args.title, y=1.0)
    fig.tight_layout()
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=240, bbox_inches="tight")
    fig.savefig(out.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)

    def _corr(x: np.ndarray, y: np.ndarray) -> float:
        m = np.isfinite(x) & np.isfinite(y)
        if m.sum() < 2 or np.nanstd(x[m]) == 0 or np.nanstd(y[m]) == 0:
            return math.nan
        return float(np.corrcoef(x[m], y[m])[0, 1])

    # Pooled summary (all partitions/rounds together).
    print(f"[POOLED] rows={len(rows)}  mean_crush_gap={np.nanmean(gap):.2f}%  "
          f"corr(mass_share, crush_gap)={_corr(mass, gap):.3f}  (expect < 0)")

    # The decisive breakdown: per partition, and per (partition, epoch).
    # The paper's claim is a *between-partition contrast*, not a pooled corr:
    # client-longtail should crush the tail harder than Dirichlet at matched
    # global class frequency. Pooling the two hides exactly that contrast.
    partitions = sorted({str(r.get("partition", "")) for r in rows})
    print("\n[BY PARTITION]  (this is the contrast that matters)")
    for part in partitions:
        idx = np.asarray([str(r.get("partition", "")) == part for r in rows])
        if not idx.any():
            continue
        print(f"  partition={part or '(blank)'}  n={int(idx.sum())}  "
              f"mean_local={np.nanmean(local_best[idx]):.1f}%  "
              f"mean_global={np.nanmean(global_post[idx]):.1f}%  "
              f"mean_crush_gap={np.nanmean(gap[idx]):.2f}%  "
              f"corr(mass,gap)={_corr(mass[idx], gap[idx]):.3f}")

    epochs = sorted({int(_f(r.get("epoch", "nan"))) for r in rows if math.isfinite(_f(r.get("epoch", "nan")))})
    if len(partitions) > 1 and len(epochs) > 1:
        print("\n[BY PARTITION x EPOCH]  (does the crush deepen over rounds?)")
        ep_arr = np.asarray([_f(r.get("epoch", "nan")) for r in rows])
        for part in partitions:
            for ep in epochs:
                idx = np.asarray([str(r.get("partition", "")) == part for r in rows]) & (ep_arr == ep)
                if not idx.any():
                    continue
                print(f"  {part or '(blank)':<24} epoch={ep:<3} "
                      f"mean_crush_gap={np.nanmean(gap[idx]):.2f}%  n={int(idx.sum())}")

    print(f"\nsaved: {out}  and  {out.with_suffix('.pdf')}")


if __name__ == "__main__":
    main()
