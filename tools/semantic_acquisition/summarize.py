from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from tools.semantic_acquisition.common import write_csv, write_json
from tools.semantic_acquisition.metrics import cluster_bootstrap


def _all_true(series: pd.Series) -> bool:
    return all(str(value).strip().lower() in ("true", "1") for value in series.tolist())


def _effect_summary(rows: list[dict], field: str) -> dict:
    frame = pd.DataFrame(rows)
    by_seed = {}
    for seed, group in frame.groupby("data_seed"):
        values = group[field].astype(float).to_numpy()
        by_seed[str(int(seed))] = {
            "mean": float(values.mean()), "median": float(np.median(values)),
            "positive_classes": int((values > 0).sum()), "negative_classes": int((values < 0).sum()),
            "class_count": int(len(values)),
        }
    clusters = {int(class_id): group[field].astype(float).tolist() for class_id, group in frame.groupby("tail_class")}
    return {"by_seed": by_seed, "combined": cluster_bootstrap(clusters, 10000, 20260811)}


def average_v3_draws(effects: list[dict]) -> list[dict]:
    """Collapse draws inside each seed/class before inferential resampling."""
    return pd.DataFrame(effects).groupby(["data_seed", "tail_class"], as_index=False).mean(numeric_only=True).to_dict("records")


def summarize_v2(input_dir: Path, mode: str) -> dict:
    metrics = pd.read_csv(input_dir / "v2_run_metrics.csv")
    fairness = pd.read_csv(input_dir / "fairness_invariants.csv")
    paired = []
    for (seed, class_id), group in metrics.groupby(["data_seed", "tail_class"]):
        related = group[group.condition == "related"]
        tail_only = group[group.condition == "tail_only_masked"]
        unrelated = group[group.condition.str.startswith("matched_unrelated")]
        expected_draws = 1 if mode == "smoke" else 3
        if len(related) != 1 or len(tail_only) != 1 or len(unrelated) != expected_draws:
            raise RuntimeError(f"Incomplete V2 paired unit {(seed, class_id)}")
        paired.append({
            "data_seed": int(seed), "tail_class": int(class_id),
            "delta_sem": float(related.g_margin.iloc[0] - unrelated.g_margin.mean()),
            "delta_pos": float(related.g_margin.iloc[0] - tail_only.g_margin.iloc[0]),
            "delta_sem_nll": float(related.g_nll.iloc[0] - unrelated.g_nll.mean()),
            "delta_pos_nll": float(related.g_nll.iloc[0] - tail_only.g_nll.iloc[0]),
            "g_related": float(related.g_margin.iloc[0]),
            "g_unrelated_mean": float(unrelated.g_margin.mean()),
            "g_tail_only": float(tail_only.g_margin.iloc[0]),
        })
    write_csv(input_dir / "v2_paired_effects.csv", paired)
    sem, pos = _effect_summary(paired, "delta_sem"), _effect_summary(paired, "delta_pos")
    valid = _all_true(fairness["pass"]) and _all_true(fairness["amp_overflow_equal"])
    if mode == "smoke":
        verdict = "IMPLEMENTATION_SMOKE_ONLY" if valid else "INVALID_COMPARISON"
    elif not valid:
        verdict = "INVALID_COMPARISON"
    else:
        seeds = sorted(sem["by_seed"])
        sem_stable = all(sem["by_seed"][seed]["mean"] > 0 and sem["by_seed"][seed]["positive_classes"] >= 12 for seed in seeds) and sem["combined"]["ci_low"] > 0
        pos_stable = all(pos["by_seed"][seed]["mean"] > 0 and pos["by_seed"][seed]["positive_classes"] >= 12 for seed in seeds) and pos["combined"]["ci_low"] > 0
        paired_frame = pd.DataFrame(paired)
        sem_nll_summary = _effect_summary(paired, "delta_sem_nll")
        pos_nll_summary = _effect_summary(paired, "delta_pos_nll")
        sem_nll_reverse = all(paired_frame.groupby("data_seed").delta_sem_nll.mean() < 0) and sem_nll_summary["combined"]["ci_high"] < 0
        pos_nll_reverse = all(paired_frame.groupby("data_seed").delta_pos_nll.mean() < 0) and pos_nll_summary["combined"]["ci_high"] < 0
        nll_reverse = sem_nll_reverse or pos_nll_reverse
        heterogeneous = (
            sem["combined"]["ci_low"] > 0 and pos["combined"]["ci_low"] > 0
            and any(
                sem["by_seed"][seed]["mean"] <= 0
                or sem["by_seed"][seed]["positive_classes"] < 12
                or pos["by_seed"][seed]["mean"] <= 0
                or pos["by_seed"][seed]["positive_classes"] < 12
                for seed in seeds
            )
        )
        if sem_stable and pos_stable and not nll_reverse:
            verdict = "POSITIVE_SEMANTIC_TRANSFER"
        elif sem_stable:
            verdict = "RELATIVE_COMPATIBILITY_ONLY"
        elif heterogeneous:
            verdict = "HETEROGENEOUS_FUNCTIONAL_TRANSFER"
        else:
            verdict = "NO_FUNCTIONAL_SUPPORT"
    summary = {
        "stage": "v2", "mode": mode, "verdict": verdict, "valid_comparison": valid,
        "delta_sem": sem, "delta_pos": pos, "paired_unit_count": len(paired),
        "bootstrap_draws": 10000, "bootstrap_seed": 20260811,
        "evidence_boundary": "isolated functional semantic-acquisition mechanism; not a full Client-LT accuracy decomposition",
    }
    write_json(input_dir / "v2_summary.json", summary)
    (input_dir / "v2_summary.md").write_text(
        f"# V2 summary\n\nVerdict: **{verdict}**\n\nPaired units: {len(paired)}. "
        "Inference averages unrelated draws within seed/class and bootstraps class IDs.\n",
        encoding="utf-8",
    )
    write_csv(input_dir / "v2_excluded_units.csv", [], ["data_seed", "tail_class", "reason"])
    return summary


def summarize_v3(input_dir: Path, mode: str, v2_summary_path: Path | None) -> dict:
    trajectory = pd.read_csv(input_dir / "v3_epoch_trajectory.csv")
    oracle = pd.read_csv(input_dir / "v3_linear_oracles.csv")
    fairness = pd.read_csv(input_dir / "fairness_invariants.csv")
    effects = []
    for (seed, class_id, draw), group in trajectory.groupby(["data_seed", "tail_class", "draw"]):
        def value(placement, epoch, role, metric="g_margin"):
            row = group[(group.placement == placement) & (group.epoch == epoch) & (group.state_role == role)]
            if len(row) != 1:
                raise RuntimeError(f"Missing V3 trajectory cell {(seed, class_id, draw, placement, epoch, role)}")
            return float(row[metric].iloc[0])
        row = {"data_seed": int(seed), "tail_class": int(class_id), "draw": int(draw)}
        for epoch in (1, 2, 3):
            row[f"delta_location_e{epoch}"] = value("R_colocated", epoch, "fedavg") - value("R_remote_U_colocated", epoch, "fedavg")
        row["delta_path_growth"] = row["delta_location_e3"] - row["delta_location_e1"]
        row["delta_support_local"] = value("R_colocated", 3, "support_local") - value("R_remote_U_colocated", 3, "support_local")
        row["remote_related_gain"] = value("R_remote_U_colocated", 3, "remote_local") - value("R_colocated", 3, "remote_local")
        row["g_colocated"] = value("R_colocated", 3, "fedavg")
        row["g_remote"] = value("R_remote_U_colocated", 3, "fedavg")
        row["delta_location_nll_e3"] = value("R_colocated", 3, "fedavg", "g_nll") - value("R_remote_U_colocated", 3, "fedavg", "g_nll")
        effects.append(row)
    write_csv(input_dir / "v3_paired_effects.csv", effects)
    averaged = average_v3_draws(effects)
    location = _effect_summary(averaged, "delta_location_e3")
    valid = _all_true(fairness["pass"]) and _all_true(oracle[oracle.oracle.isin(["raw-gradient", "plain-SGD-one-step"])]["pass"])
    main_step_bad = not _all_true(oracle[oracle.oracle == "main-optimizer-epoch1"]["pass"])
    if mode == "smoke":
        verdict = "IMPLEMENTATION_SMOKE_ONLY" if valid else "INVALID_COMPARISON"
    elif not valid:
        verdict = "INVALID_COMPARISON"
    elif main_step_bad:
        verdict = "OPTIMIZER_OR_NUMERIC_PLACEMENT_EFFECT_AT_STEP1"
    else:
        seed_stats = location["by_seed"]
        stable = location["combined"]["ci_low"] > 0 and all(value["mean"] > 0 and value["positive_classes"] >= 12 for value in seed_stats.values())
        heterogeneous = location["combined"]["ci_low"] > 0 and not stable
        coloc = _effect_summary(averaged, "g_colocated")
        remote = _effect_summary(averaged, "g_remote")
        coloc_positive = coloc["combined"]["ci_low"] > 0 and all(value["mean"] > 0 for value in coloc["by_seed"].values())
        remote_positive = remote["combined"]["ci_low"] > 0 and all(value["mean"] > 0 for value in remote["by_seed"].values())
        remote_nonpositive = remote["combined"]["ci_high"] <= 0 and all(value["mean"] <= 0 for value in remote["by_seed"].values())
        support = _effect_summary(averaged, "delta_support_local")
        support_consistent = all(value["mean"] > 0 for value in support["by_seed"].values())
        nll = _effect_summary(averaged, "delta_location_nll_e3")
        nll_reverse = nll["combined"]["ci_high"] < 0 and all(value["mean"] < 0 for value in nll["by_seed"].values())
        v2_gate = v2_summary_path is not None and json.loads(v2_summary_path.read_text(encoding="utf-8")).get("verdict") == "POSITIVE_SEMANTIC_TRANSFER"
        if stable and coloc_positive and remote_nonpositive and support_consistent and not nll_reverse and v2_gate:
            verdict = "LOCAL_COADAPTATION_NECESSARY"
        elif stable and coloc_positive and remote_positive and not nll_reverse and v2_gate:
            verdict = "LOCAL_COADAPTATION_ADVANTAGE"
        elif stable:
            verdict = "LOCAL_PLACEMENT_COMPATIBILITY_ONLY"
        elif heterogeneous:
            verdict = "HETEROGENEOUS_LOCATION_EFFECT"
        else:
            verdict = "NO_STABLE_LOCATION_ADVANTAGE"
    v2_verdict = None
    if v2_summary_path:
        v2_verdict = json.loads(v2_summary_path.read_text(encoding="utf-8")).get("verdict")
    summary = {
        "stage": "v3", "mode": mode, "verdict": verdict, "v2_verdict": v2_verdict,
        "valid_comparison": valid, "delta_location_e3": location,
        "paired_draw_count": len(effects), "paired_seed_class_count": len(averaged),
        "bootstrap_draws": 10000, "bootstrap_seed": 20260811,
        "evidence_boundary": "controlled one-round two-client equal-weight micro-federation only",
    }
    write_json(input_dir / "v3_summary.json", summary)
    (input_dir / "v3_summary.md").write_text(
        f"# V3 summary\n\nVerdict: **{verdict}**\n\nThis result is limited to the preregistered equal-weight micro-federation.\n",
        encoding="utf-8",
    )
    write_csv(input_dir / "v3_excluded_units.csv", [], ["data_seed", "tail_class", "draw", "reason"])
    return summary


def main(argv=None):
    parser = argparse.ArgumentParser(description="Summarize preregistered V2/V3 outputs")
    parser.add_argument("--stage", required=True, choices=["v2", "v3"])
    parser.add_argument("--mode", required=True, choices=["smoke", "full"])
    parser.add_argument("--input-dir", required=True, type=Path)
    parser.add_argument("--v2-summary", type=Path)
    args = parser.parse_args(argv)
    result = summarize_v2(args.input_dir, args.mode) if args.stage == "v2" else summarize_v3(args.input_dir, args.mode, args.v2_summary)
    print(json.dumps(result, ensure_ascii=False))


if __name__ == "__main__":
    main()
