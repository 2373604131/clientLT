"""Summarize Experiments B and C without changing their preregistered endpoints."""

from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.carrier_access_audit.protocol import TAIL_CLASSES, frozen_protocol
from tools.carrier_access_audit.statistics import spearman, summarize
from tools.semantic_acquisition.common import write_csv, write_json


def _read(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def summarize_b(input_dir: Path, output_dir: Path) -> dict:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    runtime = json.loads((input_dir / "runtime_contract.json").read_text(encoding="utf-8"))
    if runtime.get("stage") != "B" or runtime.get("protocol") != frozen_protocol():
        raise RuntimeError("Invalid Experiment B runtime contract")
    fairness = _read(input_dir / "runtime_fairness.csv")
    if len(fairness) != 80 or any(str(row["pass"]).lower() not in {"true", "1"} for row in fairness):
        raise RuntimeError("Experiment B fairness failed")
    rows = _read(input_dir / "transfer_matrix.csv")
    if len(rows) != 1600:
        raise RuntimeError("Experiment B transfer matrix must contain 1600 pairs")

    by_tail = defaultdict(list)
    for row in rows:
        parsed = {
            "tail_class": int(row["tail_class"]),
            "candidate_class": int(row["candidate_class"]),
            "semantic_rank": int(row["semantic_rank"]),
            "cosine_similarity": float(row["cosine_similarity"]),
            "private_margin_gain": float(row["private_margin_gain"]),
            "test_margin_gain": float(row["test_margin_gain"]),
            "test_nll_gain": float(row["test_nll_gain"]),
            "test_worst_neighbor_margin_gain": float(row["test_worst_neighbor_margin_gain"]),
        }
        by_tail[parsed["tail_class"]].append(parsed)

    per_tail = []
    budgets = [1, 3, 5, 10, 20, 40, 80]
    for tail_class in TAIL_CLASSES:
        values = sorted(by_tail[tail_class], key=lambda row: row["semantic_rank"])
        if len(values) != 80 or [row["semantic_rank"] for row in values] != list(range(1, 81)):
            raise RuntimeError(f"Tail class {tail_class} lacks a full ranked candidate matrix")
        related, unrelated = values[:10], values[-10:]
        private_selected = max(related, key=lambda row: (row["private_margin_gain"], -row["semantic_rank"], -row["candidate_class"]))
        semantic_top1 = values[0]
        row = {
            "data_seed": 42,
            "tail_class": tail_class,
            "semantic_effect_spearman": spearman(
                [value["cosine_similarity"] for value in values],
                [value["test_margin_gain"] for value in values],
            ),
            "private_test_effect_spearman": spearman(
                [value["private_margin_gain"] for value in values],
                [value["test_margin_gain"] for value in values],
            ),
            "related_positive_donor_rate": float(np.mean([value["test_margin_gain"] > 0 for value in related])),
            "unrelated_positive_donor_rate": float(np.mean([value["test_margin_gain"] > 0 for value in unrelated])),
            "related_mean_test_margin_gain": float(np.mean([value["test_margin_gain"] for value in related])),
            "unrelated_mean_test_margin_gain": float(np.mean([value["test_margin_gain"] for value in unrelated])),
            "related_mean_test_worst_neighbor_gain": float(np.mean([value["test_worst_neighbor_margin_gain"] for value in related])),
            "unrelated_mean_test_worst_neighbor_gain": float(np.mean([value["test_worst_neighbor_margin_gain"] for value in unrelated])),
            "private_selected_candidate": private_selected["candidate_class"],
            "private_selected_semantic_rank": private_selected["semantic_rank"],
            "private_selected_test_margin_gain": private_selected["test_margin_gain"],
            "semantic_top1_test_margin_gain": semantic_top1["test_margin_gain"],
            "test_oracle_top10_margin_gain_audit_only": max(value["test_margin_gain"] for value in related),
        }
        for budget in budgets:
            row[f"best_test_margin_gain_within_semantic_top{budget}"] = max(
                value["test_margin_gain"] for value in values[:budget]
            )
        per_tail.append(row)

    def field_summary(field):
        return summarize(float(row[field]) for row in per_tail)

    comparisons = {
        "related_minus_unrelated_positive_donor_rate": summarize(
            row["related_positive_donor_rate"] - row["unrelated_positive_donor_rate"] for row in per_tail
        ),
        "related_minus_unrelated_mean_test_margin_gain": summarize(
            row["related_mean_test_margin_gain"] - row["unrelated_mean_test_margin_gain"] for row in per_tail
        ),
        "related_minus_unrelated_worst_neighbor_gain": summarize(
            row["related_mean_test_worst_neighbor_gain"] - row["unrelated_mean_test_worst_neighbor_gain"]
            for row in per_tail
        ),
        "semantic_effect_spearman": field_summary("semantic_effect_spearman"),
        "private_test_effect_spearman": field_summary("private_test_effect_spearman"),
        "private_selected_test_margin_gain": field_summary("private_selected_test_margin_gain"),
    }
    directional = [
        comparisons["related_minus_unrelated_positive_donor_rate"]["mean"] > 0
        and comparisons["related_minus_unrelated_positive_donor_rate"]["positive_count"] >= 12,
        comparisons["related_minus_unrelated_mean_test_margin_gain"]["mean"] > 0
        and comparisons["related_minus_unrelated_mean_test_margin_gain"]["positive_count"] >= 12,
        comparisons["semantic_effect_spearman"]["mean"] > 0
        and comparisons["semantic_effect_spearman"]["positive_count"] >= 12,
    ]
    verdict = (
        "SEMANTIC_PRIOR_ENRICHES_FUNCTIONAL_DONORS"
        if sum(directional) >= 2
        else "NO_SEMANTIC_DONOR_ENRICHMENT"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "experiment_b_per_tail_class.csv", per_tail)
    summary = {
        "experiment": "B",
        "verdict": verdict,
        "directional_checks_passed": int(sum(directional)),
        "directional_checks_total": 3,
        "comparisons": comparisons,
        "best_attainable_gain_by_budget": {
            str(budget): field_summary(f"best_test_margin_gain_within_semantic_top{budget}")
            for budget in budgets
        },
        "test_metrics_used_for_candidate_selection": False,
        "evidence_boundary": (
            "Candidate-only local updates use equal samples and steps from theta0. Semantic similarity is "
            "evaluated as a donor-enrichment prior; positive transfer remains class-conditional."
        ),
    }
    write_json(output_dir / "experiment_b_summary.json", summary)
    return summary


def summarize_c(input_dir: Path, output_dir: Path) -> dict:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    runtime = json.loads((input_dir / "runtime_contract.json").read_text(encoding="utf-8"))
    if runtime.get("stage") != "C" or runtime.get("protocol") != frozen_protocol():
        raise RuntimeError("Invalid Experiment C runtime contract")
    fairness = _read(input_dir / "runtime_fairness.csv")
    if len(fairness) != 20 or any(str(row["pass"]).lower() not in {"true", "1"} for row in fairness):
        raise RuntimeError("Experiment C fairness failed")
    rows = _read(input_dir / "placement_metrics.csv")
    by_tail = defaultdict(dict)
    for row in rows:
        by_tail[int(row["tail_class"])][row["condition"]] = row
    expected_conditions = set(frozen_protocol()["experiment_c"]["conditions"])
    contrasts = []
    definitions = {
        "joint_related_minus_separate_merge_related": ("joint_related", "separate_merge_related"),
        "separate_readapt_related_minus_separate_merge_related": ("separate_readapt_related", "separate_merge_related"),
        "joint_related_minus_joint_unrelated": ("joint_related", "joint_unrelated"),
    }
    for tail_class in TAIL_CLASSES:
        if set(by_tail[tail_class]) != expected_conditions:
            raise RuntimeError(f"Incomplete Experiment C conditions for tail {tail_class}")
        row = {"data_seed": 42, "tail_class": tail_class}
        for name, (left_name, right_name) in definitions.items():
            left, right = by_tail[tail_class][left_name], by_tail[tail_class][right_name]
            for metric in (
                "test_margin_gain", "test_nll_gain", "test_worst_neighbor_margin_gain",
                "test_accuracy_gain", "lora_update_l2",
            ):
                row[f"{name}__{metric}"] = float(left[metric]) - float(right[metric])
        row["joint_to_separate_update_norm_ratio"] = float(
            float(by_tail[tail_class]["joint_related"]["lora_update_l2"])
            / max(float(by_tail[tail_class]["separate_merge_related"]["lora_update_l2"]), 1e-12)
        )
        row["chosen_lambda"] = float(by_tail[tail_class]["separate_readapt_related"]["chosen_lambda"])
        contrasts.append(row)

    summaries = {}
    support = {}
    for name in definitions:
        summaries[name] = {
            metric: summarize(row[f"{name}__{metric}"] for row in contrasts)
            for metric in (
                "test_margin_gain", "test_nll_gain", "test_worst_neighbor_margin_gain", "test_accuracy_gain"
            )
        }
        primary = summaries[name]["test_margin_gain"]
        support[name] = bool(primary["mean"] > 0 and primary["positive_count"] >= 12)
    verdict = (
        "JOINT_AND_PRIVATE_READAPT_BOTH_SUPPORTED"
        if support["joint_related_minus_separate_merge_related"]
        and support["separate_readapt_related_minus_separate_merge_related"]
        else "PARTIAL_PLACEMENT_SUPPORT"
        if any(support.values())
        else "NO_PLACEMENT_OR_READAPT_SUPPORT"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "experiment_c_paired_contrasts.csv", contrasts)
    summary = {
        "experiment": "C",
        "verdict": verdict,
        "contrast_support": support,
        "contrasts": summaries,
        "update_norm_diagnostic": summarize(row["joint_to_separate_update_norm_ratio"] for row in contrasts),
        "chosen_lambda": summarize(row["chosen_lambda"] for row in contrasts),
        "selection_used_test_metrics": False,
        "evidence_boundary": (
            "Joint versus separate optimizer trajectory is the treatment. Samples, per-role gradient calls, "
            "theta0 and private selection evidence are controlled; raw update norms are reported as a diagnostic."
        ),
    }
    write_json(output_dir / "experiment_c_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["b", "c"], required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = summarize_b(args.input_dir, args.output_dir) if args.stage == "b" else summarize_c(args.input_dir, args.output_dir)
    print(json.dumps({"stage": args.stage.upper(), "verdict": result["verdict"]}))


if __name__ == "__main__":
    main()
