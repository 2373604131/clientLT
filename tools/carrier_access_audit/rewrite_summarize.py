"""Summarize the frozen D1 post-write matrix and D2 cumulative replay."""

from __future__ import annotations

import argparse
import csv
import json
from collections import Counter
from pathlib import Path

import numpy as np

from tools.carrier_access_audit.rewrite_protocol import frozen_rewrite_protocol
from tools.carrier_access_audit.statistics import spearman, summarize
from tools.semantic_acquisition.common import file_sha256, write_csv, write_json


def _read(path: Path) -> list[dict[str, str]]:
    with Path(path).open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _truth(value: str | bool) -> bool:
    return str(value).lower() in {"true", "1"}


def _verify_result_hashes(runtime: dict, input_dir: Path) -> None:
    for name, expected in runtime.get("result_hashes", {}).items():
        if file_sha256(input_dir / name) != expected:
            raise RuntimeError(f"Runtime result hash mismatch: {name}")


def _rate(numerator: int, denominator: int) -> float:
    return float(numerator / denominator) if denominator else 0.0


def _sign(value: float) -> str:
    if value > 0:
        return "donor"
    if value < 0:
        return "rewriter"
    return "neutral"


def summarize_d1(input_dir: Path, output_dir: Path) -> dict:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    protocol = frozen_rewrite_protocol()
    runtime = json.loads((input_dir / "runtime_contract.json").read_text(encoding="utf-8"))
    if runtime.get("stage") != "D1" or runtime.get("protocol") != protocol:
        raise RuntimeError("Invalid D1 runtime contract")
    _verify_result_hashes(runtime, input_dir)

    writer_fairness = _read(input_dir / "tail_writer_fairness.csv")
    norm_fairness = _read(input_dir / "candidate_norm_fairness.csv")
    if len(writer_fairness) != 20 or any(not _truth(row["pass"]) for row in writer_fairness):
        raise RuntimeError("D1 tail-writer fairness failed")
    if len(norm_fairness) != 80 or any(not _truth(row["pass"]) for row in norm_fairness):
        raise RuntimeError("D1 candidate normalization fairness failed")

    writer_rows = _read(input_dir / "tail_writer_metrics.csv")
    pre_rows = _read(input_dir / "matched_pre_effects.csv")
    post_rows = _read(input_dir / "post_write_effects.csv")
    if len(writer_rows) != 20 or len(pre_rows) != 1600 or len(post_rows) != 1600:
        raise RuntimeError("D1 requires 20 writers and complete 80-by-20 pre/post matrices")
    writers = {int(row["tail_class"]): row for row in writer_rows}
    pre = {(int(row["tail_class"]), int(row["candidate_class"])): row for row in pre_rows}
    post = {(int(row["tail_class"]), int(row["candidate_class"])): row for row in post_rows}
    if set(pre) != set(post):
        raise RuntimeError("D1 matched pre and post matrices have different units")

    minimum_writers = int(protocol["tail_write"]["minimum_valid_tail_classes"])
    valid_tails = sorted(
        tail_class for tail_class, row in writers.items()
        if float(row["direct_test_margin_gain"]) > 0
    )
    valid_comparison = len(valid_tails) >= minimum_writers
    transition_names = (
        "donor_to_donor", "donor_to_rewriter",
        "rewriter_to_donor", "rewriter_to_rewriter",
    )
    per_tail = []
    transition_rows = []
    for tail_class in valid_tails:
        keys = sorted(key for key in post if key[0] == tail_class)
        pre_effects = np.asarray([float(pre[key]["test_pre_margin_gain"]) for key in keys])
        post_effects = np.asarray([float(post[key]["test_post_margin_gain"]) for key in keys])
        private_effects = np.asarray([float(post[key]["private_post_margin_gain"]) for key in keys])
        transitions = Counter(
            f"{_sign(float(pre[key]['test_pre_margin_gain']))}_to_"
            f"{_sign(float(post[key]['test_post_margin_gain']))}"
            for key in keys
        )
        post_donors = int((post_effects > 0).sum())
        post_rewriters = int((post_effects < 0).sum())
        private_positive = private_effects > 0
        private_negative = private_effects < 0
        test_positive = post_effects > 0
        test_negative = post_effects < 0
        row = {
            "data_seed": 42,
            "tail_class": tail_class,
            "direct_test_margin_gain": float(writers[tail_class]["direct_test_margin_gain"]),
            "candidate_count": len(keys),
            "pre_donor_count": int((pre_effects > 0).sum()),
            "pre_rewriter_count": int((pre_effects < 0).sum()),
            "post_donor_count": post_donors,
            "post_rewriter_count": post_rewriters,
            "has_both_post_signs": bool(post_donors > 0 and post_rewriters > 0),
            "mean_pre_test_margin_effect": float(pre_effects.mean()),
            "mean_post_test_margin_effect": float(post_effects.mean()),
            "mean_test_margin_turnover": float((post_effects - pre_effects).mean()),
            "private_test_post_spearman": spearman(private_effects, post_effects),
            "private_test_sign_agreement": float(np.mean(
                np.sign(private_effects) == np.sign(post_effects)
            )),
            "private_donor_precision": _rate(int((private_positive & test_positive).sum()), int(private_positive.sum())),
            "private_rewriter_recall": _rate(int((private_negative & test_negative).sum()), int(test_negative.sum())),
            "private_false_safe_rate": _rate(int((private_positive & test_negative).sum()), int(private_positive.sum())),
        }
        for transition in transition_names:
            row[f"{transition}_count"] = int(transitions[transition])
            transition_rows.append({
                "data_seed": 42, "tail_class": tail_class,
                "transition": transition, "count": int(transitions[transition]),
                "fraction": float(transitions[transition] / len(keys)),
            })
        per_tail.append(row)

    rules = protocol["d1"]["support_rules"]
    signed_count = sum(row["has_both_post_signs"] for row in per_tail)
    turnover_count = sum(row["donor_to_rewriter_count"] > 0 for row in per_tail)
    positive_spearman_count = sum(row["private_test_post_spearman"] > 0 for row in per_tail)
    mean_sign_agreement = float(np.mean([
        row["private_test_sign_agreement"] for row in per_tail
    ])) if per_tail else 0.0
    gates = {
        "enough_valid_tail_writers": valid_comparison,
        "signed_post_effects_are_widespread": signed_count >= int(
            rules["minimum_tail_classes_with_both_post_donors_and_rewriters"]
        ),
        "donor_to_rewriter_turnover_is_widespread": turnover_count >= int(
            rules["minimum_tail_classes_with_donor_to_rewriter_transition"]
        ),
        "private_effect_has_positive_rank_signal": positive_spearman_count >= int(
            rules["minimum_tail_classes_with_positive_private_test_spearman"]
        ),
        "private_sign_prediction_above_chance": mean_sign_agreement > float(
            rules["mean_private_test_sign_agreement_above"]
        ),
    }
    if not valid_comparison:
        verdict = "INVALID_TAIL_WRITE"
    elif all(gates.values()):
        verdict = "POST_WRITE_TURNOVER_AND_PRIVATE_DETECTION_SUPPORTED"
    elif gates["signed_post_effects_are_widespread"] and gates["private_effect_has_positive_rank_signal"]:
        verdict = "POST_WRITE_REWRITE_SUPPORTED_WITHOUT_FULL_TURNOVER_CHAIN"
    else:
        verdict = "NO_STABLE_POST_WRITE_REWRITE_SUPPORT"

    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "d1_per_tail_class.csv", per_tail)
    write_csv(output_dir / "d1_transition_matrix.csv", transition_rows)
    summary = {
        "stage": "D1",
        "verdict": verdict,
        "valid_comparison": valid_comparison,
        "valid_tail_classes": valid_tails,
        "valid_tail_writer_count": len(valid_tails),
        "gate_pass": bool(all(gates.values())),
        "gate_checks": gates,
        "tail_class_counts": {
            "both_post_donors_and_rewriters": int(signed_count),
            "at_least_one_donor_to_rewriter": int(turnover_count),
            "positive_private_test_spearman": int(positive_spearman_count),
        },
        "aggregate": {
            "direct_test_margin_gain": summarize(row["direct_test_margin_gain"] for row in per_tail),
            "post_donor_count": summarize(row["post_donor_count"] for row in per_tail),
            "post_rewriter_count": summarize(row["post_rewriter_count"] for row in per_tail),
            "test_margin_turnover": summarize(row["mean_test_margin_turnover"] for row in per_tail),
            "private_test_post_spearman": summarize(row["private_test_post_spearman"] for row in per_tail),
            "private_test_sign_agreement": summarize(row["private_test_sign_agreement"] for row in per_tail),
            "private_donor_precision": summarize(row["private_donor_precision"] for row in per_tail),
            "private_rewriter_recall": summarize(row["private_rewriter_recall"] for row in per_tail),
            "private_false_safe_rate": summarize(row["private_false_safe_rate"] for row in per_tail),
        },
        "test_metrics_used_for_candidate_selection": False,
        "evidence_boundary": (
            "D1 uses fixed, norm-equalized class-absent updates. It supports state-conditioned signed "
            "functional rewriting under this frozen vision-LoRA substrate, not a universal causal claim."
        ),
    }
    write_json(output_dir / "d1_summary.json", summary)
    return summary


def summarize_d2(input_dir: Path, d1_summary_path: Path, output_dir: Path) -> dict:
    input_dir, output_dir = Path(input_dir), Path(output_dir)
    protocol = frozen_rewrite_protocol()
    runtime = json.loads((input_dir / "runtime_contract.json").read_text(encoding="utf-8"))
    if runtime.get("stage") != "D2" or runtime.get("protocol") != protocol:
        raise RuntimeError("Invalid D2 runtime contract")
    _verify_result_hashes(runtime, input_dir)
    if file_sha256(d1_summary_path) != runtime.get("d1_summary_hash"):
        raise RuntimeError("D2 was not run against this D1 summary")
    d1_summary = json.loads(Path(d1_summary_path).read_text(encoding="utf-8"))
    if not d1_summary.get("valid_comparison", False):
        raise RuntimeError("D2 summary requires a valid D1 tail-write comparison")
    fairness = _read(input_dir / "runtime_fairness.csv")
    if len(fairness) != len(runtime["valid_tail_classes"]) or any(not _truth(row["pass"]) for row in fairness):
        raise RuntimeError("D2 runtime fairness failed")
    rows = _read(input_dir / "replay_metrics.csv")
    expected_rows = len(runtime["valid_tail_classes"]) * 3 * (2 + int(protocol["d2"]["blind_draws"]))
    if len(rows) != expected_rows:
        raise RuntimeError(f"D2 expected {expected_rows} replay rows, observed {len(rows)}")

    typed = []
    for row in rows:
        typed.append({
            **row,
            "tail_class": int(row["tail_class"]),
            "sequence_length": int(row["sequence_length"]),
            "draw": int(row["draw"]),
            "predicted_private_rewrite_risk": float(row["predicted_private_rewrite_risk"]),
            "test_forgetting": float(row["test_forgetting"]),
            "test_retention": float(row["test_retention"]),
        })
    lengths = [int(value) for value in protocol["d2"]["sequence_lengths"]]
    per_tail = []
    paired = []
    for tail_class in runtime["valid_tail_classes"]:
        values = [row for row in typed if row["tail_class"] == int(tail_class)]
        blind = [row for row in values if row["condition"] == "blind"]
        within_length_correlations = {
            length: spearman(
                [value["predicted_private_rewrite_risk"] for value in blind if value["sequence_length"] == length],
                [value["test_forgetting"] for value in blind if value["sequence_length"] == length],
            )
            for length in lengths
        }
        row = {
            "data_seed": 42,
            "tail_class": int(tail_class),
            "blind_risk_forgetting_spearman": float(np.mean(list(within_length_correlations.values()))),
        }
        for length, correlation in within_length_correlations.items():
            row[f"blind_risk_forgetting_spearman_k{length}"] = correlation
        low_advantages, high_harms, blind_means, high_values = [], [], [], []
        for length in lengths:
            low = next(value for value in values if value["condition"] == "low_risk" and value["sequence_length"] == length)
            high = next(value for value in values if value["condition"] == "high_risk" and value["sequence_length"] == length)
            blind_k = [value for value in blind if value["sequence_length"] == length]
            blind_forgetting = float(np.mean([value["test_forgetting"] for value in blind_k]))
            blind_retention = float(np.mean([value["test_retention"] for value in blind_k]))
            low_advantage = blind_forgetting - low["test_forgetting"]
            high_harm = high["test_forgetting"] - blind_forgetting
            low_advantages.append(low_advantage)
            high_harms.append(high_harm)
            blind_means.append(blind_forgetting)
            high_values.append(high["test_forgetting"])
            paired.append({
                "data_seed": 42, "tail_class": int(tail_class), "sequence_length": length,
                "low_risk_forgetting": low["test_forgetting"],
                "blind_mean_forgetting": blind_forgetting,
                "high_risk_forgetting": high["test_forgetting"],
                "blind_minus_low_forgetting": low_advantage,
                "high_minus_blind_forgetting": high_harm,
                "low_risk_retention": low["test_retention"],
                "blind_mean_retention": blind_retention,
                "high_risk_retention": high["test_retention"],
            })
        row.update({
            "mean_blind_minus_low_forgetting": float(np.mean(low_advantages)),
            "mean_high_minus_blind_forgetting": float(np.mean(high_harms)),
            "blind_length_forgetting_spearman": spearman(lengths, blind_means),
            "high_length_forgetting_spearman": spearman(lengths, high_values),
        })
        per_tail.append(row)

    positive_risk_count = sum(row["blind_risk_forgetting_spearman"] > 0 for row in per_tail)
    low_better_count = sum(row["mean_blind_minus_low_forgetting"] > 0 for row in per_tail)
    high_worse_count = sum(row["mean_high_minus_blind_forgetting"] > 0 for row in per_tail)
    rules = protocol["d2"]["support_rules"]
    gates = {
        "private_risk_predicts_blind_forgetting": positive_risk_count >= int(
            rules["minimum_tail_classes_positive_risk_forgetting_spearman"]
        ),
        "low_risk_replay_beats_blind": low_better_count >= int(
            rules["minimum_tail_classes_low_risk_better_than_blind"]
        ),
        "high_risk_replay_is_worse_than_blind": high_worse_count >= int(
            rules["minimum_tail_classes_high_risk_worse_than_blind"]
        ),
    }
    verdict = "REWRITE_RISK_PREDICTS_RETENTION" if all(gates.values()) else "NO_STABLE_RISK_RETENTION_CHAIN"
    output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(output_dir / "d2_per_tail_class.csv", per_tail)
    write_csv(output_dir / "d2_paired_conditions.csv", paired)
    summary = {
        "stage": "D2",
        "verdict": verdict,
        "gate_pass": bool(all(gates.values())),
        "gate_checks": gates,
        "tail_class_counts": {
            "positive_blind_risk_forgetting_spearman": int(positive_risk_count),
            "low_risk_better_than_blind": int(low_better_count),
            "high_risk_worse_than_blind": int(high_worse_count),
        },
        "aggregate": {
            "blind_risk_forgetting_spearman": summarize(row["blind_risk_forgetting_spearman"] for row in per_tail),
            "blind_minus_low_forgetting": summarize(row["mean_blind_minus_low_forgetting"] for row in per_tail),
            "high_minus_blind_forgetting": summarize(row["mean_high_minus_blind_forgetting"] for row in per_tail),
            "blind_length_forgetting_spearman": summarize(row["blind_length_forgetting_spearman"] for row in per_tail),
            "high_length_forgetting_spearman": summarize(row["high_length_forgetting_spearman"] for row in per_tail),
        },
        "risk_correlation_primary_subset": "blind_sequences_only_within_each_fixed_K_then_mean",
        "test_metrics_used_for_sequence_selection": False,
        "evidence_boundary": (
            "D2 replays fixed saved, norm-equalized updates. It tests whether private post-write risk "
            "orders later test forgetting; it is not a full federated trajectory with client retraining."
        ),
    }
    write_json(output_dir / "d2_summary.json", summary)
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", choices=["d1", "d2"], required=True)
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--d1-summary", type=Path, default=Path("output/post_write_rewrite_audit/analysis_d1/d1_summary.json"))
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    result = (
        summarize_d1(args.input_dir, args.output_dir)
        if args.stage == "d1"
        else summarize_d2(args.input_dir, args.d1_summary, args.output_dir)
    )
    print(json.dumps({"stage": args.stage.upper(), "verdict": result["verdict"]}))


if __name__ == "__main__":
    main()
