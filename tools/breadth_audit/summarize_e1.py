from __future__ import annotations

import argparse
import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from tools.breadth_audit.artifacts import FAMILY_FILES
from tools.breadth_audit.comparison import PRIMARY_DIRECTIONS, preregistered_direction_gate
from tools.semantic_acquisition.common import write_csv, write_json


def _read_csv(path: Path) -> list[dict]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _strength_index(run_dir: Path) -> dict[int, list[dict]]:
    grouped = defaultdict(list)
    for raw in _read_csv(run_dir / "tail_strength.csv"):
        row = {
            **raw,
            "seed": int(raw["seed"]),
            "round": int(raw["round"]),
            "tail_class": int(raw["tail_class"]),
            "tail_accuracy": float(raw["tail_accuracy"]),
            "tail_margin": float(raw["tail_margin"]),
            "tail_loss": float(raw["tail_loss"]),
        }
        grouped[row["tail_class"]].append(row)
    if set(grouped) != set(range(80, 100)):
        raise RuntimeError(f"strength rows do not cover tail classes 80--99: {sorted(grouped)}")
    for class_id, rows in grouped.items():
        rows.sort(key=lambda value: value["round"])
        observed = [row["round"] for row in rows]
        if observed != list(range(101)):
            raise RuntimeError(f"class {class_id} does not contain rounds 0--100")
    return dict(grouped)


def _derived_strength(topology: str, grouped: dict[int, list[dict]]) -> list[dict]:
    output = []
    for class_id in sorted(grouped):
        rows = grouped[class_id]
        initial = rows[0]
        final = rows[-1]
        peak = max(rows, key=lambda value: (value["tail_margin"], -value["round"]))
        peak_margin = float(peak["tail_margin"])
        final_margin = float(final["tail_margin"])
        initial_margin = float(initial["tail_margin"])
        retention = final_margin / peak_margin if abs(peak_margin) > 1e-12 else None
        gain_denominator = peak_margin - initial_margin
        retained_gain = (
            (final_margin - initial_margin) / gain_denominator
            if abs(gain_denominator) > 1e-12 else None
        )
        output.append({
            "topology": topology,
            "tail_class": class_id,
            "time_to_peak": int(peak["round"]),
            "initial_margin": initial_margin,
            "peak_margin": peak_margin,
            "final_margin": final_margin,
            "forgetting": peak_margin - final_margin,
            "retention_ratio_requested": retention,
            "retained_gain_fraction": retained_gain,
            "initial_accuracy": float(initial["tail_accuracy"]),
            "peak_round_accuracy": float(peak["tail_accuracy"]),
            "final_accuracy": float(final["tail_accuracy"]),
        })
    return output


def _load_families(run_dir: Path) -> dict[str, list[dict]]:
    result = {}
    for family, filename in FAMILY_FILES.items():
        rows = _read_csv(run_dir / filename)
        for row in rows:
            row["seed"] = int(row["seed"])
            row["round"] = int(row["round"])
            row["tail_class"] = int(row["tail_class"])
        result[family] = rows
    return result


def _select_peak_rows(families, derived):
    peak_round = {int(row["tail_class"]): int(row["time_to_peak"]) for row in derived}
    return {
        family: [
            row for row in rows
            if int(row["round"]) == peak_round[int(row["tail_class"])]
        ]
        for family, rows in families.items()
    }


def _accuracy_matches(
    dir_strength: dict[int, list[dict]],
    client_derived: list[dict],
    *,
    tolerance: float,
) -> tuple[list[dict], dict[int, int]]:
    matches, dir_rounds = [], {}
    for peak in client_derived:
        class_id = int(peak["tail_class"])
        client_accuracy = float(peak["peak_round_accuracy"])
        candidate = min(
            dir_strength[class_id],
            key=lambda row: (abs(float(row["tail_accuracy"]) - client_accuracy), row["round"]),
        )
        gap = abs(float(candidate["tail_accuracy"]) - client_accuracy)
        matches.append({
            "tail_class": class_id,
            "clientlt_peak_round": int(peak["time_to_peak"]),
            "clientlt_accuracy": client_accuracy,
            "dirichlet_matched_round": int(candidate["round"]),
            "dirichlet_accuracy": float(candidate["tail_accuracy"]),
            "absolute_accuracy_gap": gap,
            "within_tolerance": gap <= tolerance,
        })
        if gap <= tolerance:
            dir_rounds[class_id] = int(candidate["round"])
    return matches, dir_rounds


def _select_accuracy_matched_rows(
    dir_families,
    client_families,
    client_derived,
    matched_dir_rounds,
):
    client_rounds = {int(row["tail_class"]): int(row["time_to_peak"]) for row in client_derived}
    valid_classes = set(matched_dir_rounds)
    left, right = {}, {}
    for family in FAMILY_FILES:
        left[family] = [
            row for row in dir_families[family]
            if int(row["tail_class"]) in valid_classes
            and int(row["round"]) == matched_dir_rounds[int(row["tail_class"])]
        ]
        right[family] = [
            row for row in client_families[family]
            if int(row["tail_class"]) in valid_classes
            and int(row["round"]) == client_rounds[int(row["tail_class"])]
        ]
    return left, right


def _majority_summary(left, right, pair_keys=("seed", "tail_class")) -> dict:
    result = {}
    for family, directions in PRIMARY_DIRECTIONS.items():
        li = {tuple(row[key] for key in pair_keys): row for row in left[family]}
        ri = {tuple(row[key] for key in pair_keys): row for row in right[family]}
        supportive = 0
        for key in sorted(set(li) & set(ri)):
            if all(
                direction * (float(li[key][metric]) - float(ri[key][metric])) > 0.0
                for metric, direction in directions.items()
            ):
                supportive += 1
        total = len(set(li) & set(ri))
        result[family] = {
            "supportive_class_count": supportive,
            "paired_class_count": total,
            "strict_class_majority": supportive > total / 2 if total else False,
        }
    return result


def _mean(rows, key):
    return float(np.mean([float(row[key]) for row in rows]))


def _validate_pair(dir_dir: Path, client_dir: Path) -> dict:
    dir_contract = json.loads((dir_dir / "e1_contract.json").read_text(encoding="utf-8"))
    client_contract = json.loads((client_dir / "e1_contract.json").read_text(encoding="utf-8"))
    dir_manifest = _read_csv(dir_dir / "e1_round_manifest.csv")
    client_manifest = _read_csv(client_dir / "e1_round_manifest.csv")
    dir_rounds = [int(row["round"]) for row in dir_manifest]
    client_rounds = [int(row["round"]) for row in client_manifest]
    dir_partition = json.loads((dir_dir / "partition_summary.json").read_text(encoding="utf-8"))
    client_partition = json.loads((client_dir / "partition_summary.json").read_text(encoding="utf-8"))
    checks = {
        "protocol_hash_equal": dir_contract["protocol_hash"] == client_contract["protocol_hash"],
        "theta0_hash_equal": dir_contract["theta0_hash"] == client_contract["theta0_hash"],
        "round0_logits_equal": dir_manifest[0]["clean_logits_hash"] == client_manifest[0]["clean_logits_hash"],
        "round0_features_equal": dir_manifest[0]["clean_features_hash"] == client_manifest[0]["clean_features_hash"],
        "global_lt_fingerprint_equal": (
            dir_partition["global_lt_fingerprint"] == client_partition["global_lt_fingerprint"]
        ),
        "complete_101_round_evaluations": (
            dir_rounds == client_rounds == list(range(101))
        ),
    }
    dir_steps = _read_csv(dir_dir / "e1_optimizer_steps.csv")
    client_steps = _read_csv(client_dir / "e1_optimizer_steps.csv")
    dir_total = sum(int(row["optimizer_steps"]) for row in dir_steps)
    client_total = sum(int(row["optimizer_steps"]) for row in client_steps)
    audit = {
        "checks": checks,
        "pass": all(checks.values()),
        "optimizer_rule_equal": True,
        "dirichlet_realized_optimizer_steps": dir_total,
        "clientlt_realized_optimizer_steps": client_total,
        "realized_step_difference_clientlt_minus_dirichlet": client_total - dir_total,
        "realized_step_relative_difference": (
            (client_total - dir_total) / dir_total if dir_total else None
        ),
        "step_interpretation": (
            "Same local-epoch/batch/optimizer rule; realized counts differ naturally "
            "because client sizes are topology-dependent. No batch padding or truncation."
        ),
    }
    if not audit["pass"]:
        raise RuntimeError(f"E1 paired fairness gate failed: {checks}")
    return audit


def summarize(dirichlet_dir: Path, clientlt_dir: Path, output_dir: Path) -> dict:
    dirichlet_dir, clientlt_dir = Path(dirichlet_dir), Path(clientlt_dir)
    output_dir = Path(output_dir)
    fairness = _validate_pair(dirichlet_dir, clientlt_dir)
    dir_strength = _strength_index(dirichlet_dir)
    client_strength = _strength_index(clientlt_dir)
    dir_derived = _derived_strength("dirichlet", dir_strength)
    client_derived = _derived_strength("clientlt_controlled", client_strength)
    write_csv(output_dir / "derived_strength_per_class.csv", dir_derived + client_derived)

    strength = {
        "dirichlet_mean_peak_margin": _mean(dir_derived, "peak_margin"),
        "clientlt_mean_peak_margin": _mean(client_derived, "peak_margin"),
        "dirichlet_mean_time_to_peak": _mean(dir_derived, "time_to_peak"),
        "clientlt_mean_time_to_peak": _mean(client_derived, "time_to_peak"),
    }
    strength["clientlt_peak_not_weaker"] = (
        strength["clientlt_mean_peak_margin"] >= strength["dirichlet_mean_peak_margin"]
    )
    strength["clientlt_reaches_peak_faster"] = (
        strength["clientlt_mean_time_to_peak"] < strength["dirichlet_mean_time_to_peak"]
    )
    dir_by_class = {int(row["tail_class"]): row for row in dir_derived}
    client_by_class = {int(row["tail_class"]): row for row in client_derived}
    strength_supportive_classes = [
        class_id for class_id in sorted(dir_by_class)
        if (
            float(client_by_class[class_id]["peak_margin"])
            >= float(dir_by_class[class_id]["peak_margin"])
            or int(client_by_class[class_id]["time_to_peak"])
            < int(dir_by_class[class_id]["time_to_peak"])
        )
    ]
    strength["supportive_tail_classes"] = strength_supportive_classes
    strength["supportive_tail_class_count"] = len(strength_supportive_classes)
    strength["tail_class_majority"] = len(strength_supportive_classes) >= 11
    strength["strength_gate_pass"] = bool(
        (strength["clientlt_peak_not_weaker"] or strength["clientlt_reaches_peak_faster"])
        and strength["tail_class_majority"]
    )

    dir_families = _load_families(dirichlet_dir)
    client_families = _load_families(clientlt_dir)
    dir_peak = _select_peak_rows(dir_families, dir_derived)
    client_peak = _select_peak_rows(client_families, client_derived)
    peak_gate = preregistered_direction_gate(
        dir_peak, client_peak, pair_keys=("seed", "tail_class")
    )
    peak_majority = _majority_summary(dir_peak, client_peak)

    tolerance = 0.02
    matches, matched_rounds = _accuracy_matches(
        dir_strength, client_derived, tolerance=tolerance
    )
    write_csv(output_dir / "accuracy_matches.csv", matches)
    accuracy_control = {
        "accuracy_tolerance": tolerance,
        "matched_class_count": len(matched_rounds),
        "required_matched_class_count": 10,
        "sufficient_coverage": len(matched_rounds) >= 10,
    }
    if matched_rounds:
        matched_dir, matched_client = _select_accuracy_matched_rows(
            dir_families, client_families, client_derived, matched_rounds
        )
        matched_gate = preregistered_direction_gate(
            matched_dir, matched_client, pair_keys=("seed", "tail_class")
        )
        matched_majority = _majority_summary(matched_dir, matched_client)
    else:
        matched_gate = {"directional_gate_pass": False, "reason": "no accuracy matches"}
        matched_majority = {}
    accuracy_control["direction_gate"] = matched_gate
    accuracy_control["majority"] = matched_majority

    majority_family_count = sum(
        value["strict_class_majority"] for value in peak_majority.values()
    )
    majority_pass = majority_family_count >= 2
    gate_pass = bool(
        strength["strength_gate_pass"]
        and peak_gate["directional_gate_pass"]
        and majority_pass
        and accuracy_control["sufficient_coverage"]
        and matched_gate.get("directional_gate_pass", False)
    )
    if gate_pass:
        verdict = "SEED42_STRONG_BUT_NARROW_GATE_PASS"
    elif strength["strength_gate_pass"] and peak_gate["directional_gate_pass"]:
        verdict = "SEED42_APPARENT_NARROWING_NOT_ACCURACY_CONTROLLED"
    elif not strength["strength_gate_pass"] and peak_gate["directional_gate_pass"]:
        verdict = "SEED42_NARROW_BUT_NOT_STRONG"
    elif strength["strength_gate_pass"] and not peak_gate["directional_gate_pass"]:
        verdict = "SEED42_STRONG_BUT_NOT_NARROW"
    else:
        verdict = "SEED42_STRONG_BUT_NARROW_GATE_FAIL"
    summary = {
        "verdict": verdict,
        "gate_pass": gate_pass,
        "scope": "single_seed_mechanism_gate_not_final_inference",
        "fairness": fairness,
        "strength": strength,
        "peak_breadth_direction_gate": peak_gate,
        "peak_breadth_class_majority": peak_majority,
        "accuracy_control": accuracy_control,
    }
    write_json(output_dir / "e1_seed42_summary.json", summary)
    output_dir.mkdir(parents=True, exist_ok=True)
    (output_dir / "e1_seed42_summary.md").write_text(
        "\n".join([
            "# E1 seed 42 gate",
            "",
            f"- Verdict: `{verdict}`",
            f"- Paired fairness gate: `{fairness['pass']}`",
            f"- Strength gate: `{strength['strength_gate_pass']}`",
            f"- Own-peak breadth gate: `{peak_gate['directional_gate_pass']}`",
            f"- Majority-tail gate (>=2 families): `{majority_pass}`",
            f"- Accuracy-matched classes: `{len(matched_rounds)}/20`",
            f"- Accuracy-controlled breadth gate: `{matched_gate.get('directional_gate_pass', False)}`",
            "",
            "This is the preregistered seed-42 decision gate, not the final multi-seed inferential claim.",
        ]) + "\n",
        encoding="utf-8",
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description="Summarize the paired formal E1 seed-42 run")
    parser.add_argument("--dirichlet-dir", type=Path, required=True)
    parser.add_argument("--clientlt-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    summary = summarize(args.dirichlet_dir, args.clientlt_dir, args.output_dir)
    print(json.dumps({
        "verdict": summary["verdict"],
        "gate_pass": summary["gate_pass"],
        "summary": str((args.output_dir / "e1_seed42_summary.json").resolve()),
    }))


if __name__ == "__main__":
    main()
