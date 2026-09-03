#!/usr/bin/env python3
"""Audit and summarize the full-participation Client-LT diagnosis."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path


TOPOLOGY_DIRS = {
    "clientlt": "clientlt",
    "matched_dirichlet": "matched_dirichlet",
}
EXPECTED_CLIENTS = set(range(30))
EXPECTED_ROUNDS = set(range(1, 81))


def _read_csv(path: Path) -> list[dict[str, str]]:
    if not path.is_file():
        raise FileNotFoundError(path)
    with path.open("r", encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def _read_json(path: Path) -> dict:
    if not path.is_file():
        raise FileNotFoundError(path)
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(payload, ensure_ascii=False, sort_keys=True, indent=2, allow_nan=False) + "\n",
        encoding="utf-8",
    )


def _write_csv(path: Path, rows: list[dict]) -> None:
    if not rows:
        raise ValueError(f"Refusing to write empty CSV: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _count_matrix(run_dir: Path) -> list[list[int]]:
    rows = _read_csv(run_dir / "client_class_counts.csv")
    rows.sort(key=lambda row: int(row["client_id"]))
    if [int(row["client_id"]) for row in rows] != list(range(30)):
        raise RuntimeError(f"Expected client IDs 0..29 in {run_dir / 'client_class_counts.csv'}")
    return [
        [int(row[f"class_{class_id}"]) for class_id in range(100)]
        for row in rows
    ]


def _global_class_counts(matrix: list[list[int]]) -> list[int]:
    return [sum(row[class_id] for row in matrix) for class_id in range(100)]


def _client_total_samples(matrix: list[list[int]]) -> list[int]:
    return [sum(row) for row in matrix]


def _initial_hash(run_dir: Path) -> str:
    audit = _read_json(run_dir / "cliplora_initialization_audit.json")
    if not bool(audit.get("global_local_initialization_equal")):
        raise RuntimeError(f"Global/local initialization mismatch under {run_dir}")
    value = str(audit.get("initial_lora_sha256", "")).strip()
    if not value:
        raise RuntimeError(f"Missing initial LoRA hash under {run_dir}")
    return value


def _audit_full_participation(run_dir: Path) -> dict:
    grouped: dict[int, list[dict[str, str]]] = {}
    for row in _read_csv(run_dir / "lora_aggregation_weights.csv"):
        grouped.setdefault(int(row["communication_round"]), []).append(row)
    if set(grouped) != EXPECTED_ROUNDS:
        raise RuntimeError(
            f"Full-participation audit expected rounds 1..80 under {run_dir}; "
            f"observed={sorted(grouped)}"
        )

    maximum_weight_sum_error = 0.0
    for round_id, rows in sorted(grouped.items()):
        clients = [int(row["client_id"]) for row in rows]
        if len(clients) != 30 or len(set(clients)) != 30 or set(clients) != EXPECTED_CLIENTS:
            raise RuntimeError(
                f"Round {round_id} under {run_dir} is not exact 30/30 participation: {clients}"
            )
        weight_sum = sum(float(row["aggregation_weight"]) for row in rows)
        error = abs(weight_sum - 1.0)
        maximum_weight_sum_error = max(maximum_weight_sum_error, error)
        if not math.isclose(weight_sum, 1.0, rel_tol=1e-7, abs_tol=1e-7):
            raise RuntimeError(
                f"Round {round_id} under {run_dir} has FedAvg weight sum {weight_sum}"
            )
    return {
        "round_count": len(grouped),
        "clients_per_round": 30,
        "all_client_ids_present_every_round": True,
        "maximum_weight_sum_abs_error": maximum_weight_sum_error,
        "fedavg_weight_sum_one_every_round": True,
    }


def _tail_outcome(run_dir: Path) -> dict:
    trajectory = []
    for row in _read_csv(run_dir / "round_metrics.csv"):
        epoch = int(row["epoch"])
        if epoch < 0:
            continue
        trajectory.append((epoch + 1, float(row["bottom20_tail_acc"])))
    trajectory.sort()
    rounds = [round_id for round_id, _ in trajectory]
    if rounds != list(range(1, 81)):
        raise RuntimeError(
            f"Expected one tail evaluation for rounds 1..80 under {run_dir}; observed={rounds}"
        )
    final_tail = trajectory[-1][1]
    best_tail = max(value for _, value in trajectory)
    return {
        "final_tail_accuracy": final_tail,
        "best_to_final_drop": best_tail - final_tail,
    }


def _partial_outcomes(partial_root: Path) -> dict[str, dict]:
    summary = _read_json(partial_root / "analysis" / "validation_summary.json")
    raw = summary.get("raw_outcomes", {})
    result = {}
    for name in TOPOLOGY_DIRS:
        if name not in raw:
            raise RuntimeError(f"Partial baseline is missing {name}: {partial_root}")
        result[name] = {
            "final_tail_accuracy": float(raw[name]["final"]["tail"]),
            "best_to_final_drop": float(raw[name]["best_to_final_tail_drop"]),
        }
    return result


def _partial_initial_hash(partial_root: Path) -> str:
    protocol = _read_json(partial_root / "clientlt" / "functional_coverage" / "protocol.json")
    value = str(protocol.get("common_lora_anchor_sha256", "")).strip()
    if not value:
        raise RuntimeError(f"Partial baseline has no common LoRA hash: {partial_root}")
    return value


def _statuses(final_gap: float, drop_gap: float, threshold: float) -> tuple[str, str, str]:
    if final_gap > threshold:
        final_status = "FINAL_TAIL_GAP_REMAINS"
    elif abs(final_gap) <= threshold:
        final_status = "FINAL_TAIL_PRACTICALLY_EQUIVALENT"
    else:
        final_status = "FINAL_TAIL_GAP_REVERSED"

    if drop_gap > threshold:
        drop_status = "RETENTION_DEGRADATION_REMAINS"
    elif abs(drop_gap) <= threshold:
        drop_status = "BEST_TO_FINAL_DROP_PRACTICALLY_EQUIVALENT"
    else:
        drop_status = "BEST_TO_FINAL_DROP_REVERSED"

    if final_status == "FINAL_TAIL_GAP_REMAINS" and drop_status == "RETENTION_DEGRADATION_REMAINS":
        verdict = "FINAL_AND_RETENTION_GAPS_REMAIN"
    elif final_status == "FINAL_TAIL_GAP_REMAINS" and drop_status == "BEST_TO_FINAL_DROP_PRACTICALLY_EQUIVALENT":
        verdict = "FINAL_GAP_WITHOUT_EXTRA_COLLAPSE"
    elif final_status == "FINAL_TAIL_PRACTICALLY_EQUIVALENT" and drop_status == "BEST_TO_FINAL_DROP_PRACTICALLY_EQUIVALENT":
        verdict = "SPARSE_PARTICIPATION_CONFOUND_SUPPORTED"
    else:
        verdict = "MIXED_RESULT"
    return final_status, drop_status, verdict


def analyze(output_root: Path, partial_root: Path) -> dict:
    output_root = Path(output_root)
    partial_root = Path(partial_root)
    protocol = _read_json(output_root / "frozen_protocol.json")
    threshold = float(protocol["equivalence_threshold_pp"])
    runs = {name: output_root / directory for name, directory in TOPOLOGY_DIRS.items()}

    matrices = {name: _count_matrix(path) for name, path in runs.items()}
    global_counts = {name: _global_class_counts(matrix) for name, matrix in matrices.items()}
    client_totals = {name: _client_total_samples(matrix) for name, matrix in matrices.items()}
    if global_counts["clientlt"] != global_counts["matched_dirichlet"]:
        raise RuntimeError("Global class counts n_c differ between Client-LT and matched Dirichlet")
    if client_totals["clientlt"] != client_totals["matched_dirichlet"]:
        raise RuntimeError("Client total sample counts n_k differ between Client-LT and matched Dirichlet")

    initial_hashes = {name: _initial_hash(path) for name, path in runs.items()}
    if len(set(initial_hashes.values())) != 1:
        raise RuntimeError(f"Initial LoRA hashes differ across full-participation runs: {initial_hashes}")
    partial_initial_hash = _partial_initial_hash(partial_root)
    if initial_hashes["clientlt"] != partial_initial_hash:
        raise RuntimeError(
            "Full-participation initialization differs from the frozen frac=0.4 baseline: "
            f"full={initial_hashes['clientlt']} partial={partial_initial_hash}"
        )

    participation = {name: _audit_full_participation(path) for name, path in runs.items()}
    full = {name: _tail_outcome(path) for name, path in runs.items()}
    partial = _partial_outcomes(partial_root)

    def contrasts(outcomes: dict[str, dict]) -> dict:
        return {
            "final_tail_accuracy_gap_pp": (
                outcomes["matched_dirichlet"]["final_tail_accuracy"]
                - outcomes["clientlt"]["final_tail_accuracy"]
            ),
            "best_to_final_drop_gap_pp": (
                outcomes["clientlt"]["best_to_final_drop"]
                - outcomes["matched_dirichlet"]["best_to_final_drop"]
            ),
        }

    partial_contrasts = contrasts(partial)
    full_contrasts = contrasts(full)
    final_status, drop_status, verdict = _statuses(
        full_contrasts["final_tail_accuracy_gap_pp"],
        full_contrasts["best_to_final_drop_gap_pp"],
        threshold,
    )

    rows = []
    for frac, outcomes in ((0.4, partial), (1.0, full)):
        for name in ("clientlt", "matched_dirichlet"):
            rows.append(
                {
                    "frac": frac,
                    "topology": name,
                    "final_tail_accuracy": outcomes[name]["final_tail_accuracy"],
                    "best_to_final_drop": outcomes[name]["best_to_final_drop"],
                }
            )
    _write_csv(output_root / "analysis" / "main_results.csv", rows)

    summary = {
        "schema_version": "full_participation_diagnosis_analysis_v1",
        "verdict": verdict,
        "single_seed_descriptive": True,
        "equivalence_threshold_pp": threshold,
        "primary_results": {
            "frac0p4": partial_contrasts,
            "frac1p0": full_contrasts,
            "full_minus_partial_final_gap_pp": (
                full_contrasts["final_tail_accuracy_gap_pp"]
                - partial_contrasts["final_tail_accuracy_gap_pp"]
            ),
            "full_minus_partial_drop_gap_pp": (
                full_contrasts["best_to_final_drop_gap_pp"]
                - partial_contrasts["best_to_final_drop_gap_pp"]
            ),
        },
        "decision": {
            "final_tail_accuracy": final_status,
            "best_to_final_drop": drop_status,
            "binding_rule": (
                "Final-tail persistence and retention-collapse persistence are judged "
                f"separately at the preregistered {threshold:g} pp threshold."
            ),
        },
        "outcomes": {
            "frac0p4": partial,
            "frac1p0": full,
        },
        "required_audits": {
            "passed": True,
            "initial_model_hash_equal": True,
            "initial_model_hash_matches_frac0p4_baseline": True,
            "initial_lora_sha256": initial_hashes["clientlt"],
            "global_class_counts_equal": True,
            "client_total_samples_equal": True,
            "thirty_unique_clients_every_round": True,
            "fedavg_weight_sum_one_every_round": True,
            "participation_details": participation,
        },
    }
    _write_json(output_root / "analysis" / "summary.json", summary)

    full_final_gap = full_contrasts["final_tail_accuracy_gap_pp"]
    full_drop_gap = full_contrasts["best_to_final_drop_gap_pp"]
    report = "\n".join(
        [
            "# Full participation diagnosis",
            "",
            f"- Verdict: **{verdict}**",
            f"- Full-participation final tail gap (Dir - Client-LT): **{full_final_gap:+.3f} pp**",
            f"- Full-participation best-to-final drop gap (Client-LT - Dir): **{full_drop_gap:+.3f} pp**",
            f"- Final-tail decision: **{final_status}**",
            f"- Collapse decision: **{drop_status}**",
            f"- Preregistered practical-equivalence threshold: **±{threshold:.3f} pp**",
            "",
            "The final-tail and collapse conclusions are intentionally separate. No margin,",
            "retention-ratio, coverage, or correlation analysis is part of this experiment.",
            "",
        ]
    )
    report_path = output_root / "analysis" / "report.md"
    report_path.write_text(report, encoding="utf-8")
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--output-root",
        type=Path,
        default=Path("output/full_participation_diagnosis_seed42"),
    )
    parser.add_argument(
        "--partial-root",
        type=Path,
        default=Path("output/functional_coverage_validation_seed42"),
    )
    args = parser.parse_args()
    result = analyze(args.output_root, args.partial_root)
    print(json.dumps({"verdict": result["verdict"], **result["primary_results"]}, indent=2))


if __name__ == "__main__":
    main()
