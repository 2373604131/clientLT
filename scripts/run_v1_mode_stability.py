#!/usr/bin/env python
"""Run V1: stability of candidate aggregation-mode representations.

The runner consumes one or more V0 round dumps.  It uses no examples, labels,
validation set, test set, or model forward pass.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import torch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from utils.cusp_minimal import write_csv, write_json
from utils.v1_mode_stability import (
    V1_SCHEMA_VERSION,
    DisagreementSet,
    ModeSet,
    build_disagreement_set,
    compare_atom_modes,
    compare_client_modes,
    compare_degenerate_subspaces,
    degenerate_groups,
    joint_sketch_modes,
    layerwise_modes,
    stable_seed,
    svd_atom_modes,
    upload_set_from_payload,
    whole_client_modes,
)


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_source(dump_dir: Path):
    state_path = dump_dir / "round_state.pt"
    metadata_path = dump_dir / "metadata.json"
    if not state_path.is_file() or not metadata_path.is_file():
        raise FileNotFoundError(f"Incomplete V0 dump: {dump_dir}")
    payload = torch.load(state_path, map_location="cpu", weights_only=False)
    metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
    if bool(metadata.get("test_used_before_dump", False)):
        raise RuntimeError(f"V1 refuses a dump with prior test access: {dump_dir}")
    resolved = metadata.get("resolved_args", {})
    seed = int(resolved.get("seed", -1))
    round_id = int(metadata.get("communication_round", -1))
    partition = str(resolved.get("partition", "unknown"))
    source_id = f"{partition}_seed{seed}_round{round_id:03d}"
    uploads = upload_set_from_payload(payload, metadata, source_id)
    return uploads, metadata, {
        "source_id": source_id,
        "dump_dir": str(dump_dir),
        "round_state_sha256": sha256_file(state_path),
        "metadata_sha256": sha256_file(metadata_path),
        "seed": seed,
        "round": round_id,
        "partition": partition,
        "client_count": len(uploads.client_ids),
        "parameter_count": int(uploads.spec.numel),
        "layer_count": len(uploads.layers),
        "layer_names": ",".join(uploads.layers),
    }


def build_representations(
    data: DisagreementSet,
    *,
    rank: int,
    sketch_dim: int,
    sketch_seed: int,
) -> dict:
    atoms = svd_atom_modes(data, rank)
    return {
        "client_whole_update": whole_client_modes(data),
        "single_svd_atom": atoms,
        "near_degenerate_subspace": atoms,
        "layerwise_mode": layerwise_modes(data, rank),
        "cross_layer_joint_sketch": joint_sketch_modes(
            data, rank, sketch_dim=sketch_dim, seed=sketch_seed
        ),
    }


def _record(
    *,
    context: dict,
    method: str,
    layer: str,
    metrics: dict,
) -> dict:
    return {
        **context,
        "method": method,
        "layer": layer,
        "stability_score": metrics["stability_score"],
        "worst_match": metrics["worst_match"],
        "match_count": metrics["match_count"],
    }


def compare_representations(
    reference: dict,
    candidate: dict,
    *,
    context: dict,
    relative_gap: float,
) -> tuple[list[dict], list[dict]]:
    stability_rows, match_rows = [], []

    comparisons = (
        ("client_whole_update", compare_client_modes),
        ("single_svd_atom", compare_atom_modes),
        ("cross_layer_joint_sketch", compare_atom_modes),
    )
    for method, comparator in comparisons:
        metrics, matches = comparator(reference[method], candidate[method])
        stability_rows.append(_record(context=context, method=method, layer="__all__", metrics=metrics))
        match_rows.extend({**context, "method": method, "layer": "__all__", **row} for row in matches)

    metrics, matches = compare_degenerate_subspaces(
        reference["near_degenerate_subspace"],
        candidate["near_degenerate_subspace"],
        relative_gap=relative_gap,
    )
    stability_rows.append(
        _record(context=context, method="near_degenerate_subspace", layer="__all__", metrics=metrics)
    )
    match_rows.extend(
        {**context, "method": "near_degenerate_subspace", "layer": "__all__", **row}
        for row in matches
    )

    layer_records = []
    common_layers = sorted(
        set(reference["layerwise_mode"]).intersection(candidate["layerwise_mode"])
    )
    for layer in common_layers:
        layer_metrics, matches = compare_atom_modes(
            reference["layerwise_mode"][layer], candidate["layerwise_mode"][layer]
        )
        row = _record(
            context=context, method="layerwise_mode", layer=layer, metrics=layer_metrics
        )
        stability_rows.append(row)
        layer_records.append(row)
        match_rows.extend(
            {**context, "method": "layerwise_mode", "layer": layer, **match}
            for match in matches
        )
    valid = [row for row in layer_records if math.isfinite(float(row["stability_score"]))]
    total_matches = sum(int(row["match_count"]) for row in valid)
    if total_matches:
        aggregate = {
            "stability_score": sum(
                float(row["stability_score"]) * int(row["match_count"]) for row in valid
            ) / total_matches,
            "worst_match": min(float(row["worst_match"]) for row in valid),
            "match_count": total_matches,
        }
    else:
        aggregate = {"stability_score": math.nan, "worst_match": math.nan, "match_count": 0}
    stability_rows.append(
        _record(context=context, method="layerwise_mode", layer="__all__", metrics=aggregate)
    )
    return stability_rows, match_rows


def spectrum_rows(source, representations: dict, relative_gap: float) -> list[dict]:
    rows = []

    def append(method: str, layer: str, modes: ModeSet):
        groups = degenerate_groups(modes.singular_values, relative_gap)
        group_lookup = {
            mode_index: group_index for group_index, group in enumerate(groups) for mode_index in group
        }
        total_energy = float(modes.singular_values.square().sum().item())
        for index, value in enumerate(modes.singular_values.tolist()):
            next_value = (
                float(modes.singular_values[index + 1].item())
                if index + 1 < modes.singular_values.numel()
                else math.nan
            )
            gap = (
                abs(float(value) - next_value) / max(abs(float(value)), 1e-12)
                if math.isfinite(next_value)
                else math.nan
            )
            rows.append({
                "source_id": source.source_id,
                "seed": source.seed,
                "round": source.round_id,
                "partition": source.partition,
                "method": method,
                "layer": layer,
                "mode_index": index,
                "singular_value": float(value),
                "energy_fraction": float(value * value / max(total_energy, 1e-12)),
                "relative_gap_to_next": gap,
                "degenerate_group": group_lookup.get(index, -1),
            })

    append("single_svd_atom", "__all__", representations["single_svd_atom"])
    append(
        "cross_layer_joint_sketch",
        "__all__",
        representations["cross_layer_joint_sketch"],
    )
    for layer, modes in representations["layerwise_mode"].items():
        append("layerwise_mode", layer, modes)
    return rows


def summarize(stability_rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in stability_rows:
        if row["layer"] != "__all__":
            continue
        value = float(row["stability_score"])
        if math.isfinite(value):
            groups[(row["scope"], row["perturbation"], row["method"], row["rank_candidate"])].append(value)
    result = []
    for (scope, perturbation, method, rank), values in sorted(groups.items()):
        array = np.asarray(values, dtype=float)
        result.append({
            "scope": scope,
            "perturbation": perturbation,
            "method": method,
            "rank_candidate": rank,
            "unit_count": len(values),
            "stability_mean": float(array.mean()),
            "stability_std": float(array.std()),
            "stability_min": float(array.min()),
            "stability_max": float(array.max()),
        })
    return result


def build_verdict(summary_rows: list[dict], args) -> dict:
    eligible = (
        "single_svd_atom",
        "near_degenerate_subspace",
        "layerwise_mode",
        "cross_layer_joint_sketch",
    )
    components = defaultdict(dict)
    for row in summary_rows:
        method = row["method"]
        if method not in eligible:
            continue
        key = None
        if row["scope"] == "within_dump" and row["perturbation"] == "rank_only":
            previous = components[method].get("rank_stability", math.inf)
            components[method]["rank_stability"] = min(previous, float(row["stability_mean"]))
            continue
        if int(row["rank_candidate"]) != int(max(args.ranks)):
            continue
        if row["scope"] == "within_dump" and row["perturbation"] == "client_dropout_weight_jitter":
            key = "within_client_perturbation"
        elif row["scope"] == "within_dump" and row["perturbation"] == "sketch_seed":
            key = "sketch_seed"
        elif row["scope"] == "cross_dump" and row["perturbation"] == "cross_seed":
            key = "cross_seed"
        if key:
            components[method][key] = float(row["stability_mean"])

    candidates = []
    for method in eligible:
        values = components.get(method, {})
        required = [
            values.get("rank_stability", math.nan),
            values.get("within_client_perturbation", math.nan),
            values.get("cross_seed", math.nan),
        ]
        if method == "cross_layer_joint_sketch":
            required.append(values.get("sketch_seed", math.nan))
        finite = [value for value in required if math.isfinite(value)]
        conservative = min(finite) if len(finite) == len(required) else math.nan
        candidates.append({
            "method": method,
            **values,
            "conservative_score": conservative,
            "passes_within_threshold": values.get("within_client_perturbation", -math.inf)
            >= args.min_within_stability,
            "passes_rank_threshold": values.get("rank_stability", -math.inf)
            >= args.min_rank_stability,
            "passes_cross_seed_threshold": values.get("cross_seed", -math.inf)
            >= args.min_cross_seed_stability,
            "passes_sketch_seed_threshold": (
                method != "cross_layer_joint_sketch"
                or values.get("sketch_seed", -math.inf) >= args.min_sketch_stability
            ),
        })
    valid = [row for row in candidates if math.isfinite(float(row["conservative_score"]))]
    valid.sort(key=lambda row: (-float(row["conservative_score"]), row["method"]))
    recommended = valid[0]["method"] if valid else None
    chosen = valid[0] if valid else None
    passed = bool(
        chosen
        and chosen["passes_within_threshold"]
        and chosen["passes_rank_threshold"]
        and chosen["passes_cross_seed_threshold"]
        and chosen["passes_sketch_seed_threshold"]
    )
    return {
        "verdict": "READY_FOR_V2" if passed else "REPRESENTATION_NOT_STABLE_YET",
        "recommended_representation": recommended,
        "candidate_scores": candidates,
        "thresholds": {
            "minimum_within_client_perturbation": args.min_within_stability,
            "minimum_rank_stability": args.min_rank_stability,
            "minimum_cross_seed": args.min_cross_seed_stability,
            "minimum_sketch_seed": args.min_sketch_stability,
        },
        "interpretation": (
            "The recommendation is a V1 representation choice, not evidence that the selected modes "
            "have positive tail utility. That causal question belongs to V2."
        ),
    }


def write_report(output_dir: Path, verdict: dict) -> None:
    lines = [
        "# V1 mode-representation stability",
        "",
        f"Verdict: **{verdict['verdict']}**",
        "",
        f"Recommended representation: **{verdict.get('recommended_representation') or 'none'}**",
        "",
        "| representation | conservative | rank | client perturbation | cross seed | sketch seed |",
        "|---|---:|---:|---:|---:|---:|",
    ]

    def value(row: dict, key: str) -> str:
        number = float(row.get(key, math.nan))
        return f"{number:.4f}" if math.isfinite(number) else "n/a"

    for row in verdict["candidate_scores"]:
        lines.append(
            f"| {row['method']} | {value(row, 'conservative_score')} | "
            f"{value(row, 'rank_stability')} | {value(row, 'within_client_perturbation')} | "
            f"{value(row, 'cross_seed')} | {value(row, 'sketch_seed')} |"
        )
    lines.extend([
        "",
        "This verdict establishes representation stability only. It does not establish positive tail utility; that is the V2 question.",
    ])
    (output_dir / "v1_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dump-dirs", type=Path, nargs="+", required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ranks", type=int, nargs="+", default=[4, 8, 16])
    parser.add_argument("--perturb-repeats", type=int, default=5)
    parser.add_argument("--client-dropout", type=float, default=0.1)
    parser.add_argument("--weight-jitter", type=float, default=0.05)
    parser.add_argument("--sketch-dim", type=int, default=128)
    parser.add_argument("--sketch-seed", type=int, default=2026)
    parser.add_argument("--degenerate-relative-gap", type=float, default=0.05)
    parser.add_argument("--min-within-stability", type=float, default=0.75)
    parser.add_argument("--min-rank-stability", type=float, default=0.75)
    parser.add_argument("--min-cross-seed-stability", type=float, default=0.50)
    parser.add_argument("--min-sketch-stability", type=float, default=0.75)
    return parser.parse_args()


def main():
    args = parse_args()
    ranks = sorted(set(int(rank) for rank in args.ranks))
    if not ranks or ranks[0] < 1:
        raise ValueError("--ranks must contain positive integers")
    args.ranks = ranks
    if not 0.0 <= args.client_dropout < 1.0:
        raise ValueError("--client-dropout must be in [0, 1)")
    if args.perturb_repeats < 1:
        raise ValueError("--perturb-repeats must be positive")

    sources, inventory = [], []
    for dump_dir in args.dump_dirs:
        source, _, row = load_source(dump_dir)
        sources.append(source)
        inventory.append(row)
    source_ids = [source.source_id for source in sources]
    if len(set(source_ids)) != len(source_ids):
        raise ValueError(f"Duplicate V1 source identities: {source_ids}")
    reference_rank = max(ranks)
    stability_rows, match_rows, spectra = [], [], []
    clean_data, clean_representations = {}, {}

    for source in sources:
        clean = build_disagreement_set(source)
        reference = build_representations(
            clean,
            rank=reference_rank,
            sketch_dim=args.sketch_dim,
            sketch_seed=args.sketch_seed,
        )
        clean_data[source.source_id] = clean
        clean_representations[source.source_id] = reference
        spectra.extend(spectrum_rows(source, reference, args.degenerate_relative_gap))

        for rank in ranks:
            candidate = build_representations(
                clean,
                rank=rank,
                sketch_dim=args.sketch_dim,
                sketch_seed=args.sketch_seed,
            )
            context = {
                "scope": "within_dump",
                "perturbation": "rank_only",
                "source_a": source.source_id,
                "source_b": source.source_id,
                "seed_a": source.seed,
                "seed_b": source.seed,
                "round_a": source.round_id,
                "round_b": source.round_id,
                "trial_seed": args.sketch_seed,
                "rank_reference": reference_rank,
                "rank_candidate": rank,
                "retained_clients": len(clean.client_ids),
            }
            rows, matches = compare_representations(
                reference, candidate, context=context, relative_gap=args.degenerate_relative_gap
            )
            stability_rows.extend(rows)
            match_rows.extend(matches)

        for repeat in range(args.perturb_repeats):
            trial_seed = stable_seed("v1-perturb", source.source_id, repeat, args.sketch_seed)
            generator = torch.Generator(device="cpu")
            generator.manual_seed(trial_seed)
            count = len(source.client_ids)
            keep = max(reference_rank + 2, int(round(count * (1.0 - args.client_dropout))))
            keep = min(count, keep)
            indices = torch.randperm(count, generator=generator)[:keep].sort().values
            jitter = torch.exp(
                args.weight_jitter * torch.randn(keep, generator=generator, dtype=torch.float64)
            )
            perturbed = build_disagreement_set(
                source, row_indices=indices.tolist(), weight_multipliers=jitter
            )
            for rank in ranks:
                candidate = build_representations(
                    perturbed,
                    rank=rank,
                    sketch_dim=args.sketch_dim,
                    sketch_seed=trial_seed,
                )
                context = {
                    "scope": "within_dump",
                    "perturbation": "client_dropout_weight_jitter",
                    "source_a": source.source_id,
                    "source_b": source.source_id,
                    "seed_a": source.seed,
                    "seed_b": source.seed,
                    "round_a": source.round_id,
                    "round_b": source.round_id,
                    "trial_seed": trial_seed,
                    "rank_reference": reference_rank,
                    "rank_candidate": rank,
                    "retained_clients": keep,
                }
                rows, matches = compare_representations(
                    reference,
                    candidate,
                    context=context,
                    relative_gap=args.degenerate_relative_gap,
                )
                stability_rows.extend(rows)
                match_rows.extend(matches)

            sketch_candidate = build_representations(
                clean,
                rank=reference_rank,
                sketch_dim=args.sketch_dim,
                sketch_seed=trial_seed,
            )
            context = {
                "scope": "within_dump",
                "perturbation": "sketch_seed",
                "source_a": source.source_id,
                "source_b": source.source_id,
                "seed_a": source.seed,
                "seed_b": source.seed,
                "round_a": source.round_id,
                "round_b": source.round_id,
                "trial_seed": trial_seed,
                "rank_reference": reference_rank,
                "rank_candidate": reference_rank,
                "retained_clients": len(clean.client_ids),
            }
            metrics, matches = compare_atom_modes(
                reference["cross_layer_joint_sketch"],
                sketch_candidate["cross_layer_joint_sketch"],
            )
            stability_rows.append(
                _record(
                    context=context,
                    method="cross_layer_joint_sketch",
                    layer="__all__",
                    metrics=metrics,
                )
            )
            match_rows.extend(
                {**context, "method": "cross_layer_joint_sketch", "layer": "__all__", **row}
                for row in matches
            )

    for left in range(len(sources)):
        for right in range(left + 1, len(sources)):
            source_a, source_b = sources[left], sources[right]
            if source_a.spec.as_dict() != source_b.spec.as_dict():
                continue
            if source_a.partition != source_b.partition:
                continue
            if source_a.round_id == source_b.round_id and source_a.seed != source_b.seed:
                perturbation = "cross_seed"
            elif source_a.seed == source_b.seed and source_a.round_id != source_b.round_id:
                perturbation = "cross_round"
            else:
                continue
            context = {
                "scope": "cross_dump",
                "perturbation": perturbation,
                "source_a": source_a.source_id,
                "source_b": source_b.source_id,
                "seed_a": source_a.seed,
                "seed_b": source_b.seed,
                "round_a": source_a.round_id,
                "round_b": source_b.round_id,
                "trial_seed": args.sketch_seed,
                "rank_reference": reference_rank,
                "rank_candidate": reference_rank,
                "retained_clients": min(len(source_a.client_ids), len(source_b.client_ids)),
            }
            rows, matches = compare_representations(
                clean_representations[source_a.source_id],
                clean_representations[source_b.source_id],
                context=context,
                relative_gap=args.degenerate_relative_gap,
            )
            stability_rows.extend(rows)
            match_rows.extend(matches)

    summary_rows = summarize(stability_rows)
    verdict = build_verdict(summary_rows, args)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    write_csv(args.output_dir / "source_inventory.csv", inventory)
    write_csv(args.output_dir / "singular_spectra.csv", spectra)
    write_csv(args.output_dir / "mode_matches.csv", match_rows)
    write_csv(args.output_dir / "stability_units.csv", stability_rows)
    write_csv(args.output_dir / "stability_summary.csv", summary_rows)
    write_json(args.output_dir / "v1_verdict.json", verdict)
    write_json(args.output_dir / "v1_manifest.json", {
        "schema_version": V1_SCHEMA_VERSION,
        "source_count": len(sources),
        "sources": inventory,
        "protocol": {
            "parameter_space": "uploaded_raw_lora_parameter_delta",
            "centering": "selected-client FedAvg center",
            "fedavg_direction_projection": True,
            "client_dropout": args.client_dropout,
            "weight_jitter_log_sigma": args.weight_jitter,
            "perturb_repeats": args.perturb_repeats,
            "ranks": ranks,
            "sketch_dim_per_layer": args.sketch_dim,
            "sketch_seed": args.sketch_seed,
            "degenerate_relative_gap": args.degenerate_relative_gap,
            "matching": (
                "sign-invariant exact maximum-weight assignment (DP up to 12, Hungarian above 12); "
                "degenerate groups use principal-angle overlap"
            ),
            "test_or_validation_access": False,
        },
        "warning": (
            "This V1 establishes representation stability only. V2 must still measure tail gain and "
            "head damage for each selected mode. Raw LoRA-space winners should additionally be audited "
            "in basis-invariant effective-BA space before becoming the final CMSA representation."
        ),
    })
    write_report(args.output_dir, verdict)
    print(f"V1 mode stability finished: {args.output_dir}")
    print(json.dumps(verdict, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
