"""Run the V2 background-adjusted Compatibility-to-Retention Bridge.

V1 used M(theta0 + delta_tail + delta_bg) - M(theta0) as its numerator.
That quantity includes the background update's direct target-class effect and
produced a denominator artifact when the background update strongly improved
M_c.  V2 preserves all frozen V1 training and adds only the missing
background-only counterfactual:

    R*_c = [M(theta0 + delta_tail + delta_bg) - M(theta0 + delta_bg)]
           / [M(theta0 + delta_tail) - M(theta0)].
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd

from tools.boundary_evidence.core import class_cluster_summary
from tools.boundary_evidence.run import ROOT, _eval_pair
from tools.compatibility_retention.corrected_core import (
    background_adjusted_components,
    corrected_tail_retention_rows,
)
from tools.compatibility_retention.core import CONDITIONS
from tools.compatibility_retention.run import (
    _background_path,
    _load_state,
    _read_contract as _read_v1_contract,
    _runtime,
)
from tools.semantic_acquisition.common import (
    file_sha256,
    tensor_mapping_hash,
    write_csv,
    write_json,
)
from tools.semantic_acquisition.manifests import DEFAULT_DATA
from tools.semantic_acquisition.runtime import load_lora_state


DEFAULT_V1 = ROOT / "output" / "compatibility_retention_bridge"
DEFAULT_OUTPUT = ROOT / "output" / "compatibility_retention_bridge_v2"
V2_MANIFESTS = ("background_state_manifest.csv", "corrected_eval_manifest.csv")


def _expected_pairs(contract) -> int:
    return (
        len(contract["data_seeds"])
        * len(contract["tail_classes"])
        * int(contract["hard_k"])
    )


def prepare(args) -> dict:
    v1_dir, output = Path(args.v1_dir), Path(args.output_dir)
    v1_contract = _read_v1_contract(v1_dir)
    v1_metrics_path = v1_dir / "bridge_metrics.csv"
    aggregate_metrics_path = v1_dir / "background_aggregate_metrics.csv"
    if not v1_metrics_path.is_file() or not aggregate_metrics_path.is_file():
        raise FileNotFoundError("V1 bridge and background aggregate metrics are required")
    v1_metrics = pd.read_csv(v1_metrics_path)
    expected_rows = _expected_pairs(v1_contract) * len(CONDITIONS)
    if len(v1_metrics) != expected_rows:
        raise RuntimeError(f"V1 bridge is incomplete: expected {expected_rows}, found {len(v1_metrics)}")
    pair_frame = (
        v1_metrics.loc[:, ["data_seed", "tail_class", "hard_class", "hard_rank"]]
        .drop_duplicates()
        .sort_values(["data_seed", "tail_class", "hard_class"])
    )
    if len(pair_frame) != _expected_pairs(v1_contract):
        raise RuntimeError("V1 bridge does not contain the expected unique (seed,c,h) pairs")
    aggregates = pd.read_csv(aggregate_metrics_path)
    aggregate_index = {
        (int(row.data_seed), int(row.tail_class)): row for row in aggregates.itertuples()
    }
    expected_aggregates = len(v1_contract["data_seeds"]) * len(v1_contract["tail_classes"])
    if len(aggregate_index) != expected_aggregates:
        raise RuntimeError(
            f"V1 background aggregates are incomplete: expected {expected_aggregates}, "
            f"found {len(aggregate_index)}"
        )
    state_rows = []
    for seed in [int(value) for value in v1_contract["data_seeds"]]:
        for tail_class in [int(value) for value in v1_contract["tail_classes"]]:
            key = (seed, tail_class)
            state_path = _background_path(v1_dir, seed, tail_class)
            if not state_path.is_file():
                raise FileNotFoundError(
                    f"V1 background state is missing on the compute server: {state_path}"
                )
            state = _load_state(state_path, v1_contract["theta0_hash"])
            observed_hash = tensor_mapping_hash(state)
            expected_hash = str(aggregate_index[key].background_state_hash)
            if observed_hash != expected_hash:
                raise RuntimeError(f"V1 background state differs from its recorded hash: {key}")
            state_rows.append({
                "data_seed": seed,
                "tail_class": tail_class,
                "background_state_path": str(state_path.resolve()),
                "background_state_hash": observed_hash,
            })
    output.mkdir(parents=True, exist_ok=True)
    write_csv(output / "background_state_manifest.csv", state_rows)
    write_csv(output / "corrected_eval_manifest.csv", pair_frame.to_dict("records"))
    contract = {
        "schema_version": 2,
        "name": "Background-adjusted Compatibility-to-Retention Bridge",
        "supersedes": "V1 absolute-gain ratio, which is diagnostic_only_due_to_denominator_artifact",
        "v1_dir": str(v1_dir.resolve()),
        "v1_contract_sha256": file_sha256(v1_dir / "experiment_contract.json"),
        "v1_bridge_metrics_sha256": file_sha256(v1_metrics_path),
        "v1_background_metrics_sha256": file_sha256(aggregate_metrics_path),
        "theta0_hash": v1_contract["theta0_hash"],
        "data_seeds": [int(value) for value in v1_contract["data_seeds"]],
        "tail_classes": [int(value) for value in v1_contract["tail_classes"]],
        "hard_k": int(v1_contract["hard_k"]),
        "conditions": list(CONDITIONS),
        "training": "none; reuses all frozen V1 local and background updates",
        "new_evaluation": "background-only M_c on the frozen P_eval pairs",
        "primary_endpoint": (
            "R*_c=[M(theta0+delta_tail+delta_bg)-M(theta0+delta_bg)]/"
            "[M(theta0+delta_tail)-M(theta0)]"
        ),
        "ratio_order": (
            "average G_local and G_post_marginal over five hard-negative pairs and data seeds "
            "within tail class, then divide"
        ),
        "inference_unit": "20 tail classes",
        "directional_gate": (
            "mean(R*_hard-R*_control)>0 and 95% tail-class bootstrap CI excludes zero"
        ),
        "manifest_hashes": {
            name: file_sha256(output / name) for name in V2_MANIFESTS
        },
        "implementation_hashes": {
            name: file_sha256(ROOT / name)
            for name in (
                "tools/compatibility_retention/corrected_core.py",
                "tools/compatibility_retention/corrected.py",
                "tools/compatibility_retention/core.py",
                "tools/compatibility_retention/run.py",
                "tools/boundary_evidence/core.py",
                "tools/boundary_evidence/run.py",
                "tools/semantic_acquisition/runtime.py",
            )
        },
        "claim_boundary": (
            "Tests whether the marginal target contribution of the under-constrained c+r update "
            "is retained less under one identical real class-absent background state. It does "
            "not quantify how much of the 13.85pp end-to-end gap is mediated by this mechanism."
        ),
    }
    write_json(output / "experiment_contract.json", contract)
    return contract


def _read_contract(output: Path) -> dict:
    output = Path(output)
    contract = json.loads((output / "experiment_contract.json").read_text(encoding="utf-8"))
    if int(contract.get("schema_version", -1)) != 2:
        raise RuntimeError("Corrected bridge requires a schema-version-2 contract")
    v1_dir = Path(contract["v1_dir"])
    _read_v1_contract(v1_dir)
    sources = {
        "experiment_contract.json": contract["v1_contract_sha256"],
        "bridge_metrics.csv": contract["v1_bridge_metrics_sha256"],
        "background_aggregate_metrics.csv": contract["v1_background_metrics_sha256"],
    }
    for name, expected in sources.items():
        if file_sha256(v1_dir / name) != expected:
            raise RuntimeError(f"Frozen V1 bridge artifact changed: {name}")
    for name, expected in contract["manifest_hashes"].items():
        if file_sha256(output / name) != expected:
            raise RuntimeError(f"Frozen corrected-bridge manifest changed: {name}")
    for name, expected in contract["implementation_hashes"].items():
        if file_sha256(ROOT / name) != expected:
            raise RuntimeError(f"Corrected bridge implementation changed after preparation: {name}")
    return contract


def run_background_only(args) -> list[dict]:
    output = Path(args.output_dir)
    contract = _read_contract(output)
    v1_dir = Path(contract["v1_dir"])
    v1_contract = _read_v1_contract(v1_dir)
    _, store, model, _, _, eval_transform = _runtime(args, v1_contract)
    manifest = pd.read_csv(output / "corrected_eval_manifest.csv")
    state_manifest = pd.read_csv(output / "background_state_manifest.csv")
    state_index = {
        (int(row.data_seed), int(row.tail_class)): row for row in state_manifest.itertuples()
    }
    metrics_path = output / "background_only_metrics.csv"
    rows = pd.read_csv(metrics_path).to_dict("records") if metrics_path.is_file() else []
    completed = {
        (int(row["data_seed"]), int(row["tail_class"]), int(row["hard_class"]))
        for row in rows
    }
    loaded_key = None
    for row in manifest.itertuples():
        key = (int(row.data_seed), int(row.tail_class), int(row.hard_class))
        if key in completed:
            continue
        state_key = key[:2]
        state_row = state_index[state_key]
        if loaded_key != state_key:
            background_state = _load_state(
                Path(state_row.background_state_path), contract["theta0_hash"]
            )
            if tensor_mapping_hash(background_state) != str(state_row.background_state_hash):
                raise RuntimeError(f"Background state hash changed for {state_key}")
            load_lora_state(model, background_state)
            loaded_key = state_key
        metrics = _eval_pair(model, store, eval_transform, key[1], key[2])
        rows.append({
            "data_seed": key[0],
            "tail_class": key[1],
            "hard_class": key[2],
            "hard_rank": int(row.hard_rank),
            "background_only_m_c": float(metrics["m_c"]),
            "background_state_hash": str(state_row.background_state_hash),
        })
        write_csv(metrics_path, rows)
        completed.add(key)
        print(json.dumps({
            "stage": "background_only",
            "completed": key,
            "background_only_m_c": float(metrics["m_c"]),
        }, sort_keys=True))
    return rows


def summarize(args) -> dict:
    output = Path(args.output_dir)
    contract = _read_contract(output)
    v1_dir = Path(contract["v1_dir"])
    background_path = output / "background_only_metrics.csv"
    if not background_path.is_file():
        raise FileNotFoundError("Run --stage background-only before corrected summarization")
    background = pd.read_csv(background_path)
    expected_pairs = _expected_pairs(contract)
    if len(background) != expected_pairs:
        raise RuntimeError(
            f"Background-only evaluation is incomplete: expected {expected_pairs}, found {len(background)}"
        )
    if background.duplicated(["data_seed", "tail_class", "hard_class"]).any():
        raise RuntimeError("Background-only evaluation contains duplicate pair rows")
    background_index = {
        (int(row.data_seed), int(row.tail_class), int(row.hard_class)): row
        for row in background.itertuples()
    }
    v1_rows = pd.read_csv(v1_dir / "bridge_metrics.csv").to_dict("records")
    component_rows = []
    for row in v1_rows:
        pair_key = (
            int(row["data_seed"]), int(row["tail_class"]), int(row["hard_class"])
        )
        bg = background_index[pair_key]
        if str(row["background_state_hash"]) != str(bg.background_state_hash):
            raise RuntimeError(f"V1 post and V2 background-only states differ for {pair_key}")
        components = background_adjusted_components(
            theta0_m_c=float(row["theta0_m_c"]),
            local_m_c=float(row["local_m_c"]),
            background_only_m_c=float(bg.background_only_m_c),
            post_m_c=float(row["post_m_c"]),
        )
        component_rows.append({
            "data_seed": pair_key[0],
            "tail_class": pair_key[1],
            "hard_class": pair_key[2],
            "hard_rank": int(row["hard_rank"]),
            "control_class": int(row["control_class"]),
            "condition": str(row["condition"]),
            "theta0_m_c": float(row["theta0_m_c"]),
            "local_m_c": float(row["local_m_c"]),
            "background_only_m_c": float(bg.background_only_m_c),
            "post_m_c": float(row["post_m_c"]),
            **components,
            "background_state_hash": str(bg.background_state_hash),
        })
    expected_rows = expected_pairs * len(CONDITIONS)
    if len(component_rows) != expected_rows:
        raise RuntimeError(f"Corrected components expected {expected_rows} rows")
    write_csv(output / "corrected_components.csv", component_rows)
    class_rows = corrected_tail_retention_rows(component_rows)
    if len(class_rows) != len(contract["tail_classes"]):
        raise RuntimeError("Corrected bridge does not have exactly one ratio per tail class")
    write_csv(output / "corrected_per_tail_class.csv", class_rows)
    fields = {
        "hard_competitor": "hard_corrected_retention_ratio",
        "matched_control": "control_corrected_retention_ratio",
        "hard_competitor_minus_matched_control": (
            "hard_minus_control_corrected_retention_ratio"
        ),
    }
    summaries = {}
    for index, (name, field) in enumerate(fields.items()):
        values = {int(row["tail_class"]): [float(row[field])] for row in class_rows}
        summaries[name] = class_cluster_summary(values, seed=20260913 + index)
    write_csv(output / "corrected_main_results.csv", [
        {
            "condition_or_contrast": name,
            "corrected_retention_ratio": value["mean"],
            "ci_low": value["ci_low"],
            "ci_high": value["ci_high"],
            "tail_class_count": value["tail_class_count"],
        }
        for name, value in summaries.items()
    ])
    contrast = summaries["hard_competitor_minus_matched_control"]
    gate = bool(contrast["mean"] > 0.0 and contrast["ci_low"] > 0.0)
    result = {
        "schema_version": 2,
        "verdict": (
            "BACKGROUND_ADJUSTED_COMPATIBILITY_TO_RETENTION_SUPPORTED"
            if gate else "BACKGROUND_ADJUSTED_COMPATIBILITY_TO_RETENTION_NOT_SUPPORTED"
        ),
        "v1_verdict_status": "SUPERSEDED_DENOMINATOR_ARTIFACT_NOT_EVIDENCE",
        "primary_endpoint": contract["primary_endpoint"],
        "by_condition": {
            "hard_competitor": summaries["hard_competitor"],
            "matched_control": summaries["matched_control"],
        },
        "hard_competitor_minus_matched_control": contrast,
        "directional_gate": {
            "Rstar_c_plus_r_lower_than_Rstar_c_plus_h_with_95pct_CI": gate,
        },
        "inference_unit": (
            "20 tail classes; G_local and background-adjusted G_post_marginal are averaged over "
            "five hard-negative pairs and data seeds within class before R*_c is formed"
        ),
        "claim_boundary": contract["claim_boundary"],
    }
    write_json(output / "summary.json", result)
    return result


def parse_args(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--stage",
        choices=("prepare", "background-only", "summarize", "all"),
        required=True,
    )
    parser.add_argument("--v1-dir", type=Path, default=DEFAULT_V1)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    return parser.parse_args(argv)


def main(argv=None):
    args = parse_args(argv)
    if args.stage in ("prepare", "all"):
        prepare(args)
    if args.stage in ("background-only", "all"):
        run_background_only(args)
    if args.stage in ("summarize", "all"):
        result = summarize(args)
        print(json.dumps(result, ensure_ascii=False, sort_keys=True, indent=2))


if __name__ == "__main__":
    main()

