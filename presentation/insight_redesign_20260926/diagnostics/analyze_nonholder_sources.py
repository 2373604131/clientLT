"""Audit potential non-holder support on the existing Full-CP trajectory.

Read-only with respect to all training artifacts. No model or checkpoint is loaded.
Run from any directory with Python and NumPy; optional --run and --out overrides.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from pathlib import Path

import numpy as np


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def main():
    root = Path(__file__).resolve().parents[3]
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run", type=Path, default=root / "output/sfra_cp_analysis/sfra_cp/seed42/client-longtail/full-cp/lambda10_mu1_protocol42")
    parser.add_argument("--out", type=Path, default=Path(__file__).resolve().parent)
    args = parser.parse_args()
    run, out = args.run.resolve(), args.out.resolve()
    out.mkdir(parents=True, exist_ok=True)
    witness_path, partition_path = run / "private_witness_manifest.json", run / "partition_manifest.csv"
    witnesses = json.loads(witness_path.read_text(encoding="utf-8"))
    with partition_path.open(encoding="utf-8-sig", newline="") as handle:
        partition = [{k: int(v) for k, v in row.items()} for row in csv.DictReader(handle)]
    clients = sorted({row["client_id"] for row in partition})
    classes = sorted({row["class_id"] for row in partition})
    assert clients == list(range(30)) and classes == list(range(100))
    assert [row["token_id"] for row in witnesses] == list(range(len(witnesses)))
    assert len({(row["client_id"], row["class_id"]) for row in witnesses}) == len(witnesses)
    indexed_samples = {(row["client_id"], row["local_position"]): row for row in partition}
    for token in witnesses:
        assert len(token["local_positions"]) == len(token["raw_sample_ids"])
        for pos, raw_id in zip(token["local_positions"], token["raw_sample_ids"]):
            sample = indexed_samples[(token["client_id"], pos)]
            assert sample["raw_sample_id"] == raw_id and sample["class_id"] == token["class_id"]
    present = np.zeros((100, 30), dtype=bool)
    for sample in partition:
        present[sample["class_id"], sample["client_id"]] = True
    token_classes = np.array([t["class_id"] for t in witnesses])
    holders = present[token_classes]
    nonholder_candidate_share = (~holders).mean(axis=1)
    config = json.loads((run / "sfra_config.json").read_text(encoding="utf-8"))
    tail_classes = np.array(config["base_training_config"]["tail_ids"])
    assert config["positive_response_threshold"] == 1e-6
    groups = {"all": np.ones(len(witnesses), dtype=bool), "tail20": np.isin(token_classes, tail_classes), "non_tail80": ~np.isin(token_classes, tail_classes)}
    rows, source_hashes = [], {}
    max_neff_error = 0.0
    for rnd in range(1, 91):
        path = run / f"sfra_rounds/r{rnd:03d}/tokens.npz"
        source_hashes[str(path.relative_to(run))] = sha256(path)
        with np.load(path, allow_pickle=False) as data:
            response = data["responses"]
            assert response.shape == (len(witnesses), 2, len(clients))
            assert np.isfinite(response).all()
            # Exact rule in utils/sfra_math.py: minimum over two views, >1e-6.
            conservative = response.min(axis=1)
            positive = np.where(conservative > 1e-6, conservative, 0.0).astype(np.float64)
            total = positive.sum(axis=1)
            supported = total > 0
            counts = (positive > 0).sum(axis=1)
            assert np.array_equal(supported, data["supported"])
            assert np.array_equal(counts, data["positive_sources"])
            neff = np.zeros(len(witnesses))
            neff[supported] = total[supported] ** 2 / (positive[supported] ** 2).sum(axis=1)
            error = float(np.max(np.abs(neff - data["n_eff"])))
            max_neff_error = max(error, max_neff_error)
            assert np.allclose(neff, data["n_eff"], atol=2e-5, rtol=2e-5)
        nonholder_total = (positive * ~holders).sum(axis=1)
        has_nonholder = nonholder_total > 0
        share = np.divide(nonholder_total, total, out=np.zeros_like(total), where=supported)
        for group, mask in groups.items():
            supported_mask = mask & supported
            class_ids = np.unique(token_classes[mask])
            class_supported = [c for c in class_ids if np.any(supported_mask & (token_classes == c))]
            basic = dict(round=rnd, group=group, token_count=int(mask.sum()), supported_token_count=int(supported_mask.sum()), unsupported_token_count=int((mask & ~supported).sum()), class_count=len(class_ids), supported_class_count=len(class_supported))
            rows.append(dict(**basic, averaging="token_equal", supported_fraction=float(supported[mask].mean()), nonholder_exists_given_supported=float(has_nonholder[supported_mask].mean()), nonholder_positive_share_given_supported=float(share[supported_mask].mean()), nonholder_candidate_share_given_supported=float(nonholder_candidate_share[supported_mask].mean())))
            rows.append(dict(**basic, averaging="class_equal", supported_fraction=float(np.mean([supported[mask & (token_classes == c)].mean() for c in class_ids])), nonholder_exists_given_supported=float(np.mean([has_nonholder[supported_mask & (token_classes == c)].mean() for c in class_supported])), nonholder_positive_share_given_supported=float(np.mean([share[supported_mask & (token_classes == c)].mean() for c in class_supported])), nonholder_candidate_share_given_supported=float(np.mean([nonholder_candidate_share[supported_mask & (token_classes == c)].mean() for c in class_supported]))))
    summary = []
    metrics = ["supported_fraction", "nonholder_exists_given_supported", "nonholder_positive_share_given_supported", "nonholder_candidate_share_given_supported"]
    for group in groups:
        for averaging in ["token_equal", "class_equal"]:
            subset = [r for r in rows if r["group"] == group and r["averaging"] == averaging]
            item = dict(group=group, averaging=averaging, rounds=90, token_count=subset[0]["token_count"], class_count=subset[0]["class_count"], supported_token_rounds=sum(r["supported_token_count"] for r in subset), excluded_unsupported_token_rounds=sum(r["unsupported_token_count"] for r in subset), min_supported_classes=min(r["supported_class_count"] for r in subset))
            for metric in metrics:
                values = [r[metric] for r in subset]
                item[metric + "_mean_over_rounds"] = float(np.mean(values))
                item[metric + "_min_over_rounds"] = float(np.min(values))
                item[metric + "_max_over_rounds"] = float(np.max(values))
            summary.append(item)
    write_csv(out / "nonholder_sources_rounds.csv", rows)
    write_csv(out / "nonholder_sources_summary.csv", summary)
    result = dict(run=str(run), scope="Protected Full-CP, seed42, Client-LT, rounds 1-90, all 1516 client-class witness units; first-order potential support only.", response_definition="r=min over two views of the source response; a=r if r>1e-6, else 0. Source client order is sorted(client_ids) in utils/cliplora_sfra.py refresh().", ownership_definition="A source holds the class iff the actual full partition_manifest contains at least one sample for that source-client and class.", summary_definition="First compute each round; then average all 90 rounds equally. token_equal averages client-class units. class_equal averages units within class, then classes equally. Conditional metrics exclude unsupported units; class_equal additionally excludes classes without a supported unit for that round.", limitations=["Responses are gradients dotted with client A updates on an already protected Full-CP trajectory; they are not finite-update gains, unprotected-S findings, or actual transfer results.", "Positive-response shares are not FedAvg-weighted contribution or a decomposition of actual global model gains.", "Witnesses are training images; this audit does not establish generalization, causal effects, future forgetting prediction, or a universal cross-label transfer law.", "Tokens, classes, and rounds are not independent seeds. Min/max describe observed round variation, not confidence intervals."], validation=dict(token_count=len(witnesses), tail_token_count=int(groups["tail20"].sum()), token_ids_contiguous=True, unique_client_class_tokens=True, witness_sample_ids_and_local_positions_match_partition=True, all_90_shapes_finite_and_correct=True, saved_supported_and_positive_counts_exact_match=True, max_recomputed_neff_absolute_error=max_neff_error, source_client_order="0..29, sorted, full participation", manifest_sha256={"partition":sha256(partition_path), "witness":sha256(witness_path)}, token_file_sha256=source_hashes), results=summary)
    result["candidate_reference_definition"] = "For each supported token, count clients lacking its target class and divide by all 30 candidate clients; aggregate using exactly the same conditional token/class and round weights as the positive-response share. This is a candidate-count composition reference, not a null model or a test of donor efficiency."
    result["limitations"].append("Non-holders are more numerous, especially for tail classes. A high response share alone does not imply larger per-client effectiveness or excess response beyond candidate availability.")
    (out / "nonholder_sources_conclusions.json").write_text(json.dumps(result, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps({"validation": {k:v for k,v in result["validation"].items() if not k.endswith("sha256")}, "results": summary}, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
