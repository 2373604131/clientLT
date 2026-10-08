"""Recompute the setting/forgetting evidence from raw local experiment files."""
from pathlib import Path
import csv
import hashlib
import json
import subprocess
import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
DATA = HERE / "data"
DATA.mkdir(exist_ok=True)
BASE = ROOT / "output/la_control_js_topology_analysis/la_control/seed42"
DIR = ROOT / "output/la_control_standard_dirichlet_e3_j_s_analysis/la_control/seed42"

def digest(p):
    return hashlib.sha256(p.read_bytes()).hexdigest()

def read_json(p):
    return json.loads(p.read_text(encoding="utf-8-sig"))

def get_run(topology, method):
    base = DIR if topology == "noniid-labeldir-fine" else BASE
    suffix = "tau1_a1_protocol42_beta0.5" if base == DIR else "tau1_a1_protocol42"
    return base / topology / method / suffix

rows, curves, runs, counts = [], [], {}, {}
for topology in ("client-longtail", "noniid-labeldir-fine", "matched-dirichlet"):
    for method in ("j", "s", "e3"):
        path = get_run(topology, method)
        metrics = pd.read_csv(path / "round_metrics.csv")
        config = read_json(path / "control_config.json")
        complete = read_json(path / "completion.json")
        manifest = pd.read_csv(path / "partition_manifest.csv")
        matrix_df = pd.read_csv(path / "client_class_counts.csv")
        matrix = matrix_df[[f"class_{i}" for i in range(100)]].to_numpy(dtype=int)
        reconstructed = pd.crosstab(manifest.client_id, manifest.class_id).reindex(
            index=range(30), columns=range(100), fill_value=0).to_numpy()
        assert np.array_equal(matrix, reconstructed), path
        assert len(manifest) == 10847 and manifest.raw_sample_id.is_unique, path
        assert metrics["round"].tolist() == list(range(101)), path
        assert config["tail_ids"] == list(range(80, 100)), path
        pool = sorted(zip(manifest.raw_sample_id.astype(int), manifest.class_id.astype(int)))
        pool_sha = hashlib.sha256(json.dumps(pool).encode()).hexdigest()
        tail = metrics.bottom20_tail_acc
        peak_i = tail.idxmax()
        record = {
            "partition": topology, "method": method, "seed": 42,
            "initial_tail": float(tail.iloc[0]),
            "peak_tail": float(tail.loc[peak_i]),
            "peak_round": int(metrics.loc[peak_i, "round"]),
            "final_tail": float(tail.iloc[-1]),
            "peak_to_final_pp": float(tail.max() - tail.iloc[-1]),
            "last20_tail": float(tail.iloc[-20:].mean()),
            "last20_overall": float(metrics.overall_acc.iloc[-20:].mean()),
            "source": path.relative_to(ROOT).as_posix(),
        }
        rows.append(record)
        c = metrics[["round", "bottom20_tail_acc", "overall_acc"]].copy()
        c.insert(0, "method", method)
        c.insert(0, "partition", topology)
        curves.append(c)
        runs[f"{topology}/{method}"] = {
            "source": record["source"], "completion": complete,
            "config": config, "same_pool_sha256": pool_sha,
            "files": {f: digest(path / f) for f in
                      ("round_metrics.csv", "partition_manifest.csv", "client_class_counts.csv")},
            "protocol_files": sorted(p.name for p in (path / "protocol").glob("*") if p.is_file()),
        }
        if topology in counts:
            assert np.array_equal(counts[topology], matrix), (topology, method)
        else:
            counts[topology] = matrix
            matrix_df.to_csv(DATA / f"counts_{topology}.csv", index=False)

assert len({r["same_pool_sha256"] for r in runs.values()}) == 1
common_fields = (
    "seed", "protocol_seed", "la_tau", "a_lr_mult", "extra_a_lr", "extra_b_lr",
    "normal_trainable_factor", "normal_a_lr", "normal_a_weight_decay",
    "normal_b_lr", "normal_b_weight_decay", "extra_weight_decay", "aggregation",
    "control_enabled", "extra_trainable_factor", "candidate_rounds", "tail_ids",
    "rng_protocol", "primary_endpoint")
config_checks = {}
for method in ("j", "s", "e3"):
    configs = [runs[f"{top}/{method}"]["config"] for top in counts]
    mismatches = {k: [c.get(k) for c in configs] for k in common_fields
                  if len({json.dumps(c.get(k), sort_keys=True) for c in configs}) > 1}
    config_checks[method] = {"mismatches": mismatches,
        "optimizer_steps_by_partition": {
            top: runs[f"{top}/{method}"]["config"].get("normal_steps_expected", 0)
                 + runs[f"{top}/{method}"]["config"].get("extra_steps_expected", 0)
            for top in counts}}

structure = []
for top, a in counts.items():
    n = a.sum(axis=0)
    tail_by_client = a[:, 80:].sum(axis=1)
    tail_n = int(tail_by_client.sum())
    structure.append({
        "partition": top, "samples": int(a.sum()), "tail_samples": tail_n,
        "client_min": int(a.sum(axis=1).min()), "client_max": int(a.sum(axis=1).max()),
        "tail_samples_in_clients_27_29": int(tail_by_client[27:].sum()),
        "tail_share_clients_27_29": float(tail_by_client[27:].sum()/tail_n),
        "largest3_tail_share": float(np.sort(tail_by_client)[-3:].sum()/tail_n),
        "clients_with_any_tail": int((tail_by_client > 0).sum()),
        "mean_clients_per_tail_class": float((a[:, 80:] > 0).sum(axis=0).mean()),
        "mean_effective_clients_per_tail_class":
            float((1/np.square(a[:, 80:]/n[None,80:]).sum(axis=0)).mean()),
        "selected_group_total_samples": int(a[27:].sum()),
        "selected_group_fedavg_weight": float(a[27:].sum()/a.sum()),
    })
pd.DataFrame(rows).to_csv(DATA / "retention_summary.csv", index=False)
pd.concat(curves).to_csv(DATA / "training_curves.csv", index=False)
pd.DataFrame(structure).to_csv(DATA / "partition_summary.csv", index=False)

# Inventory parameters without reading event tensors/checkpoints or running training.
cmd = ["rg", "-l", "--glob", "*.json", "--glob", "!**/events/**",
       "--glob", "!**/checkpoints/**", "--glob", "!**/model/**",
       '"imb_factor"|"imbalance_factor"|"specialization_lambda"|"lambda_values"|"IF_values"|"if_values"', "output"]
found = subprocess.run(cmd, cwd=ROOT, capture_output=True, text=True, encoding="utf-8")
assert found.returncode in (0, 1), found.stderr
def param_values(obj):
    out = {}
    def visit(x, prefix=""):
        if isinstance(x, dict):
            for k, v in x.items():
                if k in ("imb_factor", "imbalance_factor", "specialization_lambda",
                         "intra_group_alpha", "head_leakage_scale", "partition",
                         "topology", "seed", "protocol_seed", "lora_rank"):
                    if not isinstance(v, (dict, list)):
                        out[prefix+k] = v
                if k in ("lambda_values", "IF_values", "if_values", "alpha_values"):
                    out[prefix+k] = v
                if isinstance(v, (dict, list)):
                    visit(v, prefix+k+".")
        elif isinstance(x, list) and len(x) < 200:
            for i, v in enumerate(x):
                if isinstance(v, dict):
                    visit(v, prefix+str(i)+".")
    visit(obj)
    return out

inventory = []
for relative in found.stdout.splitlines():
    p = ROOT / relative
    if p.stat().st_size > 3_000_000:
        continue
    try:
        params = param_values(read_json(p))
    except (ValueError, UnicodeError):
        continue
    if params:
        inventory.append({"path": p.relative_to(ROOT).as_posix(), "parameters": params,
                          "round_metrics_here": (p.parent / "round_metrics.csv").exists(),
                          "finished_here": (p.parent / "finished.flag").exists(),
                          "completion_here": (p.parent / "completion.json").exists()})
(DATA / "parameter_inventory.json").write_text(
    json.dumps(inventory, ensure_ascii=False, indent=2), encoding="utf-8")
audit = {"date": "2026-10-08", "runs": runs, "same_global_samples_and_labels": True,
         "config_checks": config_checks, "structure": structure, "retention": rows,
         "inventory_files": len(inventory),
         "scope": "Local returned files only. Config inventory is not proof that a training run completed.",
         "limitations": [
             "One split seed and one training seed in these nine runs.",
             "J is joint A/B training plus nine extra A updates; it is not plain FedAvg.",
             "Same local epochs, not identical optimizer-step counts across ordinary Dirichlet and Client-LT.",
             "Same-pool and recorded-config checks do not establish identical code binaries or initialization tensors.",
             "Peak-to-final accuracy decrease is an aggregate forgetting proxy, not individual prediction transitions."]}
(DATA / "audit.json").write_text(json.dumps(audit, ensure_ascii=False, indent=2), encoding="utf-8")
print(pd.DataFrame(rows)[["partition", "method", "peak_tail", "final_tail", "peak_to_final_pp", "last20_tail"]].to_string(index=False))
print(pd.DataFrame(structure).to_string(index=False))
print("CONFIG_CHECKS", json.dumps(config_checks))
print("PARAMETER_INVENTORY_FILES", len(inventory))
print("IMB_VALUES", sorted({str(v) for r in inventory for k,v in r["parameters"].items() if k.endswith("imb_factor")}))
