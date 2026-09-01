from __future__ import annotations

import ast
import csv
import json
import math
import re
import shlex
from collections import defaultdict
from pathlib import Path

from tools.semantic_acquisition.common import write_csv, write_json


TAIL_RE = re.compile(r"Tail accuracy \(bottom 20 classes\):\s*([0-9.]+)%")
EPOCH_LIST_RE = re.compile(r"Global Epoch List:\s*\n(\[[^\n]+\])")
COMMAND_RE = re.compile(r"(?:CUDA_VISIBLE_DEVICES=\S+\s+)?python\s+[^\n]*federated_main\.py[^\n]*")
PATH_FIELDS = {
    "seed": r"(?:^|_)seed=([^_/]+)",
    "local_epochs": r"(?:^|_)localE=([^_/]+)",
    "specialization_lambda": r"lambda=([^_/]+)",
    "intra_group_alpha": r"alpha=([^_/]+)",
    "head_leakage_scale": r"rho=([^_/]+)",
    "imb_factor": r"IF=([^_/]+)",
}


def _flag(tokens: list[str], name: str, default=""):
    try:
        position = tokens.index(name)
    except ValueError:
        return default
    return tokens[position + 1] if position + 1 < len(tokens) else default


def _float(value, default=""):
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def _int(value, default=""):
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def _path_value(path: Path, field: str):
    match = re.search(PATH_FIELDS[field], path.as_posix())
    return match.group(1) if match else ""


def _carrier(trainer: str, aggregation: str, text: str) -> str:
    lowered = f"{trainer} {aggregation} {text[:4000]}".lower()
    if "online_class_separable" in lowered or "residual_fedavg" in lowered:
        return "shared vision LoRA + class residual"
    if "promptfl" in lowered:
        return "global/general + class-aware prompt context"
    if "cliplora" in lowered or "lora" in lowered:
        return "shared vision LoRA"
    if "capt" in lowered:
        return "CAPT prompt carrier"
    return "unknown (manual review required)"


def _parse_log(path: Path) -> dict | None:
    text = path.read_text(encoding="utf-8", errors="replace")
    commands = COMMAND_RE.findall(text)
    command = commands[0].strip() if commands else ""
    try:
        tokens = shlex.split(command, posix=True) if command else []
    except ValueError:
        tokens = command.split()
    frac = _float(_flag(tokens, "--frac", ""), math.nan)
    if not math.isfinite(frac) or abs(frac - 1.0) > 1e-12:
        return None
    partition = _flag(tokens, "--partition", "")
    if partition != "client-longtail":
        return None
    tails = [float(value) for value in TAIL_RE.findall(text)]
    if not tails:
        return None
    epoch_match = EPOCH_LIST_RE.search(text)
    epochs = []
    if epoch_match:
        try:
            epochs = [int(value) for value in ast.literal_eval(epoch_match.group(1))]
        except (SyntaxError, ValueError):
            epochs = []
    trainer = _flag(tokens, "--trainer", "")
    aggregation = _flag(tokens, "--aggregation", _flag(tokens, "--model", ""))
    output_dir = _flag(tokens, "--output-dir", "")
    seed = _int(_flag(tokens, "--seed", _path_value(path, "seed")))
    local_epochs = _int(_flag(tokens, "--local_epochs", _path_value(path, "local_epochs")))
    row = {
        "source_log": str(path.resolve()),
        "declared_output_dir": output_dir,
        "model": _flag(tokens, "--config-file", "").split("/")[-1].replace(".yaml", ""),
        "trainer": trainer,
        "parameter_carrier": _carrier(trainer, aggregation, text),
        "partition": partition,
        "num_users": _int(_flag(tokens, "--num_users", "")),
        "frac": frac,
        "local_epochs": local_epochs,
        "seed": seed,
        "rounds_configured": _int(_flag(tokens, "--round", "")),
        "eval_points": len(tails),
        "final_tail_acc": tails[-1],
        "best_tail_acc": max(tails),
        "best_tail_eval_index": int(max(range(len(tails)), key=tails.__getitem__)),
        "best_tail_round": "",
        "best_to_final_drop_pp": max(tails) - tails[-1],
        "per_round_trajectory_available": len(tails) > 1,
        "trajectory_storage": "embedded_in_run_log",
        "metric_source": "Tail accuracy (bottom 20 classes)",
        "client_schedule_file": _flag(tokens, "--client_schedule_file", ""),
        "split_seed": _int(_flag(tokens, "--split_seed", "")),
        "specialization_lambda": _float(_flag(tokens, "--specialization_lambda", _path_value(path, "specialization_lambda"))),
        "intra_group_alpha": _float(_flag(tokens, "--intra_group_alpha", _path_value(path, "intra_group_alpha"))),
        "head_leakage_scale": _float(_flag(tokens, "--head_leakage_scale", _path_value(path, "head_leakage_scale"))),
        "head_client_ratio": _float(_flag(tokens, "--head_client_ratio", "")),
        "tail_client_ratio": _float(_flag(tokens, "--tail_client_ratio", "")),
        "head_class_ratio": _float(_flag(tokens, "--head_class_ratio", "")),
        "tail_class_ratio": _float(_flag(tokens, "--tail_class_ratio", "")),
        "imb_factor": _float(_flag(tokens, "--imb_factor", _path_value(path, "imb_factor"))),
        "causal_status": "descriptive_frac1_only_not_comparable_to_current_sca",
        "command": command,
    }
    best_index = int(row["best_tail_eval_index"])
    if len(epochs) == len(tails):
        row["best_tail_round"] = epochs[best_index] + 1
    return row


def _deduplicate(rows: list[dict]) -> list[dict]:
    groups = defaultdict(list)
    for row in rows:
        key = row["command"] or (
            row["trainer"], row["seed"], row["local_epochs"],
            row["specialization_lambda"], row["intra_group_alpha"],
        )
        groups[str(key)].append(row)
    output = []
    for duplicates in groups.values():
        preferred = sorted(
            duplicates,
            key=lambda row: ("/cifar100_LT/" not in row["source_log"].replace("\\", "/"), row["source_log"]),
        )[0]
        preferred = dict(preferred)
        preferred["duplicate_log_count"] = len(duplicates)
        preferred["duplicate_logs"] = json.dumps(
            [row["source_log"] for row in duplicates], ensure_ascii=False
        )
        output.append(preferred)
    return sorted(output, key=lambda row: (
        str(row["trainer"]), str(row["local_epochs"]),
        str(row["intra_group_alpha"]), str(row["seed"]),
    ))


def run(output_root: Path, audit_dir: Path) -> dict:
    output_root, audit_dir = Path(output_root), Path(audit_dir)
    audit_dir.mkdir(parents=True, exist_ok=True)
    candidates = list(output_root.rglob("run.log")) + list(output_root.rglob("log.txt"))
    parsed = []
    failures = []
    for path in sorted(set(candidates)):
        try:
            row = _parse_log(path)
            if row is not None:
                parsed.append(row)
        except Exception as exc:  # malformed legacy logs must not abort the audit
            failures.append({"path": str(path.resolve()), "error": f"{type(exc).__name__}: {exc}"})
    rows = _deduplicate(parsed)
    write_csv(audit_dir / "frac1_run_inventory.csv", rows)
    write_csv(audit_dir / "parse_failures.csv", failures)
    grouped = defaultdict(list)
    for row in rows:
        key = (
            row["trainer"], row["parameter_carrier"], row["num_users"], row["local_epochs"],
            row["specialization_lambda"], row["intra_group_alpha"], row["head_leakage_scale"],
        )
        grouped[key].append(row)
    groups = []
    for key, values in sorted(grouped.items(), key=lambda item: str(item[0])):
        finals = [float(row["final_tail_acc"]) for row in values]
        drops = [float(row["best_to_final_drop_pp"]) for row in values]
        groups.append({
            "trainer": key[0], "parameter_carrier": key[1], "num_users": key[2],
            "local_epochs": key[3], "specialization_lambda": key[4],
            "intra_group_alpha": key[5], "head_leakage_scale": key[6],
            "seeds": sorted(set(row["seed"] for row in values)),
            "run_count": len(values), "mean_final_tail_acc": sum(finals) / len(finals),
            "min_final_tail_acc": min(finals), "max_final_tail_acc": max(finals),
            "mean_best_to_final_drop_pp": sum(drops) / len(drops),
        })
    summary = {
        "schema_version": "frac1_legacy_audit_v1",
        "training_performed": False,
        "scanned_root": str(output_root.resolve()),
        "eligible_unique_runs": len(rows),
        "configuration_groups": groups,
        "parse_failure_count": len(failures),
        "interpretation": (
            "Descriptive audit only. Stability across these frac=1.0 runs may support the existence "
            "of weak tail performance or late decline, but cannot identify full-participation "
            "crowding without an architecture- and schedule-matched frac<1 comparison."
        ),
    }
    write_json(audit_dir / "p0_summary.json", summary)
    lines = [
        "# Phase 0 — legacy `frac=1.0` audit", "",
        f"- Unique eligible Client-LT runs: **{len(rows)}**",
        f"- Configuration groups: **{len(groups)}**",
        "- Training performed: **no**", "",
        "This is a descriptive inventory, not a causal comparison with the current SCA runs.", "",
        "| trainer | carrier | users | local E | lambda | alpha | seeds | final tail mean | best→final drop |",
        "|---|---|---:|---:|---:|---:|---|---:|---:|",
    ]
    for group in groups:
        lines.append(
            f"| {group['trainer']} | {group['parameter_carrier']} | {group['num_users']} | "
            f"{group['local_epochs']} | {group['specialization_lambda']} | {group['intra_group_alpha']} | "
            f"{group['seeds']} | {group['mean_final_tail_acc']:.3f} | "
            f"{group['mean_best_to_final_drop_pp']:.3f} |"
        )
    lines += ["", "Causal boundary: a matched `frac<1` run is still required to attribute any gap to full-participation crowding."]
    (audit_dir / "p0_report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return summary
