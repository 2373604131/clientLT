#!/usr/bin/env python
"""Launch the preregistered ERI 2x2 experiment and its offline stages.

Stages are deliberately separable for Slurm arrays:
  protocol -> train -> verify -> analyze -> replay -> summary.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import shlex
import subprocess
import sys
from pathlib import Path

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.eri_closure.analysis import analyze_run
from tools.eri_closure.protocol import DEFAULT_AUDIT_ROUNDS, build_protocol, parse_eri_rounds
from tools.eri_closure.replay import replay_run
from tools.eri_closure.summary import summarize


CASES = (
    "clientlt_fedavg",
    "matched_dirichlet_fedavg",
    "clientlt_support_normalized",
    "matched_dirichlet_support_normalized",
)


def case_spec(case: str) -> tuple[str, str]:
    specs = {
        "clientlt_fedavg": ("client-longtail", "fedavg"),
        "matched_dirichlet_fedavg": ("matched-dirichlet", "fedavg"),
        "clientlt_support_normalized": ("client-longtail", "support_normalized"),
        "matched_dirichlet_support_normalized": ("matched-dirichlet", "support_normalized"),
    }
    try:
        return specs[case]
    except KeyError as error:
        raise ValueError(f"Unknown ERI case {case}") from error


def _validated_frac(frac: float) -> float:
    value = float(frac)
    if not math.isfinite(value) or not 0.0 < value <= 1.0:
        raise ValueError(f"--frac must be finite and in (0, 1], got {frac!r}")
    return value


def frac_tag(frac: float) -> str:
    value = _validated_frac(frac)
    return "frac" + format(value, ".12g").replace(".", "p")


def run_dir(root: Path, case: str, seed: int, frac: float = 1.0) -> Path:
    """Keep completed full-participation paths stable and isolate new fractions."""
    base = root if math.isclose(_validated_frac(frac), 1.0) else root / "fractions" / frac_tag(frac)
    return base / "runs" / case / f"seed{int(seed)}"


def schedule_file(
    root: Path,
    seed: int,
    frac: float = 1.0,
    users: int = 30,
    rounds: int = 100,
) -> Path:
    name = f"users{int(users)}_{frac_tag(frac)}_rounds{int(rounds)}_seed{int(seed)}.json"
    return root / "protocol" / "schedules" / name


def ensure_client_schedule(
    path: Path,
    *,
    rounds: int,
    users: int,
    frac: float,
    seed: int,
) -> None:
    """Write and validate a deterministic schedule shared by paired topologies."""
    frac = _validated_frac(frac)
    clients_per_round = max(int(frac * int(users)), 1)
    if path.exists():
        data = json.loads(path.read_text(encoding="utf-8"))
        schedule = data.get("schedule", data)
        if len(schedule) < int(rounds):
            raise ValueError(f"Existing schedule has fewer than {rounds} rounds: {path}")
        for round_index, row in enumerate(schedule[: int(rounds)], start=1):
            clients = [int(item) for item in row]
            if len(clients) != clients_per_round:
                raise ValueError(
                    f"Schedule round {round_index} has {len(clients)} clients, "
                    f"expected {clients_per_round}: {path}"
                )
            if len(set(clients)) != len(clients):
                raise ValueError(f"Schedule round {round_index} contains duplicates: {path}")
            if any(client_id < 0 or client_id >= int(users) for client_id in clients):
                raise ValueError(f"Schedule round {round_index} has invalid client ids: {path}")
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    if clients_per_round == int(users):
        schedule = [list(range(int(users))) for _ in range(int(rounds))]
    else:
        rng = np.random.default_rng(int(seed))
        schedule = [
            [int(item) for item in rng.choice(int(users), clients_per_round, replace=False)]
            for _ in range(int(rounds))
        ]
    payload = {
        "num_rounds": int(rounds), "num_users": int(users), "frac": frac,
        "clients_per_round": clients_per_round, "seed": int(seed),
        "schedule": schedule,
    }
    # The two same-node workers can reach this simultaneously. Both generate
    # identical content; unique temporary files plus replace avoid partial JSON.
    temporary = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    temporary.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


def ensure_full_schedule(path: Path, *, rounds: int, users: int, seed: int) -> None:
    """Backward-compatible wrapper used by existing tests and callers."""
    ensure_client_schedule(path, rounds=rounds, users=users, frac=1.0, seed=seed)


def build_command(args, case: str, seed: int) -> list[str]:
    partition, aggregation = case_spec(case)
    protocol_file = args.output_root / "protocol" / "eri_protocol.json"
    output = run_dir(args.output_root, case, seed, args.frac)
    frozen_schedule = schedule_file(
        args.output_root, seed, args.frac, args.num_users, args.rounds
    )
    command = [
        args.python_bin, "-u", "federated_main.py",
        "--root", str(args.data_root), "--model", "fedavg", "--trainer", "ClipLora",
        "--dataset", "cifar100_LT", "--seed", str(seed), "--split_seed", str(seed),
        "--num_users", str(args.num_users), "--frac", str(args.frac), "--round", str(args.rounds),
        "--local_epochs", str(args.local_epochs), "--client_schedule_seed", str(seed),
        "--client_schedule_file", str(frozen_schedule),
        "--isolate_local_optimizer_state", "True", "--federated_single_scheduler_step", "True",
        "--lr", str(args.lr), "--gamma", "1", "--n_ctx", "4", "--n_general", "1",
        "--ctx_init", "False", "--csc", "True",
        "--dataset-config-file", "configs/datasets/cifar100_LT.yaml",
        "--config-file", "configs/trainers/PromptFL/vit_b16.yaml",
        "--output-dir", str(output), "--imb_factor", str(args.imb_factor), "--imb_type", "exp",
        "--train_batch_size", str(args.train_batch_size), "--test_batch_size", str(args.test_batch_size),
        "--global_eval_interval", "1", "--num_classes", "100", "--tail_class_ratio", "0.2",
        "--head_client_ratio", "0.9", "--tail_client_ratio", "0.1", "--head_class_ratio", "0.8",
        "--partition", partition, "--beta", str(args.dirichlet_beta),
        "--specialization_lambda", str(args.specialization_lambda),
        "--intra_group_alpha", str(args.intra_group_alpha),
        "--head_leakage_scale", str(args.head_leakage_scale),
        "--encoder", "vision", "--cliplora_position", "top3", "--cliplora_rank", "2",
        "--cliplora_alpha", "1", "--cliplora_dropout_rate", "0.0", "--cliplora_params", "q", "v",
        "--cliplora_lr_policy", "constant", "--cliplora_precision", "fp32",
        "--cliplora_common_init_seed", str(seed), "--cliplora_aggregation", aggregation,
        "--experimentD_enable", "False", "--cliplora_sca_enable", "False", "--e1_enable", "False",
        "--stage3_enable", "False", "--eri_audit_enable", "True",
        "--eri_audit_rounds", args.audit_rounds, "--eri_protocol_file", str(protocol_file),
        "DATALOADER.NUM_WORKERS", str(args.num_workers),
    ]
    return command


def is_complete(path: Path, rounds: int) -> bool:
    return (path / "eri_closure" / "dumps" / f"round_{rounds:03d}" / "round_state.pt").is_file() and (
        path / "eri_closure" / "test_per_class_metrics.csv"
    ).is_file()


def _rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


def verify_fixed_marginals(args) -> Path:
    records = []
    case_pairs = (
        ("clientlt_fedavg", "matched_dirichlet_fedavg"),
        ("clientlt_support_normalized", "matched_dirichlet_support_normalized"),
    )
    for seed in args.seeds:
        for left_case, right_case in case_pairs:
            if left_case not in selected_cases(args) or right_case not in selected_cases(args):
                continue
            left = run_dir(args.output_root, left_case, seed, args.frac)
            right = run_dir(args.output_root, right_case, seed, args.frac)
            left_csv, right_csv = left / "client_class_counts.csv", right / "client_class_counts.csv"
            if not left_csv.exists() or not right_csv.exists():
                raise FileNotFoundError(f"Cannot verify fixed marginals; missing {left_csv} or {right_csv}")
            left_rows, right_rows = _rows(left_csv), _rows(right_csv)
            def margins(rows):
                fields = [key for key in rows[0] if key.startswith("class_")]
                client_totals = [sum(int(row[key]) for key in fields) for row in rows]
                global_totals = [sum(int(row[key]) for row in rows) for key in fields]
                return client_totals, global_totals
            left_client, left_global = margins(left_rows)
            right_client, right_global = margins(right_rows)
            if left_client != right_client or left_global != right_global:
                raise RuntimeError(
                    f"Fixed client/global margins differ for seed={seed}, {left_case} vs {right_case}; ERI H1 is invalid."
                )
            left_selected = left / "selected_clients.csv"
            right_selected = right / "selected_clients.csv"
            actual_schedule_verified = False
            if left_selected.exists() or right_selected.exists():
                if not left_selected.exists() or not right_selected.exists():
                    raise FileNotFoundError(
                        "Only one paired run has selected_clients.csv: "
                        f"{left_selected}, {right_selected}"
                    )
                left_selection = [
                    (int(row["epoch_index"]), int(row["client_id"]), int(row["selection_order"]))
                    for row in _rows(left_selected)
                ]
                right_selection = [
                    (int(row["epoch_index"]), int(row["client_id"]), int(row["selection_order"]))
                    for row in _rows(right_selected)
                ]
                if left_selection != right_selection:
                    raise RuntimeError(
                        f"Executed client schedules differ for seed={seed}, "
                        f"frac={args.frac}, {left_case} vs {right_case}"
                    )
                actual_schedule_verified = True
            records.append({
                "seed": seed,
                "frac": float(args.frac),
                "clients_per_round": max(int(float(args.frac) * int(args.num_users)), 1),
                "left_case": left_case,
                "right_case": right_case,
                "fixed_nk_and_nc_verified": True,
                "actual_selected_schedule_verified": actual_schedule_verified,
            })
    if not records:
        raise ValueError("Verification needs a complete Client-LT/matched-Dirichlet case pair")
    verification_name = (
        "fixed_marginal_verification.json"
        if math.isclose(float(args.frac), 1.0)
        else f"fixed_marginal_verification_{frac_tag(args.frac)}.json"
    )
    path = args.output_root / "protocol" / verification_name
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({"schema_version": "eri_fixed_marginal_v1", "records": records}, indent=2) + "\n", encoding="utf-8")
    return path


def selected_cases(args) -> list[str]:
    return list(args.cases) if args.case is None else [args.case]


def main() -> None:
    args = parse_args()
    args.output_root = args.output_root.resolve()
    parse_eri_rounds(args.audit_rounds, args.rounds)
    cases = selected_cases(args)
    if args.stage == "protocol":
        root = build_protocol(
            args.output_root / "protocol", data_root=args.data_root,
            samples_per_class=args.probes_per_class,
            audit_rounds=parse_eri_rounds(args.audit_rounds, args.rounds), overwrite=args.overwrite_protocol,
        )
        for seed in args.seeds:
            path = schedule_file(args.output_root, seed, args.frac, args.num_users, args.rounds)
            ensure_client_schedule(
                path, rounds=args.rounds, users=args.num_users, frac=args.frac, seed=seed
            )
        print(f"Frozen ERI protocol: {root}")
        return
    if args.stage == "train":
        protocol = args.output_root / "protocol" / "eri_protocol.json"
        if not protocol.exists():
            raise FileNotFoundError("Run --stage protocol before starting any training array")
        for seed in args.seeds:
            path = schedule_file(args.output_root, seed, args.frac, args.num_users, args.rounds)
            ensure_client_schedule(
                path, rounds=args.rounds, users=args.num_users, frac=args.frac, seed=seed
            )
            for case in cases:
                output = run_dir(args.output_root, case, seed, args.frac)
                command = build_command(args, case, seed)
                print(shlex.join(command), flush=True)
                if args.dry_run:
                    continue
                if output.exists() and any(output.iterdir()):
                    if args.skip_completed and is_complete(output, args.rounds):
                        print(f"Skipping complete run: {output}", flush=True); continue
                    raise FileExistsError(f"Refusing to append to non-empty run directory: {output}")
                env = os.environ.copy()
                if args.gpu is not None:
                    visible = [item for item in env.get("CUDA_VISIBLE_DEVICES", "").split(",") if item]
                    env["CUDA_VISIBLE_DEVICES"] = visible[int(args.gpu)] if visible and str(args.gpu).isdigit() and int(args.gpu) < len(visible) else str(args.gpu)
                subprocess.run(command, cwd=REPO_ROOT, env=env, check=True)
        return
    if args.stage == "verify":
        print(f"Fixed margins verified: {verify_fixed_marginals(args)}")
        return
    if args.stage == "analyze":
        for seed in args.seeds:
            for case in cases:
                print(analyze_run(run_dir(args.output_root, case, seed, args.frac), protocol_dir=args.output_root / "protocol", data_root=args.data_root, quadrature_points=args.quadrature_points, device=args.device))
        return
    if args.stage == "replay":
        fedavg_cases = [case for case in cases if case_spec(case)[1] == "fedavg"]
        for seed in args.seeds:
            for case in fedavg_cases:
                print(replay_run(run_dir(args.output_root, case, seed, args.frac), protocol_dir=args.output_root / "protocol", data_root=args.data_root, quadrature_points=args.quadrature_points, device=args.device, permutations=args.permutations))
        return
    if args.stage == "summary":
        print(summarize(args.output_root, maintenance_start_round=args.maintenance_start_round))
        return
    raise AssertionError(args.stage)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--stage", required=True, choices=["protocol", "train", "verify", "analyze", "replay", "summary"])
    parser.add_argument("--output-root", required=True, type=Path)
    parser.add_argument("--data-root", type=Path, default=Path("DATA"))
    parser.add_argument("--python-bin", default=sys.executable)
    parser.add_argument("--cases", nargs="+", choices=CASES, default=list(CASES))
    parser.add_argument("--case", choices=CASES, help="one case for an array worker; overrides --cases")
    parser.add_argument("--seeds", nargs="+", type=int, default=[1, 2, 3, 42, 2026])
    parser.add_argument("--frac", type=float, default=1.0)
    parser.add_argument("--rounds", type=int, default=100); parser.add_argument("--num-users", type=int, default=30)
    parser.add_argument("--local-epochs", type=int, default=3); parser.add_argument("--lr", type=float, default=0.001)
    parser.add_argument("--imb-factor", type=float, default=0.01); parser.add_argument("--dirichlet-beta", type=float, default=0.5)
    parser.add_argument("--specialization-lambda", type=float, default=0.75); parser.add_argument("--intra-group-alpha", type=float, default=0.5)
    parser.add_argument("--head-leakage-scale", type=float, default=3.0); parser.add_argument("--train-batch-size", type=int, default=32)
    parser.add_argument("--test-batch-size", type=int, default=64); parser.add_argument("--num-workers", type=int, default=8)
    parser.add_argument("--audit-rounds", default=",".join(map(str, DEFAULT_AUDIT_ROUNDS)))
    parser.add_argument("--probes-per-class", type=int, default=10); parser.add_argument("--quadrature-points", type=int, default=8)
    parser.add_argument("--permutations", type=int, default=20); parser.add_argument("--device", default=None)
    parser.add_argument("--maintenance-start-round", type=int, default=10)
    parser.add_argument("--gpu", default=None); parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--skip-completed", action="store_true"); parser.add_argument("--overwrite-protocol", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    main()
