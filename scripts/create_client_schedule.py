#!/usr/bin/env python
"""Create and validate a fixed federated client schedule without training."""

import argparse
import json
import os
from pathlib import Path

import numpy as np


def create_schedule(num_rounds, num_users, frac, seed):
    clients_per_round = max(int(float(frac) * int(num_users)), 1)
    rng = np.random.default_rng(int(seed))
    return [
        [int(x) for x in rng.choice(int(num_users), clients_per_round, replace=False).tolist()]
        for _ in range(int(num_rounds))
    ]


def validate_schedule(schedule, num_rounds, num_users, frac):
    clients_per_round = max(int(float(frac) * int(num_users)), 1)
    if len(schedule) != int(num_rounds):
        raise ValueError(f"schedule has {len(schedule)} rounds, expected {num_rounds}")
    for round_idx, clients in enumerate(schedule):
        if len(clients) != clients_per_round:
            raise ValueError(
                f"round {round_idx} has {len(clients)} clients, expected {clients_per_round}"
            )
        if len(set(int(x) for x in clients)) != len(clients):
            raise ValueError(f"round {round_idx} has duplicate clients: {clients}")
        if any(int(x) < 0 or int(x) >= int(num_users) for x in clients):
            raise ValueError(f"round {round_idx} has out-of-range clients: {clients}")


def load_schedule(path):
    with Path(path).open("r", encoding="utf-8") as f:
        payload = json.load(f)
    return payload.get("schedule", payload) if isinstance(payload, dict) else payload


def write_schedule_atomic(path, payload):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = path.with_name(f"{path.name}.tmp.{os.getpid()}")
    with tmp_path.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    os.replace(tmp_path, path)


def parse_args():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--path", required=True)
    parser.add_argument("--num_rounds", type=int, required=True)
    parser.add_argument("--num_users", type=int, required=True)
    parser.add_argument("--frac", type=float, required=True)
    parser.add_argument("--seed", type=int, required=True)
    return parser.parse_args()


def main():
    args = parse_args()
    path = Path(args.path)
    clients_per_round = max(int(float(args.frac) * int(args.num_users)), 1)

    if path.exists():
        schedule = load_schedule(path)
        validate_schedule(schedule, args.num_rounds, args.num_users, args.frac)
        print(f"Validated existing client schedule: {path}")
        return

    schedule = create_schedule(args.num_rounds, args.num_users, args.frac, args.seed)
    validate_schedule(schedule, args.num_rounds, args.num_users, args.frac)
    payload = {
        "num_rounds": int(args.num_rounds),
        "num_users": int(args.num_users),
        "frac": float(args.frac),
        "clients_per_round": int(clients_per_round),
        "seed": int(args.seed),
        "schedule": schedule,
    }
    write_schedule_atomic(path, payload)
    validate_schedule(load_schedule(path), args.num_rounds, args.num_users, args.frac)
    print(f"Created and validated client schedule: {path}")


if __name__ == "__main__":
    main()
