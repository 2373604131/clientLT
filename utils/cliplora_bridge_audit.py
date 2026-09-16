"""Read-only, phase-aware observations for the C1/C2 topology bridge."""

import csv
import hashlib
import json
import os
import platform
from pathlib import Path

import numpy as np
import torch

from utils.cliplora_a_refresh import state_hash, update_diagnostics


def write_json(path, value):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(value, indent=2, ensure_ascii=False), encoding="utf-8")


def write_csv(path, rows):
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with Path(path).open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)


def json_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def copy_state(state, keys):
    return {k: state[k].detach().cpu().clone() for k in keys}


class BridgeAudit:
    def __init__(self, trainer, cfg, args, runtime, schedule):
        from tools.eri_closure.protocol import load_protocol
        from utils.functional_coverage_validation import _TrainOnlyCifar100, _exact_lt_raw_ids, _locate_cifar100

        self.root = Path(args.output_dir)
        self.runtime = runtime
        self.keys = sorted(runtime.a_keys + runtime.b_keys)
        self.a_keys = runtime.a_keys
        dataset = trainer.dm.dataset
        store = _TrainOnlyCifar100(_locate_cifar100(Path(args.root)))
        raw_ids = _exact_lt_raw_ids(store.labels, args.imb_factor, args.imb_type).tolist()
        # The data loader keeps the class-grouped LT subset in this exact order.
        pool = dataset.train_x
        assert len(raw_ids) == len(pool)
        for raw_id, item in zip(raw_ids, pool):
            assert int(store.labels[raw_id]) == int(item.label)
            assert np.array_equal(store.images[raw_id], np.asarray(item.data))
        positions = {id(item): i for i, item in enumerate(pool)}
        rows = [{"raw_sample_id": raw_ids[positions[id(item)]], "class_id": int(item.label),
                 "client_id": k, "local_position": j}
                for k, items in enumerate(dataset.federated_train_x) for j, item in enumerate(items)]
        assert sorted(r["raw_sample_id"] for r in rows) == sorted(raw_ids)
        write_csv(self.root / "partition_manifest.csv", rows)
        self.counts = torch.zeros((args.num_users, 100), dtype=torch.long)
        for row in rows:
            self.counts[row["client_id"], row["class_id"]] += 1
        self.sizes = self.counts.sum(1)
        protocol, probes = load_protocol(self.root / "protocol")
        assert not (set(raw_ids) & {int(r["raw_train_index"]) for r in probes})
        probe_digest = hashlib.sha256()
        for row in probes:
            probe_digest.update(store.images[int(row["raw_train_index"])].tobytes())
        test_digest = hashlib.sha256()
        for item in dataset.data_test:
            test_digest.update(str(int(item.label)).encode())
            test_digest.update(np.asarray(item.data).tobytes())
        state = trainer.model.state_dict()
        frozen_keys = sorted(set(state) - set(self.keys))
        repo = Path(__file__).resolve().parents[1]
        code_files = ["federated_main.py", "trainers/cliplora.py", "utils/datasplit.py",
                      "utils/loralib/layers.py", "utils/loralib/utils.py", "utils/lora_aggregation.py",
                      "utils/cliplora_loss.py", "utils/cliplora_a_refresh.py", "utils/cliplora_bridge_audit.py"]
        self.meta = {
            "schema_version": "a_refresh_bridge_v1", "seed": args.seed,
            "topology": args.partition, "method": args.a_refresh_variant,
            "resolved_args": vars(args), "resolved_config": str(cfg),
            "classnames": list(dataset.classnames),
            "initial_lora_sha256": state_hash(state, self.keys),
            "frozen_model_sha256": state_hash(state, frozen_keys),
            "schedule_sha256": json_hash(schedule), "schedule": schedule,
            "pool_sha256": json_hash(sorted(raw_ids)),
            "test_sha256": test_digest.hexdigest(), "probe_images_sha256": probe_digest.hexdigest(),
            "probe_manifest_sha256": hashlib.sha256((self.root / "protocol/probe_manifest.csv").read_bytes()).hexdigest(),
            "protocol": protocol, "client_sample_counts": self.sizes.tolist(),
            "training_code_hashes": {name: hashlib.sha256((repo/name).read_text(encoding="utf-8").encode()).hexdigest() for name in code_files},
            "environment": {"python": platform.python_version(), "torch": str(torch.__version__),
                "cuda": torch.version.cuda, "cudnn": torch.backends.cudnn.version(),
                "visible_devices": os.environ.get("CUDA_VISIBLE_DEVICES"),
                "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None},
        }
        write_json(self.root / "bridge_metadata.json", self.meta)

    def normal(self, before, after, local_states, selected, weights, round_id, factor="B"):
        keys = self.keys if factor == "AB" else self.runtime.b_keys
        # Preserve exact local parameters too: CPU (B0 + rounded dB) is not
        # guaranteed to recover the original GPU float32 local state bitwise.
        locals_cpu = {int(k): copy_state(local_states[k], keys) for k in selected}
        deltas = {int(k): {key: (local_states[k][key] - before[key]).detach().cpu().clone()
                           for key in keys} for k in selected}
        self.save(before, after, deltas, selected, weights, round_id, f"normal_{factor}", locals_cpu)

    def event_directory(self, round_id, phase):
        return self.root / "bridge_dumps" / f"round_{round_id:03d}" / phase

    def save(self, before, after, deltas, selected, weights, round_id, phase, local_states=None):
        selected = list(map(int, selected))
        normal = phase in ("normal_B", "normal_AB")
        factor = "AB" if phase == "normal_AB" else ("A" if phase == "refresh_A" else "B")
        active = self.keys if factor == "AB" else (self.runtime.a_keys if factor == "A" else self.runtime.b_keys)
        anchor, endpoint = copy_state(before, self.keys), copy_state(after, self.keys)
        reconstructed = copy_state(anchor, self.keys)
        for key in active:
            value = torch.zeros_like(anchor[key])
            for k in selected:
                source = local_states[k][key] if normal else deltas[k][key]
                value.add_(source, alpha=float(weights[k]))
            reconstructed[key] = value if normal else anchor[key] + value
        error = max(float((reconstructed[k].double()-endpoint[k].double()).abs().max()) for k in self.keys)
        frozen = sorted(set(self.keys)-set(active))
        frozen_error = max((float((anchor[k]-endpoint[k]).abs().max()) for k in frozen), default=0.)
        norms = update_diagnostics(anchor, endpoint, self.a_keys, self.runtime.scaling, self.runtime.initial)
        record = {"seed": self.meta["seed"], "topology": self.meta["topology"],
                  "method": self.meta["method"], "round": round_id, "phase": phase,
                  "active_factor": factor, "reconstruction_max_abs_error": error,
                  "frozen_factor_max_abs_error": frozen_error,
                  "reconstruction_passed": error <= 1e-5 and frozen_error == 0,
                  "optimizer_steps": sum(int(np.ceil(int(self.sizes[k])/32)) for k in selected) * (3 if normal else 1),
                  "sample_presentations": sum(int(self.sizes[k]) for k in selected) * (3 if normal else 1),
                  **norms, **getattr(self, "event_context", {})}
        payload = {**record, "schema_version": "a_refresh_bridge_v1", "active_keys": active,
                   "anchor_lora_state": anchor, "actual_after_lora_state": endpoint,
                   "local_factor_deltas": [copy_state(deltas[k], active) for k in selected],
                   "selected_client_ids": selected,
                   "server_weights": torch.tensor([float(weights[k]) for k in selected], dtype=torch.float64),
                   "client_class_counts": self.counts[selected], "client_sample_counts": self.sizes[selected],
                   "initial_lora_state": self.runtime.initial}
        if local_states is not None:
            payload["local_factor_states"] = [local_states[k] for k in selected]
        root = self.event_directory(round_id, phase)
        root.mkdir(parents=True, exist_ok=True)
        torch.save(payload, root / "state.pt")
        write_json(root / "event.json", {**record, "selected_client_ids": selected,
            "server_weights": payload["server_weights"].tolist(),
            "before_lora_sha256": state_hash(anchor, self.keys),
            "after_lora_sha256": state_hash(endpoint, self.keys)})
        print(f"Bridge dump: round={round_id} phase={phase} reconstruction={error:.3g}", flush=True)
