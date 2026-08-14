"""Run the deterministic, training-free P0/V1 context co-location audit.

The entry point reconstructs the repository's exact CIFAR-100-LT train pool,
creates matched Dirichlet and controlled Client-LT partitions, encodes only
class-name text with the frozen CLIP model, and writes auditable tables.  It
never imports a trainer or calls backward/optimizer/federated-round code.
"""

from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.util
import io
import json
import math
import pickle
import sys
import types
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch

from tools.analysis.context_colocation_metrics import (
    WEIGHTINGS,
    class_set_coverage,
    cluster_bootstrap,
    generate_frequency_matched_null_sets,
    generic_context_metrics,
    related_sample_metrics,
    stable_sha256,
    support_weights,
    topology_metrics,
)
from utils.datasplit import (
    partition_client_longtail_controlled,
    partition_fine_class_dirichlet,
)


ROOT = Path(__file__).resolve().parents[2]
DEFAULT_DATA = ROOT / "DATA" / "cifar-100" / "cifar-100-python"
DEFAULT_OUTPUT = ROOT / "output" / "p0_v1_context_colocation_v2"
DEFAULT_CLIP = Path.home() / ".cache" / "clip" / "ViT-B-16.pt"
TAIL_CLIENT_IDS = (27, 28, 29)


def sha256_bytes(*arrays: np.ndarray) -> str:
    digest = hashlib.sha256()
    for array in arrays:
        contiguous = np.ascontiguousarray(array)
        digest.update(str(contiguous.dtype).encode("ascii"))
        digest.update(np.asarray(contiguous.shape, dtype=np.int64).tobytes())
        digest.update(contiguous.tobytes())
    return digest.hexdigest()


def file_sha256(path: Path, chunk_size: int = 1024 * 1024) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        while True:
            chunk = handle.read(chunk_size)
            if not chunk:
                break
            digest.update(chunk)
    return digest.hexdigest()


def state_fingerprint(model) -> str:
    digest = hashlib.sha256()
    for name, tensor in sorted(model.state_dict().items()):
        digest.update(name.encode("utf-8"))
        value = tensor.detach().cpu().contiguous()
        digest.update(str(value.dtype).encode("ascii"))
        digest.update(np.asarray(value.shape, dtype=np.int64).tobytes())
        digest.update(value.numpy().tobytes())
    return digest.hexdigest()


def load_pickle(path: Path) -> dict:
    with path.open("rb") as handle:
        return pickle.load(handle, encoding="latin1")


def load_exact_cifar100_lt(data_dir: Path) -> tuple[np.ndarray, np.ndarray, list[str], np.ndarray]:
    """Recreate ``load_cifar100_LT_data`` labels and selected train indices."""
    train = load_pickle(data_dir / "train")
    test = load_pickle(data_dir / "test")
    meta = load_pickle(data_dir / "meta")
    raw_train_labels = np.asarray(train["fine_labels"], dtype=np.int64)
    test_labels = np.asarray(test["fine_labels"], dtype=np.int64)
    class_names = [str(name).replace("_", " ") for name in meta["fine_label_names"]]

    spec = importlib.util.spec_from_file_location(
        "repository_long_tail", ROOT / "datasets" / "long_tail.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    by_class = [np.flatnonzero(raw_train_labels == class_id).tolist() for class_id in range(100)]
    with contextlib.redirect_stdout(io.StringIO()):
        _, selected_by_class = module.train_long_tail(by_class, 100, 0.01, "exp")
    selected_raw_indices = np.asarray(module.flatten_list(selected_by_class), dtype=np.int64)
    train_labels = raw_train_labels[selected_raw_indices]
    return train_labels, test_labels, class_names, selected_raw_indices


def bottom_classes(labels: np.ndarray, count: int = 20) -> list[int]:
    """Recover the LT generator's least-frequent end, preserving index order.

    Realized integer counts tie at the boundary (classes 79 and 80 both have
    12 examples), so larger ids must be preferred for the bottom set.
    """
    counts = np.bincount(labels, minlength=100)
    return sorted(
        range(100), key=lambda class_id: (int(counts[class_id]), -class_id)
    )[:count]


def counts_matrix(labels: np.ndarray, partition: dict[int, np.ndarray], clients: int = 30) -> np.ndarray:
    output = np.zeros((clients, 100), dtype=np.int64)
    for client_id in range(clients):
        output[client_id] = np.bincount(labels[np.asarray(partition[client_id], dtype=np.int64)], minlength=100)
    return output


def partition_fingerprint(partition: dict[int, np.ndarray]) -> str:
    payload = {
        str(client_id): sorted(np.asarray(indices, dtype=np.int64).tolist())
        for client_id, indices in sorted(partition.items())
    }
    return stable_sha256(payload)


def build_non_tail_quintiles(global_counts: np.ndarray, non_tail: list[int]) -> dict[int, int]:
    ordered = sorted(non_tail, key=lambda class_id: (int(global_counts[class_id]), class_id))
    groups = np.array_split(np.asarray(ordered, dtype=np.int64), 5)
    assert all(len(group) == 16 for group in groups)
    return {int(class_id): quintile for quintile, group in enumerate(groups) for class_id in group}


def _patch_jit_for_cpu(model) -> None:
    device_holder = torch.jit.trace(lambda: torch.ones([]).to(torch.device("cpu")), example_inputs=[])
    device_node = [
        node
        for node in device_holder.graph.findAllNodes("prim::Constant")
        if "Device" in repr(node)
    ][-1]

    def constant_value(node):
        try:
            return node.output().toIValue()
        except Exception:
            try:
                return node["value"]
            except Exception:
                return None

    def patch_device(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []
        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)
        for graph in graphs:
            for node in graph.findAllNodes("prim::Constant"):
                if "value" in node.attributeNames() and str(constant_value(node)).startswith("cuda"):
                    node.copyAttributes(device_node)

    model.apply(patch_device)
    patch_device(model.encode_image)
    patch_device(model.encode_text)

    float_holder = torch.jit.trace(lambda: torch.ones([]).float(), example_inputs=[])
    float_node = list(float_holder.graph.findNode("aten::to").inputs())[1].node()

    def patch_float(module):
        try:
            graphs = [module.graph] if hasattr(module, "graph") else []
        except RuntimeError:
            graphs = []
        if hasattr(module, "forward1"):
            graphs.append(module.forward1.graph)
        for graph in graphs:
            for node in graph.findAllNodes("aten::to"):
                inputs = list(node.inputs())
                for position in (1, 2):
                    if position < len(inputs) and inputs[position].node().kind() == "prim::Constant":
                        if constant_value(inputs[position].node()) == 5:
                            inputs[position].node().copyAttributes(float_node)

    model.apply(patch_float)
    patch_float(model.encode_image)
    patch_float(model.encode_text)
    model.float()


def _tokenizer():
    # The repository tokenizer uses ftfy only for free-form Unicode cleanup.
    # CIFAR-100 canonical names and the fixed template are ASCII, so an identity
    # fallback is exact for this input if the optional ftfy package is absent.
    if "ftfy" not in sys.modules:
        try:
            import ftfy  # noqa: F401
        except ModuleNotFoundError:
            sys.modules["ftfy"] = types.SimpleNamespace(fix_text=lambda text: text)
    spec = importlib.util.spec_from_file_location(
        "repository_clip_tokenizer", ROOT / "clip" / "simple_tokenizer.py"
    )
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module.SimpleTokenizer()


def tokenize(prompts: list[str]) -> torch.Tensor:
    tokenizer = _tokenizer()
    sot = tokenizer.encoder["<|startoftext|>"]
    eot = tokenizer.encoder["<|endoftext|>"]
    result = torch.zeros((len(prompts), 77), dtype=torch.long)
    for row, prompt in enumerate(prompts):
        tokens = [sot] + tokenizer.encode(prompt) + [eot]
        if len(tokens) > 77:
            raise RuntimeError(f"Prompt exceeds CLIP context length: {prompt}")
        result[row, : len(tokens)] = torch.tensor(tokens, dtype=torch.long)
    return result


def encode_class_text(class_names: list[str], clip_path: Path) -> tuple[np.ndarray, dict]:
    model = torch.jit.load(str(clip_path), map_location="cpu").eval()
    _patch_jit_for_cpu(model)
    before = state_fingerprint(model)
    prompts = [f"a photo of a {name}." for name in class_names]
    tokens = tokenize(prompts)
    with torch.no_grad():
        features = model.encode_text(tokens).float()
        features = features / features.norm(dim=-1, keepdim=True)
    after = state_fingerprint(model)
    if before != after:
        raise RuntimeError("Frozen CLIP parameter fingerprint changed during text encoding")
    embeddings = features.cpu().numpy().astype(np.float64)
    if not np.isfinite(embeddings).all():
        raise RuntimeError("Non-finite CLIP text features")
    norms = np.linalg.norm(embeddings, axis=1)
    if not np.allclose(norms, 1.0, atol=1e-5):
        raise RuntimeError("CLIP text features are not unit-normalized")
    return embeddings, {
        "encoder_id": "ViT-B/16",
        "checkpoint": str(clip_path.resolve()),
        "checkpoint_sha256": file_sha256(clip_path),
        "template": "a photo of a {class_name}.",
        "tokenizer": str((ROOT / "clip" / "simple_tokenizer.py").resolve()),
        "feature_shape": list(embeddings.shape),
        "parameter_fingerprint_before": before,
        "parameter_fingerprint_after": after,
        "no_grad": True,
    }


def topk_non_tail(similarity: np.ndarray, tail_class: int, non_tail: list[int], k: int) -> list[int]:
    return sorted(non_tail, key=lambda class_id: (-float(similarity[tail_class, class_id]), class_id))[:k]


def topk_all_other(similarity: np.ndarray, tail_class: int, k: int) -> list[int]:
    candidates = [class_id for class_id in range(100) if class_id != tail_class]
    return sorted(candidates, key=lambda class_id: (-float(similarity[tail_class, class_id]), class_id))[:k]


def partition_invariant_row(
    seed: int,
    topology: str,
    labels: np.ndarray,
    partition: dict[int, np.ndarray],
    tail: list[int],
    universe_fingerprint: str,
) -> dict:
    merged = np.concatenate([np.asarray(partition[c], dtype=np.int64) for c in range(30)])
    matrix = counts_matrix(labels, partition)
    tail_set = set(tail)
    tail_in_specialists = int(matrix[list(TAIL_CLIENT_IDS)][:, tail].sum())
    tail_in_ordinary = int(matrix[:27, tail].sum())
    companion_by_specialist = matrix[list(TAIL_CLIENT_IDS)][:, [c for c in range(100) if c not in tail_set]].sum(axis=1)
    tail_by_specialist = matrix[list(TAIL_CLIENT_IDS)][:, tail].sum(axis=1)
    purities = tail_by_specialist / np.maximum(tail_by_specialist + companion_by_specialist, 1)
    return {
        "seed": seed,
        "topology": topology,
        "global_universe_fingerprint": universe_fingerprint,
        "partition_fingerprint": partition_fingerprint(partition),
        "assigned_total": int(merged.size),
        "unique_assigned": int(np.unique(merged).size),
        "global_total": int(labels.size),
        "coverage_exact": bool(np.array_equal(np.sort(merged), np.arange(labels.size))),
        "class_counts_conserved": bool(np.array_equal(matrix.sum(axis=0), np.bincount(labels, minlength=100))),
        "tail_samples_total": int(np.isin(labels, tail).sum()),
        "tail_samples_in_specialists": tail_in_specialists if topology == "ClientLT-controlled" else np.nan,
        "tail_samples_in_ordinary": tail_in_ordinary if topology == "ClientLT-controlled" else np.nan,
        "specialist_companion_total": int(companion_by_specialist.sum()) if topology == "ClientLT-controlled" else np.nan,
        "specialist_purity_min": float(purities.min()) if topology == "ClientLT-controlled" else np.nan,
        "role_metrics_applicable": topology == "ClientLT-controlled",
    }


def source_line(path: Path, needle: str) -> int | None:
    for number, line in enumerate(path.read_text(encoding="utf-8", errors="replace").splitlines(), 1):
        if needle in line:
            return number
    return None


def historical_round_facts() -> list[dict]:
    facts = []
    base = ROOT / "output" / "cifar100_LT" / "ClipLora_SupportNormalized_2x2_seed42" / "seed42"
    for topology, relative in (("Client-LT legacy", "clientlt/fedavg"), ("Dirichlet", "dirichlet/fedavg")):
        path = base / relative / "round_metrics.csv"
        if not path.exists():
            continue
        frame = pd.read_csv(path)
        if "bottom20_tail_acc" not in frame:
            continue
        for label, row in (("zero_shot", frame.iloc[0]), ("round_0", frame.loc[frame["epoch"] == 0].iloc[0]), ("final", frame.iloc[-1])):
            facts.append(
                {
                    "topology": topology,
                    "checkpoint": label,
                    "epoch": int(row["epoch"]),
                    "tail_accuracy": float(row["bottom20_tail_acc"]),
                    "path": str(path.relative_to(ROOT)),
                }
            )
    return facts


def write_training_audit(output_dir: Path, clip_meta: dict, invariant_frame: pd.DataFrame) -> None:
    clip_lora = ROOT / "trainers" / "cliplora.py"
    fed_main = ROOT / "federated_main.py"
    aggregation = ROOT / "utils" / "lora_aggregation.py"
    split = ROOT / "utils" / "datasplit.py"
    records = [
        {
            "item": "global candidate space and loss",
            "path": str(clip_lora.relative_to(ROOT)),
            "symbol": "CustomCLIP.forward / ClipLora.forward_backward",
            "line": source_line(clip_lora, "logits = logit_scale * image_features @ text_features.t()"),
            "status": "verified_static",
            "conclusion": "Every local example is scored against all 100 text features and optimized with ordinary cross-entropy; there is no local-class mask.",
            "story_implication": "A client without tail positives can still move the shared visual representation and tail decision boundaries.",
        },
        {
            "item": "round-start state and optimizer lifetime",
            "path": str(fed_main.relative_to(ROOT)),
            "symbol": "ClipLora client loop",
            "line": source_line(fed_main, "trainer.model.load_state_dict(global_weights, strict=False)"),
            "status": "verified_static",
            "conclusion": "Each client starts from the same round-global weights and the optimizer/scheduler are rebuilt for that client.",
            "story_implication": "Optimizer state does not persist across clients; parameter updates do compete through the common global LoRA.",
        },
        {
            "item": "FedAvg weighting",
            "path": str(aggregation.relative_to(ROOT)),
            "symbol": "average_lora_weights",
            "line": source_line(aggregation, "def sample_weighted_client_weights"),
            "status": "verified_static",
            "conclusion": "The ordinary aggregation branch weights active client LoRA states by client sample count.",
            "story_implication": "Co-location is a proxy for multi-step local interaction, not a claim that remote related-class updates never enter the global adapter.",
        },
        {
            "item": "LoRA trainable scope",
            "path": str(clip_lora.relative_to(ROOT)),
            "symbol": "ClipLora.build_model",
            "line": source_line(clip_lora, "if 'lora_' in name:"),
            "status": "verified_static",
            "conclusion": "The active ClipLora path freezes every non-LoRA parameter; the CLI defaults are vision top-3 blocks, q/v projections, rank 2, alpha 1, and dropout 0.",
            "story_implication": "All classes and clients write through the same small set of vision-adapter tensors; exact tensor shapes still belong to a runtime training configuration audit.",
        },
        {
            "item": "local traversal semantics",
            "path": "utils/dataloader.py",
            "symbol": "get_dataloader",
            "line": source_line(ROOT / "utils" / "dataloader.py", "shuffle=True, drop_last=False"),
            "status": "verified_static",
            "conclusion": "The federated image loader shuffles and retains the final partial batch. With the proposed batch size 32 and 3 local epochs, a client of size n has 3*ceil(n/32) expected optimizer steps.",
            "story_implication": "V1 records derived exposure only; it does not replay the stochastic batch order.",
        },
        {
            "item": "controlled Client-LT split",
            "path": str(split.relative_to(ROOT)),
            "symbol": "partition_client_longtail_controlled",
            "line": source_line(split, "def partition_client_longtail_controlled"),
            "status": "verified_runtime",
            "conclusion": "All 153 bottom-20 train samples remain in clients 27-29; each specialist is at least 80% tail and aggregate companion count is at most 38.",
            "story_implication": "Tail leakage and companion allowance are now independent controls; the legacy Client-LT path is unchanged.",
        },
        {
            "item": "frozen text semantic probe",
            "path": str((ROOT / "clip" / "simple_tokenizer.py").relative_to(ROOT)),
            "symbol": "ViT-B/16 encode_text",
            "line": None,
            "status": "verified_runtime",
            "conclusion": f"Text feature shape is {clip_meta['feature_shape']}; no_grad was used and parameter fingerprints match.",
            "story_implication": "V1's related sets are fixed text-semantic probes, not trained visual-gradient compatibility measurements.",
        },
        {
            "item": "visual forward runtime probe",
            "path": str(clip_lora.relative_to(ROOT)),
            "symbol": "CustomCLIP.forward",
            "line": source_line(clip_lora, "def forward(self, image)"),
            "status": "runtime_unverified",
            "conclusion": "The current lightweight environment has no torchvision, so no image forward was executed; static candidate-space and loss checks remain verified.",
            "story_implication": "P0 does not claim a runtime replay of training and starts no federated training.",
        },
    ]
    historical = historical_round_facts()
    payload = {
        "audit_records": records,
        "historical_artifact_facts": historical,
        "historical_joined_to_v1": False,
        "reason_not_joined": "Historical accuracy used the legacy Client-LT split, not the newly controlled split.",
        "controlled_invariants_all_passed": bool(
            invariant_frame["coverage_exact"].all()
            and invariant_frame["class_counts_conserved"].all()
        ),
        "interpretation_correction": (
            "Historical zero-shot and round-0 tail accuracy are nearly identical, so the current CIFAR evidence shows "
            "retention/erosion of pretrained CLIP tail capability, not strong new LoRA acquisition at round 0."
        ),
        "formation_estimand_for_v2": "A_c(theta_t + Delta_support_c) - A_c(theta_t)",
    }
    (output_dir / "training_semantics_audit.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    history_lines = "\n".join(
        f"- {x['topology']} {x['checkpoint']}: tail accuracy {x['tail_accuracy']:.2f}% (epoch {x['epoch']}, `{x['path']}`)"
        for x in historical
    ) or "- No compatible historical round-metric artifact found."
    record_lines = "\n".join(
        f"- **{item['item']}** — `{item['status']}`; `{item['path']}:{item['line'] or 'NA'}`. {item['conclusion']} {item['story_implication']}"
        for item in records
    )
    markdown = f"""# P0 training-semantics audit

This is a read-only semantic audit. No backward pass, optimizer step, local training, or federated round was run.

{record_lines}

## Correction to the two-stage story

The shared-LoRA competition/retention hypothesis is compatible with the active code path. The current CIFAR evidence does **not** yet show that LoRA newly forms strong tail knowledge early: historical zero-shot and round-0 tail accuracy are nearly equal. The defensible statement is that pretrained CLIP already has substantial tail capability and continued federated adaptation erodes it, more strongly in legacy Client-LT.

Formation must be measured as a gain from the same starting state, e.g. `A_c(theta_t + Delta_support_c) - A_c(theta_t)`. Retention must be measured separately before and after controlled non-support/global updates.

## Historical artifacts (not joined to controlled V1)

{history_lines}
"""
    (output_dir / "training_semantics_audit.md").write_text(markdown, encoding="utf-8")


def semantic_rows_for_condition(
    counts: np.ndarray,
    seed: int,
    topology: str,
    tail_class: int,
    class_name: str,
    related: list[int],
    null_sets: list[list[int]],
    similarity: np.ndarray,
    non_tail: list[int],
    input_fingerprint: str,
) -> list[dict]:
    rows = []
    logits = np.asarray([similarity[tail_class, class_id] / 0.1 for class_id in related])
    softmax = np.exp(logits - logits.max())
    softmax /= softmax.sum()
    dose = related_sample_metrics(counts, tail_class, related, non_tail)
    for weighting in WEIGHTINGS:
        clip_uniform = class_set_coverage(counts, tail_class, related, weighting)
        clip_softmax = class_set_coverage(counts, tail_class, related, weighting, softmax)
        null_values = np.asarray(
            [class_set_coverage(counts, tail_class, null_set, weighting) for null_set in null_sets],
            dtype=np.float64,
        )
        rows.append(
            {
                "seed": seed,
                "topology": topology,
                "tail_class_id": tail_class,
                "tail_class_name": class_name,
                "input_fingerprint": input_fingerprint,
                "metric_weighting": weighting,
                "candidate_scope": "non_tail_only_primary",
                "clip_neighbor_colocation": clip_uniform,
                "clip_neighbor_colocation_similarity_softmax_secondary": clip_softmax,
                "null_mean": float(null_values.mean()),
                "null_std": float(null_values.std(ddof=1)),
                "null_q025": float(np.quantile(null_values, 0.025)),
                "null_q975": float(np.quantile(null_values, 0.975)),
                "semantic_neighbor_availability_excess": float(clip_uniform - null_values.mean()),
                "related_companion_absolute_sample_count": dose[
                    f"related_companion_absolute_sample_count_{weighting}"
                ],
                "related_companion_fraction_among_companions": dose[
                    f"related_companion_fraction_among_companions_{weighting}"
                ],
            }
        )
    return rows


def paired_plot(frame: pd.DataFrame, metric: str, title: str, path_stem: Path) -> None:
    pivot = frame.pivot_table(index=["seed", "tail_class_id"], columns="topology", values=metric)
    fig, axis = plt.subplots(figsize=(5.8, 5.2))
    axis.scatter(pivot["ClientLT-controlled"], pivot["Dirichlet"], alpha=0.75)
    values = pivot[["ClientLT-controlled", "Dirichlet"]].to_numpy()
    low, high = float(np.nanmin(values)), float(np.nanmax(values))
    padding = max((high - low) * 0.05, 1e-3)
    axis.plot([low - padding, high + padding], [low - padding, high + padding], "k--", linewidth=1)
    axis.set_xlabel("Controlled Client-LT")
    axis.set_ylabel("Dirichlet")
    axis.set_title(title)
    fig.tight_layout()
    fig.savefig(path_stem.with_suffix(".png"), dpi=180)
    fig.savefig(path_stem.with_suffix(".pdf"))
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--clip-checkpoint", type=Path, default=DEFAULT_CLIP)
    parser.add_argument("--seeds", type=int, nargs="+", default=[42, 2026])
    parser.add_argument("--null-draws", type=int, default=1000)
    parser.add_argument("--bootstrap-draws", type=int, default=10000)
    args = parser.parse_args()
    output_dir = args.output_dir.resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    y_train, y_test, class_names, selected_raw = load_exact_cifar100_lt(args.data_dir)
    global_counts = np.bincount(y_train, minlength=100)
    tail = bottom_classes(y_train, 20)
    non_tail = [class_id for class_id in range(100) if class_id not in set(tail)]
    universe_fingerprint = sha256_bytes(y_train, selected_raw, global_counts)
    mapping_fingerprint = stable_sha256({str(i): name for i, name in enumerate(class_names)})
    input_fingerprint = stable_sha256(
        {
            "universe": universe_fingerprint,
            "mapping": mapping_fingerprint,
            "tail": tail,
            "config": {
                "dataset": "CIFAR-100-LT",
                "imbalance_factor": 0.01,
                "clients": 30,
                "tail_clients": list(TAIL_CLIENT_IDS),
                "controlled_tail_min_purity": 0.8,
                "dirichlet_beta": 0.5,
                "intra_group_alpha": 0.5,
            },
        }
    )
    if int(global_counts[tail].sum()) != 153:
        raise RuntimeError(f"Expected 153 bottom-20 samples, found {global_counts[tail].sum()}")

    embeddings, clip_meta = encode_class_text(class_names, args.clip_checkpoint)
    similarity = embeddings @ embeddings.T
    if not np.allclose(similarity, similarity.T, atol=1e-8):
        raise RuntimeError("CLIP similarity matrix is not symmetric")
    np.save(output_dir / "clip_similarity.npy", similarity)
    clip_meta.update(
        {
            "embedding_sha256": sha256_bytes(embeddings),
            "similarity_sha256": sha256_bytes(similarity),
            "input_mapping_fingerprint": mapping_fingerprint,
            "top_k": 10,
            "temperature": 0.1,
            "primary_candidate_scope": "80 non-tail classes; other tail classes excluded",
        }
    )
    (output_dir / "clip_similarity_meta.json").write_text(
        json.dumps(clip_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    related_by_tail = {class_id: topk_non_tail(similarity, class_id, non_tail, 10) for class_id in tail}
    all_other_by_tail = {class_id: topk_all_other(similarity, class_id, 10) for class_id in tail}
    related_rows = []
    for class_id in tail:
        for rank, neighbor in enumerate(related_by_tail[class_id], 1):
            related_rows.append(
                {
                    "tail_class_id": class_id,
                    "tail_class_name": class_names[class_id],
                    "neighbor_class_id": neighbor,
                    "neighbor_class_name": class_names[neighbor],
                    "rank": rank,
                    "cosine_similarity": float(similarity[class_id, neighbor]),
                    "candidate_scope": "non_tail_only_primary",
                    "input_fingerprint": input_fingerprint,
                }
            )
        for rank, neighbor in enumerate(all_other_by_tail[class_id], 1):
            related_rows.append(
                {
                    "tail_class_id": class_id,
                    "tail_class_name": class_names[class_id],
                    "neighbor_class_id": neighbor,
                    "neighbor_class_name": class_names[neighbor],
                    "rank": rank,
                    "cosine_similarity": float(similarity[class_id, neighbor]),
                    "candidate_scope": "all_other_classes_secondary",
                    "input_fingerprint": input_fingerprint,
                }
            )
    pd.DataFrame(related_rows).to_csv(output_dir / "clip_related_classes.csv", index=False)

    quintiles = build_non_tail_quintiles(global_counts, non_tail)
    null_sets_by_tail = {
        class_id: generate_frequency_matched_null_sets(
            class_id,
            related_by_tail[class_id],
            quintiles,
            draws=args.null_draws,
            master_seed=20260811,
        )
        for class_id in tail
    }
    null_rows = []
    for class_id in tail:
        composition = [sum(quintiles[x] == q for x in related_by_tail[class_id]) for q in range(5)]
        for draw_id, null_set in enumerate(null_sets_by_tail[class_id]):
            null_rows.append(
                {
                    "tail_class_id": class_id,
                    "draw_id": draw_id,
                    "class_ids": " ".join(map(str, null_set)),
                    "quintile_composition": " ".join(map(str, composition)),
                    "set_sha256": stable_sha256(null_set),
                    "master_seed": 20260811,
                    "candidate_scope": "non_tail_only_primary",
                }
            )
    null_frame = pd.DataFrame(null_rows)
    null_frame.to_csv(output_dir / "frequency_matched_null_sets.csv", index=False)
    (output_dir / "frequency_matched_null_sets.sha256").write_text(
        file_sha256(output_dir / "frequency_matched_null_sets.csv") + "\n", encoding="ascii"
    )

    partitions: dict[tuple[int, str], dict[int, np.ndarray]] = {}
    matrices: dict[tuple[int, str], np.ndarray] = {}
    invariant_rows = []
    for seed in args.seeds:
        dir_train, _ = partition_fine_class_dirichlet(
            y_train, y_test, 30, 100, beta=0.5, split_seed=seed
        )
        controlled = partition_client_longtail_controlled(
            y_train,
            30,
            100,
            head_client_ratio=0.9,
            tail_client_ratio=0.1,
            tail_class_ratio=0.2,
            intra_group_alpha=0.5,
            tail_client_min_purity=0.8,
            tail_class_ids=tail,
            rng=np.random.RandomState(seed),
        )
        for topology, partition in (("Dirichlet", dir_train), ("ClientLT-controlled", controlled)):
            partitions[(seed, topology)] = partition
            matrices[(seed, topology)] = counts_matrix(y_train, partition)
            invariant_rows.append(
                partition_invariant_row(seed, topology, y_train, partition, tail, universe_fingerprint)
            )
    invariant_frame = pd.DataFrame(invariant_rows)
    invariant_frame.to_csv(output_dir / "partition_invariants.csv", index=False)
    controlled_rows = invariant_frame[invariant_frame["topology"] == "ClientLT-controlled"]
    if not (
        invariant_frame["coverage_exact"].all()
        and invariant_frame["class_counts_conserved"].all()
        and (controlled_rows["tail_samples_in_ordinary"] == 0).all()
        and (controlled_rows["tail_samples_in_specialists"] == 153).all()
        and (controlled_rows["specialist_companion_total"] <= 38).all()
        and (controlled_rows["specialist_purity_min"] >= 0.8 - 1e-12).all()
    ):
        raise RuntimeError("Partition invariant failure; refusing to compute V1")

    npz_payload = {
        "global_train_labels": y_train,
        "selected_raw_train_indices": selected_raw,
        "global_class_counts": global_counts,
        "tail_class_ids": np.asarray(tail, dtype=np.int64),
    }
    for (seed, topology), matrix in matrices.items():
        key = f"counts_seed{seed}_{topology.lower().replace('-', '_')}"
        npz_payload[key] = matrix
    np.savez_compressed(output_dir / "client_class_counts.npz", **npz_payload)
    (output_dir / "client_class_counts_meta.json").write_text(
        json.dumps(
            {
                "shape": "30 clients x 100 fine classes",
                "keys": sorted(npz_payload),
                "input_fingerprint": input_fingerprint,
                "tail_class_ids": tail,
                "tail_class_names": [class_names[x] for x in tail],
                "seeds": args.seeds,
                "topologies": ["Dirichlet", "ClientLT-controlled"],
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    manifest_payload = {}
    manifest_meta = {"input_fingerprint": input_fingerprint, "entries": {}}
    for (seed, topology), partition in partitions.items():
        slug = topology.lower().replace("-", "_")
        for client_id, indices in partition.items():
            key = f"seed{seed}_{slug}_client{client_id}"
            values = np.asarray(indices, dtype=np.int64)
            manifest_payload[key] = values
            manifest_meta["entries"][key] = {
                "sample_count": int(values.size),
                "sorted_indices_sha256": sha256_bytes(np.sort(values)),
            }
    np.savez_compressed(output_dir / "partition_indices.npz", **manifest_payload)
    manifest_meta["archive_sha256"] = file_sha256(output_dir / "partition_indices.npz")
    (output_dir / "partition_indices_meta.json").write_text(
        json.dumps(manifest_meta, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    topology_rows = []
    generic_rows = []
    semantic_rows = []
    for seed in args.seeds:
        for topology in ("Dirichlet", "ClientLT-controlled"):
            matrix = matrices[(seed, topology)]
            for class_id in tail:
                base = {
                    "seed": seed,
                    "topology": topology,
                    "tail_class_id": class_id,
                    "tail_class_name": class_names[class_id],
                    "input_fingerprint": input_fingerprint,
                }
                top = topology_metrics(matrix, class_id)
                if topology == "ClientLT-controlled":
                    top["tail_mass_in_specialist_clients"] = float(matrix[list(TAIL_CLIENT_IDS), class_id].sum() / global_counts[class_id])
                    top["tail_mass_in_ordinary_clients"] = 0.0
                else:
                    top["tail_mass_in_specialist_clients"] = np.nan
                    top["tail_mass_in_ordinary_clients"] = np.nan
                topology_rows.append({**base, **top})

                generic_primary = generic_context_metrics(matrix, class_id, non_tail, embeddings)
                support, exposure_weights = support_weights(matrix, class_id)
                client_sizes = matrix[support].sum(axis=1)
                for weighting, q in exposure_weights.items():
                    generic_primary[f"derived_expected_optimizer_steps_{weighting}"] = float(
                        np.dot(q, 3 * np.ceil(client_sizes / 32.0))
                    )
                    generic_primary[f"derived_expected_non_tail_image_draws_{weighting}"] = float(
                        3 * generic_primary[f"generic_companion_sample_count_{weighting}"]
                    )
                all_non_c = [candidate for candidate in range(100) if candidate != class_id]
                generic_secondary = generic_context_metrics(matrix, class_id, all_non_c, embeddings)
                generic_rows.append(
                    {
                        **base,
                        "primary_context_scope": "non_tail_classes_only",
                        **generic_primary,
                        **{f"{key}_all_non_c_secondary": value for key, value in generic_secondary.items()},
                    }
                )
                semantic_rows.extend(
                    semantic_rows_for_condition(
                        matrix,
                        seed,
                        topology,
                        class_id,
                        class_names[class_id],
                        related_by_tail[class_id],
                        null_sets_by_tail[class_id],
                        similarity,
                        non_tail,
                        input_fingerprint,
                    )
                )
    topology_frame = pd.DataFrame(topology_rows)
    generic_frame = pd.DataFrame(generic_rows)
    semantic_frame = pd.DataFrame(semantic_rows)
    topology_frame.to_csv(output_dir / "v1a_topology_per_class.csv", index=False)
    generic_frame.to_csv(output_dir / "v1b_generic_context_per_class.csv", index=False)
    semantic_frame.to_csv(output_dir / "v1c_semantic_colocation_per_class.csv", index=False)

    paired_rows = []
    for seed in args.seeds:
        for class_id in tail:
            pair_base = {
                "seed": seed,
                "tail_class_id": class_id,
                "tail_class_name": class_names[class_id],
                "input_fingerprint": input_fingerprint,
            }
            top_pair = topology_frame[(topology_frame.seed == seed) & (topology_frame.tail_class_id == class_id)].set_index("topology")
            gen_pair = generic_frame[(generic_frame.seed == seed) & (generic_frame.tail_class_id == class_id)].set_index("topology")
            sem_pair = semantic_frame[
                (semantic_frame.seed == seed)
                & (semantic_frame.tail_class_id == class_id)
                & (semantic_frame.metric_weighting == "tail_mass_weighted")
            ].set_index("topology")
            row = dict(pair_base)
            for metric in ("support_client_count", "top2_tail_client_mass", "effective_support_clients"):
                row[f"delta_{metric}"] = float(top_pair.loc["Dirichlet", metric] - top_pair.loc["ClientLT-controlled", metric])
            generic_metric = "generic_companion_class_fraction_tail_mass_weighted"
            dose_metric = "generic_companion_sample_count_tail_mass_weighted"
            row["delta_generic_context_primary"] = float(gen_pair.loc["Dirichlet", generic_metric] - gen_pair.loc["ClientLT-controlled", generic_metric])
            row["delta_generic_companion_dose"] = float(gen_pair.loc["Dirichlet", dose_metric] - gen_pair.loc["ClientLT-controlled", dose_metric])
            row["delta_clip_neighbor_colocation"] = float(sem_pair.loc["Dirichlet", "clip_neighbor_colocation"] - sem_pair.loc["ClientLT-controlled", "clip_neighbor_colocation"])
            row["delta_null_colocation"] = float(sem_pair.loc["Dirichlet", "null_mean"] - sem_pair.loc["ClientLT-controlled", "null_mean"])
            row["delta_semantic_specific_tail_mass_weighted"] = float(
                sem_pair.loc["Dirichlet", "semantic_neighbor_availability_excess"]
                - sem_pair.loc["ClientLT-controlled", "semantic_neighbor_availability_excess"]
            )
            paired_rows.append(row)
    paired_frame = pd.DataFrame(paired_rows)
    paired_frame.to_csv(output_dir / "v1_paired_deltas.csv", index=False)

    summary_rows = []
    summary_metrics = [
        "delta_support_client_count",
        "delta_top2_tail_client_mass",
        "delta_effective_support_clients",
        "delta_generic_context_primary",
        "delta_generic_companion_dose",
        "delta_clip_neighbor_colocation",
        "delta_null_colocation",
        "delta_semantic_specific_tail_mass_weighted",
    ]
    for seed in args.seeds:
        selected = paired_frame[paired_frame.seed == seed]
        for metric in summary_metrics:
            values = selected[metric].to_numpy(dtype=np.float64)
            summary_rows.append(
                {
                    "seed": seed,
                    "metric": metric,
                    "mean": float(values.mean()),
                    "median": float(np.median(values)),
                    "positive_classes": int((values > 0).sum()),
                    "negative_classes": int((values < 0).sum()),
                    "zero_classes": int((values == 0).sum()),
                }
            )
    summary_by_seed = pd.DataFrame(summary_rows)
    summary_by_seed.to_csv(output_dir / "v1_summary_by_seed.csv", index=False)

    bootstrap_rows = []
    bootstrap_json = {}
    for metric in ("delta_generic_context_primary", "delta_semantic_specific_tail_mass_weighted"):
        values_by_class = {
            class_id: paired_frame[paired_frame.tail_class_id == class_id][metric].tolist()
            for class_id in tail
        }
        result = cluster_bootstrap(values_by_class, draws=args.bootstrap_draws, seed=20260811)
        bootstrap_rows.extend(
            {"metric": metric, "draw_id": draw_id, "bootstrap_mean": float(value)}
            for draw_id, value in enumerate(result.pop("bootstrap_means"))
        )
        bootstrap_json[metric] = result
    pd.DataFrame(bootstrap_rows).to_csv(output_dir / "v1_cluster_bootstrap.csv", index=False)
    (output_dir / "v1_cluster_bootstrap.json").write_text(
        json.dumps(bootstrap_json, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    generic_seed_means = summary_by_seed[
        summary_by_seed.metric == "delta_generic_context_primary"
    ].set_index("seed")["mean"]
    semantic_seed_means = summary_by_seed[
        summary_by_seed.metric == "delta_semantic_specific_tail_mass_weighted"
    ].set_index("seed")["mean"]
    generic_boot = bootstrap_json["delta_generic_context_primary"]
    semantic_boot = bootstrap_json["delta_semantic_specific_tail_mass_weighted"]
    both_generic_positive = bool((generic_seed_means > 0).all())
    both_semantic_positive = bool((semantic_seed_means > 0).all())
    if (
        both_generic_positive
        and both_semantic_positive
        and generic_boot["ci_low"] > 0
        and semantic_boot["ci_low"] > 0
    ):
        classification = "SEMANTIC_SPECIFIC_COLOCATION_SHRINKAGE"
    elif generic_boot["mean"] > 0 and semantic_boot["mean"] > 0:
        classification = "SUGGESTIVE_PROXY_EVIDENCE"
    elif generic_boot["mean"] > 0:
        classification = "GENERIC_CONTEXT_SHRINKAGE_ONLY"
    else:
        classification = "NOT_SUPPORTED_BY_THIS_PROXY"

    result_summary = {
        "classification": classification,
        "delta_direction": "Dirichlet - ClientLT-controlled",
        "seeds": args.seeds,
        "tail_classes": tail,
        "tail_sample_count": int(global_counts[tail].sum()),
        "primary_context_scope": "non-tail only",
        "primary_weighting": "tail_mass_weighted",
        "generic_seed_means": {str(k): float(v) for k, v in generic_seed_means.items()},
        "semantic_specific_seed_means": {str(k): float(v) for k, v in semantic_seed_means.items()},
        "bootstrap": bootstrap_json,
        "input_fingerprint": input_fingerprint,
        "no_training": True,
        "accuracy_join": "not_performed; historical runs use a different Client-LT partition",
        "evidence_boundary": (
            "V1 measures local positive-example co-location. It does not establish positive transfer, "
            "gradient compatibility, formation of new LoRA knowledge, or an accuracy causal effect."
        ),
    }
    (output_dir / "v1_summary.json").write_text(
        json.dumps(result_summary, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    paired_plot(topology_frame, "effective_support_clients", "V1-A: effective tail support", output_dir / "plot_v1a_topology")
    paired_plot(generic_frame, "generic_companion_class_fraction_tail_mass_weighted", "V1-B: non-tail context breadth", output_dir / "plot_v1b_generic_context")
    semantic_primary = semantic_frame[semantic_frame.metric_weighting == "tail_mass_weighted"]
    paired_plot(semantic_primary, "semantic_neighbor_availability_excess", "V1-C: CLIP-vs-null semantic excess", output_dir / "plot_v1c_semantic_excess")

    seed_lines = []
    for seed in args.seeds:
        g = float(generic_seed_means.loc[seed])
        s = float(semantic_seed_means.loc[seed])
        seed_lines.append(f"- Seed {seed}: generic breadth Δ={g:.6f}; semantic-specific DiD Δ={s:.6f}.")
    report = f"""# V1 topology and local-context co-location audit

All deltas are `Dirichlet - ClientLT-controlled`. The primary semantic candidate pool contains only the 80 non-tail classes; other tail classes are excluded so that co-concentration of rare classes cannot masquerade as richer external visual context.

## Controlled-split invariants

- The global CIFAR-100-LT train universe has {len(y_train)} samples; the bottom 20 contain {int(global_counts[tail].sum())} samples.
- In both controlled seeds, all 153 tail samples are in clients 27–29, no tail sample is in clients 0–26, specialist purity is at least 0.8, and companion total is at most 38.
- Companion samples were selected uniformly from the real non-tail sample pool without consulting CLIP or accuracy.

## Main results

{chr(10).join(seed_lines)}

- Generic class-breadth cluster interval: [{generic_boot['ci_low']:.6f}, {generic_boot['ci_high']:.6f}].
- Semantic-specific DiD cluster interval: [{semantic_boot['ci_low']:.6f}, {semantic_boot['ci_high']:.6f}].
- Proxy classification: `{classification}`.

This is a class-resampling interval that retains both split seeds within each class cluster. With only two split seeds, it is not a complete uncertainty estimate over training or partition randomness.

## What V1 does and does not establish

V1 can show topology concentration, generic non-tail context shrinkage, and—if the DiD is positive—reduced co-location of CLIP text-semantic neighbors beyond frequency-matched generic sparsification. It does not show that those classes provide positive gradients, cause richer tail knowledge, improve accuracy, or that non-support clients overwrite such knowledge. Those are V2 and later functional questions.
"""
    (output_dir / "v1_report.md").write_text(report, encoding="utf-8")
    write_training_audit(output_dir, clip_meta, invariant_frame)
    print(json.dumps({"output_dir": str(output_dir), "classification": classification}, ensure_ascii=False))


if __name__ == "__main__":
    main()
