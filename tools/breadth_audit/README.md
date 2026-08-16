# E1 semantic-breadth audit

This package implements the frozen evaluation side of the E1 mechanism study.
The protocol is scoped to `mechanism_validation_only`; it does not freeze
hyperparameters for later method tuning, SOTA comparisons, ablations, or
sensitivity experiments.

## 1. Write the immutable mechanism protocol

```bash
python -m tools.breadth_audit.protocol --output-dir output/e1_strength_breadth/protocol_v2
```

V2 corrects the preregistration before formal training: equal local-epoch,
batch-size, optimizer, and full-data rules are frozen, while the naturally
topology-dependent realized optimizer-step counts are audited rather than
artificially padded or truncated. The old V1 file is superseded and must not
be supplied to a formal run. The writer refuses to overwrite a different
existing protocol.

## 2. Create the independent visual subgroup artifact once

```bash
python -m tools.breadth_audit.dino_clusters \
  --data-dir DATA/cifar-100/cifar-100-python \
  --output-dir output/e1_strength_breadth/frozen_eval \
  --model-name dinov2_vitb14 \
  --device cuda
```

The command uses frozen DINOv2 ViT-B/14 embeddings and per-tail-class K-means
with four clusters, seed 20260813, and 20 initializations. It refuses to
overwrite an existing artifact. All topologies, seeds, rounds, and methods must
reuse the same resulting NPZ and its hash-checked metadata. If the DINOv2
repository is already present locally, pass `--dinov2-repo /path/to/dinov2`.

This is frozen evaluation preprocessing, not a smoke experiment.

To inspect what the class-local clusters contain without changing any frozen
assignment, generate nearest-to-centroid representative sheets:

```bash
python -m tools.breadth_audit.inspect_clusters \
  --data-dir DATA/cifar-100/cifar-100-python \
  --artifact output/e1_strength_breadth/frozen_eval/dino_tail_clusters.npz \
  --output-dir output/e1_strength_breadth/cluster_inspection
```

Cluster ids are arbitrary and class-local: cluster 0 for one class has no
semantic correspondence to cluster 0 for another class.

## 3. Metric families

`evaluate_three_breadth_families` requires all three inputs and always emits
all three families together:

- `visual_subgroup_coverage`: worst-cluster accuracy, cluster-accuracy
  standard deviation, recognized-cluster count/fraction at 50%, and
  cluster-balanced accuracy;
- `multi_view_robustness`: deterministic clean/crop/color/blur/occlusion/resize
  results, worst-view accuracy, prediction consistency, worst-view margin, and
  clean-to-corruption drop;
- `neighbor_discrimination_breadth`: mean target-to-neighbor margin,
  worst-neighbor margin, positive-margin neighbor coverage, and variance over
  the frozen V1 Top-10 non-tail neighbors.

The fixed view transforms are deterministic and operate on raw CIFAR images
before the repository's ordinary CLIP test transform.

## 4. No-cherry-picking gate

`preregistered_direction_gate` uses every primary endpoint. A family is
directionally supportive only if all its preregistered primary endpoints favor
Dirichlet over controlled Client-LT. At least two of the three families must
pass. This directional gate does not replace class-clustered confidence
intervals, which must be added in the E1 result summarizer.

## 5. Run the paired formal seed-42 experiment

The formal launcher runs standard fine-label Dirichlet (`beta=0.5`) and
controlled Client-LT as the paired topologies. Controlled Client-LT is checked
before training for all 153 tail samples in clients 27--29, zero tail leakage,
at most 38 non-tail companions, and at least 0.8 purity in each specialist.
Both runs load the same serialized LoRA theta0. Every round from 0 through 100
records tail strength, all three breadth families, representation drift,
realized optimizer steps, and a small LoRA checkpoint.
The mechanism run uses FP32 because the earlier AMP smoke exhibited
condition-dependent GradScaler skipped steps; AMP replication belongs to the
later mainline robustness check, not this causal gate.

```bash
python scripts/run_e1_seed42.py --case both
```

For a cluster job with a time limit, the cases can be submitted separately;
the completed first case is detected and skipped:

```bash
python scripts/run_e1_seed42.py --case dirichlet
python scripts/run_e1_seed42.py --case clientlt_controlled
python scripts/run_e1_seed42.py --case summarize
```

The final command writes
`output/e1_strength_breadth/formal/seed42/analysis/e1_seed42_summary.json`.
Seed 42 is a decision gate, not final multi-seed inference. Only when its
direction is valid should seeds 2026 and 3407 be launched.

## Frozen external inputs

- Tail membership is fixed by the CIFAR-100-LT generator order as class ids
  80--99 (153 samples). Realized-count ties cannot change this identity.
- Semantic neighbors are deterministically frozen from the existing
  `clip_similarity.npy` matrix after excluding those 20 tail classes. This
  replaces the stale V1 neighbor table whose boundary tie selected class 79
  instead of class 80; it does not use E1 predictions or results.
- DINO cluster assignments are created once and never recomputed per result.
