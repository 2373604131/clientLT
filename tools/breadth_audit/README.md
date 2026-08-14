# E1 semantic-breadth audit

This package implements the frozen evaluation side of the E1 mechanism study.
The protocol is scoped to `mechanism_validation_only`; it does not freeze
hyperparameters for later method tuning, SOTA comparisons, ablations, or
sensitivity experiments.

## 1. Write the immutable mechanism protocol

```bash
python -m tools.breadth_audit.protocol --output-dir output/e1_strength_breadth/protocol
```

The writer refuses to overwrite a different existing protocol.

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

## Frozen external inputs

- Tail membership is fixed by the CIFAR-100-LT generator order as class ids
  80--99 (153 samples). Realized-count ties cannot change this identity.
- Semantic neighbors are deterministically frozen from the existing
  `clip_similarity.npy` matrix after excluding those 20 tail classes. This
  replaces the stale V1 neighbor table whose boundary tie selected class 79
  instead of class 80; it does not use E1 predictions or results.
- DINO cluster assignments are created once and never recomputed per result.
