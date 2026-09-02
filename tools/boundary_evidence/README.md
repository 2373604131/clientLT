# Boundary-evidence asymmetry experiment

This package implements exactly two scientific components:

1. training-free hard-negative co-exposure under controlled Client-LT and
   fixed-marginal matched Dirichlet;
2. local `c+h` versus `c+r` adaptation from one topology-independent theta0.

The frozen top-5 hard negatives are mined from real model margins on
`P_select`. Controls are frequency matched outside frozen `Top20(c)`. The three
sample pools are disjoint:

- `P_select`: CIFAR-100 train images excluded from the federated LT pool;
- `D_local`: the exact federated CIFAR-100-LT train pool;
- `P_eval`: CIFAR-100 test images.

The exact exponential CIFAR-100-LT pool is rebuilt directly from the raw
dataset. No CLIP-text neighbor list or old semantic-acquisition artifact is an
input to this experiment.

Run on a CUDA machine:

```bash
python -m tools.boundary_evidence.run --stage prepare
python -m tools.boundary_evidence.run --stage local
python -m tools.boundary_evidence.run --stage summarize
```

The runner serializes the deterministic pre-federation initialization produced
by model-init seed 42. It intentionally has no arbitrary checkpoint option, so
a Client-LT- or Dirichlet-trained state cannot contaminate hard-negative
selection.

The three primary endpoints are `delta_m_c`, `delta_m_h`, and
`delta_pair_accuracy`. Update norm is diagnostic only. Five pairs are averaged
inside each tail class before the 20 tail classes are bootstrapped.

`delta_m_c` is not a directional gate. Its class-cluster confidence interval
controls the interpretation: a clearly negative `c+h - (c+r)` contrast is
reported as a target-side/opposite-side trade-off, not as equally strong tail
adaptation. No rewrite or fragility experiment is part of this runner.
