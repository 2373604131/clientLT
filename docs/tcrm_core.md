# TCRM-Core

TCRM is a standalone path launched by `tcrm_main.py`. It is not registered in
`federated_main.py`, Dassl `TrainerX`, CAPT, or PromptFL.

## Why Standalone

TCRM trains on frozen CLIP image features. Once the deterministic CLIP eval
transform has been used to build the feature cache, federated training no
longer needs an image augmentation pipeline or a Dassl trainer loop. The cache
metadata records dataset, split, backbone, transform, sample count, and feature
dimension; mismatches rebuild the cache.

## Prompt Learner

`trainers/tcrm/prompt_learner.py` reimplements only the CAPT/CoOp-style general
context idea:

- trainable shared context `ctx_delta`;
- frozen CLIP text encoder;
- no class-aware prompt;
- no vision prompt;
- no coupling function;
- no image adapter.

## Variants

- `prompt_only`: prompt is trained on non-tail CE; tail directions remain
  zero-shot and residual memory is not updated.
- `decoupled_residual_fedavg`: each tail class has an independent residual
  `rho_k`; updates are weighted by local adapt sample count and sanitized.
- `tcrm_core`: WIDTH, WRITE, SURVIVAL, and optional HBS.

Component flags:

- `--disable_width true`: uses `g_k=1` but still computes `R_pre`.
- `--disable_write true`: any candidate update gets `W=1`, removing admission.
- `--disable_survival true`: disables residual shrinkage but still updates age.
- `--disable_hbs true`: sets HBS contribution to zero.

TCRM-Subspace, `(U, a_k)`, Stiefel updates, QR retraction, low-rank dictionaries,
and prefix regularizers are not implemented in TCRM-Core.

## Commands

```bash
python tcrm_main.py --dataset cifar100_lt --partition client-longtail --method tcrm_core --rounds 20 --frac 0.4 --seed 1
```

Prompt-only:

```bash
python tcrm_main.py --dataset cifar100_lt --partition client-longtail --method prompt_only --rounds 20 --frac 0.4 --seed 1
```

Decoupled residual FedAvg:

```bash
python tcrm_main.py --dataset cifar100_lt --partition client-longtail --method decoupled_residual_fedavg --rounds 20 --frac 0.4 --seed 1
```

Topology report:

```bash
python scripts/tcrm_topology_report.py --dataset cifar100_lt --partition client-longtail --output output/tcrm_topology_report
```

Oracle SVD diagnostics, independent of federated training:

```bash
python scripts/tcrm_oracle_diagnostics.py --output output/tcrm_oracle_diagnostics
```

## Outputs

- `tcrm_topology_bootstrap.csv/json`: static tail exposure topology.
- `tcrm_round_metrics.csv`: global metrics and TCRM diagnostics.
- `tcrm_tail_diagnostics.csv`: per-round per-tail class state.
- `feature_cache/*.pt`: frozen CLIP image features.
- `checkpoints/*.pt`: standalone TCRM state.

## Current Limits

The oracle diagnostic script contains the SVD/reporting path and accepts a
precomputed residual bank. It does not train or inject a Subspace model into the
federated trainer. Controlled concentration sweep is provided as an experimental
allocation generator and does not modify existing partition behavior.
