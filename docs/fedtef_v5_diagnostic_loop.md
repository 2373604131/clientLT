# FedTEF-v5 Diagnostic Loop

FedTEF-v5 should be tuned as an evidence lifecycle, not as a bag of modules:

1. a post-encoder image adapter acquires transferable local evidence;
2. semantic memory preserves sparse class-wise evidence across rounds;
3. TailAgg prevents intermittent tail rows from being diluted;
4. sparse bounded residual fusion rescues useful tail predictions without replacing the base semantics.

## First Run

Run the component ladder on one Dirichlet split and one fixed client schedule:

```bash
GPU_IDS="0 1 2" bash scripts/fedtef_v5_component_ladder_cifar100lt.sh
```

For a short smoke run:

```bash
ROUNDS=10 VARIANTS="prompt_only adapter_only adapter_memory_tailagg_rescue full_v5" GPU_IDS="0" bash scripts/fedtef_v5_component_ladder_cifar100lt.sh
```

The full ladder evaluates:

| Variant | Purpose |
| --- | --- |
| `prompt_only` | clean shared semantic anchor |
| `adapter_only` | stable post-encoder evidence acquisition |
| `adapter_memory_no_fusion` | memory learning without inference effects |
| `adapter_memory_fedavg` | ordinary memory FedAvg reference |
| `adapter_memory_tailagg` | persistent TailAgg contribution |
| `adapter_memory_tailagg_rescue` | bounded residual utilization |
| `full_v5` | add the prior-balanced base auxiliary objective |

After training, inspect:

```text
output/cifar100_LT/fedtef_v5_adapter_component_ladder/fedtef_v5_component_summary.csv
output/cifar100_LT/fedtef_v5_adapter_component_ladder/fedtef_v5_component_report.md
```

## Structured Diagnostics

Each run writes:

| File | Main question |
| --- | --- |
| `shared_stream_update_diagnostics.csv` | Does the image adapter learn non-zero updates, and are they cancelled by FedAvg? |
| `tailagg_diagnostics_per_class.csv` | Does TailAgg retain more local tail evidence than ordinary FedAvg? |
| `fedtef_branch_diagnostics.csv` | Does fusion rescue tail predictions without destructive head flips? |
| `fedtef_v2_gate_history.csv` | Are low-exposure classes discovered reliably over time? |
| `per_class_metrics.csv` | Which specific classes improve or regress? |

Class-count and bottom-tail annotations in these diagnostics are debug-only analysis labels. They must not affect training.

## Decision Rules

### Adapter acquisition is weak

Symptoms:

```text
adapter_only.base_tail <= prompt_only.base_tail
adapter_only.tail_base_cosine_margin <= prompt_only.tail_base_cosine_margin
```

Run a focused acquisition sweep by changing adapter LR/eta in the base script. Do not re-enable LoRA unless adapter acquisition is already stable:

```bash
OUT_ROOT=output/cifar100_LT/fedtef_v5_adapter_eta_sweep SCHEDULE_ROOT=output/cifar100_LT/fedtef_v5_adapter_component_ladder/schedules VARIANTS="prompt_only adapter_only" GPU_IDS="0" bash scripts/fedtef_v5_component_ladder_cifar100lt.sh
```

### Semantic memory is weak

Symptoms:

```text
tail_local_evidence ~= 0
tail_memory_row_norm ~= 0
```

Inspect the positive-row mask first. If masking is correct, increase tail supervision gently:

```text
FEDTEF_V2_TAIL_ONLY_WEIGHT=0.1
FEDTEF_V2_TAIL_MARGIN_WEIGHT=0.1
```

### TailAgg does not preserve evidence

Symptoms:

```text
adapter_memory_tailagg.tail_tailagg_retention <= adapter_memory_fedavg.tail_fedavg_retention
```

Sweep memory momentum:

```text
FEDTEF_V2_EVIDENCE_MEMORY_MOMENTUM=0.25
FEDTEF_V2_EVIDENCE_MEMORY_MOMENTUM=0.5
FEDTEF_V2_EVIDENCE_MEMORY_MOMENTUM=0.75
```

If updates conflict strongly, increase agreement filtering rather than increasing fusion strength.

### Fusion harms head classes

Symptoms:

```text
head_wrong_flip_rate is large
tail_fused_minus_base > 0 but fused_overall drops
```

Reduce injection strength in this order:

```text
FEDTEF_V2_LAMBDA=0.3
FEDTEF_V2_RESIDUAL_CLAMP=2.0
FEDTEF_V2_SCALE_CLAMP_MAX=2.0
FEDTEF_V2_KEEP_KL=1.0
```

### Tail stream has no useful capacity

Symptom:

```text
oracle-gated residual fusion also fails to improve tail accuracy
```

This is a structural failure. Change the memory parameterization or its supervision before tuning the dynamic gate.

## Main Run

Only after the ladder establishes the expected mechanism chain:

```bash
GPU_IDS="0 1 2" bash scripts/fedtef_v5_adapter_dirichlet_stability_cifar100lt.sh
```

Use seeds `1`, `42`, and `3407` for the final stability report.
