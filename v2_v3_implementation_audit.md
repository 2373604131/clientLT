# V2/V3 implementation audit

Status: **PASS_FOR_IMPLEMENTATION** (static repository and frozen-artifact audit).

This pass authorizes Phases B/C and generation of the GPU runners. It does not claim that the revised CUDA/FP32 smoke gates have run on this Windows CPU environment.

## Active repository path

- Model construction: `trainers/cliplora.py::load_clip_to_cpu` -> `CustomCLIP` -> `apply_lora` -> `mark_only_lora_as_trainable`.
- Trainable scope: `ClipLora.build_model` enables only parameter names containing `lora_`; with `encoder=vision`, text LoRA parameters are rejected.
- Optimizer/scheduler: `Dassl.dassl.optim.build_optimizer(get_lora_parameters(model), cfg.OPTIM)` and `build_lr_scheduler`.
- Client lifetime: `ClipLora.reset_optimizer_and_scheduler` recreates optimizer, scheduler and AMP scaler for every local client. No state may cross an experimental episode.
- Main local loss: 100-way logits from `CustomCLIP.forward`, followed by global 100-class `F.cross_entropy`.
- Scheduler: `ClipLora.forward_backward` calls `update_lr()` at the final batch of every local epoch. `run_promptfl_local_train_with_scheduler_policy` asserts one scheduler step per local epoch when the federated control is enabled.
- Aggregation: `utils/lora_aggregation.py::sample_weighted_client_weights` and `aggregate_lora_state`; the latter changes only explicitly supplied LoRA keys and preserves frozen state.
- Evaluation: a real image forward through the current visual LoRA and 100 text classifiers. No frozen image-feature cache is scientifically valid here.

## Resolved experiment contract

The mechanism runners explicitly resolve the same current ClipLora settings rather than inheriting the one-epoch YAML value:

| Field | Resolved value |
|---|---|
| Backbone | CLIP ViT-B/16 |
| LoRA | vision, top3, q/v, rank 2, alpha 1, dropout 0 |
| Mainline precision | AMP on CUDA |
| V2/V3 mechanism precision | FP32 numerical control |
| Optimizer | SGD |
| LR | 0.002 |
| Momentum | 0.9 |
| Weight decay | 0.0005 |
| Dampening / Nesterov | 0 / false |
| Local epochs | 3 |
| LR scheduler | single-step, step size 3, gamma 1.0 |
| Warmup | disabled (`WARMUP_EPOCH=-1`) |
| Gradient clipping | none in the active ClipLora path |
| Batch size | 32 |
| Train sampler semantics | deterministic manifest order for the experiment; it replaces RandomSampler only to expose paired slots |
| `drop_last` | false |
| Loss | global 100-class CE; optional per-sample weight uses the fixed actual-batch-size denominator |

The first CUDA implementation smoke found 1--3 condition-dependent GradScaler overflows in every three-step episode, leaving only 0--2 successful optimizer steps. That smoke is permanently invalid and produced no interpretable effect result. Before formal execution, V2/V3 were therefore revised to use FP32 while keeping the mainline baseline AMP. The mechanism gate now requires three successful steps, not merely three attempted steps.

The transform builder resolves `random_resized_crop`, `random_flip`, and CIFAR-100 normalization. The experimental wrapper saves/restores Python, NumPy, CPU Torch and CUDA RNG states around every transform and derives an explicit seed per manifested sample/epoch. V2 seeds follow paired slots; V3 seeds follow the physical base sample across placements. Workers are fixed to zero so no hidden worker RNG stream exists.

All V2-B/C target episodes have `|T_c| + B_c < 32`. With `drop_last=false`, each active V2-B/C episode and each V3 client has exactly one batch per epoch, hence 3 optimizer and 3 scheduler steps. V2-A is deliberately different: it replays every complete frozen 30-client partition, retains each real client size and resulting batch count as part of the topology treatment, and asserts every attempted step succeeds. Each client's scheduler still advances exactly once per local epoch.

## Revised V2 evidence chain

- V2-A directly replays the frozen Dirichlet and ClientLT-controlled partitions from the same `theta_0`, using the same 10,847-sample global universe, three exposures per sample, sample-bound augmentations, the repository's original unweighted CE path, and sample-weighted LoRA FedAvg. It estimates the one-round total partition/topology effect on tail acquisition.
- V2-B compares related companions with frequency-matched unrelated companions under identical tail samples, slots and steps.
- V2-C compares related companions with the same physical companion inputs masked to zero loss, retaining the fixed actual-batch denominator.
- The joint gate requires both a stable Dirichlet formation advantage in V2-A and positive semantic transfer in V2-B/C. Per-class links to V1 are descriptive bridge diagnostics, not causal mediation estimates.
- V3 full reads only the joint verdict `FORMATION_CHAIN_SUPPORTED`; it cannot be unlocked by V2-B/C alone.

The CLIP ViT path contains LayerNorm but no BatchNorm, running-stat normalization, queue, or memory bank. LayerNorm is per sample. Therefore zero-loss companion replacement should leave the tail-only LoRA gradient unchanged; Phase C still tests this on the real model within dtype tolerance.

## Frozen P0/V1 inputs

- Input fingerprint: `ebd8f1d0b7765516d4f7ef8868a8262146dd3b630e7df7c1216c01d8e17b601b`.
- Global universe fingerprint: `861850202191fe1db7037afef30987b14bfda680dd37d60af1f940c5c68b3b0f`.
- Frozen CLIP checkpoint SHA-256: `5806e77cd80f8b59890b7e101eabd078d9fb84e6937f9e85e4ecb61988df416f`.
- Tail set: the 20 IDs in `client_class_counts_meta.json`; total tail train samples 153.
- Stable train sample identity: original CIFAR-100 raw train index stored in `selected_raw_train_indices`.
- Class mapping: CIFAR-100 `meta/fine_label_names`; its hash and the selected-pool hash are recomputed before manifest generation.
- Related classes: `clip_related_classes.csv`, primary `non_tail_only_primary` ranks 1-10. Top-30 is deterministically recovered from frozen `clip_similarity.npy` over the same 80 non-tail universe.
- Frequency strata: the 80 non-tail classes sorted by `(global_count, class_id)` and split into five consecutive groups of 16, exactly matching V1.
- Companion dose: recomputed from `v1b_generic_context_per_class.csv`, controlled Client-LT rows and `generic_companion_sample_count_tail_mass_weighted`, then half-up averaged across seeds 42 and 2026.

The generator must fail closed on any fingerprint, class mapping, budget, Top-K, matching, disjointness, sample conservation, or size invariant. It never rewrites the P0/V1 directory.

## State and numerical audit

`theta_0` will contain the sorted trainable LoRA tensors only. Its artifact also records: trainable names/shapes/offsets/dtypes, flatten-spec hash, LoRA hash, frozen/non-trainable hash, class-mapping hash, checkpoint hash, resolved config, and a fixed-probe logits hash. Every condition reloads tensor-exact `theta_0` before constructing a fresh optimizer/scheduler/scaler.

Per-class margin, NLL and accuracy require the complete 100-way logits. Test identities are stable `test:<raw_test_index>` IDs and are disjoint from `train:<raw_train_index>` IDs. Adaptation-tail loss uses the frozen `T_c` train IDs and is reported only as a fitting diagnostic.

Oracle A uses full-client mean gradients and fixed global accumulation order. Oracle B uses an explicitly diagnostic zero-momentum/zero-weight-decay plain-SGD step. Oracle C uses the resolved main optimizer and is diagnostic at epoch 1; raw-gradient or plain-SGD failure invalidates V3.

## Working-tree and gate note

The worktree already contains user files and earlier P0/V1 work. V2/V3 implementation is isolated under `tools/semantic_acquisition`, `tests`, and `scripts`, plus a default-off minimal ClipLora loss helper if required. Existing unrelated files must not be overwritten.

Local Phase A prerequisites are satisfied. CUDA/FP32 real CLIP construction, tensor-level augmentation equality, theta/logit reload, and model-gradient gates require the GPU environment and remain explicitly pending until the revised smoke commands finish successfully.
