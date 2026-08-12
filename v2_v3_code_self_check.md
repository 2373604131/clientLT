# V2/V3 code self-check

## Overall status

- Phase A static audit: **PASS_FOR_IMPLEMENTATION**.
- Phase B deterministic data/manifest gate: **PASS** on the real frozen P0/V1 artifacts.
- Phase C dependency-light/static/math tests: **PASS (21/21)**.
- Phase C real CLIP/CUDA assertions: implemented, **pending GPU execution**.
- Phase D V2 CUDA/FP32 smoke: **pending revised GPU run**; the original AMP smoke was correctly classified invalid because it had condition-dependent skipped steps.
- Phase E V3 smoke: **locked until V2 smoke passes**.
- Full runs: **not started in this local environment**. Both V2 full panels are fail-closed on a successful V2 smoke summary; V3 additionally requires the joint V2 verdict `FORMATION_CHAIN_SUPPORTED`.

## Prompt corrections

The prompt had two execution-order inconsistencies and was revised:

1. A read-only static Phase A cannot truthfully provide exact runtime LoRA key/shape/offset and fixed-logit hashes before constructing the real model. Phase A now audits the construction path; Phase C must populate and verify those exact runtime fields.
2. Full launcher files may exist as templates before a smoke run, but they must remain locked. The prompt and launchers now require successful smoke summaries before full execution.

A third correction adds the missing direct scientific comparison: V2 is now split into V2-A frozen topology replay and V2-B/C controlled semantic interventions. This does not alter the frozen classes, budgets, matching, seeds, or original intervention thresholds; it prevents the intervention from being misreported as a direct ClientLT-versus-Dirichlet test.

## Real-manifest audit

Generated under `output/v2_v3_semantic_acquisition/manifests`:

- V2 paired run units: 200 = `2 seeds x 20 classes x (related + 3 unrelated + tail-only)`.
- V3 placements: 240 = `2 seeds x 20 classes x 3 draws x 2 placements`.
- Matching rows: 1,200; all matches stay within the V1 frequency quintile and outside target Top-30/bottom-20/related exclusions.
- Base rows: 13,906; execution rows: 41,718.
- V2-A topology base rows: 43,388 = `2 seeds x 2 topologies x 10,847 samples`; topology execution rows: 130,164 = three exact epoch repetitions.
- Every V2-A seed/topology covers the same 10,847 distinct raw train IDs exactly once; all 30 clients are present; per-client FedAvg weights are `n_k/10847` and sum to one.
- Every frozen ClientLT-controlled replay reasserts zero tail leakage: all 153 tail samples are in clients 27/28/29, specialist companions are at most 38, and each specialist has at least 0.8 tail purity.
- V2-A augmentation seeds are bound to `seed x epoch x raw sample ID`, not topology/client, so each sample receives the same augmentation across topologies. Runtime rechecks the hashes of the actual transformed tensors.
- Maximum V2 episode and V3 client size: 25, below batch size 32.
- Every base sample occurs exactly once per base episode/client and exactly three times in its execution schedule (once per epoch).
- Every V3 placement has equal S/D sizes and exact weights `0.5/0.5`.
- V2 paired slot augmentation seeds and V3 sample-bound augmentation seeds pass.
- Filler is fixed across all draws/placements for a seed/class, has `|F_D|=|T|`, and is disjoint from T/R/all-U.
- Rebuilding manifests produces byte-identical base, execution and matching CSV files.

## Training-path audit

`trainers/cliplora.py` now exposes the existing model builder, real Dassl optimizer/scheduler builder, and one canonical optimizer-step primitive. The baseline trainer and the V2/V3 runtime both call that same primitive. With no `loss_weight` key, the loss returns the original `F.cross_entropy` tensor exactly; the optional mask is default-off.

Runtime assertions cover:

- CUDA required; CPU is rejected rather than used as a scientific substitute.
- deterministic CLIP/LoRA model initialization and exact checkpoint/class mapping hashes;
- LoRA-only trainable keys and flatten spec;
- serialized `theta_0` exact reload after deliberate parameter perturbation;
- exact fixed-probe logits after reload;
- V2-B/C and V3: one batch per epoch, three optimizer attempts and three scheduler steps; V2-A: the complete real client batch schedule with all expected optimizer steps successful and three scheduler steps;
- fresh optimizer/scheduler per episode/client;
- FP32 mechanism precision and exactly 3/3 successful optimizer steps; the mainline AMP behavior is unchanged;
- V2 fixed-denominator mask and real-model masked-gradient invariance;
- V2 target margin/NLL/accuracy, adaptation-tail loss, gradient compatibility/difficulty, update norms and safety metrics;
- V2-A per-client epoch states, sample-weighted global states, whole-tail test metrics, topology fairness checks, paired Dirichlet-minus-ClientLT effects, and the joint V1/V2 bridge table;
- V3 actual augmented-tensor multiset equality, raw-gradient oracle, actual LoRA FedAvg plain-SGD oracle, main-optimizer epoch-1 diagnostic, per-epoch S/D/FedAvg states, path effects and layer effects;
- JSON finite-value rejection and failure/exclusion artifacts.
- stale-smoke rejection using explicit implementation-file hashes, git commit/tracked-diff hash, and current manifest hashes.

## Commands actually checked locally

```text
python -m unittest tests.test_semantic_acquisition_v2_v3 -v
21 tests: PASS

python -m py_compile <all new/modified Python files>
PASS

python -m tools.semantic_acquisition.manifests --output-dir output/v2_v3_semantic_acquisition/manifests_verify
base_rows=13906, execution_rows=41718, topology_base_rows=43388,
topology_execution_rows=130164, structural_gate: PASS

python -m tools.semantic_acquisition.runtime --stage v2 --mode full ...
expected fail-closed result: rejected because no passed smoke summary

python -m tools.semantic_acquisition.runtime --stage v2 --mode smoke ...
expected local result: rejected because CUDA is unavailable
```

The failed local smoke writes `failure.json` and an exclusion row. It is an environment gate, not a passed Phase D result.

## GPU continuation

Run these one-line commands from the repository root, in order:

```text
bash scripts/run_v2_semantic_acquisition_smoke.sh
bash scripts/run_v3_local_placement_smoke.sh
```

Only if V2 smoke reports `IMPLEMENTATION_SMOKE_ONLY` with `valid_comparison=true` should the two formal V2 panels and V3 smoke unlock. Run the complete V2 evidence chain with `bash scripts/run_v2_complete_full.sh`. Formal V3 remains locked until V3 smoke passes and the V2 joint summary reports `FORMATION_CHAIN_SUPPORTED`.
