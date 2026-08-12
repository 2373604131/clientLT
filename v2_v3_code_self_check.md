# V2/V3 code self-check

## Overall status

- Phase A static audit: **PASS_FOR_IMPLEMENTATION**.
- Phase B deterministic data/manifest gate: **PASS** on the real frozen P0/V1 artifacts.
- Phase C dependency-light/static/math tests: **PASS (14/14)**.
- Phase C real CLIP/CUDA assertions: implemented, **pending GPU execution**.
- Phase D V2 CUDA/AMP smoke: **not passed locally**; the local CPU check stopped as designed before importing/running a substitute model.
- Phase E V3 smoke: **locked until V2 smoke passes**.
- Full runs: **not started**. Both full launchers are fail-closed on successful smoke summaries; V3 additionally requires the formal V2 verdict `POSITIVE_SEMANTIC_TRANSFER`.

## Prompt corrections

The prompt had two execution-order inconsistencies and was revised:

1. A read-only static Phase A cannot truthfully provide exact runtime LoRA key/shape/offset and fixed-logit hashes before constructing the real model. Phase A now audits the construction path; Phase C must populate and verify those exact runtime fields.
2. Full launcher files may exist as templates before a smoke run, but they must remain locked. The prompt and launchers now require successful smoke summaries before full execution.

Neither change alters the preregistered classes, budgets, matching, seeds, metrics, verdict thresholds, or scientific interpretation.

## Real-manifest audit

Generated under `output/v2_v3_semantic_acquisition/manifests`:

- V2 paired run units: 200 = `2 seeds x 20 classes x (related + 3 unrelated + tail-only)`.
- V3 placements: 240 = `2 seeds x 20 classes x 3 draws x 2 placements`.
- Matching rows: 1,200; all matches stay within the V1 frequency quintile and outside target Top-30/bottom-20/related exclusions.
- Base rows: 13,906; execution rows: 41,718.
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
- one full batch per epoch, three optimizer attempts and three scheduler steps;
- fresh optimizer/scheduler/GradScaler per episode/client;
- AMP scale and overflow parity;
- V2 fixed-denominator mask and real-model masked-gradient invariance;
- V2 target margin/NLL/accuracy, adaptation-tail loss, gradient compatibility/difficulty, update norms and safety metrics;
- V3 actual augmented-tensor multiset equality, raw-gradient oracle, actual LoRA FedAvg plain-SGD oracle, main-optimizer epoch-1 diagnostic, per-epoch S/D/FedAvg states, path effects and layer effects;
- JSON finite-value rejection and failure/exclusion artifacts.

## Commands actually checked locally

```text
python -m unittest tests.test_semantic_acquisition_v2_v3 -v
14 tests: PASS

python -m py_compile <all new/modified Python files>
PASS

python -m tools.semantic_acquisition.manifests --output-dir output/v2_v3_semantic_acquisition/manifests
structural_gate: PASS

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

Only if V2 smoke reports `IMPLEMENTATION_SMOKE_ONLY` with `valid_comparison=true` should V3 smoke unlock. Formal V2 remains a separate explicit command. Formal V3 remains locked until both V3 smoke passes and formal V2 reports `POSITIVE_SEMANTIC_TRANSFER`.
