# Official reference snapshots

This directory holds unmodified, commit-pinned reference files. Runtime adapters
live in `trainers/baselines/`; these snapshots are **not** placed on `sys.path`
and are not launched as standalone training repositories.

| Directory | Upstream | Commit |
|---|---|---|
| `capt` | https://github.com/shihaohou/CAPT | `840b7cab26c04fc9e65540a980b6ce020d62d618` |
| `fedpurel` | https://github.com/shihaohou/FedPuReL | `20ce20a15f449631c9aa655be585a0a39a18a4cd` |
| `fedntd` | https://github.com/Lee-Gihun/FedNTD | `be00ee598139654abe650c446a7ba3bc4e340233` |
| `fedsvd` | https://github.com/seanie12/fed-svd | `e73398ba34b6b269c3bf76ed9d74906761fc1a4d` |
| `fedlf` | https://github.com/18sym/FedLF | `127cbaf363ec3a1f40c116f6b1de3c1f7a705764` |
| `fedyoyo` | https://github.com/shanss132/FedYoYo | `fc6d728febb461ca46eedd142ef835b6c6572f88` |
| `fedrela` | https://github.com/guangzhengh/FedReLa | `20a58bd75f48b3243f80adb35537b5b4ba252773` |

`ffa-lora`, `rolora`, and `lora-a2` instead hold **paper-source provenance**
receipts, not official code snapshots. Their implementations are derived from
the cited equations/algorithms and explicitly adapted to CLIP. No verified
independent official repository is claimed for these three methods.

Each `UPSTREAM.json` records the source URL, commit, archive SHA256 and the exact
hash of every extracted file. Available upstream notices and licenses are kept.
`python scripts/vendor_paper_baselines.py` verifies existing snapshots or fetches
missing ones. It refuses to overwrite modified reference files. It is not needed
on a server receiving this directory with the project.

Adaptation scope:

- FedNTD: same not-true distillation objective, temperature 3 and beta 1; teacher
  is the frozen round-start global CLIP-LoRA. Architecture and data protocol differ
  from the original FedNTD experiments.
- FedPuReL-Global: official entropy-matched zero-shot teacher and **per-parameter**
  gradient purification; matched rank4 top3 visual q/v LoRA. The independent
  personalization stage is omitted. Do not label this as complete personalized
  FedPuReL or claim reproduction of its published main-table numbers.
- CAPT: reuse the prompt/coupling architecture and port the official prompt loss,
  clustering and aggregation. Global aggregation is fixed to every round;
  official test-driven MAB is disabled. Clients start from the shared model and
  use isolated optimizers. This is explicitly a fixed-schedule protocol adaptation.
  The upstream gradient mask occurs after `optimizer.step`; the adapter preserves
  the resulting update instead of silently adding a new pre-step mask.

All methods use the frozen raw-sample partition and base CIFAR normalization /
resize. FedLF retains crop/flip and FedYoYo retains its weak/strong views.
The new global-longtail adapters directly execute selected, hash-verified
upstream definitions: FedLF DecorrLoss; FedYoYo AutoAugment/Cutout and prior
change guard; FedReLa's complete statistics/threshold/relabel decision pipeline.
Their standalone trainers are not imported or launched. FedLF's script has
an undefined update_feature_syn call and cannot be launched unchanged.

Tests execute the upstream FedLF and FedYoYo Local.local_train bodies to check
two-step update parity, not just mirrored adapter formulas. See
`docs/longtail_benchmarks_seed42.md` for input, architecture, numerical guards,
late learning-rate schedule and checkpoint-state adaptation details.
