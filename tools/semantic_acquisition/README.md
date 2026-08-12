# V2/V3 mechanism experiments

The implementation follows `codex_prompt_v2_v3_semantic_acquisition_and_local_coadaptation.md` in gated phases.

Generate or verify the deterministic manifests without training:

```bash
python -m tools.semantic_acquisition.manifests --output-dir output/v2_v3_semantic_acquisition/manifests
```

Run the CUDA/FP32 mechanism implementation smoke tests, in order. The main federated baseline remains AMP; FP32 avoids the condition-dependent skipped steps observed in the original three-step AMP smoke:

```bash
bash scripts/run_v2_semantic_acquisition_smoke.sh
bash scripts/run_v3_local_placement_smoke.sh
```

After updating to the V2-A/V2-B/C implementation, rerun the V2 smoke even if an older FP32 smoke passed. Full launchers compare the smoke's implementation-file hashes, git revision/diff hash, and manifest hashes with the current checkout and fail closed on stale evidence.

The smoke scripts do not judge the scientific effect direction. After the V2 implementation gate passes, run both V2 evidence panels and then the joint summarizer:

```bash
bash scripts/run_v2_semantic_acquisition_full.sh
bash scripts/run_v2_topology_replay_full.sh
bash scripts/summarize_v2_joint.sh
```

Or run those three fail-fast steps with one command:

```bash
bash scripts/run_v2_complete_full.sh
```

`run_v2_semantic_acquisition_full.sh` is the controlled related/unrelated/tail-only intervention (V2-B/C). `run_v2_topology_replay_full.sh` is the direct frozen Dirichlet versus ClientLT-controlled 30-client replay (V2-A). Neither replaces the other. The joint summary reports `FORMATION_CHAIN_SUPPORTED` only if the direct topology gap and the controlled semantic mechanism both pass their frozen criteria.

V3 full is fail-closed and reads that joint V2 summary. It runs only when the recorded verdict is `FORMATION_CHAIN_SUPPORTED`:

```bash
bash scripts/run_v3_local_placement_full.sh
```

No launcher is invoked automatically. `failure.json` preserves a traceback when a runtime gate stops. Full state artifacts are stored beneath each run's `states/` directory.

The invalid original AMP smoke remains under `v2_smoke/` for audit. Revised mechanism smoke outputs use `v2_smoke_fp32/` and `v3_smoke_fp32/`, so they never overwrite that failed evidence.
