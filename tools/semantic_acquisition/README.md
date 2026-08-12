# V2/V3 mechanism experiments

The implementation follows `codex_prompt_v2_v3_semantic_acquisition_and_local_coadaptation.md` in gated phases.

Generate or verify the deterministic manifests without training:

```bash
python -m tools.semantic_acquisition.manifests --output-dir output/v2_v3_semantic_acquisition/manifests
```

Run the CUDA/AMP implementation smoke tests, in order:

```bash
bash scripts/run_v2_semantic_acquisition_smoke.sh
bash scripts/run_v3_local_placement_smoke.sh
```

The smoke scripts do not judge the scientific effect direction. After both implementation gates pass, V2 full can be started explicitly:

```bash
bash scripts/run_v2_semantic_acquisition_full.sh
```

V3 full is fail-closed and reads the completed V2 summary. It runs only when the recorded V2 verdict is `POSITIVE_SEMANTIC_TRANSFER`:

```bash
bash scripts/run_v3_local_placement_full.sh
```

No launcher is invoked automatically. `failure.json` preserves a traceback when a runtime gate stops. Full state artifacts are stored beneath each run's `states/` directory.
