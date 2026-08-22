# Carrier-access mechanism audit

This package implements the three frozen seed-42 experiments used to refine
the Client-Exposure Long Tail story before the P-FCC/D-RTC prototype.

- **A — natural carrier footprint:** reuses completed E2A outputs and performs
  no training. It compares tail-mass-weighted local functional coverage,
  positive-gain entropy, worst-neighbor gain, effective carrier count, and
  cross-carrier diversity under natural Dirichlet and controlled Client-LT.
- **B — functional transfer matrix:** trains one equal-budget, three-step
  candidate-only LoRA update for every non-tail class and evaluates all 20 tail
  classes. This yields the complete 80-by-20 signed transfer matrix. CLIP text
  similarity is only a proposal prior. Private tail-train evidence and the
  independent tail test endpoint are stored separately.
- **C — joint/separate/readapt:** selects candidates using only B's private
  tail-train evidence, then compares joint co-adaptation, separate equal merge,
  private-evidence donor gating, and an unrelated joint control. Test metrics
  never select candidates or the readaptation coefficient.

On a CUDA compute node, after E2A and E1 `theta0` are available, run one line:

```bash
bash scripts/run_carrier_access_audit.sh
```

If the old P0 output directory is absent on a fresh server, the launcher uses
the verified local `ViT-B-16.pt` checkpoint to regenerate only the 100-by-100
CLIP-text similarity matrix. This CPU-only preparation does not run a
partition, bootstrap, image encoder, backward pass, or optimizer. An existing
matrix is reused without being overwritten.

The runner is stage-resumable. Experiment B additionally checkpoints each of
the 80 candidate states, so an interrupted matrix run can continue:

```bash
python scripts/run_carrier_access_audit.py --stage b
python scripts/run_carrier_access_audit.py --stage summarize-b
python scripts/run_carrier_access_audit.py --stage c
python scripts/run_carrier_access_audit.py --stage summarize-c
```

Experiment A alone can run without CUDA:

```bash
python scripts/run_carrier_access_audit.py --stage a
```

Main conclusions are stored in `experiment_a/experiment_a_summary.json`,
`analysis_b/experiment_b_summary.json`, and
`analysis_c/experiment_c_summary.json`. A remains descriptive; B and C have
separate verdicts so one failed mechanism does not erase another supported
result.

## Post-write rewrite and retention audit

The follow-up D1/D2 audit reuses the 80 saved candidate states from Experiment
B. D1 first writes each tail class from three frozen private samples, then
measures the same norm-equalized candidate update both before and after that
write. The remaining two private samples estimate signed post-write effects;
the 100 test images per class are evaluation-only. D2 ranks fixed updates using
only those private effects and replays low-risk, blind, and high-risk sequences
of lengths 5, 10, and 20. It does not retrain clients or use test effects to
choose a sequence.

Run the complete resumable audit on a CUDA compute node:

```bash
bash scripts/run_post_write_rewrite_audit.sh
```

Or resume an individual stage:

```bash
python scripts/run_post_write_rewrite_audit.py --stage d1
python scripts/run_post_write_rewrite_audit.py --stage summarize-d1
python scripts/run_post_write_rewrite_audit.py --stage d2
python scripts/run_post_write_rewrite_audit.py --stage summarize-d2
```

The final verdicts are written to
`output/post_write_rewrite_audit/analysis_d1/d1_summary.json` and
`output/post_write_rewrite_audit/analysis_d2/d2_summary.json`.
