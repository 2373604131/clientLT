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

