# CAPT / Client-LT Paper Storyline

This document is the working storyline anchor for subsequent discussions,
experiments, result interpretation, and writing. It should be changed only if
new experimental evidence falsifies a core assumption.

## One-Sentence Claim

Federated long-tailed learning should not be characterized only by global class
imbalance and generic label skew. Rare classes may appear as weak fragmented
evidence across many clients, or as concentrated high-quality evidence in a few
specialized clients. Topology-aware class-wise residual consolidation can better
exploit the latter while preserving performance under conventional Dirichlet
splits.

中文主张：

联邦长尾学习不应只由全局类别不均衡和通用标签偏斜来刻画。全局稀有类别既可能以弱而分散的形式出现在许多客户端，也可能稳定集中于少数具有场景专长的客户端。我们的方法应显式整合后者携带的高质量类别级语义证据，并在标准 Dirichlet 划分下保持与强基线相当的性能。

## Core Position

The paper is not trying to prove:

- Client-LT is harder than Dirichlet.
- CAPT fails under Client-LT.
- Dirichlet is unrealistic or wrong.
- The method simply solves tail forgetting.

The paper is trying to prove:

- Existing federated long-tail evaluation over-compresses tail classes into global sample counts and average accuracy.
- It overlooks how tail evidence is organized across clients.
- This client-level tail evidence topology changes local evidence quality, update reliability, aggregation dynamics, and method ranking.
- The proposed method converts client-specialized tail evidence into controllable class-wise gains.

## Reality Starting Point

Real federated clients are not exchangeable random shards of a centralized dataset.
They often correspond to stable semantic entities such as hospitals, departments,
cities, road regions, factories, production lines, device environments, user groups,
or business scenarios.

Therefore, rare classes can have persistent client ownership. They may be strongly
associated with a few specialized clients rather than randomly scattered across
the federation.

The first sentence of the paper should be closer to:

> Real federated clients are semantically specialized rather than exchangeable random shards of a global dataset.

## Dirichlet vs. Client-LT

Dirichlet should be described carefully:

> Dirichlet captures generic stochastic label skew, but it does not explicitly model persistent client-class specialization.

Dirichlet is a reasonable generic heterogeneity protocol. Its limitation for this
paper is that client-class specialization is treated as a random byproduct rather
than an explicitly defined, controlled, and measured variable.

Client-LT should be described as a controlled protocol that fixes global long-tail
statistics while changing the topology of tail evidence across clients.

## Main Concept

Use the term:

> Client-Level Tail Evidence Topology

中文：

> 客户端级尾类证据拓扑

It asks:

- Which clients hold the evidence for a tail class?
- How concentrated is that evidence?
- How pure and strong is the evidence within support clients?
- How does it enter local training and federated aggregation?

Important distinction:

- Spatial topology: where the evidence is.
- Temporal exposure: when the evidence participates in training.
- Aggregation survival: whether the evidence is preserved and used.

The relationship is:

```text
client-class evidence topology
-> local evidence quality and exposure process
-> aggregation dynamics and class-wise knowledge utilization
```

## Two Tail Evidence Forms

### Dirichlet: Fragmented Weak Tail Evidence

In conventional Dirichlet splits, a tail class is often globally rare and scattered
across multiple clients. Each client may hold only a few samples, local purity is
low, gradients can be diluted by head classes, and the evidence may be weak,
fragmented, or semantically incomplete.

### Client-LT: Client-Specialized Concentrated Tail Evidence

In Client-LT, the global number of samples can remain the same, but tail samples
are concentrated in a few specialized clients. Within those clients, the tail
class can have higher local purity, stronger and more consistent gradients,
clearer tail-vs-head boundaries, and more reliable class-specific residual
evidence.

Key defense:

> Client-LT is not necessarily harder. It changes tail evidence from weak and fragmented to locally stronger but concentrated in fewer specialized clients.

## Role of PromptFL

PromptFL is not mainly a SOTA baseline to defeat. Its role is diagnostic:

> Shared prompt learning is topology-sensitive.

It helps show that topology does not only move samples across clients; it can
change how tail updates are formed in shared-parameter VLM adaptation.

Claims about different failure mechanisms under Dirichlet and Client-LT must be
supported by controlled diagnostics. If evidence is not strong enough, use the
more conservative claim:

> Different topologies induce distinct class-level exposure patterns and update statistics.

## Role of CAPT

CAPT is a strong positive control, not a failure case.

Do not write:

> CAPT fails under Client-LT.

Write:

> CAPT is robust in average tail recognition under both conventional label skew and client-specialized tail exposure.

This supports the paper because it shows Client-LT is not an artificial collapse
setting. Strong class-aware prompting can stabilize average tail recognition, but
it does not necessarily exploit specialized client-level tail residual evidence.

## Why Our Method Is Still Needed

The method should not be framed as:

> CAPT forgets tail classes, so we protect them.

Instead:

> CAPT preserves tail semantics. Ours consolidates client-specialized tail semantics.

CAPT can avoid large average tail degradation, but it may not explicitly identify,
accumulate, filter, aggregate, and inject high-confidence class-specific residual
evidence from specialized clients.

## Method Definition

Preferred framing:

> Topology-Aware Class-Wise Residual Consolidation

中文：

> 拓扑感知的类别级残差整合

If using the name TCRM, its logic should be organized around this framing rather
than around catastrophic forgetting.

The method has four conceptual steps:

1. Shared semantic backbone: keep stable global semantics and Dirichlet robustness.
2. Client-level class residual extraction: extract class-specific semantic residuals rather than raw client-private information.
3. Reliability-aware class-wise consolidation: identify high-quality support clients and aggregate class-level residuals with reliability.
4. Sparse residual injection and boundary protection: inject only trusted residuals and protect tail-vs-head decision boundaries.

Core idea:

> Convert locally strong, class-consistent specialized evidence in Client-LT into reliable global class-specific knowledge.

## Expected Result Pattern

The ideal result is not that the method crushes CAPT everywhere.

Expected pattern:

- Dirichlet: close to or slightly better than CAPT; no significant degradation.
- Client-LT: consistently better than CAPT.
- Higher tail concentration / stronger client specialization: larger relative gain over CAPT.
- Gains should appear not only in average tail accuracy, but also in worst-tail, P10/P25 tail accuracy, class-wise margin, and high-concentration tail classes.

Main evidence relation:

```text
(Ours - CAPT under Client-LT) - (Ours - CAPT under Dirichlet) > 0
```

## Experiment Chain

### Experiment 1: Same Global Long-Tail, Different Topology

Fix global class counts, imbalance factor, client number, participation rate,
communication rounds, local epochs, batch size, seeds, and initialization.

Only change support-client count, top-1/top-2 client mass, effective client number,
local purity, and client-class specialization.

Goal:

> The same global long-tail statistics can correspond to distinct client-level tail evidence structures.

### Experiment 2: PromptFL and CAPT Diagnostics

Measure local evidence strength, support-client count, concentration, active rounds,
class-level residual or gradient magnitude, hard-negative margin, per-class accuracy
trajectory, update consistency, and peak-to-final drop.

Goal:

> Different topologies alter exposure patterns and update statistics; CAPT remains more stable in average tail recognition.

### Experiment 3: Main Results

Compare PromptFL, CAPT, ours, and optionally another FL-VLM or long-tail baseline.

Report overall accuracy, head accuracy, tail macro accuracy, macro per-class
accuracy, worst-20% tail accuracy, P10/P25 tail accuracy, tail-vs-head margin,
per-class gains, and gain vs. concentration.

Goal:

> Method ranking and relative gains change with tail evidence topology.

### Experiment 4: Concentration Continuum

Gradually control tail top-1/top-2 client mass, effective client number,
support-client count, and local tail purity.

Plot per-class relative gain against concentration variables.

Goal:

> The advantage of topology-aware residual consolidation increases continuously and interpretably with client specialization.

### Experiment 5: Story-Aligned Ablations

Each ablation must correspond to a structural claim:

- Remove class-wise residual memory: Client-LT gains drop.
- Remove reliability-aware aggregation: high-concentration tail gains disappear.
- Remove sparse injection: residuals fail to become final accuracy gains.
- Remove boundary retention: tail-vs-head margin and worst-tail degrade.
- Replace class-wise residual with shared residual: specialization advantage weakens.
- Ignore topology signals and aggregate uniformly: method collapses toward generic tail enhancement.

## Forbidden and Preferred Wording

Avoid:

> Client-LT is harder than Dirichlet.

Prefer:

> Client-LT captures a distinct client-specialization axis that is not explicitly controlled by standard Dirichlet partitions.

Avoid:

> Existing methods fail under Client-LT.

Prefer:

> Existing methods exhibit different levels and modes of topology sensitivity; strong class-aware prompting can stabilize aggregate tail recognition, but topology-aware class-wise consolidation can further exploit specialized tail evidence.

Avoid:

> Dirichlet is unrealistic.

Prefer:

> Dirichlet captures generic label skew, while Client-LT explicitly models persistent client-class specialization that is common in real federated deployments.

Avoid:

> We solve tail forgetting.

Prefer:

> We improve the class-wise consolidation and utilization of client-specialized tail evidence.

## Contributions

1. We identify client-level tail evidence topology as an overlooked factor in federated long-tailed VLM adaptation.
2. We introduce Client-LT, a controlled protocol that explicitly models persistent client specialization while preserving global long-tail statistics.
3. We show that identical global long-tail statistics can induce distinct class-level evidence patterns and adaptation dynamics across federated VLM methods.
4. We propose a topology-aware class-wise residual consolidation method that preserves standard performance under Dirichlet splits and better exploits client-specialized tail evidence under Client-LT.

## Final Anchor Sentence

Client-LT 的价值不在于制造一个 CAPT 会失败的困难集，而在于把现实中的客户端类别专长从随机标签偏斜中单独抽取出来，并检验模型是否能够把这种结构化尾类证据真正转化为类别级收益。
