# Client-LT 完整故事主线（证据校正版）

> 版本：2026-09-04  
> 用途：作为论文 Introduction、Problem Formulation、Motivation、Mechanism Analysis 与 Method Motivation 的统一母稿。  
> 核心约束：保留“联邦长尾 → 双维不平衡 → Client-LT 发现 → 尾类退化 → 机制 → 方法”的原始框架，同时严格区分已证实事实、方向性证据、候选解释与已被否定的路线。

---

## 0. 全文最核心的故事

本文不从“我们提出了一个新的 Client-Level Long Tail 问题”开始，而从一个更基础的问题开始：**现有联邦长尾学习是否已经完整刻画了联邦系统中的不平衡？**

经典长尾学习以及大多数联邦长尾方法，主要用类别总样本量 $n_c$ 描述一个类别是 head 还是 tail。这一视角回答了“每个类别有多少数据”，却没有回答“同样数量的数据由哪些客户端持有、分布得多广、以多大聚合权重和多高时间频率进入全局训练”。联邦学习因此在类别频率之外天然多出一个由客户端—类别联合分布决定的维度：**class-conditioned client exposure topology**。

我们在固定全局类别长尾的前提下显式操控这一维度，由此观察到 Client-Level Long Tail（Client-LT）现象：尾类证据可能集中在少数 specialist clients，而 head 类得到广泛客户端暴露。实验表明，即使全局类别频率相同，甚至进一步固定每个客户端的数据总量，改变这种 coupling topology 仍会显著改变尾类表现。因此，Client-LT 不是论文预先假定的起点，而是从联邦长尾的第二维结构中暴露出来的现象。

机制实验进一步排除了“specialist 根本学不会尾类”这一简单解释。恰恰相反，少数持有尾类证据的客户端能够产生很强的局部尾类增益；问题在于这些正向功能进入全局模型的 access 很弱：它们只占很小的 FedAvg 权重。与此同时，所有参与客户端都在修改同一个共享适配空间，即便它们不持有该尾类，也可能对该类产生正向或负向的功能改写。于是形成核心不对称：

\[
\boxed{
\text{tail evidence ownership is sparse}
\quad\text{but}\quad
\text{shared update influence is broad}
}
\]

我们将其概括为 **Evidence–Rewrite Imbalance**：少数客户端负责提供和刷新尾类的直接证据，而广泛客户端持续拥有改写相关共享功能的能力。它首先造成聚合 access/dilution，随后在部分参与和长期共享训练中表现为 absence 与 signed rewriting，最终使已有的尾类适配从“局部可写入”变成“全局难保留”。对预训练视觉语言模型而言，这尤其不应被描述为尾类知识从未形成，因为 zero-shot 模型往往已经具备较强尾类能力；更准确的说法是：**联邦任务适配逐渐压缩并侵蚀了已有尾类能力可用的 shared functional scope。**

因此，方法不应只是再次进行类别重加权，也不能把所有缺类客户端一律当成噪声。它必须同时解决四件事：让稀缺的正向证据获得足够的全局 access；在证据暂时缺席时保持可信功能；利用真正有益的 class-absent donors；并依据私有功能证据抑制有害 rewriting。由此，方法主线自然从“tail reweighting”升级为一个完整的 **write → aggregate → persist → restore** 稀有功能生命周期。

全文逻辑可以压缩为：

```text
Federated long-tailed learning
        ↓
已有研究主要刻画 class-frequency imbalance
        ↓
发现第二维：class-conditioned client-exposure topology
        ↓
固定类别边际（严格对照中再固定客户端边际），只改变 coupling
        ↓
Client-LT：尾类证据集中在少数 specialist clients
        ↓
同样的全局长尾，不同的尾类表现
        ↓
supporters 能学，但正向功能的 aggregation access 不足
        ↓
class-absent updates 对共享功能产生 signed rewriting
        ↓
Evidence–Rewrite Imbalance
        ↓
局部强、全局脆弱的尾类适配与长期 shared-function erosion
        ↓
Evidence-conditioned access + functional persistence/restoration
```

---

## 1. 研究起点：重新审视联邦长尾中的“不平衡”

### 1.1 第一维：类别频率不平衡

设联邦系统共有 $K$ 个客户端和 $C$ 个类别，客户端—类别计数矩阵为

\[
N\in\mathbb{R}^{K\times C},
\qquad N_{kc}=\text{客户端 }k\text{ 中类别 }c\text{ 的样本数}.
\]

类别 $c$ 的全局样本量为

\[
n_c=\sum_{k=1}^{K}N_{kc}.
\]

经典 long-tailed learning 主要研究不同 $n_c$ 之间的巨大差异。例如在 CIFAR-100-LT 中，head 类的样本量远大于 tail 类：

\[
n_{\mathrm{head}}\gg n_{\mathrm{tail}}.
\]

这一类不平衡会导致 head 类主导优化方向、tail 类梯度不足以及决策边界偏移。现有 class-aware 方法通常通过重采样、损失重加权、logit adjustment 或类别条件聚合来修正这条轴。

### 1.2 联邦学习引入的第二维：类别证据在哪里

在联邦环境中，$n_c$ 并不能唯一决定 $N$。客户端 $k$ 的总样本量为

\[
n_k=\sum_{c=1}^{C}N_{kc},
\]

而类别 $c$ 在不同客户端上的暴露分布为

\[
p(k\mid c)=\frac{N_{kc}}{n_c}.
\]

即使全部 $n_c$ 与 $n_k$ 都固定，仍存在大量不同的矩阵 $N$。这些矩阵拥有相同的行、列边际，却具有不同的联合耦合结构 $p(k,c)$。我们真正研究的第二维不是客户端边际 $p(k)$，而是：

> **在给定类别总量和客户端容量之后，一个类别的证据如何耦合到客户端。**

因此，推荐将第二维正式称为：

- class-conditioned client exposure；
- client-exposure concentration；
- client support topology；
- tail-evidence topology。

“客户端边际长尾”不是严格表述，因为 $n_k$ 才是客户端边际，而我们改变的是固定边际后的 coupling。

### 1.3 $s_c$ 是入口，但不是完整定义

最直观的支持客户端数为

\[
s_c=\left|\{k:N_{kc}>0\}\right|.
\]

它回答“有多少客户端直接观察到类别 $c$”。但仅用 $(n_c,s_c)$ 仍不足以完整表示 Client-LT：两个类别可以拥有相同的 $s_c$，却分别呈现 90/10 与 50/50 的证据分配，因而在聚合权重、冗余和时间暴露上完全不同。

更完整的形式应写为

\[
\boxed{\left(n_c,\;p(k\mid c)\right)},
\]

其中 $s_c$ 只是 $p(k\mid c)$ 的一个摘要。实际实验还应报告：

1. **有效支持客户端数**

\[
N_{\mathrm{eff}}(c)
=\frac{1}{\sum_k p(k\mid c)^2}
=\frac{n_c^2}{\sum_kN_{kc}^2};
\]

2. **Top-$m$ evidence mass**

\[
T_m(c)=\sum_{k\in\operatorname{Top}m}p(k\mid c);
\]

3. **当轮 support FedAvg mass**

\[
A_c^t=\sum_{k\in S_t}q_k^t\mathbf 1[N_{kc}>0],
\]

其中 $S_t$ 是第 $t$ 轮参与集合，$q_k^t$ 是 FedAvg 权重；

4. **时间暴露**：active rounds、最大 absence gap、连续 no-support streak。

这四类量分别刻画“多少载体”“有多集中”“实际能进入多少全局权重”以及“多久得不到刷新”。

---

## 2. 现有联邦长尾设定遗漏了什么

### 2.1 隐藏假设

如果只用 $n_c$ 描述一个类别，那么下面两种情况容易被视作等价：

```text
分散支持：总计 100 个样本，分布在 20 个客户端
集中支持：总计 100 个样本，80 个在一个客户端、20 个在另一个客户端
```

然而二者在联邦训练中的行为不同：

- 独立 carrier 数量不同；
- supporter 在 FedAvg 中的总权重不同；
- 部分参与下当轮出现该类证据的概率不同；
- 连续多少轮缺少正向刷新不同；
- 局部客户端看到的竞争类别环境不同；
- 不含该类的更新相对正向证据所占的影响比例不同。

因此，同样的 $n_c$ 并不保证同样的 federated learning difficulty。

### 2.2 为什么不能简单说“Dirichlet 只改变 $p(y\mid k)$”

标准 Dirichlet 划分通常从客户端内部类别比例 $p(y\mid k)$ 出发，因此它主要被用于控制 client heterogeneity。它也会间接改变 $p(k\mid y)$，所以严格来说不能说它“只改变 $p(y\mid k)$”。真正的区别是：

> 标准 Dirichlet 并没有把某一类别的 client-conditioned exposure topology 当成独立、可解释且可匹配的实验变量。

Client-LT 则直接控制尾类证据从 ordinary clients 向 specialist clients 的转移和在 specialist group 内部的集中程度。论文应强调的是“控制目标不同”，而不是把两种条件分布错误地说成互不影响。

---

## 3. Client-LT：不是人为造难题，而是把第二维变成可控变量

### 3.1 从现象到实验操作

Client-LT 的目标不是额外减少尾类样本，而是在保持全局长尾类别计数的前提下重排尾类证据的位置：

\[
n_c\ \text{fixed},
\qquad p(k\mid c)\ \text{changed}.
\]

在最严格的 matched topology 对照中，还进一步保持

\[
n_k\ \text{fixed},
\]

使实验只改变联合矩阵的 coupling。需要明确：早期部分 Client-LT/Dirichlet 实验只保证相同 $n_c$，并不都严格匹配 $n_k$；“两个边际都固定”只能用于通过了 fixed-marginal audit 的实验。

### 3.2 两个拓扑控制参数

原 PDF 中的 Client-LT 生成器用两个参数描述尾类证据拓扑：

#### \(\lambda_T\)：跨角色专精强度

$\lambda_T$ 控制尾类证据从 ordinary clients 向 specialist clients 转移的比例。

- $\lambda_T\uparrow$：specialist mass 增加；
- tail evidence 对 ordinary clients 的 leakage 减少；
- specialist purity 增加；
- 尾类对 specialist group 的依赖增强。

在 30-client、3-specialist、20-tail-class 的 PDF 主设定中，$\lambda_T=0.75$ 时 specialist mass 约为 0.775，specialist purity 约为 0.915。这证明生成器能够控制“尾部证据由哪一类客户端持有”。

#### \(\alpha_T\)：specialist 内部集中程度

$\alpha_T$ 控制已经进入 specialist group 的尾类证据如何在组内分配。

- 较小 $\alpha_T$：证据更容易集中在极少数 specialist；
- 较大 $\alpha_T$：证据在 specialist 之间期望上更均匀；
- Top-1/Top-2 mass 应总体下降；
- $N_{\mathrm{eff}}$ 应总体上升。

必须注意：只有 3 个 specialist 且样本很少时，单个有限 Dirichlet draw 不保证每一点严格单调。正式 30-client 三种子审计中，$\lambda_T$ 的迁移与 leakage 指标单调通过，而 $\alpha_T$ 的部分 Top-2/$N_{\mathrm{eff}}$ 单调检查受有限采样影响没有全部通过。因此，$\alpha_T$ 应被描述为**分布级 concentration control**，不应声称每个有限划分都严格单调。

### 3.3 独立拓扑验证

严格 fresh partition 审计从原始 CIFAR-100 标签重新构建全局长尾池，不复用旧划分。所有协议拥有完全一致的全局类别计数，但尾类拓扑明显不同：

| Protocol | 平均 support clients | Top-2 mass | $N_{\mathrm{eff}}$ | active rounds | max gap |
|---|---:|---:|---:|---:|---:|
| IID + Global-LT | 7.650 | 0.283 | 7.650 | 98.250 | 0.600 |
| Dirichlet + Global-LT | 7.000 | 0.348 | 6.702 | 98.300 | 0.600 |
| Client-LT + Global-LT | 2.000 | 1.000 | 1.446 | 68.600 | 4.000 |
| Hybrid-LT + Global-LT | 5.400 | 0.546 | 4.176 | 95.750 | 1.250 |

这一实验只证明“拓扑变量被成功控制”，不直接证明 Client-LT 更难。

---

## 4. 由第二维发现 Client-LT 的性能现象

### 4.1 PromptFL：损伤稳定且具有 tail specificity

在 CIFAR-100-LT、PromptFL + FedAvg、30 clients、3 local epochs、seeds 42/2026 的正式 C/D 审计中，Client-LT 与 Dirichlet 的尾类差距在 concentration 扫描上稳定存在：

| concentration | Client-LT tail | Dirichlet tail | Client-LT − Dirichlet |
|---:|---:|---:|---:|
| 0.10 | 21.200 | 37.400 | −16.200 pp |
| 0.25 | 20.425 | 39.125 | −18.700 pp |
| 0.50 | 18.550 | 38.500 | −19.950 pp |
| 0.75 | 16.675 | 36.550 | −19.875 pp |
| 1.00 | 17.500 | 37.625 | −20.125 pp |

正式 D 对照中的分组差值为：

- Bottom-20 tail：(−19.15) pp；
- overall：(−4.005) pp；
- non-tail：仅 (−0.21875) pp。

这说明损伤高度集中在尾类，而不是 Client-LT 让整个训练普遍失效。但这组早期实验不能单独承担“固定两个边际后的 topology 因果效应”，它主要建立现象的稳健性和 tail specificity。

### 4.2 视觉 LoRA：现象不是 PromptFL 独有

为了避免用 prompt tuning 的诊断直接为 LoRA 方法背书，仓库完成了独立的 vision-only CLIP-LoRA 配对实验：30 clients、full participation、100 rounds、3 local epochs、相同全局类别边际和调度、seed 42。

| Topology / Aggregation | Overall | Head | Tail | H-mean |
|---|---:|---:|---:|---:|
| Zero-shot CLIP | 64.95 | 64.60 | 66.35 | 65.46 |
| Client-LT + FedAvg | 64.68 | 71.66 | 36.75 | 48.58 |
| Dirichlet + FedAvg | 66.87 | 71.41 | 48.70 | 57.91 |

Client-LT 相对 Dirichlet 的尾类差距为 (−11.95) pp，而 head 反而 (+0.25) pp。更关键的是，zero-shot tail 从 (66.35) 降到 (36.75)：尾类能力并不是从未存在，而是在联邦适配中被显著侵蚀。

用户最新重写稿还报告了一组 80-round full-participation 结果：Matched Dirichlet (53.90) vs Client-LT (40.65)，差距 (13.25) pp。这一数值与上述 100-round LoRA 结论方向一致，但当前工作区中尚未同步对应的 `output/full_participation_diagnosis_seed42` 结果目录。因此，正式投稿前可以使用它，但必须先补齐可追溯 artifact；在此之前，仓库内可完全核验的主数值仍应使用 (48.70) vs (36.75)。

### 4.3 fixed-marginal 证据：不能把差异归因于客户端总量

后续 SCA factorial 使用 matched Dirichlet 重建划分，严格固定全部 $n_c$、全部 $n_k$、模型初始化、训练预算和实际参与 schedule，只改变 client–class coupling。即使在相同 residual architecture 下，Residual-FedAvg 的最终 Head–Tail H-mean 仍为：

| Topology | H-mean |
|---|---:|
| Client-LT | 51.802 |
| Fixed-marginal Dirichlet | 62.496 |

单种子拓扑 penalty 为 (10.694) pp。这个结果比“普通 Dirichlet vs Client-LT”更直接地支持：固定两个边际后，coupling topology 本身仍然改变学习结果。它仍是 seed-42 方向性证据，多种子推断尚未完成。

### 4.4 必须保留的反例：Client-LT 不是对所有方法都更难

CAPT 的 communication-matched 双拓扑诊断显示，它在 Client-LT 上并未承受与普通共享 LoRA/SCA 相同的 topology penalty；seed 42 的分解甚至得到 CAPT tail 的 Client-LT penalty 为 (−5.9) pp，即 Client-LT 并没有更差。由此必须避免以下过强声明：

> “Client-LT 对任何联邦算法都更难。”

正确结论是：

> **Client-LT 暴露了 topology-blind shared adaptation 的系统性脆弱性；显式类别条件机制可以显著缓解甚至反转这一脆弱性。**

这个反例不是削弱故事，反而说明第二维并非“人为增加噪声”，而是一个能被算法结构有针对性处理的独立维度。

---

## 5. 机制分析的正确起点：到底是没有学到，还是没有进入并留在全局模型中

观察到 tail gap 后，不能直接把原因写成“客户端支持数少”。至少需要区分四个阶段：

1. **Local formation/adaptation**：持有尾类证据的客户端能否产生有效更新？
2. **Compatibility**：局部更新是否覆盖并保持足够的困难决策边界？
3. **Global access**：这些正向功能在 FedAvg 中拥有多少聚合质量？
4. **Persistence**：进入共享模型后，是否被后续 class-absent updates 改写？

对预训练 VLM，第一问尤其不能写成“能否从零学习出尾类知识”。更准确的问题是：

> 已有的视觉—语义能力能否被稀少的任务证据正确适配到目标边界，并在长期联邦更新中保持可用？

---

## 6. 已被排除的简单解释：specialist 并不是学不会

PromptFL 单轮反事实聚合给出：

| 指标 | Client-LT | Dirichlet |
|---|---:|---:|
| support FedAvg mass | 7.09% | 20.03% |
| support-normalized tail gain | 26.09 | 23.51 |
| 恢复真实 FedAvg 权重后的 support gain | 3.45 | 8.86 |
| 加入全客户端后的 tail gain | 0.10 | 1.73 |

第一列最关键的不是最终的 (0.10)，而是起点 (26.09)：当 support updates 被归一化为总质量 1 时，Client-LT supporters 的尾类增益不低于 Dirichlet。这直接否定了“specialist 本身不会学”的解释。

视觉 LoRA 也出现同样趋势：在 rounds 20/50/80 上平均，Client-LT 的 normalized-support tail gain 为 (7.38) pp，高于 Dirichlet 的 (5.95) pp；D1 post-write 审计中，直接 tail write 的测试 margin 增益均值为 (+0.00709)，20/20 尾类为正。

因此，主问题不应表述为：

> “尾类因为样本少、客户端少，所以本地根本学不会。”

而应表述为：

> **少数证据持有者能够产生有效尾类适配，但这些功能缺少足够的全局进入权和持续刷新权。**

---

## 7. 形成阶段的补充解释：局部边界约束可能更弱，但证据尚未闭环

这一部分保留用户新故事中的 “under-constrained specialization”，但必须降低结论强度。

### 7.1 hard-negative co-exposure 是比“语义邻居”更精确的对象

语义相似不等于真正的决策竞争。更合理的对象是从冻结模型真实 margin 中选择的 hard competitors。训练无关的 co-exposure 审计比较：直接持有 tail class $c$ 的客户端，是否同时看到其困难竞争类 $h$。用户最新结果报告：

\[
Q_{\mathrm{tail}}^{\mathrm{ClientLT}}
<
Q_{\mathrm{tail}}^{\mathrm{MatchedDirichlet}}.
\]

这说明 Client-LT 不仅减少直接 evidence carriers，也可能让 carriers 接触到更少的真实困难边界。当前工作区包含完整协议和代码，但没有同步 `output/boundary_evidence/summary.json`，所以论文落数前仍需补齐 artifact。

### 7.2 $c+h$ 与 $c+r$：观察到 margin-level trade-off

局部对照从相同的 topology-independent $\theta_0$ 出发，比较：

- $c+h$：tail class 与真实 hard competitor 联合适配；
- $c+r$：tail class 与频率匹配的普通 control class 联合适配。

当前讨论所得的模式是：$c+r$ 对 target side $m_c$ 的提升更激进，而 $c+h$ 对 opposite side $m_h$ 的保持更好。这个模式与“缺少困难竞争类时产生 under-constrained specialization”一致：更新并不弱，甚至可能更强，但其边界兼容性更差。

不过必须同时写出证据边界：预注册的 pairwise-accuracy gate 没有通过；该实验本身也不建立后续 rewrite causality。因此当前最多可以说：

> **Client-LT carriers exhibit a margin-level pattern consistent with under-constrained local specialization.**

还不能说：

> “Client-LT 已被证明因为 under-constrained specialization 而产生最终全局退化。”

### 7.3 compatibility → retention bridge 目前尚未成立

V1 bridge 曾得到表面上的正向 retention contrast，但后续审计发现：class-absent background update 本身大幅提高 target margin，使原来的 $G_{post}/G_{local}$ 被不等分母主导。仓库已明确将 V1 verdict 标记为：

```text
SUPERSEDED_DENOMINATOR_ARTIFACT_NOT_EVIDENCE
```

修正后的 V2 应使用 background-adjusted ratio：

\[
R_c^*=
\frac{
M(\theta_0+\Delta_{tail}+\Delta_{bg})
-M(\theta_0+\Delta_{bg})
}{
M(\theta_0+\Delta_{tail})-M(\theta_0)
}.
\]

当前仓库没有 `output/compatibility_retention_bridge_v2`，所以这条桥仍是待验证箭头。它可以出现在机制图中，但必须用虚线。

---

## 8. 为什么不能回到“语义共现”或“表示变窄”的旧故事

### 8.1 语义共现：结构成立，功能因果链失败

V1 审计显示，Dirichlet 相对 Client-LT 的 generic companion breadth 增量约为 $0.44\sim0.46$；扣除普通 breadth 后，语义邻居仍有约 $0.026\sim0.041$ 的额外共现收缩。因此，“Client-LT 的局部语义共现更窄”是成立的结构事实。

但 V2 的受控功能实验没有得到 related 相对 unrelated 的稳定增益，纯语义效应均值约 (0.00033)，bootstrap 区间跨 0；topology replay 也没有稳定 formation gap。正式 verdict 为：

```text
FORMATION_CHAIN_NOT_SUPPORTED
```

所以不能再使用：

\[
\text{semantic neighbors do not co-occur}
\Rightarrow
\text{tail knowledge cannot form}.
\]

语义相似度最多只能是低成本候选 prior，不能承担因果解释或安全判断。

### 8.2 “Client-LT 学出更窄的表示”也没有通过

E1 的 strength gate 通过，但 own-peak breadth、majority-family 和 accuracy-controlled breadth gates 均未通过，正式结论为：

```text
SEED42_STRONG_BUT_NOT_NARROW
```

因此不能把最终退化归因于一个稳定、统一的 narrow embedding geometry。

### 8.3 Functional Breadth 是候选，不是已证明中介

Carrier A 的自然拓扑对比显示，Dirichlet 相对 Client-LT：

- effective carrier count：(+3.741)，20/20 类同向；
- union positive-margin coverage：(+0.1385)，20/20；
- unseen coverage：(+0.1696)，20/20；
- positive-gain entropy：(+0.0793)，20/20。

Carrier B 又显示，语义相似度与真实功能效应的 Spearman 仅约 (0.148)，而 private-tail-train proxy 与测试效应约为 (0.837)。这些结果支持“功能载体比语义邻居更接近真正机制”。

但 Carrier A 仍有客户端 composition/exposure 混杂。随后试图构造 strength/norm/budget 匹配而 breadth 不同的 Broad/Narrow pair，正式 feasibility 只在 4/20 尾类上找到可用 pair，低于预设的 12/20 gate，verdict 为 `PARTIAL`。因此 Functional Breadth 不能被升级为已证实的统一中介，也不应为了保住故事而放宽 gate。

这条路线当前的正确地位是：

> **描述性上兼容、因果上未通过可操作性 gate 的候选解释。**

---

## 9. 已得到强支持的第一条主机制：Aggregation Access / Dilution

FedAvg 的全局更新为

\[
\Delta\theta^t
=\sum_{k\in S_t}q_k^t\Delta_k^t,
\qquad
q_k^t=\frac{n_k}{\sum_{j\in S_t}n_j}.
\]

对尾类 $c$，只有 $E_c=\{k:N_{kc}>0\}$ 能提供直接监督。其当轮直接证据 access 为

\[
A_c^t
=\sum_{k\in S_t\cap E_c}q_k^t.
\]

Client-LT 把尾类证据集中到少数 specialist，并不保证这些客户端拥有大的 $n_k$。于是即使它们产生很强的局部尾类更新，乘上真实 FedAvg 权重后仍会被显著压小。

PromptFL 的 $26.09\rightarrow3.45$ 正是这一阶段：support-normalized gain 很强，恢复真实样本权重后只剩很小的有效全局贡献。视觉 LoRA 的端到端干预提供了更直接的因果证据：

| Topology | FedAvg tail | Support-normalized tail | 增益 |
|---|---:|---:|---:|
| Client-LT | 36.75 | 45.80 | +9.05 pp |
| Dirichlet | 48.70 | 49.40 | +0.70 pp |

Client-LT 的 20 个尾类中 19 个改善、1 个不变。拓扑 gap 从 11.95 pp 缩小到 3.60 pp，闭合约 69.87%。这说明 access correction 是针对 Client-LT 的有效干预，而不只是普通 tail reweighting 的偶然收益。

但 support normalization 不是完整解：

- head 下降 (1.76) pp；
- 仍残留 (3.60) pp tail gap；
- 它需要类别 support 信息；
- supporter 缺席时无法产生新证据；
- 它不区分 class-absent donor 与 rewriter；
- 它没有保护 shared LoRA 中已经存在的功能。

因此，“稀释”是主机制之一，但不能独自解释全部退化。

---

## 10. 已得到强支持的第二条主机制：Signed Shared Rewriting

### 10.1 Evidence ownership 与 update influence 不对称

Client-LT 下，直接证据由少数 $E_c$ 持有，但共享 LoRA 的改写权属于全部参与客户端。对某个类 $c$ 的功能变化，可以概念性写成：

\[
\Delta F_c^t
\approx
\underbrace{
\sum_{k\in S_t\cap E_c}q_k^t g_{k,c}^t
}_{\text{direct evidence write}}
+
\underbrace{
\sum_{k\in S_t\setminus E_c}q_k^t g_{k,c}^t
}_{\text{class-absent signed rewrite}}.
\]

第一项的来源少、权重小；第二项的来源广，而且 $g_{k,c}^t$ 并不恒为负。Evidence–Rewrite Imbalance 的含义不是“所有 non-support clients 都在破坏尾类”，而是：

> **直接正证据的所有权高度集中，但对相同共享功能施加正向或负向影响的权力并未集中。**

这里必须保留一个当前证据边界：现有实验分别证明了 positive access dilution、class-absent signed effects 与长期 shared erosion，但尚未在同一条真实联邦轨迹上直接测量“destructive rewrite / positive functional refresh”的比率。因此，这三组证据目前共同提出并约束 ERI 机制，却还不能替代对 ERI 本身的直接检验。

### 10.2 反事实聚合显示全体更新会继续压缩尾类增益

PromptFL 中，从 actual support gain 到 all-client gain：

\[
3.45\rightarrow0.10
\quad(\mathrm{ClientLT}),
\]

\[
8.86\rightarrow1.73
\quad(\mathrm{Dirichlet}).
\]

视觉 LoRA 的 rounds 20/50/80 平均也从 Client-LT actual-support (0.62) pp 变为 all-client (0.00) pp。这个分解说明：在权重稀释之外，加入其他客户端更新还会进一步改变并压缩剩余尾类增益。

### 10.3 D1：class-absent clients 同时包含 donor 与 rewriter

D1 先通过 direct tail write 明确建立尾类功能，再对固定、范数均衡的 class-absent updates 做扫描。结果表明：

- 20/20 尾类都同时存在 donor 与 rewriter；
- 平均 donor 数为 (47.25)；
- 平均 rewriter 数为 (32.75)；
- private/test post-effect Spearman 约 (0.750)；
- sign agreement 约 (0.808)；
- private rewriter recall 约 (0.791)；
- private donor precision 约 (0.846)。

完整的 donor-to-rewriter turnover 没有出现，因此 verdict 是：

```text
POST_WRITE_REWRITE_SUPPORTED_WITHOUT_FULL_TURNOVER_CHAIN
```

这支持 state-conditioned signed rewriting，但不支持“一个 donor 普遍会在写入后翻转成 rewriter”，也不支持“所有缺类更新都有害”。

### 10.4 D2：私有风险能够预测后续遗忘

D2 对冻结且范数匹配的更新序列进行 replay：

- low-risk 相对 blind 的 forgetting advantage 为 (+0.0002607)，20/20 类同向；
- high-risk 相对 blind 更差 (+0.0002798)，19/20 类同向；
- private risk 与 blind forgetting 的 Spearman 约 (0.505)，19/20 为正；
- verdict：`REWRITE_RISK_PREDICTS_RETENTION`。

这说明客户端不需要向服务器公开类别列表，也能用少量私有证据判断外来更新对自己关心功能的风险。但它仍是 frozen causal replay，不是完整多轮联邦再训练；论文中应称其为风险信号的因果 replay 证据，而不是端到端方法结果。

---

## 11. 时间维度：partial participation 放大问题，但不是问题的唯一来源

### 11.1 拓扑集中会提高 no-support 概率

如果每轮从 $K$ 个客户端中均匀选择 $m$ 个，而类别 $c$ 只有 $s_c$ 个 supporters，则当轮完全看不到该类直接证据的概率为

\[
P(S_t\cap E_c=\varnothing)
=\frac{\binom{K-s_c}{m}}{\binom{K}{m}}.
\]

$s_c$ 越小，absence 和连续 absence streak 越容易出现。Client-LT 因此不仅减少空间载体冗余，也减少时间刷新机会。

### 11.2 D4：参数不变不等于功能不变

在 30 clients、participation (0.4)、80 rounds 的 SCA 运行中，共有 1600 个 tail-class×round 单元，其中 235 个发生 no-support：

- no-support 时对应 residual row delta 精确为 0；
- 但 235/235 的 class margin 仍发生变化；
- 152 次 accuracy 下降，160 次 margin 下降；
- absence 事件平均 accuracy delta 为 (−1.719) pp；
- 平均 margin delta 为 (−0.1405)。

这证明即使一个类别专属 residual row 被原样保存，其依赖的 shared LoRA 仍在变化，功能仍可能退化。

### 11.3 Stage 2-C：真正持续退化的是 shared functional substrate

清零 residual 后，shared-only 轨迹为：

| Round | Head | Tail | H-mean |
|---:|---:|---:|---:|
| 3 | 67.70 | 68.80 | 68.25 |
| 20 | 70.75 | 26.30 | 38.35 |
| 50 | 73.49 | 22.80 | 34.80 |
| 80 | 73.90 | 22.10 | 34.02 |

round 3→80，20/20 尾类 accuracy 与 20/20 tail margin 均下降；head 平均提高 6.20 pp，tail 平均下降 46.70 pp。与此同时，匹配的 residual 在后期持续提供约 $19\sim20$ pp H-mean 补偿。

这意味着 residual 不是主要退化源，而是在补救已经 head-dominant 的共享适配空间。最准确的机制描述是：

> **shared functional adaptation scope erosion**：早期已有的尾类功能在长期共享更新中系统性收缩。

full-participation LoRA 中仍存在 (11.95) pp 的尾类拓扑差距，说明 temporary absence 不是必要条件；它是 partial participation 下的重要放大器。即使所有客户端每轮都参与，support mass dilution 与 class-absent shared rewriting 仍然存在。

---

## 12. 最终统一机制：Evidence–Rewrite Imbalance

现在可以把用户原稿中的“形成阶段”和“维护阶段”改写为证据更稳固的三阶段生命周期。

### 12.1 阶段一：Local write / task adaptation

尾类证据集中在少数 carriers。已证实的是：这些 carriers 能够产生强 tail gain；尚待闭环的是：它们较窄的 hard-boundary exposure 是否稳定导致 under-constrained compatibility。

### 12.2 阶段二：Global access / aggregation

正向 tail functions 进入 FedAvg 时被 supporter 的小聚合质量压缩。这是由 PromptFL 反事实分解和 LoRA support-normalized 干预共同支持的主机制。

### 12.3 阶段三：Persistence / shared rewriting

全体客户端继续修改共享适配空间。class-absent updates 中 donor 和 rewriter 共存；partial participation 又让直接证据无法持续刷新。长期结果是 shared tail competence 逐渐向 head-dominant state 迁移。

### 12.4 最后缺失的机制闭环：直接测量 ERI

对每轮、每个尾类，需要把实际服务器加权后的客户端功能效应完整分成四项：supporter positive write $W$、supporter harmful update $H$、class-absent donor $D$ 与 class-absent destructive rewrite $R$，并检验

\[
\Delta M_c=(W_c+D_c)-(H_c+R_c).
\]

随后定义累计不平衡

\[
\operatorname{CERI}_c=
\frac{\sum_t R_c^t}{\sum_t(W_c^t+D_c^t)+\epsilon}.
\]

正式闭环要求依次通过：fixed-marginal Client-LT 的 CERI 高于 matched Dirichlet；CERI 正向预测 per-class best-to-final drop；support-aware aggregation 同时降低 CERI 并提高 retention。完整预注册设计见 [Evidence–Rewrite Imbalance 最终闭环实验](docs/evidence_rewrite_imbalance_closure_experiment.md)。在这三层通过前，ERI 应写成“由现有证据共同指向的统一机制假设”，而不是已经被直接测量的结论。

因此完整机制不是简单的

\[
\text{support count少}\Rightarrow\text{学不到},
\]

而是

\[
\boxed{
\begin{aligned}
&\text{Concentrated tail evidence}\\
&\quad\Rightarrow \text{sparse functional writers and weak aggregation access}\\
&\quad\Rightarrow \text{insufficient positive refresh relative to broad signed rewriting}\\
&\quad\Rightarrow \text{locally strong but globally fragile tail adaptation}.
\end{aligned}}
\]

建议机制图使用实线和虚线区分证据强度：

```text
                         ┌─ hard-boundary co-exposure ↓
                         │  under-constrained compatibility ?  [虚线：待 V2 bridge]
Client-LT topology ──────┤
                         ├─ supporter FedAvg mass ↓
                         │  positive functional access ↓       [实线]
                         │
                         └─ absence ↑ + class-absent signed rewriting
                            shared tail scope erosion           [实线]
                                             ↓
                              tail-specific global degradation
```

这样既保留了用户新框架中的“形成—维护”双分支，又不会把尚未通过的 compatibility-retention bridge 写成既定事实。

---

## 13. 方法如何从机制自然推出

方法不应从某个 trick 倒推故事，而应逐项回应已经验证的 failure point。

### 13.1 四个不可缺少的设计要求

#### 1. Evidence-conditioned access

少数 positive writers 的贡献不能自动被全客户端样本量权重压没。聚合应依据“对某个功能是否提供有效证据”分配 access，而不是无条件放大所有小客户端。

#### 2. Functional compatibility / signed transfer

不能把所有 class-absent clients 排除，因为其中存在大量 donors；也不能全部接收，因为其中存在 rewriters。判断必须依赖实际功能效应，而不是语义相似度或全局 scalar geometry。

#### 3. Persistence and restoration

supporter 缺席时，应保留最近可信的功能状态；当 incoming global model 相对历史可信状态退化时，应能恢复相关功能。这里保存的是 function，而不仅是某个 residual parameter row。

#### 4. Explicit information boundary

必须提前说明方法使用的信息级别：

- 类别计数可见；
- 只有 support bit；
- 不上传类别元数据，仅在客户端用私有样本进行功能评价。

不同信息级别对应不同的方法贡献和隐私声明，不能混写。

### 13.2 有类别支持信息时：class-separable aggregation 是机制上界/强对照

Support-normalized aggregation 已经证明 access correction 能显著救回 Client-LT tail。Online SCA 进一步为每个 tail class 增加独立 residual row，只允许当轮 supporters 更新，并在无 supporter 时保留旧 row：

\[
r_c^{t+1}=
\begin{cases}
r_c^t+\displaystyle\sum_{k\in S_c^t}
\frac{N_{kc}}{\sum_{j\in S_c^t}N_{jc}}\Delta r_{k,c}^t,
&S_c^t\neq\varnothing,\\[1.1em]
r_c^t,&S_c^t=\varnothing.
\end{cases}
\]

在 fixed-marginal 2×2 中，SCA 相对 architecture-matched Residual-FedAvg 的最终 H-mean：

- Client-LT：(+1.544) pp；
- matched Dirichlet：(−0.275) pp；
- difference-in-differences：(+1.819) pp。

这说明 class-separable aggregation 具有 topology-specific 净作用，但幅度有限。SCA 只能被称为 support-metadata isolation reference，而不是性能 upper bound 或最终方法，因为它：

- 仍不保护 shared LoRA；
- 无法利用 class-absent donors；
- 参数 row 保留不等于功能保留；
- 需要上传/使用类别支持信息；
- 在 matched Dirichlet 上尚无安全回退。

### 13.3 不上传类别元数据时：私有功能判断是更自然的路线

Carrier B、D1 和 D2 共同给出方法信号：客户端本地少量私有证据，比语义相似度更能判断 proposal 是 donor 还是 rewriter。因此可以形成两个概念组件：

1. **P-FCC / private functional complementation**：服务器只提供由多客户端更新形成的 mixed proposals；客户端在本地私有功能记忆上筛选正向、兼容的 donor direction，并保留零修正安全回退。
2. **D-RTC / degradation-triggered restoration**：客户端比较当前 incoming global 与历史可信 incoming-global reference；检测到私有功能退化时，生成恢复方向。

二者分别对应：

| 已验证问题 | 方法责任 |
|---|---|
| positive functional access 不足 | 找到并吸收私有验证为正向的互补方向 |
| class-absent signed rewriting | 识别风险并允许 donor/rewriter 分流 |
| no-support 与长期 shared erosion | 基于历史可信功能触发 restoration |
| head trade-off | 等范数方向合成与零修正回退 |

重要状态说明：仓库中的 P-FCC/D-RTC 目前完成的是运行契约、正确性测试和两轮 smoke，不是正式长程性能证据。它们可以作为“由机制自然推出的方法设计”，但在多轮 gate 通过前不能在论文中写成已经验证有效的最终方法。

### 13.4 为什么当前不应继续堆更复杂的旧组件

已有失败路线给出了明确边界：

- semantic co-location restoration：结构事实成立，功能链失败；
- broad representation regularization：E1 narrowness gate 失败；
- Functional CUSP：预测有信号，但标量权重表达能力不如简单 class-wise；
- boundary repair：目标可修，但 non-target safety gate 拒绝；
- FedTEF 大系统：memory、routing、fusion 多处瓶颈；
- Functional Breadth：feasibility 仅 4/20，当前定义不适合直接升级为方法。

共同教训是：最终方法必须作用在已证实的 access 或 functional retention 环节，而不能仅凭一个代理量构造复杂系统。

---

## 14. 可直接用于论文 Introduction 的六段式写法

### 第 1 段：从联邦长尾的既有定义开始

长尾学习通常把训练困难归因于类别样本量的极端不平衡。联邦长尾研究继承这一视角，主要围绕全局类别频率设计 class-aware loss、reweighting 和 aggregation。然而，联邦数据并不只是一个被切开的集中式长尾数据集：类别证据还具有客户端位置和时间暴露结构。

**转折句：** 相同数量的尾类样本，如果分散在许多客户端或集中在少数 specialist clients，并不会以相同方式进入联邦优化。

### 第 2 段：提出第二维，但暂时不先定义 Client-LT

用 $N_{kc}$ 表示客户端—类别计数后，$n_c$ 只给出列边际。即使类别总量和客户端容量相同，联合矩阵仍可以具有不同的 $p(k\mid c)$。这一 coupling 决定一个类别拥有多少独立载体、多少 FedAvg access，以及在部分参与下多久能得到一次正向刷新。现有以类别频率为中心的描述因而遗漏了 class-conditioned client-exposure topology。

**转折句：** 这促使我们问：当全局长尾保持不变、只改变尾类证据的客户端拓扑时，模型是否仍表现相同？

### 第 3 段：Client-LT 作为发现和实验工具出现

为回答这一问题，我们构造可控的 Client-LT protocol。它不减少任何尾类证据，而是用 $\lambda_T$ 控制证据向 specialists 的迁移，用 $\alpha_T$ 控制 specialist 内部集中度；在严格对照中进一步固定 $n_c$ 与 $n_k$，仅改变 coupling。由此，Client-LT 是用来暴露第二维影响的 controlled intervention，而不是随意制造的更难 Dirichlet。

### 第 4 段：先给出现象，再提出机制问题

在 PromptFL 与视觉 LoRA 中，Client-LT 都导致显著、tail-specific 的性能下降；full participation 下该差距仍存在，说明现象不能只归因于客户端抽样缺席。与此同时，CAPT 等强类别条件机制并不一定受到同样损伤，说明问题具有明确的算法结构依赖性，而不是数据质量普遍下降。

**转折句：** 更反直觉的是，持有尾类证据的 specialist clients 并没有学习失败。

### 第 5 段：给出 Evidence–Rewrite Imbalance

反事实聚合显示，supporters 归一化后能够产生强尾类增益，但真实 FedAvg 权重显著压缩其全局 access；加入其余客户端更新后，剩余增益进一步下降。后续 post-write 实验又表明，class-absent updates 对尾类功能具有带符号的影响：一些是 donors，另一些是 rewriters，并且客户端私有功能证据能够预测这种风险。Client-LT 因此造成 evidence ownership 与 shared update influence 的结构性错位。部分参与产生的 absence 会进一步放大这一错位，但不是其必要条件。

### 第 6 段：自然引出方法

这一诊断要求方法同时控制稀缺正证据的 access 和已适配功能的 persistence。一个有效方案既要让 positive writers 或私有验证为正的 donors 进入共享模型，又要在直接证据缺席时抑制有害改写并恢复历史可信功能；同时，它不应假设所有缺类客户端都无用，也不应依赖未经验证的语义邻居代理。

---

## 15. Contributions 的建议写法

在最终方法尚未完成长程 gate 前，建议只写前三项为确定贡献，第四项作为待补方法贡献。

1. **Problem characterization.** 我们揭示联邦长尾除类别频率之外还存在 class-conditioned client-exposure topology，并用固定边际后的 coupling 对其进行严格定义。
2. **Controlled protocol and phenomenon.** 我们设计由 $\lambda_T$ 与 $\alpha_T$ 控制的 Client-LT protocol，在保持全局类别长尾、并在严格对照中保持客户端容量不变的条件下，观察到 topology-dependent、tail-specific degradation。
3. **Mechanism（当前证据等级）.** 我们证明 supporters 能够产生强尾类适配，但其全局 access 被 FedAvg 权重压缩；同时，class-absent shared updates 对已写入功能产生可由私有证据预测的 signed rewriting。这两部分共同指向 Evidence–Rewrite Imbalance；只有在直接 ERI 实验通过 topology difference、outcome association 与 intervention consistency 三层 gate 后，才改写为“共同构成并直接验证了 Evidence–Rewrite Imbalance”。
4. **Method（仅在正式长程 gate 通过后）.** 我们据此提出 evidence-conditioned access 与 degradation-triggered functional restoration，在不上传额外类别元数据的条件下协调跨客户端正向迁移和长期尾类保留。

如果方法 gate 尚未通过，不应把第四项写成已完成贡献，可以改成“design implications and validated controls”。

---

## 16. 一句话贡献与摘要核心句

### 中文一句话

> 现有联邦长尾学习主要刻画类别数量的不平衡；我们进一步揭示类别证据在客户端上的暴露拓扑，并发现当尾类证据集中由少数客户端承载时，直接证据的稀缺所有权与共享更新的广泛改写权发生错位，使尾类适配虽然能在局部形成，却难以获得足够的全局进入权并在长期聚合中保持。

### 英文一句话

> Existing federated long-tailed learning mainly characterizes class-frequency imbalance; we reveal a complementary class-conditioned client-exposure topology, where sparse ownership of tail evidence is mismatched with broad influence over shared adaptation, yielding tail functions that are locally learnable yet globally difficult to access and retain.

### 方法引出句

> This evidence–rewrite imbalance calls for a functional lifecycle that grants scarce positive evidence sufficient global access, admits class-absent donors only when privately verified, and restores trusted tail functions when shared adaptation erodes them.

---

## 17. 对当前重写稿的关键修正

| 当前草稿说法 | 建议修正 | 原因 |
|---|---|---|
| 完整联邦长尾可写成 $(n_c,s_c)$ | 写成 $(n_c,p(k\mid c))$，$s_c$ 是摘要 | 相同 $s_c$ 仍可有完全不同的 concentration 和 FedAvg mass |
| 第二维是“客户端边际” | 改成固定边际后的 coupling / client-exposure topology | $n_k$ 才是客户端边际 |
| Dirichlet 改变 $p(y\mid k)$，Client-LT 改变 $p(k\mid y)$ | 写成二者控制目标不同；Dirichlet 也会间接改变 $p(k\mid y)$ | 避免条件分布的过度二分 |
| Client-LT 始终保持 $n_k$ 不变 | 只对 fixed-marginal matched 实验这样写 | 早期普通 Dirichlet 对照并未全部严格匹配 $n_k$ |
| supporters 少，所以尾类学不到 | supporters 能学，但全局 access 和长期 retention 不足 | 26.09、7.38 以及 direct-write 20/20 均否定“学不到” |
| non-support clients 都造成干扰 | class-absent updates 具有 signed effect | D1 同时发现 donor 与 rewriter |
| under-constrained specialization 已解释最终退化 | 改成 margin-level 候选解释，等待 corrected bridge V2 | pairwise gate 未通过；bridge V1 已被撤销 |
| Functional Breadth 已被证明 | 改成描述性候选，正式 feasibility 仅 4/20 | 没达到 12/20 预设 gate |
| 参数不更新即可保存功能 | 参数 persistence 不等于 functional persistence | D4 的 row 不变但 235/235 margin 仍变化 |
| Client-LT 对所有方法都更难 | 对 topology-blind shared adaptation 尤其困难 | CAPT 是明确反例 |
| support-normalized 已经解决问题 | 它验证 access，但有 head trade-off、残余 gap 和信息依赖 | LoRA 仍残留 3.60 pp gap，head −1.76 pp |

---

## 18. 当前证据账本

### 18.1 可以作为正文主证据

| 结论 | 证据 | 强度与边界 |
|---|---|---|
| 全局 class counts 可完全保持一致 | [Global-LT verification](output/experiment1_global_longtail_verification/paper_notes.md) | 强控制证据；只证明划分控制，不证明性能 |
| Client-LT 拓扑可独立操控 | [Strict topology audit](output/strict_exp1_fresh_topology/paper_notes.md) | fresh partition；拓扑描述可靠 |
| PromptFL 出现 tail-specific gap | [C/D two-seed audit](output/cd_two_seed_summary/cd_result_audit.md) | 两 seed；旧对照不等于 fixed-$n_k$ 因果实验 |
| 视觉 LoRA 也出现退化，support normalization 可部分恢复 | [LoRA 2×2 summary](output/cifar100_LT/ClipLora_SupportNormalized_2x2_seed42/lora_figure_data_summary.md) | full participation、单 seed；跨参数化支持 |
| 语义共现收缩，但 formation chain 不成立 | [V1 report](output/p0_v1_context_colocation_v2/v1_report.md)、[V2 verdict](output/v2_v3_semantic_acquisition/v2_topology_full/v2_joint_summary.md) | 适合用于排除旧故事 |
| stable narrow representation 不成立 | [E1 gate](output/e1_seed42_results_for_analysis/e1_strength_breadth/formal/seed42/analysis/e1_seed42_summary.md) | 单 seed 正式 gate |
| natural Client-LT carriers 的功能覆盖更少 | [Carrier A](output/carrier_access_audit/experiment_a/experiment_a_summary.json) | 20/20 同向，但仍是有混杂的描述性对比 |
| 私有功能信号比语义相似度更可靠 | [Carrier B](output/carrier_access_audit/analysis_b/experiment_b_summary.json) | 80×20 等预算矩阵；支持方法信号 |
| class-absent donors/rewriters 共存 | [Rewrite D1](output/post_write_rewrite_audit/analysis_d1/d1_summary.json) | frozen norm-equalized updates；无完整 turnover |
| 私有风险预测 replay forgetting | [Retention D2](output/post_write_rewrite_audit/analysis_d2/d2_summary.json) | causal replay；不是端到端 trajectory |
| 类别条件聚合具有 topology-specific 净收益 | [SCA factorial spec/results discussion](docs/sca_factorial_experiment.md)、[汇总](ClientLT方法实验与路线汇报.md) | fixed margins、单 seed、DiD +1.819 pp |
| shared substrate 是长期主要退化源 | [Stage 2-C](output/stage2c/stage2c_summary.json) | 单 seed temporal/cross-swap 诊断 |
| CAPT 的优势主要来自 topology robustness | [CAPT gap decomposition](output/online_sca_seed42_v2/stage1b_capt_gap/stage1b_report.md) | 单 seed方向性反例 |

### 18.2 只能作为候选或待补结果

| 结论 | 当前状态 |
|---|---|
| Client-LT 的 Evidence–Rewrite Imbalance 高于 matched Dirichlet，并预测最终退化 | [直接 ERI 闭环协议](docs/evidence_rewrite_imbalance_closure_experiment.md) 已设计；现有 paired local-state dump 仅够单轮 pilot，正式多轮 fixed-marginal 结果待运行 |
| hard-negative co-exposure 在 Client-LT 更低 | 用户最新实验结论与协议一致，但当前工作区缺 `output/boundary_evidence/summary.json` |
| $c+r$ 产生 under-constrained specialization | margin trade-off 可讨论；pairwise-accuracy gate 未通过 |
| under-constrained specialization 导致更低 retention | V1 bridge 无效；必须等待 corrected V2 |
| fixed-marginal topology 导致 Functional Breadth compression | topology breadth Phase 2 只有 manifests，尚无结果 |
| Broad 在 strength matched 后改善 adaptation/retention | feasibility 仅 4/20，当前 gate 未通过 |
| 80-round full participation 为 40.65 vs 53.90 | 数值来自用户新稿，当前工作区尚无对应结果 artifact |
| P-FCC/D-RTC 是有效最终方法 | 当前只有运行契约与 two-round smoke，尚无长程 gate |

### 18.3 明确不能再用作证据

- Compatibility-retention bridge V1 的绝对增益比；它已因 denominator artifact 被正式 supersede，见 [修正说明](tools/compatibility_retention/README.md)。
- “semantic co-location restoration” 的因果链。
- “Client-LT 稳定产生 narrow representation”。
- “所有 class-absent updates 都是干扰”。
- “只要保留 residual row 就保留了类别功能”。

---

## 19. 投稿前最短的证据闭环

如果论文希望把“under-constrained specialization”纳入主机制，而不只是补充分析，最短路径不是继续开发新方法，而是先补两个结果：

1. 同步并审计 `output/boundary_evidence/summary.json`，明确 Experiment A 的 $Q$ 差值、Experiment B 的 $\Delta m_c/\Delta m_h/\Delta\mathrm{pair\mbox{-}acc}$ 以及正式 verdict。
2. 运行 corrected compatibility-retention bridge V2，用 background-adjusted $R_c^*$ 判断 $c+h$ 是否确实比 $c+r$ 更能保留局部增益。

如果 V2 通过，可以把主机制升级为：

\[
\text{topology concentration}
\rightarrow
\text{under-constrained local compatibility}
\rightarrow
\text{rewrite fragility}.
\]

如果 V2 不通过，主论文仍然成立，只需把 formation branch 降级为局部 margin 现象，主机制继续由更扎实的两条证据承担：

\[
\text{aggregation access scarcity}
+
\text{signed shared rewriting}
\rightarrow
\text{global tail erosion}.
\]

此外，正式使用 40.65/53.90 前应同步 full-participation 输出；方法贡献则必须在 P-FCC/D-RTC 或其他最终方案通过真实多轮、多个 seed 和 matched-topology 安全对照后再写入摘要。

---

## 20. 最终可固定的论文主线

最终版本建议固定为以下顺序，不再把任何候选机制提前：

```text
1. Federated long-tailed learning traditionally focuses on class frequency.

2. Federated data has another degree of freedom:
   class-conditioned client-exposure topology under fixed marginals.

3. We control this degree of freedom and discover Client-LT:
   tail evidence is supported by only a few, highly concentrated clients.

4. Matched experiments reveal topology-dependent, tail-specific degradation
   in PromptFL and shared visual LoRA, while class-aware CAPT is a counterexample.

5. Support clients can learn strong tail functions.
   The failure is not simply local inability.

6. Their positive functions receive little aggregation access,
   and support normalization recovers a large fraction of the gap.

7. Meanwhile, class-absent updates have signed effects on the same shared substrate;
   partial participation creates evidence absence, and shared tail competence erodes over time.

8. This creates Evidence–Rewrite Imbalance:
   sparse evidence ownership versus broad shared-update influence.

9. The result is locally learnable yet globally fragile tail adaptation.

10. The method must jointly provide evidence-conditioned access,
    signed functional routing, persistence, and restoration under a clear information boundary.
```

最核心的收束句仍然是：

> **Client-LT 不是故事的起点，而是重新审视联邦长尾二维结构后得到的发现；真正的方法问题也不是尾类能否被少数客户端学到，而是这些稀缺证据产生的功能能否获得足够的全局 access，并在广泛的共享改写中长期存活。**
