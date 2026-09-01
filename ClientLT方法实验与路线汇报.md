# Client-LT 方法实验与路线汇报

> 汇报用途：围绕“我们为理解并解决 Client-LT 做过哪些方法实验、得到什么证据、哪些路线应继续、哪些路线已经不成立”组织。
>
> 推荐用法：正文按 `---` 分隔，每一节可直接对应 1 页 PPT；“页面内容”放入幻灯片，“讲述备注”用于现场汇报，“视觉建议”可直接交给 PPT/图片生成工具。
>
> 故事约束：联邦长尾不仅有类别频率轴，还有客户端拓扑轴。对预训练 VLM，更准确的问题不是“尾类知识能否从头形成”，而是**已有尾类能力经过联邦任务适配后，能够覆盖多少决策边界，以及这些功能能否进入并留在全局模型中。** 第一代 `semantic breadth` 与静态 representation breadth 假说已被否定；当前第二代候选是假设拓扑压缩了 **accessible functional coverage**，并在 shared rewriting 下造成保留风险。该假说仍需严格因果 gate，不能提前写成既定结论。

## 状态标记

- 🟢 **继续推进**：已有较强证据，或是当前最有判别力的下一步。
- 🟡 **保留为基线/边界**：有价值，但不能单独承担最终方法。
- 🔴 **当前停止**：关键 gate 未通过，不应继续作为主方法故事。
- ⚪ **尚待验证**：有协议或代码，但仓库内还没有正式科学结果。

---

## 第 1 页｜这次汇报要回答什么

### 页面内容

**核心问题：在已经证明 Client-LT 会改变尾类学习结果之后，我们究竟应该修复什么？**

围绕方法部分，我们依次回答了五个问题：

1. 预训练尾类能力在联邦适配中是没有被有效利用，还是写入后被持续改写？
2. Client-LT 缺少的是语义邻居，还是更一般的**功能载体冗余**？
3. 不含尾类的客户端是否真的与该类无关？
4. 应该修复局部学习、聚合权重、决策边界，还是知识保存？
5. 最终方法是否需要类别统计；如果不允许类别元数据，替代信号是什么？

### 本页结论

> 我们已经从大量候选解释中，收敛到两个可操作的方法目标：
>
> **A. 提高互补正向功能进入并覆盖尾类困难边界的 access；**
> **B. 在支持客户端缺席时，抑制有害 shared rewriting，保存已经适配出的功能。**

### 视觉建议

中心放置“Client-LT 方法问题”，向外连接五个问题；底部用两条粗箭头汇聚到 `Access` 与 `Preservation`。延续原 PDF 的深蓝标题、橙红核心结论卡片。

### 讲述备注

这一页不要先讲具体算法。先说明我们没有从某个 trick 出发，而是通过一系列有 gate 的实验，把方法目标逐步缩小。汇报的重点是“为什么最后会走到当前方法”，而不是罗列所有跑过的模型。

---

## 第 2 页｜科学定义：固定两个边际，研究联合耦合结构

### 页面内容

设客户端—类别计数矩阵为

\[
N\in\mathbb{R}^{K\times C},\qquad N_{kc}=\text{客户端 }k\text{ 中类别 }c\text{ 的样本数}.
\]

类别总量与客户端总量分别为

\[
n_c=\sum_k N_{kc},\qquad n_k=\sum_c N_{kc}.
\]

类别 \(c\) 在客户端上的暴露分布为

\[
p(k\mid c)=\frac{N_{kc}}{n_c}.
\]

即使固定全部 \(n_c\) 和 \(n_k\)，仍然存在大量不同的 \(N_{kc}\)。我们研究的不是客户端边际 \(p(k)\)，而是固定两个边际后，联合分布 \(p(k,c)\) 的**耦合结构**。

建议术语：

- **class-conditioned client exposure**；
- **client-exposure concentration**；
- **tail-evidence topology**；
- **Client-Exposure Long Tail（CELT）**。

### 本页结论

> “长尾”不只表示某类有多少数据，还表示该类获得多少独立客户端暴露与时间暴露机会。

### 视觉建议

左侧画两个具有相同 row/column sums 的矩阵：一个分散、一个集中；中间写 `same n_c + same n_k`，右侧分别画“多个 carrier”与“少数 specialist”。

### 讲述备注

这里是对原故事“第二根轴”的更严格数学化。不要叫“客户端边际”，因为客户端边际只是 \(n_k\)。我们真正改变的是固定边际后的 coupling/topology。

---

## 第 3 页｜为什么这个耦合结构会改变联邦学习

### 页面内容

改变 \(p(k\mid c)\) 会同时改变：

- 类别 \(c\) 有多少独立知识载体；
- 支持客户端在 FedAvg 中合计占多少权重；
- 部分参与时，一轮出现类别 \(c\) 的概率；
- 连续多少轮可能没有类别 \(c\) 的正向证据；
- 类别 \(c\) 是否长期依赖同一小批客户端；
- 缺失类别 \(c\) 的客户端如何改写其共享表示和决策边界。

因此，Client-LT 同时包含三层效应：

\[
\text{Topology}
\rightarrow
\begin{cases}
\text{Carrier scarcity} \\
\text{Aggregation dilution} \\
\text{Absence + signed rewriting}
\end{cases}
\rightarrow
\text{Tail retention risk}.
\]

### 本页结论

> Client-LT 不是静态计数问题，而是“谁能写入、写入占多大权重、写入后多久得不到修复”的动态问题。

### 视觉建议

制作一条从左到右的机制链。第一列为集中式尾类证据，第二列为三个机制框，第三列为随轮次下降的 tail curve。

---

## 第 4 页｜固定故事主线与后续证据的关系

### 页面内容

原故事中的反事实聚合分解：

\[
26.09\xrightarrow{\text{恢复 FedAvg 权重}}3.45
\xrightarrow{\text{加入全体客户端更新}}0.10.
\]

现在对两个阶段做更精确的定义：

| 原故事 | 后续机制定义 | 方法目标 |
|---|---|---|
| 稀释 dilution | 少量 functional carrier 的全局 access 不足 | 类别条件的支持聚合、持续保存 |
| 干扰 interference | 缺类更新产生有正有负的 signed rewriting | 私有风险判断、保护或恢复 |

需要降级的说法：

- `26.09→3.45→0.10` 是 PromptFL 的强诊断，不是所有 LoRA/数据集上的普适数值定律；
- 但后续 LoRA、carrier 和 rewrite 实验分别为两段机制补充了跨设置证据。

### 本页结论

> 主故事不变，但方法含义从“可靠性加权 + 正交保护”升级为“functional coverage access + adaptation preservation”。

### 视觉建议

沿用原 PDF 的大数字箭头。数字下方增加两行小标签：`carrier access` 与 `signed rewriting`；右侧用绿色卡片写出两个方法目标。

### 证据来源

- [PromptFL Experiment D 旧机制报告](output/cifar100_LT/PromptFL_fedavg_vit_b16_batchSize32/ExperimentD_Main/summary/experimentD_mechanism_conclusion.md)
- [C/D 两 seed 严格审计](output/cd_two_seed_summary/cd_result_audit.md)

---

## 第 5 页｜实验地图：我们如何一步步找到方法目标

### 页面内容

| 实验组 | 要排查的问题 | 结论 |
|---|---|---|
| PromptFL C/D | local learning、稀释还是全局干扰？ | 支持客户端能学，access 和 preservation 均有问题 |
| LoRA 2×2 | PromptFL 结论能否迁移到视觉 LoRA？ | 普通 LoRA 也出现尾类侵蚀，support-normalized 明显恢复 |
| V1/V2/V3 | 语义邻居不共现是否导致形成失败？ | 结构收缩成立，强功能形成链失败 |
| E1/E2 | Client-LT 是否形成更窄的知识？ | “稳定窄表示”不成立 |
| Carrier A/B | 谁是真正有效的知识载体？ | Dirichlet carrier 更宽；语义只富集 donor |
| Placement C | donor 是否必须和 tail 同客户端共现？ | 不必；合并后的 tail-conditioned readaptation 更重要 |
| Rewrite D1/D2 | 缺类更新能否破坏已学知识？风险能否预测？ | signed rewriting 成立；私有风险预测 retention |
| 方法实验 | 哪类修复真正有效？ | 简单类别条件聚合有效；多条复杂路线 gate 失败 |

### 本页结论

> 研究路径完成了两次转向：`语义共现 → 功能边界覆盖`，`知识形成 → 预训练能力的适配与保留`。

### 视觉建议

用一条时间轴，失败 gate 用红色叉号，保留下来的结论用绿色节点；末端汇聚到 `SCA` 和 `private risk control`。

---

## 第 6 页｜实验 A：支持客户端到底会不会学？

### 页面内容

PromptFL 反事实单轮聚合：

| 指标 | Client-LT | Dirichlet |
|---|---:|---:|
| 支持客户端总 FedAvg 权重 | 7.09% | 20.03% |
| support-normalized tail gain | 26.09 | 23.51 |
| 恢复真实权重后的 support gain | 3.45 | 8.86 |
| 加入全客户端后的 tail gain | 0.10 | 1.73 |
| retention ratio | 0.110 | 0.220 |

得到三条证据：

1. Client-LT specialist 并不是学不到尾类知识；
2. 正向尾类功能进入 FedAvg 的权重明显不足；
3. 非支持客户端加入后，剩余尾类增益继续收缩。

### 本页结论

> 第一个方法目标不是“让 specialist 更会学”，而是让少量正向功能有足够的全局 access，并避免随后被覆盖。

### 视觉建议

主图继续使用 `26.09 → 3.45 → 0.10`，下方放 Dirichlet 虚线对照；左上角加一个“support clients can learn”的绿色证据章。

### 讲述备注

不要把 3.45 叫作 local gain-generation deficit。它是样本权重作用后的有效全局贡献，不代表客户端自身不会学习。

---

## 第 7 页｜实验 B：普通视觉 LoRA 是否也存在这个问题？

### 页面内容

CLIP-LoRA 配对 2×2，seed 42：

| 拓扑 / 聚合 | Overall | Head | Tail |
|---|---:|---:|---:|
| Zero-shot | 64.95 | 64.60 | 66.35 |
| Client-LT + FedAvg | 64.68 | 71.66 | 36.75 |
| Client-LT + support-normalized | 65.08 | 69.90 | 45.80 |
| Dirichlet + FedAvg | 66.87 | 71.41 | 48.70 |
| Dirichlet + support-normalized | 66.69 | 71.01 | 49.40 |

关键观察：

- Client-LT FedAvg 相对 Dirichlet FedAvg：tail `−11.95 pp`；
- Client-LT 中 support-normalized：tail `+9.05 pp`；
- 约闭合 69.9% 的拓扑差距；
- 20 个 tail 类中 19 个改善、1 个不变；
- 但 head `−1.76 pp`，仍残留 3.60 pp tail gap。

### 本页结论

> PromptFL 发现可以迁移到共享视觉 LoRA；而 zero-shot `66.35 → 36.75` 表明核心更像已有知识被训练侵蚀，而不仅是知识没有形成。

### 视觉建议

左侧画五组 head/tail 双柱图；右侧画“zero-shot 高位 → FedAvg tail collapse → support-normalized 部分恢复”的三阶段曲线。

### 证据来源

- [LoRA 2×2 汇总](output/cifar100_LT/ClipLora_SupportNormalized_2x2_seed42/lora_figure_data_summary.md)

---

## 第 8 页｜路线检验 1：语义邻居不共现是否是主因？

### 页面内容

### V1：结构证据通过

- Dirichlet 相对 Client-LT 的普通 companion breadth 增量约 `0.44–0.46`；
- 扣除普通 breadth 后，语义邻居仍有约 `0.026–0.041` 的额外共现收缩；
- 结论：`SEMANTIC_SPECIFIC_COLOCATION_SHRINKAGE`。

### V2：功能因果 gate 未通过

- related 相对 unrelated 的纯语义效应均值约 `0.00033`；
- bootstrap 置信区间跨 0；
- 拓扑单轮 replay 没有预期 formation gap；
- 结论：`NO_FUNCTIONAL_SUPPORT`、`NO_STABLE_TOPOLOGY_FORMATION_GAP`；
- joint verdict：`FORMATION_CHAIN_NOT_SUPPORTED`；
- V3 按 fail-closed 规则不再运行。

### 本页结论

> Client-LT 的语义共现结构确实更窄，但“语义邻居不共现 → 尾类知识形成失败”不能作为主方法因果链。

### 路线状态

🔴 **停止把 semantic co-location restoration 作为核心方法。** 语义仍可作为低成本 donor prior，但不能承担安全判断。

### 视觉建议

上半页用两个 client-class 网络图展示共现收缩；下半页放一个红色断裂因果链：`semantic shrinkage ✓ → functional formation gap ✗`。

### 证据来源

- [V1 共现报告](output/p0_v1_context_colocation_v2/v1_report.md)
- [V2 joint summary](output/v2_v3_semantic_acquisition/v2_topology_full/v2_joint_summary.md)

---

## 第 9 页｜路线检验 2：Client-LT 是否形成“更窄的知识”？

### 页面内容

E1 strength/breadth 正式结论：`SEED42_STRONG_BUT_NOT_NARROW`。

- strength gate 通过；
- own-peak breadth gate 未通过；
- majority-family gate 未通过；
- accuracy-controlled breadth 未通过；
- 只有 multiview family 较一致；
- visual subgroup、neighbor breadth 均不稳定。

### 本页结论

> 不能把 Client-LT 稳定描述为“学出的类别表示更窄”。观察到的最终风险不能由一个统一的 narrow-representation 故事解释。

### 路线状态

🔴 **停止把 broad representation regularization 作为主方法依据。**

### 视觉建议

做一个 gate 面板：Strength 为绿色通过，其余 breadth gates 为红色未通过；右侧放“strong ≠ broad / narrow”的概念示意。

### 证据来源

- [E1 strength/breadth 报告](output/e1_seed42_results_for_analysis/e1_strength_breadth/formal/seed42/analysis/e1_seed42_summary.md)

---

## 第 10 页｜实验 C：真正缺少的是 functional carrier 冗余

### 页面内容

Carrier Experiment A：`DIRICHLET_CARRIERS_FUNCTIONALLY_BROADER`。

Dirichlet 相对 Client-LT：

- effective carrier count：`+3.741`，20/20 tail 类方向一致；
- carrier union coverage：`+0.1385`，20/20；
- tail-mass weighted positive all-class margin coverage：`+0.1538`，20/20；
- unseen coverage：`+0.1696`，20/20；
- positive-gain entropy：`+0.0793`，20/20；
- worst-neighbor gain：16/20 支持；
- cross-carrier cosine diversity 仅 4/20 支持，不能泛化为“所有多样性都更高”。

### 本页结论

> Client-LT 真正稳定缩减的是能对尾类产生正向功能的独立 carrier 数量和覆盖宽度，而不只是语义 companion 数量。

### 方法启示

1. 不能只保护“含有该类标签的客户端”；
2. 还应允许功能上有益的缺类 donor 参与；
3. 但 donor 必须通过功能证据筛选，不能仅凭语义相似。

### 视觉建议

左侧画 Dirichlet 下多个绿色 carrier 围绕 tail 类，右侧画 Client-LT 下少数 carrier；中间用五个小指标卡展示 count/coverage/entropy。

### 证据来源

- [Carrier A summary](output/carrier_access_audit/experiment_a/experiment_a_summary.json)

---

## 第 11 页｜实验 D：语义相似度只能富集 donor，私有功能评估更可靠

### 页面内容

Carrier Experiment B：80 个非尾类候选 × 20 个尾类的等预算 signed-transfer 矩阵。

### 公共语义信号

- related 相对 unrelated 的 positive donor rate：平均 `+0.27`，18/20；
- mean test margin：16/20 为正；
- worst-neighbor 不稳定，均值略负；
- semantic similarity 与真实功能效应 Spearman 仅约 `0.148`。

### 客户端私有功能信号

- private-tail-train proxy 与 test effect Spearman 约 `0.837`，20/20 为正；
- private-selected donor 的 test margin gain 约 `+0.001274`，20/20；
- 候选预算达到 10 后，20/20 类均能找到正 donor；预算为 1 时只有 55%。

### 本页结论

> Semantic similarity is a proposal prior, not a safety certificate. 客户端私有小样本功能评估才是 donor/rewriter 选择的核心信号。

### 路线状态

- 🟡 语义检索：保留为候选压缩器；
- 🟢 私有功能 gate：继续作为无类别元数据方法的核心。

### 视觉建议

画一个漏斗：CLIP semantic prior 先筛出候选，private functional test 再把候选分为绿色 donor 与红色 rewriter；突出 `0.148 vs 0.837`。

### 证据来源

- [Carrier B summary](output/carrier_access_audit/analysis_b/experiment_b_summary.json)

---

## 第 12 页｜实验 E：donor 必须与 tail 在同一客户端吗？

### 页面内容

Placement Experiment C 对比：

1. tail 与 related 在同一客户端联合训练；
2. tail 与 related 分开训练后等权合并；
3. 分开合并后，再用 tail 私有证据 readapt。

结果：`PARTIAL_PLACEMENT_SUPPORT`。

- joint-related 相对 joint-unrelated：margin/NLL 仅 13/20 支持，accuracy 无优势；
- joint-related 相对 separate equal merge：基本为零或略负；
- separate readapt 相对 separate merge：margin/NLL 20/20 改善；
- worst-neighbor 17/20 改善，accuracy 基本不变。

### 本页结论

> 有益知识不必在同一客户端物理共现；更重要的是 donor 合并之后，利用 tail 私有证据完成 tail-conditioned readaptation。

### 路线状态

🔴 不再追求人为恢复 semantic co-location。  
🟢 继续研究 server merge 后的 private selection / restoration。

### 视觉建议

三路流程图：Joint、Separate Merge、Separate + Readapt；第三路用绿色高亮并标注 20/20。

### 证据来源

- [Carrier C summary](output/carrier_access_audit/analysis_c/experiment_c_summary.json)

---

## 第 13 页｜实验 F：类别写入后，缺类客户端会怎样改写它？

### 页面内容

Post-write D1：先明确写入 tail 功能，再测试缺类客户端更新。

- direct tail write test margin：`+0.00709`，20/20 类为正；
- 每个 tail 类都同时存在 donor 与 rewriter；
- 平均 donor 数：`47.25`；
- 平均 rewriter 数：`32.75`；
- private/test post-effect Spearman：`0.750`；
- sign agreement：`0.808`；
- private rewriter recall：`0.791`；
- private donor precision：`0.846`；
- false-safe rate：`0.154`。

没有出现预注册要求的完整 donor-to-rewriter turnover，因此正式结论为：

`POST_WRITE_REWRITE_SUPPORTED_WITHOUT_FULL_TURNOVER_CHAIN`。

### 本页结论

> 缺类客户端不是统一的噪声源：它们有些能帮助尾类，有些会删除已写入功能；方法需要做 signed routing，而不是一律拒绝缺类更新。

### 视觉建议

中心放一个已经写入的 tail knowledge 节点；左侧绿色 donor 箭头、右侧红色 rewriter 箭头同时指向节点。避免画成“所有 non-support clients 都有害”。

### 证据来源

- [D1 summary](output/post_write_rewrite_audit/analysis_d1/d1_summary.json)
- [D1 transition matrix](output/post_write_rewrite_audit/analysis_d1/d1_transition_matrix.csv)

---

## 第 14 页｜实验 G：私有 rewrite risk 能否预测后续遗忘？

### 页面内容

Post-write D2：固定、范数匹配的客户端更新 replay。

- low-risk 相对 blind replay 的 forgetting advantage：`+0.0002607`，20/20；
- high-risk 相对 blind 更差：`+0.0002798`，19/20；
- blind private risk 与真实 forgetting Spearman：约 `0.505`，19/20 为正；
- verdict：`REWRITE_RISK_PREDICTS_RETENTION`。

证据边界：

- 更新已经冻结并进行范数公平控制；
- 这是 causal replay，不是完整联邦再训练；
- 证明 risk signal 存在，但仍需端到端 trajectory 验证。

### 本页结论

> 客户端不需要公开类别列表，也能用少量私有数据判断某个外来更新是否会伤害自己关心的功能。

### 方法启示

这为两类操作提供依据：

1. **选择**：优先吸收低风险 donor；
2. **恢复**：发现 incoming global 退化时，触发局部 restoration。

### 视觉建议

画三条 replay 曲线：low-risk、blind、high-risk；右侧用散点图表达 private risk 与 forgetting 正相关。

### 证据来源

- [D2 summary](output/post_write_rewrite_audit/analysis_d2/d2_summary.json)

---

## 第 15 页｜从证据自然推出的方法设计要求

### 页面内容

一个真正针对 Client-LT 的方法至少要满足四点：

### 1. Access

少量 supporter 产生的正向类功能不能被客户端总样本量自动压小。

### 2. Persistence

某轮没有 supporter 时，不应让该类别的状态被无证据更新覆盖；应该保留最近可信状态。

### 3. Signed transfer

缺类客户端不能一律排除，因为其中存在 donor；也不能一律接收，因为其中存在 rewriter。

### 4. Information boundary

必须明确方法需要哪类信息：

- 类别计数可见；
- 只有 support bit；
- 完全不上传类别元数据，只做客户端私有功能判断。

### 本页结论

> 方法目标不是笼统“增强尾类”，而是建立一条类别条件的 `write → aggregate → persist → restore` 生命周期。

### 视觉建议

用环形生命周期图，四个节点分别为 Write、Aggregate、Persist、Restore；中心写 `rare-knowledge lifecycle`。

---

## 第 16 页｜已经尝试的方法：哪些提供了正证据

### 页面内容

### 🟡 Support-normalized / class-wise aggregation

正证据：

- Client-LT LoRA tail `+9.05 pp`；
- 19/20 tail 类改善；
- 证明 access correction 是有效干预。

不足：

- head `−1.76 pp`；
- 仍残留 3.60 pp tail gap；
- 某轮 supporter 缺席时没有 preservation 机制；
- 需要类别支持信息。

### 🟡 CAPT

- matched 三 seed 中，Client-LT tail 与 Dirichlet 基本持平；
- 说明类别条件机制确实可以对 Client-LT 稳定；
- 也是“问题并非对所有算法都必然困难”的重要反例。

不足：

- 使用类别信息和非标准聚合；
- 应作为强基线和信息上界，而不是直接当作我们的方法贡献。

### 本页结论

> 简单类别条件处理已经证明“可以救”，但尚未同时解决 access、缺席保存、signed transfer 与信息约束。

### 视觉建议

用两个黄色方法卡片。每张卡片上半部绿色“已解决”，下半部橙色“未解决”。

---

## 第 17 页｜已经尝试的方法：哪些路线当前不再继续

### 页面内容

| 路线 | 原始想法 | 关键结果 | 当前状态 |
|---|---|---|---|
| Semantic co-location restoration | 把 tail 与语义邻居重新放在一起 | V2 功能链失败；Placement 不支持必须同客户端共现 | 🔴 停止 |
| Broad representation regularization | Client-LT 学到更窄表示，需要拓宽 | E1 breadth gates 未通过 | 🔴 停止 |
| Functional CUSP | 用功能/几何预测优化客户端标量权重 | 预测相关性存在，但双拓扑 gate 输给简单 class-wise | 🔴 当前版本停止 |
| Boundary repair | 直接修复脆弱 tail/non-tail 边界 | 可闭合目标 deficit，但所有安全候选被 non-target gate 拒绝 | 🔴 停止 |
| FedTEF 大系统继续叠组件 | observer、memory、TailAgg、semantic rescue、fusion | acquisition、routing、fusion 多处瓶颈，未形成稳定 Client-LT 解 | 🔴 不再堆组件 |

### 本页结论

> 失败实验的共同教训：不能从一个代理指标直接推出复杂修复；最终干预必须作用在已经验证的 access 或 retention 因果环节上。

### 视觉建议

用五条路线组成漏斗，四周复杂分支逐渐变灰，最终只保留底部两条绿色出口：`class-conditional persistence` 与 `private risk control`。

---

## 第 18 页｜CUSP、边界修复与 FedTEF 分别教会了我们什么

### 页面内容

### CUSP：oracle 上界存在，但可部署预测不足

- 早期 Client-LT round-10：FedAvg `17.6`、class-wise `26.9`、oracle CUSP `37.8`；
- 后续 functional CUSP：Client-LT `19.05`，明显低于 class-wise `26.9`；
- predicted-realized Spearman 约 `0.76–0.80`，说明几何有信号，但标量重加权表达能力不足。

### Boundary repair：局部可修不等于全局安全

- solver 能修复目标 deficit；
- 但所有候选都因 non-target safety 或 boundary reversal 被拒绝；
- 说明共享参数中的边界耦合难以用局部补丁安全隔离。

### FedTEF：保持一个 tail stream 不等于保住功能

- memory 可以记录证据；
- TailAgg 未稳定改善 retention；
- semantic rescue 和 fusion 可能伤害 tail；
- oracle tail identity 明显更强，暴露 identification/routing bottleneck。

### 本页结论

> 我们需要的是结构上可分离的类别状态，或由客户端私有功能证据控制的更新路径，而不是更复杂的全局标量权重和融合系统。

### 证据来源

- [CUSP oracle summary](output/cusp_minimal_refactor_20260801_163123/cusp_eval_client-longtail_seed42_round10/oracle_summary.md)
- [Functional CUSP gate](output/functional_cusp_gate_seed42/summary/two_topology_gate_summary.json)
- [Boundary repair 协议](docs/visual_semantic_boundary_repair_v1_experiment_spec.md)
- [FedTEF 诊断协议](docs/fedtef_v5_diagnostic_loop.md)

---

## 第 19 页｜Online SCA：已完成的类别条件隔离对照

### 页面内容

共享 LoRA 继续执行普通 sample-weighted FedAvg；每个 tail 类增加一个零初始化、feature-conditioned residual row。对类别 \(c\)，只有本地正标签出现时才允许该 row 产生梯度，服务端只在当轮 supporter 间聚合：

\[
r_c^{t+1}=
\begin{cases}
r_c^t+\displaystyle\sum_{k\in S_c^t}
\frac{N_{kc}}{\sum_{j\in S_c^t}N_{jc}}\Delta r_{k,c}^t,
& S_c^t\neq\varnothing,\\[1.2em]
r_c^t, & S_c^t=\varnothing.
\end{cases}
\]

它能够严格保证：

- supporter-only row aggregation：隔离类别条件 access/dilution；
- non-supporter 不直接更新该 residual row：阻断 row-level rewriting；
- no-support round：残差参数原样保留。

它不能保证：

- shared LoRA 中的尾类功能不被改写；
- class-absent positive donor 被 residual stream 利用；
- 参数保留等价于功能保留。

### 路线状态

🟡 **保留为 support-metadata isolation control / relaxed-information mechanistic reference。**
不再把它称为最终方法或性能 upper bound。

### 本页结论

> SCA 精确回答“显式类别 row 隔离能修复多少”，但它没有保护 shared functional substrate。

---

## 第 20 页｜SCA 2×2 已完成：聚合收益具有拓扑特异性，但幅度有限

### 页面内容

固定 \(n_k\)、\(n_c\)、residual architecture、初始化、训练预算和参与 schedule；正式配置为 `30 clients, frac=0.4, 80 rounds, seed=42`。

以最终 round 的 Head–Tail H-mean 为主指标：

| 拓扑 | Residual-FedAvg | SCA | SCA 净收益 |
|---|---:|---:|---:|
| Client-LT | 51.802 | 53.346 | **+1.544** |
| Fixed-marginal Dirichlet | 62.496 | 62.221 | **−0.275** |

\[
\mathrm{DiD}=1.544-(-0.275)=\mathbf{+1.819\ pp}.
\]

协议审计通过：两个边际完全一致、四个 cell 内训练协议一致、只改变 topology 与 residual aggregation。

### 证据边界

- 这是单 seed 描述性结果，不能代替多 seed 推断；
- SCA 的独立聚合贡献存在，但不是大幅性能救援；
- Client-LT 上相对 CAPT 仍有显著 tail/H-mean 差距；
- matched Dirichlet 上安全回退尚未实现，静态 SCA 略有负收益。

### 本页结论

> SCA 证明 class-separable aggregation 对 Client-LT 有特异性作用，但不能独自解释或解决主要拓扑损失。

### 证据来源

- [SCA factorial four-cell runs](output/online_sca_seed42_v2/)

---

## 第 21 页｜D4 与 Stage 2-C：参数保留不等于功能保留

### 页面内容

### D4：partial participation 下 absence 真实发生

- 正式 SCA 使用 `frac=0.4`，不是 full participation；
- 1600 个 tail-class×round 中出现 235 个 no-support 事件；
- no-support 时 residual row delta 精确为 0；
- 但 235/235 的 margin 仍发生变化，152 次 accuracy 下降、160 次 margin 下降；
- absence 事件平均 accuracy delta 为 `−1.719 pp`，margin delta 为 `−0.1405`。

### Stage 2-C：退化主要来自 shared LoRA 的头类化

清零 residual 后：

| checkpoint | Head | Tail | H-mean |
|---|---:|---:|---:|
| round 3 | 67.70 | 68.80 | 68.25 |
| round 20 | 70.75 | 26.30 | 38.35 |
| round 50 | 73.49 | 22.80 | 34.80 |
| round 80 | 73.90 | 22.10 | 34.02 |

round 3→80：20/20 tail accuracy 与 20/20 tail margin 均下降；head 平均提高 `6.20 pp`，tail 平均下降 `46.70 pp`。

与此同时，matched residual 在 round 20/50/80 持续提供约 `19–20 pp` H-mean 补偿。因此 residual 不是主要退化源，而是在补救已经变得 head-dominant 的 shared substrate。

### 本页结论

> 当前最强事实是 **shared functional adaptation scope erosion**：尾类能力早期已经存在，随后被共享更新系统性压缩。no-support policy 保存了参数，却没有保存其依赖的功能空间。

### 证据来源

- [Stage 2-C summary](output/stage2c/stage2c_summary.json)
- [Stage 2-C cross-swap](output/stage2c/cross_swap_summary.csv)
- [Stage 2-C aggregation stages](output/stage2c/aggregation_stage_summary.csv)

---

## 第 22 页｜第二代假说：Semantic Breadth → Functional Coverage Breadth

### 页面内容

第一代假说已经被分解并否定：

- V1：semantic co-location shrinkage 成立；
- V2：semantic relatedness → functional formation 不成立；
- E1：静态 embedding geometry narrowness 不成立。

第二代候选假说不再问“谁与 tail 语义相似或物理共现”，而是问：

> 在整个联邦适配过程中，tail class 能够利用多少**互补的正向功能方向**，这些方向覆盖多少不同的困难决策边界？

现有支持：

- Carrier A：Dirichlet 的 effective carrier 与 functional coverage 更宽，但仍是有混杂的描述性对比；
- Carrier B：semantic–effect Spearman 约 `0.148`，private functional signal–test effect 约 `0.837`；
- Placement C：物理共现不是必要条件，merge 后 tail-conditioned readaptation 有效；
- D1/D2：class-absent donor/rewriter 共存，private risk 能预测 replay forgetting；
- Stage 2-C：shared tail competence 会在长期聚合中系统性收缩。

### 当前允许使用的表述

> **Client-LT may compress the set of complementary functional directions accessible to tail adaptation, yielding boundary-incomplete competence that is more vulnerable to subsequent shared rewriting.**

### 当前不允许使用的表述

> “Client-LT 已经被证明通过 Functional Breadth Scarcity 导致遗忘。”

### 本页结论

> Functional Breadth 是被失败实验筛出的高价值候选机制，不是对 Semantic Breadth 的换名包装。

---

## 第 23 页｜怎样把“功能强度”和“功能广度”严格分开

### 页面内容

对 tail class \(c\)，使用**训练侧冻结**的 hard-negative boundary 集合

\[
\mathcal H_c=\{h_1,\ldots,h_M\}.
\]

候选 donor 集合 \(S\) 经过实际服务器合并得到 \(u_S\)。必须测量真实合并后的边界变化，而不只使用各 donor 的理想最大值：

\[
g_{S,c,h}=m_{c,h}(\theta+u_S)-m_{c,h}(\theta).
\]

总正功能强度定义为

\[
G_c(S)=\sum_h w_h[g_{S,c,h}]_+.
\]

将正增益归一化为

\[
q_h=\frac{w_h[g_{S,c,h}]_+}
{\sum_{h'}w_{h'}[g_{S,c,h'}]_+},
\]

再用有效覆盖数表示 breadth：

\[
B_c^{\mathrm{eff}}(S)=\exp\!\left(-\sum_hq_h\log q_h\right).
\]

实验目标是在严格匹配 \(G_c\) 后，只改变 \(B_c^{\mathrm{eff}}\)：

- Broad：相同总 gain 分散覆盖更多 hard boundaries；
- Narrow：相同总 gain 集中在少数 boundaries。

原始 `max-over-donors` 指标可保留为 **potential set coverage**，但不能替代 actual merged coverage，因为 shared LoRA 合并后可能发生抵消。

### 本页结论

> 只有在 strength 相同而 coverage 不同的条件下，才能证明“广度”本身有额外功能价值。

---

## 第 24 页｜Functional Breadth 生死 gate：同一实验完成适配与保留检验

### 页面内容

### Stage A：Topology → Breadth 审计

- 使用 factorial 中完全匹配边际的 Client-LT 与 Dirichlet；
- 从相同 global LoRA、相同训练预算生成 class-absent candidate updates；
- 使用训练侧/private probe 冻结 hard negatives 和 gain vectors；
- 检验 Client-LT 是否在 actual merged functional coverage 上系统性更窄。

这一步补上 Carrier A 尚未严格控制的箭头。

### Stage B：Matched Broad vs Narrow adaptation

从共同候选池构造两组 donor，严格匹配：

- donor 数量、客户端质量与聚合 mass；
- 总 update norm、optimizer steps；
- 总正增益 \(G_c\)；
- signed negative harm/head safety；
- 最好匹配与 direct-tail update 的 cosine。

两组执行实际 merge，再执行完全相同的 tail readaptation。测试集只在全部选择冻结后评估 accuracy、mean margin、worst-boundary margin。

### Stage C：Matched rewrite replay

- 不能根据测试结果事后把 Broad/Narrow 调齐；
- 只能在 private calibration 上设置初始性能 caliper；
- 对两者施加完全相同的 frozen class-absent rewrite sequence；
- 报告 forgetting trajectory、AUC、survival rounds 与相对初始值退化。

### 四种判定

1. Broad 初始更好且更稳定：统一 adaptation + retention 机制；
2. 初始接近、Broad 更稳定：只保留 robustness/retention breadth；
3. Broad 初始更好、retention 相同：只保留 adaptation coverage；
4. 二者均无差异：停止 Functional Breadth 主线。

### 本页结论

> 在完整长程方法开发前，先用一个严格匹配的因果实验决定 Functional Breadth 的生死。

---

## 第 25 页｜若 gate 通过，主方法如何自然产生

### 页面内容

### SCA 的身份

- 保留为 support-metadata isolation control；
- 证明显式类别 row 隔离能修复一部分 topology-specific damage；
- 不称为 upper bound，因为它没有保护 shared LoRA，也排除了 class-absent donor。

### P-FCC 的最小升级

当前版本选择 top-2 scalar positive donor。若 breadth gate 通过，改为互补选择：

\[
j_1=\arg\max_j G_c(j),\qquad
j_2=\arg\max_j \Delta B_c^{\mathrm{eff}}(j\mid j_1).
\]

即第二个 donor 不再追求第二大的标量 utility，而是补足第一位没有覆盖的困难边界。方法身份从 Positive Donor Selection 升级为 **Complementary Functional Consolidation**。

### Retention 组件的条件

只有 Stage C 证明 Broad/Narrow 的保留差异，或证明 private rewrite risk 能在真实长程轨迹中控制退化，才开发 D-RTC/Rewrite-Aware Retention；否则不把 retention 组件拼装进方法。

### 隐私约束

Functional Breadth gate 可以作为离线机制实验使用细粒度 gain vectors；最终方法不能默认上传带类别/边界含义的 \(g_{j,c,h}\)。互补选择必须在客户端私有完成，或使用安全聚合/匿名压缩信号，否则会破坏 no-class-metadata 叙事。

### 本页结论

> Gate 通过后，主方法应直接作用于正在退化的 shared functional stream，而不是继续堆 residual 参数。

---

## 第 26 页｜当前证据边界：已经证明什么，还没有证明什么

### 已经证明/强支持

- 固定边际后，coupling topology 是独立实验变量；
- Client-LT 下 functional carrier 数量与自然 coverage 更窄；
- semantic similarity 只提供弱 donor prior，private functional evidence 更可靠；
- class-absent 更新具有 signed functional effect；
- partial participation 产生真实 temporal absence；
- residual parameter retention 不能阻止 shared functional rewriting；
- shared LoRA 的尾类能力从早期强状态系统性退化为 head-dominant 状态。

### 尚未证明

- fixed-marginal topology 必然导致 functional breadth compression；
- functional breadth 在总 gain 匹配后仍提高 tail adaptation；
- low breadth 会因果性地导致更强 rewrite fragility；
- complementary donor selection 在真实多轮 FL 中优于 top-scalar/random；
- no-metadata 实现能在不泄露边界 utility 的条件下稳定工作。

### 本页结论

> 论文现在有一个有潜力的统一解释，但还缺决定性的中介变量干预，不应把相关证据提前升级为完整机制。

---

## 第 27 页｜从现在开始的最短闭环

### 页面内容

1. **冻结 Functional Breadth gate 协议，不开发新训练组件。**
2. 先做 feasibility：现有 candidate updates 中能否为足够多 tail 类找到 strength/norm/budget 匹配、breadth 显著不同的 Broad/Narrow pair。
3. feasibility 通过后，执行 Stage A 的 exact-topology breadth audit。
4. 再执行 Stage B adaptation 与 Stage C frozen rewrite replay；全程禁止 test-driven selection。
5. 只有 adaptation gate 通过，才实现 complementary P-FCC；只有 retention gate 通过，才实现 D-RTC。
6. 新方法在 seed 42 通过后，才进入多 seed、普通 Dirichlet 与自然异质性泛化。

### 明确停止条件

- 无法在匹配总 gain/norm/budget 后制造 breadth separation：当前定义不可操作，停止；
- Broad/Narrow adaptation 与 retention 都无差异：停止 Functional Breadth；
- 只有 potential max coverage 不同、actual merged coverage 不变：停止当前 donor 合并表达；
- complementary selection 不超过 top-scalar/random：不升级 P-FCC；
- retention replay 通过但真实长程不通过：D-RTC 不能成为主组件。

### 本页结论

> 下一步不是再跑一条 80/100-round 方法，而是先用已有更新与 replay machinery 完成一个高判别力、可失败的机制 gate。

---

# 附录 A｜方法实验总表

| 实验/方法 | 主要问题 | 证据级别 | 结果摘要 | 对方法的影响 |
|---|---|---|---|---|
| PromptFL C/D | local learning vs dilution/interference | 两 seed + 反事实聚合 | supporter 能学；真实权重和全体聚合继续压缩增益 | 保留 access + preservation 主线 |
| LoRA 2×2 | 是否迁移到视觉 LoRA | 单 seed、配对完整训练 | Client-LT tail −11.95；support norm +9.05 | 证明 LoRA 中也有强侵蚀；需多 seed |
| CAPT matched | 强 class-aware 方法是否稳定 | 三 seed | Client-LT 与 Dirichlet 基本持平 | 强基线；说明问题可被类别条件机制缓解 |
| V1 | semantic co-location 是否收缩 | 两 seed + class bootstrap | 结构收缩成立 | 仅作为结构描述 |
| V2/V3 | semantic shrinkage 是否导致 formation gap | 受控功能实验，fail-closed | 功能链失败，V3 不运行 | 停止 semantic restoration 主路线 |
| E1 | 是否形成更窄知识 | 单 seed正式 gate | strong but not narrow | 停止 broad-representation 主路线 |
| Carrier A | carrier 数量/覆盖是否缩减 | 20 类自然拓扑描述；仍有 composition/exposure 混杂 | Dirichlet carrier 更宽，coverage 指标 20/20 同向 | 支持 Functional Breadth 候选；不能单独证明 topology 因果 |
| Carrier B | donor 如何识别 | 80×20 等预算矩阵 | 语义弱、私有功能强 | 支持 private functional gate |
| Placement C | 是否必须同客户端共现 | 三种 placement 对照 | separate merge 后 readaptation 的 margin/NLL 20/20 改善；总体 verdict 为 partial support | 物理共现非必要候选解释；避免普适化 |
| Rewrite D1 | 缺类更新是否能改写已写入类 | post-write 全候选扫描 | donor 与 rewriter 共存；完整 turnover chain 未通过 | 需要 signed routing，但不宣称完整状态翻转链 |
| Rewrite D2 | 私有风险能否预测遗忘 | 范数匹配 replay | risk predicts retention | 支持 protection/restore |
| Support-normalized | access correction 是否有效 | LoRA 2×2 | 大幅恢复 tail，有 head 代价 | 必须保留为强基线 |
| CUSP | 标量功能重加权能否解决 | oracle + 双拓扑 gate | oracle 强，functional 版本输给 class-wise | 当前版本停止 |
| Boundary repair | 能否直接安全修边界 | safety gate | 目标可修，非目标不安全 | 停止 |
| FedTEF | memory/tail stream/fusion 是否解决 | 多版本组件实验 | 多处 acquisition/routing/fusion 瓶颈 | 不再堆大系统 |
| P-FCC/D-RTC | 无类别元数据能否私有控制 | 两轮 smoke | 仅证明运行正确 | 暂停长程开发，等待 breadth/retention gate |
| Online SCA factorial | 类别条件聚合是否具有 topology-specific 净收益 | fixed-marginal 2×2，单 seed | Client-LT H-mean `+1.544`，matched `−0.275`，DiD `+1.819` | 保留为机制对照，不作为最终主方法 |
| D4 absence | no-support 时参数与功能是否同时保留 | 235 个 no-support 事件 | residual row 不变，但功能继续变化并倾向下降 | 证明 parameter persistence ≠ functional persistence |
| Stage 2-C | shared 与 residual 谁导致后期退化 | 四时点 cross-swap + 三阶段分解，单 seed | shared-only tail `68.8→22.1`；residual 后期稳定补偿约 `19–20` H-mean | 转向 shared functional stream |
| Functional Breadth gate | strength 匹配后 coverage 是否影响适配与保留 | 待冻结协议 | 尚无因果结果 | 当前唯一第一优先级 |

---

# 附录 B｜汇报中建议使用与避免使用的表述

## 建议使用

- “固定类别总量和客户端总量后，改变联合分布的 coupling structure。”
- “自然 Client-LT 对比中，尾类 functional carrier redundancy 与 coverage 更低；fixed-marginal 因果关系仍待验证。”
- “支持客户端能够学习，但其正向功能的 aggregation access 不足。”
- “缺类更新具有 signed rewriting：既存在 donor，也存在 rewriter。”
- “私有功能风险可以预测后续 retention。”
- “当前最可信事实是 functional access 不足与 shared adaptation erosion；Functional Breadth 是下一项待检验中介变量。”
- “SCA 是 support-metadata isolation reference；P-FCC/D-RTC 对应待验证的 no-class-metadata 路线。”
- “对预训练 VLM，应描述 task-specific functional adaptation scope compression，而不是知识从未形成。”

## 避免使用

- “客户端边际长尾”——我们改变的不是 \(p(k)\)，而是固定边际后的 \(p(k,c)\) 耦合。
- “语义邻居不共现导致尾类知识无法形成”——V2 不支持。
- “Client-LT 一定学出更窄的表示”——E1 不支持。
- “所有缺类客户端都在干扰”——D1 表明 donor/rewriter 共存。
- “support-normalized 已经完全解决”——仍有 residual gap、head trade-off 和 absence 问题。
- “SCA 已经解决功能保留”——SCA 只保留 residual 参数，Stage 2-C 表明 shared 功能仍会退化。
- “Functional Breadth 已经被证明”——Carrier A/B/C 与 D1/D2 只构成兼容证据，缺少 matched breadth intervention。
- “SCA 是性能 upper bound”——它排除了 class-absent donor，且不保护 shared LoRA，只能作为 relaxed-information mechanism reference。
- “Client-LT 对所有模型都更难”——CAPT 是反例。

---

# 附录 C｜核心证据索引

- [原始汇报 PDF](客户端长尾日常讨论.pdf)
- [全局长尾一致性](output/experiment1_global_longtail_verification/paper_notes.md)
- [严格 Client-LT 拓扑](output/strict_exp1_fresh_topology/paper_notes.md)
- [PromptFL C/D 审计](output/cd_two_seed_summary/cd_result_audit.md)
- [LoRA support-normalized 2×2](output/cifar100_LT/ClipLora_SupportNormalized_2x2_seed42/lora_figure_data_summary.md)
- [V1 semantic co-location](output/p0_v1_context_colocation_v2/v1_report.md)
- [V2 formation-chain verdict](output/v2_v3_semantic_acquisition/v2_topology_full/v2_joint_summary.md)
- [E1 strength/breadth](output/e1_seed42_results_for_analysis/e1_strength_breadth/formal/seed42/analysis/e1_seed42_summary.md)
- [Carrier A](output/carrier_access_audit/experiment_a/experiment_a_summary.json)
- [Carrier B](output/carrier_access_audit/analysis_b/experiment_b_summary.json)
- [Placement C](output/carrier_access_audit/analysis_c/experiment_c_summary.json)
- [Rewrite D1](output/post_write_rewrite_audit/analysis_d1/d1_summary.json)
- [Retention D2](output/post_write_rewrite_audit/analysis_d2/d2_summary.json)
- [Online SCA](docs/online_sca_d4a.md)
- [SCA factorial](docs/sca_factorial_experiment.md)
- [SCA factorial four-cell runs](output/online_sca_seed42_v2/)
- [D4 per-class absence trajectory](output/d4a_per_class_round.csv)
- [Stage 2-C summary](output/stage2c/stage2c_summary.json)
- [Stage 2-C cross-swap](output/stage2c/cross_swap_summary.csv)
- [Stage 2-C aggregation stages](output/stage2c/aggregation_stage_summary.csv)
- [P-FCC / D-RTC](docs/frozen_p_fcc_d_rtc_method_contract_v1.md)
