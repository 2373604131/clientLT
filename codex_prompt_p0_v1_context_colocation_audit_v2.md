# Codex 实现 Prompt（修订版）：P0 训练语义审计与 V1 拓扑—局部语境共现审计

将下面整段 Prompt 交给项目仓库根目录中的 Codex。它取代旧版 `codex_prompt_training_semantics_audit_v1.md`。

---

```text
你需要在当前联邦长尾项目仓库中完成两个连续但证据层级不同的任务：

1. P0：把视觉 LoRA 联邦训练的真实语义固化成可复核的审计报告；
2. V1：在不训练模型的前提下，比较 matched Dirichlet 与 Client-LT 的尾类证据拓扑、一般局部共现语境，以及 CLIP 语义邻居共现是否发生超出一般语境稀疏化的特异性收缩。

本任务不得被命名或解释为“尾部知识形成验证”“正迁移验证”或“语义知识迁移实验”。V1 只分析 partition 中真实正样本的共现位置与剂量。相关类别是否真的帮助尾类，必须由后续 V2 的梯度或受控反事实训练验证。

不要启动任何联邦训练，不要实现 Semantic-Scoped LoRA、V2、V3、V4，不要修改训练、划分、损失或聚合行为。

一、研究问题与证据边界

分别回答三个问题，不得合并：

RQ1 — Topology concentration
在相同 CIFAR-100-LT 全局样本池和类别计数下，Client-LT 是否减少 bottom-20 尾类的支持客户端数、提高 Top-2 client mass，并降低有效支持客户端数？

RQ2 — Generic context shrinkage
按照尾类样本实际落点加权后，Client-LT 中尾类正样本所在客户端是否具有更少的 companion classes、companion samples 和本地正样本暴露剂量？

RQ3 — Semantic-specific co-location shrinkage
Client-LT 中 CLIP 文本语义邻居的本地共现下降，是否超过由频率匹配随机 companion 类集合所反映的一般语境收缩？

预先锁定但不得预设成立的假设：

- H1：Client-LT 的尾类支持更集中；
- H2：Client-LT 的 tail-mass-weighted generic context breadth/dose 更低；
- H3：在扣除 frequency-matched random null 后，Client-LT 的 CLIP-neighbor co-location excess 仍低于 Dirichlet。

V1 最强只允许支持以下表述：

“Client-LT is associated with reduced local co-location of semantically related positive examples beyond generic context sparsification.”

不得由 V1 推出：

- CLIP 相近类别对尾类产生了正梯度迁移；
- 语义相关类别使尾类形成了更丰富的 LoRA 知识；
- Client-LT 因语义信息不足而导致准确率下降；
- 类别越多越有利；
- 同客户端共现等于相关类更新是否进入全局模型；
- CLIP 文本相似度等于视觉 LoRA 梯度兼容性。

二、已经掌握、但仍需在当前活跃代码路径上复核的仓库事实

此前只读检查给出了以下事实。不要依赖历史行号，必须沿当前活跃入口复核 symbol 和行为：

- `trainers/cliplora.py` 的 ClipLoRA 输出全局 100 类 logits，使用标准 global cross-entropy，不做客户端本地类别 mask；
- `federated_main.py` 中每轮每个客户端从同一个 round-start global LoRA 初始化，并为客户端重新建立 optimizer/scheduler；
- `utils/lora_aggregation.py` 的普通 FedAvg 按客户端总样本数加权，所有客户端更新同一组共享视觉 LoRA；
- `utils/datasplit.py` 的 Client-LT 构造包含 `non_tail_to_tail_budget`，可能直接压缩专科客户端中的非尾类 companion 样本。

如果当前真实代码与任一事实冲突，停止真实 V1，报告当前代码、产生已有结果的代码版本/配置是否可恢复，以及冲突会如何改变研究解释。不要为了匹配上述事实改训练代码。

这些训练语义意味着：

1. 所有样本都对 100 类决策边界产生梯度影响；“尾类少见负类”不是当前故事；
2. 相关类不与尾类同客户端时，其更新仍会通过 FedAvg 进入共享 LoRA；
3. 同客户端共现只能作为多步本地优化中的局部共适配、local drift 和非线性交互的前提代理，不能直接视为正迁移；
4. 无尾类正样本客户端仍可改变尾类边界，这更直接关联后续的知识保存/干扰问题，而不是 V1 的形成验证。

三、预期主设置与 matched 输入

以下是研究预期值，优先从现有 run metadata、partition summary、split manifest、launcher 与日志确认：

- dataset：CIFAR-100-LT，imbalance factor = 0.01；
- num_clients = 30；tail_clients = 3；
- bottom-20 类为 tail，但必须从真实全局类计数排序恢复，不能默认 label 80–99；
- Client-LT：lambda = 0.75，alpha_T = 0.5，head_leakage_scale = 3.0，non_tail_to_tail_budget = 2，tail_to_tail_budget = 20；
- Dirichlet：beta = 0.5；
- matched seeds：42、2026；
- frac = 1.0，local_epochs = 3；
- 只分析普通 sample-weighted FedAvg 的 split，不把 support-normalized/oracle 聚合误作 V1 条件；
- 每个 matched seed 的两种 topology 必须使用同一全局 train index universe、相同逐类计数和相同 class mapping。

若无法唯一恢复 seed 42/2026 的 matched split，停止在只读审计，列出候选 run、参数差异、缺失证据和需要用户决定的最小问题。不得选择“最新”或结果最有利的 run，不得重新实现近似 partition。

四、阶段 A：P0 只读训练语义审计

先阅读 AGENTS.md、README、git status、现有实验说明和真实 launcher，再沿调用链检查：

1. 活跃入口：dataset、partition、client local train、loss、aggregation、evaluation 的真实 symbol；
2. 候选类别：logits 是否始终为 100 类，是否有 mask/crop，class id/name/text feature 如何对齐；
3. 最小 no-grad runtime probe：在不调用 backward、optimizer.step 或训练循环的前提下，记录 logits shape、label 范围、text-feature shape 和 class mask；环境不足时标记 `runtime_unverified`；
4. LoRA：插入模块、层数、A/B shape、rank、scaling、dropout、全部可训练参数，是否所有类别和客户端共享；
5. 客户端初始化：同轮全局起点、optimizer/scheduler 重建、状态是否跨客户端/轮保留；
6. local_steps：local_epochs、batch size、sampler、shuffle、drop_last、每客户端实际/期望 optimizer steps；
7. FedAvg：权重公式、分子分母、active-client normalization、是否存在非标准分支；
8. split：全局 LT 样本池、tail 定义、Dirichlet/Client-LT 参数与随机种子流、manifest 是否可精确恢复；
9. zero-shot 与 round metrics：只读核对已有结果中 zero-shot、round 0、final tail accuracy 的定义和 checkpoint 时点。

P0 必须明确纠正以下可能的错误叙事：如果 round-0 tail accuracy 与 zero-shot 基本相同，则不得说“LoRA 在训练早期新形成了很强的尾部知识”。应写成“冻结 CLIP 已有较强尾部能力，持续联邦适配使其发生退化”。未来真正的形成量必须定义为同一起点上的功能增益，例如：

`acquisition_gain_c = A_c(theta_t + Delta_support_c) - A_c(theta_t)`

而不是某轮的绝对 tail accuracy。知识保存量需要单独定义为 support 更新加入 non-support/global 更新前后的差异。

P0 输出 `training_semantics_audit.md/json`，每项含 `path`、`symbol/config key`、结论、证据、verified/unverified 和对论文故事的含义。

五、修改边界

允许：

- 在已有 diagnostics/analysis 目录新增一个确定性离线入口、少量纯函数、fixture tests 和只调用离线分析的 shell 脚本；
- 复用现有 partition summary/manifest、class mapping、冻结 CLIP 文本编码器和 artifact reader；
- 在全新输出目录写分析产物。

禁止：

- 修改训练入口、loss、LoRA、FedAvg、dataset split、现有 checkpoint 或结果；
- 启动 local/federated training；
- 复制一套近似 partition 实现；
- 使用 test accuracy、训练后权重、人工 superclass 或结果反馈定义相关类；
- 看结果后改变 K、temperature、random-null 数量、频率分层、seed、tail 定义或主指标；
- 将 presence、dose、tail-mass weighting 和 FedAvg weighting 混成一个指标；
- 将完整性断言列作科学指标；
- 因 H1/H2/H3 不成立而放宽断言或筛掉类别。

六、输入与公平性不变量

逐 seed 检查 Dirichlet 与 Client-LT：

- dataset/version/preprocessing 相同；
- global train index universe 完全相同，每个 index 恰好分配一次；
- global per-class count vector、total count、class id/name mapping、bottom-20 list 完全相同；
- 使用同一个冻结 CLIP encoder、template、K、temperature 和同一批 random-null sets；
- 输出完整 client×class count matrix；
- support count、client size 和 companion composition 不要求相同，因为它们是 V1 的观测对象。

使用稳定 SHA-256 或仓库惯用稳定 hash 保存 global indices、per-class counts、每客户端 indices、mapping、partition config 和输入 artifact fingerprint；不得使用 Python `hash()`。

七、统一定义与三种权重

令 `n[k,j]` 为 client k 上 class j 的训练样本数，`n[k]=sum_j n[k,j]`。对尾类 c：

- `N_c = sum_k n[k,c]`；
- `S_c = {k | n[k,c] > 0}`；
- `a[k,j] = 1[n[k,j] > 0]`；
- `m[k,c] = sum_{j != c} n[k,j]`，即 client k 上 non-c companion sample count；
- `d[k,c] = sum_{j != c} a[k,j]`，即 companion class count。

若 `S_c` 为空，视为数据错误并停止。

对支持客户端使用三个不可混淆的权重：

1. client-unweighted：`q_client[k|c] = 1 / |S_c|`；
2. tail-mass-weighted：`q_tail[k|c] = n[k,c] / N_c`；这是 V1-B/V1-C 的主要权重，回答“随机抽取一个尾类训练样本，其所在客户端具有什么语境”；
3. FedAvg-weighted：`q_fed[k|c] = n[k] / sum_{l in S_c} n[l]`；只描述支持客户端在普通 FedAvg 中的相对样本权重，不得解释为类别 c 的真实梯度贡献。

每个 context metric 都应分别输出 `_client_unweighted`、`_tail_mass_weighted`、`_fedavg_weighted`。不要用未归一化 FedAvg 权重把“支持端总质量”混进语境均值。

`tail_mass_accounted_for = sum_{k in S_c} n[k,c] / N_c` 必须恒为 1，仅作为完整性断言，不进入科学汇总或判定。

八、V1-A：Topology concentration

逐 seed×topology×tail class 输出：

- `support_client_count = |S_c|`；
- `top2_tail_client_mass`；
- `effective_support_clients = 1 / sum_k p[k|c]^2`，`p[k|c]=n[k,c]/N_c`；
- 尾类样本落在预定义专科客户端与普通客户端的质量（仅当 manifest 明确提供角色）；
- `tail_mass_accounted_for` 完整性检查。

统一 paired delta 方向：

`delta_metric = Dirichlet - ClientLT`

因此 concentration 的预期符号不是统一的：`delta_support_client_count > 0`、`delta_effective_support_clients > 0`、`delta_top2_tail_client_mass < 0` 都表示 Client-LT 更集中。报告中必须按指标含义解释，不能只说 delta 正/负。

九、V1-B：Generic context shrinkage

对每种 q 权重输出：

- `generic_companion_class_count_q = sum_{k in S_c} q[k|c] * d[k,c]`；
- `generic_companion_class_fraction_q = generic_companion_class_count_q / (C-1)`；
- `generic_companion_sample_count_q = sum_k q[k|c] * m[k,c]`；
- `generic_companion_sample_fraction_q = sum_k q[k|c] * (m[k,c] / max(n[k],1))`，即先计算每个 client 内的 companion sample fraction，再按 q 加权；
- companion presence entropy；
- 支持客户端 context 的 Jaccard diversity 与 CLIP-centroid cosine diversity。

主要 generic 指标预先锁定为：

`generic_companion_class_fraction_tail_mass_weighted`

同时将 `generic_companion_sample_count_tail_mass_weighted` 作为剂量主诊断。

若每个 local epoch 都完整遍历本地 dataloader，可根据审计到的 sampler/batch/drop_last 推导期望 companion image draws 和 optimizer-step exposure；必须标为 `derived_expected`。只有在不训练、可确定性重放相同 dataloader 顺序时才能标为 `replayed_exact`。无法恢复时标记 NA，不得把样本计数冒充实际 step exposure。

十、冻结 CLIP 语义邻居

复用项目 zero-shot baseline 的 canonical CIFAR-100 class names、tokenizer、prompt template/ensemble 和冻结 CLIP text encoder：

- 不加载训练后 LoRA/prompt/checkpoint；
- text embeddings L2 normalize；
- 计算 100×100 cosine，排除 self；
- similarity 降序、class id tie-break；
- 主分析固定 `K=10`、`tau=0.1`；
- `w[c,j] = softmax(sim(c,j)/tau)`，只在 Top-K 内归一化；
- 保存 mapping、template、encoder id、embedding/similarity cache 和 fingerprint。

正式名称统一使用：

- `clip_neighbor_colocation`
- `semantic_neighbor_availability`

不得在代码、图标题或报告中使用 `positive_transfer_coverage`。

十一、frequency-matched random null

为了区分“所有 companion 都减少”与“语义邻居发生额外收缩”，为每个尾类 c 构造预先固定的随机 null：

- 按全局 class count 排序（class id tie-break）把 100 类确定性划为 5 个 frequency quintiles；
- 记录 `R_K(c)` 在 5 个 quintile 中的组成；
- 每个 random Top-K 必须从对应 quintile 中抽取相同数量的 distinct classes，只排除 c；不要排除真实 `R_K(c)`，因为 null 表示在保持频率构成后对类别身份做随机置换，偶然抽中语义邻居属于有效的 null 波动；
- 若随机集合与完整 `R_K(c)` 恰好完全相同，使用同一确定性随机流继续抽取；若某 quintile 在只排除 c 后仍候选不足，停止并报告，不得放宽匹配；
- `B_null = 1000`；master seed 固定为 `20260811`，并由 `(master_seed, class_id, draw_id)` 派生；
- 同一个尾类的 1000 个 null sets 必须跨 topology 和 matched seed 完全共享；
- 保存每个 null set 的 class ids、quintile composition 和 hash。

对真实 CLIP Top-K 与每个 random Top-K 使用相同的 coverage 函数。为了避免随机集合没有语义权重的问题：

- 主要 null coverage 使用集合内均匀权重 `1/K`；
- 主要真实 CLIP coverage也使用均匀权重，保证差异只来自类集合身份；
- 原来的 CLIP similarity-softmax 权重结果作为次级敏感性结果，不作为 semantic-specific 主判定。

这一区分必须保留；不得把 CLIP 集合使用 similarity weight、random 集合使用 uniform weight 后直接做主差值。

十二、V1-C：Semantic-specific co-location shrinkage

对任意 class set R 和支持客户端权重 q，定义：

`C_q(c,R) = sum_{k in S_c} q[k|c] * (1/|R|) * sum_{j in R} a[k,j]`

主要 CLIP-neighbor coverage：

`C_clip_q(c) = C_q(c, R_K(c))`

随机 null 均值：

`C_null_q(c) = mean_b C_q(c, R_null_b(c))`

语义超额共现：

`E_q^T(c) = C_clip_q^T(c) - C_null_q^T(c)`

其中 T 为 topology。

主要语义特异性 estimand 是 matched difference-in-differences：

`delta_semantic_specific_q(c) = E_q^Dirichlet(c) - E_q^ClientLT(c)`

等价于：

`[C_clip^Dir - C_clip^ClientLT] - [C_null^Dir - C_null^ClientLT]`

主权重固定为 `q_tail`。因此主指标为：

`delta_semantic_specific_tail_mass_weighted`

正值表示：相较普通频率匹配类别，CLIP 语义邻居的共现优势在 Client-LT 中收缩得更多。它仍然不是正迁移证据。

同时输出：

- 三种 q 的 `C_clip`、`C_null_mean/std/2.5%/97.5%`、`E` 与 paired delta；
- `related_companion_absolute_sample_count`；
- `related_companion_fraction_among_companions`；
- tail-mass-weighted related sample dose；
- CLIP similarity-softmax 加权版本，明确标记 secondary；
- 每个 seed×class 的 1000-null empirical distribution 摘要。

十三、汇总、不确定性与判定

逐 seed 报告 20 个 tail classes 的：

- paired delta mean、median、standard deviation；
- 正/零/负类别数量，但不得把“正类数 > 10”单独作为支持门槛；
- raw per-class values，不得只保留汇总图。

合并两个 seed 时做 class-cluster bootstrap：

- 以 tail class id 为 cluster，重采样 20 个 class ids；
- 每次保留该 class 的两个 matched seeds，避免把 class×seed 当作 40 个完全独立样本；
- `B_boot = 10000`，seed 固定 `20260811`；
- 报告 combined mean/median 与 95% percentile interval；
- 只有两个 split seeds，不得宣称已充分覆盖训练随机性或把 CI 写成最终显著性证明。

最终分类只描述 proxy 结果，不作为是否执行 V2 的硬停止规则：

1. `SEMANTIC_SPECIFIC_COLOCATION_SHRINKAGE`：两个 seed 的 semantic-specific 主效应均为正，且 class-cluster bootstrap 95% interval 不含 0；
2. `SUGGESTIVE_PROXY_EVIDENCE`：点估计总体为正但 seed 方向不完全一致或区间包含 0；
3. `GENERIC_CONTEXT_SHRINKAGE_ONLY`：generic 主指标显示 Client-LT 语境更窄，但 semantic-specific difference-in-differences 不支持额外收缩；
4. `NOT_SUPPORTED_BY_THIS_PROXY`：该共现代理不支持 generic 或 semantic-specific shrinkage。

无论属于哪一类，都不得由 V1 单独否定真实梯度正迁移；报告必须说明 V2 才是功能性验证。如果只出现 generic shrinkage，论文只能写“一般局部正样本语境变窄”，不能写“语义相关语境被特异性破坏”。

十四、与已有 accuracy 的连接

若存在严格匹配 partition、method、seed、round/checkpoint 和 class mapping 的 per-class accuracy，可输出探索性关联；否则留空，不得从 tail aggregate 反推。

- delta 统一为 `Dirichlet - ClientLT`；
- Spearman/散点只标 `exploratory, non-causal`；
- 控制列至少含 global count、zero-shot per-class accuracy（仅已有可靠 artifact 时）、support count、Top-2 mass、Neff、generic context；
- 不把 accuracy correlation 作为 V1 proxy 判定门槛。

在报告中单列 zero-shot/round-0/final aggregate 事实。如果现有数据确为 zero-shot tail 66.35%、round 0 约 66.35%–66.40%、final 明显下降，则只允许表述为“已有 CLIP 尾部能力在持续适配中退化”，不得写成“LoRA 早期获得约 68% 的新知识”。所有数值必须由 artifact reader 读取，不得把本 Prompt 中的示例值硬编码。

十五、输出产物

在全新确定性目录生成：

1. `training_semantics_audit.md/json`
2. `partition_invariants.csv`
3. `client_class_counts.npz` 与 schema metadata
4. `clip_related_classes.csv`
5. `clip_similarity.npy` 与 metadata
6. `frequency_matched_null_sets.csv` 与 hash
7. `v1a_topology_per_class.csv`
8. `v1b_generic_context_per_class.csv`
9. `v1c_semantic_colocation_per_class.csv`
10. `v1_paired_deltas.csv`
11. `v1_summary_by_seed.csv`
12. `v1_cluster_bootstrap.csv/json`
13. `v1_summary.json`
14. `v1_report.md`
15. paired plots 的 PNG 与 PDF：topology、generic context、CLIP-vs-null semantic excess；如 accuracy 可连接，再输出探索性散点。

所有 CSV 同时保存 seed、topology、tail class id/name、input fingerprint、metric weighting。浮点保留足够精度；报告数值必须由 CSV/JSON 程序生成，不能手填。

十六、测试与执行门槛

1. 静态：仓库 formatter/linter（若存在）、`python -m py_compile`、`git diff --check`；
2. fixture：至少 3 clients×5 classes，手算验证 S_c、三种 q、Top-2、Neff、generic breadth/dose、CLIP coverage、random-null coverage、semantic excess 和 difference-in-differences；
3. 边界：单 support、无 companion、空 support 报错、similarity tie、null 候选不足报错、重复 null class 禁止；
4. 分区不变量：任一 global universe/count/mapping/tail-list 硬不变量失败即停止；
5. 文本特征：finite、norm、对称 similarity、self 排除、Top-K 稳定；
6. 无训练 guard：分析入口不得触发 backward、optimizer.step、local train 或 federated rounds；文本编码前后参数 fingerprint 不变；
7. seed 42 smoke：两种 topology、20 tail classes、1000 null sets、schema/delta/hash/图表检查；
8. smoke 通过后允许执行 seed 42+2026 的完整离线 V1；不要扩展其他 alpha、lambda、rank、dataset 或 seed。

实现正确性只由公式、fixture、不变量、确定性、输出 schema 和无训练保证决定。H1/H2/H3 是否成立不是代码验收条件，负结果必须原样保存。

十七、完成时汇报

最终回复必须包含：

- P0 关键事实和对“形成—保存”故事的修正；
- 新增/修改文件及用途；
- 实际执行命令；
- 静态、fixture、seed-42 smoke、42+2026 offline run 的通过情况；
- 全部产物实际路径；
- V1-A/B/C 的逐 seed 主结果、effect size、class-cluster interval；
- 四类 proxy 判定之一；
- 明确写出 V1 没有证明什么；
- 未验证项和进入 V2 时必须保持的控制变量。

不要只回复“实现完成”，不要只贴图，不要启动训练，并保留仓库中所有无关未提交修改。
```

---

## 使用说明

这版 Prompt 将 V1 的任务严格限定为三层离线审计：拓扑集中、一般局部语境收缩、语义邻居相对随机对照的特异性共现收缩。即使 V1-C 为正，也只能进入 V2 功能性验证；如果 V1-C 为负，也不能仅凭共现代理直接否定真实梯度迁移。
