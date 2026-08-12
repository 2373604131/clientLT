# Codex 实现任务书（修订版）：V2 语义增量形成与 V3 同客户端局部路径效应

你需要在当前联邦 CLIP/LoRA 仓库中实现两组相互衔接、但证据边界不同的机制实验：

- **V2：Isolated Semantic Acquisition Test**——验证在相同尾类样本、训练预算和 LoRA 起点下，CLIP 文本语义相关的 non-tail 视觉更新是否比频次匹配的无关更新带来更大的尾类增量收益，并区分“正迁移”与“只是伤害更小”。
- **V3：Equal-weight Local Placement Test**——在 episode-level global sample multiset 完全相同、两个客户端大小和 FedAvg 权重完全相同的条件下，验证 related 样本与尾类共同经历多步本地优化是否产生额外位置优势，并严格区分“共置更有利”和“共置是必要条件”。

这不是新方法实现，也不是完整 30-client/100-round 主实验。优先保证因果比较、公平性审计、可复现样本清单和原始结果完整。不要在看到结果后改变类别、预算、匹配、seed、主指标或判定门槛。

---

## 0. 证据边界与当前已知事实

已有 P0/V1 只支持：

1. 当前 `ClipLora` 对全部 100 类输出 logits，并使用全局 100 类交叉熵；
2. 只有视觉 LoRA 参数可训练，所有客户端更新同一组共享 LoRA；
3. 每个客户端从相同 round/global LoRA 开始，optimizer/scheduler 按客户端重新创建；
4. 普通聚合使用 sample-weighted FedAvg；
5. 受控 Client-LT 中全部 153 个 bottom-20 样本只在客户端 27–29，普通客户端尾类泄漏为 0，专科端 companion 总量为 37，逐端纯度不低于 0.8；
6. Client-LT 显著收缩一般 non-tail 局部语境；相对 frequency-matched null，CLIP 文本 Top-10 邻居还有约 0.0338 的小幅额外共现收缩；
7. 该 semantic-specific proxy 在类别间异质，不能视为正梯度迁移证据。

已有结果没有证明：

- related companion 对尾类产生正迁移；
- related 比 matched-unrelated 更有利；
- Client-LT 的语境收缩导致 acquisition gain 或 accuracy 下降；
- related 更新必须与尾类同客户端才有效；
- non-support 客户端覆盖了已有尾部知识。

V2 只验证隔离式、功能性的知识增量形成。V3 只验证受控单轮双客户端中的局部优化路径/位置效应。V2/V3 都不是知识保存实验，也不能直接解释完整 100 轮 Client-LT 性能差距。

---

## 1. 必须复用的仓库真实路径与语义

### 1.1 当前候选符号

开始实现前必须沿当前活跃路径复核，候选符号包括：

- `trainers/cliplora.py::ClipLora.build_model`
- `trainers/cliplora.py::ClipLora.reset_optimizer_and_scheduler`
- `trainers/cliplora.py::ClipLora.forward_backward`
- `federated_main.py::run_promptfl_local_train_with_scheduler_policy`
- `federated_main.py::get_lt_class_splits_from_counts`
- `utils/lora_aggregation.py::sample_weighted_client_weights`
- `utils/lora_aggregation.py::aggregate_lora_state`
- `utils/datasplit.py::partition_client_longtail_controlled`

不得重新实现一个与真实 `ClipLora`、optimizer、scheduler 或 FedAvg 平行的简化训练器。允许为实验增加向后兼容的可选接口，例如：逐样本 loss mask、确定性 batch plan、返回 logits、限定 episode、导出 LoRA state；默认训练行为必须不变。

### 1.2 当前候选运行配置

当前仓库候选配置为：

- backbone：CLIP ViT-B/16；
- trainable scope：视觉 LoRA；
- LoRA 默认：vision top-3、q/v、rank 2、alpha 1、dropout 0；
- batch size：32；
- local epochs：正式实验固定为 3；
- optimizer 候选：Dassl `build_optimizer` 构建的 SGD；
- YAML LR 候选：0.002；
- Dassl 默认 momentum：0.9；
- Dassl 默认 weight decay：`5e-4`；
- `cliplora_lr_policy=constant` 时，当前代码会使用不衰减的 single-step policy，并关闭 warmup；
- precision 候选：AMP；
- loss：100 类 `F.cross_entropy`，不得使用 local-class mask。

上述只是当前代码的候选值。Phase A 必须从最终 merge 后的 resolved config 和 launcher 复核并写入 contract；若真实主实验参数不同，以真实 resolved 值为准，不得静默使用 YAML 候选值。

### 1.3 P0/V1 冻结输入

默认从以下目录读取，不得覆盖：

```text
output/p0_v1_context_colocation_v2/
```

至少读取并验证：

- `client_class_counts.npz`
- `client_class_counts_meta.json`
- `partition_indices.npz`
- `partition_indices_meta.json`
- `partition_invariants.csv`
- `clip_related_classes.csv`
- `clip_similarity.npy`
- `clip_similarity_meta.json`
- `v1b_generic_context_per_class.csv`
- `v1_paired_deltas.csv`
- `v1_summary.json`

当前预期：

```text
input_fingerprint = ebd8f1d0b7765516d4f7ef8868a8262146dd3b630e7df7c1216c01d8e17b601b
global_universe_fingerprint = 861850202191fe1db7037afef30987b14bfda680dd37d60af1f940c5c68b3b0f
tail_sample_count = 153
tail_class_count = 20
```

若实际 artifact 的 fingerprint 与 contract 不同，停止，不得近似重建。

---

## 2. Phase A：只读实现审计

先阅读仓库说明和工作树状态，只读输出 `v2_v3_implementation_audit.md`，至少确认：

1. 实际模型/数据/训练/evaluation 调用链；
2. LoRA 创建、初始化、trainable key、state-dict key、加载与保存；
3. 全局 100 类 logits 和 CE；
4. resolved optimizer、momentum、weight decay、LR、scheduler、warmup、AMP、gradient clipping；
5. 每客户端 optimizer/scheduler/GradScaler 生命周期；
6. 当前 transforms 中 `random_resized_crop`、`random_flip` 等随机增强如何接收显式 seed；
7. batch size 32、`drop_last`、sampler、local epochs=3 对应的真实 optimizer steps；
8. sample-weighted FedAvg 的权重和 LoRA-only 聚合范围；
9. per-class margin/NLL/accuracy 所需 logits 与稳定 sample ID；
10. P0/V1 的 global pool、bottom-20、Top-10/Top-30、frequency quintile 和逐类 dose 字段；
11. 工作树已有修改，避免覆盖用户文件。

审计必须记录完整 frozen CLIP checkpoint hash、class mapping hash、trainable LoRA 的静态选择规则/预期生成路径和最终 resolved config。精确 trainable key、shape、offset、flatten spec hash、`theta_0` hash 与固定 probe logits hash 依赖真实模型构建，必须在 Phase C 的 CUDA runtime contract 中补全；不得用静态猜测值冒充。以下任一项在静态上找不到可行实现路径则停止，不编码：

- 无法唯一恢复 P0/V1 global pool、sample ID、bottom-20 或 Top-10；
- 从现有同一模型构建路径看，无法设计出可序列化并重新加载完全相同 `theta_0` 的接口；
- 主路径并非共享视觉 LoRA + 100 类 CE；
- 需要改变 baseline 默认训练语义才能实现；
- 无法把数据增强绑定到显式 slot seed；
- 无法稳定输出目标类的 100 类 logits。

---

## 3. 公共冻结规则

### 3.1 共同 LoRA 起点

- 所有条件从同一个序列化 `theta_0` 开始；
- `theta_0` 为主联邦训练开始前的实际 LoRA 初始化；
- 若没有初始化 artifact，只允许使用一个显式 `model_init_seed` 构建一次，立即序列化，之后所有 run 只加载该文件；
- 同时保存 frozen CLIP/full non-trainable state hash、LoRA state hash、trainable key list 和 flatten spec hash；
- 每个 run 训练前检查 logits hash 和 `theta_0` hash；
- 每个 episode 独立重建 optimizer、scheduler 和 GradScaler，不跨条件保留状态。

### 3.2 固定跨 seed companion 预算

从 `v1b_generic_context_per_class.csv` 读取 controlled Client-LT、tail-mass-weighted 的逐 `seed × class` non-tail companion sample dose：

```text
D[c,s] = generic_companion_sample_count_tail_mass_weighted
```

正式预算固定为跨两个 V1 seed 的平均，使用明确的 half-up rounding：

```text
B[c] = floor((D[c,42] + D[c,2026]) / 2 + 0.5)
```

禁止使用 Python/Pandas 的 banker's rounding。预算在任何训练前保存为 `companion_budgets.json` 并哈希。当前 artifact 的预期预算为：

```text
79:12, 81:14, 82:13, 83:14, 84:13,
85:12, 86:13, 87:12, 88:12, 89:13,
90:11, 91:13, 92:13, 93:13, 94:11,
95:13, 96:15, 97:11, 98:14, 99:14
```

生成器必须从 artifact 重算并逐项断言以上值；不得把该表当作跳过 artifact 的硬编码输入。这样两个 data seed 使用相同 (B_c)，seed 只改变预注册的样本/augmentation/matching 随机流，不改变训练剂量。

### 3.3 Related 类及 quota

- `R_c` 为 V1 冻结的 non-tail CLIP Top-10，保持 rank 顺序；
- 排除目标类和全部 bottom-20；
- 若 `B_c < 10`，使用 rank 前 `B_c`，每类 1 张；
- 若 `B_c >= 10`，先每个 Top-10 分配 1，再按 rank 从小到大循环分配剩余 quota；
- 保存 `q_c=(q_1,...,q_10)`，保证和为 `B_c`；
- 当前预算范围 11–15，因此当前每个 Top-10 至少出现 1 张，但代码仍需正确处理一般边界。

### 3.4 Frequency-matched unrelated

不要使用未定义的 log-count bin。复用 V1 的 80 个 non-tail 类 frequency quintiles：

1. 按 `(global_count, class_id)` 确定性排序；
2. 划分为 5 个各 16 类的 quintile；
3. 对每个 related rank (r_i)，unrelated (u_i) 必须来自相同 quintile；
4. unrelated 候选排除全部 bottom-20、`R_c` 和该尾类 non-tail CLIP Top-30；
5. 同一 draw 中 unrelated class 不重复；不同 draw 允许重复，但必须使用独立稳定 seed；
6. 在同 quintile 内，用二分图最小代价匹配：

```text
cost(r_i,u) = abs(log(n_global(r_i)+1) - log(n_global(u)+1))
```

7. 代价完全相同时，用由 SHA-256 派生的稳定 tie-break，不得使用 Python `hash()`；
8. exact-count 是否匹配只作为诊断字段，不作为前置要求；
9. 每个 unrelated rank 复用对应 related rank 的 quota `q_i`。

生成 manifest 时先检查完整 Hall/assignment 可行性；若失败，标记 `UNMATCHABLE` 并停止该 class，不得放宽到 Top-30、tail class 或不同 quintile。当前 P0/V1 artifact 下，五分位匹配预期对 20 类全部可行。

### 3.5 样本和随机批次流

对同一 `(data_seed, tail_class, draw)`：

- tail block 的真实 train sample ID 完全相同；
- related block 在 `related`、`tail_only_masked` 和 V3 中完全相同；
- unrelated block 按 draw 固定；
- 类内图像采样不放回；
- base episode dataset 中每个 sample ID 恰好出现一次；
- local epochs=3 时，每个 base sample 在 execution schedule 中正常出现 3 次，不得把这误报为重复样本；
- V2 中 batch slot、tail slot、companion slot 和 augmentation seed 按 paired slot 绑定；related/unrelated 只替换对应 companion slot 的 sample ID 和 label，因此匹配的 R/U 图像使用相同 slot seed；
- V3 中 augmentation seed 必须按具体 base sample/block item 绑定并随 R/U 图像跨客户端移动，保证同一真实 sample 在两个 placement、同一 epoch 中得到完全相同的 augmented tensor；不得把 V3 augmentation seed 留在客户端位置 slot 上；
- 同一输入和 seed 重建的 base manifest、execution manifest 必须 byte-identical。

分别保存并校验：

```text
base_multiset_hash
execution_schedule_hash
```

---

## 4. V2：Isolated Semantic Acquisition Test

### 4.1 研究问题和实验边界

V2 回答：在隔离的单目标类本地 adaptation episode 中，related non-tail 更新是否比 frequency-matched unrelated 更新更有利，并且是否优于没有 companion 梯度？

V2 不重建真实专科客户端中的其他 19 个尾类，因此不得直接称为“完整 Client-LT acquisition”。报告中必须写明：这是隔离式功能机制实验；它验证语义 companion 的因果作用，不直接估计完整 Client-LT 的准确率差。

### 4.2 条件

对每个 `(data_seed, tail_class)` 构造：

1. `related`：目标类全部冻结 train IDs `T_c` + related block `R_c`；
2. `matched_unrelated_r0...r2`：相同 `T_c` + 第 r 个 unrelated block `U_c^r`；
3. `tail_only_masked`：使用与 `related` 完全相同的 `T_c + R_c` forward/batch plan，但 related slots 的 loss weight 为 0。

正式实验固定 3 个 unrelated draw；smoke 固定 1 个。`tail_only_masked` 每个 seed/class 只运行一次，不按 unrelated draw 重复。

各 active 条件中：

```text
N_episode(c) = |T_c| + B_c
```

当前底部类样本数与 (B_c) 下，episode size 预期小于 batch size 32，因此每个 epoch 预期一个 full episode batch，3 local epochs 预期 3 个 optimizer steps。实现必须从实际 dataloader 计算并断言；若不是 3，停止并报告真实原因，不得静默继续。

### 4.3 Tail-only masked loss

复用 `ClipLora` 的 100 类 logits，只增加可选的逐样本 loss mask：

```text
ce_i = cross_entropy(logits_i, label_i, reduction="none")
loss = sum_i(loss_weight_i * ce_i) / actual_batch_size
```

- tail slot weight 为 1；
- companion slot weight 为 1（related/unrelated）或 0（tail-only）；
- 分母始终是 padding removal 后的真实 batch slot 数；
- tail slot 的 CE 系数跨所有条件完全相同；
- tail-only 不能缩短 dataset、减少 steps 或改变 scheduler；
- 记录 AMP scale 和 skipped-step/overflow；若不同条件出现不一致 overflow，判为公平性失败。

必须审计模型是否含 BatchNorm、memory bank、跨样本归一化或其他让 zero-loss companion 改变 tail forward/state 的模块。CLIP LayerNorm 是逐样本操作，但仍需用真实模型做梯度不变性测试：替换 masked companion 内容后，tail-only LoRA gradient 必须在预注册 dtype tolerance 内不变。

### 4.4 配对公平性

同一 `(data_seed, tail_class, draw)` 下，related 与 unrelated 必须相同：

- `theta_0`、训练前 logits；
- tail IDs、tail slot、tail augmentation seed；
- (B_c)、quota vector、companion slot；
- batch size、optimizer steps、scheduler steps；
- LR、momentum、weight decay、precision、loss denominator；
- target evaluation sample IDs。

tail-only 额外要求 forward sample 数和 slot plan 与 related 相同。

### 4.5 指标

在固定的目标类官方 test images 上以 eval/no-grad 模式评估。官方 test 只作为预注册机制评估集，不得根据 V2 test 结果重新选择 Top-K、预算、匹配或方法超参数。

对目标类样本：

```text
margin(x,c;theta) = logit_c - max_{j != c} logit_j
M_c(theta) = mean_x margin(x,c;theta)
G_margin_c(a) = M_c(theta_a) - M_c(theta_0)
```

主指标固定为 `G_margin`。同时输出：

```text
G_nll_c(a) = NLL_c(theta_0) - NLL_c(theta_a)
G_acc_c(a) = Acc_c(theta_a) - Acc_c(theta_0)
G_adaptation_tail_loss_c(a)
  = CE_on_fixed_Tc(theta_0) - CE_on_fixed_Tc(theta_a)
```

`G_adaptation_tail_loss` 明确在实际 adaptation tail block `T_c` 上计算，只作训练拟合诊断，不称为 held-out loss。

还要记录：correct-logit change、hardest-negative logit/class change、LoRA update norm、逐层 norm、optimizer steps、sample draws、AMP scale/overflow，以及 overall/non-tail test accuracy safety diagnostic。

### 4.6 梯度兼容性诊断

不额外切走稀缺尾类训练图像。在 `theta_0` 上直接使用 episode 的实际 block：

```text
g_tail = grad mean_CE(T_c)
g_comp = grad mean_CE(R_c or U_c^r)
cos_grad = dot(g_tail,g_comp) / (||g_tail||*||g_comp|| + eps)
```

- 只 flatten 真实 trainable LoRA 参数；
- 保存参数名、shape、offset 和 flatten spec hash；
- 输出全参数和逐层 cosine/norm；
- 同时记录 companion 在 `theta_0` 上的 CE、zero-shot accuracy/confidence 和 gradient norm，作为类别难度诊断；不得用这些结果重新匹配类别；
- gradient cosine 不是成立门槛，主结论只来自训练后的 paired `G_margin`。

### 4.7 效应与汇总

先在每个 `seed × class` 内平均 3 个 unrelated draw，再计算：

```text
Delta_sem(c,s)
  = G_related(c,s) - mean_r G_unrelated(c,s,r)

Delta_pos(c,s)
  = G_related(c,s) - G_tail_only(c,s)
```

不得把 draw、batch、step 或 image 当作独立统计重复。合并两个 seed 时以 `tail_class` 为 cluster，重采样类 ID 并保留该类的两个 seed。固定：

```text
B_boot = 10000
bootstrap_seed = 20260811
```

分别报告两个 seed 和合并的 mean、median、20 类正负数、逐类值和 class-cluster percentile interval。该区间仅描述类别重采样不确定性，不代表完整训练随机性。

定量异质性规则：任一主效应若两个 seed 的均值方向不一致，或任一 seed 中正向类别少于 12/20，则不能得到普遍性标签，转入 heterogeneous 标签。

稳定 NLL 反向定义为：对应 paired NLL effect 在两个 seed 均为负，且合并 class-cluster 95% CI 上界低于 0。

### 4.8 V2 verdict

- `POSITIVE_SEMANTIC_TRANSFER`：`Delta_sem` 与 `Delta_pos` 两个 seed 均为正；二者合并 CI 下界均大于 0；每个 seed 两项均至少 12/20 类为正；NLL 无稳定反向。
- `RELATIVE_COMPATIBILITY_ONLY`：`Delta_sem` 满足上述稳定性，但 `Delta_pos` 不满足；只能说 related 比 unrelated 更兼容或伤害更小。
- `HETEROGENEOUS_FUNCTIONAL_TRANSFER`：合并均值和 CI 支持正效应，但 seed 均值方向不一致或任一 seed 少于 12/20 类为正。
- `NO_FUNCTIONAL_SUPPORT`：`Delta_sem` 没有稳定正优势或稳定为负。
- `INVALID_COMPARISON`：任何关键配对不变量、theta/logits、step、mask、AMP 或样本守恒失败。

只有 `POSITIVE_SEMANTIC_TRANSFER` 自动开放完整 V3。其他 verdict 只允许完成 V3 代码/fixture/smoke，不得自动运行 V3 full；是否继续由用户单独决定。

---

## 5. V3：Equal-weight Local Placement Test

### 5.1 研究问题

V3 保持同一 episode-level global sample multiset，只交换 R/U 的客户端位置，检验多步本地路径是否带来共置优势。不得把 micro-federation 称为完整 CIFAR global pool。

### 5.2 强制等大小、等权设计

令：

- `T`：V2 的目标尾类 block；
- `R`：V2 related block；
- `U`：对应 unrelated draw；
- `F_D`：固定 remote filler，要求 `|F_D|=|T|`；
- `F_S=empty`。

两个 placement 固定为：

| 条件 | Tail-support client S | Remote client D |
|---|---|---|
| `R_colocated` | `T + R` | `U + F_D` |
| `R_remote_U_colocated` | `T + U` | `R + F_D` |

因此必须满足：

```text
n_S = n_D = |T| + B_c
FedAvg weights = 0.5 / 0.5
```

`F_D` 不是可选装饰，而是去除 client-size/FedAvg-weight 混淆的必要控制。它必须：

- 来自 train non-tail；
- 排除该尾类 Top-30、R、所有 3 个 U draw、全部 bottom-20；
- 在同一 `seed × class` 的所有 draw/placement 中逐样本相同；
- 使用稳定 seed 无放回抽样；
- 在 D 的固定 filler slot 中使用相同 augmentation seed；
- 正常 loss weight=1，并在报告中明确它是固定背景梯度。

先生成全部 U draw，再选择不与任何 U 重叠的 `F_D`。若无法选足 `|T|`，该 class 失败，不得复制或放宽到 Top-30。

### 5.3 执行

每个 `(data_seed, tail_class, draw, placement)`：

1. S、D 都从相同 `theta_0` 开始；
2. 独立创建真实主线 optimizer/scheduler/GradScaler；
3. 两端均运行 local epochs=3；
4. 两端 batch plan、steps、scheduler steps 完全相同；
5. 每个 epoch 结束保存 S/D LoRA state；
6. 在 epoch 1、2、3 分别用真实 `aggregate_lora_state` 以 0.5/0.5 聚合；
7. 评估 `theta_0`、每个 epoch 的 `theta_S`、`theta_D`、`theta_FedAvg`；
8. base episode 中 T/R/U/F 各出现一次，execution schedule 中每个 base sample 每个 epoch出现一次；
9. 两 placement 的 episode-level global sample-ID multiset hash 必须相同；
10. 两 placement 中，同一 base sample 在同一 epoch 的 augmentation seed 和 augmented-tensor hash 必须相同，因此 augmented global multiset 也相同。

当前样本规模下每端预期小于 batch size 32，因此每个 epoch 预期一个 full-batch step，共 3 steps。必须由运行时实际断言。

### 5.4 三层 oracle

#### Oracle A：raw global gradient invariance

在 `theta_0` 上使用完全相同的 augmented sample tensors、固定的全局 sample-ID 累积顺序和 full client mean loss：

```text
g_global^P(theta_0)
  = 0.5 * grad L_S^P(theta_0)
  + 0.5 * grad L_D^P(theta_0)
```

两个 placement 的 episode global multiset 相同且两个客户端等大小，因此 raw global gradient 理论上相等。记录 relative L2、max abs、cosine、dtype。默认 tolerance：实际 float32 forward/backward 为 `1e-5`；只有真实执行了 float64 forward/backward 才可使用 `1e-6`，不能仅把 float32 gradient cast 成 float64 后声称 float64 oracle。

#### Oracle B：plain-SGD one-step invariance

只作数学诊断，额外从 `theta_0` 使用一个 full-batch、zero-momentum、zero-weight-decay、固定 LR 的 plain-SGD step，再 0.5/0.5 聚合。两个 placement 的聚合 LoRA state 必须在预注册 tolerance 内相同。该 oracle 不替代主线 optimizer，也不进入效果汇总。

#### Oracle C：真实主线 optimizer 的 epoch-1 检查

当前主线候选为从零状态开始的 SGD+momentum。对真实 resolved optimizer，比较两个 placement 在第一个 full-batch local step 后的 FedAvg state：

- 若 epoch-1 state 已明显不同，标记 `OPTIMIZER_OR_NUMERIC_PLACEMENT_EFFECT_AT_STEP1`，不得把 epoch-3 差异直接归因于多步 co-adaptation；
- 若 epoch-1 近似相同，而 epoch-2/3 逐渐产生差异，才支持 local nonlinear path/co-adaptation 解释。

若真实 resolved optimizer、batching 或 AMP 使 Oracle C 理论上不应相等，Phase A 必须提前给出数学原因并把 Oracle C 标为诊断而非硬失败；不得在看到结果后修改解释。

Oracle A 或 B 超阈值一律 `INVALID_COMPARISON`，不得进入多步主结果。

### 5.5 V3 指标和分解

保存并评估：

- `theta_0`；
- epoch 1/2/3 的 `theta_S`；
- epoch 1/2/3 的 `theta_D`；
- epoch 1/2/3 的 `theta_FedAvg`。

主效应固定为 epoch 3：

```text
Delta_location(c,s,r)
  = G_after_fedavg_e3(R_colocated)
  - G_after_fedavg_e3(R_remote_U_colocated)
```

辅助：

```text
Delta_location_e1 / e2 / e3
Delta_path_growth = Delta_location_e3 - Delta_location_e1

Delta_support_local
  = G_support_local_e3(T+R) - G_support_local_e3(T+U)

Remote_related_gain
  = G_remote_local_e3(R+F_D) - G_remote_local_e3(U+F_D)
```

同时记录 S/D update norm、`cos(Delta_S,Delta_D)`、相对固定 tail probe gradient 的投影、每层 placement effect、actual FedAvg update 与 linear oracle update 的差异。

只有最终 FedAvg 的 `Delta_location` 回答位置是否重要；local state 只解释来源。

### 5.6 “优势”和“必要性”必须分开

V3 不能仅凭 `Delta_location>0` 宣称 related 必须同客户端。

先对 3 个 unrelated draw 在 `seed × class` 内取平均，再做与 V2 相同的 class-cluster bootstrap 和 12/20 异质性规则。

- `LOCAL_COADAPTATION_NECESSARY`：V2 为 `POSITIVE_SEMANTIC_TRANSFER`；Oracle A/B 通过且 Oracle C 无未解释的一步位置差；`Delta_location_e3` 两 seed 均为正、CI 下界大于 0、每 seed 至少 12/20 类为正；colocated 的 `G_margin` 稳定为正；remote placement 的 `G_margin` 两 seed 均不为正且合并 CI 上界不高于 0；`Delta_support_local` 方向一致；NLL 无稳定反向。
- `LOCAL_COADAPTATION_ADVANTAGE`：上述稳定 `Delta_location` 条件成立，colocated acquisition 为正，但 remote placement 也保留稳定正 acquisition。只能说共置额外有利，不能说必要。
- `LOCAL_PLACEMENT_COMPATIBILITY_ONLY`：`Delta_location` 稳定为正，但 colocated 相对 `theta_0` 的 acquisition 仍不稳定为正；只能说伤害更小。
- `HETEROGENEOUS_LOCATION_EFFECT`：合并均值/CI 支持正效应，但 seed 方向不一致或任一 seed 少于 12/20 类为正。
- `NO_STABLE_LOCATION_ADVANTAGE`：`Delta_location` 无稳定正差异；若 remote acquisition 为正，只能描述为 remote related updates can contribute，不能宣称与 colocated 等价。
- `OPTIMIZER_OR_NUMERIC_PLACEMENT_EFFECT_AT_STEP1`：Oracle A/B 通过，但真实 optimizer 第一步已出现未解释位置差；保留结果，但不能归因于多步 co-adaptation。
- `INVALID_COMPARISON`：样本、client size、0.5/0.5 权重、step、theta/logits、augmentation、oracle A/B 任一失败。

即使得到 `LOCAL_COADAPTATION_NECESSARY`，结论也严格限制为受控单轮、双客户端、等权 micro-federation，不能外推为 100 轮 Client-LT 差距的唯一原因。

---

## 6. 固定输出 schema

### 6.1 公共 contract/manifest

`experiment_contract.json`：

- dataset/global pool/class mapping/fingerprint；
- bottom-20、Top-10、Top-30 和五分位；
- `companion_budgets.json` path/hash；
- theta/full CLIP/LoRA/flatten hashes；
- resolved optimizer/scheduler/AMP/transforms；
- model-init/data/augmentation/bootstrap seed；
- matching和 filler规则；
- git commit、dirty diff、命令行和环境版本。

`base_sample_manifest.csv`：base episode 每个真实 sample 一行：

```text
stage,data_seed,tail_class,draw,condition,client_role,
base_sample_id,label,is_tail,is_related,is_unrelated,is_filler,
semantic_rank,frequency_quintile,match_pair_id,global_class_count,
loss_weight,base_multiset_hash
```

`execution_slot_manifest.csv`：每次真实 draw/slot 一行：

```text
stage,data_seed,tail_class,draw,condition,client_role,
epoch,batch_index,position_in_batch,base_sample_id,label,
slot_role,loss_weight,augmentation_seed,execution_schedule_hash
```

`fairness_invariants.csv` 至少包含：

```text
global_pool_hash_equal,theta0_hash_equal,pretrain_logits_equal,
tail_ids_equal,tail_slots_equal,companion_budget_equal,
quota_equal,base_multiset_conserved,execution_repetition_correct,
v2_paired_slot_augmentation_equal,
v3_per_sample_augmentation_equal,v3_augmented_multiset_equal,
batch_size_equal,optimizer_steps_equal,
scheduler_steps_equal,loss_denominator_equal,eval_ids_equal,
amp_overflow_equal,client_sizes_equal,fedavg_weights_equal,
train_test_disjoint,pass,reason
```

不适用于 V2 的 client/FedAvg 字段用 `null + reason`，不得伪造为 true。

### 6.2 V2

- `v2_run_metrics.csv`
- `v2_gradient_diagnostics.csv`
- `v2_paired_effects.csv`
- `v2_summary.json/md`
- `v2_excluded_units.csv`

必须能恢复 theta0/after 的 margin、NLL、accuracy、adaptation-tail loss、logits诊断、update norms、steps/draws、AMP状态以及逐 draw unrelated 结果。

### 6.3 V3

- `v3_placement_manifest.csv`
- `v3_client_updates.csv`
- `v3_linear_oracles.csv`
- `v3_epoch_trajectory.csv`
- `v3_paired_effects.csv`
- `v3_summary.json/md`
- `v3_excluded_units.csv`

`v3_linear_oracles.csv` 必须区分 raw-gradient、plain-SGD-one-step、main-optimizer-epoch1，并记录 tolerance、relative L2、max abs、cosine 和 pass/reason。

所有 JSON 禁止 NaN/Infinity；无法计算用 `null` 并附 reason。CSV 保留完整浮点精度。

---

## 7. 分阶段实现与门槛

### Phase A：只读审计

只生成 `v2_v3_implementation_audit.md`。前置不满足则停止，不编辑代码。

### Phase B：只实现数据和不变量

不训练。fixture 和真实 manifest dry-run 必须证明：

1. 从 V1 重算的 (B_c) 与冻结预期一致；
2. 五分位 matching 对每类可行；
3. related/unrelated/quota/slot 公平；
4. base sample 无重复，execution 每样本恰好重复 3 epochs；
5. V3 两 placement global multiset 相同；
6. V3 两客户端大小相同、权重严格 0.5/0.5、steps 相同；
7. filler 在所有 draw/placement 固定且与 T/R/U 不重叠；
8. V3 同一 sample 的 augmentation seed 随 sample 移动，两个 placement 的 augmented global multiset 相同；
9. 重跑 manifest byte-identical。

### Phase C：模型和数学测试

至少测试：

1. theta0 save/load tensor exact；
2. 条件训练前 logits exact；
3. masked companion 替换不改变 tail-only LoRA gradient；
4. 固定分母与 tail CE 系数正确；
5. margin/NLL/accuracy fixture；
6. sample-weighted LoRA FedAvg 与手算一致；
7. V3 raw-gradient oracle；
8. V3 plain-SGD one-step oracle；
9. summarizer 先平均 draw、再以 class cluster；
10. NaN/Inf 明确拒绝或转 null+reason；
11. baseline 默认训练路径在未启用实验选项时行为不变。

任一失败阻止 smoke，不得降低断言阈值。

### Phase D：V2 smoke

固定：

```text
data_seed = 42
tail classes = [90 train, 92 tulip]
unrelated_draws = 1
conditions = related, matched_unrelated_r0, tail_only_masked
batch/local epochs/optimizer/LR/scheduler/AMP = 正式 resolved 设置
```

选择已冻结：train 是 V1 semantic-specific 较强且邻居可解释的类；tulip 接近 0。Smoke 只验证实现，不以效果方向作为通过条件。

必须检查：所有 invariants、theta/logits、steps/draws、mask、AMP、可复现状态和 schema。通过后才解锁 V2 full launcher，不自动运行。允许在 smoke 前随代码生成 fail-closed launcher 模板，但它必须读取成功的 V2 smoke summary/marker；没有通过 smoke 时必须拒绝启动 full。

### Phase E：V3 smoke

在 V2 代码 smoke 通过后允许做实现 smoke：

```text
data_seed = 42
tail class = 90 train
draw = 0
placements = 2
local epochs = 3
one micro-federation round
```

先运行 Oracle A/B/C；A/B 或公平性失败则禁止多步训练。Smoke 只验证实现。通过后才解锁 V3 full launcher；允许预先生成 fail-closed 模板，但它必须同时读取成功的 V3 smoke summary/marker。V3 full launcher 还必须要求：

```text
--require-v2-verdict POSITIVE_SEMANTIC_TRANSFER
```

### Phase F：只生成完整矩阵，不自动执行

V2 full：

```text
2 seeds × 20 classes ×
(1 related + 3 unrelated + 1 tail-only)
```

V3 full，仅通过 verdict gate：

```text
2 seeds × 20 classes × 3 unrelated draws × 2 placements
```

同一 paired unit 的条件必须成组调度、成组完成；缺失条件不能用非配对均值填补。

---

## 8. 修改边界

允许：

- 新增 deterministic manifest/batch-plan、V2/V3 runner、summarizer、tests、launcher；
- 为 `ClipLora` 增加默认关闭的逐样本 mask、返回 logits/state 等最小接口；
- 复用真实 model builder、optimizer、scheduler、evaluation 和 LoRA FedAvg。

禁止：

- 改变现有 baseline 默认行为；
- 修改 P0/V1 split 或覆盖其产物；
- 使用 test 结果定义 related/unrelated、预算或超参数；
- 把 text similarity 称为 gradient compatibility；
- 使用冻结 image feature cache 代替真实视觉 LoRA forward；
- 自动启动完整 V2/V3、30-client 或 100-round 训练；
- 为实验重写一套近似 LoRA/FedAvg；
- 因结果不理想更换 seed、class、K、matching、budget、margin 或 verdict。

---

## 9. 完成报告

完成时依次报告：

1. 审计确认的真实入口和 resolved 训练语义；
2. 新增/修改文件及用途；
3. 实际静态检查、单测和 smoke 命令；
4. Phase A–E 各门槛结果；
5. V2/V3 smoke 的公平性、oracle 和产物路径；
6. 是否生成 full launcher；
7. 明确声明未自动运行哪些完整实验；
8. unmatched/excluded unit 和科学 gate 状态；
9. `git diff --check` 和相关 diff 摘要。

若任何 gate 失败，保留失败 artifact，停止后续阶段并给出最小修复建议。不要只回复“实现完成”或“测试通过”。
