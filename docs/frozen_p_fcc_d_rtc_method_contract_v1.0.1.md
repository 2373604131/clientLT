# P-FCC + D-RTC 方法运行契约 v1.0.1

状态：**FROZEN CORRECTNESS REVISION FOR STAGE-3 SEED-42 MVP**  
冻结日期：2026-08-24  
基础契约：[v1](./frozen_p_fcc_d_rtc_method_contract_v1.md)  
优先级：本文件与 v1 冲突时，以 v1.0.1 为准；未被本文件修改的条款继续继承 v1。  
变更边界：本修订不改变两个 Insight、两个组件、五个实验条件、客户端平均 CE 运行时决策或冻结超参数。

## 1. 保持不变的算法主结构

### 1.1 Insight

- **Functional Donor Access Scarcity**：Client-LT 减少尾类可利用的功能载体；语义只用于解释 donor enrichment，不是运行时输入。
- **Privately Verifiable Signed Rewrite Risk**：类别缺失更新具有类别条件且带符号的功能影响；私有证据可以判断 donor/rewrite risk，但方法不假设所有更新有害或发生普遍 donor-to-rewriter 翻转。

### 1.2 组件职责

- **P-FCC**：从上一轮 multi-source proposal prototypes 中选择当前客户端私有证据验证为正向的功能方向。
- **D-RTC**：检测 incoming global 相对客户端历史最佳 incoming-global 功能的退化，并产生恢复方向。

### 1.3 单轮顺序

```text
收到 θt 和本条件上一轮 proposal bank
        ↓
在未经修改的 θt 上更新/读取 reference，并计算 D-RTC degradation
        ↓
从同一个未经修改的 θt 独立评价每个 proposal
        ↓
执行完全标准的本地 CE，得到 δCE
        ↓
只计算 FCC direction 和 RTC restore direction
        ↓
固定范数方向合成
        ↓
上传一个普通 LoRA delta
        ↓
原始 FedAvg 聚合，并为本条件构建下一轮 bank
```

### 1.4 五个条件

1. `FedAvg-VisualLoRA`；
2. `P-FCC-only`；
3. `D-RTC-only`；
4. `P-FCC+D-RTC`；
5. `Random-Proposal`。

## 2. 数值和参数向量契约

### 2.1 固定范数合成

所有 norm、dot product、cosine、scaling 和聚类输入预处理均在 FP32 中完成。设：

\[
z_k=
\delta_k^{CE}
+\lambda_{FCC}\widehat f_k
+\lambda_{RTC}d_k^t\widehat r_k.
\]

最终上传严格定义为：

\[
\delta_k^{upload}=
\begin{cases}
0,
&\|\delta_k^{CE}\|_2\leq\epsilon_{norm},\\[4pt]
\|\delta_k^{CE}\|_2\dfrac{z_k}{\|z_k\|_2},
&\|\delta_k^{CE}\|_2>\epsilon_{norm},\ \|z_k\|_2>\epsilon_{norm},\\[8pt]
\delta_k^{CE},
&\|\delta_k^{CE}\|_2>\epsilon_{norm},\ \|z_k\|_2\leq\epsilon_{norm}.
\end{cases}
\]

冻结：

- `eps_norm = 1e-12`；
- \(z_k\) 近零时回退到 \(\delta_k^{CE}\)，禁止上传零、随机方向或历史方向；
- 合成后转回模型原 dtype；
- 转回 dtype 并完成 flatten/unflatten 后，再用 FP32 重新计算真实上传范数；
- 不声称 bitwise equality，只检验相对误差。

范数误差：

\[
e_k^{norm}=
\frac{
\left|\|\delta_k^{upload}\|_2-\|\delta_k^{CE}\|_2\right|
}{
\max(\|\delta_k^{CE}\|_2,\epsilon_{norm})
}.
\]

若两者均为零，定义 \(e_k^{norm}=0\)。正式 Gate 为：

\[
e_k^{norm}<10^{-6}.
\]

### 2.2 Flatten spec

模型初始化后、第0轮开始前，创建并冻结唯一 `flatten_spec`：

- 只包含共享、可训练的 vision-LoRA 参数；
- 使用完整参数名的字典序，不依赖模型构建时的临时遍历顺序；
- 每项保存 `name, shape, numel, dtype, offset`；
- 保存 spec hash；
- proposal、CE delta、restore gradient、clustering vector 和 upload delta 共用同一 spec。

禁止混入：

- optimizer state；
- frozen CLIP 参数；
- 文本编码器和文本原型；
- buffer；
- 非共享 LoRA 或其他临时参数。

必须满足：

\[
\operatorname{unflatten}
(\operatorname{flatten}(\theta))=\theta.
\]

## 3. Incoming-global reference 契约

客户端只保存：

- `reference_logits`：历史最佳 incoming global 在 \(E_k\) 上的 detached FP32 logits；
- `reference_ce`：FP32 scalar；
- `reference_round`；
- `reference_update_count`；
- functional-memory fingerprint。

温度分布在恢复时由 logits 计算：

\[
q_k^{ref,T}
=
\operatorname{softmax}
\left(\frac{z_k^{ref}}{T}\right).
\]

每次客户端参与时，reference 状态机固定为：

1. 在未经任何修改的 \(\theta_t\) 上，以 deterministic evaluation transform 计算当前 CE 和 FP32 logits；
2. 若无 reference：用当前 incoming global 初始化，令 \(d_k^t=0\)；
3. 若当前 CE 严格低于 `reference_ce`：更新 reference，令 \(d_k^t=0\)；
4. 否则保持旧 reference，并计算连续 degradation；
5. 本轮之后任何状态均不得再更新 reference。

禁止写入 reference：

- 本地 CE 后 logits；
- proposal probe 后 logits；
- restore/recovery 后 logits；
- 最终上传状态 logits。

deterministic transform 和 FP32 下使用严格 `<`，v1.0.1 不新增 improvement threshold。

## 4. Functional memory \(E_k\)

### 4.1 确定性选择

\(E_k\) 最多32个样本，来自客户端本地训练集且不从普通 CE 数据中删除。选择规则：

1. 获取客户端本地已观察类别；
2. 用 `stable_seed("functional-memory-classes", global_seed, client_id)` 对类别做确定性 permutation；
3. 每个类别内用 `stable_seed("functional-memory-samples", global_seed, client_id, class_id)` 对全局稳定 sample ID 排序；
4. 按打乱后的类别顺序 round-robin；
5. 达到32个样本或本地样本耗尽时停止。

保存全局稳定 dataset sample ID，禁止保存 DataLoader 临时位置作为身份。

五个实验条件使用相同 \(E_k\)。功能评价始终使用 deterministic evaluation transform；普通 CE 仍可使用这些样本及原训练 augmentation。

### 4.2 Selection-independent audit view \(A_k\)

为避免在 proposal 选择集上循环证明 accepted proposal 更好，离线研究日志增加与 \(E_k\) 不重叠的 \(A_k\)：

- 最多28个样本；
- 使用与 \(E_k\) 相同的稳定、类别均衡选择方法，从剩余 sample IDs 中选取；
- 不参与 proposal selection；
- 不参与 reference 更新；
- 不参与 degradation trigger；
- 不参与 restore gradient；
- 不参与超参数选择；
- 不进入算法通信 payload；
- 当前轮 audit 在普通本地 CE 开始前完成。

为避免从样本稀少客户端永久扣除最多28个训练样本，\(A_k\) **仍可参与所有条件完全相同的普通本地 CE**。因此它应称为 `selection-independent audit view`，不能称为完全 held-out validation set。如果除 \(E_k\) 外没有样本，则该客户端没有 \(A_k\)，不得复用测试集。

## 5. Proposal bank 的完整定义

### 5.1 每个条件独立

每个实验条件只使用自己上一轮实际上传的 delta 构建 bank。不同条件禁止共享 P-FCC、Combined 或 Random 生成的 bank。第0轮 bank 为空，bank 生命周期为一轮。

### 5.2 有效更新

上一轮更新只有同时满足以下条件才有效：

- flatten spec/hash 完全一致；
- shape 和 numel 一致；
- 所有元素 finite；
- FP32 norm 大于 `eps_norm`。

无效更新不参与 median、prototype 数量、聚类或 cluster size。

设有效更新数量为 \(N_{valid}\)：

\[
M_t=
\min\left(6,\left\lfloor\frac{N_{valid}}{4}\right\rfloor\right).
\]

若 \(N_{valid}<4\)，bank 为空。

### 5.3 范数预处理

\[
\tau_{t-1}
=
\operatorname{median}
\{\|\Delta_j^{t-1}\|_2\},
\qquad
v_j=\frac{\Delta_j}{\|\Delta_j\|_2},
\]

\[
\bar\Delta_j
=
\min\left(1,\frac{\tau_{t-1}}{\|\Delta_j\|_2}\right)
\Delta_j.
\]

- spherical k-means 只使用 \(v_j\)；
- prototype 使用裁剪后的 \(\bar\Delta_j\)；
- 聚类和 prototype 均值不使用客户端样本量或 FedAvg 权重；
- FedAvg 权重只用于最终服务器全局聚合。

### 5.4 确定性 spherical k-means

冻结：

- 初始化：k-means++；
- `n_init=1`；
- `max_iter=100`；
- `tol=1e-6`；
- seed：`stable_seed("proposal-cluster", global_seed, round_id, condition)`；
- assignment：最大 cosine；
- cosine tie：较小 cluster index；
- 空 cluster 删除，不重新补充；
- 完成 assignment 后重新计算中心。

最小 cluster 约束：

1. 重复选择成员数小于4的 cluster；成员数并列时选择较小 cluster index；
2. 将其合并到中心 cosine 最大的其他 cluster；并列时选择较小 index；
3. 每次合并后重算中心；
4. 直到所有保留 cluster 至少4个成员；
5. 合并后不重新补足 prototype 数量。

### 5.5 Prototype 和 leave-one-out

对 cluster \(C_m\)：

\[
\mu_m=\frac1{|C_m|}\sum_{j\in C_m}\bar\Delta_j,
\qquad
p_m=\tau\frac{\mu_m}{\|\mu_m\|_2}.
\]

若 \(\|\mu_m\|_2\leq\epsilon_{norm}\)，丢弃该 prototype，禁止使用随机方向填充。

若客户端 \(k\) 上一轮属于 \(C_m\)，从 cluster 裁剪更新之和中移除其贡献：

\[
\mu_{m,-k}
=
\frac1{|C_m|-1}
\sum_{j\in C_m,j\neq k}\bar\Delta_j,
\qquad
p_{m,-k}=\tau\frac{\mu_{m,-k}}{\|\mu_{m,-k}\|_2}.
\]

要求：

- leave-one-out 后至少3个来源；
- LOO mean norm 近零则不给该客户端发送；
- 上一轮未参与的客户端收到完整 prototype；
- 不能从最终归一化 prototype 上直接减去客户端 delta；
- payload 不包含 cluster 成员 ID。

## 6. P-FCC proposal evaluation

所有 proposal 必须从同一个原始 \(\theta_t\) 独立评价：

\[
L_{k,m}
=
\mathcal L_{CE}(\theta_t+\eta p_m;E_k),
\qquad
u_{k,m}=L_k(\theta_t;E_k)-L_{k,m}.
\]

每次 probe 必须：

- 临时加载 \(\theta_t+\eta p_m\)；
- `eval()` 和 `no_grad()`；
- 使用固定 evaluation transform；
- 恢复原始 \(\theta_t\)；
- 不修改 optimizer state；
- 不推进训练 DataLoader；
- 不消耗训练 augmentation RNG；
- 不更新 BatchNorm running statistics；
- 不残留 gradient；
- 不改变后续标准 CE 的 batch/augmentation 流。

禁止在已注入前一个 proposal 的状态上评价后一个 proposal。utility 相等时按 prototype index 做确定性 tie-break。

## 7. Random-Proposal 唯一定义

`Random-Proposal` 冻结为：

- \(\lambda_{RTC}=0\)；
- 使用该条件自己上一轮 delta 构建的相同 bank 规则；
- 对全部可用 proposal 执行 forward 以匹配 P-FCC 计算量；
- utility 计算后完全丢弃，不进入选择；
- 随机选择 \(R_t=\min(2,M_t)\) 个可用 proposal；
- seed：`stable_seed("random-proposal", global_seed, round_id, client_id)`；
- 选中 proposal 使用均匀权重 \(1/R_t\)；
- 不要求 utility 为正；
- correction 使用与 P-FCC 相同的 CE-norm 归一和最终范数匹配；
- bank 为空时 correction 为零。

Random-Proposal 是 P-FCC 私有筛选消融，不包含 D-RTC。

## 8. D-RTC restore direction

Restore 只计算方向，不执行额外本地优化：

1. 完成普通本地 CE；
2. 保存 \(\theta_{k,CE}^t\) 和 \(\delta_k^{CE}\)；
3. 清空已有 gradient；
4. 在 \(\theta_{k,CE}^t\) 上，以 deterministic transform 计算 restore loss；
5. backward 一次；
6. 只读取共享 trainable vision-LoRA gradient；
7. 定义 \(r_k=-\nabla L_k^{restore}\)；
8. 不执行 optimizer step；
9. 不改变 optimizer momentum/scheduler；
10. 读取后清空 gradient，禁止污染下一轮或下一客户端。

恢复损失冻结为：

\[
L_k^{restore}
=
\tfrac12L_{CE}
+\tfrac12T^2
D_{KL}(q_k^{ref,T}\|p_\theta^T),
\]

其中：

- `KLDivLoss(reduction="batchmean")`；
- \(T=2\)；
- CE/KL 权重各0.5；
- 只对 vision-LoRA 求梯度；
- \(d_k^t=0\) 时 RTC correction 严格为零；
- restore gradient 为零时 RTC correction 为零。

## 9. 隐私和通信声明修订

正式威胁模型：

- honest-but-curious server；
- honest、non-colluding clients；
- 不防御客户端串谋；
- 不提供 differential privacy；
- 不兼容服务器只能看到总和的严格 secure aggregation。

服务器与普通非安全聚合 FedAvg 一样看到单客户端普通 LoRA delta。方法只能声明：

> 相对普通非安全聚合 FedAvg，不新增显式 tail-carrier 标记、类别列表、类别计数、样本、标签、logits、loss、utility、accepted index、degradation 或恢复触发状态上行；并且不直接向客户端广播另一个客户端的单独原始更新。

`anonymous prototype` 统一改称：

> multi-source mixed prototype

它是协议层多来源混合，不是形式化 anonymity 保证。

客户端上行 payload 只能包含：

- client ID/协议标识；
- 标准 FedAvg 所需 sample count；
- 一个普通 LoRA delta。

离线研究日志必须与算法通信 payload 完全分离。

## 10. 类别级离线审计

这些字段只写入研究日志，不参与运行时客户端或服务器决策。

### 10.1 P-FCC

对客户端本地已观察类别记录：

\[
u_{k,m,c}
=
L_{k,c}(\theta_t)-L_{k,c}(\theta_t+\eta p_m).
\]

记录：

- accepted/rejected/random 的 per-class utility；
- per-class positive coverage；
- 同一 proposal 同时具有正负类别作用的比例；
- aggregate utility 为正、但至少一个本地类 utility 为负的比例；
- 仅在离线分析中使用全局 bottom-20 身份计算 tail-carrier tail utility；
- `FalseDonor = 1[aggregate utility > 0 and local-tail utility < 0]`。

### 10.2 D-RTC

记录：

\[
g_{k,c}^t
=
L_{k,c}(\theta_t)-L_{k,c}(\theta_{ref}).
\]

审计：

- mean degradation 与 worst-class degradation；
- \(d_k^t=0\) 时是否存在 class-level degradation；
- tail degradation 是否被其他类别 improvement 抵消；
- restore direction 对每个本地类的作用；
- recovery 是否改善尾类同时损害其他类。

出现 class-level miss 只说明 v1 的客户端平均粒度可能不足；不得在 seed-42 v1 中悄悄改成 per-class trigger/scoring。任何运行时粒度改变必须新建 v2。

## 11. 正确性测试 Gate

100轮正式训练前必须通过以下测试。

### A. 零系数基线等价

当 \(\lambda_{FCC}=\lambda_{RTC}=0\) 时，新代码路径必须与原 FedAvg-VisualLoRA 在以下内容数值等价：

- theta0；
- client schedule；
- batch/augmentation 流；
- optimizer/scheduler state；
- 每客户端 CE delta；
- 每轮 FedAvg global model。

这是正式实验的最高优先级 Gate。

### B. 执行顺序

- degradation 在原始 \(\theta_t\) 上计算；
- proposal 不改变 degradation；
- reference 只在本地 CE 前由 incoming global 更新；
- 本地模型永不写入 reference。

### C. Proposal 独立性

- 每个 proposal 从同一 \(\theta_t\) 评价；
- 交换 proposal 评价顺序不改变 utility；
- probe 后参数、RNG、optimizer、BatchNorm 状态恢复。

### D. Leave-one-out

- 成员客户端收到的 prototype 不包含自身；
- 非成员客户端收到完整 prototype；
- LOO 至少3个来源；
- payload 不包含成员 ID。

### E. Cluster 边界

覆盖：0、1--3、4、24以上有效更新；相同方向；相反方向；空/小 cluster；近零 mean；NaN/Inf；确定性 tie。

### F. Reference 状态

覆盖：首次初始化；更优 incoming global；更差 incoming global；本地模型禁止更新；客户端缺席；checkpoint resume；状态丢失。

### G. Restore gradient

覆盖：一次 backward；无 optimizer step；momentum/scheduler 不变；frozen CLIP 无梯度；\(d=0\)；零 restore gradient；无梯度污染。

### H. Norm matching

覆盖：普通 \(z\)；零 CE delta；FCC/RTC 与 CE 抵消；FCC 与 RTC 抵消；近零 norm；mixed precision；flatten/unflatten 后实际 norm。

### I. Random-Proposal

相同 seed、round、client、bank 必须选择相同 prototype；改变 round 或 client 允许改变。

### J. Payload

检查上行不含 functional memory 标签、reference logits、utility、accepted index、degradation、restore trigger、per-class audit 或 tail-carrier 标志。

### K. 两轮状态机 smoke

- 第0轮 bank 为空；
- 第1轮能从第0轮有效更新构建 bank；
- 首次参与只初始化 reference；
- 下一轮正确读取历史 reference；
- proposal probe 不污染 CE；
- norm matching 通过；
- checkpoint/resume 后状态一致。

两轮 smoke 是实现正确性测试，不是用于支持论文结论的实验。

## 12. Stage-3 Gate 的分层

### 12.1 硬 Gate

P-FCC：

- P-FCC-only tail margin 或 worst-neighbor margin 优于 FedAvg；
- 至少12/20尾类 margin 改善；
- non-tail accuracy drop 不超过0.5个百分点；
- 所有上传 norm error 小于 \(10^{-6}\)；
- accepted proposals 在 selection-independent \(A_k\) 上优于 random/rejected。

D-RTC：

- D-RTC-only tail retention 高于 FedAvg、forgetting 低于 FedAvg；
- degradation 对 \(A_k\) 上后续功能退化具有正向预测能力；
- recovery 在 \(A_k\) 上多数为正；
- non-tail accuracy drop 不超过0.5个百分点。

Combined：

- final tail accuracy 至少高于 FedAvg 0.5个百分点；
- tail margin、NLL、retention 至少两个方向一致；
- macro accuracy drop 不超过0.5个百分点；
- P-FCC 与 D-RTC 至少各有一个单组件机制 Gate 通过。

### 12.2 必须报告但不是预注册硬 Gate

- proposal bank 非空率、prototype 数量和来源数；
- accepted/rejected/random per-class coverage；
- false-donor rate；
- 收益的客户端、类别和轮次集中度；
- reference update rate；
- degradation 与 RTC saturation 分布；
- class-level trigger miss；
- FCC/RTC correction cosine、同向/正交/冲突比例；
- 上传方向相对 CE 的旋转角度；
- norm fallback 和组件激活率；
- 实际下行通信和额外计算。

不能仅因为 bank 非空率“看起来低”、RTC 高频触发、收益集中在少数 tail carriers、Combined 与较强单组件接近或 correction 不正交，就事后判定方法失败。这些项目用于解释结果和决定是否建立 v2。

## 13. v1.0.1 禁止新增

- 运行时 per-class P-FCC；
- worst-class D-RTC trigger；
- 第三个 Insight 或组件；
- 新数据集和强基线全量比较；
- partial-participation 正式实验；
- memory/prototype/\(\lambda\) 扫描；
- 理论收敛证明；
- secure aggregation；
- differential privacy。

这些属于 Stage-4 或 v2。边界测试仍必须覆盖少于4个有效客户端、客户端缺席和状态恢复，保证代码不会在未来 partial participation 时崩溃。

## 14. 结果分支

- 两组件均通过：保留 v1.0.1，进入 Stage-4 多 seed、强基线和多数据集；
- 仅 P-FCC 通过：根据 class-level miss、RTC trigger 和 recovery audit 新建 D-RTC v2；
- 仅 D-RTC 通过：根据 bank availability、prototype cancellation、false donor 和 \(A_k\) 泛化新建 P-FCC v2；
- Combined 上涨但两个单组件均不通过：不能声明方法成立，先复核归因；
- 只有 margin/NLL 改善而 accuracy 不变：记录为机制方向成立但固定剂量不足，随后建立独立优化协议，不回改 v1.0.1。
