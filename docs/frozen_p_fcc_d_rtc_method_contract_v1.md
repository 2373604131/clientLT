# P-FCC + D-RTC 方法运行契约 v1

状态：**FROZEN FOR STAGE-3 SEED-42 MVP**  
冻结日期：2026-08-24  
适用范围：Client-Exposure Long Tail 下的 federated vision-LoRA 方法成立性验证。  
变更规则：本文件冻结后不得根据 seed 42 的最终测试结果原地修改。任何算法性变更必须新建 v2，并保留 v1 结果。

## 1. 冻结的问题与 Insight

### Insight 1：Functional Donor Access Scarcity

Client-LT 将尾类监督集中到少数 tail-carrier clients，减少了尾类有效载体及其可利用的正向功能来源。语义相关性可以富集 donor，但不能保证更新有益，也不要求相关类别与尾类必须在同一客户端训练。因此，方法不恢复未经证明的“同客户端语义共适应”，而是扩大客户端对经过私有功能验证的 donor directions 的访问。

### Insight 2：Privately Verifiable Signed Rewrite Risk

类别缺失更新也能改变共享视觉 LoRA 中的尾类功能，其作用稳定、类别条件且带符号。少量本地私有证据能够识别 donor 与 destructive rewriter，并预测破坏性方向累积后的 retention。方法不假设所有类别缺失更新均有害，也不假设更新会在知识形成后普遍发生 donor-to-rewriter 符号翻转。

## 2. 不可违反的运行约束

1. 服务器没有公共、代理、生成或测试图像。
2. 客户端不上传类别列表、类别计数、样本、标签、logit、loss、proposal 分数或恢复触发状态。
3. 客户端上行仍然只有一个与 FedAvg-VisualLoRA 形状相同的普通 LoRA delta。
4. 所有 proposal 评价、历史退化检测和参考知识维护均发生在客户端本地。
5. 测试集不参与 proposal 构建、客户端功能评价、恢复触发、超参数选择或参考状态更新。
6. 所有客户端执行同一算法；服务器不知道也不标记 tail-carrier client。
7. 不改变 FedAvg 的客户端样本量权重，不给任何客户端统一增加权重。
8. P-FCC 与 D-RTC 只能旋转客户端本来要上传的更新方向，最终上传范数不得超过同轮普通本地 CE 更新范数。
9. proposal bank 只使用上一轮客户端更新构建，禁止使用当前轮尚未完成的更新，因而不增加同步通信轮次。
10. 运行时不使用类别名称、CLIP 文本相似度或语义类别清单。语义相似度只保留为机制分析结果，不是方法输入。

## 3. 隐私和威胁模型边界

本方法采用与普通、非安全聚合 FedAvg 相同的 honest-but-curious server 边界：服务器可以接收每个参与客户端的普通 LoRA delta。方法不声称形式化差分隐私，也不声称兼容“服务器只能看到总和”的严格 secure aggregation。

相对普通 FedAvg，本方法不增加任何客户端私有语义信息上行。服务器不接收功能分数和类别信息。为避免将单个客户端 delta 直接暴露给其他客户端，服务器只能广播由多个客户端共同形成的匿名 proposal prototype：

- 每个原始 cluster 至少包含4个客户端更新；
- 给属于该 cluster 的客户端广播时，先减去该客户端自己的贡献；
- leave-one-out prototype 仍至少由3个其他客户端更新组成；
- 客户端看不到 prototype 的成员身份和单个原始 delta。

## 4. 客户端私有功能记忆

每个客户端第一次参与时，从自己的本地训练集确定性构建固定功能记忆 \(E_k\)：

- 最大32个样本；
- 在客户端本地按已观察类别尽可能均衡采样；
- 固定 seed 为 `global_seed + client_id`；
- 样本、标签和预测永不离开客户端；
- \(E_k\) 可以继续参与普通本地 CE，本文不将其表述成独立验证集；
- 所有对照方法使用完全相同的本地训练数据，不因建立 \(E_k\) 而减少基线训练样本。

proposal 评分、历史 reference 评价和恢复梯度均使用冻结的 deterministic evaluation transform；普通本地 CE 继续使用原有训练 augmentation。这样避免把随机增强噪声误判成 donor 或 degradation。

客户端持久保存：

- \(E_k\) 的样本 ID；
- 历史最佳 **incoming global model** 在 \(E_k\) 上的参考分布 \(q_k^{ref}\)；
- 历史最佳 incoming global 状态在 \(E_k\) 上的最低交叉熵 \(\ell_k^{ref}\)；
- reference version 和最后更新时间。

这些状态均为客户端本地状态，不参与服务器聚合。

reference 只能由客户端实际收到的全局模型 \(\theta_t\) 更新，不能由本地训练后的 \(\theta_{k,CE}^t\) 或最终本地模型更新。否则 incoming global 几乎总比历史本地模型差，D-RTC 会退化成始终开启的个性化正则器，而不再检测共享知识的历史退化。

## 5. 服务器 proposal bank

设上一轮参与客户端集合为 \(S_{t-1}\)，其上传更新为 \(\Delta_j^{t-1}\)。第0轮 proposal bank 为空。

### 5.1 更新预处理

服务器计算上一轮非零更新范数中位数 \(\tau_{t-1}\)，并对每个更新进行裁剪和单位化：

\[
v_j=\frac{\Delta_j^{t-1}}{\|\Delta_j^{t-1}\|_2+\epsilon},
\qquad
\bar\Delta_j=
\min\left(1,\frac{\tau_{t-1}}{\|\Delta_j^{t-1}\|_2+\epsilon}\right)
\Delta_j^{t-1}.
\]

聚类只使用 \(v_j\) 的余弦方向；客户端样本量和类别信息不进入聚类。

### 5.2 匿名原型

- 使用确定性 spherical k-means；
- 最大 prototype 数 \(M=6\)；
- 实际 \(M_t\leq\lfloor |S_{t-1}|/4\rfloor\)；
- 小于4个成员的 cluster 必须合并到余弦最近 cluster；
- prototype 是 cluster 内裁剪更新的均值方向，并重新缩放到 \(\tau_{t-1}\)；
- 聚类 seed 固定为 `global_seed + round_id`。

对于客户端 \(k\)，如果它属于某个上一轮 cluster，则服务器从 cluster 更新之和中减去 \(k\) 的更新再构建 \(p_{m,-k}^{t-1}\)。如果 leave-one-out 后少于3个成员，该 prototype 不发送给 \(k\)。因此客户端评价的 proposal 至少由3个其他客户端共同提供。

服务器向第 \(t\) 轮客户端发送：

\[
(\theta_t, B_{-k}^{t-1}),
\qquad
B_{-k}^{t-1}=\{p_{1,-k}^{t-1},\ldots,p_{M_t,-k}^{t-1}\}.
\]

proposal bank 只有一轮寿命，不跨多轮直接累积。

## 6. 正确的单轮执行顺序

原始“先 P-FCC、再检测 D-RTC”的顺序被修正。D-RTC 必须在未注入 proposal 的原始全局状态 \(\theta_t\) 上检测历史退化，否则无法区分聚合伤害和 proposal 作用。

```text
服务器广播 θt 与上一轮匿名 proposal bank B(t-1)
                    ↓
客户端在原始 θt 上计算历史退化分数（D-RTC detection）
                    ↓
客户端独立评价每个 proposal 的当前私有功能作用（P-FCC scoring）
                    ↓
客户端从 θt 执行完全标准的本地 CE，得到基准更新 δCE
                    ↓
P-FCC 产生 donor correction；D-RTC 产生 recovery correction
                    ↓
将 δCE 与两个 correction 做固定预算的方向合成
                    ↓
上传一个与 ||δCE|| 完全相同的普通 LoRA delta
                    ↓
服务器执行原始 FedAvg，并用本轮上传 delta 构建下一轮 bank
```

## 7. P-FCC：Private Functional Cross-Client Complementation

### 7.1 私有 proposal 评分

客户端在 \(E_k\) 上计算基础 CE：

\[
L_k^0=\mathcal L_{CE}(\theta_t;E_k).
\]

对每个 proposal 使用固定探测剂量 \(\eta=0.5\)：

\[
u_{k,m}
=
L_k^0-
\mathcal L_{CE}(\theta_t+\eta p_{m,-k}^{t-1};E_k).
\]

\(u_{k,m}>0\) 才视为本地 donor。客户端只保留 utility 最大的前 \(R=2\) 个正向 proposal。若没有正向 proposal，则本轮 P-FCC correction 为零。

正向系数为：

\[
a_{k,m}
=
\frac{\max(u_{k,m},0)}
{\sum_{r\in TopR}\max(u_{k,r},0)+\epsilon}.
\]

原始 donor correction：

\[
f_k=\sum_{m\in TopR}a_{k,m}p_{m,-k}^{t-1}.
\]

P-FCC 不上传 \(u_{k,m}\)、\(a_{k,m}\) 或被接受 proposal 的索引。

### 7.2 P-FCC 方向预算

标准本地 CE 完成后得到：

\[
\delta_k^{CE}=\theta_{k,CE}^{t}-\theta_t.
\]

将 donor correction 归一到基准更新范数：

\[
\widehat f_k
=
\frac{\|\delta_k^{CE}\|_2}
{\|f_k\|_2+\epsilon}f_k.
\]

冻结系数为 \(\lambda_{FCC}=0.5\)。

## 8. D-RTC：Degradation-Triggered Retentive Correction

### 8.1 历史退化检测

客户端必须在原始 \(\theta_t\) 上计算：

\[
\ell_k^t=\mathcal L_{CE}(\theta_t;E_k),
\qquad
d_k^t=
\operatorname{clip}
\left(
\frac{\ell_k^t-\ell_k^{ref}}
{\ell_k^{ref}+\epsilon},0,1
\right).
\]

若客户端尚无 reference，令 \(d_k^t=0\)，并使用当前状态初始化 reference。D-RTC 不采用额外二值阈值；恢复强度随相对退化连续变化，避免增加 trigger threshold 超参数。

如果当前 incoming global CE 严格低于 \(\ell_k^{ref}\)，则当前全局模型成为新的历史最佳 reference，客户端先更新 \(q_k^{ref}\) 与 \(\ell_k^{ref}\)，并令本轮 \(d_k^t=0\)。否则保留历史 reference 并按上式计算退化分数。

### 8.2 恢复方向

完成普通本地 CE 后，在 \(\theta_{k,CE}^t\) 上使用功能记忆计算：

\[
L_k^{restore}
=
\frac12\mathcal L_{CE}(\theta;E_k)
+
\frac12 T^2
D_{KL}
\left(
q_k^{ref,T}\,\|\,p_{\theta}^{T}(E_k)
\right),
\]

温度冻结为 \(T=2\)。只计算一次恢复梯度：

\[
r_k=-\nabla_{\theta}L_k^{restore}
\big|_{\theta=\theta_{k,CE}^t},
\qquad
\widehat r_k
=
\frac{\|\delta_k^{CE}\|_2}
{\|r_k\|_2+\epsilon}r_k.
\]

若 \(d_k^t=0\)，则恢复 correction 为零。冻结系数为 \(\lambda_{RTC}=0.5\)。

恢复梯度只对共享 vision-LoRA 的可训练参数计算；冻结 CLIP 主干和文本原型始终不参与更新。

### 8.3 私有 reference 更新

reference 只在本轮本地训练开始前，根据原始 incoming global \(\theta_t\) 更新。P-FCC 探测状态、本地 CE 模型、recovery 后模型和最终上传方向均不得写入 reference。测试集不参与该过程。

## 9. 固定范数的组合与上传

客户端形成未归一化方向：

\[
z_k
=
\delta_k^{CE}
+\lambda_{FCC}\widehat f_k
+\lambda_{RTC}d_k^t\widehat r_k.
\]

最终上传：

\[
\delta_k^{upload}
=
\begin{cases}
\displaystyle
\frac{\|\delta_k^{CE}\|_2}
{\|z_k\|_2+\epsilon}z_k,
&\|\delta_k^{CE}\|_2>0,\\
0,&\text{otherwise}.
\end{cases}
\]

因此：

\[
\|\delta_k^{upload}\|_2
=
\|\delta_k^{CE}\|_2.
\]

P-FCC 和 D-RTC 只能改变上传方向，不能增加单客户端更新范数。服务器继续执行原始 FedAvg：

\[
\theta_{t+1}
=
\theta_t+
\sum_{k\in S_t}
\frac{n_k}{\sum_{j\in S_t}n_j}
\delta_k^{upload}.
\]

如果 \(f_k=0\)，定义 \(\widehat f_k=0\)；如果恢复梯度为零，定义 \(\widehat r_k=0\)。这些情况不得通过随机方向或历史方向填充。

### 9.1 边界情况

- 第0轮没有上一轮更新，P-FCC 自动关闭，D-RTC 只初始化 incoming-global reference；
- 上一轮少于4个有效非零客户端更新时，proposal bank 为空；
- 客户端本地样本少于32时，\(E_k\) 使用全部可用样本；
- 客户端上一轮未参与时，不执行 leave-one-out；
- proposal cluster 无法满足最小来源数量时必须合并，不能退化为广播单个更新；
- 普通本地 CE 更新范数为零时，最终上传严格为零；
- 客户端丢失持久状态时按首次参与处理，不从服务器恢复 reference。

### 9.2 计算与通信预算

- 上行通信与 FedAvg-VisualLoRA 完全相同：每轮每客户端一个 LoRA delta；
- 下行额外发送最多6个 LoRA prototype，不增加同步轮次；
- 每个客户端额外执行最多6次 proposal forward 和1次 restore backward；
- 客户端额外持久状态最多为32个样本 ID、32组参考 logits及两个标量元数据；
- 必须记录实际下行字节数、额外 forward/backward 次数和本地状态大小，最终论文不能只报告精度。

## 10. 冻结的 Stage-3 MVP 超参数

| 项目 | 冻结值 |
|---|---:|
| functional memory size | 最多32 |
| proposal source | 仅上一轮上传更新 |
| maximum prototypes \(M\) | 6 |
| minimum source clients per cluster | 4 |
| minimum leave-one-out sources | 3 |
| proposal probe dose \(\eta\) | 0.5 |
| accepted proposals \(R\) | top-2 positive |
| \(\lambda_{FCC}\) | 0.5 |
| RTC relative degradation range | clip到[0,1] |
| distillation temperature \(T\) | 2 |
| restore gradient calls per round | 1 |
| \(\lambda_{RTC}\) | 0.5 |
| proposal lifetime | 1轮 |
| upload norm | 与 \(\delta_k^{CE}\) 完全相同 |

这些数值只冻结用于 Stage-3 seed-42 MVP。后续 SOTA 优化可以建立单独调参协议，但不得回写或选择性覆盖 v1 结果。

## 11. 必须实现的对照

相同 theta0、数据划分、客户端参与序列、augmentation、local epochs、学习率和训练轮数下运行：

1. `FedAvg-VisualLoRA`：原始基线；
2. `P-FCC-only`：令 \(\lambda_{RTC}=0\)；
3. `D-RTC-only`：令 \(\lambda_{FCC}=0\)；
4. `P-FCC+D-RTC`：完整方法；
5. `Random-Proposal`：使用相同 proposal 数量和范数，但不做私有正向筛选，作为 P-FCC 必要性消融。

所有方法仍按原始 FedAvg 样本量权重聚合。

## 12. 日志与禁止项

### 12.1 允许写入实验日志但不得上传给服务器算法的离线诊断

- 本地 proposal utility 分布；
- proposal 接受率；
- RTC 退化分数和触发率；
- FCC/RTC correction 与 CE update 的余弦；
- norm matching 相对误差；
- tail-carrier 与普通客户端的离线分组统计。

这些日志只用于实验审计，不能作为运行时服务器决策输入。

### 12.2 明确禁止

- 根据客户端是否持有尾类改变其权重；
- 使用测试 accuracy/margin 选择 proposal 或 reference；
- 根据测试结果更改本轮 \(M,R,\eta,\lambda_{FCC},\lambda_{RTC},T\)；
- 将单个原始客户端 delta 直接广播给其他客户端；
- 使用服务器类别原型、公共尾类图像或生成式代理图像进行功能判断；
- 把方法描述成形式化隐私保护或 secure-aggregation-compatible，除非后续另行实现并验证。

## 13. Stage-3 成立性 Gate

### P-FCC Gate

- P-FCC-only 的 tail margin 或 worst-neighbor margin 优于 FedAvg；
- accepted proposals 的离线独立功能收益优于 rejected/random proposals；
- 上传范数匹配误差满足 \(<10^{-6}\) 相对误差；
- 至少12/20尾类的 margin 改善；
- non-tail accuracy 相对 FedAvg 下降不超过0.5个百分点。

### D-RTC Gate

- D-RTC-only 的 tail retention 高于 FedAvg、forgetting 低于 FedAvg；
- 较高本地退化分数能够预测后续尾类功能损失；
- recovery 后多数客户端的私有功能改善；
- non-tail accuracy 相对 FedAvg 下降不超过0.5个百分点。

### Combined Gate

- 完整方法 final tail accuracy 至少高于 FedAvg-VisualLoRA 0.5个百分点；
- tail margin、NLL 和 retention 至少两个方向一致；
- macro accuracy 相对 FedAvg 下降不超过0.5个百分点；
- P-FCC 与 D-RTC 至少各有一个单组件 Gate 通过。

若只改善连续 margin/NLL 而 accuracy 不变，结论只能是“机制转化成立但决策边界剂量不足”，随后进入独立的优化协议，不能改写本契约。

## 14. 允许的论文声明边界

若 Stage-3 Gate 通过，可以声明：

> Client-LT limits access to useful cross-client functions while shared LoRA permits class-absent updates to exert signed effects. P-FCC privately selects cross-client donor prototypes, and D-RTC locally detects and repairs historical functional degradation, without public proxy data or class-list disclosure.

即使 Gate 通过，也不能仅凭 Stage-3 声称：

- 所有类别缺失更新均有害；
- Client-LT 已被证明形成语义狭窄的全局知识；
- donor 会在知识形成后普遍变成 rewriter；
- 方法提供形式化隐私保证；
- 方法已经达到 SOTA。
