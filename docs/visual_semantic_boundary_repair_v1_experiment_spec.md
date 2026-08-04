# Visual-Semantic Evidence Audit + Boundary Repair

## 第一版离线 Gate 与初步生死实验实现规格（修订稿）

状态：实现前冻结稿。本文只规定第一版最小研究工具、离线 Gate 和短程工程冒烟，不启动正式长程训练。

## 1. 实验目标与归因边界

第一版只回答以下问题：

> 在不改变 PromptFL 本地训练的条件下，边界级最小写回能否比整客户端放大、ordinary classwise weighting 和 CUSP 更准确地恢复被 FedAvg 稀释的视觉—语义知识？

本地训练保持原始 PromptFL：

\[
\theta_{t,k}=\operatorname{LocalTrain}(\theta_t,D_k).
\]

第一版不加入 hard-negative 训练损失、新 tail loss、class-aware prompt、额外本地训练、tail expert 或 memory。任何收益只能归因于训练后的证据审计和边界写回。

需要避免预设“专业客户端一定学得更好”。第一版必须同时测量知识产生、支持权重稀释和非支持客户端干扰，只有证据满足相应条件时才作机制结论。

## 2. 固定视觉—语义边界与 audit 数据

对 audit 图像 \(x\)、正确类别 \(c\) 和固定易混类别 \(h\)，定义：

\[
m_{c,h}(x;\theta)
=s(f_v(x),f_t(c;\theta))-s(f_v(x),f_t(h;\theta)).
\]

CLIP 视觉编码器冻结，只训练共享 Prompt。每张本地训练图像生成一个训练流程未使用的确定性 `audit view`，并缓存归一化图像特征。audit view 属于算法内部审计数据，不称为 held-out evaluation。

hard negative 必须在诊断轮开始时由轮前模型 \(\theta_t\) 固定，所有候选共享同一集合：

\[
\mathcal H_c=\operatorname{TopK}_{h\ne c}
s(f_v(x),f_t(h;\theta_t)).
\]

第一版固定规则：轮前模型 Top-2 错误类与 zero-shot CLIP Top-1 错误类合并去重，每个类别最多保留 3 条边界。候选冻结后不得重选 hard negative。

audit cache 必须记录：样本稳定 ID、客户端 ID、标签、归一化图像特征、固定 hard-negative、audit transform 配置与随机种子、完整 cache hash。不得使用 official test 选择边界、阈值、repair、backtracking 或超参数。

## 3. 四类评估模型

本节中的四类模型都相对同一个轮前模型 \(\theta_t\) 评估。设本轮被选择客户端集合为 \(\mathcal K_t\)，客户端完整训练样本量为 \(n_k\)，真实 FedAvg 权重为：

\[
\alpha_k=\frac{n_k}{\sum_{j\in\mathcal K_t}n_j},
\qquad
\Delta_k=\theta_{t,k}-\theta_t.
\]

对边界 \(e=(c,h)\)，支持客户端集合定义为：

\[
S_e=\{k\in\mathcal K_t:n_{k,e}>0\},
\]

其中 \(n_{k,e}\) 是客户端 \(k\) 对该固定边界贡献的 audit 样本数。支持质量统计使用 \(n_{k,e}\) 聚合；参数聚合始终使用由完整客户端样本量得到的真实 \(\alpha_k\)，不得混用。

### 3.1 Local models

逐个保留支持客户端本地轮末模型：

\[
\theta^{\mathrm{local}}_{k,e}=\theta_{t,k},
\qquad k\in S_e.
\]

### 3.2 Support-normalized counterfactual

先定义该边界的真实 support mass：

\[
\mu_e=\sum_{k\in S_e}\alpha_k.
\]

只在支持客户端内部重新归一化：

\[
\theta^{\mathrm{S,norm}}_e
=\theta_t+
\sum_{k\in S_e}\frac{\alpha_k}{\mu_e}\Delta_k.
\]

若 \(\mu_e=0\)，该边界不可诊断。该模型称为 counterfactual，不称为 oracle 或理论上界。

### 3.3 Support-actual counterfactual

保留支持客户端在真实 FedAvg 分母下的原始权重，不重新归一化：

\[
\theta^{\mathrm{S,actual}}_e
=\theta_t+
\sum_{k\in S_e}\alpha_k\Delta_k.
\]

### 3.4 Full FedAvg candidate

完整 FedAvg 候选为：

\[
\widetilde\theta_{t+1}
=\theta_t+
\sum_{k\in\mathcal K_t}\alpha_k\Delta_k.
\]

以上四类状态必须由同一份轮前模型、同一批本地轮末模型和同一组 FedAvg 权重构造。

## 4. 三项并列诊断

令客户端 \(k\) 上边界 \(e\) 的 audit 平均 margin 为 \(M_{k,e}(\theta)\)。边界级 pooled 统计固定使用：

\[
\omega_{k,e}=\frac{n_{k,e}}{\sum_{j\in S_e}n_{j,e}},
\qquad
M_e(\theta)=\sum_{k\in S_e}\omega_{k,e}M_{k,e}(\theta).
\]

所有模型必须在同一 audit 样本、同一固定 \(h\) 和同一 \(\omega_{k,e}\) 下评估。

### 4.1 Local audit gain

逐支持客户端计算：

\[
G^{\mathrm{local}}_{k,e}
=M_{k,e}(\theta_{t,k})-M_{k,e}(\theta_t).
\]

同时保留完整逐客户端结果，并报告 pooled local gain：

\[
G^{\mathrm{local}}_e
=\sum_{k\in S_e}\omega_{k,e}G^{\mathrm{local}}_{k,e}.
\]

每条边界至少报告支持客户端数、均值、中位数、最小值、正增益比例和符号一致率。该指标必须命名为 `local_audit_gain`，不能写成 `local_heldout_gain`。样本充足时可以另加真正的 client-held-out 诊断，但不能替代本项。

### 4.2 Support-normalized gain

\[
G^{\mathrm{S,norm}}_e
=M_e(\theta^{\mathrm{S,norm}}_e)-M_e(\theta_t).
\]

它回答：忽略 support mass，仅看支持客户端内部聚合后，知识是否仍然存在。

### 4.3 Support-actual gain

\[
G^{\mathrm{S,actual}}_e
=M_e(\theta^{\mathrm{S,actual}}_e)-M_e(\theta_t).
\]

它回答：支持客户端知识在真实 FedAvg 权重下有多少功能可见性。

完整 FedAvg gain 作为分解所需的附加量：

\[
G^{\mathrm{all}}_e
=M_e(\widetilde\theta_{t+1})-M_e(\theta_t).
\]

### 4.4 机制判断规则

必须按以下规则解释，不能仅凭 `support-actual` 较低声称边界获取不足：

| Local audit | Support-normalized | Support-actual | 主要解释 |
|---|---|---|---|
| 低 | 低 | 低 | 支持客户端边界获取不足，或本地更新冲突严重 |
| 高 | 高 | 低 | 知识已产生，但被 support mass / FedAvg 权重稀释 |
| 高 | 低 | 低 | 单客户端有收益，但支持客户端内部方向冲突或聚合非线性明显 |
| 高 | 高 | 高，而 Full FedAvg 低 | 非支持客户端更新造成干扰 |

“高/低”的阈值必须在实现配置中预先固定，并同时报告连续值，不能只保存布尔结论。

## 5. Dilution / interference 分解

对每条边界定义：

\[
\operatorname{dilution}_e
=G^{\mathrm{S,norm}}_e-G^{\mathrm{S,actual}}_e,
\]

\[
\operatorname{interference}_e
=G^{\mathrm{S,actual}}_e-G^{\mathrm{all}}_e.
\]

因此总的支持知识可见性损失满足恒等式：

\[
G^{\mathrm{S,norm}}_e-G^{\mathrm{all}}_e
=\operatorname{dilution}_e+
\operatorname{interference}_e.
\]

必须逐边界保存并按类别、tail/non-tail、拓扑汇总。至少报告均值、中位数、正值比例以及由少量极端边界主导的程度。

注意：这四类因果诊断状态不得做范数匹配。它们的真实更新幅度正是 support mass 与 FedAvg 权重机制的一部分。可以额外报告 direction-only 的等范数辅助结果，但不得替换原始诊断。

## 6. 可见性缺口与 fragile edge

第一版 repair 的直接目标仍由 local audit 与完整 FedAvg 定义：

\[
d_e=
\left[
\gamma G^{\mathrm{local}}_e-G^{\mathrm{all}}_e-\tau
\right]_+.
\]

默认只保留：

- \(G^{\mathrm{local}}_e>0\)；
- 至少 2 个支持客户端；
- \(d_e>0\)；
- hard negative 已在 \(\theta_t\) 固定。

单支持客户端边界只记录，不执行 repair。第一版不使用预先给定的 tail 标签筛边界。

`gamma`、`tau`、每类最大边界数和总 repair 边界上限都必须写入候选冻结清单。official test 不得参与这些参数的选择。

## 7. 边界级最小写回

在完整 FedAvg 候选处计算梯度：

\[
g_e=\nabla_\theta M_e(\widetilde\theta_{t+1}).
\]

令矩阵 \(G\) 的每一行为 \(g_e^\top\)。求解最小修复：

\[
\min_\delta \frac12\|\delta\|_2^2,
\qquad
g_e^\top\delta\ge d_e.
\]

可使用非负对偶变量：

\[
\max_{\lambda\ge0}
d^\top\lambda-
\frac12\lambda^\top GG^\top\lambda,
\qquad
\delta=G^\top\lambda.
\]

求解器必须记录收敛状态、迭代次数、最大线性约束残差、Gram 条件数或正则化量。若约束在信赖域内不可满足，不得静默返回“成功”；必须报告未闭合边界并由真实 margin 复核。

## 8. 最终更新范数匹配

### 8.1 统一预算

定义 FedAvg 总更新及其预算：

\[
\Delta_{\mathrm{FedAvg}}
=\widetilde\theta_{t+1}-\theta_t,
\qquad
B=\|\Delta_{\mathrm{FedAvg}}\|_2.
\]

修复量首先受信赖域限制：

\[
\|\delta\|_2\le \rho B.
\]

但该条件只限制附加修复，不能代替最终候选的总更新范数匹配。

### 8.2 Repair 候选的最终范数

对每个 backtracking 系数 \(\alpha\)，先形成原始总更新：

\[
\Delta^{\mathrm{raw}}_{\mathrm{repair}}(\alpha)
=\Delta_{\mathrm{FedAvg}}+\alpha\delta.
\]

再执行最终更新范数匹配：

\[
\Delta^{\mathrm{match}}_{\mathrm{repair}}(\alpha)
=B\frac{\Delta^{\mathrm{raw}}_{\mathrm{repair}}(\alpha)}
{\|\Delta^{\mathrm{raw}}_{\mathrm{repair}}(\alpha)\|_2},
\]

\[
\theta^{\mathrm{repair}}_{t+1}(\alpha)
=\theta_t+
\Delta^{\mathrm{match}}_{\mathrm{repair}}(\alpha).
\]

每次 norm matching 后必须重新计算真实边界 margin、安全指标和缺口闭合；不能用匹配前的一阶约束结果代替。

若 \(B=0\) 或原始 repair 总更新范数为 0，候选必须回退为 FedAvg 并记录原因。

### 8.3 所有性能对照的匹配规则

FedAvg、inverse-support-mass、ordinary classwise、CUSP、random repair、ordinary audit-gradient 和 edge-level repair 的最终更新均以 \(B\) 为共同预算。每个候选清单必须保存：

- 匹配前总更新范数；
- 匹配系数；
- 匹配后总更新范数；
- 相对误差；
- 最终候选 hash。

norm-matched random repair 与 ordinary audit-gradient 除最终总更新匹配外，还必须匹配 edge repair 在 backtracking 后的附加修复范数，避免对照获得不同的搜索半径。

Support-normalized 与 support-actual 状态属于因果诊断，不参加上述最终性能候选的统一范数处理。

## 9. Backtracking 与安全检查

依次尝试：

\[
\alpha\in\{1,0.5,0.25,0.125\}.
\]

每个 \(\alpha\) 都必须经过最终范数匹配后再执行：

1. fragile edge 的真实 margin 与缺口闭合检查；
2. 非目标 audit 边界下降检查；
3. repair 相对 FedAvg 的额外文本原型几何漂移检查；
4. 数值有限性和候选 hash 检查。

official test、head accuracy、tail accuracy 均不得参与 backtracking。若所有 \(\alpha\) 不安全，回退 FedAvg 并记录每一级拒绝原因。

文本几何同时报告总漂移与 repair 增量漂移：

\[
D_{\mathrm{sem,total}}(\theta)
=\frac1{C^2}
\|T(\theta)T(\theta)^\top-T(\theta_0)T(\theta_0)^\top\|_F,
\]

\[
D_{\mathrm{sem,repair}}
=D_{\mathrm{sem,total}}(\theta^{\mathrm{repair}}_{t+1})
-D_{\mathrm{sem,total}}(\widetilde\theta_{t+1}).
\]

## 10. Dump 与代码结构

新增独立模块，不修改 CUSP 对照的语义：

```text
utils/
  boundary_audit.py
  boundary_repair.py
  boundary_metrics.py

scripts/
  dump_boundary_gate.py
  run_boundary_gate.py
  summarize_boundary_gate.py
  run_boundary_gate.sh
```

`federated_main.py` 只在开关启用时保存指定轮次数据；关闭开关后原始 PromptFL 结果必须逐参数一致。

最小 dump：

```text
metadata.json
round_state.pt
audit_cache.pt
```

`round_state.pt` 至少包含：

```python
{
    "global_before_trainable": ...,
    "local_trainable_states": ...,
    "fedavg_candidate_trainable": ...,
    "selected_client_ids": ...,
    "client_sample_counts": ...,
    "client_edge_counts": ...,
    "fedavg_weights": ...,
    "flatten_spec": ...,
}
```

`audit_cache.pt` 至少包含：

```python
{
    "schema_version": ...,
    "source": "train_audit_view",
    "official_test_used": False,
    "audit_transform": ...,
    "audit_seed": ...,
    "clients": {
        client_id: {
            "sample_ids": ...,
            "image_features": ...,
            "labels": ...,
            "fixed_hard_negatives": ...,
        }
    },
}
```

候选状态、清单和 hash 必须在第一次读取 official test 前落盘冻结。

## 11. Gate 0：离线单轮生死实验

固定 CIFAR100-LT、imbalance factor 0.01、PromptFL、CLIP ViT-B/16、30 clients、frac=1、local epochs=3、round 10、seed 42，以及两种拓扑：Client-LT 与 Dirichlet beta=0.5。

同一份本地模型 dump 离线构造：

1. FedAvg；
2. inverse-support-mass weighting；
3. ordinary classwise weighting；
4. CUSP minimal；
5. norm-matched random repair；
6. norm-matched ordinary audit-gradient；
7. norm-matched edge-level boundary repair；
8. support-normalized counterfactual，仅作为诊断参考，不进入公平性能排名。

Gate 0 在读取 official test 前必须完成：四类状态诊断、三项 gain、dilution/interference、边界选择、repair 求解、安全 backtracking、最终范数匹配和所有候选冻结。

若 audit transform、样本稳定 ID 与 hard-negative 规则已经在本规格中冻结，允许从可复现的原始训练划分和现有 round-10 状态确定性重建 audit cache；不因缓存生成时间较晚而强制重训。若无法验证训练划分、状态或 cache hash，则必须重新生成 dump。

## 12. Gate 1 与 Gate 2

Gate 1 为 Client-LT、seed 42 的短程在线工程冒烟。必须至少在非末轮执行一次 repair，并继续一个完整通信轮，以验证 repair 状态能够成为下一轮初始化。若只在第 5 轮 repair，则运行至少 6 轮。

Gate 2 为 20 轮方法生死实验，运行 Client-LT 与 Dirichlet，比较 FedAvg、classwise、CUSP 和 boundary repair。seed 42 只作为开发 Gate；通过并冻结公式与超参数后，使用 seed 2026 确认，暂不运行 100 轮或大规模多 seed。

## 13. 必须报告的指标

除 overall、non-tail 和 tail accuracy 外，至少报告：

- 逐客户端 `local_audit_gain` 及其 pooled 统计；
- `gain_support_normalized`；
- `gain_support_actual`；
- `gain_all_fedavg`；
- `support_mass`；
- `dilution` 与 `interference`；
- visibility deficit；
- deficit closure ratio；
- fragile-edge repair success rate；
- boundary reversal rate；
- 匹配前后最终更新范数和相对误差；
- 附加 repair/FedAvg 范数比；
- 非目标边界下降；
- 总文本几何漂移与 repair 增量漂移；
- 每轮被修复类别数和边界数；
- 额外推理、通信和求解开销；
- 所有 fallback、不可行和 backtracking 拒绝原因。

缺口闭合率定义为：

\[
R_{\mathrm{close},e}
=\frac{G^{\mathrm{repair}}_e-G^{\mathrm{all}}_e}
{\gamma G^{\mathrm{local}}_e-G^{\mathrm{all}}_e},
\]

仅在分母大于预设数值阈值时报告；同时保存未裁剪值与裁剪到 \([0,1]\) 的展示值。

## 14. 第一版判定原则

任何“边界获取不足”结论都必须由 local audit 与 support-normalized 同时偏低支持。若它们正常而 support-actual 明显较低，结论应是权重稀释；若 support-actual 正常而完整 FedAvg 较低，结论应是非支持客户端干扰。

正式性能结论只比较最终总更新范数匹配后的候选。若 boundary repair 的收益在匹配后消失，则不得归因于边界选择或求解质量。

绿色、黄色和红色三级判断沿用原实验设计，但其中“明显提升”“明显下降”和“语义漂移恶化”必须在运行前转化为配置中的数值阈值；连续指标与置信区间必须与三级标签同时报告。
