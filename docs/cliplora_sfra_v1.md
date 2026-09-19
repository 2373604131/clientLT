# SFRA V1：来源感知的 A 功能保持

本文件是可执行 V1 协议，以本次老师修订为准，替代此前设计中的加权 RMS、正裕度历史门槛、无支持默认最高权重和自适应步长等选择。当前是待验证的方法实现，不是已获得效果的结论。

## 1. 方法如何接到现有代码

沿用 `LAControlRuntime.train_phase` 的 S 训练路径、客户端顺序、分区、优化器和 LA。原有 E0–E5/J/S 默认行为不变。新增入口 `scripts/run_cliplora_sfra.py`，运行时为 `utils/cliplora_sfra.py`。

一轮只做以下事情：

1. 固定 A，各客户端按原 S 训练 B 三个 epoch，按样本量聚合 B。
2. 第 1–90 轮，所有客户端从上述共同模型出发，固定 B，用 LA 训练 A 一个 epoch，得到本地提案。样本加权平均形成普通 A 提案。
3. 在共同的 B 聚合后模型上，观察哪些客户端提案能改善哪些本地功能，形成来源优先级和本轮固定目标。
4. 从普通 A 提案出发，做三步真实功能梯度修正，固定提交第三步；不选优、不回滚、不做 B look-ahead。
5. 只用正式提交后的功能分数更新历史。新历史从下一轮起生效。

第 91–100 轮只有原 B 训练；没有 A 本地 epoch、来源梯度或功能修正。继续测量功能和登记历史，观察保持能否延续。

基础配置：CIFAR-100-LT，全局不平衡率 0.01；30 客户端全部参与；100 轮；rank=4、alpha=1、实际 LoRA scaling=0.5；vision top3 的 q/v；FP32；B 和本地 A 的 LR 都是 0.001；LA tau=1。本地 A SGD momentum=0.9、weight decay=0；B 完全使用原 S 优化器配置。数据划分的 specialization_lambda=0.75 **不是**下面搜索的保持强度。

## 2. 老师的关键修订已落地

| 项目 | V1 规则 |
|---|---|
| 功能单位 | 一个客户端中的一个非空类别；每类固定至多 8 张本地训练图像 |
| 视图 | 沿用仓库 CIFAR 的确定性预处理，再加水平翻转；不改原训练预处理 |
| 功能 | 两视图分别计算正确类余弦相似度减全部 100 类中最强错误类相似度，样本平均 |
| 来源起点 | 全部功能梯度在本轮 B 聚合后共同模型上计算 |
| 有效来源 | 两视图一阶响应的最小值严格大于 1e-6 |
| 当前改善 u | 仅对正向来源的有效响应求平均；不除以全部客户端，不乘样本量 |
| 优先级 | 正向来源集中度 1/N_eff；无支持沿用缓存，初值为 1/30 |
| 当前目标 | 当前每视图分数 + 0.5u，最高为 2 |
| 历史初值 | 仅记初始两视图最小分数作为参考，历史目标为空 |
| 历史登记 | 固定非重叠五轮块；该块所有正式轮次/视图的最小值，比初始参考或已有历史高至少 0.001 才登记 |
| 负裕度 | 稳定的负裕度改善也能登记，不要求先正确分类 |
| 合并目标 | 当前与历史取较高者；两者都没有则不激活，不额外设置初始能力保护目标 |
| 功能权重 | 对全体激活 token 只归一化一次，不再乘客户端样本权重或做客户端内平均 |
| 半径 R | 所有客户端 A 提案范数的**未加权 RMS** |
| 功能尺度 | sigma=max(0.001, R × 完整 A 功能梯度范数) |
| 来源传输 | 未截断薄 QR；反馈用 Q^T g，还原对所有客户端提案的响应 |
| 修正空间 | 完整 A 参数空间，不限制在 QR 提案子空间 |

来源只决定**功能维护优先级**，不直接修改普通提案的客户端 FedAvg 权重。同一个类别在不同客户端上是不同 token，这不是全局类别均衡。

## 3. 唯一的三步更新

令普通提案为 `A_proposal=A_previous+d0`，实际候选为 `A(z)=A_proposal+R*z`。

```text
L_func = sum_q omega_q * (1/4) * sum_{v=1,2} positive((T_qv-F_qv)/sigma_qv)^2
J(z)   = 0.5 * ||z||^2 + retention_weight * L_func(A(z))
z0     = 0
z_next = project_to_unit_ball(z - 0.1 * (z + retention_weight * R * gradient_A(L_func)))
```

三步中 B、目标、尺度、来源权重、普通提案均不变。所有客户端在同一候选上计算梯度，先求和再更新一次 A。每步重算最强错误类，不跨本地训练和三步迭代做高阶反传。最终固定提交 z3。

无激活目标或 R≤1e-12 时无需修正，直接提交普通提案，并显式记录原因。任何非有限值均保存 `failure_diagnostic.pt` 后停止，不静默换用另一个模型。约束限制的是额外修正量，不保证总 A 更新不会被部分抵消，因此同时记录实际 `0.5*B*delta_A` 的有效变化。

实现分批时先精确累计完整 token 的 F 与梯度，再形成其损失梯度；不会平方各个小批次的独立缺口。文本分支完全冻结，其确定性的标准化特征只缓存一次，图像分支依然真实前向与反传。

## 4. 第一批实验及一个参数搜索

| 组别 | 当前目标 | 来源权重 | 历史目标 | 三步修正 |
|---|---|---|---|---|
| S | 无 | 无 | 无 | 无 |
| Current | 有 | 有 | 不使用 | 有 |
| Full | 有 | 有 | 有 | 有 |
| Flat | 有 | 全部原始权重为 1 | 有 | 有 |

Flat 保留 u、目标定义、历史规则、见证数量、修正次数，只改变来源权重。各组独立训练后，目标和历史的具体数值自然会随轨迹不同，不能声称数值也保持相同。

**只搜索保持强度 `--retention-weight`，先跑 1、3、10，默认 10。** 小值接近普通学习提案、大值更重视功能目标；半径仍相同。这是开发期敏感性搜索，不能把从官方测试分数挑出的最大值作为无偏最终结果。正式确认需要固定选定参数再跑新种子；不临时改其他规则。Current/Flat 与 Full 使用同一个 λ 做消融。

以下命令均在仓库根目录、已激活 `clientlt` 环境中执行，在前台显示报错。每个节点只有一张分配到的 GPU 时，无需再设置物理 GPU 编号。

```bash
python -u scripts/run_cliplora_sfra.py --method full --retention-weight 1
```

```bash
python -u scripts/run_cliplora_sfra.py --method full --retention-weight 3
```

```bash
python -u scripts/run_cliplora_sfra.py --method full --retention-weight 10
```

默认 λ=10 的必要消融：

```bash
python -u scripts/run_cliplora_sfra.py --method current --retention-weight 10
```

```bash
python -u scripts/run_cliplora_sfra.py --method flat --retention-weight 10
```

S 已经完成且分区、初始化、调度和基础训练配置对应时，复用已有结果，不强制重跑。需要新的同协议 S 时：

```bash
python -u scripts/run_cliplora_sfra.py --method s
```

只有一张卡可用时顺序执行，例如：

```bash
python -u scripts/run_cliplora_sfra.py --method full --retention-weight 1 && python -u scripts/run_cliplora_sfra.py --method full --retention-weight 3 && python -u scripts/run_cliplora_sfra.py --method full --retention-weight 10
```

## 5. 数据协议和结果位置

默认 Client-LT 复用与原 LA-control S 相同的 `a_refresh_topology_bridge/seed42/client-longtail/c1` 分区顺序与调度。若服务器只保留了完整 S 目录，在**所有新方法命令末尾**追加：

```bash
--reference-run output/cifar100_LT/la_control/seed42/client-longtail/s/tau1_a1_protocol42
```

`--reference-run` 要求完整的 partition_manifest、bridge_metadata 和 protocol，不是仅保留分数的轻量包。显式指定不同拓扑会报错，不会将 Client-LT 容量复制给标准 Dirichlet。

如果没有历史协议，可对所有新组统一加 `--fresh-protocol`。此时应运行对应新 S，不能默认旧 S 逐客户端顺序相同。标准 Dirichlet 用 `--partition noniid-labeldir-fine`，默认 beta=0.5，不使用 matched-Dirichlet。

默认 Full10 输出：

```text
output/cifar100_LT/sfra_v1/seed42/client-longtail/full/lambda10_protocol42/
```

不同 λ 独立目录。中断后续训使用原命令加 `--resume`，例如：

```bash
python -u scripts/run_cliplora_sfra.py --method full --retention-weight 10 --resume
```

读取该目录保存的原启动配置，从最后完整轮结束恢复；未完成的整轮重算。保存模型全部变化、训练随机数、见证 ID、初始参考、历史有效位与水平、来源缓存、五轮块最小值和累计轮数，不只保存 A/B。

## 6. 怎么收集和判断结果

汇总并打包（不需要 GPU，也不加载训练模型）：

```bash
python scripts/run_cliplora_sfra.py --stage pack
```

输出 `output/cifar100_LT/sfra_v1_analysis.tar.gz`。若要把旧 S 分数一起放到汇总表，追加上述 `--reference-run`。分析包不含模型权重，不能用于续训。

主要文件：

- `round_metrics.csv` / 逐类 CSV：正式每轮结果，主指标为第 81–100 轮均值。
- `sfra_rounds.csv`：来源支持数、历史数、普通/修正后功能缺口、A 参数与有效矩阵变化。
- `sfra_rounds/rXXX/tokens.npz`：逐 token 的 B 后、普通 A 提案、正式提交后功能分数；下一轮的 B 后分数用于观察持续保持。
- `private_witness_manifest.json`：token 与本地类别/样本身份对应。含私有诊断信息，只用于这次集中模拟和研究分析，不是实际隐私保障协议。
- `sfra_costs.csv`：来源计算、QR 基下发、压缩反馈、三次修正及正式提交评估成本；`budget.csv` 为原本地训练成本。不能把全方法通信宣称为 30 维。
- `checkpoints/sfra_last.pt` 与 `base_model.pt`：实际续训状态；`completion.json` 标记完成。

读 token 数组时，当前目标仅在 `supported=True` 上存在，最终目标仅在 `active=True` 上存在；历史缺口也必须分别结合登记前/后的 `history_valid_*` 掩码。`history_gap_before` 对应本轮已经存在的历史，`history_gap_after_registration` 对应供下一轮使用的历史，两者不要混用。

先看 Full 是否改善 S 的 Overall–Tail 保持折中，再看 Full–Current 的历史价值、Full–Flat 的来源价值；同时检查 Head/Middle、有效 A 更新及 91–100 轮是否仍保留收益。不要只看 Tail 上升。

本地检查只验证数学、真实 autograd 小模型和接线，不启动 CIFAR/CLIP 实验；真实效果和 GPU 实际耗时需要服务器训练结果。
