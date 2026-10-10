# 冻结 LoRA A 的16倍加权补充实验

如需同时补充严格匹配的FedAvg，并使用GPU2/GPU3、每卡最多六客户端并行，请使用[双实验启动说明](frozen_aggregation_controls_seed42.md)。当前代码已按用户要求取消串行/并行误差阈值检查；下文关于数值对比试跑的描述仅记录早期版本，以双实验v3说明为准。

本次只增加一个 seed42 的100轮正式实验：全程冻结初始 LoRA A，只训练 B。它补齐之前 `tailrw-g16` 没有覆盖的条件，不包含保持修正、来源 C 或额外 A/B 训练。

| 设置 | 本次 TailRW16 | 已完成的新聚合 frozen |
|---|---|---|
| A | 全程固定初始化 | 全程固定初始化 |
| B 本地训练 | 每轮每客户端3个epoch | 相同 |
| 训练轮数、参与客户端 | 100轮、每轮30个 | 相同 |
| 本地优化步数 | 105600 | 相同 |
| 损失、LA | 原损失，原始全局计数先验，tau=1 | 相同 |
| 更新 A / 保持修正 / 来源 C | 全部关闭 | 全部关闭 |
| 客户端并行 | 单GPU最多4个，剩余1—3个也可运行 | 相同 |
| 聚合 | 原 TailRW gamma=16 公式 | 类别分布凸优化，lambda=0.1 |

权重严格沿用历史实现：

\[
w_j=\frac{n_j+16n_{j,\mathrm{tail}}}{N+16N_{\mathrm{tail}}}.
\]

这里的“16倍”是历史实验 gamma=16 的名称，包含原本的 n_j 基础项；不是把持有尾类的客户端整份权重统一乘16，也不修改本地损失。Tail20由训练计数确定。

**启动**

先把本次代码变更同步到服务器仓库根目录 `/data/yzh/clientLT`。本地实现没有自动推送到远端。若通过提供的补丁压缩包同步，在仓库根目录解压；它只含本次代码、测试和说明，不含数据或实验结果。

配置检查，不启动训练：

```bash
conda activate clientLT
python scripts/run_cliplora_joint_aggregation.py --stage preflight --aggregation-rule tailrw16
```

直接在服务器GPU2上完成一次 smoke，再自动运行完整实验：

```bash
python scripts/run_cliplora_joint_aggregation.py --stage server --aggregation-rule tailrw16 --gpus 2 3 --client-concurrency 4
```

TailRW16模式只允许 `frozen` 分支，GPU3不会启动任务。`server` 不是单纯检查：其smoke成功后会继续跑100轮正式实验。默认独立输出目录为：

```text
output/cifar100_LT/frozen_tailrw16_v1
```

如果只想做GPU smoke，不启动正式训练：

```bash
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=2 python scripts/run_cliplora_joint_aggregation.py --stage smoke --aggregation-rule tailrw16
```

中断后重新执行同一条server命令即可从最后提交轮次恢复，已完成并通过审计的smoke或正式结果会跳过。若启动失败，查看 `launcher_logs/frozen_server.log` 与 `launcher_logs/frozen_formal.log`（smoke为 `frozen_smoke.log`）。不得更换训练代码后直接续跑旧注册计划；如需修复应保留旧目录并使用新的 `--output-root`。

数据和参考协议默认仍是 `DATA` 与 `references/full10_clientlt`。如果原参考目录已迁移，也可以显式使用已完成的新聚合 frozen 运行目录作为 `--reference-run`，但必须在首次注册时选定，之后继续使用相同参数。

**收集结果**

```bash
python scripts/run_cliplora_joint_aggregation.py --stage status --aggregation-rule tailrw16
python scripts/run_cliplora_joint_aggregation.py --stage pack --aggregation-rule tailrw16
```

状态应为 `frozen complete round=100`。打包得到：

```text
output/cifar100_LT/frozen_tailrw16_v1_results.tar.gz
```

报告写入 `analysis/report.md`。默认只读对照目录是输出目录同级的 `client_aggregation_v2_parallel4`。若位置不同，在status/summary/pack命令中增加：

```bash
--compare-root /实际路径/client_aggregation_v2_parallel4
```

输出包含最后20轮 Overall/Head20/Middle60/Tail20、最终尾类及后期回落、逐样本新增学习/保持/遗忘、同状态FedAvg反事实、运行成本、并行试跑以及与新聚合冻结组的配对审计。比较不会改写旧实验结果。不自动混入训练过A的旧16倍实验或额外训练过B的E2。

权重、每客户端步数、实际聚合记录、每轮A哈希、所有预测及保存准确率均被核验。只有配置和CPU测试通过不代表GPU实验已经通过；服务器smoke会实际检查数值一致性、六客户端执行和最后两个客户端的短组。

**新聚合后续优化的依据**

从当前训练划分得到：

| 聚合 | 尾类集中客户端27—29的总权重 | 最小的 w_j / FedAvg权重 |
|---|---:|---:|
| FedAvg | 1.1985% | 1 |
| 当前新聚合 | 11.4629% | 接近0（客户端5、6） |
| 历史 gamma16 公式 | 15.2990% | 0.81587 |

16倍规则的基础保留比例为 N/(N+16N_tail)=0.81587；也就是说每个客户端至少保留约81.6%的原FedAvg权重，再给尾类计数额外份额。新聚合虽然给三个尾类集中客户端的总权重更低，却会更大幅度地重新分配其他客户端之间的权重。因此，不能仅用“尾类权重太高”解释当前结果。

优先考虑三个方向，但本补充实验不同时修改这些设置：

1. **保留各客户端的基础贡献。** 给 w_j 增加相对 q_j 的下界，或将新权重与FedAvg按固定比例混合，避免某些客户端贡献接近归零。它是针对权重过度重新分配的候选修正，尚不保证准确率提高。
2. **让聚合目标反映实际更新。** 当前目标使用类别比例 R_jc=n_jc/n_j，未直接描述实际有效更新 ΔW_j=sΔB_j A。该划分的三个尾类集中客户端每轮只有6个本地优化步，其他客户端为27—60步；类别比例相同的权重并不代表相同的更新幅度或作用。应先记录加权后的有效范数及训练侧类别响应，区分“更新太小”和“方向互相抵消”，再决定是否按步数或幅度校正。不能直接把每个客户端范数强行拉平。
3. **让目标与当前学习需求匹配。** 固定计数重平衡不知道哪些类别已经学够、哪些还在退步。若前两项不能解决问题，再考虑低频训练侧反馈更新权重，而非每轮增加昂贵反馈；不能根据测试准确率调权。

冻结A本身也限定了可用更新空间；聚合只能重组客户端已产生的B更新，不能保证产生新的方向。当前证据尚不能判断主要瓶颈是空间限制还是权重不合适。补充TailRW16有助于区分它们：若简单调权在同一冻结A条件下明显改善，说明原新聚合仍有可改空间；若两种调权都没有收益，也只是一条线索，不能据此证明聚合普遍无效或必须学习A。
