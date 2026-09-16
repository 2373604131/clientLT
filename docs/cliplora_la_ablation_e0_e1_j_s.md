# Client-LT 补充消融：E0、E1、J、S

## 固定实验定义

| 方法参数 | 损失（所有训练阶段） | 每轮正常阶段 | 正常聚合后的额外阶段 | 额外轮次 |
|---|---|---|---|---|
| `e0` | CE，τ=0 | 冻结 A，训练 B 三个 epoch | 冻结 A，训练 B 一个 epoch | 10、20、…、90 |
| `e1` | CE，τ=0 | 冻结 A，训练 B 三个 epoch | 冻结共同 B，训练 A 一个 epoch | 10、20、…、90 |
| `j` | 全局训练 LA，τ=1 | A/B 联合训练三个 epoch | 冻结共同 B，训练 A 一个 epoch | 10、20、…、90 |
| `s` | 全局训练 LA，τ=1 | 冻结 A，训练 B 三个 epoch | 冻结共同 B，训练 A 一个 epoch | 1、2、…、90 |

J 是预算对齐的联合训练对照，不是取消额外阶段的普通 AB 联训。
S 是高预算密集交替对照；第 91～100 轮只训练 B，第 100 轮不刷新 A。
这四组均不使用 E5 控制器、look-ahead、gap 权重、回退、PFRF 或特殊服务器聚合。

- CLIP ViT-B/16，视觉 top3 的 Q/V LoRA；rank=4、alpha=1、实际缩放 0.5、dropout=0、FP32。
- 30 客户端，每轮全部参加，共 100 轮；训练 batch size=32；种子与协议种子默认均为 42。
- 正常 B：SGD，lr=0.001、momentum=0.9、weight decay=0.0005，学习率恒定。
- J 的正常 A：同一个 SGD 优化器中的独立参数组，lr=0.001×`--a-lr-mult`、weight decay=0；正常 B 参数组不变。
- 额外阶段：新建 SGD，lr=0.001（A 乘 `--a-lr-mult`）、momentum=0.9、weight decay=0，一个 epoch，无 scheduler。
- 每个客户端从该阶段共同全局模型出发，优化器与动量重新初始化；不持久化客户端私有 A/B。
- 正常阶段平均参数，额外阶段平均增量后加回共同起点，均使用 `q_k=n_k/N`。J 分别平均 A/B，不采用有效矩阵 SVD。
- 额外 A 阶段使用本轮正常阶段聚合后的共同 B；LA 只进入客户端 CE 的 logits，不改变服务器权重，测试不添加 LA。
- 保留原有评估和额外阶段的 RNG 隔离。E0～E5 的训练分支没有改为新策略。

## 四个计算节点的前台命令

先将本次修改的代码同步到服务器，在仓库根目录激活 `clientlt` 环境。
每个节点已经只分配一张 GPU 时，继承调度器的 `CUDA_VISIBLE_DEVICES`，不用自行写 0/1。
每条命令单独占用一个节点，错误直接显示在终端；不使用 nohup 或后台启动。

节点一，E0：

```bash
python -u scripts/run_cliplora_la_control.py --method e0 --partition client-longtail
```

节点二，E1：

```bash
python -u scripts/run_cliplora_la_control.py --method e1 --partition client-longtail
```

节点三，J：

```bash
python -u scripts/run_cliplora_la_control.py --method j --partition client-longtail
```

节点四，S：

```bash
python -u scripts/run_cliplora_la_control.py --method s --partition client-longtail
```

默认数据目录为 `DATA`。继续复用原桥接实验：

```text
output/cifar100_LT/a_refresh_topology_bridge/seed42/client-longtail/c1/
```

该目录需要 `partition_manifest.csv`、`bridge_metadata.json` 和完整 `protocol/`。
数据/桥接根目录不在默认位置时，分别添加 `--data-root`、`--bridge-root`；不会重新生成划分。

输出分别写入：

```text
output/cifar100_LT/la_control/seed42/client-longtail/e0/tau0_a1_protocol42/
output/cifar100_LT/la_control/seed42/client-longtail/e1/tau0_a1_protocol42/
output/cifar100_LT/la_control/seed42/client-longtail/j/tau1_a1_protocol42/
output/cifar100_LT/la_control/seed42/client-longtail/s/tau1_a1_protocol42/
```

不会覆盖已有 E2/E3。目标目录已非空时沿用原入口的报错规则；另选 `--output-root`，不要删除已有实验。

## 预算口径

全客户端一个 epoch 为 352 个 optimizer steps。

| 方法 | 正常步数 | 额外步数 | 总 optimizer steps | A 活跃步数 | B 活跃步数 | A 刷新事件 |
|---|---:|---:|---:|---:|---:|---:|
| E0 | 105600 | 3168 | 108768 | 0 | 108768 | 0 |
| E1 | 105600 | 3168 | 108768 | 3168 | 105600 | 9 |
| J | 105600 | 3168 | 108768 | 108768 | 105600 | 9 |
| S | 105600 | 31680 | 137280 | 31680 | 105600 | 90 |

J 的一次联合 optimizer step 同时计入 A、B 活跃步数，但总 optimizer steps 只计一次。
J 与 E3 的 optimizer steps 相同，不表示精确相同 FLOPs。S 不能宣称同预算的纯频率消融。
`accepted_A_decisions=0` 表示没有控制器决策，不表示没有训练 A；查看 `a_refresh_events` 和 `normal_ab_events`。

## 汇总与分析

将四个节点结果和已有 E2/E3 放在同一 `la_control` 根目录，再运行：

```bash
python -u scripts/run_cliplora_la_control.py --stage summary
```

汇总包含 Overall、Non-tail、Head20、Middle60、Tail20、实际 A 刷新次数、AB 阶段数及预算。
Head20 是训练计数最高的 20 类；Middle60 为中间 60 类。Non-tail 仍然是除 Tail20 外的 80 类。
旧 E2/E3 没有新增分组列时，从它们原有的逐类结果计算，不重测模型。

- `analysis/comparisons.csv`：E2−E0、E3−E1、E1−E0、E3−E2，以及 J/S 对照。
- `analysis/la_refresh_interactions.csv`：四组齐全时，计算 `(E3−E2)−(E1−E0)`。
- `analysis/costs.csv`：区分总步数、A/B 活跃步数、额外预算、通信字节和阶段耗时。
- `analysis/protocol_audit.json`：划分、初始化、日程、状态重建和代码哈希核对。

本次代码扩展会改变代码文件哈希。新实验与旧 E2/E3 的数值比较会显式标记为跨代码版本描述性比较；不会假装通过同代码哈希核对。四个新实验使用同一次同步的代码。
已有 E2/E3 的训练分支保持原策略，但未运行数值复现来证明逐位一致。

训练不会自动启动 GPU 归因。需要时可另行运行（将 `j` 换成对应方法）：

```bash
python -u scripts/run_cliplora_la_control.py --stage analyze --method j --partition client-longtail
```

离线归因识别 J 的联合 A/B 参数路径和 S 的全部 90 次 A 刷新。S 归因成本也更高。

打包轻量分析文件，保留服务器原始状态：

```bash
tar --exclude='*/checkpoints' --exclude='*/events/*/state.pt' -czf la_control_ablation_analysis.tar.gz -C output/cifar100_LT la_control
```

该包不含大模型和事件 tensor；需要补做离线归因时仍要保留服务器上的 `events/`、`checkpoints/`。
本次实现只进行静态语法与差异检查，不启动训练、冒烟测试或 GPU 归因。
