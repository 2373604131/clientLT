# V2 与 CAPT 对齐实验

运行三组 V2（依次执行）：

```bash
python scripts/run_cliplora_v2.py --method v2 --data-root DATA
```

单独运行渐进版或 CAPT：

```bash
python scripts/run_cliplora_v2.py --method progressive --data-root DATA
python scripts/run_cliplora_v2.py --method capt --data-root DATA
```

`--method all` 依次运行三组 V2 和 CAPT。可通过 `--seed`、`--output-root`、`--num-workers` 修改种子、输出目录及加载线程数。重复实验使用新的输出目录，避免已有日志追加。

共同配置：CIFAR100-LT、Client-LT、IF=100、30 客户端、全参与、100 轮、本地 3 epoch、训练 batch 32、测试 batch 64、FP32、SGD、固定 lr=0.001、无 warmup；同一 seed 使用同一数据划分种子和客户端参与顺序。每次本地训练重建优化器。

V2 固定共享 A0，仅训练视觉 top3 q/v LoRA B（rank=2、alpha=1），正常接收全局模型，普通 CE。LA 求解器在训练前以 tau=1、regularization=0、max_iter=100 计算一次目标 p。三组分别使用 q、0.12q+0.88p、(1-beta)q+beta*p，其中 beta=clip((round-5)/15,0,1)，round 从 1 开始。服务器直接聚合 B，不执行 SVD。100 轮静态和渐进组累计客户端权重相同。

CAPT 使用仓库现有 cluster 路径，保留提示、损失、耦合层及客户端协作机制；固定每轮全局聚合（现有 capt_fixed_global_agg_freq=1），不使用测试精度驱动的 MAB 调度。这是预算对齐的 CAPT 对照，不等同于论文原始配置或当前全部 SOTA。CAPT 与 LoRA 的可训练结构不同，不能把模型参数量也称为完全一致。

输出位于 `output/cifar100_LT/v2_matched/seed42/{fedavg,static,progressive,capt}`：`round_metrics.csv` 包含每轮 Overall、Non-tail、Tail；V2 的 `v2_target_weights.json` 保存 p 和求解结果，`lora_aggregation_weights.csv` 保存实际逐轮权重，`lora_aggregation_summary.csv` 保存 beta。`command.json` 保存完整启动参数。

比较第 100 轮和第 81–100 轮均值，以相同种子的 CAPT 减 V2 报告精度差（百分点）。V2 初始精度使用 epoch=-1，训练第 100 轮对应 epoch=99。本次仅实现正常训练代码，未运行训练或冒烟测试。
