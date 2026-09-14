CAPT cluster：每客户端从全局模型开始的对照

新增参数 `--capt_reset_global_before_client True`，默认 `False`，仅允许与 `--model cluster --trainer CAPT` 一起使用。

启用后，每个客户端训练前严格加载完整 `global_weights`，包括共享提示、类别提示、视觉耦合层以及 state_dict 中的缓冲区。第一轮也从服务器初始化开始。相似/互补聚类、类别本地占比 >0.1 的聚合、CAPT 损失和视觉耦合逻辑均保留。建议实验名称用 `CAPT-cluster-global-start`，结果产生前不要把这个变体称为已验证的 SOTA。

本参数只控制模型起点，不控制优化器和通信频率。原始优化器状态仍复用；已有 `--capt_matched_v2 True` 可独立启用每客户端重置优化器与调度器。原始 MAB 仍由 `--capt_fixed_global_agg_freq 0` 保留。如果 MAB 跳过本轮聚合，下一轮每客户端仍从最近一次服务器模型开始，未聚合的本地结果不会成为下一轮起点。若希望每轮都更新服务器，使用已有的 `--capt_fixed_global_agg_freq 1`；该模式同时不再用官方测试反馈控制后续调度。

仓库根目录中的单卡启动命令（Linux 和 PowerShell 均可）：

```bash
python scripts/run_capt_global_start.py --gpu 0 --data-root DATA/
```

默认顺序运行六个任务：三种子 `1 42 2026` × 两种划分 `client-longtail`、`noniid-labeldir-fine`。默认 CIFAR100-LT、ViT-B/16、IF=.01、30 客户端、全参与、100 轮、本地 3 epochs、batch=32、学习率 .001、Client-LT λ=.75、组内 α=.5、Dirichlet β=.5。与现有 `scripts/capt.sh` 的主配置对应。每个种子两种划分共用相同客户端日程。

默认启动器只增加全局模型恢复；MAB 保留，优化器不重置。

先查看六条命令，不训练且不创建文件：

```bash
python scripts/run_capt_global_start.py --dry-run
```

只做一个种子的双划分试跑：

```bash
python scripts/run_capt_global_start.py --gpu 0 --seeds 42 --rounds 3 --local-epochs 1 --num-workers 0 --output-root output/cifar100_LT/capt_global_start_smoke
```

两张 GPU 分开启动，在两个终端分别运行：

```bash
python scripts/run_capt_global_start.py --gpu 0 --partitions client-longtail
```

```bash
python scripts/run_capt_global_start.py --gpu 1 --partitions noniid-labeldir-fine
```

GPU 参数按继承的 `CUDA_VISIBLE_DEVICES` 解释为可见 GPU 的索引；未设置可见设备列表时，直接使用所给设备编号。

λ=1 的双划分实验使用新输出目录：

```bash
python scripts/run_capt_global_start.py --gpu 0 --specialization-lambda 1 --output-root output/cifar100_LT/capt_global_start_lambda1
```

λ 不改变 fine-Dirichlet 的定义，所以该命令会额外重跑一组相同设定的 fine-Dirichlet。若已有其对照，只需加 `--partitions client-longtail`。

如果需要训练过程不受官方测试调度、且优化器不跨客户端继承的正式对照，显式启用两个额外设置，放到单独目录。它比默认版本多改变了优化器状态和调度，不能混作“只改模型起点”的单因素消融：

```bash
python scripts/run_capt_global_start.py --gpu 0 --fixed-global-agg-freq 1 --reset-optimizer --output-root output/cifar100_LT/capt_global_start_fixed_isolated
```

如果要复用 Stage-1 的规模和划分类型：

```bash
python scripts/run_capt_global_start.py --gpu 0 --seeds 42 --partitions client-longtail matched-dirichlet --num-users 30 --frac 0.4 --rounds 80 --local-epochs 3 --fixed-global-agg-freq 1 --output-root output/cifar100_LT/capt_global_start_stage1
```

这个命令按相同 seed 生成日程；要宣称与历史实验严格对齐，仍需对照实际 `selected_clients.csv` 和客户端计数矩阵。`matched-dirichlet` 固定 Client-LT 的双边际，`noniid-labeldir-fine` 是逐细类 Dirichlet，历史 `noniid-labeldir` 是另一种实现，三者不要混用名称。启动器也支持显式选择历史 `noniid-labeldir`。

默认输出结构：

```text
output/cifar100_LT/capt_cluster_global_start/
  schedules/
  seed1/
    client-longtail/
    noniid-labeldir-fine/
  seed42/...
  seed2026/...
```

每次运行记录 `command.json`（命令、协议开关、源代码哈希与日程哈希）、`run.log`，训练正常退出后写 `finished.json`。训练器还记录真实 `selected_clients.csv` 和原有指标文件。启动器拒绝写入已有非空运行目录；更换配置或重跑请指定新的 `--output-root`，不会覆盖历史结果。

`finished.json` 只表示训练进程正常退出，不表示 MAB 必然在最后一轮聚合。比较默认 MAB 版本应检查实际评测轮次，不能把 CSV 最后一行自动叫作第100轮成绩。指标中的 method 字段仍为 CAPT，区分变体应依据输出目录和 command.json。

要在自己的原脚本中使用，只需保持 `MODEL=cluster`、`TRAINER=CAPT`，并在配置尾部 `DATALOADER.NUM_WORKERS ...` 之前添加：

```bash
--capt_reset_global_before_client True
```

验证：`tests/test_capt_global_start.py` 从真实入口提取 CAPT 客户端循环，用 CPU 模型检查完整模型恢复、第一轮起点、未聚合轮次、更新后的服务器广播、关闭参数时的原始串行行为，以及独立的优化器重置开关。还检查参数解析、双划分共享日程、dry-run 不写文件和历史结果保护。未在本机执行完整 GPU 训练。
