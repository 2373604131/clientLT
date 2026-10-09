# 方法 A：学习机会与开放间隔实验，seed42 六卡运行

入口：`scripts/run_cliplora_a_learning_schedule.py`。

本轮使用 **E2、E3 各第 20、50、80 轮，共六个起点**。每个起点执行四组十轮短分支，总计 24 条。默认只运行 seed42，不会自动补基线或其他种子。

## 固定实验协议

起点取原始事件文件的 `anchor_lora_state`：该轮普通 B 训练完成、额外更新尚未开始。不是 `actual_after_lora_state`。

| 组别 | 第一次额外更新 | 第二次额外更新 | 后续普通训练 |
|---|---|---|---|
| BB | h=0，B，1 epoch | h=5，B，1 epoch | h=1…10，每轮 B，3 epochs |
| AB | h=0，A，1 epoch | h=5，B，1 epoch | 同上 |
| AA_short | h=0，A，1 epoch | h=1，A，1 epoch | 同上 |
| AA_long | h=0，A，1 epoch | h=5，A，1 epoch | 同上 |

- h=1、5 的额外更新在当轮普通 B 更新之后执行。来源轨迹原来的 A 刷新日程不再自动执行。
- 主指标固定为 h=8、9、10 的平均值。h<5 时 AA 两组尚未完成相同数量的额外更新，不能把这一阶段的领先当成匹配预算下的优势。
- 普通 B：SGD，lr=.001、momentum=.9、weight_decay=.0005，每客户端重新初始化优化器。
- 额外 A/B：SGD，lr=.001、momentum=.9、weight_decay=0，每客户端重新初始化优化器。
- 所有训练使用 LA，tau=1；测试使用原始 logits。30 客户端全参与、batch=32、drop_last=False。沿用原 CIFAR 包装器和数据划分。
- 配对分支的随机流不包含组名、因子或 GPU 编号。两次第二剂额外更新即使发生在 h=1、5，也使用相同的样本随机流和客户端顺序。
- 仅缓存冻结的文本特征，不启用另一套低秩 forward 实现。普通阶段沿用 FP32 参数平均；额外阶段沿用 FP32 增量平均。
- 每个起点的四组有 48 个逻辑训练阶段，复用共同前缀后实际训练 40 个：34 次普通 B、6 次额外更新。各组逻辑训练预算仍分别为 32 个全客户端 epoch。
- 额外更新阶段记录客户端本地反馈和聚合后的同样本反馈。反馈样本固定为每个客户端—类别前至多 8 个训练样本；这些是训练反馈，不能称为未见过的验证集。
- 保存官方测试集逐样本预测、CE/LA 损失和归一化预测间隔，统计相对共同起点的新答对、遗忘和保持正确样本数。
- h=0 和 h=5 的同状态 A/B 候选另外做即时有效范数匹配，目标为两者的较小非零范数。这些只是即时探针，不属于十轮主轨迹，也不用于训练选择。

## 先把新代码同步到服务器仓库

需要同步：

- `scripts/run_cliplora_a_learning_schedule.py`
- `tools/a_learning_schedule/` 下全部 Python 文件

运行时复用仓库已有 `trainers/cliplora.py`、CIFAR 数据包装器、`utils/cliplora_a_refresh.py`、`utils/sfra_execution.py` 和锁工具。服务器仓库应与当前工作区相应代码一致。

在已分配 GPU 的计算节点上：

```bash
cd ~/run/yzh/code/clientLT
conda activate clientlt
python --version
```

使用 Python 3.10 及以上版本，建议继续使用原训练的 Python 3.10 / PyTorch 环境。不要在登录节点用系统 Python 启动 GPU 训练。

## 1. 文件预检

默认查找：

```text
output/cifar100_LT/la_control/seed42/client-longtail/e2/tau1_a1_protocol42/
output/cifar100_LT/la_control/seed42/client-longtail/e3/tau1_a1_protocol42/
```

每个来源必须有 `bridge_metadata.json`、`control_config.json`、`partition_manifest.csv`、`checkpoints/base_model.pt`，以及以下三个完整事件权重：

```text
E2: events/r020_c000_main_extra_B/state.pt
    events/r050_c000_main_extra_B/state.pt
    events/r080_c000_main_extra_B/state.pt
E3: events/r020_c000_main_refresh_A/state.pt
    events/r050_c000_main_refresh_A/state.pt
    events/r080_c000_main_refresh_A/state.pt
```

轻量分析包不包含这些权重，不能拿来启动实验。

```bash
python scripts/run_cliplora_a_learning_schedule.py --stage preflight
```

来源目录改过位置时，在预检、冒烟和正式运行命令中一致追加：

```bash
--e2-run /absolute/path/to/e2/tau1_a1_protocol42 --e3-run /absolute/path/to/e3/tau1_a1_protocol42
```

只有大目录不同，可以用 `--source-root /absolute/path/to/la_control`。数据不在 `DATA` 时使用 `--data-root`。

预检通过只表示文件和配置可用，不表示 GPU 训练已验证。

## 2. 六卡冒烟

```bash
python scripts/run_cliplora_a_learning_schedule.py --stage smoke --gpus 0 1 2 3 4 5
```

每张卡加载一个起点，使用两张训练反馈图片分别执行一次 A/B forward/backward，检查非零有效更新和冻结因子，再恢复起点。不会写入正式分支，也不会把测试图片用于训练。

冒烟完成后检查命令退出码；任何失败先看对应日志。

## 3. 六卡正式运行

```bash
python scripts/run_cliplora_a_learning_schedule.py --stage run --gpus 0 1 2 3 4 5
```

默认分配：

| GPU | 工作 |
|---:|---|
| 0 | E2，第 20 轮，四组 |
| 1 | E2，第 50 轮，四组 |
| 2 | E2，第 80 轮，四组 |
| 3 | E3，第 20 轮，四组 |
| 4 | E3，第 50 轮，四组 |
| 5 | E3，第 80 轮，四组 |

`--gpus` 设置每个子进程的 `CUDA_VISIBLE_DEVICES`，会覆盖继承的环境变量。编号必须是这台计算节点上允许使用的设备。六张卡在不同节点时，使用下面的单起点方式，不能在一个节点上填写另一节点的 GPU 编号。

默认 DataLoader `num_workers=0`，避免每个客户端反复创建进程。可以在首次运行前统一选择 `--num-workers 2`，但它会成为这轮实验固定配置；续跑不要改变它。

运行器为前台进程，适合在已有调度作业或 tmux 中启动。关闭终端或中断启动器会停止它启动的子进程。阶段结果已保存，可以续跑。

## 单 GPU / 不同节点分配

例如某节点只给 GPU 0，负责 E3 第 50 轮：

```bash
python scripts/run_cliplora_a_learning_schedule.py --stage run --origins e3 --rounds 50 --gpus 0
```

分别将 `e2/e3` 与 `20/50/80` 组合分给六张卡即可。不同起点可以共享结果目录；相同起点有独占锁，禁止重复训练。同一共享目录需要使用一致的来源绝对路径和代码。

## 状态、续跑、汇总、回收

默认结果根目录：`output/cifar100_LT/a_learning_schedule_v1`。

```bash
python scripts/run_cliplora_a_learning_schedule.py --stage status
python scripts/run_cliplora_a_learning_schedule.py --stage summary
python scripts/run_cliplora_a_learning_schedule.py --stage collect
```

运行被打断后，重新执行原 `--stage run` 命令即可。它复用已经原子保存的阶段和评估结果，不会把半个客户端训练当成完成。代码、数据路径或运行配置变化时会拒绝混用缓存，应选择新的 `--output-root`。

六卡完整运行成功后自动汇总并生成 PNG/PDF 图。若服务器没有 matplotlib，CSV/报告仍正常生成，可以在分析机器上补图。

主要输出：

| 路径 | 内容 |
|---|---|
| `launcher_logs/seed42/e2_r020_run.log` 等 | 各卡训练日志 |
| `runs/seed42/e2/round020/curves.csv` 等 | 四组 h=0…10 的正式曲线 |
| 每个起点的 `evaluations/` | 每阶段测试、训练反馈逐样本 NPZ 和汇总 JSON |
| 每个起点的 `local_feedback.csv` | 同客户端—类别的更新前、本地更新后、聚合后反馈 |
| 每个起点的 `norm_probes.csv` | 两个同状态比较的即时等幅探针 |
| 每个起点的 `budget.csv`、`events.csv` | 实际执行成本、共享路径和模型状态身份 |
| `analysis/per_run.csv` | h=8…10 各组均值 |
| `analysis/paired.csv` | AB−BB、AA_long−AB、AA_long−AA_short |
| `analysis/report.md` | 固定指标报告 |
| `analysis/*_trajectories.png/pdf` | Overall/Tail 学习轨迹 |
| `analysis/*_learning_forgetting.png/pdf` | 尾类新增正确和遗忘样本数 |
| `analysis/paired_tradeoffs.png/pdf` | 配对 Overall/Tail 收益 |

`collect` 生成相邻的 `a_learning_schedule_v1_analysis.zip`，包含分析表、图、元数据和逐样本预测，不包含大模型及训练节点张量。回传这个压缩包即可做后续分析；服务器上的节点缓存保留以便续跑。

当前 seed42 的多个起点不作为独立训练种子统计。后续新增种子需要其本身的 E2/E3 来源轨迹；已有多种子结果可以使用 `--stage summary --summary-seeds 42 0 3407` 汇总，代码不会因此启动新训练。

## 验证范围

本地提供 `python -m unittest tests.test_a_learning_schedule -v`，用 CPU 小型 LoRA 模型覆盖完整 40 节点执行、共享前缀复用、冻结因子、随机流、续跑、固定主窗口和样本学习/遗忘计数。本地没有 CUDA 和完整服务器检查点，因此实际 CLIP/CIFAR GPU 执行必须在服务器通过 `--stage smoke` 验证。
