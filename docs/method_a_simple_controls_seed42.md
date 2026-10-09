# 方法 A：seed42 简单对照运行说明

本轮只运行 seed42，划分与协议种子也固定为 42。新增四次 100 轮训练：`tailrw-g1`、`tailrw-g4`、`tailrw-g16`、`cover-cp`。已有同协议 `s`、`full-cp` 优先复用，不自动追加其他种子或重跑参照。

完整公式与判定见 [实验设计](方法A_简单加权与覆盖数保持对照实验设计_20261009.md)。当前完成了实现、CPU 测试和文件/协议预检，尚无新增完整训练结果。当前本地 Python 为 `torch 2.11.0+cpu`；下面的正式训练命令用于已有 CLIP/CUDA 训练环境。

## 1. 两个对照实际改变什么

- TailRW：使用 `(n_j + gamma*n_j_tail)/(N + gamma*N_tail)`，修改第 1—100 轮 B 及第 1—90 轮普通 A 聚合；不建立功能见证、历史或功能修正。原 LA prior 和样本量权重 `q` 保留。
- Cover-CP：先按完整 A 构造目标及 active，再将 active token 的权重改为类别持有客户端数的倒数并全局归一化。历史单独激活的 token 也使用该规则。来源测量、当前目标、历史、CP 与三步修正保留。

TailRW 的 gamma 固定由组名确定，不通过底层 `--gamma` 修改；后者是原训练参数，保持 1。Full/Cover 固定 λ=10、μ=1。执行模式固定 fast-v2、反馈批量 128、设备缓存 4 GiB、FP32。

## 2. 同步与准备

在服务器项目根目录执行。同步以下文件，并保持原训练核心源码与已有 AB 实验版本一致：

- `federated_main.py`
- `scripts/run_method_a_simple_controls.py`
- `scripts/run_method_a_simple_controls_parallel.py`（四组分别使用不同 GPU 的并行入口）
- `utils/method_a_simple_controls.py`
- `tools/sfra/simple_controls.py`
- `tools/sfra/simple_controls_summary.py`
- `tests/test_method_a_simple_controls.py`（本地检查用）

沿用 `references/full10_clientlt` 的五个协议文件及现有 `DATA/cifar-100/cifar-100-python/{train,test,meta}`，无需复制参考模型检查点。新运行直接使用训练环境中原有的 CLIP 权重及依赖。

运行时在服务器生成计划。不要把本地 `tmp/` 下的预检计划复制成服务器正式计划；登记文件包含运行机器上的绝对路径。

```bash
python scripts/run_method_a_simple_controls.py --stage plan
python scripts/run_method_a_simple_controls.py --stage preflight
```

默认输出根目录为 `output/cifar100_LT/method_a_simple_controls_v1`。预检核对文件、实际样本分配、日程和源码哈希，不加载 CLIP、不验证 CUDA，也不运行训练。

入口依次查找以下根目录中的旧 seed42 S/A：

1. `output/cifar100_LT/ab_decision_42_0_3407`
2. `output/ab_decision_42_0_3407_analysis/ab_decision_42_0_3407`

目录内相对路径分别为：

```text
runs/s/seed42/client-longtail/s/baseline_protocol42_fast_v2_f128_c4
runs/a/seed42/client-longtail/full-cp/lambda10_mu1_protocol42_fast_v2_f128_c4
```

也可以在首次登记时传 `--baseline s=/absolute/path/to/S --baseline full-cp=/absolute/path/to/A`。计划打印已登记参照路径；找不到时明确显示 `not reused`，不会偷偷重跑。缺少参照时仍可训练四个新组，但对应配对结果保持 pending。

若确需重跑 S/A，首次生成该输出目录的计划时统一传 `--methods s full-cp tailrw-g1 tailrw-g4 tailrw-g16 cover-cp`，总计六次 seed42 训练。一个组不能既登记为复用又登记为新训练；已有错误计划应保留并改用新的 `--output-root`。

## 3. 训练、恢复与收集

四个新组各占一张 GPU，同时运行。默认分配如下：

| 实验 | GPU | seed |
|---|---:|---:|
| `tailrw-g1` | 0 | 42 |
| `tailrw-g4` | 1 | 42 |
| `tailrw-g16` | 2 | 42 |
| `cover-cp` | 3 | 42 |

```bash
python scripts/run_method_a_simple_controls_parallel.py --stage train --seed 42 --gpus 0 1 2 3
```

`--gpus` 可省略，默认就是 0、1、2、3；也可换成服务器上四张空闲卡的编号，例如 `--gpus 2 3 4 5`。入口要求每组使用不同 GPU，编号按照 `--methods` 顺序一一对应。

入口先预检全部四组，再启动四个独立进程。分别设置 `CUDA_VISIBLE_DEVICES=0`、`1`、`2`、`3`，覆盖启动器继承的该环境变量；每个子进程内部只看到自己分配的卡，因此进程内部设备名称均为 `cuda:0`，但实际使用的是四张不同卡。

日志写入输出根目录下 `launcher_logs/gpu<编号>_<method>.log`，输出与检查点仍按方法隔离。某组失败会报告对应名称，不会自动改参数或重试；Ctrl+C 会停止本入口启动的训练进程。独立启动入口不修改已冻结的训练源码哈希，已有计划和恢复记录可以沿用。

重启四组并行任务：

```bash
python scripts/run_method_a_simple_controls_parallel.py --stage train --seed 42 --gpus 0 1 2 3 --resume
```

单独重启 Cover-CP：

```bash
python scripts/run_method_a_simple_controls_parallel.py --stage train --seed 42 --methods cover-cp --gpus 3 --resume
```

也可以在四个终端分别执行原入口，效果相同：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_method_a_simple_controls.py --stage train --seed 42 --methods tailrw-g1
CUDA_VISIBLE_DEVICES=1 python scripts/run_method_a_simple_controls.py --stage train --seed 42 --methods tailrw-g4
CUDA_VISIBLE_DEVICES=2 python scripts/run_method_a_simple_controls.py --stage train --seed 42 --methods tailrw-g16
CUDA_VISIBLE_DEVICES=3 python scripts/run_method_a_simple_controls.py --stage train --seed 42 --methods cover-cp
```

前一版 `run_method_a_simple_controls_gpu0.py` 已改为新并行入口的兼容转发；服务器应同步新脚本，并优先使用上面的明确命令。

已完成且通过完整性检查的组自动跳过；中断组恢复 `checkpoints/sfra_last.pt`。尚未产生轮次检查点的初始化失败不支持恢复，需使用新输出目录。恢复核对原任务、原命令、源码和配置，不能中途换 gamma、分区或 workers。

训练结束后：

```bash
python scripts/run_method_a_simple_controls.py --stage summary
python scripts/run_method_a_simple_controls.py --stage pack
```

如果参照结果稍后才取回，可在汇总命令中用 `--baseline` 补充尚未登记的参照。源码和配置不匹配的结果不会进入配对差值。打包产物为输出根目录旁的 `method_a_simple_controls_v1_results.tar.gz`；包括已登记参照、逐样本预测、控制记录、汇总和新增源码，不包括训练检查点或大体积 event 状态，不用于断点续训。

路径或 workers 要调整时，在计划、预检和训练命令中保持 `--reference-run`、`--data-root`、`--output-root`、`--num-workers` 一致。仅汇总/打包通常只需相同的 `--output-root`。

## 4. 输出与解释

每个新组保存在 `runs/seed42/<method>`：

- `aggregation_weights/rNNN_B.csv` 和 `rNNN_A.csv`：实际聚合权重与原样本量权重；应为 100 个 B 文件和 90 个 A 文件。
- `functional_priority/rNNN.csv`：Full/Cover 的 coverage、原来源 rho、实际 rho、归一化权重、active 与 history-only 标记；应为第 1—90 轮。
- `predictions/r000.npz` 至 `r100.npz`：固定测试顺序的样本 ID、标签、预测和正确性。评估不更新训练模型、历史或训练随机状态。
- `client_weights.csv`、`class_coverage.csv`、原训练日志、成本、检查点、完成标记和源码登记。

`analysis/` 汇总输出：

| 文件 | 内容 |
|---|---|
| `report.md` | 各组状态与末 20 轮主要结果 |
| `status.csv`、`pair_audit.csv` | 缺失/未完成/非法结果、配对资格及原因 |
| `performance.csv`、`paired_per_seed.csv` | 第 81—100 轮均值、最终轮、回落及 seed42 配对差值 |
| `curves.csv` | 第 0—100 轮正式提交模型指标 |
| `sample_transitions.csv` | 相邻轮次错/对转换及共同初始化队列的逐轮保持和学会数量 |
| `sample_retention.csv` | 初始化正确/错误队列在末 20 轮的保持率、学会率和平均数量 |
| `costs.csv` | 正常优化、功能修正、前后向、模拟通信量、评估时间和可用显存记录 |
| `aggregation_weights.csv`、`functional_priority.csv` | 各组实际执行的控制记录 |

主指标为提交模型第 81—100 轮 Tail20 平均准确率，必须同时看 Overall、Head20、Middle60、Non-tail 与成本。三档 TailRW 全部报告，不按测试结果挑最优 gamma。这里只含一个开发种子，不计算跨种子标准差或显著性。

复用 S/A 若没有逐样本预测或显存记录，相应辅助指标留缺失；不能凭每类准确率恢复样本遗忘。Cover-CP 保留来源测量，不能将其解释为消除了该成本。不同设备的墙钟时间不直接用于宣称速度倍数。

## 5. 本地验证

```bash
python -m unittest discover -s tests -p test_method_a_simple_controls.py -v
python -m unittest discover -s tests -p test_method_a_parallel_launcher.py -v
python -m unittest discover -s tests -p "test_sfra*.py"
```

简单对照现有 16 项检查通过，包括 4 项探针清单换行恢复/拒绝篡改检查；之前的 SFRA 回归 161 项中 160 项通过、1 项 CUDA 检查跳过。检查覆盖 gamma=0 退化、两个因子的实际聚合、覆盖数及历史单独激活、共同输入上的真实三步 CP 修正、状态与 RNG 恢复、源码冻结和恢复命令、日志异常拒绝、逐样本统计及打包。另已用现有 seed42 S/A 归档验证配对资格。CPU 检查不代表完整 GPU 训练已完成或方法收益已成立。

## 6. 初始化报错 `probe_manifest_sha256` 的修复

该校验比较 `protocol/probe_manifest.csv` 的文件字节哈希与参考 `bridge_metadata.json` 的记录。当前参考记录的 SHA256 是 `8a43f25c87036442c611f69fb3dedc592ad96bbbf64b16fe2883b63a2ce1fc43`，对应 CRLF 换行。把同一份清单转换为 LF 后，哈希变为 `0bfa0a2da47623aa8ab5151f7c8522ee3a4c5425328df9bca92aea1873a1592e`。Git 检出或文本传输可能触发这种变化，即使 CSV 的样本行没有改变。

修复只需同步这两个更新文件：

- `scripts/run_method_a_simple_controls.py`
- `tools/sfra/simple_controls.py`

现在计划/预检阶段就检查清单能否匹配登记哈希。新运行准备协议时，只在 LF/CRLF 换行转换后精确匹配原哈希的情况下恢复运行目录内的副本，并保存 `probe_manifest_replay.json`。共享参考、登记哈希、样本顺序和训练算法均保持原样。若内容确实不匹配，会在加载模型前报错；不要跳过原运行时断言或用当前错误哈希覆盖参考记录。

这次错误发生在轮次检查点创建前，因此无需 `--resume`。修复更新了已冻结清单中的启动源码，使用新输出根目录，避免与旧计划冲突。确认旧启动进程已退出、同步两份文件后，在四个终端分别执行：

```bash
CUDA_VISIBLE_DEVICES=0 python scripts/run_method_a_simple_controls.py --stage train --seed 42 --methods tailrw-g1 --output-root output/cifar100_LT/method_a_simple_controls_v1_probe_fix
```

```bash
CUDA_VISIBLE_DEVICES=1 python scripts/run_method_a_simple_controls.py --stage train --seed 42 --methods tailrw-g4 --output-root output/cifar100_LT/method_a_simple_controls_v1_probe_fix
```

```bash
CUDA_VISIBLE_DEVICES=2 python scripts/run_method_a_simple_controls.py --stage train --seed 42 --methods tailrw-g16 --output-root output/cifar100_LT/method_a_simple_controls_v1_probe_fix
```

```bash
CUDA_VISIBLE_DEVICES=3 python scripts/run_method_a_simple_controls.py --stage train --seed 42 --methods cover-cp --output-root output/cifar100_LT/method_a_simple_controls_v1_probe_fix
```

原失败目录保留，已有 S/A 仍按原规则复用。之后恢复、汇总和打包都使用同一个新输出根目录。例如训练完成后的汇总：

```bash
python scripts/run_method_a_simple_controls.py --stage summary --output-root output/cifar100_LT/method_a_simple_controls_v1_probe_fix
```
