# 冻结 A+B 的最终验证代码与服务器运行说明

本入口验证已经选定的 A 与轮换采样共享 B（w=0.35）。默认只计划 A-only、A+B 的重复训练；辅助对照按需选择。所有训练都由用户在服务器显式启动。本地测试不代替真实 CLIP/CUDA 训练。

## 1. 固定什么，回答什么

| 命令中的组名 | 实际方法 | 要回答的问题 |
|---|---|---|
| `a` | Full-CP，λ=10、μ=1，关闭 B transfer | A-only 的配对参照 |
| `ab` | 同一个 A＋共享 donor C，非尾类轮换采样、w=0.35 | B 在固定 A 上是否稳定改善 Tail20？ |
| `a-calibration` | 同一个 A＋直接优化共享 B 残差，匹配 AB 反馈样本、两步优化及每步有效更新范数 | donor 更新重组是否优于这个额外分类校准对照？ |
| `s` | 同频普通 A 更新，没有 A 的功能修正和 B transfer | A 与 AB 相对同频基线有多少收益？ |
| `a-flat` | A 去掉来源优先级 | A 的来源优先级是否有贡献？ |
| `a-current` | A 去掉历史目标 | A 的历史目标是否有贡献？ |
| `a-no-cp` | A 去掉分类保持项 | A 的分类保持项是否有贡献？ |

优先完成 `a` 与 `ab`，再完成 `a-calibration`。最后四组是已有 A 验证的可选入口，不要求因为新增本脚本而全部重跑。已有 A 实验可以继续作为 A 的证据；旧结果不会自动冒充本套统一执行配置的配对重复。`s` 是同频 A 更新基线，不是普通联合训练 A/B 的 FedAvg+LoRA。

主验证默认训练种子为 42、43、44，协议种子固定为 42。相同训练种子内使用相同初始化、划分、调度和见证抽样；不同训练种子改变初始化、训练随机性和见证抽样。这样检验的是固定划分下的训练稳定性，不是多个独立划分上的稳定性。

seed42 已参与方法选择，单独标为 **development**；43、44 单独标为 **confirmation**。三个种子的合并统计也会保留，但不会把 seed42 重新解释为独立确认。两个新种子的结果仍只提供有限的稳定性证据。

固定配置：

- CIFAR-100-LT，30 客户端、全参与、100 轮、FP32、LoRA rank=4、视觉 top3 的 q/v。
- 每轮 B 三个本地 epoch；第 1–90 轮 A 一个本地 epoch。A 的三步修正、步长 0.1、历史窗口等沿用已冻结实现。
- B 在第 30、40、……、100 轮执行。每个 donor、每个模块保留独立的完整 r×r 矩阵 C；Adam lr=0.3、两步、正则 0.001、probe step=0.1。
- B 反馈来自划分中所有持有尾类的客户端，混合尾类与非尾类反馈；有两类反馈时权重为 0.35/0.65，仅含尾类时使用尾类均值。它不限定在协议标记的少数尾部客户端。
- 保留旧共享 B 的 donor 探测和取并集行为，不把此次验证写成“严格逐类定向筛选已有效”。也不顺便删除筛选或加入新版 response/direct B。
- 所有新组统一 `fast-execution-v2`、反馈 batch=128、cache=4 GiB；见证 microbatch=8。正式配对不能混用旧普通执行与 fast-v2。

## 2. 额外分类校准对照如何计算

冻结 AB 的共享残差为每个模块的 `R = mean_j(ΔB_j C_j)`，C 从零开始校准。本对照直接把同形状的 R 作为可学习参数，从零开始，固定当前 A 与普通聚合后的 B。

每一步使用 AB 对应的相同客户端和同一批本地样本位置；每个客户端计算 `0.35 × 尾类 LA 均值 + 0.65 × 非尾类 LA 均值`，仅尾类客户端使用尾类均值。所有反馈客户端等权累积梯度后，执行一次 Adam 更新。

随后对所有模块的 R 乘同一个标量，使

`sqrt(sum_module ||scaling × R_module × A_module||_F²)`

等于配对 AB 在该轮、该步记录的有效残差范数。匹配的是从普通聚合模型出发的累计残差，不是相邻两步的距离。第二步后提交一次共享残差，随后继续原 A 流程。没有 C，因此不施加 C 正则；范数匹配控制更新幅度。Adam 动量在两步之间保留，每个迁移事件重新初始化。

配对 AB 必须先完成。对照只读取 AB 的训练侧范数、反馈样本身份和配置指纹，不加载 AB 模型、不重放中间轮次、不依据测试精度选择步长或检查点。如果某轮 AB 因 donor 并集为空而跳过校准，对照也跳过该轮；该情况会保留在结果中。

这个对照匹配反馈数据、损失权重、校准步数和有效更新幅度。参数量、优化几何、筛选开销和通信量不同，会分别记录。因此，AB 优于它只能支持“相对该直接校准对照，donor 更新重组有额外价值”，不能证明筛选最优、每个类别都受益或所有额外训练方案都无效。

## 3. 服务器运行命令

从项目根目录执行。以下示例沿用服务器已有 `references/full10_clientlt` 的划分和调度；此目录需要五个协议文件，不需要其训练检查点，也不要求有 `sfra_config.json`：

`partition_manifest.csv`、`bridge_metadata.json`、`protocol/full_schedule.json`、`protocol/eri_protocol.json`、`protocol/probe_manifest.csv`。

数据应位于 `DATA/cifar-100/cifar-100-python/{train,test,meta}`。GPU 编号按服务器空闲卡替换。每条命令独立可用；多种子会在同一个进程中依次运行。

**先登记计划并做文件预检：**

```bash
python -u scripts/run_ab_validation.py --stage plan --arms a ab --seeds 42 43 44 --reference-run references/full10_clientlt --data-root DATA
python -u scripts/run_ab_validation.py --stage preflight --arms a ab --seeds 42 43 44 --reference-run references/full10_clientlt --data-root DATA
```

默认输出为 `output/cifar100_LT/ab_validation`。`plan` 只登记计划并打印后端命令；`preflight` 检查源码、协议文件、数据文件是否存在及对照依赖，不加载 CLIP，不代表 GPU 验证通过。只有 `--stage train` 启动训练。

**核心：两张卡分别运行 A-only 和 A+B。**

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_ab_validation.py --stage train --arms a --seeds 42 43 44 --reference-run references/full10_clientlt --data-root DATA
CUDA_VISIBLE_DEVICES=1 python -u scripts/run_ab_validation.py --stage train --arms ab --seeds 42 43 44 --reference-run references/full10_clientlt --data-root DATA
```

这两条命令共六次完整训练，不是六种新方法。若先只确认新种子，将计划、预检和训练中的 `--seeds 42 43 44` 统一换成 `--seeds 43 44`，共四次。需要只跑一个种子时用 `--seeds 43`。新入口不自动复用旧目录或伪造旧运行的源码登记。

**AB 完成后，运行直接校准对照。**

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/run_ab_validation.py --stage train --arms a-calibration --seeds 42 43 44 --reference-run references/full10_clientlt --data-root DATA
```

每个种子的 AB 依赖自动定位，缺失时会在训练前明确报错；不会隐式启动缺失的 AB。可以先用 `--seeds 42` 做开发种子上的对照，再按预先确定的种子集合完成确认。不要根据哪个种子的效果好而只补该种子。

**可选：补同频 S 基线或 A 的已有消融。**

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_ab_validation.py --stage train --arms s --seeds 42 43 44 --reference-run references/full10_clientlt --data-root DATA
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_ab_validation.py --stage train --arms a-flat --seeds 42 43 44 --reference-run references/full10_clientlt --data-root DATA
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_ab_validation.py --stage train --arms a-current --seeds 42 43 44 --reference-run references/full10_clientlt --data-root DATA
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_ab_validation.py --stage train --arms a-no-cp --seeds 42 43 44 --reference-run references/full10_clientlt --data-root DATA
```

**中断后续训：在原训练命令末尾添加 `--resume`。**

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/run_ab_validation.py --stage train --arms ab --seeds 42 43 44 --reference-run references/full10_clientlt --data-root DATA --resume
```

完整运行自动跳过；未完成运行从 `checkpoints/sfra_last.pt` 恢复。初始化失败、尚未形成检查点时，保留原目录并使用新的 `--output-root`。计划后不要改变训练源码、参考协议、数据路径、workers 或冻结参数；入口会拒绝将更改后的同名任务混入原计划。

不同组/种子可并行运行；文件锁防止两个进程同时训练同一个输出目录，并保护计划合并。请等相关进程结束后再汇总，避免读取正在写入的结果文件。

## 4. 可选的普通 Dirichlet 扩展

只扩展已有 CIFAR-100-LT 普通 Dirichlet β=0.5 协议，仍冻结同一组 A/B 参数。普通 Dirichlet 的客户端容量随划分变化，不能用于“客户端容量和类别总量都固定”的集中度因果主张。

```bash
python -u scripts/run_ab_validation.py --stage preflight --arms a ab --seeds 42 43 44 --partition noniid-labeldir-fine --dirichlet-beta 0.5 --data-root DATA --output-root output/cifar100_LT/ab_validation_dirichlet
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_ab_validation.py --stage train --arms a --seeds 42 43 44 --partition noniid-labeldir-fine --dirichlet-beta 0.5 --data-root DATA --output-root output/cifar100_LT/ab_validation_dirichlet
CUDA_VISIBLE_DEVICES=1 python -u scripts/run_ab_validation.py --stage train --arms ab --seeds 42 43 44 --partition noniid-labeldir-fine --dirichlet-beta 0.5 --data-root DATA --output-root output/cifar100_LT/ab_validation_dirichlet
CUDA_VISIBLE_DEVICES=2 python -u scripts/run_ab_validation.py --stage train --arms a-calibration --seeds 42 43 44 --partition noniid-labeldir-fine --dirichlet-beta 0.5 --data-root DATA --output-root output/cifar100_LT/ab_validation_dirichlet
```

最后一条仍需等配对 AB 完成。这里不传 Client-LT 的 `--reference-run`。没有参考目录时，入口为同组任务固定生成日程，运行时重新准备相应划分；不会借用 Client-LT 的客户端容量。若 Client-LT 也要从头生成协议，在所有相关命令中一致去掉 `--reference-run`，并使用一个新的输出根目录。

本次未实现其他数据集、固定边际分配对照或外部论文算法的统一适配。已有训练运行时固定 100 类等条件，这些扩展需要单独核对协议，不能仅改数据集名称后称为完成验证。

## 5. 汇总与打包

使用独立的离线汇总入口 `scripts/collect_ab_validation.py`。2026-10-03 已修复真实 `round_metrics.csv` 同时包含 `seed`、`partition` 时的重复字典参数错误。为了兼容已经启动的实验，本次只新增入口，保留纳入训练指纹的旧文件不变；同步这个新脚本即可。原 `run_ab_validation.py --stage summary/pack` 仍是冻结的旧入口，请改用下面的命令。

```bash
python -u scripts/collect_ab_validation.py --stage summary
python -u scripts/collect_ab_validation.py --stage pack
```

使用非默认根目录时加相同的 `--output-root`：

```bash
python -u scripts/collect_ab_validation.py --stage pack --output-root output/cifar100_LT/ab_validation_dirichlet
```

本轮 seed42、0、3407 的结果目录使用：

```bash
python -u scripts/collect_ab_validation.py --stage pack --output-root output/cifar100_LT/ab_decision_42_0_3407
```

默认报告为 `output/cifar100_LT/ab_validation/analysis/report.md`，默认压缩包为 `output/cifar100_LT/ab_validation_analysis.tar.gz`。包内保存分析、配置、CSV、JSON、NPZ 等证据，不包含大型 `.pt` 模型，不可用于恢复训练。

| 输出文件 | 用途 |
|---|---|
| `status.csv` | 每组缺失、未完成、完成、无效或依赖尚未验证的状态与原因 |
| `per_run.csv` | 单次运行的末 20 轮、最终轮、回落及计算/通信开销 |
| `method_summary.csv` | 每个方法按种子统计均值与样本标准差，注明实际 n 和种子集合 |
| `paired_per_seed.csv` | 每个合格种子内 AB−A、AB−校准等配对差值 |
| `paired_summary.csv` | 对配对差值求均值与样本标准差，分 all/development/confirmation |
| `pair_audit.csv` | 初始化、数据、调度、见证、执行配置、源码等配对核对 |
| `per_class.csv`、`paired_per_class.csv` | 每类末 20 轮均值，以及配对的逐类差值、尾类标记 |
| `curves.csv` | 原有正式测试曲线汇总，不新增模型重放 |
| `report.md` | 可读主报告与结论边界 |

主指标固定为 **第 81–100 轮正式提交模型的 Tail20 均值**；同时报告 Overall、Head20、Middle60、Non-tail、最终轮及回落。按训练集全局数量预先固定 Tail20，不根据测试难度重新挑尾类。标准差按训练种子计算，不把 20 轮或 20 个类别当成独立重复。只有一个种子时标准差留空。

每项比较独立汇总，A/AB 完成后不必等待无关消融。配置不合格或配对指纹不同的结果不进入该配对平均；直接校准的样本或范数来源不一致时，整组校准结果被排除。缺失结果不会当作零。无效结果仍打包保留，并以非零退出码提示查看原因。

方法均值表可能有不同的已完成种子集合，不能直接相减代替配对统计；判断 B 的增量请使用 `paired_summary.csv`。通信是模拟消息量，时间是顺序联邦模拟的耗时，并非真实部署延迟。报告不会自动宣称显著性、最优筛选或 A/B 特殊协同。

## 6. 同步代码与本地验证

需要同步新增文件：

- `scripts/run_ab_validation.py`
- `scripts/collect_ab_validation.py`（离线汇总修复；已在训练的服务器只需补这个文件）
- `tools/sfra/ab_validation.py`
- `tools/sfra/calibration_reference.py`
- `utils/cliplora_b_calibration.py`
- `tests/test_sfra_ab_validation.py`
- `tests/test_sfra_ab_collection.py`
- 本说明文档。

需要同步修改文件：`federated_main.py`、`scripts/run_cliplora_sfra.py`、`utils/cliplora_sfra.py`、`tools/sfra/summary.py`。其余现有代码也应与本地仓库版本一致。原共享 B 的采样、筛选、C 优化实现保持原样；新增对照由独立参数显式启用。

本地检查命令：

```bash
python -m unittest discover -s tests -p "test_sfra*.py"
python scripts/run_ab_validation.py --help
```

测试覆盖配对资格、种子统计、源码冻结、恢复登记、打包、空 donor 事件、直接残差的实际两步优化和范数投影、模型/RNG 恢复及异常退出。测试使用合成数据和 CPU，不证明真实 GPU 数值表现或方法收益。方法最终是否保留 B，要由服务器运行所得的固定指标和配对结果决定。

本次本地回归共 143 项：142 项通过，1 项 CUDA 检查因无 GPU 跳过；语法编译与命令行入口检查通过。另外，已用旧 w=0.35 的真实分析包验证范数参考提取，可读取 8 个事件、16 个校准步和每事件 22 个反馈客户端；该检查没有将历史运行登记为新的正式重复。
