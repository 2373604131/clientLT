# Seed42 正式对比入口与运行说明

第二轮共享因子基线已接入独立入口，见 [factor_benchmarks_seed42.md](factor_benchmarks_seed42.md)。第一批默认方法与 A/AB 方法配置保持原样。

## 启动时 `build_transform` 的 AssertionError 修复

历史参考文件的 `resolved_config` 来自 `str(CfgNode)`：`INPUT.TRANSFORMS` 和 `INPUT.SIZE` 的元组会在 YAML 读取后变成字符串。旧入口逐字符检查转换名称，因而在训练前失败。`tools/benchmarks/runtime.py` 现使用 `ast.literal_eval` 安全还原元组；保持原始预处理、数据分配和方法参数。

将修复后的 `tools/benchmarks/runtime.py` 同步到服务器。失败任务已登记旧代码指纹，重新启动时给第一批四条训练命令统一追加 `--output-root output/cifar100_LT/paper_benchmarks_seed42_v2`。原日志保留，失败位置没有产生已提交的训练轮次；无需 `--resume`。不要只删除断言或修改参考配置来绕过错误。

使用新目录后，查询和打包也要指定该目录：

```bash
python -u scripts/collect_paper_benchmarks.py --status --output-root output/cifar100_LT/paper_benchmarks_seed42_v2
python -u scripts/collect_paper_benchmarks.py --pack --output-root output/cifar100_LT/paper_benchmarks_seed42_v2
```

同一修复适用于第二轮入口；若第二轮也已经登记旧版本任务，给第二轮命令追加独立的新输出目录 `--output-root output/cifar100_LT/factor_benchmarks_seed42_v2`。

实现日期：2026-10-05。本批只使用训练 seed42、协议 seed42。

## 目录安排

| 目录/文件 | 用途 |
|---|---|
| `third_party/paper_baselines/` | 三个官方方法的固定版本源码快照、来源及逐文件哈希 |
| `trainers/baselines/` | FedNTD、FedPuReL-Global、CAPT 的适配计算与共用聚合 |
| `tools/benchmarks/` | 协议检查、训练、恢复、汇总、旧结果导入 |
| `configs/benchmarks/clientlt_seed42.json` | 本批固定超参数和类别分组 |
| `scripts/run_paper_benchmarks.py` | 统一入口 |
| `scripts/collect_paper_benchmarks.py` | 独立查询、汇总和打包入口 |

原 A/AB 训练文件没有修改。`a`、`ab` 委托原 `run_ab_validation.py` 运行，显式传入 `--seeds 42`，不使用那个旧入口的默认多种子列表。

## 比较对象与边界

| 命令名称 | 实际方法 |
|---|---|
| `fedavg-lora` | 标准样本数加权 FedAvg，客户端同时训练 LoRA 两个因子，CE 损失 |
| `capt` | CAPT 固定聚合频率适配版：保留提示/耦合层、提示损失和聚类聚合，关闭测试反馈控制的 MAB |
| `fedpurel` | FedPuReL **共享阶段**、匹配 LoRA 结构的适配版：零样本 CLIP 教师、熵匹配、逐参数梯度投影 |
| `fedntd` | FedNTD 适配 CLIP-LoRA：本轮全局模型教师，排除真类的蒸馏，T=3、β=1 |
| `a` | 已冻结的 A，Full-CP λ=10、μ=1 |
| `ab` | A 加轮换采样共享 B，w=0.35 |
| `fedavg-lora-la` | 可选基础损失对照：联合训练两个因子的 FedAvg，加全局类别先验 LA；默认不启动 |

外部 LoRA 对照与 A 使用 ViT-B/16、rank4、视觉 top3 的 q/v、alpha=1、dropout=0、FP32。CAPT 保持自身提示学习结构，并记录参数量。外部方法采用每客户端每轮 3 个本地 epoch、SGD lr=.001、momentum=.9、weight decay=.0005，100 轮全参与。A/AB 有额外阶段，不能写成相同总计算量。

FedPuReL 的个性化阶段没有接入本次共享模型比较，不把结果写成完整个性化 FedPuReL。CAPT 固定频率版也不等于原始 MAB 版本。所有表格标签会自动写明适配范围；本批用于协议下的方法比较，不能直接宣称复现了原论文数字。

本仓库的 `DatasetCifar100` 实际使用 CIFAR 均值/方差归一化后 resize 至 224。新入口复用这一真实处理流程，与 A/AB 对齐；不会误把 YAML 中的 CLIP 归一化/随机增强当成当前真实处理，也不会单独修改 A/AB 的输入协议。

## 服务器准备与短程检查

将新增的 `scripts`、`tools/benchmarks`、`trainers/baselines`、`configs/benchmarks`、`third_party/paper_baselines` 同步到 `/data/yzh/clientLT`。官方快照已保存，不需要在服务器另外克隆仓库。

以下命令均在项目根目录、原 `clientLT` 环境中运行，默认读取 `references/full10_clientlt` 和 `DATA`，默认输出 `output/cifar100_LT/paper_benchmarks_seed42`。如果参考目录不同，所有计划/预检/训练/导入命令都显式加上相同的 `--reference-run 完整参考目录`。

先检查依赖、CUDA、参考文件和固定协议，不训练：

```bash
python -u scripts/run_paper_benchmarks.py --stage preflight
```

新适配入口先各跑两轮、每轮两个客户端、每客户端一个批次，检查模型构建、真实前后向、聚合和保存。仍使用完整测试集验证评估接口。这些结果自动存到 `paper_benchmarks_seed42/smoke`，不会进入正式表：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_paper_benchmarks.py --stage smoke --methods fedavg-lora capt fedpurel fedntd
```

这一步是新代码的服务器检查，不是新增论文实验，不要求额外做原来那些逐轮机制诊断。本地只有 CPU，尚未替代服务器完成真实 CLIP/CUDA 检查。

## 复用已完成的 seed42 A 和 AB

如果服务器保留原完整 AB 验证目录，先导入，程序核对源文件、完成回执、训练配方、样本分配、调度、类别分组和结果：

```bash
python -u scripts/run_paper_benchmarks.py --stage import-ab --import-ab-root output/cifar100_LT/ab_decision_42_0_3407
```

这不会复制模型或修改原结果，只登记经过核验的 A、AB。目录内其他种子和消融不进入本批比较。已对本地回收的真实结果验证：A 的 Overall/Tail20 为 70.6330/70.9350，AB 为 70.6965/71.1575，与原汇总一致。

## 正式启动

四个外部方法可分配到四张空闲 GPU，各自在一个终端运行。GPU 编号可以按服务器资源修改：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_paper_benchmarks.py --stage train --methods fedavg-lora
```

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/run_paper_benchmarks.py --stage train --methods capt
```

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/run_paper_benchmarks.py --stage train --methods fedpurel
```

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/run_paper_benchmarks.py --stage train --methods fedntd
```

如果旧 A/AB 无法导入，才使用以下命令在一张卡上依次训练两组：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_paper_benchmarks.py --stage train --methods a ab
```

省略 `--methods` 会按顺序执行六组，已核验完成或已导入的运行自动跳过。不是并行占用六张卡。所有新训练都固定 seed42。

如果要额外检验普通 FedAvg 的 LA 损失对照，可显式运行 `--methods fedavg-lora-la`；它不是内部同频 S，不改变默认六组。

## 恢复、进度、汇总、打包

外部方法按完整轮次保存模型参数、随机状态和预算记录；每个客户端优化器独立新建。中途终止的半轮会从上一个完整轮次重跑。A/AB 保留原有恢复方式。以 FedNTD 为例：

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/run_paper_benchmarks.py --stage train --methods fedntd --resume
```

也支持外部方法 `--stop-after 1` 在完整第 1 轮后停止，再用同一命令去掉该选项并加 `--resume`；这不会改变注册的 100 轮配方。源码或配置改动后程序拒绝继续混用旧运行，不会静默覆盖。

查看已完成轮数、正在训练的客户端：

```bash
python -u scripts/collect_paper_benchmarks.py --status
```

完成后汇总并打包：

```bash
python -u scripts/collect_paper_benchmarks.py --pack
```

得到 `output/cifar100_LT/paper_benchmarks_seed42_analysis.tar.gz`。包内包含汇总、逐类数据、曲线、成本、配置、回执和导入的 A/AB 分析文件，不包含模型权重和图片。

主表为第 81–100 轮的 Overall、Head20、Middle60、Tail20 平均 ACC；同时给出末轮、逐类变化和配对差值。新基线逐轮从正确数/样本数重算；不完整、错误或 smoke 结果不会填零或进入主表。只有一个种子，不报告跨种子标准差或统计显著性。`costs.csv` 保留各阶段计数；A/AB 原始 elapsed 含诊断，而新基线 train_seconds 不含测试，不能直接当作相同口径的耗时排名。

## 本地验收

- 官方 FedNTD 损失/梯度、FedPuReL 温度匹配/实际投影更新、CAPT 损失/梯度及聚合数值对照通过。
- CPU 小模型的真实本地更新、教师冻结、客户端优化器隔离和“连续训练＝断点恢复”检查通过。
- 固定种子、输入协议、重复样本、错误结果排除、短程与正式结果隔离检查通过。
- 已完成的真实 seed42 A/AB 导入、汇总及不含模型的打包检查通过。
- 尚未运行服务器 GPU 训练，不把 CPU 验收写成真实 CUDA 实验完成。
