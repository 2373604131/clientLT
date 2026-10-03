# 方法 B：共享 C 的客户端平均保护与逐类保护

实现日期：2026-10-03。实现本轮两个候选与关闭保护的校验入口，支持完整100轮、轮边界恢复、预检、进度查看、汇总和分析包。CPU数学/优化器/入口测试通过；本机无CUDA，尚未完成真实CLIP GPU训练，不预先保证精度收益。

方法与结果判断见[设计稿](方法B_共享更新类别冲突实验设计_20261003.md)。本入口没有实现候选胜出后才需要的范数匹配或带保护的直接校准对照；不会隐式启动这些后续实验，也不会启动上一版来源调权S。

## 先同步新增文件

将以下六个新增代码文件按相同相对路径放入服务器 `/data/yzh/clientLT`：

```text
scripts/run_cliplora_b_class_guard.py
scripts/train_cliplora_b_class_guard.py
tools/sfra/b_class_guard.py
utils/b_class_guard_math.py
utils/cliplora_b_class_guard.py
utils/cliplora_b_guard_runtime.py
```

旧的 `federated_main.py`、`utils/cliplora_sfra.py`、shared B 实现、AB验证与打包文件均未修改。新worker只在自身进程中安装运行时子类，利用现有 `configure_experiment` 钩子在恢复验证之前写入完整配置，再使用新的transfer对象。

环境仍需具备原来训练成功的CAPT依赖、CLIP、CUDA及数据。`references/full10_clientlt`只需包含原来的完整划分/协议文件，不要求模型权重或 `sfra_config.json`：

```text
partition_manifest.csv
bridge_metadata.json
protocol/full_schedule.json
protocol/eri_protocol.json
protocol/probe_manifest.csv
```

## 启动两组实验

在服务器已激活的 `clientLT` 环境、项目根目录下执行。GPU编号按空闲卡替换。两条命令可在两个终端并行运行；每条命令仅启动一个seed42、100轮实验。

E1，客户端内分别保护尾/非尾组平均损失：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_cliplora_b_class_guard.py --guard client --seed 42 --reference-run references/full10_clientlt --data-root DATA --fast-execution-v2 --feedback-batch-size 128 --feedback-cache-gib 4
```

E2，客户端内先按类别判断损害，再按原样本权重汇总：

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/run_cliplora_b_class_guard.py --guard class --seed 42 --reference-run references/full10_clientlt --data-root DATA --fast-execution-v2 --feedback-batch-size 128 --feedback-cache-gib 4
```

启动前会自动检查协议、CIFAR-100 train/test/meta文件、配置和源代码指纹。也可单独文件预检，不启动模型：

```bash
python scripts/run_cliplora_b_class_guard.py --stage preflight --guard client --seed 42 --reference-run references/full10_clientlt --data-root DATA
```

`--stage plan`只打印实际worker命令。预检通过只代表文件/配置检查通过，不代表GPU可用、文件内容完整性或精度已验证。

默认输出根目录：`output/cifar100_LT/sfra_b_class_guard`。两个目录分别含 `guard_client`、`guard_class`，不会互相覆盖。A、原Tail20、22个反馈端、原donor规则、每donor独立C、class-cyclic采样、w=0.35、两步Adam/lr0.3/C正则0.001均固定。保护系数固定1，无额外参数扫描。

## 进度与恢复

一次查看本输出根目录里的所有组：

```bash
python scripts/run_cliplora_b_class_guard.py --stage status
```

中断后在原命令末尾加 `--resume`，例如：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_cliplora_b_class_guard.py --guard client --seed 42 --reference-run references/full10_clientlt --data-root DATA --resume
```

恢复必须保持原guard、seed、数据根、协议源、worker数、代码及配置。每个输出有独立OS文件锁；同一组不能被两个进程同时启动。已经完成的组在 `--resume` 时跳过。没有轮边界checkpoint的启动失败目录不自动删除，应保留并换输出根。

可选的服务器运行检查：在训练命令末尾加 `--stop-after-round 30`，第30轮包含第一次B迁移。之后在原命令追加 `--resume --stop-after-round 0` 继续到100轮，配置和样本计划保持一致。这只是可选暂停，不是正式实验的额外前置任务。

`--guard off`用于关闭保护的数值/实现校验，直接调用旧共享B的两步优化路径；不是默认要求补跑的第三组。后续复核种子使用 `--seed 0` 或 `--seed 3407`，不会自动批量启动。

## 汇总和打包

两组完成后，把已有A/旧B最终验证结果加入同表：

```bash
python -u scripts/run_cliplora_b_class_guard.py --stage pack --compare-root output/cifar100_LT/ab_decision_42_0_3407
```

若服务器旧验证结果在其他位置，将 `--compare-root` 替换为包含那些完整运行目录的根目录。该参数仅用于离线分析，与提供划分的 `--reference-run` 不同。只汇总新两组时省略 `--compare-root`。

默认分析目录：`output/cifar100_LT/sfra_b_class_guard/analysis`。

默认压缩包：`output/cifar100_LT/sfra_b_class_guard_analysis.tar.gz`。

包内保留CSV/JSON/NPZ等标量、类别、来源和配置证据，不含大模型 `.pt` 权重与原始训练图片，不是可恢复训练的checkpoint包。旧参照运行不复制进新根目录，其数值和配对审计进入analysis。

主要文件：

- `performance.csv`：每组Overall/Head20/Middle60/Tail20/Non-tail80的末20轮均值与末轮值。
- `paired.csv`、`paired_per_class.csv`：仅通过审计的逐组/逐类配对差。
- `status.csv`、`pair_audit.csv`：部分完成、无效结果或不匹配的原因，不将其混进结论。
- `optimization_steps.csv`：保护损失、保护梯度/基础梯度、有效残差范数及每步变化。
- `guard_class_trace.csv`、`guard_client_trace.csv`：同一固定反馈池在C=0/第一步后/第二步后的客户端—类别损害。
- 每事件 `guard_references.npz`：参照损失、标签、客户端本地位置，不含图像。
- `b_transfer_rounds.csv`：包括额外保护梯度诊断反向次数在内的成本。

配对比较核验初始化、数据、划分、调度、见证样本、执行设置、固定旧代码指纹、前30轮曲线和各事件反馈样本身份。没有来源指纹收据的历史基线可显示，不能当作严格匹配对照。逐类表会复算末20轮五项指标，避免口径变化或重复元数据导致汇总错误。

## 计算与解释边界

每个客户端只有一次实际混合目标反向；为了记录保护梯度是否参与，激活的客户端还执行只读autograd梯度诊断。这些额外反向单独记为 `guard_diagnostic_backward_images`，不算新的优化步，也不冒充与旧B总计算完全相同。

C=0时保护梯度为零，主要在第二步响应第一步的损害。第二步后可能仍有损害；不进行测试门控、回退、额外纠正或类别专属推理。训练损失上的受损比例不是测试精度保证。

CPU测试覆盖：损害抵消与样本权重、参考停止梯度、CE有限差分梯度、两步共享C与集中目标一致、关闭保护与旧版精确一致、整个共享事件提交和无donor跳过、状态恢复、worker隔离、配置/代码变更拒绝、真实CSV字段、配对审计及权重文件打包排除。服务器上的真实GPU运行和泛化效果仍需实际实验。
