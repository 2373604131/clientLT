**方法 A 正文实验：五组固定设置与运行说明**

状态：已实现实验入口、两种 CP 消融、配对核对和正文表格汇总。本文说明如何运行，不代表已完成 CIFAR-100-LT 的 100 轮训练。方法定义沿用 [冻结规范](cliplora_method_a_full_cp.md)。

**正文用这五组即可先形成主结果与基本消融。**

| 命令中的方法名 | 正文名称 | 相对 Full-CP 的变化 | 用途 |
|---|---|---|---|
| `s` | 同频 S | 提交普通 A 提案，不执行功能修正 | 主结果基线 |
| `full-cp` | 方法 A | 无 | 主结果与消融参照 |
| `flat-cp` | 去掉来源优先级 | 有效功能单位权重统一，保留当前目标、历史、CP | 来源优先级消融 |
| `current-cp` | 去掉历史目标 | 仅使用当前改善目标，保留来源权重和 CP | 历史消融 |
| `full` | 去掉 CP | 保留来源与历史，去掉分类正则 | CP 消融 |

`flat-cp` 仍然计算来源响应来构造当前目标，它不是移除所有来源信息。`current-cp` 保留历史记录用于日志和续训，但历史值不进入功能目标或激活判定。单因素指规则上的单项改变，长期训练后的模型与目标数值自然会分叉。

正文安排：主结果表展示 S 与 Full-CP；消融表展示 Full-CP 与三种消融。其他外部方法比较沿用整篇论文原有安排。此次不增加打乱、简单计数、等预算额外训练、匹配拓扑和参数扫描；五组结论限定为整体效果与组件作用，不据此宣称集中度已被证明是遗忘的原因。

**固定配置与数据口径**

- 单独评估方法 A，B transfer 关闭。完整方法公式与默认参数保持冻结版本。
- 功能修正 lambda=10；三个带 CP 的设置 mu=1；无 CP 的 `full` 不使用 mu。
- rank=4、scaling=0.5、FP32、LA tau=1；100 轮，每轮 B 三个本地 epoch，第 1–90 轮 A 一个本地 epoch。
- 三步修正、步长 0.1，固定提交第三步；不按测试最佳轮次选模型。
- 主指标为第 81–100 轮正式模型的 Overall、Head20、Middle60、Tail20 均值；另报 Tail 最高值到第 100 轮的回落，以及第 90 到第 100 轮的变化。
- 默认 Client-LT、训练 seed42、划分/日程 protocol seed42。三个训练种子用 `--seeds 42 43 44`；在相同划分上改变初始化、训练随机性及固定见证抽样，同一训练种子内五组必须配对。
- 现有标准 Dirichlet 可选运行，名称为 `noniid-labeldir-fine`、beta=0.5；它不是固定边际匹配拓扑，当前不要求加跑。

**运行环境**

在已有 CAPT/CLIP-LoRA 训练环境中，从项目目录运行以下命令。`DATA` 替换成训练机的数据根目录；其布局沿用当前 CIFAR-100-LT 实验。训练需要原项目依赖与 CLIP 权重；新计划和汇总入口本身只用 Python 标准库，没有新增训练依赖。

所有命令前台执行，顺序运行选定组，不自动分配多张 GPU。Linux 可在命令前设置 `CUDA_VISIBLE_DEVICES=0`；PowerShell 可先执行 `$env:CUDA_VISIBLE_DEVICES='0'`。

**1. 先查看运行计划，不加载数据、不启动训练。**

```bash
python scripts/run_method_a_maintext.py --stage plan --seeds 42 --data-root DATA
```

默认输出目录为 `output/cifar100_LT/method_a_maintext`。计划保存在 `maintext_plan.json`，记录五组的参数、路径与训练代码摘要。即使只选择其中一组执行，计划仍列出该种子的全部五组，便于提示缺失结果。

**2. 运行一个种子的五组。**

```bash
python scripts/run_method_a_maintext.py --stage train --seeds 42 --data-root DATA
```

默认自动准备协议，不依赖已有 Full-CP 或旧 E3 的目录。五组结束后自动生成配对报告。seed42 是单种子开发结果，不会生成虚构的标准差。

如果准备直接得到三个种子的正文统计，可将上一命令换成：

```bash
python scripts/run_method_a_maintext.py --stage train --seeds 42 43 44 --data-root DATA
```

共 5×3=15 次 100 轮训练，不是十五种算法。协议种子默认仍为 42；已有完成且配置一致的组自动跳过。

**3. 中断后继续，或只运行某一组。**

```bash
python scripts/run_method_a_maintext.py --stage train --seeds 42 43 44 --data-root DATA --resume
python scripts/run_method_a_maintext.py --stage train --seeds 42 --methods flat-cp --data-root DATA
```

续训使用 `checkpoints/sfra_last.pt`，恢复原保存配置、正式模型、历史、见证和随机状态。已完成组自动跳过；未完成组必须显式加 `--resume`，不会覆盖重跑。初始化阶段若失败且尚无检查点，使用新的 `--output-root` 保留原诊断。

计划/训练/续训的 `--data-root`、`--num-workers`、`--witness-batch-size`、协议和训练代码应一致。默认 workers=8、见证 microbatch=8；它们可在首次计划时指定。训练代码或已有组配置改变时，入口拒绝混用目录，请使用新输出目录；不会把更改后的代码冒充同一实验。

**4. 只汇总结果或打包。**

```bash
python scripts/run_method_a_maintext.py --stage summary
python scripts/run_method_a_maintext.py --stage pack
```

若训练时使用自定义 `--output-root`，汇总和打包时也传同一路径。打包只包含分析文件，不包含模型权重或恢复训练所需的大检查点。

**可选：沿用已有实验的划分与日程。**

```bash
python scripts/run_method_a_maintext.py --stage train --seeds 42 --data-root DATA --reference-run output/cifar100_LT/sfra_v1/seed42/client-longtail/full/lambda10_protocol42 --output-root output/cifar100_LT/method_a_maintext_replay
```

该目录需要现有 `partition_manifest.csv`、`bridge_metadata.json` 和完整 protocol 文件。只复用数据/日程，不加载其训练好的权重。指定后，后续 plan/train/resume 同样带上该参数。它不是把旧结果自动纳入新表格；新入口只汇总本次登记并保存来源记录的运行。

**生成哪些结果**

完成的运行分别位于：

```text
output/cifar100_LT/method_a_maintext/
  maintext_plan.json
  seed42/client-longtail/
    s/baseline_protocol42/
    full-cp/lambda10_mu1_protocol42/
    flat-cp/lambda10_mu1_protocol42/
    current-cp/lambda10_mu1_protocol42/
    full/lambda10_protocol42/
  maintext_analysis/
    report.md
    main_table.csv
    ablation_table.csv
    all_methods.csv
    paired_differences.csv
    per_run.csv
    status.csv
    protocol_audit.csv
```

| 文件 | 怎么使用 |
|---|---|
| `report.md` | 阅读正文表格；均值及跨种子的样本标准差 |
| `main_table.csv` | S 和 Full-CP 的主结果 |
| `ablation_table.csv` | 完整方法与三组单因素消融 |
| `paired_differences.csv` | 相同种子上 Full-CP 减各对照的差值，再跨种子汇总 |
| `per_run.csv` | 已完成运行的原始终点、CP 最终违反率、记录的成本；便于中途查看 |
| `status.csv` | 缺失、未完成、无效或已完成的每一组 |
| `protocol_audit.csv` | 同种子内划分、初始化、日程、训练设置、参数顺序、见证和代码摘要是否一致 |

正文均值表仅使用五组都完成且通过核对的种子块，避免某组多一个种子或某组用另一套见证。部分组完成时仍可查看 `per_run.csv`；无效配对不会进入正文表。单种子不填写标准差。不同拓扑、划分种子和 beta 不合并统计。

`cp_loss_increase_rate` 只统计第 1–90 轮正式提交后的分类损失相对普通提案上升超过 `1e-8` 的比例；不使用第三次更新之前的中间损失冒充最终值。

成本字段将正常训练、功能阶段及评估记录中的载荷汇总，标记为**模型估算的通信量**；墙钟时间是包含评估与保存的顺序模拟时间，不能解释为多机联邦延迟。分类前向复用功能前向，不再次叠加计算量。

**实现检查**

```bash
python -m unittest discover -s tests -p "test_sfra*.py" -q
```

测试覆盖新消融保留 CP、目标与权重只发生指定变化、真实小模型的三步梯度和 B 冻结、入口分派、续训与跳过、配对失配拦截、统计终点和成本汇总。它不下载模型或数据，也不执行论文训练。
