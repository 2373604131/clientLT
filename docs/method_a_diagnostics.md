# 方法A小规模诊断：运行说明

本入口执行已经确定的问题1/2方案：已有日志分析、两个起点的四候选单次比较、三个分支的五轮训练。无CP只在第61、81轮各做一次，不单独续训。

当前本地已完成日志分析，见 [分析报告](../output/method_a_diagnostics_20260927/report.md)。服务器短程训练尚未执行。

## 本地可直接运行

从仓库根目录执行。默认读取已有Full-CP μ1轻量结果包，输出到独立的output/method_a_diagnostics_20260927，不改写原结果。

```bash
python scripts/run_method_a_diagnostics.py --stage analyze
python scripts/run_method_a_diagnostics.py --stage preflight
```

没有matplotlib时可给分析命令加 --no-plot。文件检查不加载CLIP、不启动训练；即使文件齐全，也仍需运行时核对模型、历史和协议。

本地检查确实缺少：

- checkpoints/base_model.pt
- sfra_rounds/r060/commit.pt
- sfra_rounds/r080/commit.pt

本地torch是CPU版，未发现可用CUDA。其余已有CSV/JSON/NPZ足够完成日志分析，不能代替上述模型文件。原记录的服务器运行目录为 output/cifar100_LT/sfra_cp/seed42/client-longtail/full-cp/lambda10_mu1_protocol42，需在训练服务器确认该目录仍完整。

## 服务器执行

### GPU 1 / GPU 2 两组并行，新版加速

同步本次修改的 `scripts/run_method_a_diagnostics.py`、`utils/sfra_diagnostics.py` 及既有 v2 执行模块到服务器。在同一训练环境、仓库根目录打开两个终端，分别粘贴以下单行命令。这里只提供服务器命令，本地没有执行 CUDA 训练。

终端一，GPU 1，实验 1→2→3（第 60 轮起点）：

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/run_method_a_diagnostics.py --stage train --source-run output/cifar100_LT/sfra_cp/seed42/client-longtail/full-cp/lambda10_mu1_protocol42 --data-root DATA --anchor 60 --output-root output/method_a_diagnostics_a60_fast_v2_f128_c4 --fast-execution-v2 --feedback-batch-size 128 --feedback-cache-gib 4
```

终端二，GPU 2，实验 4→5→6（第 80 轮起点）：

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/run_method_a_diagnostics.py --stage train --source-run output/cifar100_LT/sfra_cp/seed42/client-longtail/full-cp/lambda10_mu1_protocol42 --data-root DATA --anchor 80 --output-root output/method_a_diagnostics_a80_fast_v2_f128_c4 --fast-execution-v2 --feedback-batch-size 128 --feedback-cache-gib 4
```

省略 `--branch` 时自动依次运行 `full-cp`、`ordinary`、`norm-matched`。一个分支失败会停止该组，另一张 GPU 的组可以继续。两组必须保持这里各自独立的 `--output-root`，避免共享计划/汇总写入冲突；同组的后两条分支读取本组 Full-CP 的核验和幅度记录。

128 是反馈前向批量，固定特征 GPU 缓存硬上限为 4 GiB，训练 batch 不变。只对新建的五轮配对诊断启用 v2：原起点与历史先按源实验执行方式恢复、核验；切换 v2 后再核对起点 witness margin 最大误差不超过 `5e-6`。历史继续使用原保存分数，不把加速后的数值写回历史轨迹。核验失败则停止，不放宽阈值或重置状态。切换记录在各分支的 `diagnostic_execution_audit.json`，配置也进入计划和断点，三条分支必须一致。

中断后，在对应组的同一命令末尾加 `--resume`；已完成的分支自动跳过，未完成的分支从自己的 checkpoint 恢复。不要在原未加速输出目录中切换配置。

两组各自的汇总目录只包含本组起点，因此 `suite_status.json` 仍会显示整套六分支未齐；每条分支成功以其 `diagnostic_completion.json` 为准。两组全部完成后再共同收集两份结果，不能把两份局部汇总分别当成完整六分支报告。

本地新增检查覆盖固定执行配置、单起点三分支顺序、两组输出隔离、实际小型 ViT/LoRA 的执行切换与起点误差检查、历史及模型/RNG 不变，以及启动/恢复时先核验起点再切换的顺序。14 项诊断 CPU 测试通过；没有服务器性能实测，不承诺 128 比 64 更快。

### 原执行方式及逐条启动

先同步新增脚本、工具模块和既有入口的少量接线改动，在原clientlt环境和仓库根目录运行。以下命令的DATA与原训练保持一致；若目录位置改变，替换 --source-run 和 --data-root。

```bash
python scripts/run_method_a_diagnostics.py --stage preflight --source-run output/cifar100_LT/sfra_cp/seed42/client-longtail/full-cp/lambda10_mu1_protocol42

python -u scripts/run_method_a_diagnostics.py --stage train --source-run output/cifar100_LT/sfra_cp/seed42/client-longtail/full-cp/lambda10_mu1_protocol42 --data-root DATA
```

所有训练前台、顺序运行，不自动分配GPU。顺序为anchor60的full-cp、ordinary、norm-matched，再运行anchor80的三组。固定每条五轮，不支持本入口扩大seed、改变起点或搜索超参数。

也可以用 `--anchor` 和 `--branch` 每次只启动一条分支。两项仅用于 `--stage train`；省略时仍运行整套。以下六条按顺序执行，共用同一输出根目录；同一起点的 full-cp 必须先完成，另外两条分支需要它的公共状态核验和更新幅度记录。请勿同时向同一输出根目录启动多个入口进程。

```bash
python -u scripts/run_method_a_diagnostics.py --stage train --source-run output/cifar100_LT/sfra_cp/seed42/client-longtail/full-cp/lambda10_mu1_protocol42 --data-root DATA --anchor 60 --branch full-cp
python -u scripts/run_method_a_diagnostics.py --stage train --source-run output/cifar100_LT/sfra_cp/seed42/client-longtail/full-cp/lambda10_mu1_protocol42 --data-root DATA --anchor 60 --branch ordinary
python -u scripts/run_method_a_diagnostics.py --stage train --source-run output/cifar100_LT/sfra_cp/seed42/client-longtail/full-cp/lambda10_mu1_protocol42 --data-root DATA --anchor 60 --branch norm-matched
python -u scripts/run_method_a_diagnostics.py --stage train --source-run output/cifar100_LT/sfra_cp/seed42/client-longtail/full-cp/lambda10_mu1_protocol42 --data-root DATA --anchor 80 --branch full-cp
python -u scripts/run_method_a_diagnostics.py --stage train --source-run output/cifar100_LT/sfra_cp/seed42/client-longtail/full-cp/lambda10_mu1_protocol42 --data-root DATA --anchor 80 --branch ordinary
python -u scripts/run_method_a_diagnostics.py --stage train --source-run output/cifar100_LT/sfra_cp/seed42/client-longtail/full-cp/lambda10_mu1_protocol42 --data-root DATA --anchor 80 --branch norm-matched
```

每次运行后都会更新整套实验的汇总。六条全部完成之前，`suite_status.json` 的 `complete: false` 是整套尚未完成，不表示刚运行的分支失败。单独启动和整套启动共用同一个实验计划，不能在中途修改训练代码；修改后需使用新的输出根目录。

若当前实现与原记录不一致、首轮普通提案哈希不同、历史重建失败、有效步幅匹配失败，入口报错停止，不自动换模型、重置历史或从头跑100轮。

已经完成且配置一致的分支自动跳过；中断后使用同一命令并加 --resume。从本入口自己的diagnostic_last.pt恢复，不能把普通SFRA的--resume当作分支恢复。首个诊断检查点之前失败时，修复原因后使用新的 --output-root，保留旧诊断目录。

```bash
python -u scripts/run_method_a_diagnostics.py --stage train --source-run output/cifar100_LT/sfra_cp/seed42/client-longtail/full-cp/lambda10_mu1_protocol42 --data-root DATA --resume
python scripts/run_method_a_diagnostics.py --stage summary
```

自定义输出根目录时，所有命令都带相同的 --output-root。

## 状态恢复与公平性

- 用基础模型加第60/80轮正式LoRA提交恢复模型，用下一轮普通B事件的输入哈希核验提交身份，并重算锚点见证margin。
- 初始模型和实际训练样本、划分、测试集、见证必须匹配。恢复起点与历史时执行模式必须与源实验一致；显式指定新版加速时，仅在这些核验通过后切换新诊断的执行方式，并额外核对起点 margin。历史用第1轮到起点的真实提交分数逐轮重建，并逐轮核对历史值与有效位，不重置历史。
- 历史运行通常只保留最后一个完整随机状态。因此新分支显式使用新的、按轮次及阶段固定的配对随机流；不声称精确重放原来的61–65/81–85轮结果，原片段不直接混进新表。
- 三个分支首轮普通B状态与普通A提案必须完全一致；后续各自正常训练B，状态自然分叉。
- 无CP使用同一普通提案和原full-cp路径，仅令μ=0。其历史与随机状态隔离，单独标记成本，不影响正式提交。
- norm-matched在自己的模型上构造普通提案，将全体层的有效sBΔA合并范数匹配到Full-CP同轮幅度。只用一个标量，允许实际需要放大时如实记录，不暗中截断。
- 等步幅组借用了Full-CP的幅度序列，属于诊断控制；不能用它宣传独立低成本训练。α、有效范数和匹配误差都保留。
- 每个候选保存训练见证损失及独立测试预测，测试表现不参与更新选择。单次对照包含pre-A参照；短程表固定报告五轮均值和第五轮减共同起点。

## 输出

| 文件 | 含义 |
|---|---|
| report.md、q1_group_loss.png/pdf | 已有90轮日志的分类损害分析和图 |
| q1_class_rounds.csv、q1_class_windows.csv | 逐类、逐轮及分窗口损害，含真实训练计数 |
| q1_group_rounds.csv、q1_group_windows.csv | 类均衡与样本加权两种分组口径 |
| q1_medium_late_ranking.csv | 后期中频损害排序 |
| analysis_provenance.json | 原始输入哈希、容差与重算核对 |
| preflight.json、required_files.csv | 文件缺口 |
| diagnostic_plan.json | 固定实验、输入与代码哈希 |
| anchor60或anchor80/各分支/candidate_metrics.csv | 单次与五轮正式提交的测试指标，kind字段区分用途 |
| 各分支/witness_metrics.csv | 训练见证LA与margin；不能冒充测试准确率 |
| 各分支/witness_tokens/ | 每个候选的逐客户端—类别原始LA与margin，支持继续定位具体受损类别 |
| 各分支/predictions/ | 固定测试顺序下逐样本预测、CE、LA和margin |
| 各分支/norms.csv | 有效幅度匹配、首轮公共状态哈希 |
| 各分支/no_cp_probe/ | μ0单次探针，仅存在于完整A分支 |
| 各分支/functional_costs.csv、local_training_budget.csv、diagnostic_evaluation_costs.csv | 分开记录训练、功能反馈和纯诊断成本 |
| short_run_report.md、short_run_summary.csv | 固定五轮结果，两个起点分别报告 |
| same_state_differences.csv、same_state_witness_differences.csv | 完整A分别减普通更新、无CP、等步幅更新 |
| sample_retention.csv | 共同起点正确样本的保留、转错及原先错误转对数 |
| suite_status.json | 哪些分支仍缺失 |

当前保留率队列定义为“共同起点预测正确”，不据此推断能力源自预训练还是联邦训练。代码已保存新的逐样本预测；旧能力/新学会能力的完整拆分仍需额外的初始及连续历史模型评估，不能由已有逐类准确率倒推。

## 已完成的本地验证

使用CPU小模型实际反向传播检查：有效范数而非参数范数匹配、统一标量、放大情形、零更新失败、历史重建及损坏拒绝、三分支公共首轮、μ0隔离、中断恢复、逐样本评价不消耗训练随机数、固定窗口汇总及文件缺失不启动训练。

```bash
python -m unittest discover -s tests -p "test_sfra*.py" -q
```

这些检查不是CLIP正式实验。GPU缓存放置测试在没有CUDA时跳过，服务器的真实模型状态与端到端训练仍须实际运行验证。
