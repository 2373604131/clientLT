# 方法B问题2：运行说明

已实现独立的E00/E10/E01/E11完整训练入口，以及从保存事件重放四组对照的入口。旧共享B和正在进行的非尾覆盖/权重三组仍使用原入口、原配置；新模式仅在显式选择problem2时启用。

当前已做CPU小模型测试、入口和汇总检查，未在CIFAR/CLIP上作GPU验证，未启动任何正式训练。先完成问题1并确定参考运行，再在服务器执行预检和一个事件的重放。

## 四组设置

| 参数 | 类别平均 | 逐类损害项 |
| --- | --- | --- |
| E00 | 否 | 否 |
| E10 | 是 | 否 |
| E01 | 否 | 是 |
| E11 | 是 | 是 |

四组均使用全部当轮原始donor、共享完整矩阵C、两步同步Adam和一次共享残差提交。默认C学习率0.3、C正则0.001；损害项默认beta=1。类别平均保持每步原目标的实际尾/非尾总权重，包括只有尾类的接收端。

每个“接收端—类别”的损害为同一批图片上 `relu(LA(C)-stopgrad(LA(C=0)))`；先逐类取正部，再按该组实验的权重汇总。两步各自有独立的C=0参照。E00/E10也执行这次参照前向，四组的校准数据和算法前向预算一致。参照前向计为算法工作，后续只读轨迹计为诊断工作。

## A. 先检查服务器上的参考运行

参考必须是**原共享B或问题1的shared-tradeoff运行完整目录**，包含模型状态；仅含CSV的analysis压缩包不够。重放沿用参考运行的w、采样清单、A配置、模型、划分与执行模式，不重新挑权重，也不重新采样。

以下为服务器Bash示例，将第一行替换为选定的真实目录。这里的w应在问题1结果回来后统一确定。

```bash
B_SOURCE_RUN=/path/to/complete/shared_tradeoff_run
python scripts/run_cliplora_b_problem2_replay.py --stage preflight --source-run "$B_SOURCE_RUN" --output-root output/cifar100_LT/b_problem2_replay_seed42
```

预检默认检查30、60、90轮，输出 `preflight.json`。`Ready=False`时打印每个缺失文件，不加载模型、不启动训练。预检只检查文件与配置，真正重放时还会核对张量、模型哈希、数据身份与采样清单。

所需大文件：

- `checkpoints/base_model.pt`。
- 三个 `events/r030_c000_main_normal_B/state.pt`、`r060...`、`r090...`：普通B事件，包括原始donor更新。
- 三个 `b_transfer_rounds/r030/commit.pt`、`r060...`、`r090...`：普通聚合与迁移后的LoRA状态。

还需原配置、命令、数据协议、witness/monitor清单、三个事件的C矩阵与逐类日志。预检列出完整缺项。既有训练默认会生成相应事件文件；analysis打包有意排除了这些模型大文件。

本地旧共享B分析包预检结果为7个上述大文件缺失，无配置错误，报告在 `presentation/experiment_audit_20260926/b_problem2_preflight/preflight.json`。这不是服务器目录缺失的判断。

## B. 先做一个真实事件的GPU验证

```bash
python -u scripts/run_cliplora_b_problem2_replay.py --stage run --source-run "$B_SOURCE_RUN" --data-root DATA --rounds 30 --output-root output/cifar100_LT/b_problem2_replay_smoke_seed42
```

此命令加载模型并运行四组C校准和诊断，不训练本地B、不更新A、不进入100轮训练。`DATA`需指向原数据根目录，可替换为实际路径。

先检查：

1. `r030/equivalence.csv`：并集是否等于全集；E00重放是否与原保存C、残差接近；保存probe与重放probe的LA差异。
2. 四组 `class_weights.csv`：尾/非尾总权重不随类别平均切换而变化。
3. `class_loss_trace.csv`：同一权重下，有/无损害项的第一步应在数值精度内一致；第二步可能产生差异。
4. `replay_completion.json`：本事件四组共8个共享优化步骤，以及图像访问和额外诊断耗时。

若来源池不同，等价性字段会标记不能直接作同池比较；若同池但数值复现失败，应先检查实现/环境/状态，不能直接把后续差值写成方法收益。E00与原目标在数学上相同，不承诺不同浮点求和顺序下逐位相同，更不能据单事件接近就声称完整训练轨迹一致。

## C. 三个预定事件的完整机制重放

```bash
python -u scripts/run_cliplora_b_problem2_replay.py --stage run --source-run "$B_SOURCE_RUN" --data-root DATA --output-root output/cifar100_LT/b_problem2_replay_seed42
python scripts/run_cliplora_b_problem2_replay.py --stage pack --output-root output/cifar100_LT/b_problem2_replay_seed42
```

默认四组、30/60/90轮，共24个共享C优化步骤。额外donor条件贡献与幅度匹配是前向诊断，另计图像访问和耗时。

重放会：

- 使用相同普通B状态、原始donor张量与两批校准清单执行四组目标。
- 在原保存C上逐donor移除贡献；保持原分母、其他来源和C不变；与原有缺类探测日志共有的关系比较。
- 当新方案残差范数小于E00时，把E00缩到同样的有效LoRA范数，输出幅度匹配诊断。
- 用同一训练侧monitor评价全部配置并记录与校准样本的重叠。
- 保存源文件与代码哈希，拒绝混用改变后的计划或代码。

`--skip-donor-diagnostics`可在首次环境验证时减少前向，但这会形成不同的计划，需单独输出目录；它不改变四组校准。计划中必须保留E00。

同一命令再次执行会跳过已完整结束的事件。半个事件不恢复C；如果事件中断，使用新的输出根目录。不会自动删除、覆盖半成品，也不会改写源实验。

关键输出：

| 文件 | 用途 |
| --- | --- |
| `analysis/paired_differences.csv` | E10−E00、E01−E00、E11−E10、E11−E01 |
| `analysis/class_change_summary.csv` | 统一类别宏平均的净收益、正收益量、损害量和受损数 |
| `analysis/class_changes.csv` | 每个客户端—类别的变化及校准重叠 |
| `analysis/class_loss_trace.csv` | 两批各自C=0、第一步后、第二步后的类别损失 |
| `analysis/donor_contributions.csv` | 原始探测收益与固定组合后的条件贡献 |
| `analysis/norm_matched.csv` | 缩放E00之后的收益与损害，或不适用原因 |
| `analysis/equivalence.csv` | 来源池、C与实际残差的复现检查 |

monitor并非独立验证集；本版没有自动划出额外尾类holdout。原始探测与条件贡献的尺度和组合环境也不同，不能将符号变化单独归因于C方向变换。

## D. 之后需要完整训练时

新入口为 `scripts/run_cliplora_b_problem2.py`。每组单独指定，输出目录含variant、w和有效beta，默认根目录 `output/cifar100_LT/sfra_b_problem2`。

以w=0.5为命令示例；这不是替用户确定最终w，正式比较时四组应统一替换为锁定的参考值：

```bash
python -u scripts/run_cliplora_b_problem2.py --problem2-variant E00 --transfer-tail-weight 0.5 --reference-run references/full10_clientlt --fast-execution-v2
python -u scripts/run_cliplora_b_problem2.py --problem2-variant E10 --transfer-tail-weight 0.5 --reference-run references/full10_clientlt --fast-execution-v2
python -u scripts/run_cliplora_b_problem2.py --problem2-variant E01 --transfer-tail-weight 0.5 --harm-beta 1 --reference-run references/full10_clientlt --fast-execution-v2
python -u scripts/run_cliplora_b_problem2.py --problem2-variant E11 --transfer-tail-weight 0.5 --harm-beta 1 --reference-run references/full10_clientlt --fast-execution-v2
```

这里的 `--reference-run`仍只复用原协议，不加载它的训练后模型。每条命令为一个独立100轮实验，按实际GPU资源和需要的配置分别执行；不必因为入口支持四组就立即全部启动。

原命令加 `--resume`恢复其最后完成的轮次；variant、w、beta、配置必须匹配，不允许拿旧共享B的checkpoint作为改法的普通续训。源码保留旧入口与旧配置的恢复兼容性。

```bash
python scripts/run_cliplora_b_problem2.py --stage pack
```

正式汇总沿用81—100轮。`performance.csv`记录variant、beta、Overall/Tail等既有指标，并在每类结果完整时追加 `last20_many_acc`、`last20_medium_acc`、`last20_few_acc`与各组类别数；读取epoch80—99，对应真实轮次81—100。`frequency_metrics_available=False`表示缺少逐类文件或先验，不伪造这些组别结果。

旧问题1结果是否可作正式E00基线，需要单独核对来源规则、全覆盖并集、采样和数值实现。不能只因单轮等价就保证无需新的完整E00；旧结果仍可作背景参照。

## 本地验证

```bash
python -m unittest discover -s tests -p 'test_sfra*.py'
```

新增测试覆盖实际组权重保持、损害先逐类取正部、E00与旧共享C一致性、C=0两批参照、基础参数冻结、一次残差注入、固定分母移除donor、保存事件身份检查、四组小模型重放、入口/恢复隔离、完整训练和重放汇总，以及81—100轮逐类指标的epoch映射。

本次运行共发现99项测试，98项通过，1项CUDA相关测试因本地CPU环境跳过；其中新增问题2测试13项全部通过。语法检查与 `git diff --check`通过。

通过CPU测试不等于真实GPU精度或论文效果已经验证。下一步运行B节的单事件验证，通过后再按计划运行C节。
