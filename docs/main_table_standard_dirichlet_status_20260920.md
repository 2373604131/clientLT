# 截图主表的普通狄利克雷对照：完成状态与启动方式

核查日期：2026-09-20。范围为截图中的8种方法，不是此前44行历史大表。只认可完整100轮结果。本机没有启动训练；以下状态表示当前仓库中可核验的结果，不代表服务器上未回传的运行状态。

正确划分统一为 `noniid-labeldir-fine`、β=0.5、split_seed=42，同一CIFAR100-LT全局训练池、imb_factor=0.01、30客户端全参与、100轮、正常本地3 epoch。客户端容量独立生成，不能复制Client-LT容量；旧matched-Dirichlet不能替代普通狄利克雷。

| 具体方法 | 普通狄利克雷完成状态 | Overall | Tail20 | 待启动参数 |
|---|---|---:|---:|---|
| CAPT：截图中的预算对齐配置，无LoRA | 未找到同配置的完整对照 | — | — | `capt` |
| 无LA，固定A，正常训练B＋9次额外B | 未找到完成结果 | — | — | `e0` |
| 无LA，正常只训练B＋9次额外A | 未找到完成结果 | — | — | `e1` |
| 有LA，固定A，正常训练B＋9次额外B | 只有旧固定容量对照，普通划分未找到完成结果 | — | — | `e2` |
| 有LA，正常只训练B＋9次额外A | 已完成100轮 | 69.6035 | 72.0175 | 不用补跑 |
| 有LA，功能控制选择A/B候选并前瞻 | 只有旧固定容量对照，普通划分未找到完成结果 | — | — | `e5` |
| 有LA，日常AB联合更新＋9次额外A | 已完成100轮 | 72.2925 | 71.3600 | 不用补跑 |
| 有LA，正常只训练B＋前90轮每轮更新A | 已完成100轮 | 71.0140 | 72.2975 | 不用补跑 |

精度为第81–100轮均值（%）。已完成三组的原始结果位于 `output/la_control_standard_dirichlet_e3_j_s_analysis/la_control/seed42/noniid-labeldir-fine/` 下的三个方法目录；已核对 `control_config.json`、`completion.json`、`protocol/partition_protocol.json` 及逐轮CSV。其协议明确记录 `reuse_clientlt_capacities=false`。

CAPT截图参照来源为 `output/v2_capt_analysis/output/cifar100_LT/v2_matched/seed42/capt/command.json`。它使用cluster、FP32、每客户端重置优化器、每轮全局聚合，`capt_reset_global_before_client` 保持默认False。现有global-start版本改变了模型起点，其他CAPT历史包还包含不同seed、部分参与、MAB或旧划分，均不能作为这一行的同配置对照。这里的 `capt_matched_v2=True` 表示预算对齐/优化器重置，不是matched-Dirichlet划分。

## 启动缺失的5组

把代码同步到GPU服务器，在仓库根目录激活原训练环境后执行。启动器继承节点的CUDA_VISIBLE_DEVICES，不自行指定物理GPU；前台执行，出错立即停止。

先只打印命令，不训练、不创建结果目录：

```bash
python scripts/run_main_table_standard_dirichlet.py --dry-run
```

单GPU顺序完成缺少的5组：

```bash
python -u scripts/run_main_table_standard_dirichlet.py --data-root DATA
```

如果分别分配5个单GPU节点，每个节点单独执行一条：

```bash
python -u scripts/run_main_table_standard_dirichlet.py --experiments e0 --data-root DATA
python -u scripts/run_main_table_standard_dirichlet.py --experiments e1 --data-root DATA
python -u scripts/run_main_table_standard_dirichlet.py --experiments e2 --data-root DATA
python -u scripts/run_main_table_standard_dirichlet.py --experiments e5 --data-root DATA
python -u scripts/run_main_table_standard_dirichlet.py --experiments capt --data-root DATA
```

`e0/e1/e2/e5`仅是命令所需参数：依次对应上表无LA额外B、无LA额外A、有LA额外B、功能控制。它们直接调用现有LA-control训练入口。普通Dirichlet模式独立生成划分，不要求先完成旧CE桥接实验，也无需补跑旧桥接的两组方法。

CAPT复用 `run_cliplora_v2.py` 中截图对应的命令构造，只改划分为普通Dirichlet并使用独立输出目录。保持原 `output/cifar100_LT/v2_matched/full_schedule_seed42.json` 日程路径；文件缺失时底层按原种子规则生成，也可通过 `--capt-schedule-file` 指定原日程文件。不得改用global-start入口来代替截图CAPT。

## 输出与结果回收

- 四组LoRA：`output/cifar100_LT/la_control/seed42/noniid-labeldir-fine/` 下对应方法及参数目录。
- CAPT：`output/cifar100_LT/capt_matched_standard_dirichlet/seed42/capt/`。
- 启动器默认只覆盖缺失的5种方法，不运行已完成的3组，也不启动离线GPU归因。
- 非空目标目录会报错。只剩部分组时使用 `--experiments` 选择未跑的；中断目录不要当作完整结果。另起完整重跑时用 `--la-output-root` 或 `--capt-output-root` 指定新目录。
- 完成后保留逐轮指标、逐类准确率、配置、划分清单及完成记录。LoRA应有completed_round=100；CAPT检查epoch=0…99全部存在。汇总统一取第81–100轮，Head20/Middle60由相同窗口逐类文件计算。
- 普通Dirichlet按实际客户端样本量计算步数：无控制器的9次额外更新路径为109386步；密集90次路径为138060步。功能控制另计候选与前瞻成本，不强行匹配Client-LT容量或步数。

原 `run_cliplora_standard_dirichlet_reruns.py` 默认是另一份七组补跑清单，含已完成方法与旧CE桥接，且没有截图的无LA两组和CAPT。本次应使用新的五组入口。
