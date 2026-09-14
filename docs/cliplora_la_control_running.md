# LA + 自适应 A/B 额外训练实验（E0–E5）

入口：`scripts/run_cliplora_la_control.py`。所有命令在仓库根目录、已激活的 clientlt 环境中执行。前台运行，错误直接显示；不使用 nohup，不自行分配或占用其它 GPU。

## 已冻结的协议

| 方法 | 损失 | 额外机会 |
|---|---|---|
| e0 | CE | B |
| e1 | CE | 周期 A |
| e2 | 全局训练期 LA | B |
| e3 | 全局训练期 LA | 周期 A |
| e4 | CE | A/B 双分支功能选择 |
| e5 | 全局训练期 LA | A/B 双分支功能选择 |

- rank4、alpha1、scaling0.5、视觉 top3 q/v、固定文本和 logit scale、FP32、30 客户端全参与。
- 100 轮；正常 B 每客户端 3 epochs，SGD lr=.001、momentum=.9、wd=.0005；每客户端重置 optimizer/scheduler，沿用原 gamma=1 的调度。
- 额外机会为第 10、20、…、90 轮，各 1 epoch；额外 B lr=.001，额外 A lr=.001×a_lr_mult；momentum=.9、wd=0。
- 所有训练阶段使用该实验的同一损失。LA 是 CE(raw_logits + tau×log(global_prior))。测试、在线 margin、离线 F 均不加 LA。它不是仓库的 `la_aggregation`。
- 双分支起点、客户端次序、样本顺序、增强随机流相同。h=2 的两轮只计入主路径一次；决策后从 t+h+1 继续。
- root 是第 10 轮正常 B 聚合后、额外训练前的完整全局状态。
- 历史候选仅 root 和已选中的决策终点（h=2 时为 12、22、…、92）；普通中间轮不参加历史最佳竞争。
- `best_valid_anchor` 的 valid 只是功能合格性，不是理论安全保证。
- gate 相对本次决策前固定的旧 anchor 计算。三个条件：全部类别平均 A−B > min_gain；尾类最差五个 A−B 的平均 ≥ −tail_tolerance；尾类最差五个 A−旧anchor 的平均 ≥ −history_tolerance。
- 选中状态相对旧 anchor 通过历史约束，且绝对类别平均 margin J 更高，才更新 best；同分保留旧 anchor。B 也可更新。
- 仅接受 A 会清零失败计数；拒绝 A 就加一，即使 B 更新了 best 也不清零。
- 连续 patience 次拒绝后冻结 A，并恢复 best_valid_anchor。若当前 B 正好是 best，恢复是 no-op；记录 rollback_applied=false。未来额外机会继续训练 B。
- 恢复完整 model state，不倒退逻辑轮次、RNG 或预算；不引入跨客户端 momentum。

## 数据前提：复用旧桥接结果中的协议文件

默认从以下位置读取每种拓扑的 c1：

`output/cifar100_LT/a_refresh_topology_bridge/seed42/<partition>/c1/`

必须保留：`partition_manifest.csv`、`bridge_metadata.json` 和 `protocol/` 下的 probe 清单、eri_protocol.json、full_schedule.json。不需要旧 bridge_dumps 或模型 checkpoint。

训练划分按 raw_sample_id/client_id/local_position 恢复；全局先验只由训练计数求和。离线 probe 仍在训练池外。全局样本、测试集、probe、初始化及日程与来源协议做一致性检查。

如果旧协议来自本机解压目录，命令追加：

`--bridge-root output/a_refresh_topology_bridge_analysis/a_refresh_topology_bridge`

若 CIFAR 数据不在 DATA，追加 `--data-root /absolute/path/to/DATA`。

`--seed` 控制本次模型/训练随机种子；`--protocol-seed` 默认 42，指定旧协议 seed 目录。补训练种子时保持 protocol-seed=42 可固定数据划分；若研究独立划分，先准备对应桥接协议，再指定 protocol-seed。两者都记录，不混为一种重复实验。

## 首轮六个前台命令

各计算节点只有一张分配给你的 GPU 时，无需指定 0/1；继承调度器的 CUDA_VISIBLE_DEVICES。四个节点先启动四个命令，空闲后启动剩余两个。不要在同一张卡上并发启动多项。

Client-LT E2：

```bash
python -u scripts/run_cliplora_la_control.py --method e2 --partition client-longtail
```

Client-LT E3：

```bash
python -u scripts/run_cliplora_la_control.py --method e3 --partition client-longtail
```

Client-LT E5：

```bash
python -u scripts/run_cliplora_la_control.py --method e5 --partition client-longtail
```

Dirichlet E2：

```bash
python -u scripts/run_cliplora_la_control.py --method e2 --partition matched-dirichlet
```

Dirichlet E3：

```bash
python -u scripts/run_cliplora_la_control.py --method e3 --partition matched-dirichlet
```

Dirichlet E5：

```bash
python -u scripts/run_cliplora_la_control.py --method e5 --partition matched-dirichlet
```

新框架的 e0/e1/e4 也已支持，替换 --method 即可。旧 C1/C2 仅作为历史参考：新实验把所有评价的 RNG 隔离，不能声称与旧随机流位级一致。

## 可调参数及成对运行

| 参数 | 默认 | 优先敏感性 |
|---|---:|---|
| --a-lr-mult | 1 | 0.3，然后 0.1；E3/E5 成对 |
| --la-tau | 1 | 0.5；E2/E3/E5 对应 |
| --tail-tolerance | .002 | .001 / .005 |
| --history-tolerance | 未指定 | 自动继承 tail-tolerance；首轮不独立搜索 |
| --min-gain | .0001 | 0 / .0005 |
| --lookahead-rounds | 2 | 0 / 1；必须 0≤h<10 |
| --patience | 2 | 1 / 3 |

优先成对的小 A 学习率：

```bash
python -u scripts/run_cliplora_la_control.py --method e3 --partition client-longtail --a-lr-mult 0.3
```

```bash
python -u scripts/run_cliplora_la_control.py --method e5 --partition client-longtail --a-lr-mult 0.3
```

只看即时收益的对照：

```bash
python -u scripts/run_cliplora_la_control.py --method e5 --partition client-longtail --lookahead-rounds 0
```

每个配置独立输出，默认目录：

`output/cifar100_LT/la_control/seed42/<partition>/<method>/<configuration_id>/`

相同目录不覆盖，失败直接报错；重新训练使用新的 --output-root。首版没有自动训练续跑入口，避免把未完成的双分支事务混入正式结果。完整状态与 last checkpoint 会保存；`load_global_checkpoint()` 可解码完整模型用于后续诊断，不代表恢复了整套运行事务。

## 在线反馈和测试

- training-side functional feedback：全部客户端训练图像的固定评估视图；归一化余弦 margin，无 CLIP scale、无 LA。逐客户端上传每类 sum/count，在服务器按类合并，不乘 q。
- 不切分新的 validation，不使用官方测试决策；这是训练侧反馈，存在拟合风险。
- 模拟统计上传会泄露类别存在性；没有实现安全聚合或 DP。
- official test：只对正式路径初始状态及每个逻辑轮末调用，共 101 次。分支中暂存状态，选择后才补记选中路径的对应轮次；未选中分支从不调用 test。
- 所有评价保存/恢复完整模型 state、train/eval mode 和 Python/NumPy/Torch/CUDA RNG。临时训练没有正式 TensorBoard 写入。
- 默认 h=2 若九次均尝试，最多 37 次在线 feedback，不是只有十次；h=0 即时/终点评价复用。

## 训练后离线归因

默认 --stage train **不会自动开始高成本归因**。以下命令只读取已完成训练的状态文件，不重训：

```bash
python -u scripts/run_cliplora_la_control.py --stage analyze --method e5 --partition client-longtail
```

其它组相应替换 method/partition。非默认配置归因时必须带上同样的参数，例如 `--a-lr-mult 0.3`。

默认归因额外阶段、两条分支全部 look-ahead B，以及指定正常轮（包括候选前后轮次）。需要全部正常轮：

```bash
python -u scripts/run_cliplora_la_control.py --stage analyze --method e5 --partition client-longtail --normal-rounds all
```

如数值积分闭合失败，训练结果不受影响，使用 `--quadrature-segments 2` 重算归因。沿用 F=z_c−logsumexp(z_not_c)；W/H/D/R 已含 q，不二次加权。未选中分支单独标记；恢复为服务器事件，只计算其直接 ΔF，不冒充客户端归因。

## 汇总

所有节点的各配置子目录复制到同一个 la_control 根目录后：

```bash
python -u scripts/run_cliplora_la_control.py --stage summary
```

生成 `analysis/report.md`、performance/comparisons/costs/decisions/phase_mechanisms CSV、协议核对及曲线。即使尚未离线归因，也可先汇总准确率和 controller 行为。主指标为第 81–100 轮平均；peak 仅作描述，不用测试最高点选 checkpoint 或配置。

## 必看日志与成本

- control_decisions.csv：即时/延迟 G_all、G_tail、G_hist，P_all/P_tail/P_hist，三个条件，A 接受数，失败计数，锚点变更，冻结及实际回退。
- control_class_metrics.csv：100 类 g、历史差异、root 差异、两套 Bottom5 IDs/标记；decision 表包含均值与标准差。
- history_best.csv / restore_events.csv：决策锚点竞争和恢复（包括 no-op）。
- event_manifest.csv / events/：两条分支各阶段的前后状态、上传、哈希、范数与 committed 标记。
- budget.csv：主日程与未选中分支分开；evaluation_budget.csv：功能与官方测试前向成本。
- checkpoints/：一次完整 base_model.pt + 每个锚点所有变化 tensor 的精确覆盖；不是只保存两个 LoRA 张量。恢复时组合为完整 model_state_dict。
- completion.json：100 轮结束及预算核对。normal=105600、extra=3168；分支额外步数=尝试次数×(352+h×1056)。h=2 最多 22176 步；前向评价、I/O 和通信另计。
- upload/downlink 字节是协议通信量模型，不是实际网络测速；downlink 假设每次发全部 A/B、客户端缓存冻结骨干。

回退不返还已消耗预算，不把被历史恢复抹去的学习计算当免费；主日程预算是已执行并选中的操作数量，不是最终参数的祖先路径长度。

## 打包给分析用（保留服务器原件）

```bash
tar --exclude='*/checkpoints' --exclude='*/events/*/state.pt' -czf la_control_analysis.tar.gz -C output/cifar100_LT la_control
```

这个包保留原始准确率、gate/预算、事件元数据和已完成的离线归因，但不含大模型状态。服务器原始 events/checkpoints 不要删除，补归因仍然需要它们。

## 解释边界

E5 应优于 E2，或保持 Tail 的同时提高 Overall/Non-tail；仅优于失败的 E3 不足以证明增量价值。必须同时看 accepted_A_decisions、freeze_round、实际 rollback 次数以及 D−R、W−H。全部拒绝可能意味着固定 A 更合适；gate 不保证长期安全。首轮一个种子不能宣称显著性；等计算量 E2 对照与多种子属于后续正式验证，不在首轮偷偷增加。

实现阶段仅做静态语法及差异检查，没有启动训练、冒烟实验或 GPU 归因。
