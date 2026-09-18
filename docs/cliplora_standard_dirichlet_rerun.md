# A/B框架：普通Dirichlet补跑说明

日期：2026-09-18。

## 协议与范围

后续普通Dirichlet统一使用 `noniid-labeldir-fine`、beta=0.5、split_seed=42：逐细类抽Dirichlet比例，再以multinomial分配样本；不复制Client-LT客户端容量。每客户端最少10张训练样本的重试规则沿用仓库已有实现。

继续保持全局CIFAR100-LT样本池、IF=.01、30客户端全参与、100轮、rank4、top3 Q/V、LA tau=1（E2/E3/E5/J/S）、正常3个local epochs及原刷新时机。每次生成独立协议目录；仅复用公共probe和参与日程，不复用Client-LT partition manifest或其n_k。

客户端n_k、FedAvg q_k=n_k/N和由batch取整决定的步数按新划分重新计算。相同local epochs不是相同optimizer steps；不为了凑352步重新约束客户端容量。

已用现有纯划分函数核查默认seed42：客户端训练样本量范围229–500，全客户端一个epoch为354步；E2/E3/J主路径为109386步，S主路径138060步。E5未选分支预算根据实际决策次数另记。这个检查只分配标签索引，没有运行模型。

七组需要补新对照：CE桥接C1/C2，LA系列E2/E3/E5/J/S。已有Matched结果保留，不能改名。Client-LT结果不因本次划分修订强制重跑；跨新旧代码比较会标记为描述性。E0/E1的普通Dirichlet此前未跑，是可选新增对照而非本轮七组替换。

总表：`output/ab_framework_partition_review_20260918/report.md`；完整CSV含Head20/Middle60和源路径：`all_results.csv`。

## 同步代码

将本次修改的仓库同步到服务器，至少包含：

- `scripts/cliplora_fresh_protocol.py`（新增）；
- `scripts/run_cliplora_standard_dirichlet_reruns.py`（新增）；
- `scripts/run_cliplora_la_control.py`；
- `scripts/run_cliplora_topology_bridge.py`；
- `scripts/run_cliplora_a_refresh.py`；
- `scripts/analyze_cliplora_topology_bridge.py`；
- `utils/cliplora_la_control.py`；
- `tools/la_control/summary.py`；
- `tools/a_refresh_bridge/summary.py`。

不需要修改 `utils/datasplit.py`；直接使用现有fine-class Dirichlet实现。不要只复制启动脚本而漏掉运行时预算修改。

## 一张GPU：前台依次运行全部七组

在分配到GPU的计算节点、仓库根目录、clientlt环境运行：

```bash
python -u scripts/run_cliplora_standard_dirichlet_reruns.py
```

顺序为E3、J、S、E2、E5、C1、C2。打印 `[1/7] START E3` 等进度，每个子实验的输出/错误直接显示在终端，失败立即停止，不使用nohup或后台启动。继承调度器设置的CUDA_VISIBLE_DEVICES，不擅自指定物理GPU编号。

先只跑最关键三组：

```bash
python -u scripts/run_cliplora_standard_dirichlet_reruns.py --experiments e3 j s
```

之后补其余四组：

```bash
python -u scripts/run_cliplora_standard_dirichlet_reruns.py --experiments e2 e5 c1 c2
```

若DATA不在默认位置，追加 `--data-root /实际数据目录`。此参数也会传给每个子实验。

包装器不自动跳过已有目录；重新调用时用 `--experiments` 只选择未开始的实验。中断产生的部分目录保留，不删除；需要完整重跑时使用下方单条命令并加新的 `--output-root`。本轮未增加训练断点恢复功能。

## 多节点：七条独立命令

E3：

```bash
python -u scripts/run_cliplora_la_control.py --method e3 --partition noniid-labeldir-fine --dirichlet-beta 0.5
```

J：

```bash
python -u scripts/run_cliplora_la_control.py --method j --partition noniid-labeldir-fine --dirichlet-beta 0.5
```

S：

```bash
python -u scripts/run_cliplora_la_control.py --method s --partition noniid-labeldir-fine --dirichlet-beta 0.5
```

E2：

```bash
python -u scripts/run_cliplora_la_control.py --method e2 --partition noniid-labeldir-fine --dirichlet-beta 0.5
```

E5：

```bash
python -u scripts/run_cliplora_la_control.py --method e5 --partition noniid-labeldir-fine --dirichlet-beta 0.5
```

旧CE桥接C1：

```bash
python -u scripts/run_cliplora_topology_bridge.py --stage train --method c1 --partition noniid-labeldir-fine --dirichlet-beta 0.5
```

旧CE桥接C2：

```bash
python -u scripts/run_cliplora_topology_bridge.py --stage train --method c2 --partition noniid-labeldir-fine --dirichlet-beta 0.5
```

C1/C2明确使用 `--stage train`，不会训练后自动启动耗时GPU归因。新训练拒绝matched-dirichlet，但历史 `--stage analyze` 仍然支持它。

## 输出与完整性

LA系列：

```text
output/cifar100_LT/la_control/seed42/noniid-labeldir-fine/e3/tau1_a1_protocol42_beta0.5/
output/cifar100_LT/la_control/seed42/noniid-labeldir-fine/j/tau1_a1_protocol42_beta0.5/
output/cifar100_LT/la_control/seed42/noniid-labeldir-fine/s/tau1_a1_protocol42_beta0.5/
output/cifar100_LT/la_control/seed42/noniid-labeldir-fine/e2/tau1_a1_protocol42_beta0.5/
output/cifar100_LT/la_control/seed42/noniid-labeldir-fine/e5/tau1_a1_h2_t0.002_hist0.002_gain0.0001_p2_protocol42_beta0.5/
```

CE桥接：

```text
output/cifar100_LT/a_refresh_topology_bridge/seed42/noniid-labeldir-fine/c1/
output/cifar100_LT/a_refresh_topology_bridge/seed42/noniid-labeldir-fine/c2/
```

旧client-longtail和matched-dirichlet目录均不覆盖。fine版LA不需要先完成fine版桥接C1，也不会从旧Client-LT恢复划分。存在旧桥接时，只使用其公共日程和图像/probe/初始化指纹作为参照；不存在时独立构建协议。

保存 `protocol/partition_protocol.json`、实际 `partition_manifest.csv`、`bridge_metadata.json`，便于核查到底运行了什么。LA `control_config.json` 新增 `steps_per_epoch` 与实际预算；结束断言与汇总预算均动态计算。

## 汇总与归因

```bash
python -u scripts/run_cliplora_la_control.py --stage summary
```

同一根目录下的旧CLT/Matched与新fine结果分别列出；topology_gaps.csv标注具体Dirichlet类型、客户端容量是否相同、步数是否相同及代码哈希是否相同，不再因fine版n_k不同而丢弃比较。

桥接四格汇总需要旧Client-LT C1/C2与新fine C1/C2均保留在原bridge根目录：

```bash
python -u scripts/run_cliplora_topology_bridge.py --stage summary --dirichlet-partition noniid-labeldir-fine
```

其报告位于 `output/cifar100_LT/a_refresh_topology_bridge/analysis/seed42/noniid-labeldir-fine/report.md`，不覆盖旧Matched报告。旧Matched报告复算时显式加 `--dirichlet-partition matched-dirichlet`。

LA离线归因示例（需要原事件state.pt和GPU，不是重训）：

```bash
python -u scripts/run_cliplora_la_control.py --stage analyze --method s --partition noniid-labeldir-fine --dirichlet-beta 0.5
```

桥接归因示例：

```bash
python -u scripts/run_cliplora_topology_bridge.py --stage analyze --method c2 --partition noniid-labeldir-fine
```

## 打包结果

保留服务器大模型与事件张量，打包轻量结果：

```bash
tar --exclude='*/checkpoints' --exclude='*/events/*/state.pt' --exclude='*/bridge_dumps/*/*/state.pt' -czf ab_standard_dirichlet_analysis.tar.gz -C output/cifar100_LT la_control a_refresh_topology_bridge
```

上式包含旧CLT与Matched，方便三种划分分列比较；只是打包，并不删除服务器文件。需要继续做归因时，务必保留未打包的原始state.pt。
