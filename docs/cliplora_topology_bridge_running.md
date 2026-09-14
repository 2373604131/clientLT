# 四个单 GPU 节点：C1/C2 topology bridge 启动说明

## 已实现的流程

每个节点独立运行一格：Client-LT C1、Client-LT C2、matched Dirichlet C1、matched Dirichlet C2。默认先训练 100 轮，再在同一个节点/同一张 GPU 上进行该格离线归因。

沿用已完成 C1/C2 的 rank4、FP32、top3 Q/V、正常 3 epochs、额外 1 epoch、lr=0.001、10–90 轮刷新等设置。没有加入新方法或调整训练规则。

正常 B 聚合、额外 B/A 刷新独立保存；恢复离线模型时加载完整真实 A/B，并只对该阶段活动因子求导。独立 train-only probe 不进入训练。没有复用会混淆阶段的旧 ERI dump 开关。

代码仅做了语法/静态检查，没有执行训练、GPU 冒烟测试或真实数据归因验证。

## 1. 在四个节点上分别启动

先把**整个更新后的仓库**同步到四个节点，进入仓库根目录，激活现有可运行 ClipLora 的环境，例如 `conda activate clientLT`（以实际环境名为准）。数据默认在 `DATA`，CLIP 权重使用原来的缓存。

四条命令分别放到四个节点的前台终端，不是在同一个节点上一起执行：

节点一，Client-LT C1：

```bash
python -u scripts/run_cliplora_topology_bridge.py --partition client-longtail --method c1
```

节点二，Client-LT C2：

```bash
python -u scripts/run_cliplora_topology_bridge.py --partition client-longtail --method c2
```

节点三，matched Dirichlet C1：

```bash
python -u scripts/run_cliplora_topology_bridge.py --partition matched-dirichlet --method c1
```

节点四，matched Dirichlet C2：

```bash
python -u scripts/run_cliplora_topology_bridge.py --partition matched-dirichlet --method c2
```

不使用 nohup、后台 `&` 或丢弃输出；子进程继承终端，报错直接显示并以非零状态退出。没有四卡分布式训练，也不要求节点之间通信。

脚本不覆盖 `CUDA_VISIBLE_DEVICES`。每个节点只有一张被分配的可见卡时，自动使用该卡，设备在各自进程中都可以叫 cuda:0。不要把四个节点误写成同一节点的 GPU 0/1/2/3。

如果是未做 GPU 隔离的机器，需由你在该节点命令前加 `CUDA_VISIBLE_DEVICES=<该节点被分配的物理卡号>`；不要覆盖调度系统已经设置的 GPU UUID/编号。

数据位于其他目录时，在每条命令后加 `--data-root /实际数据路径`。单独指定输出根可用 `--output-root /实际输出路径`，四个节点可以使用各自本地路径。

## 2. 协议文件不需要四节点同时写入同一目录

每格会在自己的 `protocol/` 创建确定性副本，包括相同 probe、相同调度、bridge protocol。没有共享写入竞争，适合共享文件系统和四个完全独立文件系统两种情况。

默认优先读取本仓库已有的 `output/cifar100_LT/v2_matched/full_schedule_seed42.json`，然后复制到该格；不存在时按原来的 NumPy default_rng(seed42) 全参与顺序生成。也可四格都指定同一份 `--schedule-file /path/schedule.json`。

若各节点已有的历史 schedule 内容不同，最终匹配审计会拒绝把它们当成严格对照；因此有自定义历史调度时，事先同步同一份文件。

## 3. 输出位置与完成标志

默认根目录：

```text
output/cifar100_LT/a_refresh_topology_bridge/
  seed42/
    client-longtail/c1/
    client-longtail/c2/
    matched-dirichlet/c1/
    matched-dirichlet/c2/
```

每格包含：

- `round_metrics.csv`、逐类准确率、全部原有 A-refresh 指标。
- `partition_manifest.csv`：原始图像 ID、类别、客户端和本地顺序。
- `bridge_metadata.json`：真实配置、初始化/冻结模型/代码/调度/数据指纹及环境。
- `protocol/`：固定 probe 和调度副本。
- `bridge_dumps/round_*/normal_B/state.pt`：100 轮正常更新。
- `bridge_dumps/round_*/extra_B/state.pt` 或 `refresh_A/state.pt`：9 次额外更新。
- `analysis/phase_client_effects.csv`、`phase_class_budgets.csv`：有符号功能贡献。
- `analysis/attribution_validity.csv`：功能闭合误差和真实端点误差。
- `analysis/attribution_summary.json`：该格归因的完成状态和有效性。

训练完成看 `a_refresh_progress.json` 的 completed_round=100；离线归因完成看 `analysis/attribution_summary.json` 的 valid=true。两者不是同一个阶段。

为精确重构原始 float32 参数平均，normal_B 额外保留了客户端实际 B 参数，不仅保存舍入后的 delta。存储比设计中仅保存 delta 的估算稍大，完整每格的 LoRA 事件约 0.5 GiB 原始载荷；另有日志、CSV、pickle 元数据及最终 checkpoint。建议每格预留至少 1 GiB，不含数据和 backbone 缓存。

## 4. 训练与归因可以分开执行

只训练，不自动归因：

```bash
python -u scripts/run_cliplora_topology_bridge.py --stage train --partition matched-dirichlet --method c2
```

训练已完成，单独做/重做归因：

```bash
python -u scripts/run_cliplora_topology_bridge.py --stage analyze --partition matched-dirichlet --method c2
```

第一批默认对 7 个正常 B 事件（10、20、40、60、80、90、100）和全部 9 个刷新事件归因。所有 109 个事件都已保存，后续可以加深离线分析而无需重新训练。

补充刷新后一轮：

```bash
python -u scripts/run_cliplora_topology_bridge.py --stage analyze --partition matched-dirichlet --method c2 --normal-rounds 10,11,20,21,30,31,40,41,50,51,60,61,70,71,80,81,90,91,100
```

全 100 轮正常 B 归因：

```bash
python -u scripts/run_cliplora_topology_bridge.py --stage analyze --partition matched-dirichlet --method c2 --normal-rounds all
```

这些命令覆盖该格的离线汇总 CSV，不改训练数据；应对四格使用相同的 normal-rounds，才能进行匹配的机制对照。归因当前会重新计算所选事件，不会误把旧设置的计算结果当成新结果。

若 8 点路径积分未通过闭合检查，先查看 validity CSV。可用两段各 8 点重新计算：

```bash
python -u scripts/run_cliplora_topology_bridge.py --stage analyze --partition matched-dirichlet --method c2 --quadrature-segments 2
```

这不会重训。若错误来自矩阵端点重构、缺失数据或模型指纹不同，增加积分点不能修复，应直接检查对应数据。

新训练拒绝写入非空实验目录，避免 CSV 混入两次运行。未完成的训练需要保留原目录并选择新的 `--output-root` 重跑；此最小 bridge 入口不提供自动训练断点拼接。已经训练完成则使用 `--stage analyze`，不要再次使用默认 run。

## 5. 四组完成后汇总

共享存储时，在任意一个装有项目依赖的节点执行一次；独立存储时，先把四格目录按原结构汇集到同一个输出根。

```bash
python -u scripts/run_cliplora_topology_bridge.py --stage summary
```

汇总不用 GPU 前向/反向，但需要项目现有 Python 依赖（包括 PyTorch、NumPy、PyYAML、Matplotlib）。

报告位于：

```text
output/cifar100_LT/a_refresh_topology_bridge/analysis/seed42/report.md
```

自动核对固定边际、样本身份、配置/代码/模型指纹、训练预算、109 个事件及状态连续性；计算 final/last20/last10 的 G_B、G_A、delta_G、刷新即时交互、后续 B 变化、逐类结构相关性，以及有效且匹配的 W/H/D/R/access/rho/mu。

四格都训练完但归因尚未齐全时，也能先生成性能报告；`summary.json` 会明确标记 mechanism_ready=false，不能把它当成完整机制结论。

若只有部分格归因失效，原始归因表仍保留，但不会生成声称有效的四格机制汇总。稀疏正常 B 事件的和明确标为 observed_event_sum，不当成全程累计；9 次刷新则完整覆盖。

## 6. 从四个节点打包回来

在各自仓库根目录分别执行对应命令，保存的相对路径可以直接合并：

```bash
tar -czf bridge_clientlt_c1_seed42.tar.gz -C output/cifar100_LT/a_refresh_topology_bridge seed42/client-longtail/c1
```

```bash
tar -czf bridge_clientlt_c2_seed42.tar.gz -C output/cifar100_LT/a_refresh_topology_bridge seed42/client-longtail/c2
```

```bash
tar -czf bridge_matched_c1_seed42.tar.gz -C output/cifar100_LT/a_refresh_topology_bridge seed42/matched-dirichlet/c1
```

```bash
tar -czf bridge_matched_c2_seed42.tar.gz -C output/cifar100_LT/a_refresh_topology_bridge seed42/matched-dirichlet/c2
```

完整归因建议保留 state.pt、protocol、manifest 和各 CSV，不要只带 log.txt 或最终 checkpoint。四个压缩包解压到同一个 `output/cifar100_LT/a_refresh_topology_bridge` 后即可运行 summary。
