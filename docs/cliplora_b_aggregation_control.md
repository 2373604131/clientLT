# B 迁移表达份额对照：八轮客户端等权

本次只实现两组新对照，不改 C 的学习方式、不实现共同 C，也不做类别数量或倒数加权。

| 组别 | A | 本地 B 迁移 | B 聚合 |
|---|---|---|---|
| Uniform8，无迁移 | Full-CP，lambda=10、mu=1 | 关闭 | 第 30、40、50、60、70、80、90、100 轮客户端等权 |
| Uniform8，原迁移 | 完全相同 | 原 Donor + 本地 C；C lr=0.3 | 完全相同 |

其他 B 轮次仍按原样本量聚合。A 提案、来源诊断中的样本权重、CP 分类权重始终保留原设置。
当前 30 个客户端全参与，因此指定八轮的权重均为 1/30；不是只在持有尾类的 22 个接收端之间平均。

## 聚合的准确含义

无迁移组聚合所有客户端正常训练后的整份 B；迁移组聚合所有客户端的 `B_local + R`（非接收端的 R=0）。
普通本地更新与迁移补充一起使用相同等权系数，不是保留普通 B 的样本权重、只对 R 等权。

迁移流程不改：仍在各接收端自己的本地 B 上筛选 Donor、执行两个 Adam 校准步骤；C 每次零初始化，rank 从 B 读取（本协议 r=4）。
探测步长 0.1、C 正则 0.001、LA tau=1、样本及 RNG 规则均沿用旧版。
先完成当轮 B 聚合，再执行原 A 方法；第 100 轮有 B 聚合干预和迁移（若启用），没有 A 更新。

这两组用于区分简单调权收益与迁移的附加收益，不能把新聚合本身的提升全部归因于 C。
已有同协议 A-only 和 C lr=0.3 的样本量聚合结果可作为另外两格，不强制重跑。

## 启动

在两张 GPU 的独立终端或两个单卡节点上，从更新后的仓库根目录前台执行。
若原 A 实验仍在运行，在另一份代码目录更新并运行本次对照，不要中途覆盖原 A 实验正在使用的源码。

沿用此前服务器上的 `references/full10_clientlt`。它只提供划分和训练日程；两组均从头训练，不加载参考模型权重。
如果参考目录在别处，替换该参数即可；不直接改用 fresh protocol，以免与旧结果换了划分。

无迁移：

```bash
python -u scripts/run_cliplora_sfra.py --method full-cp --retention-weight 10 --classification-weight 1 --b-aggregation uniform-transfer-rounds --reference-run references/full10_clientlt
```

原 C 迁移：

```bash
python -u scripts/run_cliplora_sfra.py --method full-cp --retention-weight 10 --classification-weight 1 --b-aggregation uniform-transfer-rounds --b-transfer --transfer-lr 0.3 --reference-run references/full10_clientlt
```

同一台多卡服务器可在两条命令前分别加 `CUDA_VISIBLE_DEVICES=4` 与 `CUDA_VISIBLE_DEVICES=5`，按实际空闲卡修改。
中断后，在对应原命令末尾加 `--resume`；自定义输出目录时保留同一个 `--output-root`。
续训使用该目录保存的原命令及最近完成轮次，不能借续训把旧样本加权实验改成等权实验。

## 输出与收集

默认独立输出根目录：`output/cifar100_LT/sfra_b_aggregation`。

```text
seed42/client-longtail/full-cp/
  lambda10_mu1_protocol42_bagg_uniform8/
  lambda10_mu1_protocol42_b_lr0.3_probe0.1_reg0.001_bagg_uniform8/
```

不传 `--b-aggregation` 时，旧命令、旧结果路径和旧聚合行为保持不变。

两组结果集中到同一个根目录后，汇总并打包：

```bash
python -u scripts/run_cliplora_sfra.py --stage pack --output-root output/cifar100_LT/sfra_b_aggregation
```

输出 `output/cifar100_LT/sfra_b_aggregation_analysis.tar.gz`。分析包不含恢复训练所需的大型检查点。

记录包括：

- `sfra_config.json`：八轮聚合协议和 B 迁移开关；也随检查点保存。
- `events/*/event.json`、`event_manifest.csv`：实际 B/A 聚合权重和冻结因子变化。
- `progress.json`：完成轮次及已完成的等权 B 轮次数。
- `analysis/aggregation_weights.csv`：汇总实际客户端权重，核对八轮等权、其余轮及 A 原权重。
- `analysis/performance.csv`：独立标记聚合规则和迁移开关，主指标仍为第 81–100 轮均值。
- 迁移组原诊断：`ordinary_global_B` 使用当轮实际等权规则计算无迁移参照，`transferred_global_B` 为同规则下带迁移的 B；因此两者只差 R。
- `receiver_summary.csv` 中 `sample_weight` 仍指原样本权重，新增 `aggregation_weight` 表示本次实际 B 权重；全局补充范数同样按实际权重计算。

关注两个差值：新聚合下“有迁移减无迁移”，以及它相对于旧聚合下迁移增量的变化。同时检查 Overall、Head/Middle、Tail 和后期回落；不能只看 Tail 提高就认定所有代价都值得。
