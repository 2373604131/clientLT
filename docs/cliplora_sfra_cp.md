# SFRA-CP：分类收益保持版 A 实验

本版本在原 Full-10 的三步 A 功能修正中，增加单侧 LA 分类损失上升正则。详细公式见 [设计协议](cliplora_sfra_classification_preservation_design.md)。B 迁移模块尚未加入。

## 已实现的协议

- 方法名 `full-cp`；保护强度 lambda=10；新增搜索参数为 `--classification-weight`，即 mu。
- 分类参照固定为本轮普通聚合 A 提案和聚合后的 B，不使用测试集。
- 正则使用包含 CLIP logit scale 和全局 LA 的完整分类 logits，tau=1。
- 复用每本地类别至多 8 张见证及两视图，按原客户端样本权重和本地类别频率计算整体分类损失。即便某个功能单位未激活，分类项仍包含它。
- 先聚合分类损失，再计算单侧平方惩罚；先聚合分类梯度，再计算归一化尺度。
- 首步缓存分类参照和尺度；三步内固定。首步分类正则梯度为零，后两步与功能保持梯度共同参与更新。
- lambda、来源识别、五轮历史、rank=4、三步修正、步长 0.1 和本地训练预算不改。第 1–90 轮刷新 A，第 91–100 轮仅训练 B。
- 记录额外分类梯度成本，不将其描述为零开销。保留原版所有方法的入口及结果目录。

## 启动三组实验

在服务器仓库根目录、已激活原 `clientlt` 环境中执行。每个节点只有一张已分配 GPU 时，不需要额外指定物理 GPU 编号。以下均为前台运行，终端直接显示日志和报错。

建议明确复用已完成 Full-10 的数据划分、顺序与日程。`--reference-run` 要指向服务器上的完整运行目录，不是下载到本地的轻量分析包；不需要从 Full-10 的模型继续训练，新实验仍从原共同初始化开始。

**节点一：mu=0.3**

```bash
python -u scripts/run_cliplora_sfra.py --method full-cp --retention-weight 10 --classification-weight 0.3 --reference-run output/cifar100_LT/sfra_v1/seed42/client-longtail/full/lambda10_protocol42
```

**节点二：mu=1**

```bash
python -u scripts/run_cliplora_sfra.py --method full-cp --retention-weight 10 --classification-weight 1 --reference-run output/cifar100_LT/sfra_v1/seed42/client-longtail/full/lambda10_protocol42
```

**节点三：mu=3**

```bash
python -u scripts/run_cliplora_sfra.py --method full-cp --retention-weight 10 --classification-weight 3 --reference-run output/cifar100_LT/sfra_v1/seed42/client-longtail/full/lambda10_protocol42
```

若旧实验使用自定义输出根目录，只替换上述参考路径。如果服务器只保留了完整 S 目录，可将三条命令的参考路径统一改为 `output/cifar100_LT/la_control/seed42/client-longtail/s/tau1_a1_protocol42`。不要为了绕过缺失参考目录而仅对某一组使用 `--fresh-protocol`。

默认输出分别为：

```text
output/cifar100_LT/sfra_cp/seed42/client-longtail/full-cp/lambda10_mu0.3_protocol42/
output/cifar100_LT/sfra_cp/seed42/client-longtail/full-cp/lambda10_mu1_protocol42/
output/cifar100_LT/sfra_cp/seed42/client-longtail/full-cp/lambda10_mu3_protocol42/
```

不会覆盖 `sfra_v1` 中的 Full、Current、Flat 或 S。已完成的 Full-10 和 S 作为参照，不要求重跑。

只有一个节点一张 GPU 时，三条训练命令按上述顺序用 `&&` 连接即可；前一个报错后不会自动启动下一个。

## 日志与续训

终端每个修正步骤显示 `LA`、`LA_increase`、`cls_penalty`、`|g_func|`、`|g_cls|` 和 mu。首步的分类惩罚及分类梯度为零符合设计；后续是否激活取决于分类损失是否上升。每轮完成显示：

```text
SFRA COMMITTED 100/100: full-cp, lambda=10, mu=1
```

正常完成后存在 `completion.json`。中断后，例如恢复 mu=1：

```bash
python -u scripts/run_cliplora_sfra.py --method full-cp --retention-weight 10 --classification-weight 1 --resume
```

续训重放原保存配置，恢复最后一个完整轮次；若首轮前便失败且没有 checkpoint，应先解决具体报错，不使用此续训命令冒充恢复。使用过自定义 `--output-root` 时，续训也须指定同一路径。

## 汇总与打包

三组在不同节点产生的结果先汇集到同一 `output/cifar100_LT/sfra_cp` 根目录，再执行：

```bash
python -u scripts/run_cliplora_sfra.py --method full-cp --stage pack --reference-run output/cifar100_LT/sfra_v1/seed42/client-longtail/full/lambda10_protocol42
```

输出：

```text
output/cifar100_LT/sfra_cp/analysis/report.md
output/cifar100_LT/sfra_cp_analysis.tar.gz
```

其中参考 Full-10 的汇总分数进入报表；压缩包主要包含新版实验本身，不会打包参考目录的大模型。若各节点不是共享存储，必须先复制完整的新实验子目录或分别打包，汇总命令不会自动收集其他节点。

必须带 `--method full-cp`，否则默认汇总的是旧 `sfra_v1` 根目录。打包不加载模型或使用 GPU。

## 分析文件

- `round_metrics.csv`：原有全部正式指标，主口径仍为第 81–100 轮平均。
- `sfra_rounds.csv`：分类参照、尺度、提交后损失增量、正则、实际激活步数、两项梯度范数和原 A 更新诊断。
- `sfra_rounds/rXXX/correction_steps.csv`：每步输入候选上的两项损失、总目标、两项梯度范数和夹角；提交第三步后的总目标单独记录于轮次汇总。
- `sfra_rounds/rXXX/tokens.npz`：保留原功能记录，新增修正前后逐功能单位的分类损失、逐客户端分类损失、客户端权重及全局单位权重。
- `sfra_costs.csv`：总前向、反向、通信和耗时，以及分类子项计数。分类前向复用了功能前向，不能把分类前向计数再次加到总数上；总反向计数已经计入分类梯度。
- `analysis/correction_steps.csv`：跨运行的三步修正明细。

先看尾类保持是否仍成立，再看 Head/Middle/Overall 是否恢复。分类训练侧损失下降不是测试收益的保证；不以单一 Overall 最大值忽略尾类重新遗忘。

## 本地检查范围

本次不启动训练、不下载数据、不运行 GPU 或冒烟实验；交付前仅检查 Python 语法与参数接线。仓库中已有的小模型测试保留，但本次不执行，其存在不代表本次已经通过运行测试。
