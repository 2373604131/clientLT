# 方法 B 新版：同一共享模型上的联合 C 迁移

## 本次实现的边界

只实现刚确定的低计算预算版本：**保留旧版接收端、两批校准图片、每端本地损失和两个反馈步骤，只将 donor 探测和 C 学习放到真正的共享模型上。**

不增加无尾类客户端的分类反馈，不修改 A、LA、普通本地训练、普通 FedAvg、数据划分或迁移轮次。旧入口 `run_cliplora_b_transfer.py` 仍运行原来的本地 C 方法。

本次尚未实现标量 C 或额外 B 训练的消融入口，也没有新增短分支检查点实验入口；下面的训练命令执行完整 100 轮。已有实验不需要重跑。

## 固定协议

| 项目 | 新版设置 |
|---|---|
| A 底座 | Full-CP，λ=10，μ=1；原始 A 更新及修正不变 |
| 日常 B | 每客户端三个 epoch，LA τ=1，按样本量 FedAvg |
| LoRA | rank=4，最后三个视觉 block 的 q/v 六个模块；C 的实际形状从 B 读取 |
| 迁移时机 | 第 30、40、50、60、70、80、90、100 轮 |
| 接收端 | 旧版所有持有 bottom20 尾类的客户端，不扩大到全体客户端 |
| donor 候选 | 其他客户端缺少接收端至少一个目标尾类；仅在缺少的类别上判断正向 LA 收益 |
| 探测 | 普通共享 B + 0.1 × 原始 donor ΔB，每个 donor 单独探测 |
| donor 集合 | 各接收端通过筛选的 donor 并集；两步内不重新筛选 |
| C | 每个 donor、每个模块一个共享 r×r 全矩阵，每次迁移零初始化 |
| 优化 | 共享 Adam，默认 lr=0.3，betas=(0.9,0.999)，eps=1e-8，weight_decay=0 |
| 步数 | 两次联合梯度更新；每步遍历原接收端各自对应的校准 batch |
| 正则 | 0.001 × 所有 C 的平方和 / (donor 数 × 模块数)，每个全局步骤只计算一次 |
| 提交 | 第二步结束后直接提交；不按测试或训练指标挑选步骤、门控或回退 |

### 校准数据及权重：严格复用旧版

调用旧版 `calibration_batches`，种子、轮次及客户端 ID 的规则完全相同：混合客户端每步最多 16 张尾类 + 16 张非尾类，只有尾类的客户端最多 32 张尾类；样本少则使用实际数量。

每个客户端的损失也复用旧版：有两组时为 `0.5 * mean(LA_tail) + 0.5 * mean(LA_non_tail)`，只有尾类则用尾类均值。共享目标是**所有原接收端本地损失的等权平均**，不是先在全部客户端内分别合并尾类与非尾类再进行 50:50 加权。这样不会悄悄改变只有尾类客户端的权重。

当 donor 并集非空时，所有原接收端都反馈，即使某个接收端本轮没有独立筛出 donor。旧版在这种情况下会跳过该接收端的本地 C 校准，因此两版实际反向图片数可能略有差别；日志记录实际数量，不声称预算逐图片绝对相同。没有任何合格 donor 时记录一个无迁移事件。

缺类筛选针对某条“接收端—类别—donor”关系，并不要求 donor 缺少所有尾类。一个接收端自身也可能作为其他接收端的 donor 进入并集；联合校准时所有人都看到完整并集，不能将其解释成每个客户端只吸收与自身完全无关的来源。

数据均为本地训练集，探测与校准允许重叠，沿用确定性视图、原全局 LA 先验和全部类别 logits。正式评估依然使用 raw logits。训练侧改善不等于独立测试收益。

## 运算和接入点

1. 从共同 A/B 开始完成全部客户端普通 B 训练，缓存原始 `ΔB_j = B_j_loc - B_start`。
2. 继承的普通 B 聚合完整执行，得到实际 `B_bar`；不修改本地状态或客户端权重。
3. 在同一个 `B_bar` 上探测 donor 并取并集 D。
4. 构造每层 `R(C) = mean_j(ΔB_j @ C_j)`，候选模型为 `B_bar + R(C)`。
5. 每一步清零一次 C 梯度，逐接收端前向与反向；**客户端循环内部不执行 optimizer.step**。每个本地损失除以接收端数量，最后对共享正则反向一次，再执行一次 Adam。
6. 每个客户端重建小矩阵残差的计算图，计算完释放该客户端的视觉网络图，不同时保留所有客户端的网络图。
7. 第二步后重建 R，提交 `B_bar + R`，不再乘接收端样本权重，也不进行第二次 FedAvg。
8. 按原时间表继续方法 A。

A、基础 B、骨干、文本特征和 donor ΔB 都固定，只对 C 求梯度。使用现有 `differentiable_b_residual` 在前向添加 `scaling * R * A * x`，保留到 C 的完整梯度；最终直接并回 B，推理 rank 不变。

该协议有两次客户端反馈同步。图片仍在客户端；服务器接收 donor ID 和关于共享 C 的梯度。记录密集 FP32 消息的建模通信量，不声称差分隐私或安全聚合。当前仓库仍为单 GPU 串行联邦模拟。

## 日志与恢复

- `sfra_config.json` / `b_transfer_config.json`：明确 `mode=shared`、反馈端范围和实际矩阵 rank。
- `b_transfer_rounds/rXXX/calibration_manifest.json`：原来的两批图片、每端正向 donor、最终共享 donor 顺序。
- `donor_scores.csv`：参照为 `ordinary_shared_B`。
- `optimization_steps.csv`：每轮最多两行共享更新；记录梯度、C 范数及有效 LoRA 改动幅度。
- `client_feedback_steps.csv`：每端每步的损失、图片数量、权重和共享 C 版本；同一步版本必须相同。
- `receiver_summary.csv`：同一共享模型补充前后，不再冒充本地模型前后。
- `matrices.npz`：共享 donor IDs 和每个模块的 C。
- `commit.pt`：普通共享 A/B、补充后的 A/B 和残差。
- `b_transfer_rounds.csv`：算法/诊断图片数、两个全局步骤、实际客户端反向 batch 数、建模通信和耗时。

原 `events/...normal_B/state.pt` 保持真实的普通 FedAvg 重建，不伪造经过迁移的本地更新。对应事件的 `state_role=ordinary_B_before_shared_transfer`，`shared_transfer_state_path` 指向实际迁移提交文件。之后 A 阶段和全局 checkpoint 使用已经补充的 B。

逐轮 checkpoint 复用原机制；中断后恢复到最后完成的轮次，半轮 C 不继续使用，而是重做该轮。旧版 checkpoint 不能作为新版的 `--resume`。

## 启动实验

在服务器仓库根目录、原训练环境中执行。一张 GPU 即可，前台显示日志和报错：

```bash
CUDA_VISIBLE_DEVICES=4 python -u scripts/run_cliplora_b_shared_transfer.py --method full-cp --retention-weight 10 --classification-weight 1 --transfer-lr 0.3 --reference-run references/full10_clientlt --fast-execution
```

`--reference-run` 只复制原数据划分和训练协议，不加载该实验训练完成后的模型。从 CLIP/LoRA 初始化开始完整训练 100 轮。该目录应是之前已复制好的协议目录；不需要拷贝旧权重。

输出与旧实验隔离在：

```text
output/cifar100_LT/sfra_b_shared_transfer/seed42/client-longtail/full-cp/
  lambda10_mu1_protocol42_bshared_b_lr0.3_probe0.1_reg0.001_fast/
```

同一命令末尾加 `--resume` 可续训。GPU 编号可以变，影响输出目录的其他参数保持一致。

先只运行这一组，不同时搜索多个新参数；已有 Full-CP μ=1 和旧 B lr=0.3 用作开发比较。它与旧版不同之处包含共享参照的 donor 筛选和共享 C 联合校准，不能把差值仅归因于“改变相加位置”。

完成后汇总、打包：

```bash
python -u scripts/run_cliplora_b_shared_transfer.py --stage pack
```

输出 `output/cifar100_LT/sfra_b_shared_transfer_analysis.tar.gz`。不要给此打包命令附加只有协议、没有性能结果的 `references/full10_clientlt`；如需合并 A-only 结果，`--reference-run` 必须指向该 A-only 的完整结果目录。

## 验证边界

代码以 CPU 小张量单元检查核对共享梯度、参数冻结、单次残差提交、旧入口隔离及恢复/汇总接线，不运行数据集或 GPU 冒烟训练。正式 CIFAR100-LT 训练的耗时、显存及准确率提升仍需服务器实验确认。

本次共通过 61 项 CPU 单元检查：新版 5 项、原 B 聚合 5 项、现有加速执行 18 项、A 正文实验入口 10 项、A/CP 数值及恢复 23 项。
