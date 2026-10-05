# 第二轮：共享 LoRA 因子基线（seed42）

本轮加入 FFA-LoRA、RoLoRA、FedSVD、LoRA-A²。所有客户端最终评估同一个共享模型，不加入 FedSA-LoRA 等客户端私有因子的个性化方法。原 A/AB 训练文件没有修改。

## 目录和实现来源

| 文件 | 用途 |
|---|---|
| `trainers/baselines/factor_freezing.py` | 因子冻结、交替训练、FedSVD 重参数化、自适应秩选择 |
| `configs/benchmarks/factor_baselines_seed42.json` | 第二轮独立配置 |
| `scripts/run_factor_benchmarks.py` | 第二轮入口；复用第一轮的协议检查、恢复和汇总 |
| `third_party/paper_baselines/{ffa-lora,rolora,lora-a2}/UPSTREAM.json` | 原论文链接、算法位置及适配说明；不是官方源码声明 |
| `third_party/paper_baselines/fedsvd/` | 官方代码固定版本 `e73398ba34b6b269c3bf76ed9d74906761fc1a4d`，含校验哈希 |

## 四个方法的实际计算

统一记号为 `delta_W = scaling * B @ A`。模型、数据分配、100 轮全参与、SGD、输入处理和 seed42 初始化与当前比较协议一致；保留仓库的 `alpha/sqrt(rank)` 缩放。它们都是 **CLIP 协议适配版**，不是原论文语言模型实验的原样复现。共同使用 CE，不加入我们自己的保持损失或 B 组件。

- **FFA-LoRA**：A 全程固定，只更新 B；服务器按客户端样本量聚合 B。为配对初始化，使用当前仓库的 Kaiming A，而非原论文的 Gaussian A。采用非 DP 版本。[论文第 4 节](https://arxiv.org/abs/2403.12313)。
- **RoLoRA**：第 1、3、5…通信轮只训练 B，第 2、4、6…轮只训练 A；每次只聚合活动因子，另一个因子保持完全相同。这里 100 轮指 100 次通信，不能将论文包含两个通信阶段的一次外循环算成一轮。按现有协议采用样本加权。论文 A/B 名称与实现可能不同，以矩阵形状映射。[论文第 3 节、算法 1](https://proceedings.neurips.cc/paper_files/paper/2025/file/adf17e7346b6be4c2f2bb40de572e5bc-Paper-Conference.pdf)。
- **FedSVD**：每轮客户端只训练 B，服务器聚合后令 `U,S,Vh = svd(B@A)`，取 `A=Vh[:r]`、`B=U[:,:r]*S[:r]`。不采用两边平分 `sqrt(S)` 的分解，因为本方法要求 A 行正交；模型缩放保持不变。每轮重参数化、无 DP 噪声。SVD 对有效矩阵的保持和官方函数数值一致性有单元测试。[论文公式 7](https://arxiv.org/abs/2505.12805)。
- **LoRA-A²**：保留交替冻结和自适应秩选择两部分。全局秩 `rG=4`，各客户端平均秩预算 `ri=2`，跨全部 LoRA 模块选取 `ri*N` 个槽位，而不是每层固定取两个。每轮先用本地训练数据试训 1 epoch，根据 `||delta_B[:,i]||*||A[i,:]||` 或 `||B[:,i]||*||delta_A[i,:]||` 评分；重置到轮初模型和新优化器，再训练 3 epochs。仅所选槽位的增量允许改变，未选位置保留既有参数。B 学习率是 A 的 5 倍，沿用原文比例；基础学习率为本协议 `.001`。预算 2 是显式预设，未根据测试集挑选。[论文公式 4–6、算法 1–2、附录 B](https://aclanthology.org/2025.acl-long.19/)。

LoRA-A² 的试训额外消耗必须计入，总本地计算是 1 个筛选 epoch 加 3 个训练 epochs。每步同时处理梯度掩码、weight decay 和 momentum，避免未选槽位偷偷改变。实现通过密集张量模拟稀疏更新，不宣称节省了实际 GPU 算力或真实网络流量。

FedSVD/RoLoRA/FFA 每客户端每轮 3 epochs；学习率不是针对测试集调好的最优值。本轮先核验同一设置下的机制表现，不能仅凭这一个种子就宣称达到 SOTA。

## 第一批：四张卡各启动一个方法

先一次性同步 `scripts/`、`tools/benchmarks/`、`trainers/baselines/`、`configs/benchmarks/` 和整个 `third_party/paper_baselines/` 到服务器。不要在训练中途覆盖源文件；入口会检查代码是否发生改变。

在服务器 `/data/yzh/clientLT`、`clientLT` 环境中，**分别在四个终端执行下列四行**。所有命令使用 seed42；输出目录相同，但每个方法的运行子目录、日志及锁文件独立，不会相互覆盖。

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_paper_benchmarks.py --stage train --methods fedavg-lora --seed 42 --reference-run references/full10_clientlt --data-root DATA
CUDA_VISIBLE_DEVICES=1 python -u scripts/run_paper_benchmarks.py --stage train --methods capt --seed 42 --reference-run references/full10_clientlt --data-root DATA
CUDA_VISIBLE_DEVICES=2 python -u scripts/run_paper_benchmarks.py --stage train --methods fedpurel --seed 42 --reference-run references/full10_clientlt --data-root DATA
CUDA_VISIBLE_DEVICES=3 python -u scripts/run_paper_benchmarks.py --stage train --methods fedntd --seed 42 --reference-run references/full10_clientlt --data-root DATA
```

这四条只启动第一批，不会启动第二轮或重新训练 A/AB。`capt` 是固定聚合的 CAPT 适配版；`fedpurel` 只使用 FedPuReL 的共享阶段。默认输出 `output/cifar100_LT/paper_benchmarks_seed42`。

中断后在相应命令后追加 `--resume`。若此前已用旧代码在同一目录登记过任务，新版会拒绝混用不同代码的登记；请使用新的 `--output-root`，不要删除旧实验来绕过检查。

查看第一批四个任务的当前轮次及客户端位置：

```bash
python -u scripts/collect_paper_benchmarks.py --status
```

第一批结束后汇总和打包：

```bash
python -u scripts/collect_paper_benchmarks.py --pack
```

## 第二轮：已实现，待第一批完成后运行

第二轮默认输出 `output/cifar100_LT/factor_benchmarks_seed42`。`plan`/`preflight` 不训练；`smoke` 只检查两个通信轮、每轮两个客户端的一批数据，保存到独立 `smoke/`，不会进入正式结果表。

```bash
python -u scripts/run_factor_benchmarks.py --stage preflight
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_factor_benchmarks.py --stage smoke
```

第二轮正式训练，同样每行使用一个终端：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_factor_benchmarks.py --stage train --methods ffa-lora
CUDA_VISIBLE_DEVICES=1 python -u scripts/run_factor_benchmarks.py --stage train --methods rolora
CUDA_VISIBLE_DEVICES=2 python -u scripts/run_factor_benchmarks.py --stage train --methods fedsvd
CUDA_VISIBLE_DEVICES=3 python -u scripts/run_factor_benchmarks.py --stage train --methods lora-a2
```

也支持 `--stop-after 1` 和 `--resume`。恢复保存完整 A/B 状态，交替阶段由已提交的通信轮编号恢复，无需持久保存客户端优化器。

导入已经完成的 seed42 A/AB，只读取核验，不重新训练：

```bash
python -u scripts/run_factor_benchmarks.py --stage import-ab --import-ab-root output/cifar100_LT/ab_decision_42_0_3407
```

查询、汇总和打包第二轮：

```bash
python -u scripts/run_factor_benchmarks.py --stage status
python -u scripts/run_factor_benchmarks.py --stage pack
```

得到 `output/cifar100_LT/factor_benchmarks_seed42_analysis.tar.gz`，包含四组固定 ACC、逐类结果、每轮成本、A² 选中槽位、SVD 保持误差和来源记录，不包含模型权重。

通信成本字段是参数载荷估算，排除共同初始模型分发和传输协议开销；A² 包含槽位索引/模块计数，FedSVD 每次重分解后需下发两个因子。实际模拟仍在一个进程内。

## 验证范围

本地测试覆盖冻结因子完全不变、两个方向的交替训练、掩码与非零历史参数、全模型秩排序、SVD 有效矩阵保持/行正交/官方函数对照、四组连续训练与断点恢复一致、正式 A² 计算预算核验，以及第一批四个任务并发登记。

本地没有 CUDA，真实 CLIP/GPU 运行仍需在服务器验证。
