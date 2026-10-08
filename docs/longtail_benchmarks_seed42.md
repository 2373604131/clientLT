# 全局联邦长尾四组对照：实现、核验与启动

本批包含 `fedavg-lora-la`、`fedlf`、`fedyoyo`、`fedrela`，固定训练 seed42、协议 seed42，使用已有 Client-LT 原始样本分配、30 个客户端、100 轮全参与、每轮每客户端 3 个本地 epoch。推理始终评价同一个共享模型。

新入口为 `scripts/run_longtail_benchmarks.py`，默认输出 `output/cifar100_LT/longtail_benchmarks_seed42_v1`。第一批、因子第二批的默认方法列表及原 A/A+B 训练实现不变。新增方法是保留核心计算的 CLIP-LoRA 协议适配，不能称为复现了原论文的 ResNet 主表数字。

**官方代码与复用方式**

| 方法 | 固定官方版本 | 本地接入 |
|---|---|---|
| FedAvg＋LoRA＋LA | 本仓库既有实现 | 复用 `runtime.local_train`，训练用 `CE(logits + log(global_prior), label)`，推理使用原 logits，tau=1 |
| [FedLF](https://github.com/18sym/FedLF) | `127cbaf363ec3a1f40c116f6b1de3c1f7a705764` | 直接执行原 `DecorrLoss`；移植 logit 缩放、类中心损失及一次本地预扫描 |
| [FedYoYo](https://github.com/shanss132/FedYoYo) | `fc6d728febb461ca46eedd142ef835b6c6572f88` | 直接复用原 AutoAugment、Cutout 和分布变化保护；移植完整有效类别数估计、DLA 和 ASD |
| [FedReLa](https://github.com/guangzhengh/FedReLa) | `20a58bd75f48b3243f80adb35537b5b4ba252773` | 直接执行原局部分布归一化、逐类 z-score、阈值和随机重标记函数；宿主为共享 FedAvg-LoRA＋CE |

官方文件放在 `third_party/paper_baselines/{fedlf,fedyoyo,fedrela}/`，保持原始内容，并保存 commit、下载包 SHA256 和逐文件 SHA256。`trainers/baselines/upstream.py` 在核验哈希后只加载指定函数/类定义，不执行官方启动脚本中的命令行解析、重新划分数据或下载数据。服务器同步源码目录即可，不需要重新从 GitHub 下载。

FedLF 官方启动脚本调用了未定义的 `update_feature_syn`，并残留写死的特征/类别维度，所以不能原样启动。本批重用其本地训练计算，并沿用已有可靠的全局训练、聚合和评估入口。

**计算与适配细节**

- 四组均使用相同的 CLIP ViT-B/16、视觉 top3 q/v LoRA rank4、固定文本头和 FP32。其余骨干、文本编码器、文本提示参数保持冻结；不叠加 A 或 B。
- 全局模型及初始 LoRA 都检查参考哈希。每个客户端从同一轮全局状态出发；SGD 优化器不在客户端之间继承。公共 SGD 仍为 lr=0.001、momentum=0.9、weight_decay=0.0005。这些属于统一协议，区别于部分上游的原生 SGD 参数。
- 三个新增方法不默认叠加本文的全局 LA。FedLF 与 FedYoYo 保留自身校准，FedReLa 宿主使用 CE。

FedLF：根据本地类别计数构造 `s_c = 0.25 + 0.75 * n_c / max(n)`，训练使用缩放后的 CLIP logits。每轮本地训练前，完整扫描本地训练样本计算固定类中心；中心和去相关损失使用 CLIP L2 归一化之前的图像特征，与产生分类 logits 的同一次前向共享计算。缺失类中心按原代码置 `1e-8`；类别中心间最大距离截断至 100，加入真类距离，中心项权重 0.01、去相关项权重 0.01。三个损失及实际两步参数更新均与官方函数核对。

FedLF 只增加一处数值保护：平方距离开根前截断到 `1e-12`，避免浮点负零或重合中心导致 NaN；单样本 batch 的去相关项仍为零。保留原 crop/flip 增强，改为逐样本处理后 resize 至 224。该输入实现与上游在整个 batch 上共用随机裁剪坐标的写法不同，属于输入适配，未声称增强随机数轨迹相同。

FedYoYo：每轮先使用轮初全局模型在各客户端训练数据上扫描两次，第一次得到类中心，第二次按原代码的 batch 内中心化特征相关矩阵计算有效类别数。服务器累加各客户端估计、加全 1 初值、处理首轮异常值，并从第二轮起以 0.9 动量平滑，再应用官方变化阈值 100。这里没有用真实全局标签计数替代估计。

本地训练每个 batch 更新 `prior = 0.9 * prior + 0.1 * local_distribution`，再加入 `log(prior^1.5 + 1e-9)`。弱增强与强增强一起前向，两支都接受 CE；只选校准后弱增强预测正确的样本，使用停止梯度的弱分支分布蒸馏强分支。温度 1.5，KD 权重 4，前 50 轮线性 warm-up；按照官方实现，不额外乘温度平方。弱增强保留 crop/flip/rotation15，强增强直接复用官方 CIFAR10Policy＋Cutout，即使数据为 CIFAR100 也与其官方分支一致。

FedYoYo 空筛选集合返回可微的零 KD，避免先产生 NaN 再替换。首轮若所有类别均触发异常阈值，使用均匀有效类别数作为退化保护，避免官方空均值的 NaN。正常输入分支与官方计算保持一致。默认参数来自官方运行示例和参数解析器，尚未声称是当前 CLIP 设置下的最优超参数。

FedReLa：在完成 90 轮后、第 91 轮本地更新前，由同一个全局模型分别预测各客户端的原始训练样本，按官方流程进行 5 遍 posterior 收集与平均。沿用宿主确定性的原输入处理，因此这 5 遍没有新增随机增强。随后直接调用官方自适应 top-5% 阈值与随机重标记流程，只允许本地较多样本类向较少样本类转移。

重标记映射按原始样本 ID 保存，一次生成，后续复用；不修改 DATA 文件、不修改分配清单，不重新定义头中尾类别。按官方训练实现，在重标记后将学习率乘 0.1。该行应标为 **FedAvg-LoRA＋FedReLa（含上游后期学习率安排）**，不能将它与恒定学习率 CE 宿主之差解释为纯重标记消融。重标记轮数由原示例的后 10% 训练阶段映射到当前 100 轮协议。

所有方法继续按样本量 FedAvg，报告相同测试集第 81–100 轮平均 Overall／Head20／Middle60／Tail20 ACC，以及末轮结果。测试输出不参与阈值、标签、分布状态或最优模型选择。FedReLa 的测试驱动早停和上游脚本的 best-test checkpoint 不接入。

**状态、成本和汇总**

- FedYoYo 的上一轮有效类别数、FedReLa 的标签映射/触发轮次和模型/RNG 一起保存到完整轮次检查点。中断重放不会偷偷重新采样已提交的标签映射。
- `method_audits/rNNN.json` 保存 FedYoYo 的估计先验，或 FedReLa 的修改数量、前后计数、阈值和样本映射。
- `client_costs/` 和 `costs.csv` 记录额外统计前向、双视图训练、有效 KD 样本、重标记使用量及统计通信字节。成本表不把辅助扫描当作免费。
- `--stage pack` 会打包配置、来源记录、曲线、逐类结果和方法审计；不打包模型权重。短程结果标为 `smoke_only`，不会进入正式性能表。

**服务器启动**

将本次新增和修改的 `scripts/`、`tools/benchmarks/`、`trainers/baselines/`、`configs/benchmarks/`、`third_party/paper_baselines/` 同步到服务器。不要用新代码恢复前两批尚未结束的旧源码指纹任务；本批使用独立输出目录。

在仓库根目录、此前成功运行对比实验的环境中，先做四组短程检查。这会依次运行四种方法，每种 2 轮、每轮 2 个客户端、每客户端 1 个训练 batch，并执行完整测试集评估：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_longtail_benchmarks.py --stage smoke --seed 42 --reference-run references/full10_clientlt --data-root DATA --num-workers 4 --output-root output/cifar100_LT/longtail_benchmarks_seed42_v1
```

短程的 FedReLa 在第 2 轮提前触发重标记，以实际覆盖后期路径；正式实验仍在第 91 轮触发。四项检查完成后，分别在四个终端启动：

```bash
CUDA_VISIBLE_DEVICES=0 python -u scripts/run_longtail_benchmarks.py --stage train --methods fedavg-lora-la --seed 42 --reference-run references/full10_clientlt --data-root DATA --num-workers 4 --output-root output/cifar100_LT/longtail_benchmarks_seed42_v1
```

```bash
CUDA_VISIBLE_DEVICES=1 python -u scripts/run_longtail_benchmarks.py --stage train --methods fedlf --seed 42 --reference-run references/full10_clientlt --data-root DATA --num-workers 4 --output-root output/cifar100_LT/longtail_benchmarks_seed42_v1
```

```bash
CUDA_VISIBLE_DEVICES=2 python -u scripts/run_longtail_benchmarks.py --stage train --methods fedyoyo --seed 42 --reference-run references/full10_clientlt --data-root DATA --num-workers 4 --output-root output/cifar100_LT/longtail_benchmarks_seed42_v1
```

```bash
CUDA_VISIBLE_DEVICES=3 python -u scripts/run_longtail_benchmarks.py --stage train --methods fedrela --seed 42 --reference-run references/full10_clientlt --data-root DATA --num-workers 4 --output-root output/cifar100_LT/longtail_benchmarks_seed42_v1
```

中断恢复：在对应原命令末尾追加 `--resume`，其余参数不变。查看进度与汇总打包：

```bash
python -u scripts/run_longtail_benchmarks.py --stage status --output-root output/cifar100_LT/longtail_benchmarks_seed42_v1
python -u scripts/run_longtail_benchmarks.py --stage pack --output-root output/cifar100_LT/longtail_benchmarks_seed42_v1
```

结果包为 `output/cifar100_LT/longtail_benchmarks_seed42_v1_analysis.tar.gz`。已有 A/A+B 结果也可通过相同入口的 `--stage import-ab --import-ab-root 原AB验证目录` 审核后导入。

**检查记录与范围**

检查命令：

```bash
python -m unittest tests.test_paper_benchmarks tests.test_benchmark_reference_config tests.test_factor_benchmarks tests.test_benchmark_model_startup tests.test_longtail_benchmarks -q
```

新增检查覆盖官方两步更新与梯度效果、有效类别数估计、z-score 和随机重标记、空 KD 集合、缺失类、单样本 batch、原始样本 ID 对齐、数据增强的多进程加载、逐轮恢复以及恢复后汇总打包。模型构建检查使用实际 CLIP 实现、随机小模型权重，验证仅预定 LoRA 参数发生变化。

2026-10-08 上述组合检查实际执行 **57 项，全部通过，没有跳过项**。其中 FedLF/FedYoYo 的参数更新对照直接执行下载的官方 `Local.local_train`，FedReLa 的决策对照直接执行官方统计及重标记函数。短程训练不是性能实验，不能据此比较精度高低。

本机真实 GPU 验证使用 RTX 4070 Ti SUPER、PyTorch 2.10.0+cu126、完整 CIFAR100 与官方 CLIP ViT-B/16 权重。实测骨干哈希与正式参考一致，但本机重建的初始 LoRA 哈希与服务器参考不同；在本机 PyTorch 2.6 环境也观察到这一差异，具体成因尚未定位。

因此真实数据执行检查使用独立的 `tmp/longtail_gpu_validation_20261008_final/local_initialization_fixture`，记录原初始化哈希和本地初始化哈希，原 `references/full10_clientlt` 未修改。此检查只能验证完整 CUDA 执行，不能替代服务器上原始初始化的一致性验收。正式入口仍然拒绝不匹配的初始化，不提供跳过开关。检查来源记录为该验证目录下的 `validation_provenance.json`。

四组真实 GPU 短程检查已全部完成：每组 2 轮、4 次客户端优化步、3 次完整 10,000 张测试图像评估。FedLF 的类中心和去相关损失实际非零；FedYoYo 记录了 74 个有效蒸馏样本；FedReLa 在短程第 2 轮对两个客户端合计 646 个训练样本生成映射，其中 196 个被重标记，并完成新标签更新。汇总显示四组均为 `smoke_only`，正式性能表为空，打包成功。原始检查输出位于上述目录的 `suite/smoke/`，检查压缩包为 `suite/smoke_analysis.tar.gz`。这些数值只证明对应路径实际执行，不证明任何方法的完整训练性能。

正式训练前使用上面的服务器 `--stage smoke`。若服务器同样出现 `Initial LoRA factors differ from reference seed42`，应核对其此前成功实验的环境及初始化来源，不能通过改参考哈希或删检查继续正式比较。
