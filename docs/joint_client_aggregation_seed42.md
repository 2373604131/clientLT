# 新聚合下冻结 LoRA A 与当前 A+B：两组正式实验

本轮只有两个100轮训练，均为seed42、同一Client-LT划分、同一初始化、同一新聚合权重。
聚合强度固定为lambda=0.1，是本轮预先指定的探索配置，不是验证集或测试集选出的最优值。

| 实验参数 | frozen | ab |
|---|---|---|
| LoRA A | 初始化后全程冻结 | 第1—90轮各学习1个本地epoch，随后原Full-CP修正 |
| 普通LoRA B | 每轮3个本地epoch | 相同 |
| 方法A | 无 | Full-CP：保持权重10、CP权重1、原来源优先级/历史/三步修正 |
| 方法B | 无额外补充更新 | 原共享C：30/40/50/60/70/80/90/100轮、两步Adam、lr0.3、reg0.001、probe0.1、class-cyclic、tail权重0.35 |
| 普通模型聚合 | 新权重用于每轮B | 同一新权重用于每轮B和开放轮次的普通A提案 |
| LA训练先验 | 原训练频数、tau=1 | 相同，不因聚合改变 |
| 普通B优化步数 | 105600 | 105600 |
| 额外A本地优化步数 | 0 | 31680 |

`ab`是既有Full-CP＋共享donor-C方法，不是自由校准F，不是新设计的日常B学习算法。
固定方法A指固定实现和超参数，不是冻结AB组的LoRA A参数。
新权重只替换模型聚合；来源统计中的原样本权重、CP样本权重、B反馈目标和残差组合形式保持原定义。
B反馈的基点为本轮实际新聚合后的普通B，残差只注入一次；历史、来源响应和反馈均沿各组新轨迹产生，不从旧结果拷贝。

## 这两组能回答什么

能判断：在同一个新聚合基础上，从“永久冻结A、普通B训练”增加整套A学习、保持修正和B迁移，性能、保持、学习以及成本如何变化。

不能单独证明：A和B各自的长期贡献、额外优化预算没有作用、新聚合优于旧聚合、某个梯度方向是唯一因果原因。
本轮尊重只跑两组的安排，不自动增加S、A-only、B-only或其他lambda的完整训练。
同轮原样本量聚合反事实只用于即时诊断，不是第三条训练，也不是旧聚合的完整结果。

## 聚合定义

客户端类别计数n_jc、总量n_j，R_jc=n_jc/n_j，原权重q_j=n_j/N，类别目标u_c=1/100。
服务器一次性解严格凸问题：

    minimize KL(u || R.T @ w) + lambda/2 * sum_j (w_j-q_j)^2/q_j
    subject to w_j >= 0 and sum_j w_j = 1

求解器使用FP64 SLSQP，并检查权重归一化、KKT条件及相对FedAvg的目标改善。
所有客户端全参与且划分固定，权重一次生成后冻结，两个运行读取相同的权重及指纹。
程序不根据测试准确率更新权重。R.T@w是类别分布代理，不是实际梯度或知识贡献。
真实部署需要每客户端类别计数；30×100个int32计数的一次逻辑上传为12000字节，当前模拟已有这些信息。

## 详细观察设计

两组都保存第0—100轮的完整测试预测、原始logit间隔、逐类正确率；最终指标统一取第81—100轮均值。
不挑最好测试轮次。单种子不报告显著性，不把100个类别或20个轮次当独立重复。

在预先指定的1、20、30、50、70、80、90、100轮，增加只读测试阶段诊断：

| 对比 | 可以观察什么 |
|---|---|
| 普通新聚合B－上一轮提交模型 | 日常B训练与聚合之后的即时学习/损失 |
| 普通新聚合B－相同本地更新的原样本量聚合B | 同一起点同一批更新，仅改变B聚合权重的即时影响 |
| B补充后－普通新聚合B | 共享C补充是否改善测试预测，是否只改置信度 |
| 普通A提案－本轮B阶段终点 | 开放A产生多少学习收益和损失 |
| 新聚合A提案－相同本地更新的原样本量聚合A | A因子聚合权重的即时影响 |
| 最终提交－普通A提案 | 保持修正保住了哪些预测，又牺牲了哪些预测 |

阶段测量严格恢复模型状态、训练模式、Python/NumPy/Torch随机状态；使用独立的global_trainer，不返回给训练决策。
额外测试开销单列，不算作新聚合算法成本。原A/B训练侧反馈仍按原实现计费。
正式冻结组增加16次完整测试，AB组增加36次完整测试；两组通常每轮的提交模型测试另计。这里的额外前向是实验诊断开销，聚合求解本身只在初始化时运行一次。
阶段差分只解释当前轨迹上的即时动作，不等于移除该组件重训的长期效果。

分析同时输出：

- Overall、Head20、Middle60、Tail20、Non-tail、尾类峰值至末轮回落和90→100轮变化。
- 逐类结果及训练样本数、持有客户端数，观察收益是否只集中于少数类别。
- 起初答对样本的保持、起初答错样本的学会、曾答对后来答错、始终没有答对的样本。
- 以第0、50、80轮为参考的保持/新学习样本数，避免仅凭组平均准确率推断遗忘。
- 两组最后20轮的“只有AB答对／只有冻结组答对／都答对／都答错”。
- 本地训练步数、A修正步数、B优化步数、算法与诊断成本，以及原始运行环境。

安装Matplotlib时，汇总会自动生成三套PNG/SVG图：四组精度曲线、仍未学会与后来答错的样本曲线、各操作的阶段即时变化。图嵌入report.md，CSV保留原始数值；没有Matplotlib也可以完成表格汇总。

## 用结果决定下一步

| 观察到的结果 | 优先检查或优化 | 尚不能直接下的结论 |
|---|---|---|
| AB改善头中部且尾类维持/提高 | 保留整套配置，之后再验证各组件必要性 | A、B各自必不可少 |
| 普通A改善头中部，保持修正又削弱该改善 | 保持对象、历史参考、修正范围 | 单轮差分已证明长期头类损失原因 |
| 普通A本身就损害尾类，修正无法补足 | A开放日程、学习步幅及需要保持的功能 | 固定A必然最优 |
| 共享B改善训练反馈但测试收益小 | 反馈覆盖、来源利用、目标与测试泛化的差异 | 只要加大C或训练步数就会好 |
| 冻结组学会少，AB明显增加原本不会的正确样本 | 进一步判断A开放和B补充分别在哪个阶段带来收益 | 额外训练预算已排除 |
| 两组都留有较多始终错误样本 | 检查类别支持、表征、局部学习及反馈覆盖 | 已证明LoRA容量不足 |
| 新聚合在即时反事实中仍损害部分组 | 检查统计代理与实际更新作用的偏差 | 已证明新聚合完整训练无效 |

## 服务器 GPU 2 / GPU 3 与四客户端并行

默认采用四客户端执行，新输出目录为`output/cifar100_LT/client_aggregation_v2_parallel4`，避免与旧串行计划混用。
GPU 2运行`frozen`，GPU 3运行`ab`；每个实验进程只看见自己的那张GPU，不会启用跨卡DataParallel。
每张卡最多并行4个客户端，两个实验合计最多8个客户端。

每个客户端保留独立的LoRA A/B、梯度、优化器与CUDA stream；冻结的主干参数共享存储，缓冲区独立。
主线程先按原串行顺序生成本地批次，再交给任务队列。普通B保持原DataLoader随机流；A保持原有按轮次、客户端隔离的种子规则。
客户端本地的batch size、epoch数、优化器、学习率、LA先验及全局聚合权重不变。
局部数据处理使用4个工作线程，CPU算子线程数固定1；`--num-workers`仍控制原评估等DataLoader。
仅普通B本地训练和普通A提案并行，A保持修正、共享C反馈和最终聚合仍按原顺序运行。

队列没有“必须凑满4个”的条件。30个客户端完成时，末尾剩余1、2、3个任务都正常退出；既不复制客户端也不丢弃任务。
必须等本阶段全部30个客户端完成，再按原客户端顺序聚合一次。某客户端真实失败时不会聚合不完整的结果。

每组smoke会做以下加速尝试：

1. 各CUDA stream做短预热。
2. 相同起点和批次下，6个客户端分别串行与四路运行，比较参数（atol=1e-6、rtol=1e-5）和训练步数。
3. 额外提交只有2个客户端的任务，核对不足4个时的结果。
4. 完成整个30客户端的一轮运行；AB还实际触发A保持修正和共享C。

这些计时和数值对照不提交给正式训练。只有smoke通过才启动对应组的正式训练。
`smoke/seed42/<arm>/parallel_benchmark/factor_B.json`记录串行/四路耗时、速度比、显存和数值误差；AB另有`factor_A.json`。
`analysis/parallel_pilot.csv`汇总计时；速度比小于1就是这次尝试更慢，不会把四路并行直接宣称为四倍加速。
四路执行会增加激活显存，本地CPU测试不能判定你的GPU是否足够；OOM等真实错误会明确报出，不会悄悄改训练batch size。
如需回到原串行路径，显式设置`--client-concurrency 1`并使用另一个输出根目录。

## 启动

需要原clientlt训练环境、CUDA、NumPy、SciPy及CIFAR-100。使用conda环境中的Python。
默认参考是仓库`references/full10_clientlt`，只重放划分/协议/初始化约定，不需要搬运旧模型检查点。
如果该目录缺失，显式传`--reference-run`指向已有完整S/Full-10运行。

先在仓库根目录执行：

```bash
python scripts/run_cliplora_joint_aggregation.py --stage preflight
```

预检只检查文件、配置和CPU求解器，不代表GPU训练已通过。
在同一台服务器上，一条命令启动两张卡，每组先smoke再正式训练：

```bash
python scripts/run_cliplora_joint_aggregation.py --stage server --gpus 2 3 --client-concurrency 4
```

也可以分别在两个终端手动启动；`run`会自动执行或验证对应的smoke：

```bash
CUDA_VISIBLE_DEVICES=2 CUDA_DEVICE_ORDER=PCI_BUS_ID python scripts/run_cliplora_joint_aggregation.py --stage run --arms frozen --client-concurrency 4

CUDA_VISIBLE_DEVICES=3 CUDA_DEVICE_ORDER=PCI_BUS_ID python scripts/run_cliplora_joint_aggregation.py --stage run --arms ab --client-concurrency 4
```

两个正式训练共享输出根目录，但各有独立目录和进程锁，可并发运行。
总调度日志位于`launcher_logs/frozen_server.log`和`launcher_logs/ab_server.log`；训练过程日志为各自的`*_smoke.log`及`*_formal.log`。

smoke是独立目录的一轮GPU检查；AB smoke仅在测试目录把共享B事件提前到第1轮以实际触发路径。
它不是正式结果，也不会作为正式训练检查点。正式B仍按第30轮开始。
中断后重跑相同run命令，会从最后完成轮次自动恢复；不需要手动改历史/配置。
运行中不要修改这套训练实现。配置或代码变更需要新`--output-root`，不复用旧实验目录。
日志位于`launcher_logs`；失败时启动器会打印日志末尾。

结果分析和打包不需要GPU：

```bash
python scripts/run_cliplora_joint_aggregation.py --stage summary
python scripts/run_cliplora_joint_aggregation.py --stage pack
```

默认结果根目录为`output/cifar100_LT/client_aggregation_v2_parallel4`。
`analysis/report.md`包含两组汇总与判读边界。`pack`产生旁边的`client_aggregation_v2_parallel4_results.tar.gz`，包含预测、阶段诊断、加速尝试和审计元数据，排除模型检查点及smoke训练预测。

CPU验证命令：

```bash
python -m unittest discover -s tests -p test_joint_aggregation.py
python -m unittest discover -s tests -p test_joint_client_parallel.py
```
