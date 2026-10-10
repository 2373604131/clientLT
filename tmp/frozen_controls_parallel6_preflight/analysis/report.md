# 冻结 A 的聚合对照（seed42）

新增两条100轮训练：16倍加权与FedAvg。仅聚合权重不同；A全程固定，B每轮3个epoch，共105600步。
无额外A/B训练、保持修正、来源C、直接校准。本地损失与原始计数LA先验不变。
主指标预先固定为第81—100轮Overall均值，同时报告头、中、尾类；只有一个种子，不作跨种子显著性结论。

| 聚合 | Overall | Head20 | Middle60 | Tail20 |
|---|---:|---:|---:|---:|
| joint | 67.8600 | 70.7050 | 66.3308 | 69.6025 |

配对结果（左减右，单位：百分点）：
- tailrw16 − fedavg：unavailable
- joint − fedavg：unavailable
- tailrw16 − joint：unavailable

运行状态：
- tailrw16：missing，round=0 
- fedavg：missing，round=0 
- joint：complete，round=100 

sample_dynamics.csv 区分初始能力保持、新答对、逐轮遗忘与始终未答对；per_class.csv 保存每类结果。
stage_effects.csv 是同状态同客户端更新的即时对照，不代替完整训练轨迹。
parallel_pilot.csv 是同批次串行/并行试跑；六并行不代表六倍加速。历史新聚合只读，未重新训练。
历史E2额外训练过B，旧16倍实验训练过A，均不放入此表。
新聚合参考目录：G:\桌面\CAPT\output\client_aggregation_v2_parallel4_results\client_aggregation_v2_parallel4
