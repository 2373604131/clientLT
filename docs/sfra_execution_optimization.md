# SFRA 工程加速：实现范围与审核

本次只优化执行方式，不设计新方法，不改变 SFRA / Full-CP / B-transfer 的训练协议。

## 开启方式

在现有 `run_cliplora_sfra.py` 或 `run_cliplora_b_transfer.py` 命令后添加：

```bash
--fast-execution
```

默认关闭。打开后，原配置的实验目录名追加 `_fast`，不覆盖旧结果。加速设置单独保存为
`execution_config.json`，不混入算法的 `sfra_config.json`。

两组等权聚合实验的新启动命令如下。分别在两个终端前台执行；已经运行中的旧实验不需要中断或重跑。

```bash
CUDA_VISIBLE_DEVICES=4 python -u scripts/run_cliplora_sfra.py --method full-cp --retention-weight 10 --classification-weight 1 --b-aggregation uniform-transfer-rounds --reference-run references/full10_clientlt --fast-execution
```

```bash
CUDA_VISIBLE_DEVICES=5 python -u scripts/run_cliplora_sfra.py --method full-cp --retention-weight 10 --classification-weight 1 --b-aggregation uniform-transfer-rounds --b-transfer --transfer-lr 0.3 --reference-run references/full10_clientlt --fast-execution
```

加速版本中断后，在同一条命令后追加 `--resume`。启动器重放原始保存命令，并校验独立的执行配置。
旧的非加速实验仍按旧命令恢复，不在中途切换计算实现。

## 已实现的优化

| 项目 | 实现 | 保留的语义 |
| --- | --- | --- |
| 类别文本缓存 | 普通训练和测试复用固定、归一化的文本特征 | 文本、提示词、缩放系数与分类头不变；权重重新加载时缓存失效 |
| 固定图片缓存 | 固定 witness 的确定性预处理结果保存在 CPU | 样本身份、顺序、两个视图不变；普通训练随机增强不缓存 |
| 只恢复 A/B | 首次完整加载模型，后续客户端/阶段切换恢复全部 A/B | A/B 都恢复，不让客户端继承前一客户端参数；优化器重建规则不变 |
| 冻结视觉前段缓存 | 缓存进入第一个 LoRA 块前的完整 patch/CLS tokens | 当前 top3 对应前九个块固定；后三个块始终正常前向/反向；不缓存最终视觉特征 |
| 低秩前向 | 显式计算 `B(Ax)`，不临时构造稠密 `BA` | 原始骨干、alpha/sqrt(rank)、dropout 和 A/B/C 梯度不变 |
| 反馈合批 | 同一客户端内合并反馈前向，默认最多 32 张 | 每个 token、每个视图仍分别求平均；保留原样本量权重；不跨客户端合批 |
| 精确零梯度跳过 | 修正阶段功能系数恰好为零时，不调用相应反向 | 不是小梯度截断；下一步重新判断；LA 分类梯度照算 |

来源识别仍使用逐 token、逐视图的独立完整 A 梯度，保存原范数与 QR 投影。
合批主要用于修正和纯分数测量，不用一个总梯度替代来源识别所需的独立梯度。
功能梯度与分类梯度仍分开返回；CP 的非线性惩罚仍在全局 LA 损失聚合以后计算。

## 明确没有修改的部分

- 数据划分、客户端参与、顺序、随机流和训练增强。
- rank、alpha、LA、学习率、损失权重、epoch、90 次 A 刷新与每次三步修正。
- 来源统计、阈值、目标构建、sigma、历史注册、投影和最终提交规则。
- A/B 聚合权重，以及等权聚合的八个指定轮次。
- B 的 donor 筛选、C 初始化、C 优化器、两个校准 batch、迁移重构。
- 逐轮测试、结果指标、断点保存时点和分析所需的事件文件。

`utils/sfra_math.py`、`utils/cliplora_b_transfer.py`、`utils/lora_aggregation.py`、
`utils/b_aggregation.py`、`utils/cliplora_a_refresh.py` 均未修改。

## 修改文件边界

- `utils/sfra_execution.py`：新增缓存、冻结前段拆分和 A/B 状态恢复工具。
- `utils/sfra_fast_feedback.py`：新增等价的反馈执行路径。
- `utils/sfra_feedback.py`：增加显式分派及来源识别时的前段缓存读取；旧路径保留。
- `trainers/cliplora.py`：只在 forward 中增加可选文本缓存。
- `utils/loralib/layers.py`：只在 LinearLoRA.forward 增加可选低秩计算。
- `utils/cliplora_la_control.py`：只增加状态恢复调用入口，默认仍完整加载。
- `utils/cliplora_sfra.py`：仅加速路径安装、状态恢复调用、执行计时及断点元数据接线。
- 两个命令入口：只增加开关和独立目录后缀。
- 新增本说明与 `tests/test_sfra_execution.py`。

## 内存与结果边界

所有缓存保留原精度（当前 FP32），放在 CPU，不使用半精度压缩。
按现有 5,842 张反馈图片、两个视图估算，图片和视觉前段缓存合计额外约 **10 GiB 主机内存/进程**。
同时运行两个实验约需额外 20 GiB，尚不包括训练原本的内存和数据加载器。缓存不写入 checkpoint，
重启后按需重建；首次填充缓存计入运行耗时。

这不是逐位确定性保证。低秩计算与合批改变 FP32 运算顺序，可能产生小数值差异；临近筛选阈值时，
不能由局部单元测试推断整个 100 轮的所有决策都相同，更不能承诺最终准确率完全一致。

`sfra_costs.csv` 保留逻辑图片访问计数，并添加合批次数/零梯度批次数；缓存省掉的是前段重复计算，
不能简单根据 `forward_images` 推断 FLOPs。真实速度需在服务器观察同阶段耗时，未给出预估加速倍数。

## 审核与验证

本次本地结果：**56 项相关 CPU 单元测试通过**（原有 38 项、新增 18 项），10 个 Python 文件语法检查通过。
另对照修改前版本进行 AST 审核：`refresh`、`restore`、`run`、`effective_norms`、
`phase_aggregation_weights`、`prepare_b_aggregation`、`train_phase` 的算法表达式保持不变，
只有完整模型恢复调用替换成可选的状态恢复入口；上列五个受保护方法文件内容未变。
固定测试样例中，两种执行方式最终提交的 A 参数最大绝对差小于 `5e-7`；这不是完整训练误差上界。

运行：

```bash
python -m unittest discover -s tests -p "test_sfra*.py" -q
```

新增检查使用 CPU、小尺寸的仓库真实 VisionTransformer / LoRA 实现及固定合成张量，
不下载权重、不读取训练集、不启动任何完整训练或冒烟实验。覆盖：

- rank 1/4/8、A-only/B-only/AB、dropout 随机流和输入梯度。
- 冻结前段拆分、原图/翻转、前段缓存复用与失效。
- 普通 CustomCLIP 文本缓存的 logits 与图像梯度。
- 逐项来源梯度范数、QR 投影、来源数量/支持掩码。
- 不同微批和合批大小、不同类别样本权重、未激活 token、完整 LA 梯度。
- 零功能梯度跳过，但保留分类梯度。
- 三步修正后的实际 A 参数，以及测试样例上的 CP/历史决策。
- 真正 SFRARuntime.refresh 接口、B/骨干不变、C 梯度不变。
- 全部 A/B 恢复、参数对象身份、模式/随机流的正常及异常恢复。
- 加速命令独立目录、原始命令续跑、执行元数据保存、冷缓存重建。

本地无 CUDA，且完整 Trainer 导入缺少 TensorBoard 依赖；因此测试提取实际 CustomCLIP 类体进行隔离验证，
未宣称完整 GPU/训练环境已验证。额外的旧 `test_lora_aggregation.py` 检查因本地缺少 pytest 未执行，
没有为此安装依赖或改动无关代码。
