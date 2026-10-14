三维样本分布图重绘
==================

重要：默认数据是根据截图手工构造的近似示例，不是恢复的原始实验数据。
截图中的细小文字不能完全确认，本脚本采用 10 类、20 个客户端、
Class / Client ID / Count 三个轴，类别和客户端从 0 开始编号。

1. 在 CAPT 项目根目录运行：
   python presentation/reconstructed_3d_histogram/plot_3d_histogram.py

   输出位于同目录 output/：
   reconstructed_3d_histogram.png  300 dpi 预览
   reconstructed_3d_histogram.pdf  矢量版
   reconstructed_3d_histogram.svg  矢量版，文字可编辑
   counts_used.csv                本次绘图使用的数值及数据来源说明

2. 控制参数：
   修改脚本顶部 COUNTS：行是客户端，列是类别，数值是柱高。
   修改 STYLE：视角、宽度、画布、字体、网格、坐标范围、坐标标题。
   修改 CLASS_COLORS：每个类别一种颜色。
   类别数和客户端数自动跟随矩阵形状；0 值不会画出假柱子。
   zmax 是期望的纵轴上限；数值超过它时上限自动扩展，避免裁掉柱高。

3. 命令行示例：
   python presentation/reconstructed_3d_histogram/plot_3d_histogram.py --elev 30 --azim -60
   python presentation/reconstructed_3d_histogram/plot_3d_histogram.py --show
   python presentation/reconstructed_3d_histogram/plot_3d_histogram.py --data my_counts.csv

   --show 打开可用鼠标旋转的窗口（需要带 GUI 的本机 Python）。
   导出发生在窗口打开之前。要把新视角导出，使用 --elev 和 --azim 重跑。
   CSV 不要包含标题行或索引列；以 # 开头的注释行会被忽略。
   counts_used.csv 可以直接作为 --data 的输入。
   不使用 CSV 时，脚本完全独立，可单独复制到其他目录运行。

4. 在 Python 中调用：
   from plot_3d_histogram import plot_distribution
   fig, ax = plot_distribution(my_counts, {"elev": 30, "azim": -60})
   ax.set_zlabel("Number of samples")
   fig.savefig("my_figure.png", dpi=300, bbox_inches="tight")

依赖安装（若尚未安装）：
   python -m pip install numpy matplotlib

已在 Python 3.13 / Matplotlib 3.10.9 环境中验证导出。

ClientLT 模拟（新增）
====================
运行：
   python presentation/reconstructed_3d_histogram/simulate_clientlt.py

固定类别 0-9、客户端 0-19；尾部类为 7/8/9，尾部客户端为 18/19。
模拟标签使用指数长尾分布，最大类 5000 个样本、最小类 100 个样本。
先调用仓库 utils.datasplit.partition_client_longtail 生成基础划分，
组内 Dirichlet alpha=0.1，随机种子为 1；随后对模拟计数施加指定的
稀疏泄露约束。不读取或修改真实训练数据，也不修改仓库训练划分实现。
该模拟脚本依赖当前仓库及其 torch 依赖，原绘图脚本仍可独立使用。

默认尾部类对客户端 0-17 的泄露总量为 60：
   Class 7 -> Client 3/9/15，分别 10/10/9 个，共 29 个；
   Class 8 -> Client 1/6/12，分别 7/6/6 个，共 19 个；
   Class 9 -> Client 4/7/11，各 4 个，共 12 个。
尾部客户端接收的非尾部类别因客户端而异，合计 20 个样本：
   Client 18: 175 个尾部样本 + 10 个其他类别样本（1/3/5: 4/2/4）；
   Client 19: 257 个尾部样本 + 10 个其他类别样本（2/6: 6/4）。
18 仅接收 1/3/5，19 仅接收 2/6。各类别数量采用带随机权重的
多项分布抽样；预算充足时每个指定类别至少有 1 个样本。
默认使用独立 leakage-seed=1，结果可复现。
全局各类总数、外泄总量、混入总量和 18/19 原有尾部样本数均守恒。
这些是实际生成矩阵的统计值。不同 seed/alpha 会改变基础组内分配。
脚本顶部 TAIL_LEAKAGE_TARGETS 和 INCOMING_ROUTES 控制泄露范围。

改变泄露量或浓度：
   python presentation/reconstructed_3d_histogram/simulate_clientlt.py --tail-leakage 45 --head-leakage 10 --alpha 0.1

只更换混入类别的随机数量（保持基础划分与外泄分配）：
   python presentation/reconstructed_3d_histogram/simulate_clientlt.py --leakage-seed 2

--tail-leakage 是三个尾部类向外泄露的合计样本数，不是每个客户端的数量。
--head-leakage 是客户端 18/19 合计接收的非尾部样本数。
尾部类的外泄预算按各类原有样本数量分摊，再均分到指定客户端；
混入预算先均分到 18/19（奇数预算的余数分给 18），再在各客户端
指定的类别内随机分配。默认各客户端均混入 10 个样本。
混入类别改变时，从非尾部客户端调整对应计数，保证各类总数不变。
脚本由预算反推基础 ClientLT 划分的 lambda/rho，再施加位置约束。
默认 lambda=0.8644986449864499、rho=3.0。
受仓库公式约束，tail-leakage=0 时 head-leakage 也必须为 0。

clientlt_output/ 保存 PNG、PDF、SVG、数据矩阵和 simulation_summary.json。
图中 7/8/9 类和 18/19 客户端刻度用棕红色标记，柱高仍为真实的模拟计数。
ClientLT 图默认与新参考图统一：仰角 30 度、水平视角 -45 度、
三轴视觉比例 1:1:1（原比例为 1:1:0.75）。修改视图不会改变模拟数据。
可显式指定：
   python presentation/reconstructed_3d_histogram/simulate_clientlt.py --elev 30 --azim -45 --box-aspect 1 1 1
若需直接用原绘图脚本重画这份 CSV：
   python presentation/reconstructed_3d_histogram/plot_3d_histogram.py --data presentation/reconstructed_3d_histogram/clientlt_output/clientlt_counts.csv --output presentation/reconstructed_3d_histogram/clientlt_output/plain

新参考图近似重绘
================
reference_recreation/ 是按新上传图片近似还原的独立输出。
使用原绘图脚本的截图估算数据，三维坐标盒比例改为 1:1:1：
   python presentation/reconstructed_3d_histogram/plot_3d_histogram.py --box-aspect 1 1 1 --elev 30 --azim -45 --output presentation/reconstructed_3d_histogram/reference_recreation
详细参数和数据编辑方式见 reference_recreation/README.txt。
