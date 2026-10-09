# 方法 A 右侧修正机制：单独展开版

本次专门修改原模型示意图右侧。图中直接展示：

1. 双视图的更新影响筛选，正向影响均值与集中程度的不同用途。
2. 当前目标与登记历史目标取较大值。
3. 预测间隔的目标缺口、分类损失上升的两个惩罚条件，以及只回到 A 的梯度回路。

查看 figures/tikz_method_a_detail_zh.png 和 figures/tikz_method_a_detail_en.png。同名 PDF 为论文矢量文件，SVG 可编辑图形，TEX 可编辑文字与结构。生成脚本为 build_method_a_detail.py。执行该脚本将重新生成中英文 PDF / SVG / PNG，并测量实际 TeX 文字边界。

这是原右侧区域的独立展开图，按 177.8 mm 通栏排版时最小正文约 8.64 pt。请使用独立方法图的位置，不要将其原样缩进原概览图约 8 cm 的右栏。原左侧与中间图未修改，原完整概览也保留。

构图记录：METHOD_A_DETAIL_PLAN.md。

图注与实现口径：method_a_detail_captions.md。

实际成图审查：method_a_detail_review.json。

文字几何检查：method_a_detail_geometry_review.json。

所有示意数值形状都不是实验结果。输入图为原始 CIFAR 训练缩略图，其他元素为原生矢量。
