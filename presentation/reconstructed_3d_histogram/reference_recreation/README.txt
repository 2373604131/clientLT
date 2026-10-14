新参考图的近似重绘
==================

此版本按用户新提供的小尺寸参考图近似还原，使用 plot_3d_histogram.py
中的截图估算数据 COUNTS。它不是 ClientLT 约束模拟，也不是原始实验数据。

保留 10 类、20 个客户端、按类别着色的三维柱状图；
三轴视觉比例设为 1:1:1，仰角 30 度，水平视角 -45 度。
纵轴上限 4000。图中精确数据和细小文字无法从参考图完整恢复。

在 CAPT 项目根目录重新生成：
python presentation/reconstructed_3d_histogram/plot_3d_histogram.py --box-aspect 1 1 1 --elev 30 --azim -45 --output presentation/reconstructed_3d_histogram/reference_recreation

输出：
reconstructed_3d_histogram.png   300 dpi 图片
reconstructed_3d_histogram.pdf   矢量 PDF
reconstructed_3d_histogram.svg   可编辑矢量 SVG
counts_used.csv                 本次使用的近似数据

修改 counts_used.csv 后，使用该文件作为输入：
python presentation/reconstructed_3d_histogram/plot_3d_histogram.py --data presentation/reconstructed_3d_histogram/reference_recreation/counts_used.csv --box-aspect 1 1 1 --elev 30 --azim -45 --output presentation/reconstructed_3d_histogram/reference_recreation
