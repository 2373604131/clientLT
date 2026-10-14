#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""用 Matplotlib 重绘参考图中的三维类别/客户端样本分布。

默认数据是根据低分辨率截图手工构造的近似示例，不是恢复的实验数据。
运行：python plot_3d_histogram.py
旋转查看：python plot_3d_histogram.py --show
真实数据：python plot_3d_histogram.py --data your_counts.csv
依赖：numpy、matplotlib。
"""

import argparse
from pathlib import Path

import matplotlib
import numpy as np
from matplotlib.ticker import MaxNLocator, MultipleLocator


# ==================== 1. 数据 ====================
# 行 = Client ID (0~19)，列 = Class (0~9)。
# COUNTS[client_id, class_id] = 该客户端中该类别的样本数。
# 请用原始统计值替换下面的近似数据，或使用 --data 读取 CSV。
COUNTS = np.array([
    # c0    c1    c2   c3   c4   c5   c6   c7   c8   c9
    [   0,    0,    0,   0,   0,   0,   0,   0,   0,   0],  # client 0
    [   0,    0,  110,   0,   0,   0,  24,   0,   0,   0],
    [   0,    0,    0,   0,  35,   0,   0,  16,   0,  16],
    [   0,    0,    0,   0,   0,   0,  42,   0,   0,   0],
    [   0,    0,    0,   0,  58,   0,   0,   0,   0,  17],
    [4000,    0,    0,   0,   0,   0,   0,   0,   0,   0],
    [   0, 2300,   15,   0,   0,   0,   0,   0,   0,   0],
    [   0,  250,   70,   0,  85,   0,   0,   0,   0,  18],
    [  90,    0, 1320,   0,   0,  45,   0,   0,   0,   0],
    [   0,    0,    0, 120,   0,   0,   0,   0,   0,   0],
    [  60,    0,    0, 900,   0,   0,   0, 120,   0,   0],
    [  35,    0,    0, 125,   0,   0,   0,   0,  28,  22],
    [   0,    0,    0, 110,   0,   0,   0,   0,   0,   0],
    [   0,    0,   95,   0,   0, 450,   0,   0,   0,   0],
    [  70,    0,   80,   0,   0,   0, 100,   0,   0,   0],
    [ 320,    0,    0,   0,   0,   0,   0,   0,   0,  20],
    [   0,  170,   75,   0,   0,   0,   0,   0,   0,   0],
    [   0,    0,    0,   0,   0,   0,   0,  25,   0,   0],
    [   0,    0,    0,   0, 620,   0,  30,   0,   0,   0],
    [ 440,  220,    0,  55,   0,   0,   0,   0,  20,  20],
], dtype=float)


# ==================== 2. 外观：在这里集中修改 ====================
STYLE = {
    "figsize": (8.4, 6.8),     # 英寸；导出尺寸 = figsize × dpi
    "dpi": 300,
    "elev": 30,                # 仰角
    "azim": -45,               # 水平旋转角
    "box_aspect": (1, 1, 0.75), # 三个坐标轴的视觉长度比例
    "bar_width": 0.58,          # 类别方向柱宽，单位为类别间距
    "bar_depth": 0.55,          # 客户端方向柱宽
    "zmax": 4000,              # 数据更大时会自动上调，避免截断
    "z_tick_step": 500,
    "font_family": "DejaVu Sans",
    "font_size": 10,
    "tick_size": 8,
    "x_label": "Class",
    "y_label": "Client ID",
    "z_label": "Count",
    "label_pad": 8,
    "grid_color": "#d4d4d4",
    "grid_width": 0.65,
    "pane_color": "#f4f4f4",
    "shade": True,             # 立体明暗
    "projection": "persp",     # 改为 'ortho' 可用正交投影
}

# 每列（类别）的颜色固定，不随客户端或柱高变化。
CLASS_COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd",
    "#8c564b", "#e377c2", "#7f7f7f", "#bcbd22", "#17becf",
]


def validate_counts(counts):
    """拒绝空矩阵、缺失值和负样本数，避免静默画出错误的数据。"""
    counts = np.asarray(counts, dtype=float)
    if counts.ndim != 2 or 0 in counts.shape:
        raise ValueError("counts must be a nonempty 2D [client, class] matrix")
    if not np.isfinite(counts).all() or (counts < 0).any():
        raise ValueError("counts must contain finite, nonnegative values")
    return counts


def plot_distribution(counts=COUNTS, style=None, class_colors=None):
    """返回 (fig, ax)，可在其他 Python 程序中继续设置坐标轴或添加标注。

    示例：
        fig, ax = plot_distribution(my_counts, {"elev": 30, "azim": -60})
        ax.set_zlabel("Number of samples")
        fig.savefig("my_figure.pdf", bbox_inches="tight")
    """
    import matplotlib.pyplot as plt

    counts = validate_counts(counts)
    config = {**STYLE, **(style or {})}
    colors = CLASS_COLORS if class_colors is None else list(class_colors)
    if not colors:
        raise ValueError("class_colors must contain at least one color")
    if config["z_tick_step"] <= 0 or config["zmax"] <= 0:
        raise ValueError("zmax and z_tick_step must be positive")
    if config["bar_width"] <= 0 or config["bar_depth"] <= 0:
        raise ValueError("bar_width and bar_depth must be positive")
    if len(config["box_aspect"]) != 3 or not all(
        np.isfinite(value) and value > 0 for value in config["box_aspect"]
    ):
        raise ValueError("box_aspect must contain three finite, positive values")

    with matplotlib.rc_context({
        "font.family": config["font_family"],
        "font.size": config["font_size"],
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
        "svg.fonttype": "none",
    }):
        fig = plt.figure(figsize=config["figsize"], facecolor="white")
        ax = fig.add_subplot(111, projection="3d")
        fig.subplots_adjust(left=0.02, right=0.90, bottom=0.08, top=0.98)

        # meshgrid 的形状与 counts 一致：先客户端行，再类别列。
        num_clients, num_classes = counts.shape
        xx, yy = np.meshgrid(np.arange(num_classes), np.arange(num_clients))
        nonzero = counts.ravel() > 0
        x = xx.ravel()[nonzero]
        y = yy.ravel()[nonzero]
        heights = counts.ravel()[nonzero]
        if heights.size:
            # 一次绘制所有柱子，使 Matplotlib 能统一处理三维遮挡关系。
            ax.bar3d(
                x - config["bar_width"] / 2,
                y - config["bar_depth"] / 2,
                np.zeros_like(heights),
                config["bar_width"], config["bar_depth"], heights,
                color=[colors[int(cls) % len(colors)] for cls in x],
                shade=config["shade"], edgecolor="none", linewidth=0,
                zsort="average",
            )

        ax.set_xlim(-0.6, num_classes - 0.4)
        ax.set_ylim(-0.6, num_clients - 0.4)
        z_step = config["z_tick_step"]
        z_top = max(config["zmax"], np.ceil(counts.max() / z_step) * z_step)
        ax.set_zlim(0, z_top)
        ax.set_xticks(np.arange(num_classes))
        ax.set_yticks(np.arange(num_clients))
        if z_top / z_step <= 12:
            ax.zaxis.set_major_locator(MultipleLocator(z_step))
        else:
            ax.zaxis.set_major_locator(MaxNLocator(nbins=8, integer=True))
        ax.set_xlabel(config["x_label"], labelpad=config["label_pad"])
        ax.set_ylabel(config["y_label"], labelpad=config["label_pad"])
        ax.set_zlabel(config["z_label"], labelpad=config["label_pad"])
        ax.tick_params(axis="both", labelsize=config["tick_size"], pad=0)
        ax.tick_params(axis="z", labelsize=config["tick_size"], pad=2)
        ax.view_init(elev=config["elev"], azim=config["azim"])
        ax.set_box_aspect(config["box_aspect"])
        ax.set_proj_type(config["projection"])
        ax.grid(True)

        for axis in (ax.xaxis, ax.yaxis, ax.zaxis):
            axis.set_pane_color(matplotlib.colors.to_rgba(config["pane_color"]))
            axis.line.set_color("#bdbdbd")
            axis.line.set_linewidth(0.65)
            # mplot3d 的网格细节目前通过 _axinfo 设置；其余为公开接口。
            axis._axinfo["grid"].update({
                "color": config["grid_color"],
                "linewidth": config["grid_width"],
                "linestyle": "-",
            })

    return fig, ax


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path,
                        help="Numeric CSV: rows=clients, columns=classes; no header/index")
    parser.add_argument("--output", type=Path,
                        default=Path(__file__).resolve().parent / "output",
                        help="Output directory")
    parser.add_argument("--formats", nargs="+", choices=["png", "pdf", "svg"],
                        default=["png", "pdf", "svg"])
    parser.add_argument("--elev", type=float, default=STYLE["elev"])
    parser.add_argument("--azim", type=float, default=STYLE["azim"])
    parser.add_argument("--box-aspect", type=float, nargs=3, default=STYLE["box_aspect"],
                        metavar=("X", "Y", "Z"),
                        help="Visual axis proportions, e.g. 1 1 1 for a taller box")
    parser.add_argument("--zmax", type=float, default=STYLE["zmax"])
    parser.add_argument("--dpi", type=int, default=STYLE["dpi"])
    parser.add_argument("--show", action="store_true",
                        help="Open an interactive Matplotlib window after export")
    args = parser.parse_args()
    if args.dpi <= 0:
        parser.error("--dpi must be positive")

    # 默认批量导出；--show 时保留本机 GUI backend，可用鼠标旋转。
    if not args.show:
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    try:
        counts = COUNTS if args.data is None else np.loadtxt(
            args.data, delimiter=",", comments="#", ndmin=2,
        )
        counts = validate_counts(counts)
        fig, ax = plot_distribution(counts, {
            "elev": args.elev, "azim": args.azim, "zmax": args.zmax,
            "box_aspect": tuple(args.box_aspect),
        })
    except (ValueError, OSError) as exc:
        parser.error(str(exc))

    args.output.mkdir(parents=True, exist_ok=True)
    source = (
        "VISUAL APPROXIMATION from screenshot; NOT recovered experimental data."
        if args.data is None else f"Source CSV: {args.data.resolve()}"
    )
    print(source)
    np.savetxt(args.output / "counts_used.csv", counts, delimiter=",", fmt="%.18g",
               header=f"{source}\nRows=Client ID (0-based); columns=Class (0-based).")
    # 保存时也设置字体类型：PDF 内嵌 TrueType，SVG 保留可编辑文字。
    with matplotlib.rc_context({"pdf.fonttype": 42, "svg.fonttype": "none"}):
        for extension in args.formats:
            destination = args.output / f"reconstructed_3d_histogram.{extension}"
            fig.savefig(destination, dpi=args.dpi, bbox_inches="tight",
                        pad_inches=0.16, facecolor="white",
                        bbox_extra_artists=[ax.xaxis.label, ax.yaxis.label, ax.zaxis.label])
            print(f"Saved: {destination}")
    if args.show:
        plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
