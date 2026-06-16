#!/usr/bin/env python3
"""从 stress_test_data.csv 绘制 MOS 温度曲线。

用法::

    uv run python example/plot_stress_data.py
    uv run python example/plot_stress_data.py --csv stress_test_data.csv
    uv run python example/plot_stress_data.py --csv data.csv --output temp_curve.png --smooth 5
"""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import matplotlib.pyplot as plt
import numpy as np


def moving_average(data: np.ndarray, window: int) -> np.ndarray:
    """计算滑动平均，边界用有效窗口大小归一化。"""
    if window <= 1:
        return data
    cumsum = np.cumsum(np.insert(data, 0, 0))
    result = (cumsum[window:] - cumsum[:-window]) / window
    # 前 window-1 个点用逐步增大的窗口
    prefix = []
    for i in range(1, min(window, len(data) + 1)):
        prefix.append(np.mean(data[:i]))
    return np.concatenate([prefix, result])


def main() -> int:
    parser = argparse.ArgumentParser(description="绘制 MOS 温度曲线")
    parser.add_argument("--csv", type=str, default="stress_test_data.csv", help="输入 CSV 文件路径")
    parser.add_argument("--output", type=str, default="temperature_curve.png", help="输出 PNG 文件路径")
    parser.add_argument("--smooth", type=int, default=1, help="滑动平均窗口大小（点数，默认 1 不平滑）")
    parser.add_argument("--dpi", type=int, default=150, help="输出图片 DPI")
    args = parser.parse_args()

    csv_path = Path(args.csv)
    if not csv_path.exists():
        print(f"错误: 文件不存在: {csv_path.resolve()}")
        return 1

    # 读取 CSV
    with open(csv_path, "r") as f:
        header = f.readline().strip().split(",")

    if not header or header[0] != "time":
        print(f"错误: CSV 格式不正确，首行应为 time,xxx_mos,...")
        return 1

    joint_names = [h for h in header[1:] if h.endswith("_mos")]
    if not joint_names:
        print("错误: CSV 中未找到 MOS 温度列")
        return 1

    data = np.loadtxt(csv_path, delimiter=",", skiprows=1)
    if data.ndim == 1:
        data = data.reshape(1, -1)

    t_arr = data[:, 0]
    joint_cols = {name: i + 1 for i, name in enumerate(joint_names)}

    # 绘图
    matplotlib.use("Agg")
    fig, ax = plt.subplots(1, 1, figsize=(12, 6))

    colors = plt.cm.tab10(np.linspace(0, 1, len(joint_names)))
    for i, name in enumerate(joint_names):
        col = joint_cols[name]
        raw = data[:, col]
        # 去除可能的 nan
        valid = ~np.isnan(raw)
        if not np.any(valid):
            continue
        t_v = t_arr[valid]
        v_v = raw[valid]
        if args.smooth > 1:
            v_v = moving_average(v_v, args.smooth)
            # 平滑后长度可能变短，需对齐时间轴
            if len(v_v) < len(t_v):
                t_v = t_v[: len(v_v)]
        label = name.replace("_mos", "")
        ax.plot(t_v, v_v, label=label, color=colors[i], linewidth=1.2)

    ax.set_xlabel("Time (s)")
    ax.set_ylabel("MOS Temperature (°C)")
    ax.set_title("reBotArm Stress Test — MOS Temperature")
    ax.legend(loc="upper left", fontsize=9)
    ax.grid(True, alpha=0.3)
    plt.tight_layout()

    out_path = Path(args.output)
    plt.savefig(out_path, dpi=args.dpi, bbox_inches="tight")
    print(f"温度曲线已保存: {out_path.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
