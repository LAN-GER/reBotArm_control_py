#!/usr/bin/env python3
"""机械臂耐久测试 + 实时温度监测（基于 RebotArmEndPose）。

核心设计：
    - 运动模式：分组循环。A 点（默认机械零点）↔ B 点（远点）往返 reps 次，
                然后在 A 点休息 rest 秒，继续下一组。
    - A 点可调：默认机械零点；也可用 --ratio-low 指定臂展比例搜索 A 点位置。
    - 关节偏移：A→B 过程中可同步平滑偏移 4 号关节，减小远端压力。
    - 温度监测：读取 MOS 温度，5 秒滑动平均平滑，超温自动安全停止。
    - 数据记录：默认保留 12 小时数据，CSV + 温度曲线图自动保存。

用法::

    # 默认：A=机械零点，B=70%臂展，3次往返+休息5秒
    uv run python example/11_reach_stress_test.py --disable-gripper

    # joint4 减压（A→B 同步向下偏移 45°）
    uv run python example/11_reach_stress_test.py --disable-gripper --joint4-offset -45

    # 自定义往返次数和休息时间
    uv run python example/11_reach_stress_test.py --disable-gripper --reps 5 --rest 10

    # A 点改为 15% 臂展位置（非零点）
    uv run python example/11_reach_stress_test.py --disable-gripper --ratio-low 0.15

    # 纯运动测试，无温度，最平滑
    uv run python example/11_reach_stress_test.py --disable-gripper --arm-mode posvel --no-temp

    # 快速验证：1 次往返，不休息
    uv run python example/11_reach_stress_test.py --disable-gripper --reps 1 --rest 0

    # 调试模式（打印电机通信诊断信息）
    uv run python example/11_reach_stress_test.py --disable-gripper --debug

参数说明：
    --ratio-low     A 点臂展比例（默认 0 使用机械零点，>0 时按该比例搜索）
    --ratio-high    B 点臂展比例（默认 0.70 = 70%）
    --period        单次往返周期，秒（默认 4.0）
    --dwell         端点停留时间，秒（默认 0.5）
    --reps          每组往返次数（默认 3）
    --rest          每组结束后在 A 点休息时间，秒（默认 5）
    --joint4-offset A→B 时 4 号关节同步偏移角度，度（默认 0，向下一般为负值）
    --temp-rate     温度采样频率 Hz（默认 1）
    --max-duration  最长数据记录时长，秒（默认 43200 = 12 小时）
    --z-height      末端固定高度，米（默认 0.25）
    --max-temp      MOS 温度上限 °C（默认 80），超温触发安全停止
    --arm-mode      arm 控制模式：mit（默认）或 posvel
    --no-plot       禁用 matplotlib 实时绘图
    --no-temp       禁用温度读取（纯运动测试，最平滑）
    --disable-gripper   完全禁用夹爪
    --debug         启用调试输出，打印电机通信诊断

推荐组合：
    # 长时间耐久测试（拆夹爪 + joint4 减压 + 温度监测）
    uv run python example/11_reach_stress_test.py --disable-gripper --joint4-offset -45 --reps 5 --rest 10

    # 纯运动测试，无温度，最平滑
    uv run python example/11_reach_stress_test.py --disable-gripper --arm-mode posvel --no-temp

    #测试使用参数：
    uv run python 11_reach_stress_test.py --disable-gripper --joint4-offset -90 --max-temp 140 --ratio-high 70 --reps 1 --rest 0 --ratio-low 0.1

"""

from __future__ import annotations

import argparse
import signal
import sys
import threading
import time
from collections import deque
from pathlib import Path
from typing import Dict

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from reBotArm_control_py.actuator import RebotArm
from reBotArm_control_py.actuator.rebotarm import NoOpGroup
from reBotArm_control_py.controllers import RebotArmEndPose
from reBotArm_control_py.kinematics import (
    compute_fk,
    get_end_effector_frame_id,
    load_robot_model,
    pad_q_for_model,
    pos_rot_to_se3,
)
from reBotArm_control_py.kinematics.inverse_kinematics import solve_ik
from reBotArm_control_py.trajectory import TrajProfile

plt = None

_g_running = True


def _sigint_handler(signum, frame):
    global _g_running
    print("\n[stress_test] 收到停止信号，正在安全退出...")
    _g_running = False


signal.signal(signal.SIGINT, _sigint_handler)


# ---------------------------------------------------------------------------
# 辅助函数
# ---------------------------------------------------------------------------

def _find_reach_targets(model, data, end_frame_id, z_height, ratio_low, ratio_high):
    """搜索可达构型，返回两个目标位姿的位置坐标。"""
    configs = []
    for x in np.linspace(0.05, 0.70, 60):
        target = pos_rot_to_se3(np.array([x, 0.0, z_height]), rot=np.eye(3))
        result = solve_ik(
            model, data, end_frame_id, target,
            np.zeros(model.nq), controlled_joints=6,
        )
        if result.success:
            q_full = pad_q_for_model(model, result.q, 6)
            pos, _, _ = compute_fk(model, q_full)
            configs.append({
                "reach": float(np.linalg.norm(pos)),
                "q": result.q.copy()[:6],
            })
    configs.sort(key=lambda c: c["reach"])

    reach_min, reach_max = configs[0]["reach"], configs[-1]["reach"]
    target_A = reach_min + ratio_low * (reach_max - reach_min)
    target_B = reach_min + ratio_high * (reach_max - reach_min)

    config_A = min(configs, key=lambda c: abs(c["reach"] - target_A))
    config_B = min(configs, key=lambda c: abs(c["reach"] - target_B))

    pos_A = compute_fk(model, pad_q_for_model(model, config_A["q"], 6))[0]
    pos_B = compute_fk(model, pad_q_for_model(model, config_B["q"], 6))[0]
    return pos_A, pos_B


def _read_motor_data(rebotarm: RebotArm) -> tuple[Dict[str, float], Dict[str, int]]:
    """读取电机状态：温度和状态码。

    同时检查各电机是否正常通信（状态码非零表示保护/故障）。
    """
    for ctrl in rebotarm._ctrl_map.values():
        try:
            ctrl.poll_feedback_once()
        except Exception:
            pass

    temps: Dict[str, float] = {}
    statuses: Dict[str, int] = {}
    for jc in rebotarm._all_joints:
        st = rebotarm._motor_map[jc.name].get_state()
        if st is not None:
            temps[jc.name] = float(st.t_mos)
            statuses[jc.name] = int(st.status_code)
    return temps, statuses


# ---------------------------------------------------------------------------
# 主程序
# ---------------------------------------------------------------------------

def main() -> int:
    global _g_running, plt

    parser = argparse.ArgumentParser(description="机械臂臂展耐久测试 + 温度监测")
    parser.add_argument("--ratio-low", type=float, default=0.0, help="A 点臂展比例，0 表示机械零点（默认 0）")
    parser.add_argument("--ratio-high", type=float, default=0.70)
    parser.add_argument("--period", type=float, default=4.0)
    parser.add_argument("--dwell", type=float, default=0.5)
    parser.add_argument("--temp-rate", type=float, default=1.0, help="温度采样 Hz (默认 1)")
    parser.add_argument("--z-height", type=float, default=0.25)
    parser.add_argument("--max-temp", type=float, default=80.0)
    parser.add_argument("--max-duration", type=float, default=43200.0, help="最长数据记录时长，秒（默认 43200 = 12 小时）")
    parser.add_argument("--joint4-offset", type=float, default=0.0, help="到达B点后4号关节偏移角度，度（正值/负值取决于电机方向，默认 0）")
    parser.add_argument("--reps", type=int, default=3, help="每组 A→B 往返次数（默认 3）")
    parser.add_argument("--rest", type=float, default=5.0, help="每组结束后在零点休息时间，秒（默认 5）")
    parser.add_argument("--arm-mode", choices=["mit", "posvel"], default="mit", help="arm 控制模式")
    parser.add_argument("--no-plot", action="store_true")
    parser.add_argument("--no-temp", action="store_true")
    parser.add_argument("--disable-gripper", action="store_true")
    parser.add_argument("--debug", action="store_true", help="启用调试输出，打印电机通信状态")
    args = parser.parse_args()

    # matplotlib
    use_plot = not args.no_plot
    if use_plot:
        try:
            import matplotlib
            for backend in ("TkAgg", "Qt5Agg", "QtAgg", "GTK3Agg", "WXAgg"):
                try:
                    matplotlib.use(backend, force=True)
                    import matplotlib.pyplot as _plt
                    _plt.ion()
                    _test_fig = _plt.figure()
                    _plt.close(_test_fig)
                    plt = _plt
                    break
                except Exception:
                    continue
            else:
                matplotlib.use("Agg", force=True)
                import matplotlib.pyplot as _plt
                plt = _plt
        except Exception as e:
            print(f"[绘图] 初始化失败: {e}")
            use_plot = False
            plt = None

    # =====================================================================
    # 1. 搜索目标位姿
    # =====================================================================
    print("=" * 64)
    print("  机械臂臂展耐久测试 + 实时温度监测")
    print("=" * 64)

    model = load_robot_model()
    data = model.createData()
    end_frame_id = get_end_effector_frame_id(model)

    print(f"\n[1/4] 搜索目标位姿 (z={args.z_height:.2f}m)...")
    _, pos_B = _find_reach_targets(
        model, data, end_frame_id, args.z_height, 0.0, args.ratio_high
    )
    # A 点：默认机械零点，也可用 --ratio-low 指定搜索比例
    if args.ratio_low <= 0.0:
        q_zero = np.zeros(model.nq)
        pos_A = compute_fk(model, q_zero)[0]
        print(f"  目标 A (机械零点): [{pos_A[0]:.3f}, {pos_A[1]:.3f}, {pos_A[2]:.3f}] m")
    else:
        pos_A, _ = _find_reach_targets(
            model, data, end_frame_id, args.z_height, args.ratio_low, args.ratio_high
        )
        print(f"  目标 A ({args.ratio_low*100:.0f}%): [{pos_A[0]:.3f}, {pos_A[1]:.3f}, {pos_A[2]:.3f}] m")
    print(f"  目标 B ({args.ratio_high*100:.0f}%): [{pos_B[0]:.3f}, {pos_B[1]:.3f}, {pos_B[2]:.3f}] m")

    # 机械零点（组间休息用）
    q_zero = np.zeros(model.nq)
    pos_zero = compute_fk(model, q_zero)[0]
    print(f"  机械零点: [{pos_zero[0]:.3f}, {pos_zero[1]:.3f}, {pos_zero[2]:.3f}] m")

    # =====================================================================
    # 2. 连接机械臂 + 禁用夹爪
    # =====================================================================
    print("\n[2/4] 连接机械臂...")
    rebotarm = RebotArm()
    rebotarm.connect()

    if args.disable_gripper:
        rebotarm._groups["gripper"] = NoOpGroup()
        print("  [夹爪] 已禁用（替换为 NoOpGroup）")

    # =====================================================================
    # 3. 初始化 RebotArmEndPose
    # =====================================================================
    print("\n[3/4] 初始化 RebotArmEndPose...")
    ctrl = RebotArmEndPose(
        rebotarm,
        dt=0.01,
        profile=TrajProfile.LINEAR,
        arm_control_mode=args.arm_mode,
    )
    ctrl.start()
    print(f"  arm 模式: {args.arm_mode.upper()}")

    # =====================================================================
    # 4. 初始化绘图
    # =====================================================================
    fig = ax_mos = None
    lines_mos: Dict[str, object] = {}

    if args.disable_gripper and "arm" in rebotarm.groups:
        joint_names = list(rebotarm.groups["arm"].joint_names)
    else:
        joint_names = [jc.name for jc in rebotarm._all_joints]

    if use_plot and plt is not None:
        print("\n[4/4] 初始化温度绘图...")
        fig, ax_mos = plt.subplots(1, 1, figsize=(11, 5))
        fig.suptitle("reBotArm Stress Test — MOS Temperature Monitoring")
        colors = plt.cm.tab10(np.linspace(0, 1, len(joint_names)))
        for i, name in enumerate(joint_names):
            (line_mos,) = ax_mos.plot([], [], label=name, color=colors[i], linewidth=1.2)
            lines_mos[name] = line_mos
        ax_mos.set_ylabel("MOS Temperature (°C)")
        ax_mos.set_xlabel("Time (s)")
        ax_mos.legend(loc="upper left", fontsize=8)
        ax_mos.grid(True, alpha=0.3)
        plt.tight_layout()
        if plt.get_backend().lower() != "agg":
            try:
                plt.show(block=False)
            except Exception:
                pass
    else:
        print("\n[4/4] 绘图已禁用。")

    max_history = int(args.temp_rate * args.max_duration)
    smooth_window = max(1, int(args.temp_rate * 5))  # 约 5 秒滑动窗口
    time_history: deque = deque(maxlen=max_history)
    temp_mos_history: Dict[str, deque] = {name: deque(maxlen=max_history) for name in joint_names}
    temp_mos_smooth_history: Dict[str, deque] = {name: deque(maxlen=max_history) for name in joint_names}

    # =====================================================================
    # 5. 主循环：A → B → A → B ...
    # =====================================================================
    half_period = (args.period - 2.0 * args.dwell) / 2.0
    if half_period <= 0:
        half_period = 0.5

    print(f"\n开始测试 (Ctrl+C 停止)...")
    print(f"  每组: {args.reps} 次往返 (A→B→A) + {args.rest:.1f}s 休息")
    print(f"  周期: {args.period:.1f}s (运动 {half_period:.2f}s + 停留 {args.dwell:.1f}s)")
    print("-" * 64)

    dt_temp = 1.0 / args.temp_rate
    t_start = time.monotonic()
    t_last_plot = t_start
    t_last_print = t_start
    t_last_move_check = t_start
    cycle_count = 0
    group_count = 0
    q_prev_check: np.ndarray | None = None
    stuck_counter = 0

    # 启动温度后台线程（持续读取，和运动循环解耦）
    temp_thread: threading.Thread | None = None
    if not args.no_temp:
        def _temp_worker():
            while _g_running:
                t0 = time.monotonic()
                try:
                    temps, statuses = _read_motor_data(rebotarm)
                    _record_temps(temps, time_history, temp_mos_history, temp_mos_smooth_history, joint_names, t_start, smooth_window, args.max_temp, statuses)
                except Exception:
                    pass
                elapsed = time.monotonic() - t0
                sleep_time = dt_temp - elapsed
                if sleep_time > 0:
                    time.sleep(sleep_time)
        temp_thread = threading.Thread(target=_temp_worker, name="temp-worker", daemon=True)
        temp_thread.start()

    while _g_running:
        group_count += 1
        for rep in range(args.reps):
            if not _g_running:
                break

            # ---- A → B ----
            # dwell=0 时，joint4 偏移和运动一起做；dwell>0 时，偏移放到 dwell 期间
            traj_kwargs = {}
            if args.joint4_offset != 0.0 and args.dwell <= 0.0:
                traj_kwargs["joint_offsets"] = {3: np.radians(args.joint4_offset)}
                print(f"  [joint4] A→B 同步偏移 {args.joint4_offset:+.1f}° (dwell=0)")
            ctrl.move_to_traj(
                x=float(pos_B[0]), y=float(pos_B[1]), z=float(pos_B[2]),
                duration=half_period,
                **traj_kwargs,
            )
            while ctrl._moving and _g_running:
                time.sleep(0.02)

                # 运动卡死检测：每 3 秒检查一次位置变化
                t_now = time.monotonic()
                if t_now - t_last_move_check >= 3.0:
                    t_last_move_check = t_now
                    try:
                        q_now = rebotarm.get_positions()[:rebotarm.arm.num_joints]
                        if q_prev_check is not None:
                            max_delta = float(np.max(np.abs(q_now - q_prev_check)))
                            if max_delta < 0.02:  # 3 秒内位移 < 0.02 rad ≈ 1.1°
                                stuck_counter += 1
                                if stuck_counter >= 2:  # 连续 2 次（约 6 秒）几乎不动
                                    print(
                                        f"\n  {'='*52}\n"
                                        f"  [紧急停止] 机械臂卡住！{6}s 内几乎无位移\n"
                                        f"  {'='*52}"
                                    )
                                    _g_running = False
                            else:
                                stuck_counter = 0
                        q_prev_check = q_now.copy()
                    except Exception as e:
                        if args.debug:
                            print(f"  [debug] 位置检查失败: {e}")

            # dwell 期间完成 joint4 偏移（dwell>0 时）
            if args.joint4_offset != 0.0 and args.dwell > 0.0 and _g_running:
                print(f"  [joint4] dwell 期间偏移 {args.joint4_offset:+.1f}°")
                joint4_start = float(ctrl._q_target[3])
                joint4_end = joint4_start + np.radians(args.joint4_offset)
                n_steps = max(2, int(args.dwell / 0.02))
                for i in range(n_steps):
                    if not _g_running:
                        break
                    alpha = (i + 1) / n_steps
                    ctrl._q_target[3] = joint4_start + alpha * (joint4_end - joint4_start)
                    time.sleep(args.dwell / n_steps)

            time.sleep(args.dwell if args.joint4_offset == 0.0 else 0.0)

            if not _g_running:
                break

            # ---- B → A ----
            ctrl.move_to_traj(
                x=float(pos_A[0]), y=float(pos_A[1]), z=float(pos_A[2]),
                duration=half_period,
            )
            while ctrl._moving and _g_running:
                time.sleep(0.02)

                # 运动卡死检测
                t_now = time.monotonic()
                if t_now - t_last_move_check >= 3.0:
                    t_last_move_check = t_now
                    try:
                        q_now = rebotarm.get_positions()[:rebotarm.arm.num_joints]
                        if q_prev_check is not None:
                            max_delta = float(np.max(np.abs(q_now - q_prev_check)))
                            if max_delta < 0.02:
                                stuck_counter += 1
                                if stuck_counter >= 2:
                                    print(
                                        f"\n  {'='*52}\n"
                                        f"  [紧急停止] 机械臂卡住！6s 内几乎无位移\n"
                                        f"  {'='*52}"
                                    )
                                    _g_running = False
                            else:
                                stuck_counter = 0
                        q_prev_check = q_now.copy()
                    except Exception as e:
                        if args.debug:
                            print(f"  [debug] 位置检查失败: {e}")

            time.sleep(args.dwell)
            cycle_count += 1

        # ---- 组结束后回到机械零点休息 ----
        if _g_running:
            print(f"  [回零] 第 {group_count} 组完成，回到机械零点...")
            ctrl.move_to_traj(
                x=float(pos_zero[0]), y=float(pos_zero[1]), z=float(pos_zero[2]),
                duration=half_period,
            )
            while ctrl._moving and _g_running:
                time.sleep(0.02)

        if _g_running and args.rest > 0.0:
            print(f"  [休息] 在机械零点休息 {args.rest:.1f}s...")
            rest_start = time.monotonic()
            while _g_running and (time.monotonic() - rest_start < args.rest):
                time.sleep(0.02)

        # ---- 终端打印（每 1 秒）----
        t_now = time.monotonic()
        if t_now - t_last_print >= 1.0:
            t_last_print = t_now
            t_elapsed = t_now - t_start
            temps_list = [temp_mos_smooth_history[n][-1] for n in joint_names if temp_mos_smooth_history[n]]
            avg_temp = np.nanmean(temps_list) if temps_list else 0.0
            max_temp = np.nanmax(temps_list) if temps_list else 0.0
            print(
                f"  Group {group_count:3d} | Rep {cycle_count:4d} | "
                f"Avg {avg_temp:5.1f}°C | Max {max_temp:5.1f}°C | "
                f"Time {t_elapsed:6.1f}s"
            )

            # 绘图更新（只显示最近 1 小时数据，避免 matplotlib 卡顿）
            if use_plot and plt is not None and len(time_history) > 1:
                plot_max_points = int(args.temp_rate * 3600)  # 最近 1 小时
                t_arr = np.array(time_history)
                for name in joint_names:
                    if temp_mos_smooth_history[name]:
                        s_arr = np.array(temp_mos_smooth_history[name])
                        if len(t_arr) > plot_max_points:
                            lines_mos[name].set_data(t_arr[-plot_max_points:], s_arr[-plot_max_points:])
                        else:
                            lines_mos[name].set_data(t_arr, s_arr)
                ax_mos.relim()
                ax_mos.autoscale_view()
                fig.canvas.draw_idle()
                fig.canvas.flush_events()

    # =====================================================================
    # 6. 停止与保存
    # =====================================================================
    print("\n" + "-" * 64)
    print("正在安全停止...")

    ctrl.end()

    if time_history:
        csv_path = Path("stress_test_data.csv")
        with open(csv_path, "w") as f:
            header = ["time"] + [f"{n}_mos" for n in joint_names]
            f.write(",".join(header) + "\n")
            for i, t in enumerate(time_history):
                row = [f"{t:.3f}"]
                for name in joint_names:
                    row.append(f"{temp_mos_smooth_history[name][i]:.2f}")
                f.write(",".join(row) + "\n")
        print(f"  原始数据已保存: {csv_path.resolve()}")

    if use_plot and plt is not None and fig is not None:
        png_path = Path("temperature_curve.png")
        try:
            import matplotlib
            matplotlib.use("Agg", force=True)
        except Exception:
            pass
        plt.savefig(png_path, dpi=150, bbox_inches="tight")
        print(f"  温度曲线已保存: {png_path.resolve()}")
        if plt.get_backend().lower() != "agg":
            try:
                plt.ioff()
                plt.show()
            except Exception:
                pass

    print(f"\n测试完成！总循环: {cycle_count} 次")
    return 0


def _record_temps(temps, time_history, temp_mos_history, temp_mos_smooth_history, joint_names, t_start, smooth_window: int, max_temp: float = 80.0, statuses: Dict[str, int] | None = None):
    """记录温度到缓冲区，并计算滑动平均平滑值。"""
    global _g_running
    t_elapsed = time.monotonic() - t_start
    time_history.append(t_elapsed)
    for name in joint_names:
        if name in temps:
            temp_mos_history[name].append(temps[name])
        else:
            pm = temp_mos_history[name][-1] if temp_mos_history[name] else np.nan
            temp_mos_history[name].append(pm)

        # 滑动平均平滑
        vals = list(temp_mos_history[name])
        if len(vals) >= smooth_window:
            smooth = float(np.nanmean(vals[-smooth_window:]))
        elif vals:
            smooth = float(np.nanmean(vals))
        else:
            smooth = np.nan
        temp_mos_smooth_history[name].append(smooth)

    # 温度保护：超温时触发安全停止
    if _g_running:
        for name, val in temps.items():
            if val > max_temp:
                print(
                    f"\n  {'='*52}\n"
                    f"  [紧急停止] {name} MOS 温度 {val:.1f}°C 超过阈值 {max_temp:.1f}°C！\n"
                    f"  {'='*52}"
                )
                _g_running = False
                break

    # 电机状态码检查
    if _g_running and statuses:
        for name, code in statuses.items():
            if code != 0:
                print(
                    f"\n  {'='*52}\n"
                    f"  [电机异常] {name} 状态码 {code}，可能进入保护/故障模式！\n"
                    f"  {'='*52}"
                )
                _g_running = False
                break


if __name__ == "__main__":
    sys.exit(main())
