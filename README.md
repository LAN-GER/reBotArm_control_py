# reBotArm Control Python

基于 Pinocchio + motorbridge 的 reBotArm 机械臂 Python 控制库，支持达妙（DM）、RobStride（RS）等多种电机，提供正逆运动学、轨迹规划、末端位姿控制以及长时耐久压测。

## 主要特性

- **配置驱动**：所有硬件参数集中在 `config/rebotarm_*.yaml`，切换电机类型只需改 `config/rebotarm.yaml` 中的 `hardware_yaml`
- **分组控制**： arm / gripper 可独立切换 MIT、位置-速度、速度等模式
- **运动学与控制**：
  - 正逆运动学（IK）、雅可比、力矩前馈
  - SE(3) 测地线轨迹规划（线性 / 梯形 / min-jerk）
  - 末端位姿闭环控制器 `RebotArmEndPose`
- **长时耐久压测**：`example/11_reach_stress_test.py` 支持 A↔B 往返、温度监测、超温/卡死/电机故障保护
- **实时可视化**：温度曲线 matplotlib 实时绘制，测试数据自动保存 CSV/PNG

## 硬件支持

| 电机类型 | 配置文件 | 说明 |
|---|---|---|
| 达妙（Damiao） | `config/rebotarm_dm.yaml` | DM 系列关节电机 |
| RobStride | `config/rebotarm_rs.yaml` | 本末/RobStride 系列 |

切换配置：

```python
# config/rebotarm.yaml
hardware_yaml: "rebotarm_rs.yaml"   # 或 rebotarm_dm.yaml
```

## 环境要求

- Python 3.10
- CAN 接口（如 `can0`）已配置，RobStride 默认 bitrate 1 Mbps
- 依赖：`pin`, `motorbridge`, `meshcat`, `numpy`, `matplotlib`, `pyyaml`

## 安装

使用 uv（推荐）：

```bash
uv sync
```

或使用 pip：

```bash
python -m venv .venv
source .venv/bin/activate
pip install -e .
```

## 快速开始

```python
from reBotArm_control_py.actuator import RebotArm

arm = RebotArm()
arm.connect()
arm.arm.mode_mit()
arm.arm.enable()

def loop(ref, dt):
    ref.arm.send_mit(pos, vel, kp, kd, tau)

arm.start_control_loop(loop)
# ...
arm.disconnect()
```

更多示例见 `example/`。

## 示例说明

| 示例 | 说明 |
|---|---|
| `0x01damiao_test.py` / `0x01rs06_test.py` | 单电机基础测试 |
| `2_zero_and_read.py` | 读取电机状态、设置零点 |
| `3_mit_control.py` | MIT 力控模式测试 |
| `4_pos_vel_control.py` | 位置-速度模式测试 |
| `5_fk_test.py` / `6_ik_test.py` | 正逆运动学测试 |
| `7_arm_ik_control.py` | 末端位姿 IK 控制 |
| `8_arm_traj_control.py` | 末端轨迹跟踪 |
| `9_gravity_compensation.py` | 关节重力补偿 |
| `10_gravity_compensation_lock.py` | 零重力锁定 |
| `11_reach_stress_test.py` | **长时耐久压测 + 温度监测** |
| `plot_stress_data.py` | 压测 CSV 离线绘图 |

## 耐久压测 (`11_reach_stress_test.py`)

### 用法

```bash
# 默认：A=机械零点，B=70% 臂展，3 次往返 + 休息 5 秒
uv run python example/11_reach_stress_test.py --disable-gripper

# 远点 70% 臂展，近点 10% 臂展，不休息，joint4 向下偏移 90°
uv run python example/11_reach_stress_test.py \
  --disable-gripper \
  --ratio-low 0.10 \
  --ratio-high 0.70 \
  --reps 1 \
  --rest 0 \
  --joint4-offset -90 \
  --max-temp 140
```

### 主要参数

| 参数 | 默认值 | 说明 |
|---|---|---|
| `--ratio-low` | `0.0` | A 点臂展比例，`0` 表示机械零点 |
| `--ratio-high` | `0.70` | B 点臂展比例 |
| `--period` | `4.0` | 单次 A→B→A 周期（秒） |
| `--dwell` | `0.0` | 端点停留时间（秒） |
| `--reps` | `3` | 每组往返次数 |
| `--rest` | `5.0` | 每组结束后在机械零点休息时间（秒） |
| `--joint4-offset` | `0.0` | A→B 过程中 joint4 偏移角度（度） |
| `--max-temp` | `80.0` | MOS 温度超温保护阈值（°C） |
| `--temp-rate` | `1.0` | 温度采样频率（Hz） |
| `--arm-mode` | `mit` | 手臂控制模式：`mit` / `posvel` |
| `--disable-gripper` | `False` | 禁用夹爪（只测 arm） |
| `--no-plot` | `False` | 关闭实时绘图 |
| `--no-temp` | `False` | 关闭温度读取 |
| `--debug` | `False` | 打印调试信息 |

### 安全保护

1. **超温保护**：任一电机 MOS 温度超过 `--max-temp` 立即触发紧急停止
2. **卡死检测**：每 3 秒检查一次位置变化，连续 6 秒位移 < 0.02 rad 判定为卡住，触发紧急停止
3. **电机故障检测**：读取电机 `status_code`，非零状态码触发紧急停止
4. **Ctrl+C**：安全停止并保存已记录的温度数据

### 压测后绘图

如果程序崩溃未保存图片，可用：

```bash
uv run python example/plot_stress_data.py --csv stress_test_data.csv --output curve.png
```

## 本次改进汇总

### 1. 新增长时耐久压测脚本

- `example/11_reach_stress_test.py`
  - A↔B 分组往返，支持自定义往返次数、周期、停留时间
  - 组间自动回到**机械零点**休息
  - 支持 A→B 过程中 joint4 同步偏移，减小远端负载
  - 使用 `TrajProfile.LINEAR` 避免 min-jerk 端点低速爬行

### 2. 新增温度监测与绘图工具

- 后台线程持续读取各电机 MOS 温度
- 5 秒滑动平均平滑曲线
- 实时 matplotlib 温度曲线（默认显示最近 1 小时）
- 自动保存 `stress_test_data.csv` 和 `temperature_curve.png`
- `example/plot_stress_data.py` 支持离线补绘

### 3. 末端位姿控制器增强

- `reBotArm_control_py/controllers/rebotarm_endpose_controller.py`
  - 启动时增加 0.3s 模式切换→使能、使能→控制循环的间隔，缓解 CAN 总线拥堵
  - `_send_loop` 起点加入平滑过渡，避免轨迹跳变
  - `move_to_traj` 支持 `joint_offsets` 参数，可在轨迹中渐变偏移指定关节
  - 使用 `_q_target` 作为轨迹起点，避免 CAN 反馈旧帧导致轨迹偏移
  - `end()` 先停止轨迹发送线程再 `safe_home()`，避免抖动

### 4. CAN 总线健壮性提升

- `reBotArm_control_py/actuator/rebotarm.py`
  - `mode_mit` / `mode_pos_vel` / `mode_vel` / `enable` 遇到 `No buffer space available` 时自动重试最多 5 次，按指数退避
  - 大幅减少 CAN TX 队列瞬满导致的初始化失败

### 5. 重力补偿微调

- `example/9_gravity_compensation.py`：joint3 重力补偿系数从 `1.5` 调整为 `1.6`

### 6. 仓库清理

- 更新 `.gitignore`，忽略 `__pycache__`、虚拟环境、IDE 配置及压测运行时 CSV/PNG 产物
- 移除已跟踪的 pycache 二进制文件

## CAN 总线配置示例

RobStride 默认 1 Mbps：

```bash
sudo ip link set can0 down
sudo ip link set can0 up type can bitrate 1000000 txqueuelen 1000
```

若仍出现 `socketcan write failed: No buffer space available`，可尝试增大发送队列：

```bash
sudo ip link set can0 txqueuelen 1000
```

## 注意事项

- `ratio-low` / `ratio-high` 为 **0.0~1.0 之间的比例**，不是百分比。例如 70% 臂展应写 `0.70`，而不是 `70`
- 压测前请确认机械臂运动空间无障碍物，并设置合理的 `--max-temp`
- 首次运行建议先用 `--reps 1 --rest 0` 短行程验证

## 许可证

MIT
