#!/usr/bin/env python3
"""
CartPole 物理引擎 —— 揭开 gym.step() 的黑盒

这个文件从零实现了 CartPole 的物理模拟，不依赖任何外部库。
所有物理方程和常数与 Gymnasium CartPole-v1 完全一致。

核心思想：
    给定当前状态 (x, x_dot, theta, theta_dot) 和动作 (0 或 1)，
    通过牛顿力学计算下一时刻的状态。
    这就是所谓的 "world model"（世界模型）。

运行方式：
    python3 cartpole_physics.py

参考文献：
    Florian, R.V., "Correct equations for the dynamics of the cart-pole system"
    https://coneural.org/florian/papers/05_cart_pole.pdf
"""

import math
import sys
import os
import random
import time

# ============================================================================
#  第一部分：物理常数（与 Gymnasium CartPole-v1 完全一致）
# ============================================================================

GRAVITY = 9.8          # 重力加速度 (m/s²)
CART_MASS = 1.0        # 小车质量 (kg)
POLE_MASS = 0.1        # 杆的质量 (kg)
TOTAL_MASS = CART_MASS + POLE_MASS  # 总质量 = 1.1 kg
POLE_HALF_LENGTH = 0.5 # 杆的半长 (m)，注意：这是半长，完整杆长 = 1.0m
POLE_MASS_LENGTH = POLE_MASS * POLE_HALF_LENGTH  # 杆质量 × 半长 = 0.05
FORCE_MAG = 10.0       # 推力大小 (N)，向左或向右
TAU = 0.02             # 时间步长 (s)，即每一步模拟 0.02 秒

# 终止条件
X_THRESHOLD = 2.4      # 小车位置超过 ±2.4m 时游戏结束
THETA_THRESHOLD = 12 * 2 * math.pi / 360  # 杆倾斜超过 ±12° 时游戏结束（≈ 0.2095 弧度）


# ============================================================================
#  第二部分：核心物理引擎 —— 这就是 gym.step() 背后在做的事情
# ============================================================================

def step(state, action):
    """
    CartPole 的物理引擎（world model）

    这个函数是整个文件的核心。它实现了一个完整的 "状态转移函数"：
        next_state = f(current_state, action)

    在强化学习中，这就是环境的 "dynamics" 或 "transition function"。
    DQN 不需要知道这个函数的具体形式，它通过试错来学习最优策略。
    但理解这个函数，能帮你理解"环境"到底在做什么。

    输入：
        state: (x, x_dot, theta, theta_dot)
            x        — 小车在轨道上的位置 (m)，0 是中心
            x_dot    — 小车的速度 (m/s)，正 = 向右
            theta    — 杆与竖直方向的夹角 (rad)，正 = 顺时针（向右倒）
            theta_dot — 杆的角速度 (rad/s)
        action: 0（向左推）或 1（向右推）

    输出：
        next_state: 下一帧的 (x, x_dot, theta, theta_dot)
        reward: 奖励（只要没倒就是 1.0）
        done: 是否结束（杆倒了或小车出界）
    """
    x, x_dot, theta, theta_dot = state

    # ── 第 1 步：确定施加的力 ──
    # action=1 → 向右推 +10N；action=0 → 向左推 -10N
    force = FORCE_MAG if action == 1 else -FORCE_MAG

    # ── 第 2 步：预计算三角函数 ──
    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    # ── 第 3 步：计算加速度（这是最核心的物理方程）──
    #
    # 这里用的是刚体力学中"小车上的倒立摆"模型。
    # 推导来源：拉格朗日力学（Lagrangian mechanics）
    # 参考论文：Florian 2007, "Correct equations for the dynamics of the cart-pole system"
    #
    # 直觉理解：
    #   - 杆和小车通过铰链连接，它们的运动是耦合的
    #   - 推小车时，杆会因为惯性而倾斜
    #   - 杆倾斜时，重力会让它继续倒下
    #   - 我们需要同时求解小车加速度和杆角加速度

    # ── 3a：中间变量 temp ──
    # temp = (F + m_pole * L * θ̇² * sin(θ)) / M_total
    #
    # 物理含义：
    #   F                                → 外力（我们施加的推力）
    #   m_pole * L * θ̇² * sin(θ)        → 杆旋转产生的离心力在水平方向的分量
    #   除以 M_total                     → 按照总质量归一化
    #
    # 这个 temp 可以理解为"如果忽略杆的角加速度，小车大致的加速度"
    temp = (
        force + POLE_MASS_LENGTH * (theta_dot ** 2) * sin_theta
    ) / TOTAL_MASS

    # ── 3b：杆的角加速度 θ̈ ──
    # θ̈ = (g * sin(θ) - cos(θ) * temp) / (L * (4/3 - m_pole * cos²(θ) / M_total))
    #
    # 物理含义（分子）：
    #   g * sin(θ)      → 重力让杆倒下的力矩（杆越倾斜，力矩越大）
    #   - cos(θ) * temp → 小车加速度通过铰链对杆产生的反作用力矩
    #
    # 物理含义（分母）：
    #   L * (4/3 - ...)  → 杆的有效转动惯量
    #   4/3 来自均匀杆绕端点的转动惯量 I = (1/3)mL²
    #   减去的项是小车-杆耦合效应的修正
    theta_acc = (GRAVITY * sin_theta - cos_theta * temp) / (
        POLE_HALF_LENGTH * (4.0 / 3.0 - POLE_MASS * (cos_theta ** 2) / TOTAL_MASS)
    )

    # ── 3c：小车的加速度 ẍ ──
    # ẍ = temp - m_pole * L * θ̈ * cos(θ) / M_total
    #
    # 物理含义：
    #   temp                              → 之前算的"粗略加速度"
    #   - m_pole * L * θ̈ * cos(θ) / M_total → 杆角加速度对小车的反作用力
    #
    # 注意这里的耦合：θ̈ 的计算用到了 temp，而 ẍ 又用到了 θ̈
    # 这就是为什么不能简单地独立计算两个加速度
    x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS

    # ── 第 4 步：欧拉积分（Euler integration）──
    # 这是最简单的数值积分方法：
    #   新位置 = 旧位置 + 速度 × 时间步长
    #   新速度 = 旧速度 + 加速度 × 时间步长
    #
    # 时间步长 TAU = 0.02 秒，即每秒模拟 50 帧
    #
    # 更精确的方法（如 Runge-Kutta）会更准确，
    # 但欧拉法足够简单，而且 Gymnasium 用的就是这个。

    # 位置更新：x_new = x + v * dt
    x_new = x + TAU * x_dot
    # 速度更新：v_new = v + a * dt
    x_dot_new = x_dot + TAU * x_acc
    # 角度更新：θ_new = θ + ω * dt
    theta_new = theta + TAU * theta_dot
    # 角速度更新：ω_new = ω + α * dt
    theta_dot_new = theta_dot + TAU * theta_acc

    next_state = (x_new, x_dot_new, theta_new, theta_dot_new)

    # ── 第 5 步：判断是否结束 ──
    # 两种失败情况：
    #   1. 小车跑出边界：|x| > 2.4
    #   2. 杆倒下：|θ| > 12°（约 0.2095 弧度）
    done = bool(
        x_new < -X_THRESHOLD
        or x_new > X_THRESHOLD
        or theta_new < -THETA_THRESHOLD
        or theta_new > THETA_THRESHOLD
    )

    # 奖励：只要还没倒，就给 +1（鼓励尽可能久地保持平衡）
    reward = 1.0

    return next_state, reward, done


def reset(seed=None):
    """
    重置环境到初始状态

    与 Gymnasium 一致：每个状态变量从 [-0.05, 0.05] 均匀随机采样
    这意味着杆初始时几乎是直立的，只有微小的随机扰动
    """
    if seed is not None:
        random.seed(seed)
    return (
        random.uniform(-0.05, 0.05),  # x: 小车位置
        random.uniform(-0.05, 0.05),  # x_dot: 小车速度
        random.uniform(-0.05, 0.05),  # theta: 杆角度
        random.uniform(-0.05, 0.05),  # theta_dot: 杆角速度
    )


# ============================================================================
#  第三部分：验证 —— 确认我们的实现和 Gymnasium 完全一致
# ============================================================================

def verify_against_gym():
    """
    验证我们的物理引擎和 Gymnasium 的输出完全一致。
    如果 Gymnasium 没有安装，跳过验证。
    """
    try:
        import gymnasium as gym
        import numpy as np
    except ImportError:
        print("(跳过验证：未安装 gymnasium)")
        return True

    print("=" * 60)
    print("  验证：与 Gymnasium CartPole-v1 对比")
    print("=" * 60)

    env = gym.make("CartPole-v1")
    env.reset(seed=42)

    # 设置相同的初始状态
    test_state = (0.03, -0.02, 0.01, 0.04)
    env.unwrapped.state = np.array(test_state, dtype=np.float64)

    all_match = True
    for action in [0, 1, 1, 0, 1, 0, 0, 1]:
        # 我们的实现
        our_next, our_reward, our_done = step(test_state, action)

        # Gymnasium 的实现
        gym_obs, gym_reward, gym_terminated, _, _ = env.step(action)
        gym_next = tuple(gym_obs.astype(float))

        # 比较
        # 容差说明：我们用 math.cos/sin，Gymnasium 用 numpy.cos/sin，
        # 浮点实现略有差异（约 1e-9 量级），这是正常的数值误差
        match = all(abs(a - b) < 1e-6 for a, b in zip(our_next, gym_next))
        status = "MATCH" if match else "MISMATCH!"
        if not match:
            all_match = False

        print(f"  action={action}  [{status}]")
        print(f"    我们: x={our_next[0]:.6f}  v={our_next[1]:.6f}  "
              f"θ={our_next[2]:.6f}  ω={our_next[3]:.6f}")
        print(f"    Gym:  x={gym_next[0]:.6f}  v={gym_next[1]:.6f}  "
              f"θ={gym_next[2]:.6f}  ω={gym_next[3]:.6f}")

        # 用我们的结果继续（保持同步）
        test_state = our_next
        env.unwrapped.state = np.array(our_next, dtype=np.float64)

    env.close()

    if all_match:
        print("\n  *** 所有结果完全匹配！我们的物理引擎与 Gymnasium 一致 ***\n")
    else:
        print("\n  *** 存在不匹配，请检查 ***\n")

    return all_match


# ============================================================================
#  第四部分：ASCII 可视化
# ============================================================================

def render_ascii(state, action_label="", step_num=0, total_reward=0):
    """
    用 ASCII 字符画出 CartPole 的当前状态

    可视化说明：
        |    ← 杆（会根据 theta 倾斜）
        ●    ← 铰链（杆和小车的连接点）
      ===●===  ← 小车（会根据 x 左右移动）
    ________________  ← 轨道
    """
    x, x_dot, theta, theta_dot = state
    theta_deg = math.degrees(theta)

    # 轨道宽度（字符数）
    track_width = 60
    track_center = track_width // 2

    # 小车在轨道上的位置（将 x 从 [-2.4, 2.4] 映射到 [0, track_width]）
    cart_pos = int(track_center + (x / X_THRESHOLD) * (track_width // 2))
    cart_pos = max(3, min(track_width - 4, cart_pos))

    # 杆的倾斜方向（用字符近似表示）
    pole_height = 4  # 杆的高度（字符行数）

    # 计算杆顶端的水平偏移（基于 theta）
    # sin(theta) 给出水平偏移的方向和大小
    tip_offset = math.sin(theta) * pole_height * 2  # 放大以便可视化

    lines = []

    # 状态信息（带中文说明）
    lines.append("")
    lines.append(f"  步数: {step_num:4d}   累计奖励: {total_reward:.0f}   {action_label}")
    lines.append(f"  ┌─────────────────────────────────────────────────────┐")
    lines.append(f"  │ x = {x:+8.4f} m    (小车位置，0=中心)              │")
    lines.append(f"  │ v = {x_dot:+8.4f} m/s  (小车速度，+=向右)            │")
    lines.append(f"  │ θ = {theta_deg:+8.4f}°   (杆角度，+=向右倒)           │")
    lines.append(f"  │ ω = {theta_dot:+8.4f} r/s  (杆角速度)                │")
    lines.append(f"  └─────────────────────────────────────────────────────┘")

    # 绘制杆
    for i in range(pole_height, 0, -1):
        # 杆每一行的水平偏移
        offset = int(tip_offset * (i / pole_height))
        line = [' '] * track_width
        pole_x = cart_pos + offset
        if 0 <= pole_x < track_width:
            # 根据倾斜程度选择字符
            if abs(theta_deg) < 2:
                line[pole_x] = '|'
            elif theta_deg > 0:
                line[pole_x] = '/'  if i > pole_height // 2 else '/'
            else:
                line[pole_x] = '\\' if i > pole_height // 2 else '\\'
        lines.append('  ' + ''.join(line))

    # 绘制小车
    cart_line = [' '] * track_width
    # 小车体 ===●===
    cart_chars = "===O==="
    start = cart_pos - len(cart_chars) // 2
    for i, ch in enumerate(cart_chars):
        pos = start + i
        if 0 <= pos < track_width:
            cart_line[pos] = ch
    lines.append('  ' + ''.join(cart_line))

    # 绘制轨道
    track_line = '_' * track_width
    lines.append('  ' + track_line)

    # 边界标记
    boundary_line = [' '] * track_width
    left_b = int(track_center - (track_width // 2))
    right_b = int(track_center + (track_width // 2)) - 1
    boundary_line[max(0, left_b)] = '|'
    boundary_line[min(track_width - 1, right_b)] = '|'
    boundary_line[track_center] = '^'
    lines.append('  ' + ''.join(boundary_line))
    lines.append('  ' + ' ' * left_b + '-2.4' + ' ' * (track_center - left_b - 4) +
                 '0.0' + ' ' * (right_b - track_center - 3) + '+2.4')

    return '\n'.join(lines)


def clear_screen():
    """清屏"""
    os.system('cls' if os.name == 'nt' else 'clear')


# ============================================================================
#  第五部分：单步演示 —— 展示物理方程的计算过程
# ============================================================================

def demo_single_step():
    """
    详细展示一步物理计算的全过程，让你看到每个方程在做什么
    """
    print("\n" + "=" * 60)
    print("  单步物理计算演示")
    print("  ——看看 gym.step() 背后到底发生了什么")
    print("=" * 60)

    state = (0.0, 0.0, 0.03, 0.0)  # 杆微微向右倾斜 (约 1.7°)
    action = 1  # 向右推

    x, x_dot, theta, theta_dot = state
    force = FORCE_MAG if action == 1 else -FORCE_MAG

    print(f"\n  当前状态:")
    print(f"    x     = {x:.4f} m     (小车在中心)")
    print(f"    x_dot = {x_dot:.4f} m/s   (小车静止)")
    print(f"    θ     = {theta:.4f} rad   (杆微微向右倾 ≈ {math.degrees(theta):.2f}°)")
    print(f"    θ_dot = {theta_dot:.4f} rad/s (杆没有角速度)")
    print(f"\n  动作: {action} → 向右推 F = {force:+.1f} N")

    cos_theta = math.cos(theta)
    sin_theta = math.sin(theta)

    print(f"\n  ── 计算过程 ──")
    print(f"\n  cos(θ) = {cos_theta:.6f}")
    print(f"  sin(θ) = {sin_theta:.6f}")

    temp = (force + POLE_MASS_LENGTH * (theta_dot ** 2) * sin_theta) / TOTAL_MASS
    print(f"\n  temp = (F + m_p * L * ω² * sin(θ)) / M_total")
    print(f"       = ({force:.1f} + {POLE_MASS_LENGTH:.2f} × {theta_dot**2:.4f} × {sin_theta:.6f}) / {TOTAL_MASS:.1f}")
    print(f"       = {temp:.6f}")

    theta_acc_numer = GRAVITY * sin_theta - cos_theta * temp
    theta_acc_denom = POLE_HALF_LENGTH * (4.0/3.0 - POLE_MASS * (cos_theta**2) / TOTAL_MASS)
    theta_acc = theta_acc_numer / theta_acc_denom

    print(f"\n  θ̈ = (g * sin(θ) - cos(θ) * temp) / (L * (4/3 - m_p * cos²(θ) / M_total))")
    print(f"     分子 = {GRAVITY:.1f} × {sin_theta:.6f} - {cos_theta:.6f} × {temp:.6f}")
    print(f"          = {theta_acc_numer:.6f}")
    print(f"     分母 = {POLE_HALF_LENGTH:.1f} × (1.333 - {POLE_MASS:.1f} × {cos_theta**2:.6f} / {TOTAL_MASS:.1f})")
    print(f"          = {theta_acc_denom:.6f}")
    print(f"     θ̈   = {theta_acc:.6f} rad/s²")

    x_acc = temp - POLE_MASS_LENGTH * theta_acc * cos_theta / TOTAL_MASS
    print(f"\n  ẍ = temp - m_p * L * θ̈ * cos(θ) / M_total")
    print(f"    = {temp:.6f} - {POLE_MASS_LENGTH:.2f} × {theta_acc:.6f} × {cos_theta:.6f} / {TOTAL_MASS:.1f}")
    print(f"    = {x_acc:.6f} m/s²")

    x_new = x + TAU * x_dot
    x_dot_new = x_dot + TAU * x_acc
    theta_new = theta + TAU * theta_dot
    theta_dot_new = theta_dot + TAU * theta_acc

    print(f"\n  ── 欧拉积分 (dt = {TAU}s) ──")
    print(f"    x_new     = {x:.4f} + {TAU} × {x_dot:.4f}     = {x_new:.6f}")
    print(f"    v_new     = {x_dot:.4f} + {TAU} × {x_acc:.6f} = {x_dot_new:.6f}")
    print(f"    θ_new     = {theta:.4f} + {TAU} × {theta_dot:.4f}     = {theta_new:.6f}")
    print(f"    ω_new     = {theta_dot:.4f} + {TAU} × {theta_acc:.6f} = {theta_dot_new:.6f}")

    print(f"\n  ── 结果 ──")
    print(f"    新状态: ({x_new:.6f}, {x_dot_new:.6f}, {theta_new:.6f}, {theta_dot_new:.6f})")
    print(f"\n  直觉解读:")
    print(f"    - 向右推了小车，小车开始向右加速 (ẍ = {x_acc:.4f})")
    print(f"    - 但杆因为惯性，反而稍微向右倒得更多了 (θ̈ = {theta_acc:.4f})")
    print(f"    - 这就是控制的难点：推力和杆的运动方向相反！")
    print()


# ============================================================================
#  第六部分：交互式游戏模式
# ============================================================================

def play_interactive():
    """
    在终端中手动玩 CartPole
    用 ← → 方向键控制小车
    """
    print("\n" + "=" * 60)
    print("  交互模式 —— 手动控制 CartPole")
    print("=" * 60)
    print("\n  操作说明：")
    print("    a 或 ← : 向左推")
    print("    d 或 → : 向右推")
    print("    q      : 退出")
    print("\n  目标：保持杆不倒！看你能坚持多少步。")
    print("\n  按 Enter 开始...")

    try:
        input()
    except EOFError:
        return

    state = reset(seed=42)
    total_reward = 0
    step_num = 0
    done = False

    # 尝试使用非阻塞输入
    try:
        import tty
        import termios
        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        can_raw = True
    except (ImportError, termios.error):
        can_raw = False

    try:
        while not done:
            clear_screen()
            action_label = ""
            print(render_ascii(state, action_label, step_num, total_reward))
            print("\n  [a/←] 向左推    [d/→] 向右推    [q] 退出")

            # 读取按键
            if can_raw:
                tty.setraw(fd)
                ch = sys.stdin.read(1)
                termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)

                if ch == 'q' or ch == '\x03':  # q 或 Ctrl+C
                    break
                elif ch == 'a' or ch == 'h':
                    action = 0
                    action_label = "← 向左推 (-10N)"
                elif ch == 'd' or ch == 'l':
                    action = 1
                    action_label = "→ 向右推 (+10N)"
                elif ch == '\x1b':  # 方向键序列
                    tty.setraw(fd)
                    seq = sys.stdin.read(2)
                    termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)
                    if seq == '[D':  # 左
                        action = 0
                        action_label = "← 向左推 (-10N)"
                    elif seq == '[C':  # 右
                        action = 1
                        action_label = "→ 向右推 (+10N)"
                    else:
                        continue
                else:
                    continue
            else:
                # 降级模式：用 input()
                try:
                    key = input("  输入 a(左) d(右) q(退出): ").strip().lower()
                except EOFError:
                    break
                if key == 'q':
                    break
                elif key == 'a':
                    action = 0
                    action_label = "← 向左推 (-10N)"
                elif key == 'd':
                    action = 1
                    action_label = "→ 向右推 (+10N)"
                else:
                    continue

            state, reward, done = step(state, action)
            total_reward += reward
            step_num += 1

            if done:
                clear_screen()
                print(render_ascii(state, action_label, step_num, total_reward))
                x, _, theta, _ = state
                print("\n  *** 游戏结束! ***")
                if abs(x) > X_THRESHOLD:
                    print(f"  原因：小车出界 (x = {x:.4f}，阈值 = ±{X_THRESHOLD})")
                else:
                    print(f"  原因：杆倒了 (θ = {math.degrees(theta):.2f}°，阈值 = ±12°)")
                print(f"  总步数: {step_num}，总奖励: {total_reward:.0f}")
                print()

    finally:
        if can_raw:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ============================================================================
#  第七部分：自动运行模式 —— 观察物理演化
# ============================================================================

def run_random_policy(max_steps=50):
    """
    随机策略：每步随机选择向左或向右
    可以观察到：随机策略很快就会失败
    """
    print("\n" + "=" * 60)
    print("  自动运行：随机策略")
    print("  （每步随机选择方向，观察状态如何变化）")
    print("=" * 60)

    state = reset(seed=42)
    total_reward = 0

    print(f"\n  {'步':>4}  {'动作':>6}  {'x':>8}  {'v':>8}  {'θ(°)':>8}  {'ω':>8}  {'奖励':>4}")
    print("  " + "-" * 56)
    print(f"  {'0':>4}  {'初始':>6}  {state[0]:>8.4f}  {state[1]:>8.4f}  "
          f"{math.degrees(state[2]):>8.4f}  {state[3]:>8.4f}  {'-':>4}")

    for i in range(1, max_steps + 1):
        action = random.randint(0, 1)
        state, reward, done = step(state, action)
        total_reward += reward
        action_str = "→ 右推" if action == 1 else "← 左推"

        print(f"  {i:>4}  {action_str}  {state[0]:>8.4f}  {state[1]:>8.4f}  "
              f"{math.degrees(state[2]):>8.4f}  {state[3]:>8.4f}  {reward:>4.0f}")

        if done:
            x, _, theta, _ = state
            if abs(x) > X_THRESHOLD:
                reason = f"小车出界 (x={x:.2f})"
            else:
                reason = f"杆倒下 (θ={math.degrees(theta):.1f}°)"
            print(f"\n  游戏结束！原因: {reason}")
            print(f"  坚持了 {i} 步，总奖励: {total_reward:.0f}")
            break
    else:
        print(f"\n  达到最大步数 {max_steps}，总奖励: {total_reward:.0f}")

    return total_reward


def run_heuristic_policy(max_steps=200):
    """
    启发式策略：杆往哪边倒，就往哪边推
    这是人类直觉的控制策略：
        - θ > 0（杆向右倒）→ 向右推（action=1）
        - θ < 0（杆向左倒）→ 向左推（action=0）

    可以观察到：这个简单策略比随机策略好得多！
    """
    print("\n" + "=" * 60)
    print("  自动运行：启发式策略（杆往哪边倒就往哪边推）")
    print("  策略逻辑: if θ > 0 then 右推 else 左推")
    print("=" * 60)

    state = reset(seed=42)
    total_reward = 0

    print(f"\n  {'步':>4}  {'动作':>6}  {'x':>8}  {'v':>8}  {'θ(°)':>8}  {'ω':>8}  {'奖励':>4}")
    print("  " + "-" * 56)
    print(f"  {'0':>4}  {'初始':>6}  {state[0]:>8.4f}  {state[1]:>8.4f}  "
          f"{math.degrees(state[2]):>8.4f}  {state[3]:>8.4f}  {'-':>4}")

    for i in range(1, max_steps + 1):
        x, x_dot, theta, theta_dot = state

        # 启发式策略：杆往哪边倒就往哪边推
        action = 1 if theta > 0 else 0

        state, reward, done = step(state, action)
        total_reward += reward
        action_str = "→ 右推" if action == 1 else "← 左推"

        # 只打印前 30 步和最后几步
        if i <= 30 or done or i == max_steps:
            print(f"  {i:>4}  {action_str}  {state[0]:>8.4f}  {state[1]:>8.4f}  "
                  f"{math.degrees(state[2]):>8.4f}  {state[3]:>8.4f}  {reward:>4.0f}")
        elif i == 31:
            print(f"  {'...':>4}  {'...':>6}  {'...':>8}  {'...':>8}  "
                  f"{'...':>8}  {'...':>8}  {'...':>4}")

        if done:
            x, _, theta, _ = state
            if abs(x) > X_THRESHOLD:
                reason = f"小车出界 (x={x:.2f})"
            else:
                reason = f"杆倒下 (θ={math.degrees(theta):.1f}°)"
            print(f"\n  游戏结束！原因: {reason}")
            print(f"  坚持了 {i} 步，总奖励: {total_reward:.0f}")
            break
    else:
        print(f"\n  达到最大步数 {max_steps}，总奖励: {total_reward:.0f}")
        print("  启发式策略成功平衡了杆！")

    return total_reward


def run_animated_demo(max_steps=200, delay=0.1):
    """
    动画模式：用 ASCII 动画展示 CartPole 运行过程
    使用启发式策略自动控制
    """
    print("\n" + "=" * 60)
    print("  动画演示模式（启发式策略）")
    print("  按 Ctrl+C 可随时退出")
    print("=" * 60)
    print("\n  3 秒后开始...")
    time.sleep(2)

    state = reset(seed=42)
    total_reward = 0
    step_num = 0

    try:
        for i in range(1, max_steps + 1):
            x, x_dot, theta, theta_dot = state
            action = 1 if theta > 0 else 0
            action_label = "→ 向右推 (+10N)" if action == 1 else "← 向左推 (-10N)"

            state, reward, done = step(state, action)
            total_reward += reward
            step_num += 1

            clear_screen()
            print(render_ascii(state, action_label, step_num, total_reward))
            print("\n  模式: 自动（启发式策略）  按 Ctrl+C 退出")

            if done:
                x, _, theta, _ = state
                print(f"\n  *** 游戏结束! ***")
                if abs(x) > X_THRESHOLD:
                    print(f"  原因：小车出界 (x = {x:.4f})")
                else:
                    print(f"  原因：杆倒了 (θ = {math.degrees(theta):.2f}°)")
                print(f"  总步数: {step_num}，总奖励: {total_reward:.0f}")
                break

            time.sleep(delay)

    except KeyboardInterrupt:
        print(f"\n\n  手动停止。总步数: {step_num}，总奖励: {total_reward:.0f}")


# ============================================================================
#  第八部分：状态转移矩阵可视化 —— 理解 world model 的结构
# ============================================================================

def explore_state_transitions():
    """
    探索不同动作对状态的影响

    对比同一状态下：
        action=0（向左推）→ next_state_left
        action=1（向右推）→ next_state_right

    这就是 DQN 要学的东西：
        "在这个状态下，选 action=0 和 action=1，哪个未来更好？"
    """
    print("\n" + "=" * 60)
    print("  状态转移对比 —— world model 做了什么")
    print("=" * 60)
    print("\n  同一状态下，不同动作会导致完全不同的结果。")
    print("  这就是 DQN 需要学习评估的：哪个动作让未来更好？\n")

    test_cases = [
        ((0.0, 0.0, 0.0, 0.0), "杆完全直立，小车静止"),
        ((0.0, 0.0, 0.05, 0.0), "杆微微向右倾 (≈2.9°)"),
        ((0.0, 0.0, -0.1, 0.0), "杆向左倾 (≈5.7°)"),
        ((1.0, 0.5, 0.05, 0.1), "小车偏右且在移动，杆在向右倒"),
    ]

    for state, desc in test_cases:
        print(f"  ┌── 初始状态: {desc}")
        print(f"  │   (x={state[0]:.2f}, v={state[1]:.2f}, θ={math.degrees(state[2]):.2f}°, ω={state[3]:.2f})")

        next_left, _, done_left = step(state, 0)
        next_right, _, done_right = step(state, 1)

        print(f"  │")
        print(f"  ├── action=0 (← 左推):")
        print(f"  │   x={next_left[0]:+.4f}  v={next_left[1]:+.4f}  "
              f"θ={math.degrees(next_left[2]):+.4f}°  ω={next_left[3]:+.4f}")
        print(f"  │")
        print(f"  └── action=1 (→ 右推):")
        print(f"      x={next_right[0]:+.4f}  v={next_right[1]:+.4f}  "
              f"θ={math.degrees(next_right[2]):+.4f}°  ω={next_right[3]:+.4f}")

        # 分析差异
        dx = next_right[0] - next_left[0]
        dtheta = math.degrees(next_right[2] - next_left[2])
        print(f"      差异: Δx={dx:+.4f}m, Δθ={dtheta:+.4f}°")
        print()


# ============================================================================
#  主程序
# ============================================================================

def print_menu():
    print("\n" + "=" * 60)
    print("  CartPole 物理引擎演示")
    print("  —— 揭开 gym.step() 的黑盒")
    print("=" * 60)
    print("""
  这个程序从零实现了 CartPole 的物理模拟。
  所有物理方程和常数与 Gymnasium CartPole-v1 完全一致。

  World Model 的核心：
    (x, v, θ, ω) + action → (x', v', θ', ω')
    4 个状态变量 + 1 个动作 → 下一时刻的 4 个状态变量

  请选择模式：

    [1] 单步计算演示  — 详细展示一步物理计算的全过程
    [2] 状态转移对比  — 对比不同动作对状态的影响
    [3] 随机策略      — 观察随机控制下的状态变化
    [4] 启发式策略    — 观察简单策略如何控制杆
    [5] 动画演示      — ASCII 动画展示 CartPole 运行
    [6] 交互式游戏    — 手动控制 CartPole
    [7] 验证正确性    — 与 Gymnasium 对比验证
    [0] 退出
    """)


def main():
    # 显示物理常数
    print("\n  ╔══════════════════════════════════════════╗")
    print("  ║   CartPole 物理常数 (与 Gymnasium 一致)   ║")
    print("  ╠══════════════════════════════════════════╣")
    print(f"  ║  重力加速度  g    = {GRAVITY:>5.1f} m/s²           ║")
    print(f"  ║  小车质量    M    = {CART_MASS:>5.1f} kg             ║")
    print(f"  ║  杆质量      m    = {POLE_MASS:>5.1f} kg             ║")
    print(f"  ║  杆半长      L    = {POLE_HALF_LENGTH:>5.1f} m              ║")
    print(f"  ║  推力大小    F    = {FORCE_MAG:>5.1f} N              ║")
    print(f"  ║  时间步长    dt   = {TAU:>5.2f} s              ║")
    print(f"  ║  位置阈值    x_th = ±{X_THRESHOLD:.1f} m             ║")
    print(f"  ║  角度阈值    θ_th = ±12°               ║")
    print("  ╚══════════════════════════════════════════╝")

    while True:
        print_menu()
        try:
            choice = input("  请输入选项 [0-7]: ").strip()
        except (EOFError, KeyboardInterrupt):
            print("\n\n  再见！")
            break

        if choice == '1':
            demo_single_step()
        elif choice == '2':
            explore_state_transitions()
        elif choice == '3':
            run_random_policy(50)
        elif choice == '4':
            run_heuristic_policy(200)
        elif choice == '5':
            run_animated_demo(200, delay=0.08)
        elif choice == '6':
            play_interactive()
        elif choice == '7':
            verify_against_gym()
        elif choice == '0':
            print("\n  再见！\n")
            break
        else:
            print("\n  无效选项，请重新输入")

        if choice in ('1', '2', '3', '4', '7'):
            try:
                input("\n  按 Enter 继续...")
            except (EOFError, KeyboardInterrupt):
                print("\n\n  再见！")
                break


if __name__ == "__main__":
    main()
