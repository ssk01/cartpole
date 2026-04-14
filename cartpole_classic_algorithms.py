"""
CartPole 经典算法实现 —— 从 1968 到 2013 的历史重现
=====================================================
1. BOXES (Michie & Chambers, 1968) — 最早的 CartPole AI，查表法
2. Actor-Critic (Barto, Sutton & Anderson, 1983) — 两个"神经元"
3. DQN 对比数据从已有的 dqn_demo.py 加载

运行方式: python3 cartpole_classic_algorithms.py
"""

import random
import numpy as np
import gymnasium as gym
import collections
import os

# ============================================================
# 公共工具：状态离散化 —— BOXES 和 Actor-Critic 共用
# ============================================================
# CartPole-v1 的 4 个连续状态变量：
#   x        : 小车位置      (范围约 -2.4 ~ 2.4)
#   x_dot    : 小车速度      (范围约 -inf ~ inf，实际常在 -3 ~ 3)
#   theta    : 杆子角度      (范围约 -0.209 ~ 0.209 rad ≈ ±12°)
#   theta_dot: 杆子角速度    (范围约 -inf ~ inf，实际常在 -3 ~ 3)
#
# 我们把连续空间切成离散的"格子"(boxes)：
#   x        → 3 bins: 左 / 中 / 右
#   x_dot    → 3 bins: 向左移 / 静止 / 向右移
#   theta    → 6 bins: 6 段不同倾斜角
#   theta_dot→ 3 bins: 向左倒 / 静止 / 向右倒
# 总共 3 × 3 × 6 × 3 = 162 个格子

# 每个变量的分割边界
# x: [-0.8, 0.8] 把位置分成 3 段
X_BINS = [-0.8, 0.8]
# x_dot: [-0.5, 0.5] 把速度分成 3 段
X_DOT_BINS = [-0.5, 0.5]
# theta: 6 段，更细的划分因为角度是最关键的状态
# 约 [-12°, -6°, -1°, 0°, 1°, 6°, 12°] → 6 个区间
THETA_BINS = [-0.1047, -0.0349, -0.0175, 0.0175, 0.0349, 0.1047]
# theta_dot: [-0.87, 0.87] 把角速度分成 3 段（约 ±50°/s）
THETA_DOT_BINS = [-0.87, 0.87]

# 各变量的 bin 数量
N_X = 3           # len(X_BINS) + 1
N_X_DOT = 3       # len(X_DOT_BINS) + 1
N_THETA = 7       # len(THETA_BINS) + 1  → 注意：6 个边界产生 7 个区间
N_THETA_DOT = 3   # len(THETA_DOT_BINS) + 1

# 格子总数（这里是 3×3×7×3 = 189，比原始 162 稍多，因为 theta 用 7 bins）
N_BOXES = N_X * N_X_DOT * N_THETA * N_THETA_DOT  # = 189


def digitize(value, bins):
    """把一个连续值映射到离散 bin 编号（从 0 开始）"""
    # np.digitize 返回 [1, len(bins)]，减 1 变成 [0, len(bins)-1]
    # 但我们需要 [0, len(bins)]，因为两端也各有一个区间
    return int(np.digitize(value, bins))


def get_box(state):
    """
    把 4 维连续状态映射到一个整数 box_id ∈ [0, N_BOXES-1]
    就像把 4 维坐标展平成 1 维索引（类似 C 语言多维数组的行优先展平）
    """
    x, x_dot, theta, theta_dot = state

    ix = digitize(x, X_BINS)                   # 0, 1, 2
    ix_dot = digitize(x_dot, X_DOT_BINS)       # 0, 1, 2
    itheta = digitize(theta, THETA_BINS)        # 0, 1, ..., 6
    itheta_dot = digitize(theta_dot, THETA_DOT_BINS)  # 0, 1, 2

    # 行优先展平：box_id = ix * (3*7*3) + ix_dot * (7*3) + itheta * 3 + itheta_dot
    box_id = ix * (N_X_DOT * N_THETA * N_THETA_DOT) \
           + ix_dot * (N_THETA * N_THETA_DOT) \
           + itheta * N_THETA_DOT \
           + itheta_dot

    return box_id


# ============================================================
# 算法 1: BOXES (Michie & Chambers, 1968)
# ============================================================
# 历史背景：
#   Donald Michie 和 R.A. Chambers 在 1968 年的论文
#   "BOXES: An Experiment in Adaptive Control" 中提出了这个算法。
#   它是最早的 CartPole 平衡 AI 之一（当时叫 "pole-balancing"）。
#
# 核心思想极其简单：
#   1. 把连续状态空间切成离散的"格子"（boxes）
#   2. 每个格子里记录两个计数：选动作 0（左推）多少次成功，选动作 1（右推）多少次成功
#   3. 选成功次数更多的那个动作
#   4. 如果杆子倒了，就惩罚最近走过的那些格子-动作对
#
# 它本质上是一种"查表 + 统计"的方法，没有任何"学习率"或"梯度"的概念。
# 但它真的能学会平衡杆子！虽然学得慢，而且上限不高。

def train_boxes(n_episodes=1000, seed=42):
    """
    BOXES 算法训练

    核心数据结构：
      scores[box_id][action] = 该格子中选该动作的"信用分"
      正分 → 这个动作在这个格子里曾经帮助杆子存活
      负分 → 这个动作在这个格子里曾经导致杆子倒下
    """
    print("=" * 65)
    print("训练 BOXES 算法 (Michie & Chambers, 1968)")
    print("=" * 65)

    env = gym.make("CartPole-v1")
    rng = np.random.RandomState(seed)

    # --- 核心数据结构：每个格子 × 每个动作 的分数 ---
    # scores[box_id][action] 初始化为 0（没有先验偏好）
    scores = np.zeros((N_BOXES, 2), dtype=np.float64)

    # 探索率：以一定概率随机选动作，避免过早锁定在错误的动作上
    epsilon = 0.3  # 初始探索率 30%
    epsilon_decay = 0.997  # 每局衰减
    epsilon_min = 0.02     # 最低探索率

    all_rewards = []  # 记录每局的回报（存活步数）

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0.0

        # --- 记录本局走过的 (box, action) 轨迹 ---
        # 用于在杆子倒下时做"延迟惩罚"（credit assignment）
        trajectory = []

        while True:
            box = get_box(state)

            # --- 选动作：ε-greedy ---
            if rng.random() < epsilon:
                # 探索：随机选
                action = rng.randint(0, 2)
            else:
                # 利用：选分数更高的动作
                if scores[box][0] == scores[box][1]:
                    action = rng.randint(0, 2)  # 分数相同就随机
                else:
                    action = int(np.argmax(scores[box]))

            trajectory.append((box, action))

            # 执行动作
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            if done:
                if terminated:
                    # --- 杆子倒了！惩罚最近访问的格子-动作对 ---
                    # Michie 的关键洞察：不只惩罚最后一步，而是惩罚"最近"的若干步
                    # 因为导致失败的原因可能在好几步之前就已经埋下了
                    # 越近的步骤越可能是"罪魁祸首"，所以惩罚随距离衰减
                    n_blame = min(len(trajectory), 25)  # 最多回溯 25 步
                    for i in range(n_blame):
                        b, a = trajectory[-(i + 1)]
                        # 衰减惩罚：最近的惩罚最重，越远越轻
                        penalty = 1.0 * (0.9 ** i)
                        scores[b][a] -= penalty
                else:
                    # --- 达到最大步数（500 步），奖励最近的格子-动作对 ---
                    n_credit = min(len(trajectory), 50)
                    for i in range(n_credit):
                        b, a = trajectory[-(i + 1)]
                        scores[b][a] += 0.5 * (0.95 ** i)
                break
            else:
                # --- 杆子还活着！给当前格子-动作对加一点分 ---
                # 每存活一步就是一次小小的"成功"
                scores[box][action] += 0.1

            state = next_state

        # 衰减探索率
        epsilon = max(epsilon_min, epsilon * epsilon_decay)

        all_rewards.append(episode_reward)

        # 每 100 局打印进度
        if episode % 100 == 0:
            avg = np.mean(all_rewards[-100:])
            print(f"  Episode {episode:5d} | "
                  f"最近 100 局平均: {avg:6.1f} | "
                  f"本局: {episode_reward:5.0f} | "
                  f"ε: {epsilon:.3f}")

    env.close()
    print(f"  训练完成。最终 100 局平均回报: {np.mean(all_rewards[-100:]):.1f}")
    return all_rewards


# ============================================================
# 算法 2: Barto-Sutton-Anderson Actor-Critic (1983)
# ============================================================
# 历史背景：
#   Andrew Barto, Richard Sutton 和 Charles Anderson 在 1983 年发表了
#   "Neuronlike Adaptive Elements That Can Solve Difficult Learning
#    Control Problems"。这篇论文引入了两个"类神经元"单元：
#
#   1. ACE (Adaptive Critic Element) = Critic
#      估计每个状态的价值 V(s)，使用 TD(0) 学习
#      类似于"评论家"：告诉你"当前状态有多好/多差"
#
#   2. ASE (Adaptive Search Element) = Actor
#      学习策略：在每个状态下应该选什么动作
#      类似于"演员"：根据评论家的反馈调整自己的表演
#
# 这就是现代 Actor-Critic 方法的鼻祖！
# 后来的 A2C, A3C, PPO, SAC 等都是这个框架的后代。
#
# 与 BOXES 的区别：
#   - BOXES 只统计"成功/失败次数"，没有"价值"的概念
#   - Actor-Critic 有明确的 V(s) 估计，用 TD error 来指导学习
#   - Actor-Critic 的数学基础更扎实（强化学习理论）

def train_actor_critic(n_episodes=1000, seed=42):
    """
    Barto-Sutton-Anderson Actor-Critic 算法训练

    两个核心组件：
      ACE (Critic): V(s) — 状态价值函数，一个 N_BOXES 维的向量
      ASE (Actor):  w(s,a) — 策略权重，一个 N_BOXES × 2 的矩阵

    Barto 1983 原文用 r=0（存活）和 r=-1（失败），这里等价地使用
    Gymnasium 的 r=1（存活）和终止时 V(s')=0 的约定。
    """
    print("\n" + "=" * 65)
    print("训练 Actor-Critic 算法 (Barto, Sutton & Anderson, 1983)")
    print("=" * 65)

    env = gym.make("CartPole-v1")
    rng = np.random.RandomState(seed)

    # === Critic (ACE): 状态价值函数 ===
    # V[box_id] = 该状态的估计价值
    # 初始化为 0：一开始我们对所有状态一无所知
    V = np.zeros(N_BOXES, dtype=np.float64)

    # === Actor (ASE): 策略权重 ===
    # w[box_id][action] = 该状态下选该动作的"偏好程度"
    # 通过 sigmoid 函数转化为概率：P(action=1) = sigmoid(w[box][1] - w[box][0])
    w = np.zeros((N_BOXES, 2), dtype=np.float64)

    # === 超参数 ===
    alpha = 0.3     # Critic 学习率：V(s) 更新步长
    beta = 0.3      # Actor 学习率：w(s,a) 更新步长
    gamma = 0.99    # 折扣因子：未来奖励的衰减
    # 注意这些超参数比 DQN 大得多（0.3 vs 0.0001）
    # 因为这里没有神经网络，不需要小步长来避免震荡

    # 资格迹(eligibility traces)：记录最近访问的状态，用于加速信用分配
    # 这是 Barto 1983 论文的一个关键技巧
    lambda_trace = 0.8  # 迹衰减系数

    all_rewards = []

    for episode in range(1, n_episodes + 1):
        state, _ = env.reset(seed=seed + episode)
        episode_reward = 0.0

        # 每局重置资格迹
        # Barto 原文在每局开始时清零迹
        e_critic = np.zeros(N_BOXES, dtype=np.float64)    # Critic 的资格迹
        e_actor = np.zeros((N_BOXES, 2), dtype=np.float64)  # Actor 的资格迹

        box = get_box(state)

        while True:
            # === Actor: 选动作 ===
            # 用 sigmoid 把权重差转化为概率
            # P(action=1) = 1 / (1 + exp(-(w[box][1] - w[box][0])))
            # 如果 w[box][1] > w[box][0]，就更倾向选动作 1（右推）
            preference = w[box][1] - w[box][0]
            # 防止 overflow
            preference = np.clip(preference, -500, 500)
            prob_right = 1.0 / (1.0 + np.exp(-preference))
            # 随机决策（概率性策略，不是贪心）
            if rng.random() < prob_right:
                action = 1
            else:
                action = 0

            # 标记 Actor 资格迹：只对选中的动作标记
            # Barto 原文：e_actor(s,a) 在选中 a 时递增，表示"这个状态-动作对有资格获得更新"
            e_actor *= gamma * lambda_trace     # 先衰减所有迹
            e_actor[box, action] += (1.0 - prob_right) if action == 1 else prob_right
            # 上面这行很关键：加的是"surprise"——选了不太可能的动作时，迹更大
            # 这等价于策略梯度 ∂log π(a|s) / ∂w

            # 标记 Critic 资格迹
            e_critic *= gamma * lambda_trace    # 先衰减
            e_critic[box] += 1.0

            # === 执行动作 ===
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            episode_reward += reward

            # === Critic: 计算 TD error ===
            # TD error δ = r + γ * V(s') - V(s)
            # 这是 Actor-Critic 的核心信号！
            # δ > 0 → "比预期好"，鼓励当前的动作
            # δ < 0 → "比预期差"，惩罚当前的动作
            # δ ≈ 0 → "跟预期差不多"，不怎么更新
            if terminated:
                # 杆子倒了：终止状态 V(s')=0，reward=1（Gymnasium 给的）
                # TD error = 1 + γ*0 - V(s) = 1 - V(s)
                # 如果 V(s) 已经学到这个状态价值约 1（即将结束），δ ≈ 0
                # 如果 V(s) 很高（本以为能活很久），δ 很负 → 强烈惩罚
                td_error = reward + 0.0 - V[box]
            else:
                next_box = get_box(next_state)
                # 正常的一步或 truncated：r=1, V(s') 正常估计
                td_error = reward + gamma * V[next_box] - V[box]

            # === Critic 更新：V(s) += α × δ × e(s) ===
            # 用资格迹实现了类似 TD(λ) 的效果
            # 所有最近访问过的状态都会被更新，不只是当前状态
            V += alpha * td_error * e_critic

            # === Actor 更新：w(s,a) += β × δ × e(s,a) ===
            # TD error 就是"强化信号"(reinforcement signal)
            # 如果 δ > 0（比预期好），就增强最近选过的动作
            # 如果 δ < 0（比预期差），就减弱最近选过的动作
            w += beta * td_error * e_actor

            if done:
                break

            box = next_box
            state = next_state

        all_rewards.append(episode_reward)

        # 每 100 局打印进度
        if episode % 100 == 0:
            avg = np.mean(all_rewards[-100:])
            print(f"  Episode {episode:5d} | "
                  f"最近 100 局平均: {avg:6.1f} | "
                  f"本局: {episode_reward:5.0f}")

    env.close()
    print(f"  训练完成。最终 100 局平均回报: {np.mean(all_rewards[-100:]):.1f}")
    return all_rewards


# ============================================================
# 画对比图
# ============================================================
def plot_comparison(boxes_rewards, ac_rewards):
    """
    把两种经典算法的学习曲线画在同一张图上，
    并用水平线标注 DQN 的典型性能作为参考。
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # macOS 中文字体：PingFang SC
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "STHeiti",
                                        "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 9), sharex=True)
    fig.suptitle("CartPole 经典算法对比 —— 从 1968 到 2013",
                 fontsize=16, fontweight="bold")

    episodes = np.arange(1, len(boxes_rewards) + 1)
    window = 50  # 滑动平均窗口

    # --- 上图：原始回报 + 滑动平均 ---
    # BOXES (1968)
    ax1.plot(episodes, boxes_rewards, alpha=0.15, color="steelblue", linewidth=0.5)
    boxes_smooth = np.convolve(boxes_rewards, np.ones(window) / window, mode="valid")
    ax1.plot(np.arange(window, len(boxes_rewards) + 1), boxes_smooth,
             color="steelblue", linewidth=2.5,
             label="BOXES (Michie & Chambers, 1968)")

    # Actor-Critic (1983)
    ax1.plot(episodes, ac_rewards, alpha=0.15, color="darkorange", linewidth=0.5)
    ac_smooth = np.convolve(ac_rewards, np.ones(window) / window, mode="valid")
    ax1.plot(np.arange(window, len(ac_rewards) + 1), ac_smooth,
             color="darkorange", linewidth=2.5,
             label="Actor-Critic (Barto, Sutton & Anderson, 1983)")

    # DQN 参考线（典型训练 600 ep 后能达到 475+）
    ax1.axhline(y=475, color="crimson", linestyle="--", linewidth=1.5,
                label="DQN 解决标准 (Mnih et al., 2013): 475", alpha=0.8)
    ax1.axhline(y=500, color="gray", linestyle=":", linewidth=1,
                label="CartPole-v1 最大回报: 500", alpha=0.5)

    ax1.set_ylabel("每局回报 (Episode Reward)", fontsize=12)
    ax1.set_title("学习曲线（50 局滑动平均）", fontsize=13)
    ax1.legend(fontsize=10, loc="upper left")
    ax1.grid(True, alpha=0.3)
    ax1.set_ylim(0, 520)

    # --- 下图：100 局滑动平均（更清晰的趋势对比）---
    window100 = 100
    if len(boxes_rewards) >= window100:
        boxes_avg100 = np.convolve(boxes_rewards, np.ones(window100) / window100,
                                    mode="valid")
        ac_avg100 = np.convolve(ac_rewards, np.ones(window100) / window100,
                                 mode="valid")
        ep100 = np.arange(window100, len(boxes_rewards) + 1)

        ax2.plot(ep100, boxes_avg100, color="steelblue", linewidth=2.5,
                 label="BOXES (1968)")
        ax2.plot(ep100, ac_avg100, color="darkorange", linewidth=2.5,
                 label="Actor-Critic (1983)")
        ax2.axhline(y=475, color="crimson", linestyle="--", linewidth=1.5,
                    alpha=0.8, label="DQN 解决标准: 475")

    ax2.set_xlabel("Episode", fontsize=12)
    ax2.set_ylabel("100 局滑动平均回报", fontsize=12)
    ax2.set_title("长期学习趋势（100 局滑动平均）", fontsize=13)
    ax2.legend(fontsize=10, loc="upper left")
    ax2.grid(True, alpha=0.3)
    ax2.set_ylim(0, 520)

    plt.tight_layout()

    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                              "cartpole_history_comparison.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n对比图已保存到 {save_path}")


# ============================================================
# 打印总结表格
# ============================================================
def print_summary(boxes_rewards, ac_rewards):
    """打印最终对比总结表"""
    print("\n" + "=" * 65)
    print("总结对比表")
    print("=" * 65)

    headers = ["算法", "年份", "最终100局平均", "最高100局平均", "首次>200局"]
    row_fmt = "{:<35s} {:>6s} {:>14s} {:>14s} {:>12s}"

    print(row_fmt.format(*headers))
    print("-" * 85)

    for name, year, rewards in [
        ("BOXES (Michie & Chambers)", "1968", boxes_rewards),
        ("Actor-Critic (Barto, Sutton, Anderson)", "1983", ac_rewards),
        ("DQN (Mnih et al.)", "2013", None),
    ]:
        if rewards is not None:
            final_avg = np.mean(rewards[-100:])
            # 计算 100 局滑动平均的最大值
            if len(rewards) >= 100:
                avgs = np.convolve(rewards, np.ones(100) / 100, mode="valid")
                best_avg = np.max(avgs)
            else:
                best_avg = np.mean(rewards)
            # 找到第一次 100 局平均超过 200 的位置
            first_200 = "未达到"
            if len(rewards) >= 100:
                for i, a in enumerate(avgs):
                    if a >= 200:
                        first_200 = f"Ep {i + 100}"
                        break

            print(row_fmt.format(
                name, year,
                f"{final_avg:.1f}",
                f"{best_avg:.1f}",
                first_200
            ))
        else:
            # DQN：给出典型数据作为参考
            print(row_fmt.format(
                name, year,
                "~475+",
                "~500",
                "~Ep 200"
            ))

    print("-" * 85)
    print("\n注：DQN 数据为典型值，实际取决于超参数和随机种子。")
    print("    BOXES 和 Actor-Critic 的上限受限于状态离散化的精度，")
    print("    不太可能稳定达到 DQN 的水平。但它们的思想对后续研究影响深远！")

    # 打印历史意义
    print("\n" + "=" * 65)
    print("历史脉络")
    print("=" * 65)
    print("""
  1968  BOXES (Michie & Chambers)
        ↓  最早的 CartPole AI，纯查表法
        ↓  关键创新：把连续空间离散化 + 延迟信用分配
        ↓
  1983  Actor-Critic (Barto, Sutton & Anderson)
        ↓  引入 TD learning + 两个"神经元"（Actor + Critic）
        ↓  关键创新：TD error 作为强化信号，资格迹加速学习
        ↓  这是现代 Actor-Critic 方法（A2C, PPO, SAC）的鼻祖
        ↓
  2013  DQN (Mnih et al.)
        ↓  深度神经网络 + 经验回放 + 目标网络
        ↓  关键创新：端到端从像素到动作，不需要手工特征
        ↓  开启了深度强化学习时代
        ↓
  2015  DQN Nature 论文
        ↓  49 个 Atari 游戏，超越人类水平
        ↓
  今天  PPO, SAC, AlphaGo, ChatGPT/RLHF ...
        都站在这些先驱的肩膀上
""")


# ============================================================
# 主入口
# ============================================================
if __name__ == "__main__":
    # 固定随机种子
    SEED = 42
    random.seed(SEED)
    np.random.seed(SEED)

    N_EPISODES = 1000

    # 1. 训练 BOXES
    boxes_rewards = train_boxes(n_episodes=N_EPISODES, seed=SEED)

    # 2. 训练 Actor-Critic
    ac_rewards = train_actor_critic(n_episodes=N_EPISODES, seed=SEED)

    # 3. 画对比图
    plot_comparison(boxes_rewards, ac_rewards)

    # 4. 打印总结
    print_summary(boxes_rewards, ac_rewards)
