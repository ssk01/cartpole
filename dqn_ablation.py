"""
DQN 消融实验：有 vs 无 Target Network
======================================
对比实验：在 CartPole-v1 上分别训练「有 target 网络」和「无 target 网络」的 DQN，
然后画在同一张图上，直观展示 target network 对训练稳定性的影响。

「无 target 网络」版本：用同一个 policy_net 既选动作又计算 TD 目标，
即 target = r + gamma * max policy_net(s')，不再维护一份冻结的参数副本。

超参数差异说明：
  无 target 网络时，TD 目标随每次参数更新而变化（"自己追自己"），
  Q 值会无限膨胀（maximization bias + bootstrapping 正反馈循环）。
  经过多轮调参实验，无 target 版本采用：
    - 更高学习率 (5e-4 vs 1e-4)：趁 Q 值还没发散时快速学习
    - MSE loss (而非 Huber)：保留大 TD error 的梯度信号（2013 论文也没用 Huber）
    - 严格梯度裁剪 (clip_grad_value_=1)：限制每步参数变化幅度
    - Q 值裁剪 (clamp +-100)：防止 Q 值无限膨胀，打破正反馈循环
    - 更快的 epsilon 衰减 (1000 vs 3000)：尽快开始利用，获得更长的 episode
    - 更多训练 episode (2000 vs 600)：无 target 版本收敛更慢且不稳定
  这些调整让无 target 版本能够学到不错的策略（100ep 平均可达 300-470），
  但训练曲线明显更不稳定、有大幅振荡——正好展示了 target network 的核心价值。

用法：python3 -u dqn_ablation.py
"""

import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# ============================================================
# 共享超参数
# ============================================================
BATCH_SIZE = 128           # mini-batch 大小
GAMMA = 0.99               # 折扣因子
REPLAY_SIZE = 10000        # 回放缓冲区容量
MIN_REPLAY = 1000          # 开始训练前最少 transition 数
EPSILON_START = 0.9        # 初始探索率
EPSILON_END = 0.05         # 最终探索率
TAU = 0.005                # target 网络软更新系数（仅有 target 版本使用）
SEED = 0                   # 随机种子，两组实验用相同种子

# ============================================================
# 各版本独立的超参数
# ============================================================
# 有 target 网络版本（标准 DQN）— 与 dqn_demo.py 一致
WITH_TARGET_LR = 1e-4          # 学习率
WITH_TARGET_GRAD_CLIP = 100    # 梯度裁剪阈值（clip_grad_value_）
WITH_TARGET_USE_MSE = False    # False = Huber loss
WITH_TARGET_EPSILON_DECAY = 3000  # epsilon 衰减速度（步数）— 与 dqn_demo.py 一致
WITH_TARGET_MAX_EPISODES = 600    # 训练 episode 数

# 无 target 网络版本 — 经多轮调参实验确定
# 核心问题：没有 target 网络，Q 值会无限膨胀（max 操作 + 自举 = 正反馈循环）
# 解决方案：
#   - 更高学习率 + 严格梯度裁剪：在 Q 值失控前尽快学到有用信号
#   - MSE loss：保留大 TD error 的梯度方向信息
#   - Q 值裁剪：直接限制 Q 值上界，打破正反馈循环
#   - 更快 epsilon 衰减：尽快开始利用已学到的策略
#   - 更多 episode：收敛更慢，需要更多训练
NO_TARGET_LR = 5e-4            # 更高学习率
NO_TARGET_GRAD_CLIP = 1        # 严格梯度裁剪（clip_grad_value_=1）
NO_TARGET_USE_MSE = True       # MSE loss
NO_TARGET_EPSILON_DECAY = 1000 # 更快的 epsilon 衰减
NO_TARGET_MAX_EPISODES = 2000  # 更多训练 episode
NO_TARGET_Q_CLIP = 100         # Q 值裁剪上限（CartPole 理论 Q 值 ≈ 100）


# ============================================================
# Q 网络 — 和 dqn_demo.py 一致
# ============================================================
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, action_dim),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


# ============================================================
# 经验回放缓冲区 — 和 dqn_demo.py 一致
# ============================================================
class ReplayBuffer:
    def __init__(self, capacity: int):
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


# ============================================================
# epsilon-greedy 动作选择
# ============================================================
def select_action(state, policy_net, epsilon, n_actions, device):
    if random.random() < epsilon:
        return random.randrange(n_actions)
    with torch.no_grad():
        state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
        return policy_net(state_t).argmax(dim=1).item()


# ============================================================
# epsilon 衰减（基于步数的指数衰减）
# ============================================================
def get_epsilon(steps_done: int, decay: int) -> float:
    return EPSILON_END + (EPSILON_START - EPSILON_END) * \
        np.exp(-steps_done / decay)


# ============================================================
# 软更新 target 网络
# ============================================================
def soft_update(target_net, policy_net, tau):
    for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
        tp.data.copy_(tau * pp.data + (1.0 - tau) * tp.data)


# ============================================================
# 训练一步 — 支持有/无 target 网络两种模式
# ============================================================
def train_step(policy_net, target_net, optimizer, replay, device,
               use_mse=False, grad_clip=100, q_clip=None):
    states, actions, rewards, next_states, dones = replay.sample(BATCH_SIZE)

    states_t = torch.tensor(states, device=device)
    actions_t = torch.tensor(actions, device=device)
    rewards_t = torch.tensor(rewards, device=device)
    next_states_t = torch.tensor(next_states, device=device)
    dones_t = torch.tensor(dones, device=device)

    # Q(s, a) — 当前网络对已选动作的 Q 值
    q_values = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)

    # TD 目标 y = r + gamma * max Q(s', a')
    # 关键区别：用哪个网络计算 max Q(s', a')
    with torch.no_grad():
        if target_net is not None:
            # 有 target 网络：用冻结的 target_net 计算（标准 DQN）
            next_q_max = target_net(next_states_t).max(1)[0]
        else:
            # 无 target 网络：用同一个 policy_net 计算（消融版本）
            next_q_max = policy_net(next_states_t).max(1)[0]
            # Q 值裁剪：防止 Q 值无限膨胀
            # 没有 target 网络时，max Q 会通过 bootstrapping 不断增长
            # 裁剪打破这个正反馈循环，让 Q 值保持在合理范围
            if q_clip is not None:
                next_q_max = next_q_max.clamp(-q_clip, q_clip)
        td_target = rewards_t + GAMMA * next_q_max * (1.0 - dones_t)

    # 损失函数
    if use_mse:
        loss = nn.functional.mse_loss(q_values, td_target)
    else:
        loss = nn.functional.smooth_l1_loss(q_values, td_target)

    optimizer.zero_grad()
    loss.backward()
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), grad_clip)
    optimizer.step()

    return loss.item()


# ============================================================
# 训练函数 — 可选是否使用 target 网络
# ============================================================
def run_training(use_target_net: bool, label: str):
    """
    训练 DQN 并返回每个 episode 的回报列表和损失列表。

    参数：
        use_target_net: True 表示使用 target 网络（标准 DQN），
                        False 表示不使用（消融实验）
        label: 显示用的标签名
    返回：
        all_rewards: 每个 episode 的总回报列表
        all_losses:  每个 episode 的平均 TD loss 列表
    """
    # --- 选择对应版本的超参数 ---
    if use_target_net:
        lr = WITH_TARGET_LR
        grad_clip = WITH_TARGET_GRAD_CLIP
        use_mse = WITH_TARGET_USE_MSE
        epsilon_decay = WITH_TARGET_EPSILON_DECAY
        max_episodes = WITH_TARGET_MAX_EPISODES
        q_clip = None  # 有 target 网络不需要 Q 值裁剪
    else:
        lr = NO_TARGET_LR
        grad_clip = NO_TARGET_GRAD_CLIP
        use_mse = NO_TARGET_USE_MSE
        epsilon_decay = NO_TARGET_EPSILON_DECAY
        max_episodes = NO_TARGET_MAX_EPISODES
        q_clip = NO_TARGET_Q_CLIP

    # --- 固定种子，确保两组实验起点相同 ---
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    # --- 环境和设备 ---
    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_actions = env.action_space.n
    state_dim = env.observation_space.shape[0]

    # --- 创建网络 ---
    policy_net = QNetwork(state_dim, n_actions).to(device)

    if use_target_net:
        target_net = QNetwork(state_dim, n_actions).to(device)
        target_net.load_state_dict(policy_net.state_dict())
        target_net.eval()
    else:
        target_net = None

    optimizer = optim.AdamW(policy_net.parameters(), lr=lr, amsgrad=True)
    replay = ReplayBuffer(REPLAY_SIZE)

    reward_history = collections.deque(maxlen=100)  # 最近 100 episode 的回报
    all_rewards = []       # 记录每个 episode 的回报
    all_losses = []        # 记录每个 episode 的平均 TD loss
    steps_done = 0

    print(f"\n{'='*65}")
    print(f"开始训练：{label}")
    print(f"Target 网络: {'是' if use_target_net else '否'} | 设备: {device}")
    print(f"LR: {lr} | Loss: {'MSE' if use_mse else 'Huber'} | "
          f"梯度裁剪: {grad_clip} | Q值裁剪: {q_clip or '无'}")
    print(f"epsilon 衰减: {epsilon_decay} 步 | 训练 episode 数: {max_episodes}")
    print(f"{'='*65}")

    for episode in range(1, max_episodes + 1):
        if episode == 1:
            state, _ = env.reset(seed=SEED)
        else:
            state, _ = env.reset()

        episode_reward = 0.0
        episode_losses = []

        while True:
            epsilon = get_epsilon(steps_done, epsilon_decay)
            steps_done += 1

            action = select_action(state, policy_net, epsilon, n_actions, device)
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            replay.push(state, action, reward, next_state, float(done))
            state = next_state
            episode_reward += reward

            # 缓冲区数据够了就训练
            if len(replay) >= MIN_REPLAY:
                loss_val = train_step(policy_net, target_net, optimizer,
                                      replay, device,
                                      use_mse=use_mse, grad_clip=grad_clip,
                                      q_clip=q_clip)
                episode_losses.append(loss_val)
                # 如果使用 target 网络，每步软更新
                if use_target_net:
                    soft_update(target_net, policy_net, TAU)

            if done:
                break

        reward_history.append(episode_reward)
        avg_reward = np.mean(reward_history)
        all_rewards.append(episode_reward)
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0
        all_losses.append(avg_loss)

        # 每 50 个 episode 打印一次进度
        if episode % 50 == 0:
            print(f"  [{label}] Episode {episode:4d}/{max_episodes} | "
                  f"回报: {episode_reward:6.1f} | "
                  f"平均回报(100ep): {avg_reward:6.1f} | "
                  f"loss: {avg_loss:.4f} | "
                  f"epsilon: {epsilon:.3f}")

    env.close()
    final_avg = np.mean(list(reward_history))
    print(f"  [{label}] 训练完成！最终 100ep 平均回报: {final_avg:.1f}")
    return all_rewards, all_losses


# ============================================================
# 画对比图 — 2x2 布局
# ============================================================
def plot_comparison(rewards_with, losses_with, rewards_without, losses_without):
    """
    画 2x2 对比图：
      左上：回报对比 — 两条移动平均在同一图上
      右上：每局回报明细 — 两个版本各自的原始回报 + 移动平均
      左下：有 Target Network 的 TD Loss
      右下：无 Target Network 的 TD Loss
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    # macOS 中文字体支持
    plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "STHeiti",
                                        "SimHei", "Arial Unicode MS"]
    plt.rcParams["axes.unicode_minus"] = False

    episodes_with = range(1, len(rewards_with) + 1)
    episodes_without = range(1, len(rewards_without) + 1)

    # 计算移动平均
    def moving_avg(data, window=100):
        result = []
        for i in range(len(data)):
            start = max(0, i - window + 1)
            result.append(np.mean(data[start:i + 1]))
        return result

    avg_with = moving_avg(rewards_with)
    avg_without = moving_avg(rewards_without)
    loss_smooth_with = moving_avg(losses_with, window=50)
    loss_smooth_without = moving_avg(losses_without, window=50)

    # --- 创建 2x2 图 ---
    fig, axes = plt.subplots(2, 2, figsize=(16, 11))
    fig.suptitle("DQN 消融实验：Target Network 的影响",
                 fontsize=16, fontweight="bold", y=0.98)

    # ==========================================
    # 左上：回报对比 — 两条移动平均叠在一起
    # ==========================================
    ax = axes[0, 0]
    ax.plot(episodes_with, avg_with, color="#2196F3", linewidth=2.5,
            label=f"有 Target Network（{len(rewards_with)} ep）")
    ax.plot(episodes_without, avg_without, color="#F44336", linewidth=2.5,
            label=f"无 Target Network（{len(rewards_without)} ep）")
    ax.axhline(y=475, color="gray", linestyle="--", linewidth=1,
               alpha=0.7, label="解决标准 (475)")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("100-Episode 移动平均回报", fontsize=11)
    ax.set_title("回报对比", fontsize=13, fontweight="bold")
    # 超参数注释
    param_note = (
        f"有Target: LR={WITH_TARGET_LR}, Huber, clip={WITH_TARGET_GRAD_CLIP}\n"
        f"无Target: LR={NO_TARGET_LR}, MSE, clip={NO_TARGET_GRAD_CLIP}, "
        f"Q_clip={NO_TARGET_Q_CLIP}")
    ax.text(0.02, 0.96, param_note, transform=ax.transAxes,
            fontsize=7.5, verticalalignment='top',
            bbox=dict(boxstyle='round,pad=0.3', facecolor='lightyellow', alpha=0.8))
    ax.legend(fontsize=10, loc="lower right")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(len(rewards_with), len(rewards_without)))
    ax.set_ylim(0, 520)

    # ==========================================
    # 右上：每局回报明细
    # ==========================================
    ax = axes[0, 1]
    ax.plot(episodes_with, rewards_with, alpha=0.15, color="#2196F3",
            linewidth=0.5)
    ax.plot(episodes_with, avg_with, color="#1565C0", linewidth=1.8,
            label="有 Target — 100ep 平均")
    ax.plot(episodes_without, rewards_without, alpha=0.15, color="#F44336",
            linewidth=0.5)
    ax.plot(episodes_without, avg_without, color="#C62828", linewidth=1.8,
            label="无 Target — 100ep 平均")
    ax.axhline(y=475, color="gray", linestyle="--", linewidth=1, alpha=0.5)
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("回报", fontsize=11)
    ax.set_title("每局回报明细", fontsize=13, fontweight="bold")
    ax.legend(fontsize=9, loc="upper left")
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, max(len(rewards_with), len(rewards_without)))
    ax.set_ylim(0, 520)

    # ==========================================
    # 左下：有 Target Network 的 TD Loss
    # ==========================================
    ax = axes[1, 0]
    ax.plot(episodes_with, losses_with, alpha=0.3, color="#2196F3",
            linewidth=0.5, label="原始 loss")
    ax.plot(episodes_with, loss_smooth_with, color="#1565C0", linewidth=2,
            label="50ep 移动平均")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("平均 TD Loss", fontsize=11)
    ax.set_title("有 Target Network — TD Loss", fontsize=13, fontweight="bold",
                 color="#1565C0")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(losses_with))

    # ==========================================
    # 右下：无 Target Network 的 TD Loss
    # ==========================================
    ax = axes[1, 1]
    ax.plot(episodes_without, losses_without, alpha=0.3, color="#F44336",
            linewidth=0.5, label="原始 loss")
    ax.plot(episodes_without, loss_smooth_without, color="#C62828", linewidth=2,
            label="50ep 移动平均")
    ax.set_xlabel("Episode", fontsize=11)
    ax.set_ylabel("平均 TD Loss", fontsize=11)
    ax.set_title("无 Target Network — TD Loss", fontsize=13, fontweight="bold",
                 color="#C62828")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)
    ax.set_xlim(1, len(losses_without))

    # 各自独立设置 y 轴范围 — 两个版本的 loss 量级差异巨大
    # （有 target ~0.2，无 target ~几十甚至上百），共享 y 轴会导致小值不可见
    y_max_with = max(losses_with) * 1.1 if max(losses_with) > 0 else 1.0
    y_max_without = max(losses_without) * 1.1 if max(losses_without) > 0 else 1.0
    axes[1, 0].set_ylim(0, y_max_with)
    axes[1, 1].set_ylim(0, y_max_without)

    plt.tight_layout(rect=[0, 0, 1, 0.96])

    # --- 保存 ---
    import os
    save_path = os.path.join(os.path.dirname(os.path.abspath(__file__)),
                             "dqn_ablation_curves.png")
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n对比图已保存到 {save_path}")


# ============================================================
# 主程序
# ============================================================
if __name__ == "__main__":
    print("=" * 65)
    print("DQN 消融实验：有 vs 无 Target Network")
    print("环境: CartPole-v1 | seed=0")
    print("-" * 65)
    print("超参数差异：")
    print(f"  有 Target: LR={WITH_TARGET_LR}, Loss=Huber, "
          f"grad_clip={WITH_TARGET_GRAD_CLIP}, "
          f"episodes={WITH_TARGET_MAX_EPISODES}")
    print(f"  无 Target: LR={NO_TARGET_LR}, Loss=MSE, "
          f"grad_clip={NO_TARGET_GRAD_CLIP}, "
          f"Q_clip={NO_TARGET_Q_CLIP}, "
          f"episodes={NO_TARGET_MAX_EPISODES}")
    print(f"  (其余参数相同: batch={BATCH_SIZE}, gamma={GAMMA}, "
          f"replay={REPLAY_SIZE})")
    print("=" * 65)

    # --- 第一组：有 target 网络（标准 DQN）---
    rewards_with, losses_with = run_training(
        use_target_net=True,
        label="有Target网络"
    )

    # --- 第二组：无 target 网络（消融版本）---
    rewards_without, losses_without = run_training(
        use_target_net=False,
        label="无Target网络"
    )

    # --- 画对比图 ---
    print("\n正在生成对比图...")
    plot_comparison(rewards_with, losses_with, rewards_without, losses_without)

    # --- 打印总结表格 ---
    def compute_stats(rewards, losses):
        avg_last100 = np.mean(rewards[-100:])
        avg_all = np.mean(rewards)
        max_r = np.max(rewards)
        avg_loss_last100 = np.mean(losses[-100:])
        # 首次达到 100ep 平均 >= 200 的 episode
        first_200 = None
        for i in range(99, len(rewards)):
            if np.mean(rewards[max(0, i-99):i+1]) >= 200:
                first_200 = i + 1
                break
        # 首次达到 475 的 episode
        first_475 = None
        for i in range(99, len(rewards)):
            if np.mean(rewards[max(0, i-99):i+1]) >= 475:
                first_475 = i + 1
                break
        return avg_last100, avg_all, max_r, first_200, first_475, avg_loss_last100

    stats_with = compute_stats(rewards_with, losses_with)
    stats_without = compute_stats(rewards_without, losses_without)

    print("\n" + "=" * 70)
    print("实验结果总结")
    print("=" * 70)
    print(f"{'指标':<30} {'有Target网络':>15} {'无Target网络':>15}")
    print("-" * 70)
    print(f"{'训练 episode 数':<27} "
          f"{len(rewards_with):>15d} {len(rewards_without):>15d}")
    print(f"{'最后100ep平均回报':<25} "
          f"{stats_with[0]:>15.1f} {stats_without[0]:>15.1f}")
    print(f"{'全程平均回报':<27} "
          f"{stats_with[1]:>15.1f} {stats_without[1]:>15.1f}")
    print(f"{'单局最高回报':<27} "
          f"{stats_with[2]:>15.1f} {stats_without[2]:>15.1f}")
    ep200_w = str(stats_with[3]) if stats_with[3] else "未达到"
    ep200_wo = str(stats_without[3]) if stats_without[3] else "未达到"
    print(f"{'首次100ep平均>=200 (ep)':<23} "
          f"{ep200_w:>15s} {ep200_wo:>15s}")
    ep475_w = str(stats_with[4]) if stats_with[4] else "未达到"
    ep475_wo = str(stats_without[4]) if stats_without[4] else "未达到"
    print(f"{'首次100ep平均>=475 (ep)':<23} "
          f"{ep475_w:>15s} {ep475_wo:>15s}")
    print(f"{'平均 TD Loss (最后100ep)':<23} "
          f"{stats_with[5]:>15.4f} {stats_without[5]:>15.4f}")
    print("=" * 70)
