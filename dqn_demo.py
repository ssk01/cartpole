"""
DQN 最简实现 — 对应 Mnih et al. 2013 (arXiv) / 2015 (Nature) 论文
=================================================================
环境：CartPole-v1（平衡杆，状态是 4 维向量，动作是左/右两个离散动作）
虽然比 Atari 简单，但核心算法完全一致：Q 网络 + 经验回放 + 目标网络。

参考代码：
  - DeepMind 原版 Lua/Torch 实现 (NeuralQLearner.lua, TransitionTable.lua)
  - DeepMind dqn_zoo JAX 实现 (dqn/agent.py, replay.py)

依赖：pip install torch gymnasium
"""

import random
import collections
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import gymnasium as gym

# ============================================================
# 超参数 — 每个值都对应论文中的某个设计选择
# ============================================================
BATCH_SIZE = 128           # 每次从回放缓冲区采样的 transition 数量（PyTorch 教程用 128）
GAMMA = 0.99               # 折扣因子 γ：未来奖励的衰减系数（论文 0.99）
LR = 1e-4                  # 学习率（较小的学习率让训练更稳定）
REPLAY_SIZE = 10000        # 经验回放缓冲区容量（PyTorch 教程用 10000）
MIN_REPLAY = 1000          # 开始训练前最少收集多少 transition（确保采样多样性）
EPSILON_START = 0.9        # ε-greedy 起始探索率
EPSILON_END = 0.05         # ε-greedy 最终探索率
EPSILON_DECAY = 3000       # ε 指数衰减速度（基于步数）
TAU = 0.005                # target 网络软更新系数：每步微量同步
MAX_EPISODES = 600         # 最大训练 episode 数
SOLVED_REWARD = 475        # CartPole-v1 的"解决"标准：最近 100 episode 平均 >= 475


# ============================================================
# 1. Q 网络 — 输入状态，输出每个动作的 Q 值
# ============================================================
# 论文中用 CNN 处理 Atari 像素，这里 CartPole 的状态是 4 维向量，用 MLP 即可。
# 网络结构：state(4) -> 128 -> 128 -> Q(2)
# 关键思想：Q(s,a) ≈ 该状态下执行动作 a 能获得的期望累计回报
class QNetwork(nn.Module):
    def __init__(self, state_dim: int, action_dim: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(state_dim, 128),   # 全连接层：4 -> 128
            nn.ReLU(),                    # 非线性激活（论文用 ReLU）
            nn.Linear(128, 128),          # 全连接层：128 -> 128
            nn.ReLU(),
            nn.Linear(128, action_dim),   # 输出层：128 -> 2（左/右各一个 Q 值）
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)  # 输出 shape: (batch, action_dim)


# ============================================================
# 2. 经验回放缓冲区 (Experience Replay Buffer)
# ============================================================
# 论文的关键创新之一（2013 论文的核心贡献）：
# - 打破数据的时间相关性：连续帧高度相关，直接训练会导致不稳定
# - 提高数据效率：同一条经验可以被多次采样学习
# 实现：一个固定大小的环形缓冲区，存满后覆盖最旧的数据
class ReplayBuffer:
    def __init__(self, capacity: int):
        # deque 自带 maxlen，满了之后自动丢弃最旧的元素（等价于环形缓冲区）
        self.buffer = collections.deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """存入一条 transition (s, a, r, s', done)"""
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """均匀随机采样一个 mini-batch（论文的 uniform replay）"""
        batch = random.sample(self.buffer, batch_size)
        # 把 list of tuples 转成 tuple of arrays，方便后续转 tensor
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
# 3. 选择动作 — ε-greedy 策略
# ============================================================
# 论文的探索策略：以概率 ε 随机选动作（探索），以概率 1-ε 选 Q 值最大的动作（利用）
# ε 随训练推进逐渐衰减：早期多探索，后期多利用
def select_action(state: np.ndarray, policy_net: QNetwork, epsilon: float,
                  n_actions: int, device: torch.device) -> int:
    if random.random() < epsilon:
        # 探索：随机选一个动作
        return random.randrange(n_actions)
    else:
        # 利用：选 Q 值最大的动作（不需要梯度，用 torch.no_grad 节省内存）
        with torch.no_grad():
            state_t = torch.tensor(state, dtype=torch.float32, device=device).unsqueeze(0)
            q_values = policy_net(state_t)          # shape: (1, n_actions)
            return q_values.argmax(dim=1).item()    # 返回 Q 最大的动作索引


# ============================================================
# 4. 计算 epsilon 值 — 基于步数的指数衰减（PyTorch 教程方式）
# ============================================================
def get_epsilon(steps_done: int) -> float:
    # 从 EPSILON_START 指数衰减到 EPSILON_END，基于总步数而非 episode 数
    # 论文用线性衰减（从 1.0 到 0.1，100 万帧），这里用指数衰减更适合短训练
    # 基于步数衰减比基于 episode 更稳定，因为 episode 长度会随训练变化
    return EPSILON_END + (EPSILON_START - EPSILON_END) * \
        np.exp(-steps_done / EPSILON_DECAY)


# ============================================================
# 5. 训练一步 — DQN 的核心：TD 目标 + 梯度下降
# ============================================================
# 这就是 DQN 算法最关键的部分，对应论文 Algorithm 1 的内循环
def train_step(policy_net: QNetwork, target_net: QNetwork,
               optimizer: optim.Optimizer, replay: ReplayBuffer,
               device: torch.device):
    # --- 5a. 从回放缓冲区均匀采样一个 mini-batch ---
    states, actions, rewards, next_states, dones = replay.sample(BATCH_SIZE)

    # 转为 PyTorch tensor 并放到对应设备上
    states_t = torch.tensor(states, device=device)           # (B, 4)
    actions_t = torch.tensor(actions, device=device)         # (B,)
    rewards_t = torch.tensor(rewards, device=device)         # (B,)
    next_states_t = torch.tensor(next_states, device=device) # (B, 4)
    dones_t = torch.tensor(dones, device=device)             # (B,)

    # --- 5b. 计算 Q(s, a) — 当前网络对"已选动作"的 Q 值估计 ---
    # policy_net(states_t) 输出所有动作的 Q 值 (B, 2)
    # .gather(1, actions_t.unsqueeze(1)) 取出实际执行的那个动作对应的 Q 值
    q_values = policy_net(states_t).gather(1, actions_t.unsqueeze(1)).squeeze(1)
    # 结果 shape: (B,) — 每条 transition 对应一个 Q(s,a) 值

    # --- 5c. 计算 TD 目标 y = r + γ * max_a' Q_target(s', a') ---
    # 这是 DQN 的关键公式！用 TARGET 网络（不是 policy 网络）来计算下一状态的最大 Q 值
    # 2015 Nature 论文的核心改进：target 网络的参数是 policy 网络的"旧拷贝"
    # 这样 TD 目标更稳定，避免"自己追自己"导致的训练震荡
    with torch.no_grad():  # target 计算不需要梯度
        # max(1) 返回 (values, indices)，取 [0] 即最大 Q 值
        next_q_max = target_net(next_states_t).max(1)[0]
        # 如果 s' 是终止状态，没有未来回报，所以乘以 (1 - done)
        td_target = rewards_t + GAMMA * next_q_max * (1.0 - dones_t)

    # --- 5d. 计算损失并反向传播 ---
    # 论文使用 Huber loss（等价于将 TD error clip 到 [-1,1]），比 MSE 更稳定
    # 当误差较小时等价于 MSE，当误差较大时等价于 L1（不会因极端值爆梯度）
    loss = nn.functional.smooth_l1_loss(q_values, td_target)

    optimizer.zero_grad()  # 清除上一步的梯度
    loss.backward()        # 反向传播，计算梯度 dL/dθ

    # 论文中的梯度裁剪（clip_delta）：防止梯度爆炸
    torch.nn.utils.clip_grad_value_(policy_net.parameters(), 100)
    optimizer.step()       # 用 AdamW 更新参数 θ ← θ - lr * gradient

    return loss.item()     # 返回 loss 数值，用于记录


# ============================================================
# 6. 软更新 target 网络
# ============================================================
# 论文原文用硬更新（每 C 步完全复制）。这里用软更新——效果等价但更平滑：
# θ_target = τ * θ_online + (1 - τ) * θ_target
# 每步只融合一点点新参数（τ=0.005），等价于约每 200 步的硬更新
# 优势：避免 target Q 值的突变，训练更稳定
def soft_update(target_net: QNetwork, policy_net: QNetwork, tau: float):
    for tp, pp in zip(target_net.parameters(), policy_net.parameters()):
        tp.data.copy_(tau * pp.data + (1.0 - tau) * tp.data)


# ============================================================
# 7. 主训练循环
# ============================================================
def train():
    # --- 环境和设备 ---
    env = gym.make("CartPole-v1")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    n_actions = env.action_space.n         # CartPole: 2（左/右）
    state_dim = env.observation_space.shape[0]  # CartPole: 4

    # --- 创建两个结构相同的网络 ---
    policy_net = QNetwork(state_dim, n_actions).to(device)   # 在线网络：用于选动作、计算梯度
    target_net = QNetwork(state_dim, n_actions).to(device)   # 目标网络：用于计算 TD 目标
    target_net.load_state_dict(policy_net.state_dict())      # 初始参数完全相同
    target_net.eval()  # target 网络不训练，只做推理

    # AdamW = Adam + weight decay（L2 正则化），amsgrad 提高训练稳定性
    optimizer = optim.AdamW(policy_net.parameters(), lr=LR, amsgrad=True)
    replay = ReplayBuffer(REPLAY_SIZE)

    # --- 记录最近 100 episode 的回报，用于判断是否"解决" ---
    reward_history = collections.deque(maxlen=100)

    # --- 记录训练曲线数据 ---
    all_rewards = []       # 每个 episode 的总回报
    all_avg_rewards = []   # 每个 episode 的 100ep 滑动平均
    all_losses = []        # 每个 episode 的平均 loss
    all_epsilons = []      # 每个 episode 的 epsilon 值

    # --- 记录特定状态的 Q 值变化（用于可视化"学到了什么"）---
    # 杆直立 + 小车在中心 + 全部静止 = 最理想状态
    probe_ideal = torch.tensor([0.0, 0.0, 0.0, 0.0], dtype=torch.float32, device=device)
    # 杆往右歪 5 度 = 有点危险
    probe_tilt_right = torch.tensor([0.0, 0.0, 0.087, 0.0], dtype=torch.float32, device=device)
    # 杆往左歪 5 度
    probe_tilt_left = torch.tensor([0.0, 0.0, -0.087, 0.0], dtype=torch.float32, device=device)
    all_q_ideal = []       # 每个 episode 记录理想状态的 max Q
    all_q_tilt_right = []  # 杆右歪时的 [Q左, Q右]
    all_q_tilt_left = []   # 杆左歪时的 [Q左, Q右]

    steps_done = 0  # 全局步数计数器，用于 epsilon 衰减

    print("开始训练 DQN ...")
    print(f"设备: {device} | 状态维度: {state_dim} | 动作数: {n_actions}")
    print("-" * 65)

    for episode in range(1, MAX_EPISODES + 1):
        state, _ = env.reset()      # 重置环境，获取初始状态
        episode_reward = 0.0
        episode_losses = []          # 本 episode 内每步的 loss

        # --- 一个 episode 的循环 ---
        while True:
            # 基于全局步数计算 epsilon（而非 episode，更稳定）
            epsilon = get_epsilon(steps_done)
            steps_done += 1

            # (a) 用 ε-greedy 选动作
            action = select_action(state, policy_net, epsilon, n_actions, device)

            # (b) 执行动作，观察下一状态和奖励
            next_state, reward, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # (c) 存入回放缓冲区
            replay.push(state, action, reward, next_state, float(done))

            state = next_state
            episode_reward += reward

            # (d) 如果缓冲区里数据够了，就训练一步
            #     论文：每个时间步都训练一次（update_freq=1）
            if len(replay) >= MIN_REPLAY:
                loss_val = train_step(policy_net, target_net, optimizer, replay, device)
                episode_losses.append(loss_val)
                # (e) 每步软更新 target 网络
                soft_update(target_net, policy_net, TAU)

            if done:
                break

        # --- 记录和打印 ---
        reward_history.append(episode_reward)
        avg_reward = np.mean(reward_history)
        avg_loss = np.mean(episode_losses) if episode_losses else 0.0

        all_rewards.append(episode_reward)
        all_avg_rewards.append(avg_reward)
        all_losses.append(avg_loss)

        # 记录探针状态的 Q 值
        with torch.no_grad():
            q_ideal = policy_net(probe_ideal.unsqueeze(0)).squeeze()
            q_tr = policy_net(probe_tilt_right.unsqueeze(0)).squeeze()
            q_tl = policy_net(probe_tilt_left.unsqueeze(0)).squeeze()
        all_q_ideal.append(q_ideal.max().item())
        all_q_tilt_right.append([q_tl[0].item(), q_tl[1].item()])  # 杆左歪时：Q左, Q右
        all_q_tilt_left.append([q_tr[0].item(), q_tr[1].item()])    # 杆右歪时：Q左, Q右
        all_epsilons.append(epsilon)

        if episode % 10 == 0 or avg_reward >= SOLVED_REWARD:
            print(f"Episode {episode:4d} | "
                  f"回报: {episode_reward:6.1f} | "
                  f"平均回报(100ep): {avg_reward:6.1f} | "
                  f"ε: {epsilon:.3f} | "
                  f"缓冲区: {len(replay)}")

        # --- 判断是否"解决"了 CartPole ---
        if len(reward_history) >= 100 and avg_reward >= SOLVED_REWARD:
            print("=" * 65)
            print(f"CartPole 已解决！平均回报 {avg_reward:.1f} >= {SOLVED_REWARD}")
            print(f"共训练 {episode} 个 episode")
            break

    # --- 保存模型权重 ---
    import os
    save_path = os.path.join(os.path.dirname(__file__), "dqn_cartpole.pth")
    torch.save({
        "policy_net": policy_net.state_dict(),
        "target_net": target_net.state_dict(),
        "episode": episode,
        "avg_reward": avg_reward,
    }, save_path)
    print(f"模型已保存到 {save_path}")

    env.close()
    print("训练完成。")

    # --- 画训练曲线 ---
    plot_training_curves(all_rewards, all_avg_rewards, all_losses, all_epsilons,
                         all_q_ideal, all_q_tilt_right, all_q_tilt_left)


def plot_training_curves(rewards, avg_rewards, losses, epsilons,
                         q_ideal=None, q_tilt_right=None, q_tilt_left=None):
    """画 6 张子图：回报、学习进度、Loss、Epsilon、Q值变化、动作偏好"""
    try:
        import matplotlib
        matplotlib.use("Agg")  # 无需 GUI，直接保存文件
        import matplotlib.pyplot as plt
        # macOS 中文字体支持
        plt.rcParams["font.sans-serif"] = ["PingFang SC", "Heiti TC", "STHeiti", "SimHei", "Arial Unicode MS"]
        plt.rcParams["axes.unicode_minus"] = False
    except ImportError:
        print("matplotlib 未安装，跳过画图。安装方式：pip install matplotlib")
        return

    fig, axes = plt.subplots(3, 2, figsize=(14, 13))
    fig.suptitle("DQN CartPole 训练曲线", fontsize=15, fontweight="bold")
    episodes = range(1, len(rewards) + 1)

    # (1) 每局回报
    ax = axes[0, 0]
    ax.plot(episodes, rewards, alpha=0.3, color="steelblue", linewidth=0.8)
    ax.plot(episodes, avg_rewards, color="orange", linewidth=2, label="100ep 滑动平均")
    ax.axhline(y=475, color="red", linestyle="--", linewidth=1, label="解决标准 (475)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("回报 (Reward)")
    ax.set_title("每局回报 & 滑动平均")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (2) 滑动平均回报
    ax = axes[0, 1]
    ax.plot(episodes, avg_rewards, color="orange", linewidth=2)
    ax.axhline(y=475, color="red", linestyle="--", linewidth=1, label="解决标准 (475)")
    ax.set_xlabel("Episode")
    ax.set_ylabel("100ep 滑动平均回报")
    ax.set_title("学习进度")
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # (3) Loss 曲线
    ax = axes[1, 0]
    ax.plot(episodes, losses, color="crimson", alpha=0.6, linewidth=0.8)
    if len(losses) > 20:
        window = min(50, len(losses) // 5)
        smoothed = np.convolve(losses, np.ones(window) / window, mode="valid")
        ax.plot(range(window, len(losses) + 1), smoothed, color="darkred", linewidth=2, label=f"{window}ep 滑动平均")
        ax.legend(fontsize=9)
    ax.set_xlabel("Episode")
    ax.set_ylabel("平均 Loss (Huber)")
    ax.set_title("TD Loss 曲线")
    ax.grid(True, alpha=0.3)

    # (4) Epsilon 衰减
    ax = axes[1, 1]
    ax.plot(episodes, epsilons, color="green", linewidth=2)
    ax.set_xlabel("Episode")
    ax.set_ylabel("ε")
    ax.set_title("探索率 (ε-greedy) 衰减")
    ax.grid(True, alpha=0.3)

    # (5) Q 值变化：杆直立时的 max Q（网络觉得"最好状态"值多少分）
    if q_ideal:
        ax = axes[2, 0]
        ax.plot(episodes, q_ideal, color="purple", linewidth=1.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("max Q 值")
        ax.set_title("杆直立 + 小车居中时的 Q 值\n（网络觉得\"最好状态\"还能活多久）")
        ax.grid(True, alpha=0.3)
        ax.annotate("Q 值从 0 逐渐涨上去\n= 网络在学习", xy=(len(q_ideal)//3, q_ideal[len(q_ideal)//3]),
                    fontsize=9, color="purple")

    # (6) 动作偏好：杆歪时 Q(左) vs Q(右) 的差异
    if q_tilt_right and q_tilt_left:
        ax = axes[2, 1]
        q_tr = np.array(q_tilt_right)  # 杆左歪时的 [Q左, Q右]
        q_tl = np.array(q_tilt_left)   # 杆右歪时的 [Q左, Q右]
        # 杆左歪时应该 Q(左推) > Q(右推)，差值 > 0 说明学对了
        pref_left = q_tr[:, 0] - q_tr[:, 1]   # 杆左歪：Q左 - Q右
        pref_right = q_tl[:, 1] - q_tl[:, 0]  # 杆右歪：Q右 - Q左
        ax.plot(episodes, pref_left, color="blue", linewidth=1.5, label="杆左歪 → 偏好左推", alpha=0.8)
        ax.plot(episodes, pref_right, color="red", linewidth=1.5, label="杆右歪 → 偏好右推", alpha=0.8)
        ax.axhline(0, color="gray", linestyle="--", alpha=0.5)
        ax.set_xlabel("Episode")
        ax.set_ylabel("Q(正确方向) - Q(错误方向)")
        ax.set_title("动作偏好：杆歪 5° 时选对方向的 Q 值优势\n（> 0 说明学会了\"往哪边倒就往哪边推\"）")
        ax.legend(fontsize=8)
        ax.grid(True, alpha=0.3)

    plt.tight_layout()
    import os
    fig_path = os.path.join(os.path.dirname(__file__), "dqn_training_curves.png")
    plt.savefig(fig_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"训练曲线已保存到 {fig_path}")


# ============================================================
if __name__ == "__main__":
    # 固定随机种子以获得可复现结果（可修改 seed 值观察不同运行的差异）
    SEED = 0
    random.seed(SEED)
    np.random.seed(SEED)
    torch.manual_seed(SEED)

    train()
