from typing import Tuple
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch


class DynamicServiceEncoder(nn.Module):
    """动态服务时序特征编码器"""

    def __init__(self, feat_dim=25, hidden_dim=64):
        super().__init__()
        # 轻量化时序卷积
        self.conv = nn.Sequential(
            nn.Conv1d(feat_dim, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv1d(32, hidden_dim, kernel_size=3, padding=1),
        )
        # 时序注意力
        self.attn = nn.MultiheadAttention(hidden_dim, num_heads=2)

    def forward(self, x):
        """
        输入: (B, num_services, time_steps, feat_dim)
        输出: (B, hidden_dim)
        """
        B, S, T, F = x.shape
        # 并行处理所有服务
        x = x.view(B * S, T, F)
        # 时序卷积 [B*S, T, F] -> [B*S, F', T]
        x = self.conv(x.permute(0, 2, 1))  # 转换为通道优先
        # 时序注意力 [B*S, F', T] -> [B*S, F']
        x, _ = self.attn(x, x, x)
        x = x.mean(dim=1)
        # 服务聚合 [B*S, F'] -> [B, F']
        return x.view(B, S, -1).mean(dim=1)


class Q_Net(nn.Module):
    """适应动态服务数量的Q网络"""

    def __init__(self, num_actions=5, time_steps=10):
        super().__init__()
        # 服务编码器
        self.encoder = DynamicServiceEncoder()
        # 动作价值头
        self.q_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(),
                                    nn.Linear(32, num_actions))

    def forward(self, state):
        """
        输入: (B, time_steps, num_services, feat_dim)
        输出: (B, num_actions)
        """
        # 动态编码服务
        state_feat = self.encoder(state)
        # 生成Q值
        return self.q_head(state_feat)


class Duel_Q_Net(nn.Module):
    """支持动态服务输入的Dueling DQN"""

    def __init__(self, opt):
        super(Duel_Q_Net, self).__init__()
        self.encoder = DynamicServiceEncoder(feat_dim=opt.feat_dim)
        self.fc = nn.Linear(64, opt.fc_width)  # 编码器输出64维

        # Dueling 结构
        self.A = nn.Linear(opt.fc_width, opt.action_dim)  # Advantage流
        self.V = nn.Linear(opt.fc_width, 1)  # Value流

    def forward(self, obs):
        """
        输入: obs形状 (B, T, S, F)
        输出: Q值 (B, action_dim)
        """
        s = self.encoder(obs)  # [B, 64]
        s = F.relu(self.fc(s))  # [B, fc_width]

        Adv = self.A(s)  # [B, action_dim]
        Val = self.V(s)  # [B, 1]
        Q = Val + (Adv - Adv.mean(dim=1, keepdim=True))
        return Q


class Double_Q_Net(nn.Module):
    """双Q网络结构"""

    def __init__(self, opt):
        super(Double_Q_Net, self).__init__()
        self.Q1 = Duel_Q_Net(opt)
        self.Q2 = Duel_Q_Net(opt)

    def forward(self, s):
        return self.Q1(s), self.Q2(s)


class Double_Duel_Q_Net(nn.Module):

    def __init__(self, opt):
        super(Double_Duel_Q_Net, self).__init__()

        self.Q1 = Duel_Q_Net(opt)
        self.Q2 = Duel_Q_Net(opt)

    def forward(self, s):
        q1 = self.Q1(s)
        q2 = self.Q2(s)
        return q1, q2


class Policy_Net(nn.Module):
    """策略网络 (基于Dueling结构)"""

    def __init__(self, opt):
        super(Policy_Net, self).__init__()
        self.encoder = DynamicServiceEncoder(feat_dim=opt.feat_dim)
        self.fc = nn.Linear(64, opt.fc_width)
        self.head = nn.Linear(opt.fc_width, opt.action_dim)


class ReplayBuffer:
    """经验回放池"""

    def __init__(
        self,
        buffer_size: int = 100000,
        num_services: int = 28,
        state_window: int = 10,
        state_features: int = 25,
        num_actions: int = 5,
    ):
        """
        参数:
            buffer_size: 缓冲池最大容量
            num_services: 固定服务数量 (必须为28)
            state_window: 状态时间窗口 (固定为10步)
            state_features: 每个服务的特征维度 (固定为25)
            num_actions: 全局有限动作的数量 (例如5种策略)
        """
        assert num_services == 28, "仅支持28个服务的配置"

        # 预分配内存 (优化内存访问模式)
        self.states = np.zeros(
            (buffer_size, state_window, num_services, state_features),
            dtype=np.float32)  # 状态序列
        self.actions = np.zeros(buffer_size,
                                dtype=np.int64)  # 动作索引 (0~num_actions-1)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)  # 即时奖励
        self.next_states = np.zeros_like(self.states)  # 下一状态
        self.dones = np.zeros(buffer_size, dtype=np.bool_)  # 终止标志

        # 指针管理
        self.buffer_size = buffer_size
        self.current_idx = 0  # 当前写入位置
        self.current_size = 0  # 当前有效数据量

    def add_experience(
        self,
        state: np.ndarray,  # (10,28,25)
        action: int,  # 动作索引 (标量)
        reward: float,
        next_state: np.ndarray,  # (10,28,25)
        done: bool,
    ) -> None:
        """添加单条经验"""
        # 输入数据验证
        self._validate_shape(state, (10, 28, 25), "State")
        self._validate_scalar(action, "Action")
        self._validate_shape(next_state, (10, 28, 25), "Next State")

        # 写入存储
        idx = self.current_idx
        self.states[idx] = state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_states[idx] = next_state
        self.dones[idx] = done

        # 更新环形缓冲指针
        self.current_idx = (idx + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample_batch(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor,
               torch.Tensor]:
        """采样训练批次"""
        indices = np.random.choice(self.current_size,
                                   batch_size,
                                   replace=False)

        return (
            torch.FloatTensor(self.states[indices]),  # (B,10,28,25)
            torch.LongTensor(self.actions[indices]),  # (B,)
            torch.FloatTensor(self.rewards[indices]).unsqueeze(1),  # (B,1)
            torch.FloatTensor(self.next_states[indices]),  # (B,10,28,25)
            torch.FloatTensor(self.dones[indices].astype(
                np.float32)).unsqueeze(1),  # (B,1)
        )

    def _validate_shape(self, data: np.ndarray, expected_shape: tuple,
                        name: str):
        """验证状态数据维度"""
        if data.shape != expected_shape:
            raise ValueError(
                f"Invalid {name} shape. Expected: {expected_shape}, Got: {data.shape}"
            )

    def _validate_scalar(self, value: int, name: str):
        """验证动作是否为标量"""
        if not (0 <= value < self.num_actions):
            raise ValueError(
                f"{name} index out of range. Max allowed: {self.num_actions - 1}"
            )

    @property
    def num_actions(self) -> int:
        """返回动作空间维度"""
        return self.actions.max() + 1 if self.current_size > 0 else 0

    @property
    def is_full(self) -> bool:
        """缓冲池是否已满"""
        return self.current_size >= self.buffer_size


def evaluate_policy(env, agent, seed, turns=3):
    total_scores = 0
    for j in range(turns):
        s, info = env.reset(seed=seed)
        done = False
        while not done:
            # Take deterministic actions at test time
            a = agent.select_action(s, deterministic=True, evaluate=False)
            s_next, r, dw, tr, info = env.step(a)
            done = dw or tr

            total_scores += r
            s = s_next

    return int(total_scores / turns)


# You can just ignore 'str2bool'. Is not related to the RL.
def str2bool(v):
    """transfer str to bool for argparse"""
    if isinstance(v, bool):
        return v
    if v.lower() in ("yes", "True", "true", "TRUE", "t", "y", "1"):
        return True
    elif v.lower() in ("no", "False", "false", "FALSE", "f", "n", "0"):
        return False
    else:
        raise argparse.ArgumentTypeError("Boolean value expected.")


class LinearSchedule(object):

    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """在schedule_timesteps时间步内，将initial_p线性插值到final_p。
        超过此时限后，始终返回final_p。

        参数
        ----------
        schedule_timesteps: int
            用于将initial_p线性衰减至final_p的总时间步数
        initial_p: float
            初始输出值
        final_p: float
            最终输出值
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
