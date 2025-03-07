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

    def __init__(self, num_actions=5, time_steps=30):
        super().__init__()
        # 服务编码器
        self.encoder = DynamicServiceEncoder()
        # 动作价值头
        self.q_head = nn.Sequential(nn.Linear(64, 32), nn.ReLU(), nn.Linear(32, num_actions))

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
    """支持双模态状态的经验回放池"""

    def __init__(
            self,
            buffer_size: int = 100000,
            service_shape: Tuple[int, int, int] = (30, 28, 26),  # (时间步, 服务数, 特征)
            latency_shape: Tuple[int, int] = (30, 6),  # (时间步, 延迟指标)
            num_actions: int = 8,
    ):
        """
        参数:
            buffer_size: 缓冲池容量
            service_shape: 服务状态形状 (T,S,F)
            latency_shape: 延迟状态形状 (T,L)
            num_actions: 动作空间大小
        """
        # 服务状态相关存储 (性能指标)
        self.service_states = np.zeros((buffer_size, *service_shape), dtype=np.float32)  # (buffer_size, 30,28,26)
        self.next_service_states = np.zeros_like(self.service_states)

        # 延迟状态相关存储
        self.latency_states = np.zeros((buffer_size, *latency_shape), dtype=np.float32)  # (buffer_size, 30,6)
        self.next_latency_states = np.zeros_like(self.latency_states)

        # 通用存储
        self.actions = np.zeros(buffer_size, dtype=np.int64)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)
        self.dones = np.zeros(buffer_size, dtype=np.bool_)

        # 指针管理
        self.buffer_size = buffer_size
        self.current_idx = 0
        self.current_size = 0

    def add_experience(
        self,
        service_state: np.ndarray,  # (30,28,26)
        latency_state: np.ndarray,  # (30,6)
        action: int,
        reward: float,
        next_service_state: np.ndarray,
        next_latency_state: np.ndarray,
        done: bool,
    ) -> None:
        """添加双模态经验"""
        # 输入验证
        self._validate_shape(service_state, self.service_states.shape[1:], "Service State")
        self._validate_shape(latency_state, self.latency_states.shape[1:], "Latency State")
        self._validate_scalar(action, "Action")

        # 写入存储
        idx = self.current_idx
        self.service_states[idx] = service_state
        self.latency_states[idx] = latency_state
        self.actions[idx] = action
        self.rewards[idx] = reward
        self.next_service_states[idx] = next_service_state
        self.next_latency_states[idx] = next_latency_state
        self.dones[idx] = done

        # 更新指针
        self.current_idx = (idx + 1) % self.buffer_size
        self.current_size = min(self.current_size + 1, self.buffer_size)

    def sample_batch(
        self, batch_size: int
    ) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样批次数据"""
        indices = np.random.choice(self.current_size, batch_size, replace=False)

        return (
            torch.FloatTensor(self.service_states[indices]),  # (B,30,28,26)
            torch.FloatTensor(self.latency_states[indices]),  # (B,30,6)
            torch.LongTensor(self.actions[indices]),  # (B,)
            torch.FloatTensor(self.rewards[indices]).unsqueeze(1),  # (B,1)
            torch.FloatTensor(self.next_service_states[indices]),  # (B,30,28,26)
            torch.FloatTensor(self.next_latency_states[indices]),  # (B,30,6)
            torch.FloatTensor(self.dones[indices].astype(np.float32)).unsqueeze(1),  # (B,1)
        )

    def _validate_shape(self, data: np.ndarray, expected_shape: tuple, name: str):
        """形状验证"""
        if data.shape != expected_shape:
            raise ValueError(f"{name} 形状错误，应为 {expected_shape}，实际为 {data.shape}")

    def _validate_scalar(self, value: int, name: str):
        """动作验证"""
        if not (0 <= value < self.num_actions):
            raise ValueError(f"{name} 超出范围，允许的最大值：{self.num_actions - 1}")

    @property
    def num_actions(self) -> int:
        """动作空间维度"""
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
