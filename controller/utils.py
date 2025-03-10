from typing import Tuple
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch


class Permute(nn.Module):

    def __init__(self, *dims):
        super(Permute, self).__init__()
        self.dims = dims

    def forward(self, x):
        return x.permute(*self.dims)


class SAC_StateEncoder(nn.Module):
    """SAC专用的轻量级状态编码器，融合服务和延迟特征"""

    def __init__(self, service_feature_dim=26, latency_feature_dim=6, time_steps=30, service_num=28, hidden_dim=128):
        super().__init__()

        # 服务特征轻量编码 (处理形状: B, T, S, F)
        self.service_encoder = nn.Sequential(
            nn.Conv2d(
                in_channels=time_steps,  # 时间维度作为通道
                out_channels=32,
                kernel_size=(3, 3),
                padding=1),  # (B, T, S, F) -> (B, 32, S, F)
            Permute(0, 2, 1, 3),  # (B, 32, S, F) -> (B, S, 32, F)
            nn.ReLU(),
            nn.Flatten(start_dim=2),  # (B, S, 32, F) -> (B, S, 32 * F)
            nn.Linear(32 * service_feature_dim, 64),  # (B, S, 32 * F) -> (B, S, 64)
            nn.LayerNorm(64),
            nn.ReLU(),
            nn.Linear(64, hidden_dim // 2))  # (B, S, 64) -> (B, S, hidden_dim // 2)

        # 延迟特征时序编码 (处理形状: B,T,D)
        self.latency_encoder = nn.Sequential(
            nn.Linear(latency_feature_dim, 32),
            nn.ReLU(),  # (B, T, D) -> (B, T, 32)
            nn.LSTM(input_size=32, hidden_size=hidden_dim // 2, num_layers=1,
                    batch_first=True))  # (B, T, 32) -> (B, T, 64)
        self.latency_proj = nn.Linear(hidden_dim // 2, hidden_dim // 2)

        # 特征融合层
        self.fusion = nn.Sequential(nn.Linear(hidden_dim, hidden_dim), nn.LayerNorm(hidden_dim), nn.Mish())

    def forward(self, service_data, latency_data):
        """处理两种输入并生成联合特征"""
        # 服务特征处理 (B,T,S,F) → (B,H//2)
        B, T, S, F = service_data.size()
        service_feat = self.service_encoder(service_data)  # [B, T, S, F] -> [B, S, 64]
        service_feat = service_feat.mean(dim=1)  # [B, S, 64] -> [B, 64]

        # 延迟特征处理 (B,T,D) → (B,H//2)
        latency_out, _ = self.latency_encoder(latency_data.view(B, T, -1))
        latency_feat = self.latency_proj(latency_out.mean(dim=1))  # 取所有时间步的平均 （B, 64）-> (B, 64)

        # 特征融合
        combined = torch.cat([service_feat, latency_feat], dim=1)  # (B, 128)
        return self.fusion(combined)


class Q_Net(nn.Module):
    """离散SAC专用Q网络，集成状态编码器"""

    def __init__(self, num_actions=8, service_feature_dim=26, latency_feature_dim=6, time_steps=30, hidden_dim=128):
        super().__init__()

        # 联合特征编码器
        self.encoder = SAC_StateEncoder(service_feature_dim=service_feature_dim,
                                        latency_feature_dim=latency_feature_dim,
                                        time_steps=time_steps,
                                        hidden_dim=hidden_dim)

        # Q值预测头（支持离散动作空间）
        self.q_head = nn.Sequential(nn.Linear(hidden_dim, 256), nn.LayerNorm(256), nn.Mish(), nn.Dropout(0.1),
                                    nn.Linear(256, num_actions))

    def forward(self, service_data, latency_data):
        """
        输入:
        - service_data: (B, T, S, F) 服务特征矩阵，支持动态S
        - latency_data: (B, T, D=6) 延迟时序数据
        输出:
        - q_values: (B, A) 各动作Q值
        """
        # 联合特征编码
        state_feat = self.encoder(service_data, latency_data)  # (B, H)

        # 生成动作价值
        return self.q_head(state_feat)


class Duel_Q_Net(nn.Module):
    """支持动态服务输入的Dueling DQN，整合SAC编码器"""

    def __init__(self,
                 num_actions=8,
                 service_feature_dim=26,
                 latency_feature_dim=6,
                 time_steps=30,
                 hidden_dim=128,
                 fc_width=256):
        super().__init__()

        # 联合特征编码器
        self.encoder = SAC_StateEncoder(service_feature_dim=service_feature_dim,
                                        latency_feature_dim=latency_feature_dim,
                                        time_steps=time_steps,
                                        hidden_dim=hidden_dim)

        # Dueling结构分支
        self.value_stream = nn.Sequential(nn.Linear(hidden_dim, fc_width), nn.LayerNorm(fc_width), nn.Mish(),
                                          nn.Linear(fc_width, 1))

        self.advantage_stream = nn.Sequential(nn.Linear(hidden_dim, fc_width), nn.LayerNorm(fc_width), nn.Mish(),
                                              nn.Linear(fc_width, num_actions))

    def forward(self, service_data, latency_data):
        """
        输入: 
        - service_data: (B, T, S, F) 动态服务数据
        - latency_data: (B, T, D) 延迟时序数据
        输出: 
        - Q值 (B, A)
        """
        # 特征编码
        state_feat = self.encoder(service_data, latency_data)  # (B, H)

        # Dueling计算
        V = self.value_stream(state_feat)
        A = self.advantage_stream(state_feat)
        Q = V + (A - A.mean(dim=-1, keepdim=True))

        return Q


class Double_Q_Net(nn.Module):
    """基于SAC状态编码器的双Q网络，支持动态服务输入"""

    def __init__(self,
                 num_actions=8,
                 service_feature_dim=26,
                 latency_feature_dim=6,
                 time_steps=30,
                 hidden_dim=128,
                 fc_width=256):
        super().__init__()

        # 共享编码器结构的双Q网络
        self.Q1 = Duel_Q_Net(num_actions=num_actions,
                             service_feature_dim=service_feature_dim,
                             latency_feature_dim=latency_feature_dim,
                             time_steps=time_steps,
                             hidden_dim=hidden_dim,
                             fc_width=fc_width)

        self.Q2 = Duel_Q_Net(num_actions=num_actions,
                             service_feature_dim=service_feature_dim,
                             latency_feature_dim=latency_feature_dim,
                             time_steps=time_steps,
                             hidden_dim=hidden_dim,
                             fc_width=fc_width)

    def forward(self, service_data, latency_data):
        """
        输入: 
        - service_data: (B, T, S, F) 动态服务数据
        - latency_data: (B, T, D) 延迟时序数据
        输出: 
        - Q1, Q2: (B, A) 两个Q网络的预测
        """
        return self.Q1(service_data, latency_data), self.Q2(service_data, latency_data)


class Double_Duel_Q_Net(nn.Module):
    """基于SAC编码器的双Dueling Q网络"""

    def __init__(self,
                 num_actions=8,
                 service_feature_dim=26,
                 latency_feature_dim=6,
                 time_steps=30,
                 hidden_dim=128,
                 fc_width=256):
        super().__init__()

        # 共享编码器架构的双Q网络
        self.Q1 = Duel_Q_Net(num_actions=num_actions,
                             service_feature_dim=service_feature_dim,
                             latency_feature_dim=latency_feature_dim,
                             time_steps=time_steps,
                             hidden_dim=hidden_dim,
                             fc_width=fc_width)

        self.Q2 = Duel_Q_Net(num_actions=num_actions,
                             service_feature_dim=service_feature_dim,
                             latency_feature_dim=latency_feature_dim,
                             time_steps=time_steps,
                             hidden_dim=hidden_dim,
                             fc_width=fc_width)

    def forward(self, service_data, latency_data):
        """
        输入: 
        - service_data: (B, T, S, F) 动态服务数据
        - latency_data: (B, T, D) 延迟时序数据
        输出: 
        - q1, q2: (B, A) 双Q网络预测值
        """
        return self.Q1(service_data, latency_data), self.Q2(service_data, latency_data)


class Policy_Net(nn.Module):
    """适配动态服务的策略网络"""

    def __init__(self,
                 num_actions=8,
                 service_feature_dim=26,
                 latency_feature_dim=6,
                 time_steps=30,
                 hidden_dim=128,
                 fc_width=256):
        super().__init__()

        # 共享状态编码器
        self.encoder = SAC_StateEncoder(service_feature_dim=service_feature_dim,
                                        latency_feature_dim=latency_feature_dim,
                                        time_steps=time_steps,
                                        hidden_dim=hidden_dim)

        # 策略网络主体
        self.policy_net = nn.Sequential(nn.Linear(hidden_dim, fc_width), nn.LayerNorm(fc_width), nn.Mish(),
                                        nn.Dropout(0.1), nn.Linear(fc_width, num_actions))

    def forward(self, service_data, latency_data):
        """
        输入:
        - service_data: (B, T, S, F)
        - latency_data: (B, T, D)
        输出:
        - action_probs: (B, A) 动作概率分布
        """
        # 特征编码
        state_feat = self.encoder(service_data, latency_data)

        # 生成策略
        logits = self.policy_net(state_feat)
        return F.gumbel_softmax(logits, tau=1.0, hard=False)


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
        self.num_actions = num_actions

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
        
        # 获取设备信息
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        return (
            torch.FloatTensor(self.service_states[indices]).to(device),  # (B,30,28,26)
            torch.FloatTensor(self.latency_states[indices]).to(device),  # (B,30,6)
            torch.LongTensor(self.actions[indices]).to(device),  # (B,)
            torch.FloatTensor(self.rewards[indices]).unsqueeze(1).to(device),  # (B,1)
            torch.FloatTensor(self.next_service_states[indices]).to(device),  # (B,30,28,26)
            torch.FloatTensor(self.next_latency_states[indices]).to(device),  # (B,30,6)
            torch.FloatTensor(self.dones[indices].astype(np.float32)).unsqueeze(1).to(device),  # (B,1)
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
