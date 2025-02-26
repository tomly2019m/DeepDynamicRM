from typing import Tuple
import numpy as np
import torch.nn.functional as F
import torch.nn as nn
import argparse
import torch


class Q_Net(nn.Module):
    def __init__(self, opt):
        super(Q_Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten())
        self.fc1 = nn.Linear(64 * 7 * 7, opt.fc_width)
        self.fc2 = nn.Linear(opt.fc_width, opt.action_dim)

    def forward(self, obs):
        s = obs.float() / 255  # convert to f32 and normalize before feeding to network
        s = self.conv(s)
        s = torch.relu(self.fc1(s))
        q = self.fc2(s)
        return q


class Duel_Q_Net(nn.Module):
    def __init__(self, opt):
        super(Duel_Q_Net, self).__init__()
        self.conv = nn.Sequential(
            nn.Conv2d(4, 32, 8, stride=4),
            nn.ReLU(),
            nn.Conv2d(32, 64, 4, stride=2),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, stride=1),
            nn.ReLU(),
            nn.Flatten())
        self.fc1 = nn.Linear(64 * 7 * 7, opt.fc_width)
        self.A = nn.Linear(opt.fc_width, opt.action_dim)
        self.V = nn.Linear(opt.fc_width, 1)

    def forward(self, obs):
        s = obs.float() / 255  # convert to f32 and normalize before feeding to network
        s = self.conv(s)
        s = torch.relu(self.fc1(s))
        Adv = self.A(s)
        V = self.V(s)
        Q = V + (Adv - torch.mean(Adv, dim=-1, keepdim=True))  # Q(s,a)=V(s)+A(s,a)-mean(A(s,a))
        return Q


class Double_Q_Net(nn.Module):
    def __init__(self, opt):
        super(Double_Q_Net, self).__init__()

        self.Q1 = Q_Net(opt)
        self.Q2 = Q_Net(opt)

    def forward(self, s):
        q1 = self.Q1(s)
        q2 = self.Q2(s)
        return q1, q2


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
    def __init__(self, opt):
        super(Policy_Net, self).__init__()
        self.P = Q_Net(opt)

    def forward(self, s):
        logits = self.P(s)
        probs = F.softmax(logits, dim=1)
        return probs


class ReplayBuffer:
    """经验回放池"""
    def __init__(self, 
                 buffer_size: int = 100000, 
                 num_services: int = 28,
                 state_window: int = 10,
                 state_features: int = 25,
                 num_actions: int = 5):
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
        self.states = np.zeros((buffer_size, state_window, num_services, state_features),
                              dtype=np.float32)  # 状态序列
        self.actions = np.zeros(buffer_size, dtype=np.int64)         # 动作索引 (0~num_actions-1)
        self.rewards = np.zeros(buffer_size, dtype=np.float32)        # 即时奖励
        self.next_states = np.zeros_like(self.states)                # 下一状态
        self.dones = np.zeros(buffer_size, dtype=np.bool_)           # 终止标志
        
        # 指针管理
        self.buffer_size = buffer_size
        self.current_idx = 0    # 当前写入位置
        self.current_size = 0   # 当前有效数据量

    def add_experience(self,
                      state: np.ndarray,       # (10,28,25)
                      action: int,            # 动作索引 (标量)
                      reward: float,
                      next_state: np.ndarray, # (10,28,25)
                      done: bool) -> None:
        """添加单条经验"""
        # 输入数据验证
        self._validate_shape(state, (10,28,25), "State")
        self._validate_scalar(action, "Action")
        self._validate_shape(next_state, (10,28,25), "Next State")
        
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

    def sample_batch(self, 
                    batch_size: int) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
        """采样训练批次"""
        indices = np.random.choice(self.current_size, batch_size, replace=False)
        
        return (
            torch.FloatTensor(self.states[indices]),        # (B,10,28,25)
            torch.LongTensor(self.actions[indices]),        # (B,)
            torch.FloatTensor(self.rewards[indices]).unsqueeze(1),  # (B,1)
            torch.FloatTensor(self.next_states[indices]),   # (B,10,28,25)
            torch.FloatTensor(self.dones[indices].astype(np.float32)).unsqueeze(1)  # (B,1)
        )

    def _validate_shape(self, data: np.ndarray, expected_shape: tuple, name: str):
        """验证状态数据维度"""
        if data.shape != expected_shape:
            raise ValueError(f"Invalid {name} shape. Expected: {expected_shape}, Got: {data.shape}")

    def _validate_scalar(self, value: int, name: str):
        """验证动作是否为标量"""
        if not isinstance(value, (int, np.integer)):
            raise TypeError(f"{name} must be integer scalar. Got type: {type(value)}")
        if not (0 <= value < self.num_actions):
            raise ValueError(f"{name} index out of range. Max allowed: {self.num_actions-1}")

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
            done = (dw or tr)

            total_scores += r
            s = s_next

    return int(total_scores / turns)


# You can just ignore 'str2bool'. Is not related to the RL.
def str2bool(v):
    '''transfer str to bool for argparse'''
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'True', 'true', 'TRUE', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'False', 'false', 'FALSE', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('Boolean value expected.')


class LinearSchedule(object):
    def __init__(self, schedule_timesteps, final_p, initial_p=1.0):
        """Linear interpolation between initial_p and final_p over
        schedule_timesteps. After this many timesteps pass final_p is
        returned.

        Parameters
        ----------
        schedule_timesteps: int
            Number of timesteps for which to linearly anneal initial_p
            to final_p
        initial_p: float
            initial output value
        final_p: float
            final output value
        """
        self.schedule_timesteps = schedule_timesteps
        self.final_p = final_p
        self.initial_p = initial_p

    def value(self, t):
        fraction = min(float(t) / self.schedule_timesteps, 1.0)
        return self.initial_p + fraction * (self.final_p - self.initial_p)
