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


class ReplayBuffer(object):
    def __init__(self, device, gpu_size, mem_size, max_size=int(1e5)):
        self.dvc = device
        self.max_size = max_size

        self.size = 0
        self.ptr = 0

        self.gpu_size = gpu_size
        self.mem_size = mem_size

        # self.state = torch.zeros((max_size, 4, 84, 84), dtype=torch.uint8, device=self.dvc)
        # self.action = torch.zeros((max_size, 1), dtype=torch.int64, device=self.dvc)
        # self.reward = torch.zeros((max_size, 1), device=self.dvc)
        # self.next_state = torch.zeros((max_size, 4, 84, 84), dtype=torch.uint8, device=self.dvc)
        # self.dw = torch.zeros((max_size, 1), dtype=torch.bool, device=self.dvc)

        # 显存中的数据
        self.s_gpu = torch.zeros((self.gpu_size, 4, 84, 84), dtype=torch.uint8, device=self.dvc)
        self.a_gpu = torch.zeros((self.gpu_size, 1), dtype=torch.int64, device=self.dvc)
        self.r_gpu = torch.zeros((self.gpu_size, 1), device=self.dvc)
        self.s_next_gpu = torch.zeros((self.gpu_size, 4, 84, 84), dtype=torch.uint8, device=self.dvc)
        self.dw_gpu = torch.zeros((self.gpu_size, 1), dtype=torch.bool, device=self.dvc)

        # 内存中的数据
        self.s_cpu = torch.zeros((self.mem_size, 4, 84, 84), dtype=torch.uint8)
        self.a_cpu = torch.zeros((self.mem_size, 1), dtype=torch.int64)
        self.r_cpu = torch.zeros((self.mem_size, 1))
        self.s_next_cpu = torch.zeros((self.mem_size, 4, 84, 84), dtype=torch.uint8)
        self.dw_cpu = torch.zeros((self.mem_size, 1), dtype=torch.bool)

    def add(self, s, a, r, s_next, dw):
        """将新的经验添加到回放池中"""
        if self.size < self.gpu_size:
            # 向显存添加数据
            self.s_gpu[self.ptr] = s
            self.a_gpu[self.ptr] = a
            self.r_gpu[self.ptr] = r
            self.s_next_gpu[self.ptr] = s_next
            self.dw_gpu[self.ptr] = dw
        else:
            # 向内存添加数据
            index_mem = self.ptr - self.gpu_size
            if index_mem < self.mem_size:
                self.s_cpu[index_mem] = s
                self.a_cpu[index_mem] = a
                self.r_cpu[index_mem] = r
                self.s_next_cpu[index_mem] = s_next
                self.dw_cpu[index_mem] = dw
            else:
                print("Warning: Replay buffer has overflowed. New experiences will not be added.")

        # 更新指针和大小
        self.ptr = (self.ptr + 1) % self.max_size
        self.size = min(self.size + 1, self.max_size)

    def sample(self, batch_size):
        # ind = np.random.choice((self.size-1), batch_size, replace=False)  # Time consuming, but no duplication
        # ind = np.random.randint(0, (self.size - 1), batch_size)  # Time effcient, might duplicates

        ind = torch.randint(0, self.size, size=(batch_size,), device=self.dvc)

        # 初始化存储结果的空张量
        s_batch = torch.zeros((batch_size, 4, 84, 84), dtype=torch.float32, device=self.dvc)
        a_batch = torch.zeros((batch_size, 1), dtype=torch.int64, device=self.dvc)
        r_batch = torch.zeros((batch_size, 1), device=self.dvc)
        s_next_batch = torch.zeros((batch_size, 4, 84, 84), dtype=torch.float32, device=self.dvc)
        dw_batch = torch.zeros((batch_size, 1), dtype=torch.bool, device=self.dvc)

        # 逐个检查每个索引并从显存/内存中提取数据
        for i in range(batch_size):
            idx = ind[i].item()

            if idx < self.gpu_size:
                # 从显存中提取数据
                s_batch[i] = self.s_gpu[idx]
                s_next_batch[i] = self.s_next_gpu[idx]
                a_batch[i] = self.a_gpu[idx]
                r_batch[i] = self.r_gpu[idx]
                dw_batch[i] = self.dw_gpu[idx]
            else:
                # 从内存中提取数据
                idx_mem = idx - self.gpu_size
                s_batch[i] = self.s_cpu[idx_mem]
                s_next_batch[i] = self.s_next_cpu[idx_mem]
                a_batch[i] = self.a_cpu[idx_mem]
                r_batch[i] = self.r_cpu[idx_mem]
                dw_batch[i] = self.dw_cpu[idx_mem]

        return s_batch, a_batch, r_batch, s_next_batch, dw_batch

        # return self.state[ind].to(self.dvc), self.action[ind].to(self.dvc), self.reward[ind].to(self.dvc), \
        #     self.next_state[ind].to(self.dvc), self.dw[ind].to(self.dvc)


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
