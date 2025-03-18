import copy
import os
import torch
import numpy as np
import torch.nn.functional as F
from utils import Double_Q_Net, ReplayBuffer


class DQN_agent:

    def __init__(self, **kwargs):
        # 初始化超参数，类似于"self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        self.tau = 0.005
        self.train_counter = 0
        self.epsilon = 1.0  # 初始探索率
        self.epsilon_min = 0.02  # 最小探索率
        self.epsilon_decay = 0.995  # 探索率衰减因子
        
        # 初始化经验回放缓冲区
        self.replay_buffer = ReplayBuffer(service_shape=(self.time_steps, self.service_num, self.service_feat_dim),
                                          latency_shape=(self.time_steps, self.latency_feat_dim),
                                          buffer_size=int(1e6),
                                          num_actions=self.action_dim)

        # 初始化Q网络
        self.q_net = Double_Q_Net(num_actions=self.action_dim,
                                 service_feature_dim=self.service_feat_dim,
                                 latency_feature_dim=self.latency_feat_dim,
                                 time_steps=self.time_steps,
                                 hidden_dim=self.hidden_dim,
                                 fc_width=self.fc_width).to(self.dvc)

        # 初始化优化器
        self.q_optimizer = torch.optim.Adam(self.q_net.parameters(), lr=self.lr)
        
        # 初始化目标网络
        self.q_target = copy.deepcopy(self.q_net)
        for p in self.q_target.parameters():
            p.requires_grad = False

    def select_action(self, service_data, latency_data, deterministic=False):
        with torch.no_grad():
            # 输入预处理
            service = torch.FloatTensor(service_data).unsqueeze(0).to(self.dvc)  # [1,T,S,F]
            latency = torch.FloatTensor(latency_data).unsqueeze(0).to(self.dvc)  # [1,T,D]
            
            # ε-greedy策略
            if deterministic or np.random.rand() > self.epsilon:
                # 选择Q值最大的动作
                q_values, _ = self.q_net(service, latency)
                action = q_values.argmax(-1).item()
            else:
                # 随机选择动作
                action = np.random.randint(0, self.action_dim)
                
            return action

    def train(self):
        self.train_counter += 1
        service, latency, a, r, service_next, latency_next, dw = self.replay_buffer.sample_batch(self.batch_size)

        # ------------------------------------------ 训练Q网络 ----------------------------------------#
        # 计算目标Q值
        with torch.no_grad():
            # 获取下一状态的最大Q值
            next_q1, next_q2 = self.q_target(service_next, latency_next)  # [b,a_dim]
            next_q = torch.min(next_q1, next_q2)
            max_next_q = next_q.max(dim=1, keepdim=True)[0]  # [b,1]
            target_Q = r + (1 - dw) * self.gamma * max_next_q

        # 计算当前Q值
        q1, q2 = self.q_net(service, latency)  # [b,a_dim]
        a = a.long().unsqueeze(-1)  # 增加维度 [batch_size] => [batch_size, 1]
        q1_a = q1.gather(1, a)
        q2_a = q2.gather(1, a)
        
        # 计算损失并更新网络
        q_loss = F.mse_loss(q1_a, target_Q) + F.mse_loss(q2_a, target_Q)
        self.q_optimizer.zero_grad()
        q_loss.backward()

        # 梯度裁剪
        torch.nn.utils.clip_grad_norm_(self.q_net.parameters(), 0.5)
        self.q_optimizer.step()

        # 更新目标网络
        if self.train_counter % self.update_steps == 0:
            for param, target_param in zip(self.q_net.parameters(), self.q_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
        
        # 衰减探索率
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def save(self, time, steps):
        save_path = f"./model/{time}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(self.q_net.state_dict(), f"{save_path}/dqn_{time}_{steps}.pth")

    def load(self, time, steps, dir):
        save_path = f"{dir}/"
        self.q_net.load_state_dict(torch.load(f"{save_path}/dqn_{time}_{steps}.pth", map_location=self.dvc))
        self.q_target = copy.deepcopy(self.q_net)


def test_dqn():
    """
    测试DQN算法的基本功能
    参数设置：T=30, S=28, F=26, D=6, 动作空间=8
    """
    import numpy as np
    import torch

    # 创建一个DQN_agent实例
    agent = DQN_agent(action_dim=8,
                     dvc="cuda" if torch.cuda.is_available() else "cpu",
                     time_steps=30,
                     service_num=28,
                     service_feat_dim=26,
                     latency_feat_dim=6,
                     hidden_dim=128,
                     fc_width=256,
                     lr=0.001,
                     gamma=0.99,
                     update_steps=1000,
                     batch_size=64)

    # 创建模拟的服务状态数据 (T=30, S=28, F=26)
    service_state = np.random.rand(30, 28, 26).astype(np.float32)

    # 创建模拟的延迟状态数据 (T=30, D=6)
    latency_state = np.random.rand(30, 6).astype(np.float32)

    # 测试确定性动作选择
    deterministic_action = agent.select_action(service_state, latency_state, deterministic=True)
    print(f"确定性动作: {deterministic_action}")
    assert 0 <= deterministic_action < 8, "确定性动作超出范围"

    # 测试随机动作选择（通过epsilon-greedy）
    agent.epsilon = 1.0  # 设置为最大探索率确保随机选择
    random_action = agent.select_action(service_state, latency_state, deterministic=False)
    print(f"随机动作: {random_action}")
    assert 0 <= random_action < 8, "随机动作超出范围"

    # 填充回放缓冲区用于训练测试
    for _ in range(agent.batch_size * 2):
        # 添加随机经验
        service_data = np.random.rand(30, 28, 26).astype(np.float32)
        latency_data = np.random.rand(30, 6).astype(np.float32)
        action = np.random.randint(0, 8)
        reward = np.random.randn()
        next_service_data = np.random.rand(30, 28, 26).astype(np.float32)
        next_latency_data = np.random.rand(30, 6).astype(np.float32)
        done = np.random.rand() < 0.1
        
        agent.replay_buffer.add_experience(
            service_data, latency_data, action, reward, 
            next_service_data, next_latency_data, done)
    
    # 测试训练
    initial_params = [p.clone() for p in agent.q_net.parameters()]
    
    print("开始训练测试...")
    for i in range(10):
        agent.train()
        if i % 5 == 0:
            print(f"完成训练步骤 {i+1}/10, 当前探索率: {agent.epsilon:.4f}")
    
    # 检查参数是否更新
    params_changed = any(not torch.equal(p1, p2) for p1, p2 in zip(initial_params, agent.q_net.parameters()))
    print(f"Q网络参数已更新: {params_changed}")
    
    print("测试通过！DQN算法工作正常。")


if __name__ == "__main__":
    test_dqn()
