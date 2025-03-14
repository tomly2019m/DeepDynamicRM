import copy
import os
import torch
import numpy as np
import torch.nn.functional as F
from torch.distributions.categorical import Categorical
from utils import Double_Duel_Q_Net, Double_Q_Net, Policy_Net, ReplayBuffer


class SACD_agent:

    def __init__(self, **kwargs):
        # Init hyperparameters for agent, just like "self.gamma = opt.gamma, self.lambd = opt.lambd, ..."
        self.__dict__.update(kwargs)
        self.tau = 0.005
        self.train_counter = 0
        self.H_mean = 0
        self.replay_buffer = ReplayBuffer(service_shape=(self.time_steps, self.service_num, self.service_feat_dim),
                                          latency_shape=(self.time_steps, self.latency_feat_dim),
                                          buffer_size=int(1e6),
                                          num_actions=self.action_dim)

        self.actor = Policy_Net(num_actions=self.action_dim,
                                service_feature_dim=self.service_feat_dim,
                                latency_feature_dim=self.latency_feat_dim,
                                time_steps=self.time_steps,
                                hidden_dim=self.hidden_dim,
                                fc_width=self.fc_width).to(self.dvc)

        self.actor_optimizer = torch.optim.Adam(self.actor.parameters(), lr=self.lr)

        self.q_critic = Double_Duel_Q_Net(num_actions=self.action_dim,
                                          service_feature_dim=self.service_feat_dim,
                                          latency_feature_dim=self.latency_feat_dim,
                                          time_steps=self.time_steps,
                                          hidden_dim=self.hidden_dim,
                                          fc_width=self.fc_width).to(self.dvc)

        self.q_critic_optimizer = torch.optim.Adam(self.q_critic.parameters(), lr=self.lr)
        self.q_critic_target = copy.deepcopy(self.q_critic)
        for p in self.q_critic_target.parameters():
            p.requires_grad = False

        self.alpha = kwargs.get('alpha', 0.2)  # 默认值为0.2
        if self.adaptive_alpha:
            # use 0.6 because the recommended 0.98 will cause alpha explosion.
            self.target_entropy = 0.6 * (-np.log(1 / self.action_dim))  # H(discrete)>0
            self.log_alpha = torch.tensor(np.log(self.alpha), dtype=float, requires_grad=True, device=self.dvc)
            self.alpha_optim = torch.optim.Adam([self.log_alpha], lr=self.lr)

    def select_action(self, service_data, latency_data, deterministic):
        with torch.no_grad():
            # 输入预处理
            service = torch.FloatTensor(service_data).unsqueeze(0).to(self.dvc)  # [1,T,S,F]
            latency = torch.FloatTensor(latency_data).unsqueeze(0).to(self.dvc)  # [1,T,D]
            # 获取预测结果
            probs = self.actor(service, latency)

            p = 0.02 if deterministic else self.exp_noise
            if np.random.rand() < p:
                return np.random.randint(0, self.action_dim)

            if deterministic:
                a = probs.argmax(-1).item()
            else:
                a = Categorical(probs).sample().item()
            return a

    def train(self):
        self.train_counter += 1
        service, latency, a, r, service_next, latency_next, dw = self.replay_buffer.sample_batch(self.batch_size)

        # ------------------------------------------ Train Critic ----------------------------------------#
        """Compute the target soft Q value"""
        with torch.no_grad():
            next_probs = self.actor(service_next, latency_next)  # [b,a_dim]
            next_log_probs = torch.log(next_probs + 1e-8)  # [b,a_dim]
            next_q1_all, next_q2_all = self.q_critic_target(service_next, latency_next)  # [b,a_dim]
            min_next_q_all = torch.min(next_q1_all, next_q2_all)
            v_next = torch.sum(
                next_probs * (min_next_q_all - self.alpha * next_log_probs),
                dim=1,
                keepdim=True,
            )  # [b,1]
            target_Q = r + (1 - dw) * self.gamma * v_next
        """Update soft Q net"""
        q1_all, q2_all = self.q_critic(service, latency)  # [b,a_dim]
        a = a.long().unsqueeze(-1)  # 增加维度 [batch_size] => [batch_size, 1]
        q1 = q1_all.gather(1, a)
        q2 = q2_all.gather(1, a)
        q_loss = F.mse_loss(q1, target_Q) + F.mse_loss(q2, target_Q)
        self.q_critic_optimizer.zero_grad()
        q_loss.backward()

        # 梯度裁剪 可调参数
        torch.nn.utils.clip_grad_norm_(self.q_critic.parameters(), 0.5)
        self.q_critic_optimizer.step()

        # ------------------------------------------ Train Actor ----------------------------------------#
        probs = self.actor(service, latency)  # [b,a_dim]
        log_probs = torch.log(probs + 1e-8)  # [b,a_dim]

        with torch.no_grad():
            q1_all, q2_all = self.q_critic(service, latency)  # [b,a_dim]
        min_q_all = torch.min(q1_all, q2_all)

        a_loss = torch.sum(probs * (self.alpha * log_probs - min_q_all), dim=1, keepdim=False)  # [b,]

        self.actor_optimizer.zero_grad()
        a_loss.mean().backward()

        # 梯度裁剪 可调参数
        torch.nn.utils.clip_grad_norm_(self.actor.parameters(), 0.5)

        self.actor_optimizer.step()

        # ------------------------------------------ Train Alpha ----------------------------------------#
        if self.adaptive_alpha:
            with torch.no_grad():
                self.H_mean = -torch.sum(probs * log_probs, dim=1).mean()
            alpha_loss = self.log_alpha * (self.H_mean - self.target_entropy)

            self.alpha_optim.zero_grad()
            alpha_loss.backward()
            self.alpha_optim.step()

            self.alpha = self.log_alpha.exp().item()

        # ------------------------------------------ Update Target Net ----------------------------------#
        if self.train_counter % self.update_steps == 0:
            for param, target_param in zip(self.q_critic.parameters(), self.q_critic_target.parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    def save(self, time, steps):
        save_path = f"./model/{time}/"
        if not os.path.exists(save_path):
            os.makedirs(save_path)

        torch.save(self.actor.state_dict(), f"{save_path}/sacd_actor_{time}_{steps}.pth")
        torch.save(self.q_critic.state_dict(), f"{save_path}/sacd_critic_{time}_{steps}.pth")

    def load(self, time, steps, dir):
        save_path = f"{dir}/"
        self.actor.load_state_dict(torch.load(f"{save_path}/sacd_actor_{time}_{steps}.pth", map_location=self.dvc))
        self.q_critic.load_state_dict(torch.load(f"{save_path}/sacd_critic_{time}_{steps}.pth", map_location=self.dvc))


def test_select_action():
    """
    测试select_action函数是否正常工作
    参数设置：T=30, S=28, F=26, D=6, 动作空间=8
    """
    import numpy as np
    import torch

    # 创建一个SACD_agent实例
    agent = SACD_agent(action_dim=8,
                       dvc="cuda" if torch.cuda.is_available() else "cpu",
                       time_steps=30,
                       service_num=28,
                       service_feat_dim=26,
                       latency_feat_dim=6,
                       hidden_dim=128,
                       fc_width=256,
                       lr=0.001,
                       gamma=0.99,
                       alpha=0.2,
                       update_steps=1000,
                       batch_size=64,
                       exp_noise=0.2,
                       adaptive_alpha=True)

    # 创建模拟的服务状态数据 (T=30, S=28, F=26)
    service_state = np.random.rand(30, 28, 26).astype(np.float32)

    # 创建模拟的延迟状态数据 (T=30, D=6)
    latency_state = np.random.rand(30, 6).astype(np.float32)

    # 测试确定性动作选择
    deterministic_action = agent.select_action(service_state, latency_state, deterministic=True)
    print(f"确定性动作: {deterministic_action}")
    assert 0 <= deterministic_action < 8, "确定性动作超出范围"

    # 测试随机动作选择
    agent.exp_noise = 0.2  # 设置探索噪声
    random_action = agent.select_action(service_state, latency_state, deterministic=False)
    print(f"随机动作: {random_action}")
    assert 0 <= random_action < 8, "随机动作超出范围"

    print("测试通过！select_action函数工作正常。")


def test_train():
    """
    测试train函数是否正常工作
    参数设置参考main.py中的argparse配置
    """
    import numpy as np
    import torch
    from torch.utils.checkpoint import checkpoint

    # ================== 参数设置 ==================
    args = {
        'service_num': 28,
        'service_feat_dim': 26,
        'latency_feat_dim': 6,
        'time_steps': 30,
        'seed': 1024,
        'exp_noise': 1.0,
        'init_noise': 1.0,
        'noise_steps': 5000,
        'final_noise': 0.02,
        'hidden_dim': 128,
        'fc_width': 256,
        'gamma': 0.99,
        'tau': 0.005,
        'lr': 1e-4,
        'batch_size': 256,
        'adaptive_alpha': True,
        'stop_steps': 100,  # 缩短测试步数
        'replay_size': int(1e6),
        'alpha': 0.2,
        'random_steps': 500,
        'action_dim': 8,
        'update_steps': 1000,
        'dvc': "cuda" if torch.cuda.is_available() else "cpu"
    }

    # ================== 初始化智能体 ==================
    agent = SACD_agent(**args)

    # ================== 生成模拟数据填充回放缓冲区 ==================
    num_samples = max(2 * args['batch_size'], args['random_steps'])
    print(f"Generating {num_samples} simulated experiences...")

    for _ in range(num_samples):
        # 生成随机服务数据 [T, S, F]
        service_data = np.random.randn(args['time_steps'], args['service_num'],
                                       args['service_feat_dim']).astype(np.float32)

        # 生成随机延迟数据 [T, D]
        latency_data = np.random.randn(args['time_steps'], args['latency_feat_dim']).astype(np.float32)

        # 随机生成动作、奖励、终止标志
        action = np.random.randint(0, args['action_dim'])
        reward = np.random.randn()
        done = np.random.rand() < 0.05  # 5%概率终止

        # 添加至回放缓冲区
        agent.replay_buffer.add_experience(
            service_data,
            latency_data,
            action,
            reward,
            service_data.copy(),  # 简化：next_state=state
            latency_data.copy(),
            done)

    # ================== 训练验证 ==================
    print("\nStart training test...")
    initial_actor_params = [p.clone() for p in agent.actor.parameters()]
    initial_critic_params = [p.clone() for p in agent.q_critic.parameters()]

    loss_history = []
    alpha_history = []

    for step in range(50):  # 运行50次训练步骤
        agent.train()

        # 记录训练指标
        if hasattr(agent, 'alpha'):
            alpha_history.append(agent.alpha)

        # 每10步打印状态
        if step % 10 == 0:
            status = f"Step {step+1}/50"
            if loss_history:
                status += f" | Latest Loss: {loss_history[-1]:.4f}"
            if alpha_history:
                status += f" | Alpha: {alpha_history[-1]:.4f}"
            print(status)

    # ================== 验证训练效果 ==================
    # 1. 检查网络参数是否更新
    params_changed = any(not torch.equal(p1, p2) for p1, p2 in zip(initial_actor_params, agent.actor.parameters()))
    print(f"\nActor parameters changed: {params_changed}")

    params_changed = any(not torch.equal(p1, p2) for p1, p2 in zip(initial_critic_params, agent.q_critic.parameters()))
    print(f"Critic parameters changed: {params_changed}")

    # 2. 检查熵系数调整是否生效
    if args['adaptive_alpha']:
        print(f"\nAlpha values during training: {alpha_history[-5:]}...")
        assert abs(alpha_history[-1] - alpha_history[0]) > 0.01, "Alpha not adapting!"

    # 3. 检查目标网络更新
    if agent.train_counter >= args['update_steps']:
        target_diff = sum((t_p - q_p).abs().sum().item()
                          for t_p, q_p in zip(agent.q_critic_target.parameters(), agent.q_critic.parameters()))
        print(f"Target network difference: {target_diff:.4f}")
        assert target_diff > 0, "Target network not updating"

    print("\n测试通过！train函数工作正常。")


if __name__ == "__main__":
    # test_select_action()
    test_train()
