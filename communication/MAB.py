import os
import random
import sys
import numpy as np
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

class MAB:
    def __init__(self):
        self.actions = [
            # 固定增加
            {"type" : "increase", "value" : 1},
            # 固定减少
            {"type" : "decrease", "value" : 0.5},
            # 批量增加
            {"type" : "increase_batch", "value" : 1},
            # 批量减少
            {"type" : "decrease_batch", "value" : 0.5},
            # 百分比增加
            {"type" : "increase_percent", "value" : 0.25},
            # 百分比减少
            {"type" : "decrease_percent", "value" : 0.1},
            # 重置
            {"type" : "reset", "value" : 0},
            # 保持
            {"type" : "hold", "value" : 0},
            # 恢复到上一次的分配
            {"type" : "recover", "value" : 0}
        ]

         # 上下文特征列表（环境提供的指标）
        self.context_features = [
            "cpu_mean",
            "cpu_max",
            "cpu_min",
            "latencys",
            "action_count"  # 可用于记录探索步数或其他计数信息
        ]

        # cpu成本因子
        self.cpu_cost_factor = 1
        # 延迟成本因子
        self.latency_cost_factor = 1
        
        # Epsilon-Greedy 参数
        self.base_epsilon = 0.05
        self.min_epsilon = 0.01
        self.epsilon_decay = 0.99
        self.epsilon = self.base_epsilon

        # 学习率（在线更新线性模型时使用）
        self.learning_rate = 0.01

        # 为每个动作初始化一个线性模型（模型维度为：上下文特征数量+1，用于偏置）
        self.models = {}
        d = len(self.context_features) + 1  # +1 是偏置项
        for i in range(len(self.actions)):
            # 可用0初始化，也可用较小的随机数： np.random.randn(d)*0.01
            self.models[i] = np.zeros(d)
        
        # 记录整体的探索步数（可作为上下文的一个特征）
        self.iteration = 0
    
    # 计算动作成本
    # cpu_allocation: 当前的cpu分配 服务->cpu分配数
    # latency: 当前的端到端延迟
    # TODO 参数与数值待定
    def calculate_cost(self, cpu_allocation : dict[str, int], latency : int):
        # 计算cpu成本
        cpu_cost = sum(cpu_allocation.values())
        # 计算延迟成本
        if latency < 100:
            latency_cost = 0
        elif latency < 300:
            latency_cost = 1
        elif latency < 500:
            latency_cost = 2
        else:
            latency_cost = 10
        return self.cpu_cost_factor * cpu_cost + self.latency_cost_factor * latency_cost


    def get_actions(self):
        return self.actions

    def get_context_features(self):
        return self.context_features

    def get_context_vector(self, context_data: dict):
        """
        根据上下文数据构造特征向量。
        此处将偏置项（1）放在第一位，其后按照 context_features 顺序添加特征值。
        如果某个特征在 context_data 中不存在，默认取0。
        """
        vector = [1]  # 偏置项
        for feature in self.context_features:
            vector.append(context_data.get(feature, 0))
        return np.array(vector)

    def predict_reward(self, action_index: int, context_vector: np.ndarray):
        """
        利用线性模型预测给定动作在当前上下文下的奖励。
        注意：我们的奖励定义为负成本。
        """
        weights = self.models[action_index]
        prediction = np.dot(weights, context_vector)
        return prediction

    def select_action(self, context_data: dict):
        """
        根据当前上下文数据使用 epsilon-greedy 策略选择动作。
        返回选中动作的索引以及对应的上下文向量。
        """
        context_vector = self.get_context_vector(context_data)
        if random.random() < self.epsilon:
            action_index = random.randint(0, len(self.actions) - 1)
        else:
            # 选择预测奖励最高的动作
            predictions = [self.predict_reward(i, context_vector) for i in range(len(self.actions))]
            action_index = int(np.argmax(predictions))
        return action_index, context_vector
    
    def update_model(self, action_index: int, context_vector: np.ndarray, reward: float):
        """
        使用梯度下降更新选定动作的线性模型参数。
        我们最小化 (w^T x - reward)^2 的均方误差损失。
        """
        weights = self.models[action_index]
        prediction = np.dot(weights, context_vector)
        error = prediction - reward
        # 梯度： error * context_vector
        gradient = error * context_vector
        self.models[action_index] = weights - self.learning_rate * gradient
    
    def decay_epsilon(self):
        """按照衰减因子更新 epsilon，但不低于最小值。"""
        self.epsilon = max(self.min_epsilon, self.epsilon * self.epsilon_decay)

    def update(self, context_data: dict, action_index: int, observed_cost: float, action_count: int):
        """
        将观察到的成本转换为奖励（负成本），更新对应动作的模型，
        并将当前探索步数作为上下文的一部分进行更新。
        """
        reward = -observed_cost  # 奖励定义为负成本
        # 将当前步数作为上下文中的 action_count 特征
        context_data['action_count'] = action_count
        context_vector = self.get_context_vector(context_data)
        self.update_model(action_index, context_vector, reward)
        self.decay_epsilon()

    def run_step(self, env):
        """
        执行一次决策步骤。需要一个环境对象 env，该对象需要提供：
          - get_context() 方法：返回当前上下文数据，格式为 dict，包含 cpu_mean、cpu_max、cpu_min、latencys 等指标；
          - execute_action(action) 方法：根据给定动作执行资源调整，返回调整后的 cpu_allocation（dict）与 latency（int）。
        步骤：
          1. 根据上下文数据选择动作。
          2. 通过 env 执行动作并获取新的状态。
          3. 计算新的成本（包括 cpu 和延迟成本）。
          4. 用负成本作为奖励更新模型。
          5. 返回选中动作、奖励及新的状态数据。
        """
        # 获取当前环境上下文
        context_data = env.get_context()
        action_index, context_vector = self.select_action(context_data)
        action = self.actions[action_index]
        # 执行动作（实际部署时，这里应调用相应的 API）
        new_cpu_allocation, new_latency = env.execute_action(action)
        # 计算调整后的成本
        cost = self.calculate_cost(new_cpu_allocation, new_latency)
        reward = -cost

        self.iteration += 1
        # 更新模型：将当前探索步数加入上下文
        self.update(context_data, action_index, cost, self.iteration)

        return action, reward, new_cpu_allocation, new_latency



import numpy as np

class UCB_Bandit:
    """
    UCB算法实现的多臂赌博机
    """
    def __init__(self, allocate_dict, replica_dict):
        self.actions = [
            # 固定增加
            {"type" : "increase", "value" : 1},
            # 固定减少
            {"type" : "decrease", "value" : 0.5},
            # 批量增加
            {"type" : "increase_batch", "value" : 1},
            # 批量减少
            {"type" : "decrease_batch", "value" : 0.5},
            # 百分比增加
            {"type" : "increase_percent", "value" : 0.25},
            # 百分比减少
            {"type" : "decrease_percent", "value" : 0.1},
            # 重置
            {"type" : "reset", "value" : 0},
            # 保持
            {"type" : "hold", "value" : 0},
            # 恢复到上一次的分配
            {"type" : "recover", "value" : 0}
        ]

        self.allocate_dict : dict[str, int] = allocate_dict  #初始分配字典
        self.replica_dict : dict[str, int]  = replica_dict

        self.k = len(self.actions)        # 臂的数量
        self.counts = np.zeros(self.k)  # 每个臂的尝试次数
        self.values = np.zeros(self.k)  # 当前价值估计
        self.total_counts = 0           # 总尝试次数

        self.last_allocate = self.allocate_dict # 上一次的分配 用于recover动作
        self.config = self.setup_config()


    def setup_config(self):
        """
        从 'communication/mab.json' 文件中读取配置，返回配置字典。
        """
        config_file_path = os.path.join(PROJECT_ROOT, "communication", "mab.json")
        
        try:
            with open(config_file_path, 'r') as f:
                config = json.load(f)  # 解析 JSON 文件
            return config
        except FileNotFoundError:
            print(f"配置文件 '{config_file_path}' 未找到！")
            return {}
        except json.JSONDecodeError:
            print(f"配置文件 '{config_file_path}' 解析失败！")
            return {}


    def select_arm(self):
        """选择要拉动的臂（基于UCB公式）"""
        ucb_values = np.zeros(self.k)
        
        for arm in range(self.k):
            if self.counts[arm] == 0:   # 未尝试过的臂优先选择
                ucb_values[arm] = float('inf')
            else:
                # UCB公式：价值估计 + 探索项
                exploration = np.sqrt(2 * np.log(self.total_counts) / self.counts[arm])
                ucb_values[arm] = self.values[arm] + exploration
        
        # 选择UCB值最大的臂（若有多个则随机选择）
        max_ucb = np.max(ucb_values)
        candidates = np.where(ucb_values == max_ucb)
        return np.random.choice(candidates)
    
    def update(self, chosen_arm, reward):
        """更新选定臂的统计信息"""
        self.counts[chosen_arm] += 1    # 更新尝试次数
        self.total_counts += 1          # 更新总次数
        
        # 增量式更新价值估计（避免存储所有奖励）
        n = self.counts[chosen_arm]
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / n
    

    # 根据分配的CPU数量和latency来计算奖励
    def calculate_reward(self, latency):
        """
        根据分配的CPU数量和延迟来计算奖励
        CPU数量映射：从0到max_cpu映射到1到0
        延迟映射：从0到500映射到0.5到0，超过500则返回-1的奖励
        """
        # 计算当前的总CPU分配
        total_cpu_allocation = 0
        for service in self.allocate_dict:
            total_cpu_allocation += self.allocate_dict[service] * self.replica_dict.get(service, 1)

        # 获取配置中的最大CPU数
        max_cpu = self.config.get('max_cpu', 100)  # 默认值为100，如果配置文件中没有该项

        # 计算CPU奖励（映射：0->max_cpu 映射到 1->0）
        if total_cpu_allocation <= max_cpu:
            cpu_reward = 1 - (total_cpu_allocation / max_cpu)
        else:
            cpu_reward = 0  # 超过最大CPU分配时，奖励为0

        # 延迟奖励映射：如果延迟小于500，按比例映射；如果超过500，返回-1
        if latency < 500:
            latency_reward = 0.5 - (latency / 1000)  # 映射到0.5到0之间
        elif latency >= 500:
            latency_reward = -1  # 超过500的延迟，奖励为-1

        # 总奖励为CPU奖励和延迟奖励的加权和
        total_reward = self.config['cpu_factor'] * cpu_reward + self.config['latency_factor'] * latency_reward

        return total_reward


    def execute_action(self, chosen_arm, cpu_state):
        """
        根据选择的 arm 下标和 cpu_state 执行相应的动作
        cpu_state 是一个字典，包含所有服务的 CPU 利用率，如 {'service1': 0.8, 'service2': 0.5, ...}
        """
        action = self.actions[chosen_arm]  # 获取选定的动作
        action_type = action["type"]
        action_value = action["value"]
        
        # 判断动作类型并根据 cpu_state 执行相应的调整
        if action_type == "increase":
            # 固定增加 CPU 使用率
            service = list(cpu_state.keys())[0]  # 假设我们只增加第一个服务的CPU
            cpu_state[service] = min(1.0, cpu_state[service] + action_value)  # 不超过 100%
        
        elif action_type == "decrease":
            # 固定减少 CPU 使用率
            service = list(cpu_state.keys())[0]  # 假设我们只减少第一个服务的CPU
            cpu_state[service] = max(0.0, cpu_state[service] - action_value)  # 不低于 0%

        elif action_type == "increase_batch":
            # 批量增加 CPU 使用率
            for service in cpu_state:
                cpu_state[service] = min(1.0, cpu_state[service] + action_value)  # 不超过 100%

        elif action_type == "decrease_batch":
            # 批量减少 CPU 使用率
            for service in cpu_state:
                cpu_state[service] = max(0.0, cpu_state[service] - action_value)  # 不低于 0%

        elif action_type == "increase_percent":
            # 百分比增加 CPU 使用率
            for service in cpu_state:
                cpu_state[service] = min(1.0, cpu_state[service] * (1 + action_value))  # 不超过 100%

        elif action_type == "decrease_percent":
            # 百分比减少 CPU 使用率
            for service in cpu_state:
                cpu_state[service] = max(0.0, cpu_state[service] * (1 - action_value))  # 不低于 0%

        elif action_type == "reset":
            # 重置 CPU 使用率
            for service in cpu_state:
                cpu_state[service] = 0.0  # 重置为 0%

        elif action_type == "hold":
            # 保持当前状态，不做任何改变
            pass

        elif action_type == "recover":
            # 恢复到上一次的 CPU 分配
            cpu_state = self.last_allocate  # 假设 self.last_allocate 保存了上一次的分配

        return cpu_state


# 示例使用
if __name__ == "__main__":
    np.random.seed(42)  # 设置随机种子保证可重复性
    
    # 创建赌博机实例（5个臂，真实均值为1-5）
    bandit = UCB_Bandit(true_means=[1.0, 2.0, 3.0, 4.0, 5.0])
    
    # 进行1000次试验
    num_trials = 1000
    rewards = []  # 记录每次奖励
    
    for _ in range(num_trials):
        arm = bandit.select_arm()       # 选择臂
        reward = bandit.pull_arm(arm)   # 获得奖励
        bandit.update(arm, reward)      # 更新统计信息
        rewards.append(reward)          # 记录奖励
    
    # 结果分析
    print(f"各臂尝试次数: {bandit.counts}")
    print(f"平均奖励: {np.mean(rewards):.2f}")
    print(f"最优臂选择比例: {bandit.counts[-1]/num_trials:.2%}")

# 典型输出：
# 各臂尝试次数: [  3.   3.   4.  32. 958.]
# 平均奖励: 4.94
# 最优臂选择比例: 95.80%