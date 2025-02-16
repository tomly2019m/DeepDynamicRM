# 上下文感知的混合赌博机

import random
import numpy as np


class HCB:
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
        self.cpu_cost_factor = 0.5
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

# 以下给出一个简单的模拟环境示例，仅用于演示 HCB 的调用方式
class DummyEnv:
    def __init__(self):
        # 初始模拟状态
        self.cpu_allocation = {"service1": 40, "service2": 55, "service3": 60}
        self.latency = 30  # 初始延迟

    def get_context(self):
        # 模拟生成上下文数据（在实际环境中可根据监控数据构造）
        context = {
            "cpu_mean": np.mean(list(self.cpu_allocation.values())),
            "cpu_max": max(self.cpu_allocation.values()),
            "cpu_min": min(self.cpu_allocation.values()),
            "latencys": self.latency
        }
        return context

    def execute_action(self, action: dict):
        """
        根据动作更新 cpu_allocation 和 latency。
        注意：这里仅作简单模拟，不代表实际逻辑。
        """
        action_type = action["type"]
        value = action["value"]

        # 简单模拟：调整cpu_allocation和延迟的关系
        if action_type == "increase":
            # 增加某个服务的cpu分配
            self.cpu_allocation["service1"] += value
            self.latency = max(50, self.latency - 10)
        elif action_type == "decrease":
            self.cpu_allocation["service1"] = max(0, self.cpu_allocation["service1"] - value)
            self.latency += 10
        elif action_type == "increase_batch":
            for key in self.cpu_allocation:
                self.cpu_allocation[key] += value
            self.latency = max(50, self.latency - 5)
        elif action_type == "decrease_batch":
            for key in self.cpu_allocation:
                self.cpu_allocation[key] = max(0, self.cpu_allocation[key] - value)
            self.latency += 5
        elif action_type == "increase_percent":
            for key in self.cpu_allocation:
                self.cpu_allocation[key] = int(self.cpu_allocation[key] * (1 + value))
            self.latency = max(50, self.latency - 8)
        elif action_type == "decrease_percent":
            for key in self.cpu_allocation:
                self.cpu_allocation[key] = int(self.cpu_allocation[key] * (1 - value))
            self.latency += 8
        elif action_type == "reset":
            # 重置到初始状态
            self.cpu_allocation = {"service1": 40, "service2": 55, "service3": 60}
            self.latency = 30
        elif action_type == "hold":
            # 保持不变
            pass
        elif action_type == "recover":
            # 恢复到上一次状态（这里简单模拟为重置）
            self.cpu_allocation = {"service1": 40, "service2": 55, "service3": 60}
            self.latency = 30
        return self.cpu_allocation.copy(), self.latency

# 测试示例
if __name__ == "__main__":
    bandit = HCB()
    env = DummyEnv()
    
    for i in range(500):
        action, reward, cpu_alloc, latency = bandit.run_step(env)
        print(f"步骤 {i+1}: 选择动作 {action}，获得奖励 {reward:.2f}")
        print(f"当前状态：cpu_allocation = {cpu_alloc}，latency = {latency}")
        print("-" * 50)