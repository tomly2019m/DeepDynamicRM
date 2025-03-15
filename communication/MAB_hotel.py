from copy import deepcopy
import os
import sys
import numpy as np
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from monitor.data_collector import set_cpu_limit


class UCB_Bandit:
    """
    UCB算法实现的多臂赌博机
    """

    def __init__(self, min_core):
        self.actions = [
            # 固定增加
            {
                "type": "increase",
                "value": 1
            },
            # 固定减少
            {
                "type": "decrease",
                "value": 0.5
            },
            # 批量增加
            {
                "type": "increase_batch",
                "value": 1
            },
            # 批量减少
            {
                "type": "decrease_batch",
                "value": 0.5
            },
            # 百分比增加
            {
                "type": "increase_percent",
                "value": 0.25
            },
            # 百分比减少
            {
                "type": "decrease_percent",
                "value": 0.1
            },
            # # 重置
            # {
            #     "type": "reset",
            #     "value": 0
            # },
            # 保持
            {
                "type": "hold",
                "value": 0
            },
            # 恢复到降低之前的有效分配
            {
                "type": "recover",
                "value": 0
            },
        ]

        self.allocate_dict: dict[str, float] = {}  # 每个服务总的cpu分配
        self.replica_dict: dict[str, int] = {}

        self.initial_allocation: dict[str, float] = {}  # 初始分配方案

        self.k = len(self.actions)  # 臂的数量
        self.counts = np.zeros(self.k)  # 每个臂的尝试次数
        self.values = np.zeros(self.k)  # 当前价值估计
        self.total_counts = 0  # 总尝试次数

        self.last_allocate = self.allocate_dict  # 上一次的分配 用于recover动作
        self.config = self.setup_config()
        self.last_action = {}
        self.hold_count = 0

        # 保存资源配置历史，当发生降低动作时，记录下来 默认长度为10
        self.allocation_history = []
        self.history_length = 10

        # 保存从部署配置文件中读取默认服务配置
        self.default_cpu_config: dict[str, dict[str, float]] = {}

        # 保存可调节的服务列表
        self.scalable_service = []

        self.min_core = min_core
        self.min_perrep = 0.0

        # 配置初始化
        self.setup_config()
        self._load_deployment_config()

    def get_init_allocate(self):
        return self.initial_allocation

    def setup_config(self):
        """
        从 'communication/mab_hotel.json' 文件中读取配置，返回配置字典。
        """
        config_file_path = os.path.join(PROJECT_ROOT, "communication", "mab_hotel.json")

        try:
            with open(config_file_path, "r") as f:
                config = json.load(f)  # 解析 JSON 文件
            return config
        except FileNotFoundError:
            print(f"配置文件 '{config_file_path}' 未找到！")
            return {}
        except json.JSONDecodeError:
            print(f"配置文件 '{config_file_path}' 解析失败！")
            return {}

    def _load_deployment_config(self):
        """
        从配置文件中加载并生成 allocate_dict 和 replica_dict
        """
        config_file_path = os.path.join(PROJECT_ROOT, "deploy", "config", "hotelreservation.json")

        try:
            with open(config_file_path, "r") as f:
                config = json.load(f)

            # 提取可扩展服务列表
            self.scalable_service = config.get("scalable_service", [])

            # 提取完整服务配置
            self.default_cpu_config = config.get("service", {})

            # 自动生成 allocate_dict 和 replica_dict
            for service_name, service_conf in self.default_cpu_config.items():
                # CPU分配：直接使用配置中的 cpus 字段
                self.allocate_dict[service_name] = service_conf.get("cpus", 0.0)
                self.initial_allocation = deepcopy(self.allocate_dict)

                # 副本数：使用配置中的 replica 字段
                self.replica_dict[service_name] = service_conf.get("replica", 1)

                self.min_perrep = self.min_core / sum(self.replica_dict.values())

            print(f"已自动生成初始分配：{len(self.allocate_dict)} 个服务的CPU配置")

        except FileNotFoundError:
            print(f"[错误] 部署配置文件 {config_file_path} 不存在！")
            raise  # 配置文件不存在时直接抛出异常
        except json.JSONDecodeError as e:
            print(f"[错误] 配置文件解析失败: {str(e)}")
            raise

    def select_arm(self, latency):
        """选择要拉动的臂（基于UCB公式）"""
        ucb_values = np.full(self.k, -np.inf, dtype=np.float32)
        # 在没超出slo之前 一直使用decrease策略
        arm_list = []
        decrease = [1, 3, 5]
        increase = [0, 2, 4, 6, 7]
        increase_no_hold_no_recover = [0, 2, 4]
        increase_no_recover = [0, 2, 4, 6]
        hold = [6]

        latency = latency[-2]  # P99延迟

        if self.hold_count > 0:
            if latency > 180:
                if self.hold_count >= 8:
                    arm_list = increase_no_hold_no_recover
                else:
                    arm_list = increase_no_recover
            else:
                arm_list = hold
            self.hold_count -= 1
        else:
            if latency > 180:
                arm_list = increase
                self.hold_count = 10
            else:
                arm_list = decrease

        for arm in arm_list:
            if self.counts[arm] == 0:  # 未尝试过的臂优先选择
                ucb_values[arm] = float("inf")
            else:
                # UCB公式：价值估计 + 探索项
                exploration = np.sqrt(2 * np.log(self.total_counts) / self.counts[arm])
                ucb_values[arm] = self.values[arm] + exploration

        # 选择UCB值最大的臂（若有多个则随机选择）
        max_ucb = np.max(ucb_values)
        # np.where(condition)返回一个元组，其中每个元素对应一个维度的索引数组。
        candidates = np.where(ucb_values == max_ucb)[0]
        return np.random.choice(candidates)

    def update(self, chosen_arm, reward):
        """更新选定臂的统计信息"""
        self.counts[chosen_arm] += 1  # 更新尝试次数
        self.total_counts += 1  # 更新总次数

        # 增量式更新价值估计（避免存储所有奖励）
        n = self.counts[chosen_arm]
        self.values[chosen_arm] += (reward - self.values[chosen_arm]) / n

    # 根据分配的CPU数量和latency来计算奖励
    def calculate_reward(self, latency):
        """
        根据分配的CPU数量和延迟来计算奖励
        CPU数量映射：从0到max_cpu映射到1到0
        延迟映射：从0到500映射到0.1到0，超过500则返回-1的奖励
        """
        # 计算当前的总CPU分配
        total_cpu_allocation = 0
        for service in self.allocate_dict:
            total_cpu_allocation += self.allocate_dict[service]

        latency = latency[-2]  # P99延迟

        # 获取配置中的最大CPU数
        max_cpu = self.config.get("max_cpu", 270)  # 默认值为100，如果配置文件中没有该项

        cpu_reward, latency_reward = 0, 0
        # 计算CPU奖励（映射：0->max_cpu 映射到 1->0）
        if total_cpu_allocation <= max_cpu:
            cpu_reward = 1 - (total_cpu_allocation / max_cpu)
        else:
            cpu_reward = 0  # 超过最大CPU分配时，奖励为0

        # 延迟奖励映射：如果延迟小于500，按比例映射；如果超过500，返回-1
        if latency < 200:
            latency_reward = 0.1 - (latency / 2000)  # 映射到0.5到0之间
        elif latency >= 200:
            latency_reward = -1  # 超过500的延迟，奖励为-1

        print(f"CPU奖励:{cpu_reward}, 延迟奖励:{latency_reward}")
        # 总奖励为CPU奖励和延迟奖励的加权和
        total_reward = (self.config["cpu_factor"] * cpu_reward + self.config["latency_factor"] * latency_reward)

        return total_reward

    def execute_action(self, chosen_arm: int, cpu_state: dict[str, list[float]]):
        """
        执行选定动作并返回新的资源分配

        参数：
        chosen_arm -- 选择的动作索引
        cpu_state -- CPU状态字典，格式如：
            {"service1": [max_util, min_util, mean_util, std_util], ...}

        返回：
        new_allocation -- 新的资源分配字典
        """
        action = self.actions[chosen_arm]
        action_type: str = action["type"]
        action_value = action["value"]

        # 保存当前状态到历史记录（用于recover）
        self.allocation_history.append(deepcopy(self.allocate_dict))
        if len(self.allocation_history) > self.history_length:  # 保留最近10次记录
            self.allocation_history.pop(0)

        # 计算所有服务的负载水平（利用率 / 分配资源）
        load = {}
        for service in self.scalable_service:
            mean_util = cpu_state[service][2]  # 获取平均利用率
            load[service] = (mean_util * 100) / self.allocate_dict[service]

        # 执行动作逻辑
        new_allocation = deepcopy(self.allocate_dict)

        if action_type == "increase":
            # 找到负载最高的服务增加资源
            target_service = max(load, key=lambda k: load[k])
            new_cpu = self.allocate_dict[target_service] + action["value"]
            new_cpu = min(new_cpu, self.default_cpu_config[target_service]["max_cpus"])
            new_allocation[target_service] = new_cpu

        elif action_type == "decrease":
            # 找到负载最低的服务减少资源
            # 过滤掉 allocate 值为 self.min_perrep 的服务
            candidates = [
                service for service in load if self.allocate_dict[service] -
                self.min_perrep * self.replica_dict[service] > 1e-4  # 检查 allocate 值是否为 self.min_perrep
            ]
            if len(candidates) > 0:
                target_service = min(candidates, key=lambda k: load[k])
                new_allocation[target_service] = max(
                    self.min_perrep * self.replica_dict[target_service],  # 保持最小分配量
                    new_allocation[target_service] - action_value,
                )
                # 记录降低之前的配置信息
                # self.allocation_history.append(deepcopy(self.allocate_dict))
                # if len(self.allocation_history) > self.history_length:
                #     self.allocation_history.pop(0)

        elif action_type == "increase_batch":
            # 增加前所有可调服务的CPU配额
            candidates = self.scalable_service
            for service in candidates:
                new_allocation[service] = min(
                    self.default_cpu_config[service]["max_cpus"],
                    new_allocation[service] + action["value"],
                )

        elif action_type == "decrease_batch":
            candidates = [
                service for service in load if self.allocate_dict[service] -
                self.min_perrep * self.replica_dict[service] > 1e-4  # 检查 allocate 值是否为 self.min_perrep
            ]

            if len(candidates) > 0:
                for service in candidates:
                    new_allocation[service] = max(self.min_perrep * self.replica_dict[service],
                                                  new_allocation[service] - action["value"])

                # self.allocation_history.append(deepcopy(self.allocate_dict))
                # if len(self.allocation_history) > self.history_length:
                #     self.allocation_history.pop(0)

        elif action_type == "increase_percent":
            # 按百分比增加高负载服务
            target_service = max(load, key=lambda k: load[k])
            new_allocation[target_service] = min(
                self.default_cpu_config[target_service]["max_cpus"],
                new_allocation[target_service] * (1 + action["value"]),
            )

        elif action_type == "decrease_percent":
            # 按百分比减少低负载服务
            # 过滤掉 allocate 值为 self.min_perrep 的服务
            candidates = [
                service for service in load if self.allocate_dict[service] -
                self.min_perrep * self.replica_dict[service] > 1e-4  # 检查 allocate 值是否为 self.min_perrep
            ]
            if len(candidates) > 0:
                target_service = min(candidates, key=lambda k: load[k])
                new_allocation[target_service] = max(self.min_perrep * self.replica_dict[target_service],
                                                     new_allocation[target_service] * (1 - action["value"]))

                # self.allocation_history.append(deepcopy(self.allocate_dict))
                # if len(self.allocation_history) > self.history_length:
                #     self.allocation_history.pop(0)

        elif action_type == "reset":
            # 重置到初始配置
            new_allocation = deepcopy(self.initial_allocation)

        elif action_type == "hold":
            # 保持当前状态
            pass

        elif action_type == "recover":
            # 恢复到self.history_length之前有效分配
            if len(self.allocation_history) == self.history_length:
                new_allocation = deepcopy(self.allocation_history[0])

        key_service = self.config["key_service"]

        # 验证分配有效性
        for service in new_allocation:
            if new_allocation[service] < self.min_perrep * self.replica_dict[service]:  # 资源分配下限保护
                new_allocation[service] = self.min_perrep * self.replica_dict[service]

        # set_cpu_limit(new_allocation, self.replica_dict)

        # 更新状态记录
        self.last_allocate = deepcopy(self.allocate_dict)
        self.allocate_dict = deepcopy(new_allocation)

        return new_allocation
