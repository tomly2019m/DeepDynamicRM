import asyncio
from copy import deepcopy
import json
import os
from pathlib import Path
import pickle
import signal
import sys
import time
from typing import Dict, Tuple
import joblib
import numpy as np
from collections import deque
import paramiko
from sklearn.preprocessing import StandardScaler
import torch
import torch.nn.functional as F

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from communication.master import SlaveConnection
from predictor.slo_predictor import OnlineScaler
from monitor.data_collector import *
from monitor.shell import execute_command
from predictor.slo_predictor import DynamicSLOPredictor
from monitor.data_collector import concat_data, process_data, transform_data
from mylocust.util.get_latency_data import get_latest_latency


class Env:

    def __init__(self, window_size=30):

        self.config_path = f"{PROJECT_ROOT}/communication/comm.json"
        # 读取配置文件
        self.master, self.slaves, self.port = self._load_config(self.config_path)

        self.username = "tomly"
        # self._setup_slaves()
        # 等待监听进程拉起
        time.sleep(5)

        self.episode_count = 0

        # 新增连接存储结构
        self.connections: Dict[Tuple[str, int], SlaveConnection] = {}

        # 配置其余参数
        self.window_size = window_size

        # 保存缓存数据 当达到30个时间步的数据之后再开始执行决策
        self.buffer = deque(maxlen=self.window_size)
        self.latency_buffer = deque(maxlen=self.window_size)
        self.config_buffer = deque(maxlen=self.window_size)

        # 在预测器训练时，得到的归一化器
        self.scalers = []
        self._load_scalers()

        # 在线标准化器 用于处理动态的服务结构
        self.online_scaler = OnlineScaler()

        # 可选动作列表
        self.actions = []
        self._load_actions()
        # 最低分配数量
        self.min_allocate = 60
        self.constraint = None
        # 保存资源配置历史
        self.allocation_history = []
        self.history_length = 10

        # 保存从部署配置文件中读取默认服务配置
        self.default_cpu_config: dict[str, dict[str, float]] = {}
        self.services = []

        self.allocate_dict: dict[str, float] = {}  # 每个服务总的cpu分配
        self.replica_dict: dict[str, int] = {}
        self.replica_ndarray = []

        self.initial_allocation: dict[str, float] = {}  # 初始分配方案，用于reset
        self.max_cpu = 0
        self._load_service_default_config()

        # 当前的cpu状态，从gathered中获取
        self.cpu_state = {}

        # locust进程的pid
        self.locust_pid = None

        # 预测器
        self.predictor = DynamicSLOPredictor(service_mode="hier_attention")
        self._load_predictor()

        # 奖励参数
        self.w1, self.w2, self.w3, self.w4 = 0, 0, 0, 0
        self.pv = 0  # pv阈值
        self.threshold = 0  # SLO阈值
        self.punish_factor = 4.5
        self._load_reward_config()

        self.steps = 0  # 统计step

        self.done_steps = 20 * 3600

        self.every_episode_steps = 1000

    # 加载集群配置文件
    def _load_config(self, path):
        with open(path, "r") as f:
            config = json.load(f)
            hosts = config["slaves"]
            port = config["port"]
            slaves = [(host, port) for host in hosts]
            print(config["master"], slaves, port)
            return config["master"], slaves, port

    def _load_service_default_config(self):
        """
        从配置文件中加载并生成 allocate_dict 和 replica_dict，初始化初始分配字典，初始化默认cpu配置
        """
        config_file_path = os.path.join(PROJECT_ROOT, "deploy", "config", "socialnetwork.json")

        try:
            with open(config_file_path, "r") as f:
                config = json.load(f)

            # 提取可扩展服务列表
            self.scalable_service = config.get("scalable_service", [])

            # 服务列表
            self.services = config.get("service_list", [])

            self.max_cpu = config.get("max_cpu", 256)

            # 提取完整服务配置
            self.default_cpu_config = config.get("service", {})

            # 自动生成 allocate_dict 和 replica_dict
            for service_name, service_conf in self.default_cpu_config.items():
                # CPU分配：直接使用配置中的 cpus 字段
                self.allocate_dict[service_name] = service_conf.get("cpus", 0.0)
                self.initial_allocation = deepcopy(self.allocate_dict)

                # 副本数：使用配置中的 replica 字段
                self.replica_dict[service_name] = service_conf.get("replica", 1)

            replica_list = [self.replica_dict.get(service, 1) for service in services]
            self.replica_ndarray = np.array(replica_list)

            print(f"已自动生成初始分配：{len(self.allocate_dict)} 个服务的CPU配置")

        except FileNotFoundError:
            print(f"[错误] 部署配置文件 {config_file_path} 不存在！")
            raise  # 配置文件不存在时直接抛出异常
        except json.JSONDecodeError as e:
            print(f"[错误] 配置文件解析失败: {str(e)}")
            raise

    def _close_all_slaves(self):
        for host in self.slaves:
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            ssh.connect(host, username=self.username)
            command = f"sudo kill -9 $(sudo lsof -t -i :{self.port})"
            stdin, stdout, stderr = ssh.exec_command(command)
            ssh.close()

    # 配置slave节点，在所有slave节点上启动监听
    def _setup_slaves(self):
        for host in self.slaves:
            # 通过SSH连接到slave节点
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                ssh.connect(host[0], username=self.username)
                # 先切换到目标目录在slave节点上启动监听程序

                # 清理旧的进程
                command = f"sudo kill -9 $(sudo lsof -t -i :{self.port})"
                stdin, stdout, stderr = ssh.exec_command(command)

                command = ("cd ~/DeepDynamicRM/communication && "
                           "nohup ~/miniconda3/envs/DDRM/bin/python3 "
                           f"slave.py --port {self.port} > /dev/null 2>&1 &")

                stdin, stdout, stderr = ssh.exec_command(command)

                print(f"在 {host} 上启动监听服务,端口:{self.port}")

            except Exception as e:
                print(f"连接到 {host} 失败: {str(e)}")
            finally:
                ssh.close()

    def _load_scalers(self):
        """加载预测器训练好的标准化器"""
        save_dir = f"{PROJECT_ROOT}/predictor/model"

        try:
            # 加载服务级标准化器
            service_scaler_path = f"{save_dir}/service_scalers.pkl"
            with open(service_scaler_path, 'rb') as f:
                service_scalers = joblib.load(f)

            # 加载延迟标准化器
            latency_scaler_path = f"{save_dir}/latency_scaler.pkl"
            with open(latency_scaler_path, 'rb') as f:
                latency_scaler = joblib.load(f)

            # 保存到实例变量
            self.scalers = {
                'service': service_scalers,  # List[StandardScaler] 每个服务一个
                'latency': latency_scaler  # 单个StandardScaler
            }
            print("成功加载标准化器")

        except FileNotFoundError as e:
            raise RuntimeError(f"标准化器文件 {e.filename} 不存在\n"
                               "请先执行预测器训练并保存标准化器")
        except Exception as e:
            raise RuntimeError(f"加载标准化器失败: {str(e)}")

    def _load_actions(self):
        """加载并验证动作配置文件"""
        try:
            # 使用pathlib构建路径
            action_path = Path(PROJECT_ROOT) / "controller" / "actions.json"

            # 检查文件是否存在
            if not action_path.is_file():
                raise FileNotFoundError(f"动作配置文件不存在: {action_path}")

            # 明确指定utf-8编码
            with action_path.open('r', encoding='utf-8') as f:
                self.actions = json.load(f)

        except FileNotFoundError as e:
            print(f"关键错误: {str(e)}")
            raise
        except json.JSONDecodeError:
            print("JSON格式错误，请检查配置文件语法")
            raise
        except (ValueError, KeyError) as e:
            print(f"配置验证失败: {str(e)}")
            raise
        except Exception as e:
            print(f"未知错误: {str(e)}")
            raise

    def _load_reward_config(self):
        config_path = Path(PROJECT_ROOT) / "controller" / "reward.json"
        with open(config_path, 'r') as f:
            config = json.load(f)
            self.w1, self.w2, self.w3, self.w4 = config["w1"], config["w2"], config["w3"], config["w4"]
            self.pv = config["pv"]
            self.threshold = config["threshold"]

    # 建立slave连接
    async def create_connections(self):
        """异步建立所有Slave连接并初始化"""
        for slave_host, slave_port in self.slaves:
            connection = SlaveConnection(slave_host, slave_port)
            await connection.connect()
            self.connections[(slave_host, slave_port)] = connection
            print("初始化slave连接")
            connection.send_command_sync("init")

    def close_all_connections(self):
        """关闭所有连接"""
        for connection in self.connections.values():
            connection.close()

    def _load_predictor(self):
        """加载预测器"""
        model_path = Path(PROJECT_ROOT) / "predictor" / "model" / "best_model.pth"
        constraint_path = Path(PROJECT_ROOT) / "predictor" / "model" / "constraint.pkl"

        if os.path.exists(constraint_path):
            with open(constraint_path, 'rb') as f:
                self.constraint = pickle.load(f)
        else:
            self.constraint = None

        # 加载完整的检查点文件
        checkpoint = torch.load(model_path)

        # 检查是否包含model_state键
        if "model_state" in checkpoint:
            # 只加载模型状态部分
            self.predictor.load_state_dict(checkpoint["model_state"])
        else:
            # 尝试直接加载（兼容旧格式）
            try:
                self.predictor.load_state_dict(checkpoint)
            except Exception as e:
                print(f"模型加载失败: {str(e)}")
                print("请确保模型文件格式正确，或重新训练模型")
                raise

        self.predictor.eval()
        print("预测器加载成功")

    def gather_data(self):
        """数据采集接口 获取一个时间步的数据"""
        gathered = {"cpu": {}, "memory": {}, "io": {}, "network": {}}
        while True:
            modify = False
            for connection in self.connections.values():
                result = connection.send_command_sync("collect")
                if result == "modify":
                    for connection in self.connections.values():
                        # 确保容器状态稳定再flush
                        print("等待容器状态稳定")
                        time.sleep(5)
                        connection.send_command_sync("flush")
                    modify = True
                    break
                data_dict = json.loads(result)
                gathered["cpu"] = concat_data(gathered["cpu"], data_dict["cpu"])
                gathered["memory"] = concat_data(gathered["memory"], data_dict["memory"])
                gathered["io"] = concat_data(gathered["io"], data_dict["io"])
                gathered["network"] = concat_data(gathered["network"], data_dict["network"])
            if not modify:
                break

        for k, v in gathered["cpu"].items():
            gathered["cpu"][k] = [item / 1e6 for item in v]
        self.cpu_state = gathered["cpu"]
        latency = get_latest_latency()
        gathered["cpu"] = process_data(gathered["cpu"])
        gathered["memory"] = process_data(gathered["memory"])
        gathered["io"] = process_data(gathered["io"])
        gathered["network"] = process_data(gathered["network"])
        gathered = transform_data(gathered)  # 转化为(service_num, 6, 4)
        status = gathered.reshape(gathered.shape[0], -1)  # -1 表示自动计算维度
        if self.constraint is not None:
            self.min_allocate = self.constraint(latency[0])
        return status, latency

    def convert_to_ndarray(self, allocate):
        """
        将输入的allocate字典按照self.services列表的顺序转换为numpy数组
        
        Args:
            allocate: 包含服务资源分配的字典
            
        Returns:
            numpy.ndarray: 按services顺序排列的资源分配数组
        """
        # 初始化一个与services长度相同的数组
        ndarray = np.zeros(len(self.services))

        # 按照services的顺序填充数组
        for i, service in enumerate(self.services):
            ndarray[i] = allocate[service]

        return ndarray

    def warmup(self):
        """预热，倒计时40秒填充buffer"""
        # 多填充10秒数据
        countdown = self.window_size + 10

        while countdown > 0:
            data, latency = self.gather_data()
            self.buffer.append(data)
            self.latency_buffer.append(latency)

            # 配置列表预填充最初的配置
            self.config_buffer.append(self.convert_to_ndarray(self.initial_allocation))

            time.sleep(1)
            countdown -= 1
            print(f"剩余预热时间: {countdown}秒")

    def get_state_and_latency(self):
        """返回形状为 (30,28,26) 和 (30,6) 的归一化数据"""
        service_data = np.array(self.buffer)  # 形状 (30,28,24)
        latency_data = np.array(self.latency_buffer)  # 形状 (30,6)
        config_data = np.array(self.config_buffer)  # 形状 (30,28)
        replica_data = np.array(self.replica_ndarray)  # 形状 (28, )

        # ========== 特征拼接 ==========
        # 1. 将 config_data 扩展维度后与 service_data 拼接
        config_expanded = config_data[..., np.newaxis]  # 形状 (30, 28, 1)
        service_config = np.concatenate([service_data, config_expanded], axis=2)  # 形状 (30, 28, 25)

        # 2. 将 replica_data 扩展后与上述结果拼接
        replica_expanded = np.tile(replica_data[np.newaxis, :, np.newaxis],
                                   (service_config.shape[0], 1, 1))  # 形状 (30, 28, 1)
        combined_data = np.concatenate([service_config, replica_expanded], axis=2)  # 最终形状 (30, 28, 26)

        # ========== 服务数据处理 ==========
        processed_serv = np.zeros_like(combined_data)
        for s in range(28):  # 遍历每个服务
            # 提取特征 (10,25)
            features = combined_data[:, s, :25]
            scaled = self.scalers["service"][s].transform(features)  # 输入 (10,25)
            processed_serv[:, s, :25] = scaled
        # 保留第26个特征（replica_data）不归一化
        processed_serv[:, s, 25] = combined_data[:, s, 25]

        # ========== 延迟数据处理 ==========
        processed_lat = self.scalers["latency"].transform(latency_data)  # 输入 (10,6)

        return processed_serv, processed_lat  # 形状 (30, 28, 26), (30, 6)

    def step(self, action):
        """执行一个环境步骤
        
        Args:
            action: 要执行的动作索引
            
        Returns:
            stacked_state: 当前状态 (30,28,26)
            reward: 奖励值
            done: 是否结束
        """
        # 1. 执行动作 会更新self.allocate_dict
        print(f"执行动作: {action}, {self.actions[action]}")
        self._execute_action(action)

        # 2. 采集新数据
        gathered, latency = self.gather_data()  #返回(28, 24) 和(6,)
        self.buffer.append(gathered)
        self.latency_buffer.append(latency)
        self.config_buffer.append(self.convert_to_ndarray(deepcopy(self.allocate_dict)))

        stacked_state, stacked_latency = self.get_state_and_latency()
        result = (deepcopy(stacked_state), deepcopy(stacked_latency))
        # 添加批次维度
        state_batch = np.expand_dims(stacked_state, axis=0)  # shape (1,30,28,26)
        latency_batch = np.expand_dims(stacked_latency, axis=0)  # shape (1,30,6)

        # 得到预测概率
        with torch.no_grad():
            # 获取预测结果
            predictions = self.predictor(torch.FloatTensor(state_batch), torch.FloatTensor(latency_batch))
            # 应用softmax获取概率分布
            probs = F.softmax(predictions, dim=1)
            # 获取第五类的概率作为违例概率
            pv = probs[0, 5].item()  # 索引5对应第六类
        print(f"延迟概率分布：{probs}")
        print(f"违例概率: {pv * 100:.2f}%")
        print(f"当前延迟: {latency}")
        p99_latency = latency[-2]
        reward = self._calculate_reward(pv, p99_latency)

        # 两阶段反馈
        self.steps += 1
        done = False
        if self.steps < self.done_steps:
            if self.steps % self.every_episode_steps == 0:
                done = True
        return result[0], result[1], reward, done

    def _execute_action(self, action):
        """执行资源分配动作"""
        action_type = self.actions[action]["type"]
        action_value = self.actions[action]["value"]

        self.min_perrep = self.min_allocate / sum(self.replica_dict.values())

        # 保存当前状态到历史记录（用于recover）
        self.allocation_history.append(deepcopy(self.allocate_dict))
        if len(self.allocation_history) > 10:  # 保留最近10次记录
            self.allocation_history.pop(0)

        load = {}
        for service in self.allocate_dict:
            mean_util = self.cpu_state[service][2]  # 获取平均利用率
            load[service] = (mean_util * 100) / self.allocate_dict[service]

        new_allocation = deepcopy(self.allocate_dict)

        if action_type == "increase":
            # 找到负载最高的服务增加资源
            target_service = max(load, key=lambda k: load[k])
            new_cpu = self.allocate_dict[target_service] + action_value
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

        elif action_type == "increase_batch":
            # 增加前所有可调服务的CPU配额
            candidates = self.scalable_service
            for service in candidates:
                new_allocation[service] = min(
                    self.default_cpu_config[service]["max_cpus"],
                    new_allocation[service] + action_value,
                )

        elif action_type == "decrease_batch":
            candidates = [
                service for service in load if self.allocate_dict[service] -
                self.min_perrep * self.replica_dict[service] > 1e-4  # 检查 allocate 值是否为 self.min_perrep
            ]

            if len(candidates) > 0:
                for service in candidates:
                    new_allocation[service] = max(self.min_perrep * self.replica_dict[service],
                                                  new_allocation[service] - action_value)

        elif action_type == "increase_percent":
            # 按百分比增加高负载服务
            target_service = max(load, key=lambda k: load[k])
            new_allocation[target_service] = min(
                self.default_cpu_config[target_service]["max_cpus"],
                new_allocation[target_service] * (1 + action_value),
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
                                                     new_allocation[target_service] * (1 - action_value))

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

        for service in new_allocation:
            if new_allocation[service] < self.min_perrep * self.replica_dict[service]:  # 资源分配下限保护
                new_allocation[service] = self.min_perrep * self.replica_dict[service]

        # 如果总资源上限，则恢复到初始配置
        if sum(new_allocation.values()) > self.max_cpu:
            new_allocation = deepcopy(self.initial_allocation)

        # 更新状态记录
        self.last_allocate = deepcopy(self.allocate_dict)
        self.allocate_dict = new_allocation

    def _calculate_reward(self, pv, latency):
        """综合奖励计算
        Args:
            pv: 未来5个时间步的平均违例概率（0~1） 由预测器给出
            latency: 当前延迟值
        """
        # 获取参数配置
        w1 = self.w1  # 资源节约奖励系数
        w2 = self.w2  # 未来风险惩罚系数
        w3 = self.w3  # 违例惩罚强度系数
        w4 = self.w4  # 持续风险抑制系数
        threshold = self.threshold  # SLO延迟阈值
        allocated = sum(self.allocate_dict.values())  # 当前已分配CPU

        def get_allocated_reward(allocated):
            if allocated < 0:
                return 0  # 负数CPU数量无效
            elif 0 <= allocated <= self.max_cpu / 2:
                # 使用平方根函数实现增速递减
                return 5 * math.sqrt(allocated / (self.max_cpu / 2))
            else:
                return 5

        def discount_factor(latency):
            if latency <= threshold * 0.6:
                return 1
            else:
                return (1 - latency / (threshold * 0.7))

        def clip(x):
            if x >= 10:
                return 10
            else:
                return x

        if latency < threshold * 0.7:
            # --------------------------
            # 状态1：SLO未违例（资源优化模式）
            # --------------------------
            # 资源节约奖励（标准化到0~1）
            cpu_util = allocated / self.max_cpu
            resource_reward = w1 * (1 - cpu_util)

            # 未来风险惩罚（非线性响应）
            if pv <= self.pv:
                # 低风险区平方惩罚
                risk_penalty = w2 * (pv**2)
            else:
                # 高风险区线性放大
                risk_penalty = w2 * (2 * pv - 0.5)

            return (resource_reward - risk_penalty) * discount_factor(latency)

        else:
            # --------------------------
            # 状态2：SLO已违例（紧急恢复模式）
            # --------------------------
            # 延迟超阈值惩罚（1.5次方梯度）
            violation_degree = (latency - threshold * 0.7) / (threshold * 0.7)
            delay_penalty = w3 * (violation_degree**1.5)

            # 持续未来风险惩罚
            ongoing_penalty = w4 * pv

            # 总惩罚取负 鼓励分配更多资源
            return -clip((delay_penalty + ongoing_penalty) * 4.5) + get_allocated_reward(allocated)

    async def start_locust(self, user_count):
        locust_cmd = [
            "locust",  # 命令名称
            "-f",  # 参数：指定locust文件路径
            f"{PROJECT_ROOT}/mylocust/src/socialnetwork_mixed.py",
            "--host",  # 参数：目标主机
            "http://127.0.0.1:8080",
            "--users",  # 用户数参数
            f"{user_count}",
            "--csv",  # 输出CSV文件
            f"{PROJECT_ROOT}/mylocust/locust_log",
            "--headless",  # 无头模式
            "-t",  # 测试时长
            f"{10 * 2000}s",
            "-r",
            "10"  # 每秒启动10个用户
        ]

        print(f"locust command:{locust_cmd}")

        try:
            # 创建子进程，不等待立即返回
            process = await asyncio.create_subprocess_exec(
                *locust_cmd,
                stdout=asyncio.subprocess.DEVNULL,  # 丢弃输出
                stderr=asyncio.subprocess.DEVNULL)

            print(f"Locust已启动，PID: {process.pid}")

            self.locust_pid = process.pid

        except Exception as e:
            # 捕获启动错误（如命令不存在、路径错误等）
            print(f"启动Locust失败: {str(e)}")
            raise

    def stop_locust(self):
        if self.locust_pid:
            _, _ = execute_command(f"sudo kill {self.locust_pid}")
            print("Locust已停止")
        else:
            print("Locust未启动")

    def reset_deploy(self):
        time.sleep(10)
        #重置实验环境
        command = ("cd ~/DeepDynamicRM/deploy && "
                   "~/miniconda3/envs/DDRM/bin/python3 "
                   "deploy_benchmark.py")
        execute_command(command, stream_output=True)

    async def reset(self):
        """重置环境"""
        self.episode_count += 1
        if self.episode_count % 10 == 0:
            print("停止locust")
            self.stop_locust()
            self.reset_deploy()

        user_count_list = [50, 100, 150, 200, 250, 300, 350, 400, 450]
        print("重置环境")
        print("停止locust")
        self.stop_locust()
        self.locust_pid = None
        print("清空缓存")
        self.buffer.clear()
        self.latency_buffer.clear()
        self.config_buffer.clear()
        self.allocation_history.clear()
        # 每个回合 重置cpu分配方案 防止上一个回合的cpu分配方案影响下一个回合
        self._load_service_default_config()
        # 重新启动locust
        print("重新启动locust")
        try:
            # 随机选择一个用户数
            user_count = np.random.choice(user_count_list)
            await self.start_locust(user_count)
            print(f"随机选择用户数: {user_count}")
            print("等待30秒")
            # TODO 把预热时间改为30秒
            time.sleep(30)
            print("预热")
            self.warmup()
            print("返回状态")
            return self.get_state_and_latency()
        except Exception as e:
            print(f"重置环境失败: {str(e)}")
