import asyncio
import json
import os
import sys
import time
from typing import Dict, Tuple
import joblib
import numpy as np
from collections import deque
import paramiko
from sklearn.preprocessing import StandardScaler
from monitor.data_collector import concat_data, process_data, transform_data
from mylocust.util.get_latency_data import get_latest_latency

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from communication.master import SlaveConnection


class Env:

    def __init__(self, connections, exp_time=300, window_size=10):

        self.config_path = f"{PROJECT_ROOT}/communication/comm.json"
        # 读取配置文件
        self.master, self.slaves, self.port = self._load_config(
            self.config_path)

        self.username = "tomly"
        self._setup_slaves()
        # 等待监听进程拉起
        time.sleep(5)

        # 新增连接存储结构
        self.connections: Dict[Tuple[str, int], SlaveConnection] = {}

        # 配置其余参数
        self.window_size = window_size

        # 保存缓存数据 当达到10个时间步的数据之后再开始执行决策
        self.buffer = []

        # 在预测器训练时，得到的归一化器
        self.scalers = []
        self._load_scalers()

    # 加载配置文件
    def _load_config(self, path):
        with open(path, "r") as f:
            config = json.load(f)
        return config["master"], config["slaves"], config["port"]

    # 配置slave节点，在所有slave节点上启动监听
    def _setup_slaves(self):
        for host in self.slaves:
            # 通过SSH连接到slave节点
            ssh = paramiko.SSHClient()
            ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
            try:
                ssh.connect(host, username=self.username)
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
        save_dir = f"{PROJECT_ROOT}/predictor/data"

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

    # 建立slave连接
    async def create_connections(self):
        """异步建立所有Slave连接并初始化"""
        tasks = []

        # 遍历配置中的slave节点
        for host in self.slaves:
            slave_addr = (host, self.port)

            # 创建连接对象
            conn = SlaveConnection(host, self.port)

            try:
                # 执行异步连接
                await conn.connect()
                self.connections[slave_addr] = conn

                # 创建初始化任务
                tasks.append(asyncio.create_task(conn.send_command("init")))

            except ConnectionError as e:
                print(f"连接 {host}:{self.port} 失败: {str(e)}")
                continue

        # 批量执行初始化命令
        if tasks:
            await asyncio.gather(*tasks)

        print(f"成功建立 {len(self.connections)} 个Slave连接")

    # 数据采集接口 获取一次数据
    async def gather_data(self):
        tasks = []
        replicas = []
        gathered = {"cpu": {}, "memory": {}, "io": {}, "network": {}}
        for connection in self.connections.values():
            tasks.append(
                asyncio.create_task(connection.send_command("collect")))

        results = await asyncio.gather(*tasks)
        for result in results:
            data_dict = json.loads(result)
            gathered["cpu"] = concat_data(gathered["cpu"], data_dict["cpu"])
            gathered["memory"] = concat_data(gathered["memory"],
                                             data_dict["memory"])
            gathered["io"] = concat_data(gathered["io"], data_dict["io"])
            gathered["network"] = concat_data(gathered["network"],
                                              data_dict["network"])
        if replicas == []:
            replicas = np.array([
                len(cpu_list) for cpu_list in gathered["cpu"].values()
            ]).flatten()
        for k, v in gathered["cpu"].items():
            gathered["cpu"][k] = [item / 1e6 for item in v]
        latency = get_latest_latency()
        gathered["cpu"] = process_data(gathered["cpu"])
        gathered["memory"] = process_data(gathered["memory"])
        gathered["io"] = process_data(gathered["io"])
        gathered["network"] = process_data(gathered["network"])
        # 转化为(service_num, 6, 4)
        gathered = transform_data(gathered)

    async def warmup(self):
        """异步预热10秒"""
        print("Starting async warmup...")
        for _ in range(10):
            gathered = await self._async_gather()
            processed = self._process_data(gathered)
            self.state_buffer.append(processed)
            await asyncio.sleep(1)
        print(f"Warmup completed. Buffer size: {len(self.state_buffer)}")

    async def _async_gather(self):
        """异步数据采集核心逻辑"""
        gathered = {"cpu": {}, "memory": {}, "io": {}, "network": {}}

        # 并发采集所有节点数据
        tasks = []
        for conn in self.connections.values():
            tasks.append(asyncio.create_task(conn.send_command("collect")))

        results = await asyncio.gather(*tasks)

        # 合并数据
        for result in results:
            data = json.loads(result)
            for metric in ["cpu", "memory", "io", "network"]:
                gathered[metric] = concat_data(gathered[metric], data[metric])

        # CPU单位转换
        for k in gathered["cpu"]:
            gathered["cpu"][k] = [v / 1e6 for v in gathered["cpu"][k]]

        return gathered

    def _process_data(self, raw_gathered):
        """数据处理流水线"""
        processed = {}
        for metric in ["cpu", "memory", "io", "network"]:
            # 先转换再处理
            transformed = transform_data(raw_gathered[metric])
            processed[metric] = process_data(transformed)
        return processed

    def _get_state_window(self):
        """生成(10,28,25)状态窗口"""
        # 自动填充不足的时间步
        while len(self.state_buffer) < self.window_size:
            self.state_buffer.appendleft(self.state_buffer[-1] if self.
                                         state_buffer else self._empty_state())

        # 转换为numpy数组 (10,28,25)
        window = np.stack(
            [self._vectorize_state(s) for s in self.state_buffer])

        # 标准化处理
        return self._normalize_window(window)

    def _vectorize_state(self, processed_data):
        """将处理后的数据转换为(28,25)特征矩阵"""
        vector = np.zeros((28, 25))

        for service_id in range(28):
            # 原始特征 (假设每个metric处理后得到6个特征)
            features = []
            for metric in ["cpu", "memory", "io", "network"]:
                features.extend(processed_data[metric].get(
                    service_id, [0] * 6))

            # 添加副本数特征
            replicas = len(processed_data["cpu"].get(service_id, []))
            features.append(replicas)

            vector[service_id] = features[:25]  # 截断至25维

        return vector

    def _normalize_window(self, window):
        """窗口数据标准化 (10,28,25)"""
        normalized = window.copy()
        for service_id in range(28):
            # 仅标准化前24个特征
            service_data = window[:, service_id, :24]
            self.scalers[service_id].partial_fit(service_data)
            normalized[:,
                       service_id, :24] = self.scalers[service_id].transform(
                           service_data)
        return normalized

    async def step(self, action):
        """异步执行环境步"""
        # 1. 执行动作
        await self._execute_action(action)

        # 2. 采集新数据
        gathered = await self._async_gather()
        processed = self._process_data(gathered)

        # 3. 更新状态缓冲区
        self.state_buffer.append(processed)
        self.gathered_list.append(processed)

        # 4. 获取延迟并计算奖励
        latency = get_latest_latency()
        self.latency_list.append(latency)

        # 5. 生成状态窗口
        state = self._get_state_window()
        reward = self._calculate_reward(state, latency)

        # 6. 更新时间
        self.current_exp_time += 1
        done = self.current_exp_time >= self.exp_time

        return state, reward, done, {}

    async def _execute_action(self, action):
        """执行资源分配动作（与MAB集成）"""
        # 这里与您现有的MAB逻辑集成
        from mylocust.strategy.mab import MABController

        # 选择臂
        arm_id = MABController.select_arm(action)

        # 执行动作（假设异步接口）
        await MABController.async_execute_action(
            arm_id, self.state_buffer[-1]["cpu"] if self.state_buffer else {})

        # 记录副本数变化
        current_replicas = np.array([
            len(cpu_list)
            for cpu_list in self.state_buffer[-1]["cpu"].values()
        ])
        self.replicas_history.append(current_replicas)

    def _calculate_reward(self, state_window, latency):
        """综合奖励计算"""
        # 1. SLO违例惩罚（假设SLO为1.0秒）
        slo_violation = 1.0 if latency > 1.0 else 0.0
        slo_penalty = -10.0 * slo_violation

        # 2. 资源利用率奖励（取窗口平均）
        cpu_util = np.mean(state_window[:, :, 0])  # 假设第0列是CPU利用率
        util_reward = 5.0 * cpu_util

        # 3. 负载均衡奖励
        balance_penalty = -2.0 * np.std(state_window[-1, :, 24])  # 副本数标准差

        # 4. 调整成本惩罚
        if len(self.replicas_history) >= 2:
            delta = np.abs(self.replicas_history[-1] -
                           self.replicas_history[-2]).sum()
            adjust_penalty = -0.1 * delta
        else:
            adjust_penalty = 0.0

        return slo_penalty + util_reward + balance_penalty + adjust_penalty

    def _empty_state(self):
        """生成空状态占位"""
        return {
            "cpu": {
                i: []
                for i in range(28)
            },
            "memory": {
                i: []
                for i in range(28)
            },
            "io": {
                i: []
                for i in range(28)
            },
            "network": {
                i: []
                for i in range(28)
            }
        }

    async def reset(self):
        """重置环境"""
        self.current_exp_time = 0
        self.state_buffer.clear()
        self.gathered_list.clear()
        self.latency_list.clear()
        await self.warmup()
        return self._get_state_window()
