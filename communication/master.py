import argparse
from asyncio import subprocess
import json
import os
import socket
import sys
import time
import asyncio
from typing import Dict, Tuple
import paramiko
from sync import distribute_project
from MAB import UCB_Bandit

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from monitor.data_collector import *
from mylocust.util.get_latency_data import get_latest_latency
from deploy.util.ssh import *

parser = argparse.ArgumentParser()
parser.add_argument("--exp_time",
                    type=int,
                    default=500,
                    help="experiment time")
parser.add_argument("--username",
                    type=str,
                    default="tomly",
                    help="username for SSH connection")
parser.add_argument("--save", action="store_true", help="whether to save data")

args = parser.parse_args()

exp_time = args.exp_time
username = args.username
save = args.save

gathered_list = []  # 用于存储每次循环处理后的 gathered 数据
replicas = []
service_replicas = {}
latency_list = []


class SlaveConnection:

    def __init__(self, slave_host, slave_port):
        self.slave_host = slave_host
        self.slave_port = slave_port
        self.socket = None

    async def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.slave_host, self.slave_port))
        print(f"Connected to slave at {self.slave_host}:{self.slave_port}")

    def send_command_sync(self, command) -> str:
        if self.socket:
            # 添加结束标记
            command = f"{command}\r\n\r\n"
            self.socket.sendall(command.encode())
            data = b""
            while True:
                chunk = self.socket.recv(20480)
                # 连接关闭时退出
                data += chunk
                # 检测服务端的结束符
                if data.endswith(b"\r\n\r\n"):
                    # 去除结束符并解码
                    data = data[:-4]
                    break
            # print(f'Received from {self.slave_host}:{self.slave_port}:',
            #       data.decode())
            return data.decode()

    async def send_command(self, command) -> str:
        if self.socket:
            # 添加结束标记
            command = f"{command}\r\n\r\n"
            self.socket.sendall(command.encode())
            data = b""
            while True:
                chunk = self.socket.recv(20480)
                # 连接关闭时退出
                data += chunk
                # 检测服务端的结束符
                if data.endswith(b"\r\n\r\n"):
                    # 去除结束符并解码
                    data = data[:-4]
                    break
            # print(f'Received from {self.slave_host}:{self.slave_port}:',
            #       data.decode())
            return data.decode()

    def close(self):
        if self.socket:
            self.socket.close()
            print(f"Connection to {self.slave_host}:{self.slave_port} closed.")


async def start_experiment(slaves):
    global exp_time, gathered_list, replicas, service_replicas
    connections: Dict[Tuple[str, int], SlaveConnection] = {}
    tasks = []

    # 建立与每个slave的连接
    for slave_host, slave_port in slaves:
        connection = SlaveConnection(slave_host, slave_port)
        await connection.connect()
        connections[(slave_host, slave_port)] = connection
        tasks.append(asyncio.create_task(connection.send_command("init")))

    # 等待所有 slave 初始化完毕
    await asyncio.gather(*tasks)
    tasks.clear()

    # 启动locust负载，同时使用MAB探索
    locust_cmd = [
        "locust",  # 命令名称
        "-f",  # 参数：指定locust文件路径
        f"{PROJECT_ROOT}/mylocust/src/socialnetwork.py",  # 你的Locust文件路径
        "--host",  # 参数：目标主机
        "http://127.0.0.1:8080",
        "--users",  # 用户数参数
        "50",
        "--csv",  # 输出CSV文件
        f"{PROJECT_ROOT}/mylocust/locust_log",
        "--headless",  # 无头模式
        "-t",  # 测试时长
        f"{10 * exp_time}s",  # 100秒运行时间
    ]

    print(f"locust command:{locust_cmd}")

    try:
        # 创建子进程，不等待立即返回
        process = await asyncio.create_subprocess_exec(
            *locust_cmd,
            stdout=asyncio.subprocess.DEVNULL,  # 丢弃输出
            stderr=asyncio.subprocess.DEVNULL)

        print(f"Locust已启动，PID: {process.pid}")

    except Exception as e:
        # 捕获启动错误（如命令不存在、路径错误等）
        print(f"启动Locust失败: {str(e)}")
        raise

    mab = UCB_Bandit()

    # 等待负载稳定
    time.sleep(5)

    current_exp_time = 0
    start_time = time.time()
    try:
        while True:
            # 数据采集阶段
            collect_start = time.time()
            gathered = {"cpu": {}, "memory": {}, "io": {}, "network": {}}
            tasks.clear()
            for connection in connections.values():
                result = connection.send_command_sync("collect")
                data_dict = json.loads(result)
                gathered["cpu"] = concat_data(gathered["cpu"],
                                              data_dict["cpu"])
                gathered["memory"] = concat_data(gathered["memory"],
                                                 data_dict["memory"])
                gathered["io"] = concat_data(gathered["io"], data_dict["io"])
                gathered["network"] = concat_data(gathered["network"],
                                                  data_dict["network"])
            print(f"同步采集耗时：{time.time() - collect_start}")

            # for connection in connections.values():
            #     tasks.append(
            #         asyncio.create_task(connection.send_command("collect")))
            # results = await asyncio.gather(*tasks)
            # for result in results:
            #     data_dict = json.loads(result)
            #     gathered["cpu"] = concat_data(gathered["cpu"],
            #                                   data_dict["cpu"])
            #     gathered["memory"] = concat_data(gathered["memory"],
            #                                      data_dict["memory"])
            #     gathered["io"] = concat_data(gathered["io"], data_dict["io"])
            #     gathered["network"] = concat_data(gathered["network"],
            #                                       data_dict["network"])
            # collect_time = time.time() - collect_start
            # print(f"数据采集耗时: {collect_time:.3f}秒")

            # 副本初始化阶段
            if len(replicas) == 0:
                replicas = np.array([
                    len(cpu_list) for cpu_list in gathered["cpu"].values()
                ]).flatten()
                service_replicas = {
                    key: len(cpu_list)
                    for key, cpu_list in gathered["cpu"].items()
                }

            print(f"当前实验进度: {current_exp_time}/{args.exp_time}")

            # 数据处理阶段
            process_start = time.time()
            for k, v in gathered["cpu"].items():
                gathered["cpu"][k] = [item / 1e6 for item in v]

            gathered["cpu"] = process_data(gathered["cpu"])
            gathered["memory"] = process_data(gathered["memory"])
            gathered["io"] = process_data(gathered["io"])
            gathered["network"] = process_data(gathered["network"])
            process_time = time.time() - process_start
            print(f"数据处理耗时: {process_time:.3f}秒")

            # MAB决策阶段
            mab_start = time.time()
            latency = get_latest_latency()
            print(f"当前延迟{latency}")
            arm_id = mab.select_arm(latency=latency)
            print(f"选择动作{arm_id}, {mab.actions[arm_id]}")
            new_allocate = mab.execute_action(arm_id, gathered["cpu"])
            print(f"新的分配方案：{new_allocate}")
            print(f"总CPU分配数量：{sum(new_allocate.values())}")
            mab_time = time.time() - mab_start
            print(f"MAB决策耗时: {mab_time:.3f}秒")

            # 配置更新阶段
            update_start = time.time()
            print(f"更新cpu配置....")
            for service in new_allocate:
                new_allocate[service] /= service_replicas[service]
            tasks.clear()
            for connection in connections.values():
                connection.send_command_sync(
                    f"update{json.dumps(new_allocate)}")

            reward = mab.calculate_reward(latency)
            mab.update(arm_id, reward)
            update_time = time.time() - update_start
            print(f"配置更新耗时: {update_time:.3f}秒")

            # 数据存储阶段
            store_start = time.time()
            gathered = transform_data(gathered)
            gathered_list.append(gathered)
            latency_list.append(latency)
            store_time = time.time() - store_start
            print(f"数据存储耗时: {store_time:.3f}秒")

            total_time = time.time() - start_time
            print(f"时: {total_time:.3f}秒")
            print("-" * 50)

            time.sleep(1)
            current_exp_time += 1
            if current_exp_time == exp_time:
                break
    finally:
        # 清理locust进程
        _, _ = execute_command(f"sudo kill {process.pid}")
        # 关闭所有连接
        for connection in connections.values():
            connection.close()


# 配置好slave，在slave上启动监听
def setup_slave():
    # 从配置文件中读取主机名和端口
    comm_config = ""
    with open("./comm.json", "r") as f:
        comm_config = json.load(f)
    hosts = comm_config["slaves"]
    port = comm_config["port"]

    # 在每个slave节点上启动监听服务
    for host in hosts:
        # 通过SSH连接到slave节点

        # 清理旧的进程
        command = f"sudo kill -9 $(sudo lsof -t -i :{port})"
        execute_command_via_system_ssh(host, username, command)

        command = ("cd ~/DeepDynamicRM/communication && "
                   "nohup ~/miniconda3/envs/DDRM/bin/python3 "
                   f"slave.py --port {port} > /dev/null 2>&1 &")

        execute_command_via_system_ssh(host,
                                       username,
                                       command,
                                       async_exec=True)

        print(f"在 {host} 上启动监听服务,端口:{port}")


def save_data(gathered_list, replicas):
    """保存实验数据到本地文件"""
    # 创建数据目录(如果不存在)
    if not os.path.exists("./data"):
        os.makedirs("./data")

    # 保存gathered数据
    gathered_path = f"./data/gathered.npy"
    np.save(gathered_path, gathered_list)
    print(f"已保存gathered数据到: {gathered_path}")

    # 保存replicas数据
    replicas_path = f"./data/replicas.npy"
    np.save(replicas_path, replicas)
    print(f"已保存replicas数据到: {replicas_path}")

    # 保存延迟latency数据
    latency_path = f"./data/latency.npy"
    np.save(latency_path, latency_list)
    print(f"已保存latency数据到: {latency_path}")


class Executor:
    pass


async def main():
    global gathered_list, replicas
    # 从配置文件中读取主机名和端口，然后创建连接
    comm_config = ""
    with open("./comm.json", "r") as f:
        comm_config = json.load(f)
    hosts = comm_config["slaves"]
    port = comm_config["port"]
    slaves = [(host, port) for host in hosts]

    distribute_project(username=username)
    setup_slave()
    # 等待slave监听进程启动完成
    time.sleep(5)
    await start_experiment(slaves)
    if save:
        save_data(gathered_list, replicas)


def test_setup_slave():
    # setup_slave()
    print("🔧 开始测试slave节点配置...")

    # 从配置文件中读取主机名和端口
    with open("./comm.json", "r") as f:
        comm_config = json.load(f)
    hosts = comm_config["slaves"]
    port = comm_config["port"]

    # 测试每个slave节点的连通性
    for host in hosts:
        try:
            # 创建socket连接测试
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(5)  # 设置超时时间为5秒
                result = s.connect_ex((host, port))

                if result == 0:
                    print(f"✅ {host}:{port} 连接成功")
                else:
                    print(f"❌ {host}:{port} 连接失败")

        except Exception as e:
            print(f"⚠️ 测试 {host} 时发生错误: {str(e)}")

    print("🔍 slave节点配置测试完成")


if __name__ == "__main__":
    asyncio.run(main())
