import argparse
from asyncio import subprocess
from copy import deepcopy
import json
import os
import socket
import sys
import time
import asyncio
from typing import Dict, Tuple
import paramiko
import psutil

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(f"{PROJECT_ROOT}/deploy")

from monitor.data_collector_hotel import *
from mylocust.util.get_latency_data import get_latest_latency
from deploy.util.ssh import *
from communication.sync import distribute_project
from communication.MAB_hotel import UCB_Bandit

parser = argparse.ArgumentParser()
parser.add_argument("--exp_time", type=int, default=300, help="experiment time")
parser.add_argument("--username", type=str, default="tomly", help="username for SSH connection")
parser.add_argument("--save", action="store_true", help="whether to save data")

args = parser.parse_args()

exp_time = args.exp_time
username = args.username
save = args.save

gathered_list = []  # 用于存储每次循环处理后的 gathered 数据
replicas = []
service_replicas = {}
latency_list = []
cpu_config_list = []
services = []
locust_process = []

with open(f"{PROJECT_ROOT}/deploy/config/hotelreservation.json", 'r') as f:
    config = json.load(f)
    services = config["service_list"]


class SlaveConnection:

    def __init__(self, slave_host, slave_port):
        self.slave_host = slave_host
        self.slave_port = slave_port
        self.socket = None

    async def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.slave_host, self.slave_port))
        self.socket.setsockopt(socket.SOL_SOCKET, socket.SO_KEEPALIVE, 1)  # 开启TCP保活
        print(f"Connected to slave at {self.slave_host}:{self.slave_port}")

    def send_command_sync(self, command) -> str:
        if self.socket:
            # 添加结束标记
            command = f"{command}\r\n\r\n"
            self.socket.sendall(command.encode())
            data = ""
            while True:
                chunk = self.socket.recv(1024)
                # 连接关闭时退出
                if not chunk:
                    print("connection closed")
                    break
                data += chunk.decode()
                # 检测服务端的结束符
                if "\r\n\r\n" in data:
                    # 去除结束符并解码
                    data = data.split("\r\n\r\n")[0]
                    break
            return data

    def close(self):
        if self.socket:
            self.socket.close()
            print(f"Connection to {self.slave_host}:{self.slave_port} closed.")


async def start_experiment(connections: Dict[Tuple[str, int], SlaveConnection], users: int, load_type: str, min_core):
    global exp_time, gathered_list, replicas, service_replicas, cpu_config_list, locust_process

    tasks = []

    pids = []

    # 启动master节点
    master_cmd = [
        "locust",
        "-f",
        f"{PROJECT_ROOT}/mylocust/src/hotelreservation_{load_type}.py",
        "--host",
        "http://127.0.0.1:5000",
        "--master",
        "--headless",
        "--users",
        f"{users}",
        "-r",
        "50",  # 启动速率参数
        "-t",
        f"{3 * exp_time}s",
        "--csv",
        f"{PROJECT_ROOT}/mylocust/locust_log",
        "--expect-workers=8",
        "--master-bind-host=0.0.0.0"
    ]

    print(f"Master command: {' '.join(master_cmd)}")

    # 启动worker节点的基本命令
    worker_cmd_base = [
        "locust",
        "-f",
        f"{PROJECT_ROOT}/mylocust/src/hotelreservation_{load_type}.py",
        "--host",
        "http://127.0.0.1:5000",
        "--worker",
        "--master-host=127.0.0.1"  # 假设master在本地运行
    ]
    while True:
        try:
            # 启动master进程
            master_process = await asyncio.create_subprocess_exec(*master_cmd,
                                                                  stdout=asyncio.subprocess.DEVNULL,
                                                                  stderr=asyncio.subprocess.DEVNULL)
            print(f"Locust master started with PID: {master_process.pid}")
            pids.append(master_process.pid)
            # 启动8个worker进程
            worker_processes = []
            for i in range(8):
                worker_process = await asyncio.create_subprocess_exec(*worker_cmd_base,
                                                                      stdout=asyncio.subprocess.DEVNULL,
                                                                      stderr=asyncio.subprocess.DEVNULL)
                worker_processes.append(worker_process)
                print(f"Worker {i} started with PID: {worker_process.pid}")

            # 将进程对象保存在全局变量中以便后续管理
            locust_process = [master_process] + worker_processes
            pids.extend([master_process.pid] + [worker_process.pid for worker_process in worker_processes])
            # 检查pids列表中的进程是否存在
            check_pids = True
            for pid in pids:
                if not psutil.pid_exists(pid):
                    print(f"进程{pid}不存在")
                    check_pids = False
                    break
            if check_pids:
                break
            else:
                for pid in pids:
                    print(f"清理进程{pid}")
                    _, _ = execute_command(f"sudo kill {pid}")
                time.sleep(5)
                print(f"等待5秒后重新启动")
        except Exception as e:
            print(f"启动Locust失败: {str(e)}")
            # 清理已启动的进程
            if 'master_process' in locals():
                master_process.terminate()
            for wp in worker_processes:
                wp.terminate()
            raise

    mab = UCB_Bandit(min_core)
    init_allocate = deepcopy(mab.get_init_allocate())

    # 初始化配置
    print("执行初始化配置...")
    for service in init_allocate:
        init_allocate[service] /= mab.replica_dict[service]
    for connection in connections.values():
        connection.send_command_sync(f"update{json.dumps(init_allocate)}")

    # 等待负载稳定
    time.sleep(30)

    current_exp_time = 0

    try:
        while True:
            # 数据采集阶段
            collect_start = time.time()
            gathered = {"cpu": {}, "memory": {}, "io": {}, "network": {}}
            tasks.clear()
            while True:
                start_time = time.time()
                gathered = {"cpu": {}, "memory": {}, "io": {}, "network": {}}
                modify = False
                for connection in connections.values():
                    result = connection.send_command_sync("collect")
                    if result == "modify":
                        for connection in connections.values():
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

            print(f"同步采集耗时：{time.time() - collect_start}")

            # 副本初始化阶段
            if len(replicas) == 0:
                replicas = np.array([len(cpu_list) for cpu_list in gathered["cpu"].values()]).flatten()
                service_replicas = {key: len(cpu_list) for key, cpu_list in gathered["cpu"].items()}
            print(f"当前实验进度: {current_exp_time}/{exp_time}")

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
            stored_allocate = deepcopy(new_allocate)
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
                connection.send_command_sync(f"update{json.dumps(new_allocate)}")

            reward = mab.calculate_reward(latency)
            mab.update(arm_id, reward)
            update_time = time.time() - update_start
            print(f"配置更新耗时: {update_time:.3f}秒")

            # 数据存储阶段
            store_start = time.time()
            gathered = transform_data(gathered)
            gathered_list.append(gathered)
            latency_list.append(latency)
            # 保存cpu配置信息
            cpu_config_list.append([stored_allocate[service] for service in services])
            store_time = time.time() - store_start
            print(f"数据存储耗时: {store_time:.3f}秒")

            total_time = time.time() - start_time
            print(f"总时间: {total_time:.3f}秒")
            print("-" * 50)

            elapsed_time = time.time() - start_time
            if elapsed_time < 1:
                time.sleep(1 - elapsed_time)

            current_exp_time += 1
            if current_exp_time == exp_time:
                for pid in pids:
                    _, _ = execute_command(f"sudo kill {pid}")
                break

    finally:
        # 清理locust进程
        for pid in pids:
            _, _ = execute_command(f"sudo kill {pid}")


# 配置好slave，在slave上启动监听
def setup_slave():
    hosts = ["rm1", "rm2", "rm3", "rm4"]
    port = 12345
    username = "tomly"
    python_path = "/home/tomly/miniconda3/envs/DDRM/bin/python3"
    # 将两个命令组合在一起，第一个命令执行完后立即执行第二个命令
    # 此处假设第一个命令用于清理旧进程，第二个命令启动新的后台服务
    command = (
        f"sudo kill -9 $(sudo lsof -t -i :{port}) || true; "  # 清理旧进程
        "cd /home/tomly/DeepDynamicRM/communication && "  # 切换到工作目录
        f"nohup {python_path} slave_hotel.py --port {port} > /dev/null 2>&1 &"  # 后台启动服务
    )

    for host in hosts:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(hostname=host, username=username, timeout=10)
            # 一次性发送组合命令，不读取任何输出
            client.exec_command(command)
            print(f"{host} 服务已启动")
        except Exception as e:
            print(f"{host} 错误: {str(e)}")
        finally:
            client.close()


def kill_slave():
    import paramiko

    hosts = ["rm1", "rm2", "rm3", "rm4"]
    port = 12345
    username = "tomly"
    command = f"sudo kill -9 $(sudo lsof -t -i :{port}) || true"
    for host in hosts:
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(hostname=host, username=username, timeout=10)
            client.exec_command(command)
            print(f"{host} 服务已关闭")
        except Exception as e:
            print(f"{host} 错误: {str(e)}")
        finally:
            client.close()


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

    cpu_config_path = f"./data/cpu_config.npy"
    np.save(cpu_config_path, cpu_config_list)
    print(f"已保存cpu_config_path数据到: {cpu_config_path}")


async def main():
    global gathered_list, replicas, exp_time
    distribute_project(username=username)
    # 从配置文件中读取主机名和端口，然后创建连接
    # kill_slave()
    # time.sleep(10)
    # setup_slave()
    # time.sleep(10)
    comm_config = ""
    with open("./comm.json", "r") as f:
        comm_config = json.load(f)

    mab_config = ""
    with open("./mab_hotel.json", "r") as f:
        mab_config = json.load(f)

    hosts = comm_config["slaves"]
    port = comm_config["port"]
    slaves = [(host, port) for host in hosts]

    connections: Dict[Tuple[str, int], SlaveConnection] = {}

    # 建立与每个slave的连接
    for slave_host, slave_port in slaves:
        connection = SlaveConnection(slave_host, slave_port)
        await connection.connect()
        connections[(slave_host, slave_port)] = connection
        connection.send_command_sync("init")

    for users in [2500, 2800, 3100, 3400, 3700]:
        # for users in [3700]:
        #重置实验环境
        # if users == 1000 or users == 2500:
        #     command = ("cd ~/DeepDynamicRM/deploy && "
        #                "~/miniconda3/envs/DDRM/bin/python3 "
        #                "deploy_hotel.py")
        #     execute_command(command, stream_output=True)
        # time.sleep(10)

        for load_type in ["constant", "daynight", "bursty", "noisy"]:
            # for load_type in ["constant"]:
            await start_experiment(connections, users, load_type, mab_config[str(users)])
            if save:
                save_data(gathered_list, replicas)

    for connection in connections.values():
        connection.close()


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
