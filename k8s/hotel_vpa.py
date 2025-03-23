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
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
sys.path.append(f"{PROJECT_ROOT}/deploy")

# 修改导入，使用 k8s_vpa 替代 MAB_hotel
from k8s.k8s_vpa import MultiServiceVPAManager
from monitor.data_collector_hotel import *
from mylocust.util.get_latency_data import get_latest_latency
from deploy.util.ssh import *
from communication.sync import distribute_project

parser = argparse.ArgumentParser()
parser.add_argument("--exp_time", type=int, default=500, help="experiment time")
parser.add_argument("--username", type=str, default="tomly", help="username for SSH connection")

args = parser.parse_args()

exp_time = args.exp_time
username = args.username

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


async def start_experiment(connections: Dict[Tuple[str, int], SlaveConnection], users: int, load_type: str):
    global exp_time, gathered_list, replicas, service_replicas, cpu_config_list, locust_process

    # 创建实验数据保存目录
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    exp_data_dir = f"{PROJECT_ROOT}/k8s/exp_data/hotel/{time_str}_users{users}_{load_type}"
    os.makedirs(exp_data_dir, exist_ok=True)

    # 创建单个数据CSV文件并写入表头
    exp_data_csv_path = f"{exp_data_dir}/experiment_data.csv"
    with open(exp_data_csv_path, "w") as f:
        # 写入表头：时间戳、延迟指标、各服务CPU分配、总CPU
        header = "timestamp,rps,latency_90,latency_95,latency_98,latency_99,latency_999"
        # 添加所有服务的CPU分配列
        for service in services:
            header += f",cpu_{service}"
        # 添加总CPU列
        header += ",total_cpu\n"
        f.write(header)

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
            pids.extend([worker_process.pid for worker_process in worker_processes])
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
                pids = []  # 重置PID列表
                time.sleep(5)
                print(f"等待5秒后重新启动")
        except Exception as e:
            print(f"启动Locust失败: {str(e)}")
            # 清理已启动的进程
            for pid in pids:
                try:
                    _, _ = execute_command(f"sudo kill {pid}")
                except:
                    pass
            raise

    # 初始化 VPA 管理器
    vpa_manager = MultiServiceVPAManager(config_path=f"{PROJECT_ROOT}/k8s/config/hotelreservation_config.json")

    # 初始化配置
    print("执行初始化配置...")
    init_allocate = {}
    with open(f"{PROJECT_ROOT}/k8s/config/hotelreservation_config.json", "r") as f:
        config = json.load(f)
        for service in services:
            init_allocate[service] = config[service]["max_allowed"]

    # 应用初始配置
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

            # 获取每个服务总cpu使用率
            cpu_usage = {k: sum(v) for k, v in gathered["cpu"].items()}
            print(f"CPU使用情况: {cpu_usage}")

            # 添加样本到VPA管理器
            vpa_manager.add_samples(cpu_usage, current_exp_time)

            # 获取延迟数据
            latency_data = get_latest_latency()
            print(f"当前延迟{latency_data}")

            if current_exp_time < 100:
                print(f"预热阶段，{current_exp_time}/100")
            else:
                # 使用VPA获取推荐配置
                new_allocate = vpa_manager.get_recommendations()
                print(f"VPA推荐配置: {new_allocate}")

                # 配置更新阶段
                update_start = time.time()
                print(f"更新CPU配置....")

                # 计算总CPU分配量
                total_cpu = sum(new_allocate.values())
                print(f"总CPU分配数量：{total_cpu}")

                # 为了更新容器配置，计算每个副本的配置
                per_replica_allocation = {}
                for service in new_allocate:
                    per_replica_allocation[service] = new_allocate[service] / service_replicas[service]

                # 将所有数据写入同一个CSV文件
                with open(exp_data_csv_path, "a") as f:
                    # 构建一行数据：开始是时间戳和延迟指标
                    if len(latency_data) >= 6:  # 确保延迟数据完整
                        line = f"{current_exp_time},{latency_data[0]},{latency_data[1]},{latency_data[2]},"
                        line += f"{latency_data[3]},{latency_data[4]},{latency_data[5]}"
                    else:
                        # 如果延迟数据不完整，填充0
                        line = f"{current_exp_time}," + ",".join(["0"] * 6)

                    # 添加每个服务的总CPU分配
                    for service in services:
                        allocation = new_allocate.get(service, 0)
                        line += f",{allocation}"

                    # 添加总CPU使用量
                    line += f",{total_cpu}\n"
                    f.write(line)

                # 更新服务配置
                for connection in connections.values():
                    connection.send_command_sync(f"update{json.dumps(per_replica_allocation)}")

            total_time = time.time() - start_time
            print(f"总时间: {total_time:.3f}秒")
            print("-" * 50)

            elapsed_time = time.time() - start_time
            if elapsed_time < 1:
                time.sleep(1 - elapsed_time)

            current_exp_time += 1
            if current_exp_time == exp_time:
                print(f"实验数据已保存到 {exp_data_dir}/experiment_data.csv")
                for pid in pids:
                    _, _ = execute_command(f"sudo kill {pid}")
                    time.sleep(1)
                break

    finally:
        # 清理locust进程
        for pid in pids:
            try:
                _, _ = execute_command(f"sudo kill {pid}")
            except:
                pass


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


async def main():
    global gathered_list, replicas, exp_time
    # 从配置文件中读取主机名和端口，然后创建连接
    comm_config = ""
    with open(f"{PROJECT_ROOT}/communication/comm.json", "r") as f:
        comm_config = json.load(f)

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

    for users in [1000, 1300, 1600, 1900, 2200, 2500, 2800, 3100, 3400, 3700]:
        command = ("cd ~/DeepDynamicRM/deploy && "
                   "~/miniconda3/envs/DDRM/bin/python3 "
                   "deploy_hotel.py")
        execute_command(command, stream_output=True)
        time.sleep(10)

        for load_type in ["bursty"]:
            # 移除 min_core 参数，使用 None 代替
            await start_experiment(connections, users, load_type)

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
