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

parser = argparse.ArgumentParser()
parser.add_argument("--exp_time", type=int, default=30, help="experiment time")
parser.add_argument("--username", type=str, default="tomly", help="username for SSH connection")
parser.add_argument("--save", action="store_true", help="whether to save data")

args = parser.parse_args()

exp_time = args.exp_time
username = args.username
save     = args.save

gathered_list = []  # 用于存储每次循环处理后的 gathered 数据
replicas = []
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

    async def send_command(self, command) -> str:
        if self.socket:
            self.socket.sendall(command.encode())
            data = self.socket.recv(20480)
            # print(f'Received from {self.slave_host}:{self.slave_port}:', data.decode())
            return data.decode()

    def close(self):
        if self.socket:
            self.socket.close()
            print(f"Connection to {self.slave_host}:{self.slave_port} closed.")


async def start_experiment(slaves):
    global exp_time, gathered_list, replicas
    connections : Dict[Tuple[str, int], SlaveConnection] = {}
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
            "locust",          # 命令名称
            "-f",              # 参数：指定locust文件路径
            "src/socialnetwork.py",  # 你的Locust文件路径
            "--host",           # 参数：目标主机
            "http://127.0.0.1:8080",
            "--users",          # 用户数参数
            "2",
            "--csv",           # 输出CSV文件
            "locust_log",
            "--headless",       # 无头模式
            "-t",               # 测试时长
            f"{exp_time + 100}s"             # 100秒运行时间
        ]
    
    try:
        # 创建子进程，不等待立即返回
        process = await asyncio.to_thread(
            lambda: subprocess.Popen(
                locust_cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,  # 自动解码文本
                start_new_session=True  # 独立会话
            )
        )
        
        # 可选：立即返回成功信息（或根据需要返回进程ID）
        print(f"Locust进程已启动 (PID: {process.pid})")
        
    except Exception as e:
        # 捕获启动错误（如命令不存在、路径错误等）
        print(f"启动Locust失败: {str(e)}")
        raise

    mab = UCB_Bandit()

    current_exp_time = 0
    try:
        while True:
            gathered = {
                'cpu':{},
                'memory':{},
                'io':{},
                'network':{}
            }
            # 遍历所有slave连接，发送collect命令采集数据
            tasks.clear()
            for connection in connections.values():
                tasks.append(asyncio.create_task(connection.send_command("collect")))
            results = await asyncio.gather(*tasks)
            # s = time.time()
            for result in results:
                data_dict = json.loads(result)
                gathered['cpu'] = concat_data(gathered['cpu'], data_dict['cpu'])
                gathered['memory'] = concat_data(gathered['memory'], data_dict['memory'])
                gathered['io'] = concat_data(gathered['io'], data_dict['io'])
                gathered['network'] = concat_data(gathered['network'], data_dict['network'])
            if replicas == []:
                replicas = np.array([len(cpu_list) for cpu_list in gathered['cpu'].values()]).flatten()
            # print(replicas)
            print(f"当前实验进度: {current_exp_time}/{exp_time}")
            print(gathered)
            for k, v in gathered['cpu'].items():
                gathered['cpu'][k] = [item / 1e6 for item in v]

            arm_id = mab.select_arm()
            mab.execute_action(arm_id, gathered["cpu"])
            latency = get_latest_latency()
            reward = mab.calculate_reward(latency)
            mab.update(arm_id, reward)
            
            gathered['cpu'] = process_data(gathered['cpu'])
            gathered['memory'] = process_data(gathered['memory'])
            gathered['io'] = process_data(gathered['io'])
            gathered['network'] = process_data(gathered['network'])

            gathered = transform_data(gathered)
            # print(time.time() - s)
            gathered_list.append(gathered)  # 将处理后的 gathered 数据存储到列表中
            latency_list.append(latency)
            time.sleep(1)
            exp_time -= 1
            current_exp_time += 1
            
            # 实验结束
            if exp_time == 0:
                break
    finally:
        for connection in connections.values():
            connection.close()


# 配置好slave，在slave上启动监听
def setup_slave():
    # 从配置文件中读取主机名和端口
    comm_config = ''
    with open("./comm.json", 'r') as f:
        comm_config = json.load(f)
    hosts = comm_config["slaves"] 
    port = comm_config["port"]

    # 在每个slave节点上启动监听服务
    for host in hosts:
        # 通过SSH连接到slave节点
        ssh = paramiko.SSHClient()
        ssh.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh.connect(host, username=username)
            # 先切换到目标目录在slave节点上启动监听程序

            # 清理旧的进程
            command = f"sudo kill -9 $(sudo lsof -t -i :{port})"
            stdin, stdout, stderr = ssh.exec_command(command)

            command = (
                'cd ~/DeepDynamicRM/communication && '
                'nohup ~/miniconda3/envs/DDRM/bin/python3 '
                f'slave.py --port {port} > /dev/null 2>&1 &'
            )
            
            stdin, stdout, stderr = ssh.exec_command(command)
            
            print(f'在 {host} 上启动监听服务,端口:{port}')
            
        except Exception as e:
            print(f'连接到 {host} 失败: {str(e)}')
        finally:
            ssh.close()

def save_data(gathered_list, replicas):
    """保存实验数据到本地文件"""
    # 创建数据目录(如果不存在)
    if not os.path.exists('./data'):
        os.makedirs('./data')

    # 保存gathered数据
    gathered_path = f'./data/gathered.npy'
    np.save(gathered_path, gathered_list)
    print(f"已保存gathered数据到: {gathered_path}")
    
    # 保存replicas数据 
    replicas_path = f'./data/replicas.npy'
    np.save(replicas_path, replicas)
    print(f"已保存replicas数据到: {replicas_path}")

    # 保存延迟latency数据
    latency_path = f'./data/latency.npy'
    np.save(latency_path, latency_list)
    print(f"已保存latency数据到: {latency_path}")

class Executor:
    pass


async def main():
    global gathered_list, replicas
    # 从配置文件中读取主机名和端口，然后创建连接
    comm_config = ''
    with open("./comm.json", 'r') as f:
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
    with open("./comm.json", 'r') as f:
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

