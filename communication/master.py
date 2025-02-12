import argparse
import json
import os
import socket
import sys
import time
import asyncio
from typing import Dict, Tuple
import paramiko
from sync import distribute_project

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--exp_time", type=int, default=30, help="experiment time")
parser.add_argument("--username", type=str, default="tomly", help="username for SSH connection")

args = parser.parse_args()

exp_time = args.exp_time
username = args.username


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
            print(f'Received from {self.slave_host}:{self.slave_port}:', data.decode())
            return data.decode()

    def close(self):
        if self.socket:
            self.socket.close()
            print(f"Connection to {self.slave_host}:{self.slave_port} closed.")


async def start_experiment(slaves):
    global exp_time
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

    try:
        while True:
            # 遍历所有slave连接，发送collect命令采集数据
            for connection in connections.values():
                tasks.append(asyncio.create_task(connection.send_command("collect")))

            results = await asyncio.gather(*tasks)
            for result in results:
                print(f"Received data: {result}")
            
            time.sleep(1)
            exp_time -= 1
            
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


async def main():
    # 从配置文件中读取主机名和端口，然后创建连接
    comm_config = ''
    with open("./comm.json", 'r') as f:
        comm_config = json.load(f)
    hosts = comm_config["slaves"]
    port = comm_config["port"]
    slaves = [(host, port) for host in hosts]
    
    distribute_project(username=username)
    setup_slave()
    await start_experiment(slaves)


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
    # asyncio.run(main())
    test_setup_slave()
