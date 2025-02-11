import argparse
import json
import os
import socket
import sys
import time
import asyncio
from typing import Dict, Tuple

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--exp_time", type=int, default=30, help="experiment time")
args = parser.parse_args()

exp_time = args.exp_time


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


async def main():
    # 从配置文件中读取主机名和端口，然后创建连接
    comm_config = ''
    with open("./comm.json", 'r') as f:
        comm_config = json.load(f)
    hosts = comm_config["slaves"]
    port = comm_config["port"]
    slaves = [(host, port) for host in hosts]
    
    await start_experiment(slaves)


if __name__ == "__main__":
    asyncio.run(main())
