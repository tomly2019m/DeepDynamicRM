import argparse
import json
import os
import socket
import sys
import time

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
        self.connect()

    def connect(self):
        self.socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.socket.connect((self.slave_host, self.slave_port))
        print(f"Connected to slave at {self.slave_host}:{self.slave_port}")

    def send_command(self, command) -> str:
        if self.socket:
            self.socket.sendall(command.encode())
            data = self.socket.recv(1024)
            print(f'Received from {self.slave_host}:{self.slave_port}:', data.decode())
            return data.decode()

    def close(self):
        if self.socket:
            self.socket.close()
            print(f"Connection to {self.slave_host}:{self.slave_port} closed.")

def start_experiment(slaves):
    global exp_time
    connections : dict[tuple[str, int], SlaveConnection] = {}
    for slave_host, slave_port in slaves:
        connection = SlaveConnection(slave_host, slave_port)
        connection.send_command("init")
        connections[(slave_host, slave_port)] = connection
    try:
        while True:
            # 遍历所有slave连接，发送collect命令采集数据
            for connection in connections.values():
                connection.send_command("collect")
            
            time.sleep(1)
            exp_time -= 1
            
            # 实验结束
            if exp_time == 0:
                break
    finally:
        for connection in connections.values():
            connection.close()


def main():
    # 从配置文件中读取主机名和端口，然后创建连接
    comm_config = ''
    with open("./comm.json", 'r') as f:
        comm_config = json.load(f)
    hosts = comm_config["slaves"]
    port = comm_config["port"]
    slaves = []
    for host in hosts:
        slaves.append((host, port))
    start_experiment(slaves)

if __name__ == "__main__":
    main()