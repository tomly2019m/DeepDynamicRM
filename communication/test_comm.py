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
            data = self.socket.recv(20480)
            print(f"Received from {self.slave_host}:{self.slave_port}:", data.decode())
            return data.decode()

    def close(self):
        if self.socket:
            self.socket.close()
            print(f"Connection to {self.slave_host}:{self.slave_port} closed.")


def test_slave_connection():
    s = time.time()
    slave_host = "127.0.0.1"
    slave_port = 12345
    connection = SlaveConnection(slave_host, slave_port)
    s = time.time()
    response = connection.send_command("init")
    print(time.time() - s)
    print(response)
    # 测试多个不同的命令
    test_commands = ["collect", "collect", "collect", "collect"]

    print("\n测试多个命令:")
    for cmd in test_commands:
        print(f"\n执行命令: {cmd}")
        s = time.time()
        response = connection.send_command(cmd)
        print(time.time() - s)
        # print(f"响应: {response}")

    # 测试连续快速发送命令
    print("\n测试连续快速发送命令:")
    for i in range(5):
        cmd = "collect"
        response = connection.send_command(cmd)
        # print(f"命令 {i} 响应: {response}")

    # 清理连接
    connection.close()


if __name__ == "__main__":
    test_slave_connection()
