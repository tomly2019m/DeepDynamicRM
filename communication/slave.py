import os
import socket
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from monitor.data_collector import *

import argparse

parser = argparse.ArgumentParser()
parser.add_argument("--port", type=int, default=12345, help="port to listen")
args = parser.parse_args()
# 从命令行参数获取端口号
port = args.port
host = '0.0.0.0'  # 监听所有可用接口


def handle_command(command: str):
    # 在这里解析并执行命令
    print('Executing command:', command)

    response = ""
    if command == 'init':
        init_collector()
        print("Collector initialized")
        response = "Collector initialized"

    elif command == 'collect':
        latest_data = {
            "cpu": get_container_cpu_usage(),
            "memory": get_memory_usage(),
            "io": get_io_usage(),
            "network": get_network_usage()
        }
        response = json.dumps(latest_data)

    elif "update" in command:
        command = command.replace("update", "")
        allocate_dict = json.loads(command)
        set_cpu_limit(allocate_dict)
    return response


def slave_listen(master_host, master_port):
    # 创建 socket 对象
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # 绑定到指定的主机和端口
        s.bind((master_host, master_port))
        # 监听连接
        s.listen()
        print('等待连接...')
        conn, addr = s.accept()
        with conn:
            print(f'连接成功: {addr}')
            while True:
                data = b''

                while True:
                    chunk = conn.recv(20480)
                    data += chunk
                    if data.endswith(b"\r\n\r\n"):
                        # 去除结束符并解码
                        data = data[:-4]
                        break

                command = data.decode()
                # 处理命令
                result = handle_command(command)

                if result == "stop" or not result:
                    break
                # 返回结果，添加结束符
                result = f"{result}\r\n\r\n"
                conn.sendall(result.encode())


if __name__ == "__main__":
    slave_listen(host, port)
