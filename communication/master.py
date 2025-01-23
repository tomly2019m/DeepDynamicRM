import socket
import threading

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

def manage_connections(slaves):
    connections = {}
    for slave_host, slave_port in slaves:
        connection = SlaveConnection(slave_host, slave_port)
        connections[(slave_host, slave_port)] = connection

    try:
        while True:
            command = input("Enter command (or 'exit' to quit): ")
            if command.lower() == 'exit':
                break
            target = input("Enter target slave (host:port): ")
            host, port = target.split(':')
            port = int(port)
            if (host, port) in connections:
                connections[(host, port)].send_command(command)
            else:
                print("Invalid target.")
    finally:
        for connection in connections.values():
            connection.close()

if __name__ == "__main__":
    slaves = [
        ('slave1_host', 12345),  # 替换为实际的 slave 主机名或 IP 和端口
        ('slave2_host', 12346),  # 添加更多的 slave 信息
        # ...
    ]
    pass