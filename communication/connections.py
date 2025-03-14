import socket


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
            # print(f'Received from {self.slave_host}:{self.slave_port}:',
            #       data.decode())
            return data

    # async def send_command(self, command) -> str:
    #     if self.socket:
    #         # 添加结束标记
    #         command = f"{command}\r\n\r\n"
    #         self.socket.sendall(command.encode())
    #         data = b""
    #         while True:
    #             chunk = self.socket.recv(1024)
    #             # 连接关闭时退出
    #             data += chunk
    #             # 检测服务端的结束符
    #             if data.endswith(b"\r\n\r\n"):
    #                 # 去除结束符并解码
    #                 data = data[:-4]
    #                 break
    #         # print(f'Received from {self.slave_host}:{self.slave_port}:',
    #         #       data.decode())
    #         return data.decode()

    def close(self):
        if self.socket:
            self.socket.close()
            print(f"Connection to {self.slave_host}:{self.slave_port} closed.")
