import socket

def handle_command(command):
    # 在这里解析并执行命令
    # 这是一个简单的示例，实际操作需要根据命令内容进行
    print('Executing command:', command)
    result = f"Executed: {command}"
    return result

def slave_listen(master_host, master_port):
    # 创建 socket 对象
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        # 绑定到指定的主机和端口
        s.bind((master_host, master_port))
        # 监听连接
        s.listen()
        print('Waiting for connection...')
        conn, addr = s.accept()
        with conn:
            print('Connected by', addr)
            while True:
                # 接收数据
                data = conn.recv(1024)
                if not data:
                    break
                command = data.decode()
                # 处理命令
                result = handle_command(command)
                # 返回结果
                conn.sendall(result.encode())

if __name__ == "__main__":
    master_host = '0.0.0.0'  # 监听所有可用接口
    master_port = 12345      # 替换为实际的端口号
    slave_listen(master_host, master_port) 