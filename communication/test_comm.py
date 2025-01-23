from master import SlaveConnection


def test_slave_connection():
    slave_host = '127.0.0.1'
    slave_port = 12345
    connection = SlaveConnection(slave_host, slave_port)
    response = connection.send_command("Test command")
    print(response)
    # 测试多个不同的命令
    test_commands = [
        "status",
        "help",
        "version", 
        "echo Hello World",
        "ping"
    ]
    
    print("\n测试多个命令:")
    for cmd in test_commands:
        print(f"\n执行命令: {cmd}")
        response = connection.send_command(cmd)
        print(f"响应: {response}")
        
    # 测试连续快速发送命令
    print("\n测试连续快速发送命令:")
    for i in range(5):
        cmd = f"quick_command_{i}"
        response = connection.send_command(cmd)
        print(f"命令 {i} 响应: {response}")
        
    # 清理连接
    connection.close()


if __name__ == "__main__":
    test_slave_connection()