import subprocess
import os
import sys

def setup_ssh_multiplexing(host: str, username: str):
    """
    设置 SSH 连接复用
    """
    control_path = f"/tmp/ssh_mux_{username}_{host}"
    
    # 检查控制socket是否已经存在
    if not os.path.exists(control_path):
        # 建立主连接，添加更多控制选项
        ssh_command = (
            f"ssh -fNM "
            f"-o ControlMaster=yes "
            f"-o ControlPath={control_path} "
            f"-o ControlPersist=1h "  # 空闲连接保持1小时
            f"-o ServerAliveInterval=60 "  # 每60秒发送一次保活包
            f"-o ServerAliveCountMax=3 "   # 最多允许3次保活失败
            f"{username}@{host}"
        )
        subprocess.run(ssh_command, shell=True, check=True)
    
    return control_path

def execute_command_via_system_ssh(host: str, username: str, command: str, async_exec: bool = False, stream_output: bool = False):
    """
    使用系统 ssh 命令执行远程命令（使用连接复用）
    
    Args:
        host: 目标主机
        username: 用户名
        command: 要执行的命令
        async_exec: 是否异步执行，默认为False
        stream_output: 是否实时输出，默认为False
    """
    control_path = setup_ssh_multiplexing(host, username)
    ssh_command = f"ssh -o ControlPath={control_path} {username}@{host} '{command}'"
    
    if stream_output:
        # 实时输出模式, 输出到标准输出
        result = subprocess.run(
            ssh_command, 
            shell=True, 
            stdout=sys.stdout, 
            stderr=sys.stderr
        )
        return None, None
        
    elif async_exec:
        # 异步执行模式
        process = subprocess.Popen(
            ssh_command,
            shell=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True
        )
        return process
    else:
        # 同步执行模式
        try:
            result = subprocess.run(ssh_command, shell=True, check=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True)
            return result.stdout, result.stderr
        except subprocess.CalledProcessError as e:
            print(f"系统 SSH 执行失败: {e.stderr}")
            return e.stdout, e.stderr

def cleanup_ssh_multiplexing(host: str, username: str):
    """
    清理 SSH 复用连接
    """
    control_path = f"/tmp/ssh_mux_{username}_{host}"
    if os.path.exists(control_path):
        ssh_command = f"ssh -O exit -o ControlPath={control_path} {username}@{host}"
        subprocess.run(ssh_command, shell=True)

def test_ssh_command():
    host = "debian2"
    username = 'tomly'
    command = 'ls -la'
    output, error = execute_command_via_system_ssh(host, username, command)
    print(output, error)
    if error:
        print("执行命令出错！")


def test_ssh_command_timing():
    import time
    
    host = "debian2"
    username = 'tomly'
    command = 'ls -la'
    
    start_time = time.time()
    output, error = execute_command_via_system_ssh(host, username, command)
    end_time = time.time()
    
    execution_time = end_time - start_time
    print(f"SSH命令执行时间: {execution_time:.2f} 秒")
    print(f"命令输出:\n{output}")
    if error:
        print(f"执行命令出错: {error}")


def test_ssh_multiplexing_performance():
    """
    测试SSH连接复用的性能
    """
    import time
    
    host = "debian2"
    username = 'tomly'
    command = 'ls -la'
    iterations = 5  # 测试次数
    
    print("开始性能测试...")
    print("\n1. 第一次连接（建立主连接）:")
    start_time = time.time()
    output, error = execute_command_via_system_ssh(host, username, command)
    first_execution_time = time.time() - start_time
    print(f"首次执行时间: {first_execution_time:.3f} 秒")
    
    print("\n2. 后续复用连接测试:")
    times = []
    for i in range(iterations):
        start_time = time.time()
        output, error = execute_command_via_system_ssh(host, username, command)
        execution_time = time.time() - start_time
        times.append(execution_time)
        print(f"第 {i+1} 次执行时间: {execution_time:.3f} 秒")
    
    avg_time = sum(times) / len(times)
    print(f"\n复用连接平均执行时间: {avg_time:.3f} 秒")
    print(f"最快执行时间: {min(times):.3f} 秒")
    print(f"最慢执行时间: {max(times):.3f} 秒")


def test_long_running_command():
    """
    测试长时间运行命令的实时输出
    """
    host = "debian2"
    username = "tomly"
    # 一个长时间运行的命令示例
    command = "for i in {1..10}; do echo $i; sleep 1; done"
    
    print("开始执行长时间运行的命令...")
    output, error = execute_command_via_system_ssh(
        host=host,
        username=username,
        command=command,
        stream_output=True
    )
    print("命令执行完成！")



if __name__ == "__main__":
    # test_ssh_multiplexing_performance()
    test_long_running_command()