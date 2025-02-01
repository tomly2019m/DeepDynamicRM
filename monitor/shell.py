import subprocess
import asyncio
import sys
from typing import Tuple, Optional

def execute_command(command: str, timeout: int = -1, stream_output: bool = False) -> Tuple[str, Optional[str]]:
    """
    同步执行命令并返回结果
    
    Args:
        command: 要执行的命令
        timeout: 超时时间(秒)，-1表示不超时
        stream_output: 是否流式输出
    Returns:
        (output, error): 输出结果和错误信息的元组
    """
    try:
        if stream_output:
            result = subprocess.run(
                command, 
                shell=True, 
                stdout=sys.stdout, 
                stderr=sys.stderr
            )
            return None, None
        else:
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                shell=True,
                universal_newlines=True
            )
            
            if timeout == -1:
                output, error = process.communicate()
            else:
                output, error = process.communicate(timeout=timeout)
                
            return output.strip(), error.strip() if error else None
        
    except subprocess.TimeoutExpired:
        process.kill()
        return "", f"命令执行超时(>{timeout}秒)"
    except Exception as e:
        return "", str(e)

async def execute_command_async(command: str, timeout: int = -1) -> Tuple[str, Optional[str]]:
    """
    异步执行命令并返回结果
    
    Args:
        command: 要执行的命令
        timeout: 超时时间(秒)，-1表示不超时
    Returns:
        (output, error): 输出结果和错误信息的元组
    """
    try:
        process = await asyncio.create_subprocess_shell(
            command,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE
        )
        
        if timeout == -1:
            output, error = await process.communicate()
        else:
            try:
                output, error = await asyncio.wait_for(
                    process.communicate(),
                    timeout=timeout
                )
            except asyncio.TimeoutError:
                process.kill()
                return "", f"命令执行超时(>{timeout}秒)"
                
        return output.decode().strip(), error.decode().strip() if error else None
        
    except Exception as e:
        return "", str(e)

def test_commands():
    # 测试同步命令执行
    print("测试同步命令执行:")
    output, error = execute_command("ls -l", timeout=5, stream_output=True)
    print(f"输出: {output}")
    print(f"错误: {error}")
    
    # 测试超时情况
    print("\n测试超时情况:")
    output, error = execute_command("sleep 10", timeout=2) 
    print(f"输出: {output}")
    print(f"错误: {error}")

    # 测试异常情况
    print("\n测试异常情况:")
    output, error = execute_command("不存在的命令")
    print(f"输出: {output}")
    print(f"错误: {error}")

async def test_async_commands():
    # 测试异步命令执行
    print("测试异步命令执行:")
    output, error = await execute_command_async("ls -l", timeout=5)
    print(f"输出: {output}")
    print(f"错误: {error}")
    
    # 测试异步超时情况
    print("\n测试异步超时情况:")
    output, error = await execute_command_async("sleep 10", timeout=2)
    print(f"输出: {output}")
    print(f"错误: {error}")

    # 测试异步异常情况
    print("\n测试异步异常情况:")
    output, error = await execute_command_async("不存在的命令")
    print(f"输出: {output}")
    print(f"错误: {error}")

if __name__ == "__main__":
    # 测试同步命令
    test_commands()
    
    # 测试异步命令
    asyncio.run(test_async_commands())
