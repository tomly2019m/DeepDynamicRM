import argparse
import os
import sys
import json

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from monitor.shell import execute_command


def distribute_project(username: str):
    """分发项目到所有slave节点"""
    # 读取节点配置
    with open("./comm.json", "r") as f:
        config = json.load(f)

    master_host = config["master"]
    slave_hosts = [h for h in config["slaves"] if h != master_host]

    # 获取要分发的项目路径
    project_name = os.path.basename(PROJECT_ROOT)
    local_path = os.path.join(PROJECT_ROOT, "..", project_name)

    print(f"🚀 开始分发项目到 {len(slave_hosts)} 个节点")

    # 遍历所有从节点
    for slave in slave_hosts:
        try:
            print(f"\n🔧 正在处理节点 {slave}")

            # 构造远程目录路径
            remote_path = f"{username}@{slave}:~/{project_name}"

            # 执行同步命令（带进度显示）
            cmd = f"rsync -avz --delete {local_path}/ {remote_path}"
            print(f"执行命令: {cmd}")

            # 使用流式输出执行命令
            output, error = execute_command(cmd, stream_output=True)

            # 检查执行结果（流式模式下通过异常捕获错误）
            if error:
                print(f"❌ 同步到 {slave} 失败: {error}")
            else:
                print(f"✅ 成功同步到 {slave}")

        except Exception as e:
            print(f"⚠️ 处理节点 {slave} 时发生意外错误: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--username", type=str, default="tomly", help="SSH登录用户名（所有节点需相同）")
    args = parser.parse_args()

    distribute_project(args.username)
