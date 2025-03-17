import argparse
import os
import sys
import time
import json
import paramiko

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

parser = argparse.ArgumentParser()
parser.add_argument("--bench_dir",
                    type=str,
                    default="~/DeepDynamicRM/benchmarks/hotelReservation/",
                    help="benchmark data dir")
parser.add_argument("--benchmark_config",
                    type=str,
                    default="./config/hotelreservation.json",
                    help="benchmark config file path")
parser.add_argument("--username", type=str, default="tomly", help="username for ssh")
parser.add_argument("--benchmark_name", type=str, default="hotel", help="benchmark name")
args = parser.parse_args()

username = args.username
benchmark_config = args.benchmark_config
benchmark_name = args.benchmark_name


# 加载配置文件
def load_config(file_path: str):
    try:
        with open(file_path, 'r') as f:
            config = json.load(f)
        return config
    except FileNotFoundError:
        print(f"配置文件 {file_path} 未找到")
        return None
    except json.JSONDecodeError as e:
        print(f"配置文件格式错误: {e}")
        return None


config = load_config('./config/config.json')


# 使用paramiko执行远程命令
def execute_command(host, username, command, stream_output=False):
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=host, username=username)
        stdin, stdout, stderr = client.exec_command(command)

        if stream_output:
            # 实时输出结果
            while True:
                line = stdout.readline()
                if not line:
                    break
                print(line.strip())

        stdout_str = stdout.read().decode('utf-8')
        stderr_str = stderr.read().decode('utf-8')

        return stdout_str, stderr_str
    except Exception as e:
        return "", str(e)
    finally:
        client.close()


# 停止所有节点上的服务
def stop_all_services():
    nodes = [config["cluster"]["master"]] + config["cluster"]["workers"]

    for node in nodes:
        print(f"正在停止节点 {node['name']} 上的服务...")
        # 使用sudo强制删除所有容器
        stop_command = "sudo docker rm -f $(sudo docker ps -aq) 2>/dev/null || true"
        stdout, stderr = execute_command(node["host"], username, stop_command)
        if stderr:
            print(f"警告: 停止节点 {node['name']} 上的服务时出错: {stderr}")

    print("所有服务已强制停止并删除")


# 在指定节点上部署服务
def deploy_to_node(node_name, node_host, compose_file):
    compose_path = f"{PROJECT_ROOT}/benchmarks/hotelReservation/{compose_file}"

    # 确保目标目录存在
    mkdir_command = f"mkdir -p {PROJECT_ROOT}/benchmarks/hotelReservation"
    stdout, stderr = execute_command(node_host, username, mkdir_command)
    if stderr:
        raise RuntimeError(f"在节点 {node_name} 上创建目录时出错: {stderr}")

    # 部署服务
    deploy_command = f"cd {PROJECT_ROOT}/benchmarks/hotelReservation && docker compose -f {compose_file} up -d"
    print(f"正在节点 {node_name} 上部署服务，命令: {deploy_command}")
    stdout, stderr = execute_command(node_host, username, deploy_command, stream_output=True)
    # if stderr:
    #     raise RuntimeError(f"在节点 {node_name} 上部署服务时出错: {stderr}")

    print(f"节点 {node_name} 上的服务部署成功")


# 检查服务状态
def check_service_status(node_name, node_host):
    check_command = "sudo docker ps --format '{{.Names}}: {{.Status}}'"
    result, stderr = execute_command(node_host, username, check_command)
    if stderr:
        raise RuntimeError(f"检查节点 {node_name} 上的服务状态时出错: {stderr}")

    print(f"节点 {node_name} 上的服务状态：")
    print(result)

    # 检查每个容器的状态
    all_running = True
    if result.strip():
        containers = result.strip().split('\n')
        for container in containers:
            if "Up" not in container:
                print(f"警告：容器 {container.split(':')[0]} 未处于运行状态")
                all_running = False

        if all_running:
            print(f"节点 {node_name} 上的所有容器都在正常运行")
        else:
            print(f"警告：节点 {node_name} 上有容器未正常运行")
        return all_running
    else:
        print(f"警告：节点 {node_name} 上没有运行中的容器")
        return False


# 部署基准测试
def deploy_benchmark():
    resource_config = load_config(benchmark_config)

    # 停止所有现有服务
    stop_all_services()

    # 在四个节点上分别部署服务
    node_configs = {
        "rm1": {
            "file": "docker-compose-rm1.yml",
            "host": "rm1"
        },
        "rm2": {
            "file": "docker-compose-rm2.yml",
            "host": "rm2"
        },
        "rm3": {
            "file": "docker-compose-rm3.yml",
            "host": "rm3"
        },
        "rm4": {
            "file": "docker-compose-rm4.yml",
            "host": "rm4"
        }
    }

    # 按照4-1-2-3的顺序部署
    deployment_order = ["rm4", "rm1", "rm2", "rm3"]
    for node_name in deployment_order:
        node_config = node_configs[node_name]
        deploy_to_node(node_name, node_config["host"], node_config["file"])

    # 等待所有服务启动完成
    print("等待所有服务启动...")
    time.sleep(10)

    # 检查所有节点上的服务状态
    print("检查所有节点上的服务状态...")
    all_nodes_running = True

    for node_name in deployment_order:
        node_running = check_service_status(node_name, node_name)
        if not node_running:
            all_nodes_running = False

    if all_nodes_running:
        print("所有节点上的服务都已成功启动！")
    else:
        print("警告：部分节点上的服务未能正常启动，请检查日志获取详细信息。")

    return all_nodes_running


if __name__ == "__main__":
    deploy_benchmark()
