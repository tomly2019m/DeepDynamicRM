import argparse
import os
import subprocess
import time
import docker
import json

import paramiko
from util.ssh import execute_command_via_system_ssh
from util.parser import parse_swarm_output, parse_node_label

join_cluster_command = ''
client = docker.from_env()

parser = argparse.ArgumentParser()
parser.add_argument("--docker_compose",
                    type=str,
                    default="~/DeepDynamicRM/benchmarks/socialNetwork-ml-swarm/docker-compose-swarm.yml",
                    help="benchmark yaml file path")
parser.add_argument("--bench_dir",
                    type=str,
                    default="~/DeepDynamicRM/benchmarks/socialNetwork-ml-swarm/",
                    help="benchmark data dir")
parser.add_argument("--benchmark_config",
                    type=str,
                    default="./config/socialnetwork.json",
                    help="benchmark config file path")
parser.add_argument("--username", type=str, default="tomly", help="username for ssh")
parser.add_argument("--benchmark_name", type=str, default="socialnetwork", help="benchmark name")
args = parser.parse_args()

username = args.username
docker_compose_file = args.docker_compose
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

# 初始化docker swarm集群，返回加入集群命令
# def init_master():
#     global join_cluster_command
#     swarm_init_command = "docker swarm init"
#     master_host = config["cluster"]["master"]["host"]
#     result, err = execute_command_via_system_ssh(master_host, username, swarm_init_command)
#     if err:
#         raise RuntimeError(f"执行初始化命令出错: {err}")

#     join_cluster_command = parse_swarm_output(result)["worker_command"]
#     assert "token" in join_cluster_command
#     print(join_cluster_command)
#     print("初始化管理节点完成")


def init_master():
    global join_cluster_command
    master_host = config["cluster"]["master"]["host"]
    swarm_init_command = "docker swarm init"

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=master_host, username=username, timeout=10)
        stdin, stdout, stderr = client.exec_command(swarm_init_command)
        # 等待命令执行完成
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            error_msg = stderr.read().decode().strip()
            raise RuntimeError(f"执行初始化命令出错: {error_msg}")

        result = stdout.read().decode()
        # 解析返回结果，提取 worker 加入集群的命令
        join_cluster_command = parse_swarm_output(result)["worker_command"]
        assert "token" in join_cluster_command
        print(join_cluster_command)
        print("初始化管理节点完成")
    except Exception as e:
        raise RuntimeError(f"连接管理节点 {master_host} 时出错: {e}")
    finally:
        client.close()


# 解散docker swarm集群
# def dissolve_cluster():
#     for worker in config["cluster"]["workers"]:
#         leave_command = "docker swarm leave"
#         _, err = execute_command_via_system_ssh(worker["host"], username, leave_command)
#         if err:
#             raise RuntimeError(f"{worker['name']}执行离开集群命令出错: {err}")
#         print(f"工作节点 {worker['name']} 离开集群成功")
#     master_host = config["cluster"]["master"]["host"]
#     leave_command = "docker swarm leave -f"
#     _, err = execute_command_via_system_ssh(master_host, username, leave_command)
#     if err:
#         raise RuntimeError(f"管理节点 {master_host} 执行离开集群命令出错: {err}")
#     print(f"管理节点 {master_host} 离开集群成功")


def dissolve_cluster():
    # 对所有工作节点执行 "docker swarm leave"
    for worker in config["cluster"]["workers"]:
        host = worker["host"]
        command = "docker swarm leave"
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(hostname=host, username=username, timeout=10)
            stdin, stdout, stderr = client.exec_command(command)
            # 等待命令执行完毕
            exit_status = stdout.channel.recv_exit_status()
            error_msg = stderr.read().decode().strip()
            if exit_status != 0 or error_msg:
                raise RuntimeError(f"{worker['name']} 执行离开集群命令出错: {error_msg}")
            print(f"工作节点 {worker['name']} 离开集群成功")
        except Exception as e:
            raise RuntimeError(f"连接 {worker['name']} 时出错: {e}")
        finally:
            client.close()

    # 对管理节点执行 "docker swarm leave -f"
    master_host = config["cluster"]["master"]["host"]
    command = "docker swarm leave -f"
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=master_host, username=username, timeout=10)
        stdin, stdout, stderr = client.exec_command(command)
        exit_status = stdout.channel.recv_exit_status()
        error_msg = stderr.read().decode().strip()
        if exit_status != 0 or error_msg:
            raise RuntimeError(f"管理节点 {master_host} 执行离开集群命令出错: {error_msg}")
        print(f"管理节点 {master_host} 离开集群成功")
    except Exception as e:
        raise RuntimeError(f"连接管理节点 {master_host} 时出错: {e}")
    finally:
        client.close()


# 配置节点标签
# def config_node_label(node_name: str, label: str):
#     config_node_label_command = f"docker node update --label-add {label} {node_name}"
#     result, err = execute_command_via_system_ssh(config["cluster"]["master"]["host"], username,
#                                                  config_node_label_command)
#     if err:
#         raise RuntimeError(f"执行配置节点标签命令出错: {err}")
#     print(result)


def config_node_label(node_name: str, label: str):
    master_host = config["cluster"]["master"]["host"]
    config_node_label_command = f"docker node update --label-add {label} {node_name}"

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(hostname=master_host, username=username, timeout=10)
        stdin, stdout, stderr = client.exec_command(config_node_label_command)
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            error_msg = stderr.read().decode().strip()
            raise RuntimeError(f"执行配置节点标签命令出错: {error_msg}")
        result = stdout.read().decode().strip()
        print(result)
    except Exception as e:
        raise RuntimeError(f"连接管理节点 {master_host} 时出错: {e}")
    finally:
        client.close()


# 检查节点标签,确保节点标签与配置文件一致
# def check_node_label() -> bool:
#     check_command = 'docker node inspect --format "{{ .Description.Hostname }}: {{ .Spec.Labels }}" $(docker node ls -q)'
#     result, err = execute_command_via_system_ssh(config["cluster"]["master"]["host"], username, check_command)
#     if err:
#         raise RuntimeError(f"执行检查节点标签命令出错: {err}")
#     print(result)
#     nodes_label = parse_node_label(result)
#     master_label = nodes_label[config["cluster"]["master"]["name"]]["type"]
#     if master_label not in config["cluster"]["master"]["label"]:
#         raise RuntimeError(f"管理节点标签与配置文件不一致: {master_label}")
#     for worker in config["cluster"]["workers"]:
#         worker_label = nodes_label[worker["name"]]["type"]
#         if worker_label not in worker["label"]:
#             raise RuntimeError(f"工作节点标签与配置文件不一致: {worker_label}")
#     return True


def check_node_label() -> bool:
    # 构造检查节点标签的命令
    check_command = 'docker node inspect --format "{{ .Description.Hostname }}: {{ .Spec.Labels }}" $(docker node ls -q)'
    master_host = config["cluster"]["master"]["host"]

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

    try:
        client.connect(hostname=master_host, username=username, timeout=10)
        stdin, stdout, stderr = client.exec_command(check_command)
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            error_msg = stderr.read().decode().strip()
            raise RuntimeError(f"执行检查节点标签命令出错: {error_msg}")
        result = stdout.read().decode().strip()
        print(result)
    except Exception as e:
        raise RuntimeError(f"连接管理节点 {master_host} 时出错: {e}")
    finally:
        client.close()

    # 解析节点标签信息
    nodes_label = parse_node_label(result)

    # 检查管理节点标签是否一致
    master_node_name = config["cluster"]["master"]["name"]
    master_label = nodes_label[master_node_name]["type"]
    if master_label not in config["cluster"]["master"]["label"]:
        raise RuntimeError(f"管理节点标签与配置文件不一致: {master_label}")

    # 检查各工作节点标签是否一致
    for worker in config["cluster"]["workers"]:
        worker_label = nodes_label[worker["name"]]["type"]
        if worker_label not in worker["label"]:
            raise RuntimeError(f"工作节点标签与配置文件不一致: {worker_label}")

    return True


# 初始化docker swarm集群
# def setup_swarm_cluster():
#     for worker in config["cluster"]["workers"]:
#         _, err = execute_command_via_system_ssh(worker["host"], username, join_cluster_command)
#         if err:
#             raise RuntimeError(f"执行加入集群命令出错: {err}")
#         print(f"工作节点 {worker['name']} 加入集群成功")
#     print("集群初始化完成")
#     cluster_info_command = "docker node ls"
#     result, err = execute_command_via_system_ssh(config["cluster"]["master"]["host"], username, cluster_info_command)
#     if err:
#         raise RuntimeError(f"执行集群信息命令出错: {err}")
#     print(result)

#     # 配置节点标签
#     config_node_label(config["cluster"]["master"]["name"], config["cluster"]["master"]["label"])
#     for worker in config["cluster"]["workers"]:
#         config_node_label(worker["name"], worker["label"])
#     print("节点标签配置完成")
#     if check_node_label():
#         print("节点标签检查完成")
#     else:
#         raise RuntimeError("节点标签检查失败")


def setup_swarm_cluster():
    # 1. 让所有工作节点加入集群
    for worker in config["cluster"]["workers"]:
        host = worker["host"]
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            client.connect(hostname=host, username=username, timeout=10)
            stdin, stdout, stderr = client.exec_command(join_cluster_command)
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                error_msg = stderr.read().decode().strip()
                raise RuntimeError(f"工作节点 {worker['name']} 执行加入集群命令出错: {error_msg}")
            print(f"工作节点 {worker['name']} 加入集群成功")
        except Exception as e:
            raise RuntimeError(f"连接工作节点 {worker['name']} 时出错: {e}")
        finally:
            client.close()

    print("集群初始化完成")

    # 2. 在管理节点上执行 docker node ls 命令获取集群信息
    master_host = config["cluster"]["master"]["host"]
    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=master_host, username=username, timeout=10)
        cluster_info_command = "docker node ls"
        stdin, stdout, stderr = client.exec_command(cluster_info_command)
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            error_msg = stderr.read().decode().strip()
            raise RuntimeError(f"管理节点 {master_host} 执行集群信息命令出错: {error_msg}")
        result = stdout.read().decode().strip()
        print(result)
    except Exception as e:
        raise RuntimeError(f"连接管理节点 {master_host} 时出错: {e}")
    finally:
        client.close()

    # 3. 配置节点标签
    config_node_label(config["cluster"]["master"]["name"], config["cluster"]["master"]["label"])
    for worker in config["cluster"]["workers"]:
        config_node_label(worker["name"], worker["label"])
    print("节点标签配置完成")

    # 检查节点标签配置是否正确
    if check_node_label():
        print("节点标签检查完成")
    else:
        raise RuntimeError("节点标签检查失败")


# def docker_stack_rm(stack_name: str):
#     docker_stack_rm_command = f"docker stack rm {stack_name}"
#     _, err = execute_command_via_system_ssh(config["cluster"]["master"]["host"],
#                                             username,
#                                             docker_stack_rm_command,
#                                             stream_output=True)
#     if err:
#         raise RuntimeError(f"执行删除栈命令出错: {err}")
#     print(f"栈 {stack_name} 删除成功")
def docker_stack_rm(stack_name: str):
    docker_stack_rm_command = f"docker stack rm {stack_name}"
    master_host = config["cluster"]["master"]["host"]

    client = paramiko.SSHClient()
    client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        client.connect(hostname=master_host, username=username, timeout=10)
        # 执行删除栈命令，stream_output=True 时逐行读取输出
        stdin, stdout, stderr = client.exec_command(docker_stack_rm_command)

        # 流式输出：逐行读取 stdout
        for line in iter(stdout.readline, ""):
            print(line, end="")  # 保持原有换行

        # 等待命令执行结束
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            error_msg = stderr.read().decode().strip()
            raise RuntimeError(f"执行删除栈命令出错: {error_msg}")
        print(f"栈 {stack_name} 删除成功")
    except Exception as e:
        raise RuntimeError(f"连接管理节点 {master_host} 时出错: {e}")
    finally:
        client.close()


# def deploy_benchmark():
#     resource_config = load_config(benchmark_config)
#     docker_stack_deploy_command = f"docker stack deploy -c {docker_compose_file} {resource_config['name']}"
#     _, err = execute_command_via_system_ssh(config["cluster"]["master"]["host"],
#                                             username,
#                                             docker_stack_deploy_command,
#                                             stream_output=True)
#     if err:
#         raise RuntimeError(f"执行部署命令出错: {err}")
#     print("等待所有副本拉起")
#     converged = False
#     waits = 0
#     while not converged:
#         for service in client.services.list():
#             command = "docker service ls --format '{{.Replicas}}' --filter 'id=" + service.id + "'"
#             out = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True,
#                                           universal_newlines=True).strip()
#             if err:
#                 raise RuntimeError(f"执行检查服务副本命令出错: {err}")
#             print("service: ", service.name, "replicas: ", out)
#             raw_replicas = out.split('(')[0].strip()

#             actual = int(raw_replicas.split('/')[0])
#             desired = int(raw_replicas.split('/')[1])
#             converged = actual == desired
#             if not converged:
#                 break

#         time.sleep(5)
#         waits += 1
#         if waits > 30:
#             docker_stack_rm(resource_config["name"])
#             raise RuntimeError("服务副本未完全拉起，等待超时")
#     print("部署完成")


def deploy_benchmark():
    # 加载 benchmark 配置文件，得到资源配置信息
    resource_config = load_config(benchmark_config)
    stack_name = resource_config['name']
    docker_stack_deploy_command = f"docker stack deploy -c {docker_compose_file} {stack_name}"
    master_host = config["cluster"]["master"]["host"]

    # 1. 在管理节点上执行 docker stack deploy 命令
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh_client.connect(hostname=master_host, username=username, timeout=10)
        stdin, stdout, stderr = ssh_client.exec_command(docker_stack_deploy_command)
        # 流式输出部署过程
        for line in iter(stdout.readline, ""):
            print(line, end="")
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            error_msg = stderr.read().decode().strip()
            raise RuntimeError(f"执行部署命令出错: {error_msg}")
    finally:
        ssh_client.close()

    print("等待所有副本拉起")
    converged = False
    waits = 0

    # 2. 循环检查 stack 内所有服务的副本是否达到期望状态
    while not converged:
        ssh_client = paramiko.SSHClient()
        ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
        try:
            ssh_client.connect(hostname=master_host, username=username, timeout=10)
            # 使用 docker stack services 命令检查指定 stack 下各服务状态
            # 输出格式类似：service_name 1/1
            check_command = f"docker stack services {stack_name} --format '{{{{.Name}}}} {{{{.Replicas}}}}'"
            stdin, stdout, stderr = ssh_client.exec_command(check_command)
            exit_status = stdout.channel.recv_exit_status()
            if exit_status != 0:
                error_msg = stderr.read().decode().strip()
                raise RuntimeError(f"执行检查服务副本命令出错: {error_msg}")
            output = stdout.read().decode().strip()

            all_converged = True
            # 对每一行进行解析，检查副本状态
            for line in output.splitlines():
                parts = line.split()
                if len(parts) < 2:
                    continue
                service_name, replicas = parts[0], parts[1]
                try:
                    actual_str, desired_str = replicas.split('/')
                    actual = int(actual_str)
                    desired = int(desired_str)
                except Exception as parse_err:
                    raise RuntimeError(f"解析服务 {service_name} 副本信息出错: {parse_err}")
                print(f"service: {service_name} replicas: {replicas}")
                if actual != desired:
                    all_converged = False
                    break
            converged = all_converged
        finally:
            ssh_client.close()

        if not converged:
            time.sleep(5)
            waits += 1
            if waits > 30:
                docker_stack_rm(stack_name)
                raise RuntimeError("服务副本未完全拉起，等待超时")
    print("部署完成")


# 初始化socialnetwork数据
# def init_socialnetwork_data():
#     bench_dir = args.bench_dir
#     script_path = os.path.join(bench_dir, "scripts", "setup_social_graph_init_data_sync.py")
#     command = f"python3 {script_path} {bench_dir}/datasets/social-graph/socfb-Reed98/socfb-Reed98.mtx"
#     _, err = execute_command_via_system_ssh(config["cluster"]["master"]["host"], username, command, stream_output=True)
#     if err:
#         raise RuntimeError(f"执行初始化数据命令出错: {err}")
#     print("初始化数据完成")


def init_socialnetwork_data():
    bench_dir = args.bench_dir
    script_path = os.path.join(bench_dir, "scripts", "setup_social_graph_init_data_sync.py")
    data_path = os.path.join(bench_dir, "datasets", "social-graph", "socfb-Reed98", "socfb-Reed98.mtx")
    command = f"python3 {script_path} {data_path}"
    master_host = config["cluster"]["master"]["host"]

    # 使用 Paramiko 进行远程命令执行
    ssh_client = paramiko.SSHClient()
    ssh_client.set_missing_host_key_policy(paramiko.AutoAddPolicy())
    try:
        ssh_client.connect(hostname=master_host, username=username, timeout=10)
        stdin, stdout, stderr = ssh_client.exec_command(command)

        # 流式输出命令执行过程
        for line in iter(stdout.readline, ""):
            print(line, end="")

        # 检查命令执行结果
        exit_status = stdout.channel.recv_exit_status()
        if exit_status != 0:
            error_msg = stderr.read().decode().strip()
            raise RuntimeError(f"执行初始化数据命令出错: {error_msg}")
        print("初始化数据完成")
    except Exception as e:
        raise RuntimeError(f"初始化数据过程出错: {e}")
    finally:
        ssh_client.close()


# 初始化benchmark数据
def init_data():
    bench_name = args.benchmark_name
    if bench_name == "socialnetwork":
        init_socialnetwork_data()


def test_init():
    print(init_master())


def test_dissolve():
    dissolve_cluster()


def test_setup():
    init_master()
    setup_swarm_cluster()


def test_check_node_label():
    check_node_label()


if __name__ == "__main__":
    # test_dissolve()
    # test_setup()
    # deploy_benchmark()
    docker_stack_rm(benchmark_name)
    dissolve_cluster()
    init_master()
    setup_swarm_cluster()
    deploy_benchmark()
    init_data()
    pass
