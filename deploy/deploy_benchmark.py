import argparse
import os
import subprocess
import time
import docker
import json
from util.ssh import execute_command_via_system_ssh
from util.parser import parse_swarm_output, parse_node_label

join_cluster_command = ''
client = docker.from_env()

parser = argparse.ArgumentParser()
parser.add_argument("--docker_compose",
                    type=str,
                    default="~/DeepDynamicRM/benchmarks/socialNetwork-ml-swarm/docker-compose-swarm.yml",
                    help="benchmark yaml file path")
parser.add_argument("--bench_dir", type=str, default="~/DeepDynamicRM/benchmarks/socialNetwork-ml-swarm/", help="benchmark data dir")
parser.add_argument("--benchmark_config", type=str, default="./config/socialnetwork.json", help="benchmark config file path")
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
def init_master():
    global join_cluster_command
    swarm_init_command = "docker swarm init"
    master_host = config["cluster"]["master"]["host"]
    result, err = execute_command_via_system_ssh(master_host, username, swarm_init_command)
    if err:
        raise RuntimeError(f"执行初始化命令出错: {err}")

    join_cluster_command = parse_swarm_output(result)["worker_command"]
    assert "token" in join_cluster_command
    print(join_cluster_command)
    print("初始化管理节点完成")


# 解散docker swarm集群
def dissolve_cluster():
    for worker in config["cluster"]["workers"]:
        leave_command = "docker swarm leave"
        _, err = execute_command_via_system_ssh(worker["host"], username, leave_command)
        if err:
            raise RuntimeError(f"{worker['name']}执行离开集群命令出错: {err}")
        print(f"工作节点 {worker['name']} 离开集群成功")
    master_host = config["cluster"]["master"]["host"]
    leave_command = "docker swarm leave -f"
    _, err = execute_command_via_system_ssh(master_host, username, leave_command)
    if err:
        raise RuntimeError(f"管理节点 {master_host} 执行离开集群命令出错: {err}")
    print(f"管理节点 {master_host} 离开集群成功")


# 配置节点标签
def config_node_label(node_name: str, label: str):
    config_node_label_command = f"docker node update --label-add {label} {node_name}"
    result, err = execute_command_via_system_ssh(config["cluster"]["master"]["host"], username, config_node_label_command)
    if err:
        raise RuntimeError(f"执行配置节点标签命令出错: {err}")
    print(result)


# 检查节点标签,确保节点标签与配置文件一致
def check_node_label() -> bool:
    check_command = 'docker node inspect --format "{{ .Description.Hostname }}: {{ .Spec.Labels }}" $(docker node ls -q)'
    result, err = execute_command_via_system_ssh(config["cluster"]["master"]["host"], username, check_command)
    if err:
        raise RuntimeError(f"执行检查节点标签命令出错: {err}")
    print(result)
    nodes_label = parse_node_label(result)
    master_label = nodes_label[config["cluster"]["master"]["name"]]["type"]
    if master_label not in config["cluster"]["master"]["label"]:
        raise RuntimeError(f"管理节点标签与配置文件不一致: {master_label}")
    for worker in config["cluster"]["workers"]:
        worker_label = nodes_label[worker["name"]]["type"]
        if worker_label not in worker["label"]:
            raise RuntimeError(f"工作节点标签与配置文件不一致: {worker_label}")
    return True


# 初始化docker swarm集群
def setup_swarm_cluster():
    for worker in config["cluster"]["workers"]:
        _, err = execute_command_via_system_ssh(worker["host"], username, join_cluster_command)
        if err:
            raise RuntimeError(f"执行加入集群命令出错: {err}")
        print(f"工作节点 {worker['name']} 加入集群成功")
    print("集群初始化完成")
    cluster_info_command = "docker node ls"
    result, err = execute_command_via_system_ssh(config["cluster"]["master"]["host"], username, cluster_info_command)
    if err:
        raise RuntimeError(f"执行集群信息命令出错: {err}")
    print(result)

    # 配置节点标签
    config_node_label(config["cluster"]["master"]["name"], config["cluster"]["master"]["label"])
    for worker in config["cluster"]["workers"]:
        config_node_label(worker["name"], worker["label"])
    print("节点标签配置完成")
    if check_node_label():
        print("节点标签检查完成")
    else:
        raise RuntimeError("节点标签检查失败")


def docker_stack_rm(stack_name: str):
    docker_stack_rm_command = f"docker stack rm {stack_name}"
    _, err = execute_command_via_system_ssh(config["cluster"]["master"]["host"], username, docker_stack_rm_command, stream_output=True)
    if err:
        raise RuntimeError(f"执行删除栈命令出错: {err}")
    print(f"栈 {stack_name} 删除成功")


def deploy_benchmark():
    resource_config = load_config(benchmark_config)
    docker_stack_deploy_command = f"docker stack deploy -c {docker_compose_file} {resource_config['name']}"
    _, err = execute_command_via_system_ssh(config["cluster"]["master"]["host"], username, docker_stack_deploy_command, stream_output=True)
    if err:
        raise RuntimeError(f"执行部署命令出错: {err}")
    print("等待所有副本拉起")
    converged = False
    waits = 0
    while not converged:
        for service in client.services.list():
            command = "docker service ls --format '{{.Replicas}}' --filter 'id=" + service.id + "'"
            out = subprocess.check_output(command, stderr=subprocess.STDOUT, shell=True, universal_newlines=True).strip()
            if err:
                raise RuntimeError(f"执行检查服务副本命令出错: {err}")
            print("service: ", service.name, "replicas: ", out)
            raw_replicas = out.split('(')[0].strip()

            actual = int(raw_replicas.split('/')[0])
            desired = int(raw_replicas.split('/')[1])
            converged = actual == desired
            if not converged:
                break

        time.sleep(5)
        waits += 1
        if waits > 30:
            docker_stack_rm(resource_config["name"])
            raise RuntimeError("服务副本未完全拉起，等待超时")
    print("部署完成")


# 初始化socialnetwork数据
def init_socialnetwork_data():
    bench_dir = args.bench_dir
    script_path = os.path.join(bench_dir, "scripts", "setup_social_graph_init_data_sync.py")
    command = f"python3 {script_path} {bench_dir}/datasets/social-graph/socfb-Reed98/socfb-Reed98.mtx"
    _, err = execute_command_via_system_ssh(config["cluster"]["master"]["host"], username, command, stream_output=True)
    if err:
        raise RuntimeError(f"执行初始化数据命令出错: {err}")
    print("初始化数据完成")


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
