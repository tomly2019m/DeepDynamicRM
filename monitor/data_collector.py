import json
import os
from shell import execute_command
import sys


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from deploy.util.parser import parse_service_name

running_container_list = []

# 服务名 -> 容器名列表
service_container : dict[str, list[str]] = {}

# 服务名列表
services: list[str] = []

# 容器名 -> 容器id
container_name_id : dict[str, str] = {}

# 容器id -> 容器pid
container_id_pid : dict[str, str] = {}

# 容器cpu_time

benchmark_name = "socialnetwork"

def load_services():
    global services, service_container, container_name_id, container_id_pid
    with open(f"{PROJECT_ROOT}/deploy/config/socialnetwork.json", "r") as f:
        config = json.load(f)
        # 从配置文件中获取所有服务名
        service_dict = config["service"]
        services = list(service_dict.keys())
        # 初始化service_container
        service_container = {service: [] for service in services}
        # 初始化container_name_id
        container_name_id = {}
        # 初始化container_id_pid
        container_id_pid = {}

# 获取当前节点正在运行的容器列表
def set_running_container_list():
    global running_container_list
    command = "docker ps"
    result, err = execute_command(command, stream_output=False)
    if err:
        raise RuntimeError(f"获取正在运行的容器列表失败: {err}")
    print("获取正在运行的容器列表成功")

    # 解析结果
    for line in result.split("\n"):
        # 跳过标题行
        if "CONTAINER ID".lower() in line.lower():
            continue
        if benchmark_name in line and line.strip():
            container_name = line.split()[-1]
            service_name = parse_service_name(container_name)
            if service_name in services:
                service_container[service_name].append(container_name)

# 依据容器名获取容器id
def get_container_id(container_name: str) -> str:
    command = f"docker inspect -f '{{{{.Id}}}}' {container_name}"
    result, err = execute_command(command, stream_output=False)
    if err:
        raise RuntimeError(f"获取容器id失败: {err}")
    return result.strip()

# 获取容器id
def set_container_name_id():
    global container_name_id, service_container
    for container_list in service_container.values():
        for container_name in container_list:
            container_id = get_container_id(container_name)
            container_name_id[container_name] = container_id

# 依据容器id获取容器pid
def get_container_pid(container_id: str) -> str:
    command = f"docker inspect -f '{{{{.State.Pid}}}}' {container_id}"
    result, err = execute_command(command, stream_output=False)
    if err:
        raise RuntimeError(f"获取容器pid失败: {err}")
    return result.strip()

# 配置容器pid
def set_container_pids():
    global container_id_pid, container_name_id, service_container
    for container_list in service_container.values():
        for container_name in container_list:
            container_id = container_name_id[container_name]
            container_pid = get_container_pid(container_id)
            container_id_pid[container_id] = container_pid

# 获取cgroup版本
def get_cgroup_version():
    cgroup_type = os.popen("stat -fc %T /sys/fs/cgroup").read().strip()
    return "v2" if cgroup_type == "cgroup2fs" else "v1"

# 获取容器cpu使用率
def get_container_cpu():
    global service_container, container_name_id
    cgroup_version = get_cgroup_version()

    # 服务名 -> 服务cpu使用率列表（对应于所有replicas）
    service_cpu_time : dict[str, list[float]] = {}
    # 一个服务可能包含多个容器，需要遍历所有容器
    for service_name, container_list in service_container.items():
        if container_list != []:
            service_cpu_time[service_name] = []
            for container_name in container_list:
                container_id = container_name_id[container_name]
                assert cgroup_version == "v2"
                pseudo_file = f"/sys/fs/cgroup/system.slice/docker-{container_id}.scope/cpu.stat"
                with open(pseudo_file, "r") as f:
                    line = f.readline()
                    # 在V2版本中，CPU计数的单位是微秒不是纳秒
                    cum_cpu_time = int(line.split(' ')[1])/1000.0
                    service_cpu_time[container_name] = max(cum_cpu_time - service_cpu_time[container_name]['cpu_time'], 0)
    return service_cpu_time

def main():
    pass

def test_load_services():
    load_services()
    print(services)
    print(len(services))
    print(service_container)

def test_set_container_name_id():
    set_container_name_id()
    print(container_name_id)

def test_set_running_container_list():
    set_running_container_list()
    print(service_container)

def test_set_container_pids():
    set_container_pids()
    print(container_id_pid)

if __name__ == "__main__":
    load_services()
    test_set_running_container_list()
    test_set_container_name_id()
    test_set_container_pids()