import json
import math
import os
import time

import sys
import numpy as np

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from monitor.shell import execute_command
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

# 容器cpu_time 用于保存每个容器总的cpu时间，从而计算在一段时间内容器占用的CPU时间
container_id_total_cpu: dict[str, float] = {}

# 容器id -> 容器总io量和io次数，用于计算在一段时间内容器占用的io量 (int, int)->(io量, io次数)
container_id_total_io: dict[str, tuple[int, int]] = {}

# 容器id -> 容器总网络接收量和网络发送量，用于计算在一段时间内容器占用的网络量 (int, int)->(接收量, 发送量)
container_id_total_network: dict[str, tuple[int, int]] = {}

# 数据采集指标列表
metrics = ["cpu_usage", "memory_usage", "io_write", "io_read", "network_recv", "network_send"]

# 数据采集间隔 单位：秒
collect_interval = 1

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
def get_container_cpu_usage():
    global service_container, container_name_id, container_id_total_cpu
    cgroup_version = get_cgroup_version()

    # 服务名 -> 服务cpu使用率列表（对应于所有replicas）
    service_cpu_time : dict[str, list[float]] = {}
    # 一个服务可能包含多个容器，需要遍历所有容器
    for service_name, container_list in service_container.items():
        service_cpu_time[service_name] = []
        if container_list != []:
            for container_name in container_list:
                container_id = container_name_id[container_name]
                assert cgroup_version == "v2"
                pseudo_file = f"/sys/fs/cgroup/system.slice/docker-{container_id}.scope/cpu.stat"
                with open(pseudo_file, "r") as f:
                    line = f.readline()
                    # 在V2版本中，CPU计数的单位是微秒不是纳秒
                    cum_cpu_time = int(line.split(' ')[1])/1000.0
                    previous_cpu_time = container_id_total_cpu.get(container_id, 0)
                    cpu_usage_time = max(cum_cpu_time - previous_cpu_time, 0)
                    container_id_total_cpu[container_id] = cum_cpu_time
                    service_cpu_time[service_name].append(cpu_usage_time)
    return service_cpu_time


# 获取每个服务的内存使用情况
def get_memory_usage():
    global service_container, container_name_id

    # 服务名 -> 服务内存使用率列表（对应于所有replicas）
    service_memory_usage : dict[str, list[float]] = {}

    cgroup_version = get_cgroup_version()
    for service, container_list in service_container.items():
        service_memory_usage[service] = []
        for container_name in container_list:
            container_id = container_name_id[container_name]
            assert cgroup_version == "v2"
            pseudo_file = f"/sys/fs/cgroup/system.slice/docker-{container_id}.scope/memory.stat"
            with open(pseudo_file, "r") as f:
                total_memory = 0
                for line in f:
                    parts = line.split()
                    if parts:
                        key = parts[0]
                        value = int(parts[1])
                        if key in ["anon", "file", "kernel"]:
                            total_memory += value
                # 单位转换为MB
                service_memory_usage[service].append(total_memory / 1024.0 ** 2)
    return service_memory_usage


# 获取每个服务的io使用情况 tuple[int, int] 分别表示io量和io次数
def get_io_usage():
    global service_container, container_name_id

    # 服务名 -> 服务io使用率列表（对应于所有replicas）
    service_io_usage : dict[str, list[tuple[int, int]]] = {}

    cgroup_version = get_cgroup_version()
    for service, container_list in service_container.items():
        service_io_usage[service] = []
        for container_name in container_list:
            container_id = container_name_id[container_name]
            assert cgroup_version == "v2"
            pseudo_file = f"/sys/fs/cgroup/system.slice/docker-{container_id}.scope/io.stat"
            if not os.path.exists(pseudo_file):
                print(f"容器{container_name}的io.stat文件不存在")
                continue
            with open(pseudo_file, "r") as f:
                lines = f.readlines()
                total_rbytes, total_wbytes, total_rios, total_wios = 0, 0, 0, 0
                for line in lines:
                    parts = line.strip().split()
                    if len(parts) < 2:
                        continue  # Skip invalid lines
                    stats = {k.split('=')[0]: int(k.split('=')[1]) for k in parts[1:]}  # Parse key-value pairs

                    # Accumulate values across all devices
                    total_rbytes += stats.get('rbytes', 0)
                    total_wbytes += stats.get('wbytes', 0)
                    total_rios += stats.get('rios', 0)
                    total_wios += stats.get('wios', 0)
            total_bytes = total_rbytes + total_wbytes
            total_operations = total_rios + total_wios
            service_io_usage[service].append((total_bytes - container_id_total_io.get(container_id, (0, 0))[0], 
                                              total_operations - container_id_total_io.get(container_id, (0, 0))[1]))
            container_id_total_io[container_id] = (total_bytes, total_operations)
    return service_io_usage


# 获取每个服务的网络使用情况
def get_network_usage():
    global service_container, container_name_id, container_id_pid

    # 服务名 -> 服务网络使用率列表（对应于所有replicas）tuple[int, int] 分别表示网络接收量和网络发送量
    service_network_usage : dict[str, list[tuple[int, int]]] = {}

    for service, container_list in service_container.items():
        service_network_usage[service] = []
        for container_name in container_list:

            recv_bytes, send_bytes = 0, 0
            
            container_id = container_name_id[container_name]
            container_pid = container_id_pid[container_id]
            pseudo_file = f"/proc/{container_pid}/net/dev"
            if not os.path.exists(pseudo_file):
                print(f"容器{container_name}的net/dev文件不存在")
                continue

            with open(pseudo_file, "r") as f:
                lines = f.readlines()
                for line in lines:
                    if 'Inter-|   Receive' in line or 'face |bytes    packets errs' in line:
                        continue
                    data = line.strip().split()
                    data = [d for d in data if (d != '' and '#' not in d and ":" not in d)]
                    recv_bytes += int(data[0])
                    send_bytes += int(data[8])
                last_recv_bytes, last_send_bytes = container_id_total_network.get(container_id, (0, 0))
                service_network_usage[service].append((recv_bytes - last_recv_bytes, send_bytes - last_send_bytes))
                container_id_total_network[container_id] = (recv_bytes, send_bytes)
    return service_network_usage


# 获取运行在当前节点上的所有服务的replicas数量
def get_replicas():
    global service_container
    replicas = {}
    for service, container_list in service_container.items():
        replicas[service] = len(container_list)
    return replicas

# 计算一个列表中的最大值
def calculate_max(data: list | list[tuple], position: int):
    if isinstance(data[0], tuple):
        return max(data, key=lambda x: x[position])
    return max(data)

# 计算一个列表中的最小值
def calculate_min(data: list | list[tuple], position: int):
    if isinstance(data[0], tuple):
        return min(data, key=lambda x: x[position])
    return min(data)

# 计算一个列表中的平均值
def calculate_mean(data: list | list[tuple], position: int):
    if isinstance(data[0], tuple):
        return sum(data, key=lambda x: x[position]) / len(data)
    return sum(data) / len(data)

# 计算一个列表中的标准差
def calculate_std(data: list | list[tuple], position: int):
    if isinstance(data[0], tuple):
        return math.sqrt(sum((x[position] - calculate_mean(data, position)) ** 2 for x in data) / len(data))
    return math.sqrt(sum((x - calculate_mean(data, position)) ** 2 for x in data) / len(data))

# 将不同节点上的数据合并
def concat_data(data1: dict, data2: dict):
    for k, v in data2.items():
        dict.setdefault(k, []).extend(v)
    return data1

# 汇聚不同节点上的replicas数据 相同服务，数据相加
def gather_replicas_data(data1: dict[str, list[int]], data2: dict[str, list[int]]):
    for k, v in data2.items():
        if k in data1:
            data1[k] = data1[k] + data2[k]
        else:
            data1[k] = data2[k]
    return data1

# 处理数据，计算每个服务的最大值、最小值、平均值、标准差
def process_data(data: dict):
    for k, v in data.items():
        if isinstance(v[0], tuple):
            data[k] = [calculate_max(v, 0), calculate_min(v, 0), calculate_mean(v, 0), calculate_std(v, 0)]
            data[k].append([calculate_max(v, 1), calculate_min(v, 1), calculate_mean(v, 1), calculate_std(v, 1)])
        else:
            data[k] = [calculate_max(v), calculate_min(v), calculate_mean(v), calculate_std(v)]
    return data

# 将处理好的dict转换为numpy数组 shape = (service_num, metric_num, [max, min, mean, std] -> 4)
# dict的key为服务名，value为列表[max1, min1, mean1, std1, max2, min2, mean2, std2, ...] 
# 每4个为一组，分别表示指标[cpu_usage, memory_usage, io_write, io_read, network_recv, network_send]的max, min, mean, std
def to_numpy(data: dict):
    service_num = len(data)
    metric_num = len(metrics)
    numpy_data = np.zeros((service_num, metric_num, 4))  
    
    for i, service_name in enumerate(data):
        service_data = data[service_name] 
        
        for j in range(metric_num):
            # 每个指标的 4 个数值分别为 max, min, mean, std
            numpy_data[i, j] = service_data[j*4:(j+1)*4]

    return numpy_data


# 初始化数据采集器
def init_collector():
    load_services()
    set_running_container_list()
    set_container_name_id()
    set_container_pids()


# 定时的逻辑应该在master中，在master中定时采集数据，而不是在data_collector中
# 主函数的职责
# 1. 初始化数据采集器
# 2. 以一定间隔采集数据
# 3. 发送数据
# def main():
#     init_collector()
#     while True:
#         # 定时采集逻辑
#         time.sleep(collect_interval)
#         # 存储最新数据到内存缓存
#         global latest_data
#         latest_data = {
#             "cpu": get_container_cpu_usage(),
#             "memory": get_memory_usage(),
#             "io": get_io_usage(),
#             "network": get_network_usage()
#         }



def test_to_numpy():
    data = {
        "serviceA": [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 26],
        "serviceB": [26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10, 9, 8, 7, 6, 5, 4, 3, 2, 1]
    }
    print(to_numpy(data).shape)

def test_get_network_usage():
    init_collector()
    print(get_network_usage())
    print(get_network_usage())
    print(get_network_usage())
    print(get_network_usage())
    print(get_network_usage())


def test_get_container_cpu_usage():
    init_collector()
    print(get_container_cpu_usage())
    print(get_container_cpu_usage())
    print(get_container_cpu_usage())
    print(get_container_cpu_usage())
    print(get_container_cpu_usage())

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

def test_get_memory_usage():
    init_collector()
    print(get_memory_usage())
    print(get_memory_usage())
    print(get_memory_usage())
    print(get_memory_usage())
    print(get_memory_usage())


def test_get_io_usage():
    init_collector()
    print(get_io_usage())
    print(get_io_usage())
    print(get_io_usage())
    print(get_io_usage())


if __name__ == "__main__":
    test_get_network_usage()