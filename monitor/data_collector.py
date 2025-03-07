from collections import OrderedDict
from concurrent.futures import ThreadPoolExecutor
from copy import deepcopy
import json
import math
import os
import subprocess
import time
import shlex
import sys
from typing import List
import numpy as np
import docker

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from monitor.shell import execute_command, execute_command_async
from deploy.util.parser import parse_service_name

running_container_list = []

# 服务名 -> 容器名列表
service_container: dict[str, list[str]] = {}

# 服务名列表
services: list[str] = []

# 可分配的服务列表
scalable_service: list[str] = []

# 容器名 -> 容器id
container_name_id: dict[str, str] = {}

# 容器id -> 容器pid
container_id_pid: dict[str, str] = {}

# 容器cpu_time 用于保存每个容器总的cpu时间，从而计算在一段时间内容器占用的CPU时间
container_id_total_cpu: dict[str, float] = {}

# 容器id -> 容器总io量和io次数，用于计算在一段时间内容器占用的io量 (int, int)->(io量, io次数)
container_id_total_io: dict[str, tuple[int, int]] = {}

# 容器id -> 容器总网络接收量和网络发送量，用于计算在一段时间内容器占用的网络量 (int, int)->(接收量, 发送量)
container_id_total_network: dict[str, tuple[int, int]] = {}

# 数据采集指标列表
metrics = [
    "cpu_usage", "memory_usage", "io_write", "io_read", "network_recv",
    "network_send"
]

# 数据采集间隔 单位：秒
collect_interval = 1

benchmark_name = "socialnetwork"

client = docker.from_env()


def load_services():
    global services, service_container, container_name_id, container_id_pid, scalable_service
    with open(f"{PROJECT_ROOT}/deploy/config/socialnetwork.json", "r") as f:
        config = json.load(f)
        # 从配置文件中获取所有服务名
        service_dict = config["service"]
        services = config["service_list"]
        scalable_service = config["scalable_service"]
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


def set_running_container_list_via_docker_api():
    """使用Docker API高效获取容器列表，保持原有数据结构"""
    global running_container_list, service_container, services, benchmark_name, client

    # 清空全局容器列表
    running_container_list = []
    service_container = {s: [] for s in services}  # 保持原有结构

    try:
        containers = client.containers.list(filters={"status": "running"})

        # 防御性编程：添加空值检查
        valid_containers = [
            c for c in containers if getattr(c, 'name', None) is not None
        ]
        invalid_count = len(containers) - len(valid_containers)
        if invalid_count > 0:
            print(f"警告：发现{invalid_count}个无名容器，已自动过滤")

        # 提取容器信息
        for container in containers:
            try:
                # 处理名称格式（兼容Docker不同版本）
                raw_name = container.name
                container_name = raw_name.lstrip(
                    '/') if raw_name else f"unnamed_{container.id[:12]}"

                # 过滤逻辑（增加异常捕获）
                if benchmark_name in container_name:
                    running_container_list.append(container_name)

                    # 服务名称解析（防止解析异常）
                    service_name = parse_service_name(
                        container_name) if container_name else "unknown"
                    if service_name in service_container:
                        service_container[service_name].append(container_name)
                    else:
                        print(f"未注册服务：{service_name} 容器：{container_name}")

            except Exception as e:
                print(f"处理容器{container.id}时发生异常：{str(e)}")
                continue

        # print(f"成功获取{len(running_container_list)}个运行容器")

    except docker.errors.APIError as e:
        print(f"Docker API请求失败：{e.explanation}")
    except Exception as e:
        print(f"未知错误：{str(e)}")


def set_running_container_list_subprocess() -> None:
    """使用subprocess高效获取容器列表，兼容原有全局变量"""
    global running_container_list, service_container, services, benchmark_name

    # 清空全局容器列表
    running_container_list = []
    service_container = {s: [] for s in services}

    try:
        # 构建安全命令 (使用格式化输出减少解析复杂度)
        cmd = shlex.split("docker ps --filter 'status=running' "
                          "--format '{{.ID}}|{{.Names}}' --no-trunc")

        # 执行命令并捕获输出
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5  # 设置超时防止卡死
        )

        # 解析输出
        containers: List[str] = result.stdout.splitlines()
        if not containers:
            print("没有运行中的容器")
            return

        # 处理每行数据
        for line in containers:
            # 安全分割字段
            if '|' not in line:
                print(f"异常数据行: {line}")
                continue

            cid, raw_name = line.split('|', 1)
            container_name = raw_name.strip()

            # 空名称处理
            if not container_name:
                container_name = f"unnamed_{cid[:12]}"

            # 应用过滤逻辑
            if benchmark_name in container_name:
                running_container_list.append(container_name)

                # 服务分类
                service_name = parse_service_name(container_name)
                if service_name in service_container:
                    service_container[service_name].append(container_name)

        print(f"成功获取 {len(running_container_list)} 个容器")

    except subprocess.CalledProcessError as e:
        error_msg = f"命令执行失败: {e.stderr.strip()}" if e.stderr else "未知错误"
        raise RuntimeError(f"{error_msg}\n命令: {e.cmd}") from None
    except subprocess.TimeoutExpired:
        raise RuntimeError("容器列表获取超时(5秒)")
    except Exception as e:
        raise RuntimeError(f"未知错误: {str(e)}") from e


# 依据容器名获取容器id
def get_container_id(container_name: str) -> str:
    command = f"docker inspect -f '{{{{.Id}}}}' {container_name}"
    result, err = execute_command(command, stream_output=False)
    if err:
        raise RuntimeError(f"获取容器id失败: {err}")
    return result.strip()


def get_container_id_subprocess(container_name: str) -> str:
    """
    安全获取容器完整ID
    :param container_name: 容器名称/ID
    :return: 64位完整容器ID
    :raises ContainerNotFoundError: 当容器不存在时
    :raises RuntimeError: 其他执行错误
    """
    try:
        # 安全构造命令（避免注入攻击）
        cmd = shlex.split(
            f"docker inspect -f '{{{{.Id}}}}' {shlex.quote(container_name)}")

        # 执行命令（带超时控制）
        result = subprocess.run(
            cmd,
            check=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            timeout=5  # 超时时间可根据需求调整
        )

        # 清理输出（处理可能的换行符）
        container_id = result.stdout.strip()

        if not container_id:
            raise

        return container_id

    except subprocess.CalledProcessError as e:
        # 分析错误类型
        if "No such object" in e.stderr:
            raise
        elif "Permission denied" in e.stderr:
            raise RuntimeError("Docker权限不足，请检查用户组权限") from e
        else:
            raise RuntimeError(f"命令执行失败: {e.stderr.strip() or '未知错误'}") from e

    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"获取容器ID超时，容器: {container_name}") from e

    except FileNotFoundError:
        raise RuntimeError("Docker CLI 未安装或不在PATH中") from None


# 获取容器id
def set_container_name_id():
    global container_name_id, service_container
    for container_list in service_container.values():
        for container_name in container_list:
            container_id = get_container_id_subprocess(container_name)
            container_name_id[container_name] = container_id


# 依据容器id获取容器pid
def get_container_pid(container_id: str) -> str:
    command = f"docker inspect -f '{{{{.State.Pid}}}}' {container_id}"
    result, err = execute_command(command, stream_output=False)
    if err:
        raise RuntimeError(f"获取容器pid失败: {err}")
    return result.strip()


def get_container_pid_subprocess(container_id: str) -> int:
    """
    安全获取容器PID
    :param container_id: 容器完整ID
    :return: 容器进程PID
    :raises ContainerNotFoundError: 容器不存在时
    :raises ContainerNotRunningError: 容器未运行时
    :raises InvalidPIDError: PID无效时
    """
    try:
        # 安全构造命令（处理特殊字符）
        cmd = shlex.split(
            f"docker inspect --format '{{{{.State.Pid}}}}' {shlex.quote(container_id)}"
        )

        # 执行命令（带超时控制）
        result = subprocess.run(cmd,
                                check=True,
                                stdout=subprocess.PIPE,
                                stderr=subprocess.PIPE,
                                text=True,
                                timeout=5)

        # 提取并验证PID
        pid_str = result.stdout.strip()
        if not pid_str.isdigit():
            raise

        pid = int(pid_str)
        if pid <= 0:
            raise

        return pid

    except subprocess.CalledProcessError as e:
        # 分析错误类型
        if "No such container" in e.stderr:
            raise
        elif "is not running" in e.stderr:
            raise
        else:
            raise RuntimeError(f"命令执行失败: {e.stderr.strip() or '未知错误'}") from e

    except subprocess.TimeoutExpired as e:
        raise RuntimeError(f"获取PID超时，容器: {container_id}") from e

    except ValueError:
        raise


# 配置容器pid
def set_container_pids():
    global container_id_pid, container_name_id, service_container
    for container_list in service_container.values():
        for container_name in container_list:
            container_id = container_name_id[container_name]
            container_pid = get_container_pid_subprocess(container_id)
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
    service_cpu_time: dict[str, list[float]] = {}
    # 一个服务可能包含多个容器，需要遍历所有容器
    for service_name, container_list in service_container.items():
        service_cpu_time[service_name] = []
        if container_list != []:
            for container_name in container_list:
                container_id = container_name_id[container_name]
                assert cgroup_version == "v2"
                pseudo_file = f"/sys/fs/cgroup/system.slice/docker-{container_id}.scope/cpu.stat"
                if not os.path.exists(pseudo_file):
                    raise FileNotFoundError(f"文件不存在: {pseudo_file}")
                # print(pseudo_file)
                with open(pseudo_file, "r") as f:
                    line = f.readline()
                    # 在V2版本中，CPU计数的单位是微秒
                    cum_cpu_time = int(line.split(' ')[1])
                    previous_cpu_time = container_id_total_cpu.get(
                        container_id, 0)
                    cpu_usage_time = max(cum_cpu_time - previous_cpu_time, 0)
                    container_id_total_cpu[container_id] = cum_cpu_time
                    service_cpu_time[service_name].append(cpu_usage_time)
    return service_cpu_time


# 获取每个服务的内存使用情况
def get_memory_usage():
    global service_container, container_name_id

    # 服务名 -> 服务内存使用率列表（对应于所有replicas）
    service_memory_usage: dict[str, list[float]] = {}

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
                service_memory_usage[service].append(total_memory / 1024.0**2)
    return service_memory_usage


# 获取每个服务的io使用情况 tuple[int, int] 分别表示io量和io次数
def get_io_usage():
    global service_container, container_name_id

    # 服务名 -> 服务io使用率列表（对应于所有replicas）
    service_io_usage: dict[str, list[tuple[int, int]]] = {}

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
                    stats = {
                        k.split('=')[0]: int(k.split('=')[1])
                        for k in parts[1:]
                    }  # Parse key-value pairs

                    # Accumulate values across all devices
                    total_rbytes += stats.get('rbytes', 0)
                    total_wbytes += stats.get('wbytes', 0)
                    total_rios += stats.get('rios', 0)
                    total_wios += stats.get('wios', 0)
            total_bytes = total_rbytes + total_wbytes
            total_operations = total_rios + total_wios
            service_io_usage[service].append(
                (total_bytes - container_id_total_io.get(container_id,
                                                         (0, 0))[0],
                 total_operations -
                 container_id_total_io.get(container_id, (0, 0))[1]))
            container_id_total_io[container_id] = (total_bytes,
                                                   total_operations)
    return service_io_usage


# 获取每个服务的网络使用情况
def get_network_usage():
    global service_container, container_name_id, container_id_pid

    # 服务名 -> 服务网络使用率列表（对应于所有replicas）tuple[int, int] 分别表示网络接收量和网络发送量
    service_network_usage: dict[str, list[tuple[int, int]]] = {}

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
                    data = [
                        d for d in data
                        if (d != '' and '#' not in d and ":" not in d)
                    ]
                    recv_bytes += int(data[0])
                    send_bytes += int(data[8])
                last_recv_bytes, last_send_bytes = container_id_total_network.get(
                    container_id, (0, 0))
                service_network_usage[service].append(
                    (recv_bytes - last_recv_bytes,
                     send_bytes - last_send_bytes))
                container_id_total_network[container_id] = (recv_bytes,
                                                            send_bytes)
    return service_network_usage


# 获取运行在当前节点上的所有服务的replicas数量
def get_replicas():
    global service_container
    replicas = {}
    for service, container_list in service_container.items():
        replicas[service] = len(container_list)
    return replicas


# 计算一个列表中的最大值
def calculate_max(data: list | list[list], position: int = 0):
    if isinstance(data[0], list):
        return max(data, key=lambda x: x[position])[position]
    return max(data)


# 计算一个列表中的最小值
def calculate_min(data: list | list[list], position: int = 0):
    if isinstance(data[0], list):
        return min(data, key=lambda x: x[position])[position]
    return min(data)


# 计算一个列表中的平均值
def calculate_mean(data: list | list[list], position: int = 0):
    if isinstance(data[0], list):
        return sum(x[position] for x in data) / len(data)
    return sum(data) / len(data)


# 计算一个列表中的标准差
def calculate_std(data: list | list[list], position: int = 0):
    if isinstance(data[0], list):
        return math.sqrt(
            sum((x[position] - calculate_mean(data, position))**2
                for x in data) / len(data))
    return math.sqrt(
        sum((x - calculate_mean(data, position))**2 for x in data) / len(data))


# 将不同节点上的数据合并
def concat_data(data1: dict, data2: dict):
    for k, v in data2.items():
        data1.setdefault(k, []).extend(v)
    return data1


# 汇聚不同节点上的replicas数据 相同服务，数据相加
def gather_replicas_data(data1: dict[str, list[int]], data2: dict[str,
                                                                  list[int]]):
    for k, v in data2.items():
        if k in data1:
            data1[k] = data1[k] + data2[k]
        else:
            data1[k] = data2[k]
    return data1


# 处理数据，计算每个服务的最大值、最小值、平均值、标准差
def process_data(data: dict):
    for k, v in data.items():
        if isinstance(v[0], list):
            data[k] = [
                calculate_max(v, 0),
                calculate_min(v, 0),
                calculate_mean(v, 0),
                calculate_std(v, 0)
            ]
            data[k].extend([
                calculate_max(v, 1),
                calculate_min(v, 1),
                calculate_mean(v, 1),
                calculate_std(v, 1)
            ])
        else:
            data[k] = [
                calculate_max(v),
                calculate_min(v),
                calculate_mean(v),
                calculate_std(v)
            ]
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
            numpy_data[i, j] = service_data[j * 4:(j + 1) * 4]

    return numpy_data


load_services()


def transform_data(gathered_data):
    # 定义指标处理顺序和对应原始字段
    metric_mapping = OrderedDict([
        ('cpu_usage', ('cpu', 4)),
        ('memory_usage', ('memory', 4)),
        ('io_write', ('io', 0)),  # io前4个元素
        ('io_read', ('io', 4)),  # io后4个元素
        ('network_recv', ('network', 0)),  # network前4个
        ('network_send', ('network', 4))  # network后4个
    ])

    # 获取所有服务并保持顺序一致
    all_services = services

    # 初始化结果数组 (service_num, metric_num, 4)
    service_num = len(all_services)
    metric_num = len(metric_mapping)
    result_array = np.zeros((service_num, metric_num, 4), dtype=np.float64)

    # 填充数据
    for service_idx, service_name in enumerate(all_services):
        for metric_idx, (metric_name,
                         (src_key,
                          offset)) in enumerate(metric_mapping.items()):
            try:
                src_data = gathered_data[src_key][service_name]

                # 处理不同长度的数据
                if len(src_data) == 8:  # IO和Network数据
                    metric_data = src_data[offset:offset + 4]
                elif len(src_data) == 4:  # CPU和Memory数据
                    metric_data = src_data
                else:
                    raise ValueError(f"非预期数据长度: {len(src_data)}")

                result_array[service_idx, metric_idx] = metric_data

            except KeyError as e:
                print(f"警告: 服务 {service_name} 缺少指标 {src_key}")
                result_array[service_idx, metric_idx] = np.nan

    return result_array


# 初始化数据采集器
def init_collector():
    load_services()
    set_running_container_list_via_docker_api()
    set_container_name_id()
    set_container_pids()


# 定期更新全局变量
def flush():
    set_running_container_list_via_docker_api()
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


# 配置cpu限制 输入参数是一个字典，key是service name value是cpu limit的具体值 已经经过处理 转化为每个副本的cpu limit
def set_cpu_limit(cpu_limit: dict[str, int]):
    """并行设置容器CPU限制（无阻塞等待）"""
    global service_container, scalable_service

    # 生成所有需要执行的命令列表
    commands = [
        f"docker update --cpus={limit} {container_name}"
        for service, limit in cpu_limit.items() if service in scalable_service
        for container_name in service_container[service]
    ]

    # 使用线程池并行执行（根据CPU核心数动态调整工作线程）
    with ThreadPoolExecutor(max_workers=min(32, len(commands))) as executor:
        for cmd in commands:
            # 提交任务到线程池（不等待结果）
            executor.submit(subprocess.run,
                            cmd,
                            shell=True,
                            stdout=subprocess.DEVNULL,
                            stderr=subprocess.DEVNULL)


def test_to_numpy():
    data = {
        "serviceA": [
            1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19,
            20, 21, 22, 23, 24, 25, 26
        ],
        "serviceB": [
            26, 25, 24, 23, 22, 21, 20, 19, 18, 17, 16, 15, 14, 13, 12, 11, 10,
            9, 8, 7, 6, 5, 4, 3, 2, 1
        ]
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


def testflush():
    init_collector()
    s = time.time()
    for i in range(10):
        flush()
        print(service_container)
    print(f"{time.time() - s}")


if __name__ == "__main__":
    testflush()
