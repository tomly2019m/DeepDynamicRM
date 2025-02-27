import re


# 解析命令行的输出 得到加入docker集群的命令
def parse_swarm_output(output):
    """
    解析 Docker Swarm 初始化输出，提取加入集群的命令
    :param output: Docker Swarm 初始化输出字符串
    :return: 包含 worker 和 manager 加入命令的字典
    """
    result = {}

    # 使用正则表达式提取 worker 加入命令
    worker_command_pattern = r"docker swarm join --token\s+([^\n]+)"
    worker_command_match = re.search(worker_command_pattern, output)
    if worker_command_match:
        result[
            'worker_command'] = f"docker swarm join --token {worker_command_match.group(1)}"

    # 使用正则表达式提取 manager 加入命令的提示
    manager_command_pattern = r"docker swarm join-token manager"
    if manager_command_pattern in output:
        result[
            'manager_command_hint'] = "Run 'docker swarm join-token manager' and follow the instructions."

    return result


# 解析docker ps 的输出，得到服务名
# 输入样例：socialnetwork_media-service.1.pfjrt565lfwu1m0qvljq9fq9i
# 输出样例：media-service
def parse_service_name(docker_ps_names: str) -> str:
    return docker_ps_names.split("_")[1].split(".")[0]


# 解析节点标签
def parse_node_label(output: str) -> dict:
    # 字符串按行分割
    lines = output.strip().split("\n")
    # 存储解析结果的字典
    parsed_labels = {}
    for line in lines:
        # 跳过空行
        if not line.strip():
            continue
        # 按冒号分割，提取节点名称和标签部分
        node, label_map = [part.strip() for part in line.split(":", 1)]
        # 提取 map[] 中的内容
        label_content = label_map.strip()
        if label_content.startswith("map[") and label_content.endswith("]"):
            label_content = label_content[4:-1]  # 移除 "map[" 和 "]"

        # 将标签部分解析为字典
        labels = {}
        if label_content:
            # 处理多个标签的情况
            label_pairs = label_content.split(" ")
            for pair in label_pairs:
                if ":" in pair:
                    key, value = pair.split(":", 1)
                    labels[key.strip()] = value.strip()

        # 将解析结果存储到字典中
        parsed_labels[node] = labels

    return parsed_labels


def test_parse_sawrm_output():
    swarm_output = """
    Swarm initialized: current node (mdkewmt2yr35m4h6p047zi3zp) is now a manager.

    To add a worker to this swarm, run the following command:

    docker swarm join --token SWMTKN-1-1fq19wz4ua2p1cruuc7dt7zqb7tepheaq27yaotpm5j3uwb09d-bk4yjglfzvv99uf5dp01x7ybt 172.110.0.103:2377

    To add a manager to this swarm, run 'docker swarm join-token manager' and follow the instructions.
    """
    result = parse_swarm_output(swarm_output)
    print(result)


def test_parse_node_label():
    node_label_output = """debian1: map[type:compute]
    debian2: map[type:compute]
    ubuntu2: map[type:data]"""
    result = parse_node_label(node_label_output)
    print(result)


if __name__ == "__main__":
    # test_parse_sawrm_output()
    test_parse_node_label()
