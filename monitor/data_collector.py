import docker
from cmd import execute_command

running_container_list = []


# 获取当前节点正在运行的容器列表
def set_running_container_list():
    global running_container_list
    command = "docker ps"
    _, err = execute_command(command, stream_output=True)
    if err:
        raise RuntimeError(f"获取正在运行的容器列表失败: {err}")
    print("获取正在运行的容器列表成功")

    # TODO:
    # 一个服务可能包含了多个容器，需要再定义一个map来存储服务和容器的关系
    # 服务名从配置文件中获取

def get_running_container_info():
    pass


def main():
    pass


if __name__ == "__main__":
    main()