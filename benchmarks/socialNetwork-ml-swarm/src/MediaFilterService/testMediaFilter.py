#!/usr/bin/env python
# -*- coding: utf-8 -*-

import sys
import glob
import base64
import os

# 添加生成的 Thrift 代码路径
sys.path.append('./gen-py')

from social_network import MediaFilterService # type: ignore
from thrift.transport import TSocket
from thrift.transport import TTransport
from thrift.protocol import TBinaryProtocol

def load_base64_images(directory_path):
    """
    读取指定目录下所有 .jpg 和 .png 文件中的 Base64 字符串。
    返回一个包含所有 Base64 字符串的列表。
    """
    base64_images = []
    # 支持 .jpg 和 .png 文件
    for file_path in glob.glob(os.path.join(directory_path, '*.jpg')) + glob.glob(os.path.join(directory_path, '*.png')):
        try:
            with open(file_path, 'r') as f:
                base64_str = f.read().strip()
                if base64_str:
                    base64_images.append(base64_str)
                else:
                    print("警告：文件 {} 为空，跳过。".format(file_path))
        except Exception as e:
            print("错误：无法读取文件 {}: {}".format(file_path, e))
    return base64_images

def main():
    # 设置服务器地址和端口
    server_host = 'localhost'  # 如果客户端在不同机器上，请替换为服务器的IP地址
    server_port = 40000

    # 创建 Thrift 传输和协议
    transport = TSocket.TSocket(server_host, server_port)
    transport = TTransport.TFramedTransport(transport)  # 使用 TFramedTransport
    protocol = TBinaryProtocol.TBinaryProtocol(transport)

    # 创建客户端
    client = MediaFilterService.Client(protocol)

    # 打开传输
    try:
        transport.open()
    except Exception as e:
        print("错误：无法连接到 Thrift 服务器 {}:{} - {}".format(server_host, server_port, e))
        sys.exit(1)

    # 准备请求数据
    req_id = 1  # 请求 ID，可以根据需要递增
    media_types = ['image']  # 媒体类型列表，根据实际需求调整
    carrier = 'test_carrier'  # 载体信息，可以根据需要调整

    # 读取 Base64 图像
    images_directory = '/home/tomly/sinan-local/locust/base64_images'
    medium = load_base64_images(images_directory)

    if not medium:
        print("错误：未找到任何有效的 Base64 图像文件。")
        transport.close()
        sys.exit(1)

    try:
        # 发送 UploadMedia 请求
        print("发送 UploadMedia 请求，包含 {} 张图像...".format(len(medium)))
        response = client.UploadMedia(req_id, media_types, medium, carrier)
        
        # 处理响应
        print("收到响应：", response)
    except Exception as e:
        print("错误：在发送 UploadMedia 请求时发生异常 - {}".format(e))
    finally:
        # 关闭传输
        transport.close()

if __name__ == '__main__':
    main()
