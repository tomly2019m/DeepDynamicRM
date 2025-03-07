import os
import numpy as np


def check_npy_files():
    """检查data目录下所有npy文件的形状和数据情况"""
    data_dir = "data"

    if not os.path.exists(data_dir):
        print("错误: data目录不存在")
        return

    npy_files = [f for f in os.listdir(data_dir) if f.endswith('.npy')]

    if not npy_files:
        print("未找到任何npy文件")
        return

    print(f"找到{len(npy_files)}个npy文件:")
    print("-" * 50)

    for file in npy_files:
        file_path = os.path.join(data_dir, file)
        try:
            data = np.load(file_path)
            print(f"\n文件名: {file}")
            print(f"数据形状: {data.shape}")
            print(f"数据类型: {data.dtype}")
            print(f"数据范围: [{np.min(data)}, {np.max(data)}]")
            print(f"均值: {np.mean(data):.4f}")
            print(f"标准差: {np.std(data):.4f}")
            print(f"是否包含NaN: {np.isnan(data).any()}")
            print(f"是否包含Inf: {np.isinf(data).any()}")
        except Exception as e:
            print(f"\n读取文件 {file} 时发生错误:")
            print(str(e))
        print("-" * 50)


if __name__ == "__main__":
    check_npy_files()
