import time
import pandas as pd
import numpy as np

project_root = "~/DeepDynamicRM"

# 定义文件路径
log_path = f"{project_root}/locust/locust_log_stats_history.csv"

# 定义需要提取的字段（对应CSV列名）
percentile_columns = [
    '90%',  # 第90百分位延迟
    '95%',  # 第95百分位延迟
    '98%',  # 第98百分位延迟
    '99%',  # 第99百分位延迟
    '99.9%'  # 第99.9百分位延迟
]


def get_latest_latency():
    # 读取CSV文件
    df = pd.read_csv(log_path)

    # 提取最后一行数据
    last_row = df.iloc[-1]

    # 选择目标字段并转为numpy数组
    latency_data = last_row[percentile_columns].astype(float).values

    return latency_data


if __name__ == "__main__":
    # 加载数据
    latency_data = get_latest_latency()

    # 每秒获取并打印延迟数据
    while True:
        latency_data = get_latest_latency()
        print(f"当前延迟数据(90%,95%,98%,99%,99.9%): {latency_data}")
        time.sleep(1)
