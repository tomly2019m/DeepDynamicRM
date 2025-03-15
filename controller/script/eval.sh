#!/bin/bash

# 从50到450，步长为50执行评估
for user_count in $(seq 100 50 450)
do
    echo "正在执行用户数量为 $user_count 的评估..."
    python eval.py --user_count $user_count
    echo "用户数量为 $user_count 的评估已完成"
    # 等待一段时间，确保系统恢复稳定
    sleep 10
done
