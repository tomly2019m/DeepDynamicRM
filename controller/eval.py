import argparse
import asyncio
import csv
import json
import os
import random
import sys
import time
from utils import LinearSchedule, str2bool
from copy import deepcopy
import paramiko

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)
from deploy.util.ssh import execute_command_via_system_ssh

import numpy as np
import torch
from SACD import SACD_agent
from controller.utils import ReplayBuffer
from env import Env
from communication.sync import distribute_project
from mylocust.util.get_latency_data import get_latest_latency


def parse_args():
    parser = argparse.ArgumentParser(description='SACD算法评估参数配置')

    # ================== 环境特征参数 ==================
    parser.add_argument('--service-num', type=int, default=28, help='服务数量 (默认: 28)')
    parser.add_argument('--service-feat-dim', type=int, default=26, help='服务特征维度 (默认: 26)')
    parser.add_argument('--latency-feat-dim', type=int, default=6, help='延迟特征维度 (默认: 6)')
    parser.add_argument('--time-steps', type=int, default=30, help='时间序列长度 (默认: 30)')
    parser.add_argument('--seed', type=int, default=1024, help='随机种子 (默认: 1024)')
    parser.add_argument('--exp-noise', type=float, default=0.02, help='探索噪声 (默认: 0.02)')

    # ================== 网络结构参数 ==================
    parser.add_argument('--hidden-dim', type=int, default=128, help='编码器隐藏层维度 (默认: 128)')
    parser.add_argument('--fc-width', type=int, default=256, help='全连接层宽度 (默认: 256)')

    # ================== 训练超参数 ==================
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子 (默认: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, help='目标网络软更新系数 (默认: 0.005)')
    parser.add_argument('--lr', type=float, default=1e-4, help='统一学习率 (默认: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=256, help='训练批次大小 (默认: 256)')
    parser.add_argument('--adaptive-alpha', action='store_true', help='启用自动熵系数调整 (默认: False)')
    parser.add_argument('--alpha', type=float, default=0.2, help='init alpha')
    parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='使用自适应alpha调整')
    parser.add_argument('--action-dim', type=int, default=8, help='动作维度 (默认: 8)')
    parser.add_argument('--dvc', type=str, default="cuda", help='设备 (默认: cuda)')

    # ================== 评估参数 ==================
    parser.add_argument('--model-path', type=str, default="./model/socialnetwork", help='模型路径')
    parser.add_argument('--model-step', type=int, default=72000, help='要加载的模型步数')
    parser.add_argument('--eval-episodes', type=int, default=3, help='评估的episode数量')
    parser.add_argument('--eval-episode-steps', type=int, default=500, help='评估的episode步数')

    # ================== 运行模式 ==================
    parser.add_argument('--username', type=str, default="tomly", help='用户名 (默认: tomly)')
    parser.add_argument('--locustfile_name',
                        type=str,
                        default="socialnetwork_constant",
                        help='locustfile名称 (默认: socialnetwork)')
    parser.add_argument('--user_count', type=int, default=250, help='用户数量 (默认: 50)')

    return parser.parse_args()


def seed_everything(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True


async def main(args):
    seed_everything(args.seed)

    # 初始化环境，创建slave连接
    env = Env()
    await env.create_connections()
    connections = env.connections

    # 重置实验环境
    env.reset_benchmark()

    # 初始化智能体
    agent = SACD_agent(**vars(args))

    # 加载训练好的模型
    # 模型文件名格式：sacd_actor_20250313_065721_72000.pth
    model_dir = args.model_path
    model_time = "20250313_065721"  # 从文件名提取的时间戳
    model_step = args.model_step  # 72000
    print(f"加载模型: 时间戳 {model_time}, 步数 {model_step}")
    agent.load(model_time, model_step, model_dir)
    print("模型加载成功！")

    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    eval_data_path = f"./eval_data/{time_str}/"
    if not os.path.exists(eval_data_path):
        os.makedirs(eval_data_path)

    # 创建评估结果文件
    eval_results_path = os.path.join(eval_data_path, "eval_results.csv")
    with open(eval_results_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'total_reward', 'avg_latency_90', 'avg_latency_95', 'avg_latency_99'])

    try:
        all_total_rewards = []
        all_latency_metrics = {'90%': [], '95%': [], '98%': [], '99%': [], '99.9%': []}

        for episode_num in range(args.eval_episodes):
            # 先执行初始化分配
            cpu_allocate = deepcopy(env.initial_allocation)
            for service in cpu_allocate:
                cpu_allocate[service] /= env.replica_dict[service]
            for connection in connections.values():
                connection.send_command_sync(f"update{json.dumps(cpu_allocate)}")

            # 重置环境
            state, latency = await env.reset_eval(args.user_count, args.locustfile_name)
            done = False

            services = list(env.allocate_dict.keys())
            episode_dir = os.path.join(eval_data_path, f"episode{episode_num:03d}")
            os.makedirs(episode_dir, exist_ok=True)

            # 初始化step记录文件
            step_csv_path = os.path.join(episode_dir, "step_data.csv")
            with open(step_csv_path, 'w', newline='') as f:
                writer = csv.writer(f)
                # 构建表头：公共字段 + 服务字段
                header = ['step', 'action', 'reward', 'total_cpu', 'rps', '90%', '95%', '98%', '99%', '99.9%'] + \
                        [f'{s}_cpu' for s in services]
                writer.writerow(header)

            episode_step = 0
            total_reward = 0
            episode_latencies = {'90%': [], '95%': [], '98%': [], '99%': [], '99.9%': []}

            while not done:
                start_time = time.time()
                action = agent.select_action(state, latency, deterministic=False)

                # 执行动作
                next_state, next_latency, reward, done = env.step(action)
                raw_latency = get_latest_latency()

                # 记录性能指标
                for i, percentile in enumerate(['90%', '95%', '98%', '99%', '99.9%']):
                    episode_latencies[percentile].append(raw_latency[i])

                print(
                    f"Episode {episode_num+1}, Step {episode_step}, Action: {action}, Reward: {reward}, Latency: {raw_latency}"
                )

                state = next_state
                latency = next_latency
                if np.isnan(reward):
                    print(f"reward is nan, skip this step")
                    continue

                total_reward += reward

                original_cpu_allocate = deepcopy(env.allocate_dict)
                total_cpu = sum(original_cpu_allocate.values())

                # 记录step数据
                with open(step_csv_path, 'a', newline='') as f:
                    writer = csv.writer(f)
                    row = [
                        episode_step,
                        action,
                        reward,
                        total_cpu,
                        *raw_latency,  # 展开延迟特征
                        *[original_cpu_allocate[s] for s in services]  # 各服务CPU分配
                    ]
                    writer.writerow(row)

                # 转化为每replica的cpu分配
                cpu_allocate = deepcopy(env.allocate_dict)
                print(f"cpu_allocate: {cpu_allocate}")
                print(f"总cpu分配: {sum(cpu_allocate.values())}")
                for service in cpu_allocate:
                    cpu_allocate[service] /= env.replica_dict[service]
                for connection in connections.values():
                    connection.send_command_sync(f"update{json.dumps(cpu_allocate)}")

                episode_step += 1

                # 每个episode评估500步
                if episode_step == args.eval_episode_steps:
                    break

                # 如果时间小于1秒，则等待
                elapsed_time = time.time() - start_time
                print(f"elapsed_time: {elapsed_time}")
                if elapsed_time < 1:
                    await asyncio.sleep(1 - elapsed_time)
                else:
                    pass

            # 计算每个episode的平均延迟指标
            avg_latencies = {k: np.mean(v) for k, v in episode_latencies.items()}

            # 记录评估结果
            with open(eval_results_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(
                    [episode_num, total_reward, avg_latencies['90%'], avg_latencies['95%'], avg_latencies['99%']])

            # 收集统计数据
            all_total_rewards.append(total_reward)
            for k, v in avg_latencies.items():
                all_latency_metrics[k].append(v)

            print(f"Episode {episode_num+1} 完成, 总奖励: {total_reward}")
            print(f"平均延迟指标: {avg_latencies}")

        # 输出总体评估结果
        print("\n========== 评估结果汇总 ==========")
        print(f"平均总奖励: {np.mean(all_total_rewards):.4f} ± {np.std(all_total_rewards):.4f}")
        for k, v in all_latency_metrics.items():
            print(f"平均{k}延迟: {np.mean(v):.4f} ± {np.std(v):.4f}ms")

        # 保存总体评估结果
        summary_path = os.path.join(eval_data_path, "eval_summary.json")
        summary = {
            "model_path": args.model_path,
            "model_step": args.model_step,
            "episodes": args.eval_episodes,
            "avg_reward": float(np.mean(all_total_rewards)),
            "std_reward": float(np.std(all_total_rewards)),
            "latency_metrics": {
                k: {
                    "mean": float(np.mean(v)),
                    "std": float(np.std(v))
                }
                for k, v in all_latency_metrics.items()
            }
        }
        with open(summary_path, 'w') as f:
            json.dump(summary, f, indent=4)

        print(f"评估结果已保存至 {eval_data_path}")

    except Exception as e:
        raise e
    finally:
        env.stop_locust()
        for connection in connections.values():
            connection.send_command_sync("close")
            connection.close()


def setup_slave():
    import paramiko

    hosts = ["rm1", "rm2", "rm3", "rm4"]
    port = 12345
    username = "tomly"

    # 在每个slave节点上启动监听服务
    for host in hosts:
        # 创建SSH客户端
        client = paramiko.SSHClient()
        client.set_missing_host_key_policy(paramiko.AutoAddPolicy())

        try:
            # 连接到远程服务器
            client.connect(hostname=host, username=username)

            # 清理旧的进程
            command = f"sudo kill -9 $(sudo lsof -t -i :{port})"
            client.exec_command(command)

            # 启动监听服务
            command = ("cd ~/DeepDynamicRM/communication && "
                       "nohup ~/miniconda3/envs/DDRM/bin/python3 "
                       f"slave.py --port {port} > /dev/null 2>&1 &")
            client.exec_command(command)

            print(f"在 {host} 上启动监听服务,端口:{port}")
        except Exception as e:
            print(f"连接到 {host} 时出错: {str(e)}")


def test_setup_slave():
    import socket
    setup_slave()
    s = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
    try:
        s.connect(('rm1', 12345))  # 替换实际地址和端口
    except socket.error as e:
        print(f"连接失败: {e}")
        s.close()
        return

    try:
        # 发送数据前检查套接字状态
        if s.fileno() == -1:
            print("套接字无效")
            return

        s.sendall(b"close\n\n\n\n")
        print("指令发送成功")
    except OSError as e:
        print(f"发送失败: {e}")
    finally:
        s.close()


if __name__ == "__main__":
    # 启动slave节点
    args = parse_args()
    setup_slave()
    time.sleep(5)
    print("评估参数配置：")
    print(f"模型路径: {args.model_path}")
    print(f"模型步数: {args.model_step}")
    print(f"评估轮数: {args.eval_episodes}")
    print(f"动作维度: {args.action_dim}")
    print(f"服务特征维度: {args.service_feat_dim}")
    print(f"延迟特征维度: {args.latency_feat_dim}")
    print(f"时间序列长度: {args.time_steps}")
    asyncio.run(main(args))
