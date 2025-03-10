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

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import numpy as np
import torch
from SACD import SACD_agent
from controller.utils import ReplayBuffer
from env import Env
from communication.sync import distribute_project
from mylocust.util.get_latency_data import get_latest_latency


def parse_args():
    parser = argparse.ArgumentParser(description='SACD算法参数配置')

    # ================== 环境特征参数 ==================
    parser.add_argument('--service-num', type=int, default=28, help='服务数量 (默认: 28)')
    parser.add_argument('--service-feat-dim', type=int, default=26, help='服务特征维度 (默认: 26)')
    parser.add_argument('--latency-feat-dim', type=int, default=6, help='延迟特征维度 (默认: 6)')
    parser.add_argument('--time-steps', type=int, default=30, help='时间序列长度 (默认: 30)')
    parser.add_argument('--seed', type=int, default=1024, help='随机种子 (默认: 1024)')
    parser.add_argument('--exp-noise', type=float, default=1, help='随机探索概率')
    parser.add_argument('--init-noise', type=float, default=1, help='初始探索噪声')
    parser.add_argument('--noise-steps', type=int, default=5000, help='噪声衰减步数')
    parser.add_argument('--final-noise', type=float, default=0.02, help='最终探索噪声')

    # ================== 网络结构参数 ==================
    parser.add_argument('--hidden-dim', type=int, default=128, help='编码器隐藏层维度 (默认: 128)')
    parser.add_argument('--fc-width', type=int, default=256, help='全连接层宽度 (默认: 256)')

    # ================== 训练超参数 ==================
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子 (默认: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, help='目标网络软更新系数 (默认: 0.005)')
    parser.add_argument('--lr', type=float, default=1e-4, help='统一学习率 (默认: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=256, help='训练批次大小 (默认: 256)')
    parser.add_argument('--adaptive-alpha', action='store_true', help='启用自动熵系数调整 (默认: False)')
    parser.add_argument('--stop-steps', type=int, default=20 * 3600, help='最大训练步数 (默认: 72000)')
    parser.add_argument('--replay-size', type=int, default=int(1e6), help='回放缓冲区容量 (默认: 1e6)')
    parser.add_argument('--alpha', type=float, default=0.2, help='init alpha')
    parser.add_argument('--adaptive_alpha', type=str2bool, default=True, help='Use adaptive alpha turning')
    # TODO 修改为1000
    parser.add_argument('--random-steps', type=int, default=500, help='纯随机探索步数 (默认: 1000)')
    parser.add_argument('--action-dim', type=int, default=8, help='动作维度 (默认: 8)')
    parser.add_argument('--update-steps', type=int, default=1000, help='更新步数 (默认: 1000)')
    parser.add_argument('--dvc', type=str, default="cuda", help='设备 (默认: cuda)')

    # ================== 运行模式 ==================
    parser.add_argument('--username', type=str, default="tomly", help='用户名 (默认: tomly)')

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
    total_steps = 0
    episode_num = 0

    # 初始化环境，创建slave连接
    env = Env()
    await env.create_connections()
    connections = env.connections

    # 初始化智能体
    agent = SACD_agent(**vars(args))
    agent.exp_noise = args.init_noise
    schedualer = LinearSchedule(schedule_timesteps=args.noise_steps,
                                final_p=args.final_noise,
                                initial_p=args.init_noise)
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())

    exp_data_path = f"./exp_data/{time_str}/"
    if not os.path.exists(exp_data_path):
        os.makedirs(exp_data_path)

    total_reward_path = os.path.join(exp_data_path, "episode_rewards.csv")
    with open(total_reward_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['episode', 'total_reward'])

    try:
        while total_steps < args.stop_steps:
            # 重置环境
            state, latency = await env.reset()
            done = False

            services = list(env.allocate_dict.keys())
            episode_dir = os.path.join(exp_data_path, f"episode{episode_num:03d}")
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
            while not done:
                start_time = time.time()
                action = agent.select_action(state, latency, deterministic=False)

                # 执行动作
                next_state, next_latency, reward, done = env.step(action)
                raw_latency = get_latest_latency()
                print(f"action: {action}, reward: {reward}, latency: {raw_latency}")
                agent.replay_buffer.add_experience(state, latency, action, reward, next_state, next_latency, done)

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

                if agent.replay_buffer.current_size > args.random_steps:
                    agent.train()
                    agent.exp_noise = schedualer.value(total_steps)  # e-greedy decay

                total_steps += 1
                episode_step += 1
                if total_steps % 1000 == 0:
                    agent.save(time_str, total_steps)
                # 如果时间小于1秒，则等待
                elapsed_time = time.time() - start_time
                print(f"elapsed_time: {elapsed_time}")
                if elapsed_time < 1:
                    await asyncio.sleep(1 - elapsed_time)
                else:
                    await asyncio.sleep(1)

            with open(total_reward_path, 'a', newline='') as f:
                writer = csv.writer(f)
                writer.writerow([episode_num, total_reward])

            print(f"总奖励: {total_reward}")
            episode_num += 1
    except Exception as e:
        raise e
    finally:
        env.stop_locust()
        for connection in connections.values():
            connection.send_command_sync("close")
            connection.close()


if __name__ == "__main__":
    args = parse_args()
    # distribute_project(args.username)
    print("参数配置：")
    print(f"学习率: {args.lr:.0e}")
    print(f"批大小: {args.batch_size}")
    print(f"随机探索步数: {args.random_steps}")
    print(f"总探索步数: {args.stop_steps}")
    print(f"动作维度: {args.action_dim}")
    print(f"服务特征维度: {args.service_feat_dim}")
    print(f"延迟特征维度: {args.latency_feat_dim}")
    print(f"时间序列长度: {args.time_steps}")
    print(f"编码器隐藏层维度: {args.hidden_dim}")
    print(f"全连接层宽度: {args.fc_width}")
    print(f"折扣因子: {args.gamma}")
    print(f"目标网络软更新系数: {args.tau}")
    asyncio.run(main(args))
