import argparse
import asyncio
from SACD import SACD_agent
from env import Env


def parse_args():
    parser = argparse.ArgumentParser(description='环境和训练参数')

    # 训练超参数
    parser.add_argument('--learning-rate', type=float, default=1e-4, help='模型训练的学习率 (默认: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=32, help='训练采样批次大小 (默认: 32)')
    parser.add_argument('--target-update-interval', type=int, default=1000, help='目标网络更新周期（每隔多少步更新）(默认: 1000)')
    parser.add_argument('--replay-size', type=int, default=int(1e6), help='经验回放池容量 (默认: 1,000,000)')
    parser.add_argument('--alpha', type=float, default=0.2, help='初始熵系数 (默认: 0.2)')

    # 探索策略参数
    parser.add_argument('--random-steps', type=int, default=200, help='初始随机探索步数，不更新策略网络 (默认: 200)')
    parser.add_argument('--exploration-steps', type=int, default=45 * 200, help='e-greedy探索总步数，含随机探索阶段 (默认: 9000)')

    # 训练终止参数
    parser.add_argument('--stop-steps', type=int, default=20 * 3600, help='训练最大总步数，超过此值停止训练 (默认: 72000)')
    # 运行模式参数
    parser.add_argument('--train', action='store_true', help='启用训练模式（不指定则默认验证模式）')

    return parser.parse_args()


def main():
    env = Env()
    env.create_connections()
    env.warmup()

    agent = SACD_agent()


if __name__ == "__main__":
    args = parse_args()
    print("参数配置：")
    print(f"学习率: {args.learning_rate:.0e}")
    print(f"批大小: {args.batch_size}")
    print(f"随机探索步数: {args.random_steps}")
    print(f"总探索步数: {args.exploration_steps}")
    main()
