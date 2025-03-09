import argparse
import asyncio
from SACD import SACD_agent
from env import Env


def parse_args():
    parser = argparse.ArgumentParser(description='SACD算法参数配置')

    # ================== 环境特征参数 ==================
    parser.add_argument('--service-feat-dim', type=int, default=26, help='服务特征维度 (默认: 26)')
    parser.add_argument('--latency-feat-dim', type=int, default=6, help='延迟特征维度 (默认: 6)')
    parser.add_argument('--time-steps', type=int, default=30, help='时间序列长度 (默认: 30)')

    # ================== 网络结构参数 ==================
    parser.add_argument('--hidden-dim', type=int, default=128, help='编码器隐藏层维度 (默认: 128)')
    parser.add_argument('--fc-width', type=int, default=256, help='全连接层宽度 (默认: 256)')

    # ================== 训练超参数 ==================
    parser.add_argument('--gamma', type=float, default=0.99, help='折扣因子 (默认: 0.99)')
    parser.add_argument('--tau', type=float, default=0.005, help='目标网络软更新系数 (默认: 0.005)')
    parser.add_argument('--lr', type=float, default=3e-4, help='统一学习率 (默认: 1e-4)')
    parser.add_argument('--batch-size', type=int, default=256, help='训练批次大小 (默认: 256)')
    parser.add_argument('--adaptive-alpha', action='store_true', help='启用自动熵系数调整 (默认: False)')
    parser.add_argument('--stop-steps', type=int, default=20 * 3600, help='最大训练步数 (默认: 72000)')
    parser.add_argument('--replay-size', type=int, default=int(1e6), help='回放缓冲区容量 (默认: 1e6)')
    parser.add_argument('--random-steps', type=int, default=500, help='纯随机探索步数 (默认: 200)')
    parser.add_argument('--exploration-steps', type=int, default=45 * 200, help='探索阶段总步数 (默认: 9000)')
    parser.add_argument('--action-dim', type=int, default=8, help='动作维度 (默认: 8)')

    # ================== 运行模式 ==================
    parser.add_argument('--train', action='store_true', help='训练模式 (默认: 验证模式)')

    return parser.parse_args()


def main():
    env = Env()
    env.create_connections()
    env.warmup()
    agent = SACD_agent()


if __name__ == "__main__":
    args = parse_args()
    print("参数配置：")
    print(f"学习率: {args.lr:.0e}")
    print(f"批大小: {args.batch_size}")
    print(f"随机探索步数: {args.random_steps}")
    print(f"总探索步数: {args.exploration_steps}")
    print(f"动作维度: {args.action_dim}")
    print(f"训练模式: {args.train}")
    print(f"服务特征维度: {args.service_feat_dim}")
    print(f"延迟特征维度: {args.latency_feat_dim}")
    print(f"时间序列长度: {args.time_steps}")
    print(f"编码器隐藏层维度: {args.hidden_dim}")
    print(f"全连接层宽度: {args.fc_width}")
    print(f"折扣因子: {args.gamma}")
    print(f"目标网络软更新系数: {args.tau}")
    main()
