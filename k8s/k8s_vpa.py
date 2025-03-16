import numpy as np
from typing import List, Optional
from collections import deque
import matplotlib.pyplot as plt


class VPACPURecommender:

    def __init__(
            self,
            target_percentile: float = 0.95,
            margin: float = 0.2,
            min_allowed: float = 0.1,
            max_allowed: float = 4.0,
            window_size: int = 86400,  # 默认24小时数据 (秒级采样)
            decay_factor: float = 0.99,  # 时间衰减因子
    ):
        self.target_percentile = target_percentile
        self.margin = margin
        self.min_allowed = min_allowed
        self.max_allowed = max_allowed
        self.window_size = window_size
        self.decay_factor = decay_factor

        # 存储带时间戳的样本 (timestamp, value)
        self.samples = deque(maxlen=window_size)

    def add_sample(self, value: float, timestamp: float):
        """添加带权重的样本"""
        self.samples.append((timestamp, value))

    def _apply_time_decay(self) -> List[float]:
        """计算时间衰减权重并返回加权样本"""
        if not self.samples:
            return []

        # 获取时间范围
        timestamps = np.array([s[0] for s in self.samples])
        values = np.array([s[1] for s in self.samples])

        # 计算相对时间差（最近的时间权重最大）
        max_time = np.max(timestamps)
        time_diffs = max_time - timestamps
        weights = np.exp(-self.decay_factor * time_diffs)

        # 归一化权重
        weights /= np.sum(weights)

        # 返回加权样本（通过重复值模拟权重）
        weighted_samples = []
        for val, w in zip(values, weights):
            weighted_samples.extend([val] * int(round(w * 1000)))  # 权重放大1000倍
        return weighted_samples

    def _filter_outliers(self, samples: List[float]) -> List[float]:
        """基于IQR过滤异常值"""
        if len(samples) < 4:
            return samples

        q25, q75 = np.percentile(samples, [25, 75])
        iqr = q75 - q25
        lower_bound = q25 - 1.5 * iqr
        upper_bound = q75 + 1.5 * iqr
        return [x for x in samples if lower_bound <= x <= upper_bound]

    def recommend(self) -> Optional[float]:
        """生成CPU推荐值"""
        if not self.samples:
            return self.min_allowed

        # 1. 应用时间衰减权重
        weighted_samples = self._apply_time_decay()

        # 2. 过滤异常值
        filtered = self._filter_outliers(weighted_samples)
        if not filtered:
            return self.min_allowed

        # 3. 计算目标百分位数
        sorted_samples = np.sort(filtered)
        n = len(sorted_samples)
        index = int(self.target_percentile * n)
        percentile_val = sorted_samples[min(index, n - 1)]

        # 4. 应用边际缓冲
        recommendation = percentile_val * (1 + self.margin)

        # 5. 边界约束
        return np.clip(recommendation, self.min_allowed, self.max_allowed)

    def visualize(self, save_path: str = None):
        """可视化当前样本分布与推荐值"""
        if not self.samples:
            return

        weighted = self._apply_time_decay()
        filtered = self._filter_outliers(weighted)
        rec = self.recommend()

        plt.figure(figsize=(10, 6))

        # 原始样本分布
        plt.subplot(2, 1, 1)
        plt.hist([s[1] for s in self.samples], bins=50, alpha=0.5, label='Raw Samples')
        plt.axvline(rec, color='r', linestyle='--', label='Recommendation')
        plt.title("Raw CPU Usage Samples")
        plt.legend()

        # 处理后的样本分布
        plt.subplot(2, 1, 2)
        plt.hist(filtered, bins=50, alpha=0.5, color='g', label='Processed Samples')
        plt.axvline(rec, color='r', linestyle='--', label='Recommendation')
        plt.title("After Time Decay & Outlier Filtering")
        plt.legend()

        if save_path:
            plt.savefig(save_path)
        else:
            plt.show()


# 测试用例
def test_vpa():
    # 模拟24小时数据（峰值在最后4小时）
    vpa = VPACPURecommender()
    base_time = 1609459200  # 2021-01-01 00:00:00

    # 前20小时：低负载
    for t in range(0, 20 * 3600, 60):
        vpa.add_sample(value=0.5 + 0.1 * np.random.rand(), timestamp=base_time + t)

    # 最后4小时：高负载
    for t in range(20 * 3600, 24 * 3600, 60):
        vpa.add_sample(value=1.5 + 0.5 * np.random.rand(), timestamp=base_time + t)

    # 添加异常值
    vpa.add_sample(10.0, base_time + 24 * 3600)  # 异常高值

    rec = vpa.recommend()
    print(f"Recommended CPU: {rec:.2f} cores")

    # 可视化
    vpa.visualize()


if __name__ == "__main__":
    test_vpa()
