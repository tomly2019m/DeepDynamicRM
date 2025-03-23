import numpy as np
from typing import Dict, List, Optional
from collections import deque
import json
from pathlib import Path


class VPACPURecommender:

    def __init__(
        self,
        min_allowed: float,
        max_allowed: float,
        target_percentile: float = 1.00,
        margin: float = 3.5,
        window_size: int = 100,  # 调整为100秒窗口（每秒1个样本）
        decay_factor: float = 0.99,
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
        """添加带时间戳的样本"""
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
        return samples

    def recommend(self) -> Optional[float]:
        """生成CPU推荐值"""
        # if not self.samples:
        #     return self.min_allowed

        # # 1. 应用时间衰减权重
        # weighted_samples = self._apply_time_decay()

        # # 2. 过滤异常值
        # filtered = self._filter_outliers(weighted_samples)
        # if not filtered:
        #     return self.min_allowed

        # # 3. 计算目标百分位数
        # sorted_samples = np.sort(filtered)
        # n = len(sorted_samples)
        # index = int(self.target_percentile * n)
        # percentile_val = sorted_samples[min(index, n - 1)]

        # 4. 应用边际缓冲
        # recommendation = percentile_val * (1 + self.margin)

        #——————————————————————————————————————————————————————————————————
        # 直接获取原始样本值
        values = [sample[1] for sample in self.samples]

        # 获取最大值
        max_value = max(values)

        # 应用边际缓冲
        recommendation = max_value * (1 + self.margin)

        # 5. 边界约束
        return np.clip(recommendation, self.min_allowed, self.max_allowed)


class MultiServiceVPAManager:

    def __init__(self, config_path: str = "./config/service_config.json"):
        # 加载服务配置
        self.config = self._load_config(config_path)

        # 初始化每个服务的推荐器
        self.recommenders: Dict[str, VPACPURecommender] = {}
        for svc, params in self.config.items():
            self.recommenders[svc] = VPACPURecommender(min_allowed=params["min_allowed"],
                                                       max_allowed=params["max_allowed"],
                                                       window_size=100)

    @staticmethod
    def _load_config(path: str) -> Dict:
        """加载服务配置文件"""
        config_file = Path(path)
        if not config_file.exists():
            raise FileNotFoundError(f"Config file {path} not found")

        with open(config_file, 'r') as f:
            return json.load(f)

    def add_samples(self, samples: Dict[str, float], timestamp: float):
        """批量添加服务样本（字典格式）"""
        for svc, usage in samples.items():
            if svc not in self.recommenders:
                raise KeyError(f"服务 '{svc}' 未在配置文件中定义")
            self.recommenders[svc].add_sample(usage, timestamp)

    def get_recommendations(self) -> Dict[str, float]:
        """获取所有服务的CPU推荐值"""
        return {svc: recommender.recommend() for svc, recommender in self.recommenders.items()}
