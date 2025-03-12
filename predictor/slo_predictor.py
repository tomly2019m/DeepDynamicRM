import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
from sklearn.preprocessing import StandardScaler
import joblib
from torch.utils.data import DataLoader, TensorDataset
from sklearn.metrics import accuracy_score, roc_auc_score, f1_score

# ----------------------------------
# 步骤1：加载原始数据
# ----------------------------------
data_dir = f"{PROJECT_ROOT}/communication/data"

save_dir = f"{PROJECT_ROOT}/predictor/model"

# 加载三个数据集
gathered = np.load(f"{data_dir}/gathered.npy")  # 形状 (n,28,6,4)
latency = np.load(f"{data_dir}/latency.npy")  # 形状 (n,6)
replicas = np.load(f"{data_dir}/replicas.npy")  # 形状 (28,)
cpu_configs = np.load(f"{data_dir}/cpu_config.npy")  # 形状（n,28）

# 加载三个数据集后添加
print("\n原始数据NaN检查:")
print("gathered:", np.isnan(gathered).any())
print("latency:", np.isnan(latency).any())
print("replicas:", np.isnan(replicas).any())

# ----------------------------------
# 步骤2：数据完整性验证
# ----------------------------------
# 检查样本数量一致性
assert gathered.shape[0] == latency.shape[0], \
    f"时间步数量不一致: gathered {gathered.shape[0]} vs latency {latency.shape[0]}"

# 检查服务数量一致性
assert gathered.shape[1] == replicas.shape[0] == 28, \
    f"服务数量不一致: gathered {gathered.shape[1]}, replicas {replicas.shape[0]}"

n_timesteps = gathered.shape[0]
print(f"数据加载成功！总时间步: {n_timesteps}")


# ----------------------------------
# 步骤3：特征工程处理
# ----------------------------------
def process_gathered(gathered, replicas):
    global latency, cpu_configs
    """处理第一类数据：融合副本数量（不标准化）并对统计量标准化"""

    #数据清洗
    # 定义36个阶段的起始索引和长度
    stage_config = [
        # 50用户（4个阶段）
        (0, 500),
        (500, 500),
        (1000, 500),
        (1500, 500),
        # 100用户（4个阶段）
        (2000, 500),
        (2500, 500),
        (3000, 500),
        (3500, 500),
        # 150用户（4个阶段）
        (4000, 500),
        (4500, 500),
        (5000, 500),
        (5500, 500),
        # 200用户（4个阶段）
        (6000, 500),
        (6500, 500),
        (7000, 500),
        (7500, 500),
        # 250用户（4个阶段）
        (8000, 500),
        (8500, 500),
        (9000, 500),
        (9500, 500),
        # 300用户（4个阶段）
        (10000, 1500),
        (11500, 1500),
        (13000, 1500),
        (14500, 1500),
        # 350用户（4个阶段）
        (16000, 1500),
        (17500, 1500),
        (19000, 1500),
        (20500, 1500),
        # 400用户（4个阶段）
        (22000, 1500),
        (23500, 1500),
        (25000, 1500),
        (26500, 1500),
        # 450用户（4个阶段）
        (28000, 1500),
        (29500, 1500),
        (31000, 1500),
        (32500, 1500)
    ]
    # 生成需要删除的索引列表
    remove_indices = []
    for start_idx, _ in stage_config:
        remove_indices.extend(range(start_idx, start_idx + 9))
    # 创建保留掩码
    n_samples = gathered.shape[0]
    mask = np.ones(n_samples, dtype=bool)
    mask[remove_indices] = False
    # 应用清洗
    gathered = gathered[mask]
    latency = latency[mask]
    cpu_configs = cpu_configs[mask]
    # 验证结果
    print(f"原始数据量: {n_samples}")
    print(f"清洗后数据量: {len(gathered)}")
    print(f"删除数据量: {n_samples - len(gathered)}")

    n_timesteps = gathered.shape[0]

    # 展平最后两个维度 (n,28,6,4) → (n,28,24)
    flattened = gathered.reshape(n_timesteps, 28, -1)

    replicas = np.log1p(replicas)  # 加1避免log(0)

    cpu_configs_expanded = cpu_configs[:, :, np.newaxis].astype(np.float32)  # 添加新轴 → (n,28,1)

    # 直接广播原始副本数（不标准化）到所有时间步：(28,) → (n,28,1)
    replicas_expanded = np.tile(
        replicas.reshape(1, -1, 1),  # 先变形为 (1,28,1)
        (n_timesteps, 1, 1)  # 沿时间步复制n次 → (n,28,1)
    ).astype(np.float32)  # 统一数据类型

    # 拼接特征 → (n,28,26)
    merged = np.concatenate([flattened, cpu_configs_expanded, replicas_expanded], axis=-1)

    # 仅对前25列（统计量）进行服务级标准化
    service_scalers = []
    for i in range(28):
        scaler = StandardScaler()
        # 只标准化前25列，保留副本数原始值
        merged[:, i, :25] = scaler.fit_transform(merged[:, i, :25])
        service_scalers.append(scaler)

    return merged, service_scalers


def process_data(window_size=30, pred_window=5, threshold=500, save_scalers=True):
    """
    处理时序数据生成带滑动窗口的训练数据集
    
    参数:
    window_size (int): 输入模型的历史时间步数，默认用过去30个时间步预测未来
    pred_window (int): 预测未来多少个时间步的违例状态，默认预测未来5个时间步
    threshold (int): SLO违例阈值(单位:ms)，默认500ms，超过则标记为违例
    
    返回:
    tuple: (
        (训练集服务数据, 训练集延迟数据, 训练标签),
        (验证集服务数据, 验证集延迟数据, 验证标签),
        (测试集服务数据, 测试集延迟数据, 测试标签),
        service_scalers,  # 服务数据标准化器列表(每个服务一个)
        latency_scaler   # 延迟数据标准化器
    )
    
    处理流程:
    1. 服务数据预处理: 融合副本数量并标准化统计量
    2. 延迟数据预处理: 标准化延迟指标(除P99外)
    3. 滑动窗口生成: 用window_size步历史数据预测未来pred_window步的违例状态
    4. 时序数据划分: 按80-10-10比例分割训练/验证/测试集，保持时间连续性
    5. 标签生成: 计算预测窗口内P99延迟的平均值，超过阈值标记为违例(1)
    """

    # 处理第一类数据
    X_service, service_scalers = process_gathered(gathered, replicas)

    # 处理第二类数据：保留原始延迟数据用于标签生成
    raw_latency = latency.copy()  # 原始未标准化的延迟数据
    percentile_idx = -2  # 假设'99%'延迟在latency数据的倒数第二列

    # 标准化延迟指标（仅用于模型输入）
    latency_scaler = StandardScaler()
    X_latency = latency_scaler.fit_transform(latency)

    def create_stage_samples(X_serv, X_lat, raw_lat, window_size=30, pred_window=5):
        samples_serv = []
        samples_lat = []
        labels = []

        cleaned_stage_boundaries = [
            # 50用户（4个阶段）
            (0, 490),
            (491, 981),
            (982, 1472),
            (1473, 1963),
            # 100用户（4个阶段）
            (1964, 2454),
            (2455, 2945),
            (2946, 3436),
            (3437, 3927),
            # 150用户（4个阶段）
            (3928, 4418),
            (4419, 4909),
            (4910, 5400),
            (5401, 5891),
            # 200用户（4个阶段）
            (5892, 6382),
            (6383, 6873),
            (6874, 7364),
            (7365, 7855),
            # 250用户（4个阶段）
            (7856, 8346),
            (8347, 8837),
            (8838, 9328),
            (9329, 9819),
            # 300用户（4个阶段）
            (9820, 11310),
            (11311, 12801),
            (12802, 14292),
            (14293, 15783),
            # 350用户（4个阶段）
            (15784, 17274),
            (17275, 18765),
            (18766, 20256),
            (20257, 21747),
            # 400用户（4个阶段）
            (21748, 23238),
            (23239, 24729),
            (24730, 26220),
            (26221, 27711),
            # 450用户（4个阶段）
            (27712, 29202),
            (29203, 30693),
            (30694, 32184),
            (32185, 33675)
        ]

        for stage_start, stage_end in cleaned_stage_boundaries:
            # 跳过长度不足的阶段
            if (stage_end - stage_start + 1) < (window_size + pred_window):
                continue

            # 阶段内数据范围
            stage_serv = X_serv[stage_start:stage_end + 1]  # +1因end是闭区间
            stage_lat = X_lat[stage_start:stage_end + 1]
            stage_raw = raw_lat[stage_start:stage_end + 1]

            # 生成该阶段的样本
            for i in range(window_size, len(stage_serv) - pred_window + 1):
                # 服务特征窗口 (window_size, 28, 26)
                serv_window = stage_serv[i - window_size:i]

                # 延迟特征窗口 (window_size, 6)
                lat_window = stage_lat[i - window_size:i]

                # 预测窗口的P99延迟 (pred_window,)
                pred_values = stage_raw[i:i + pred_window, percentile_idx]

                # 生成分类标签
                max_p99 = np.max(pred_values)
                label = 0
                if max_p99 <= 100:
                    label = 0
                elif max_p99 <= 200:
                    label = 1
                elif max_p99 <= 300:
                    label = 2
                elif max_p99 <= 400:
                    label = 3
                elif max_p99 <= 500:
                    label = 4
                else:
                    label = 5

                samples_serv.append(serv_window)
                samples_lat.append(lat_window)
                labels.append(label)

        return (np.array(samples_serv), np.array(samples_lat), np.array(labels))

    # 生成所有样本
    X_serv_all, X_lat_all, y_all = create_stage_samples(X_service, X_latency, raw_latency)


    num_samples = len(y_all)
    indices = np.arange(num_samples)
    np.random.shuffle(indices) 
    X_serv_all = X_serv_all[indices]
    X_lat_all = X_lat_all[indices]
    y_all = y_all[indices]
    train_end = int(num_samples * 0.8)
    val_end = int(num_samples * 0.9)

    # 训练集（前80%样本）
    X_train_serv = X_serv_all[:train_end]
    X_train_lat = X_lat_all[:train_end]
    y_train = y_all[:train_end]

    # 验证集（中间10%样本）
    X_val_serv = X_serv_all[train_end:val_end]
    X_val_lat = X_lat_all[train_end:val_end]
    y_val = y_all[train_end:val_end]

    # 测试集（后10%样本）
    X_test_serv = X_serv_all[val_end:]
    X_test_lat = X_lat_all[val_end:]
    y_test = y_all[val_end:]

    # ----------------------------------
    # 输出数据集信息
    # ----------------------------------
    def print_dataset_info(name, serv, lat, labels):
        print(f"\n{name}数据集:")
        print(f"样本数: {len(labels)}")
        print(f"服务数据形状: {serv.shape}")
        print(f"延迟数据形状: {lat.shape}")
        print(
            f"标签分布: 0={np.sum(labels==0)}, 1={np.sum(labels==1)}, 2={np.sum(labels==2)}, 3={np.sum(labels==3)}, 4={np.sum(labels==4)}, 5={np.sum(labels==5)}"
        )

    print_dataset_info("训练集", X_train_serv, X_train_lat, y_train)
    print_dataset_info("验证集", X_val_serv, X_val_lat, y_val)
    print_dataset_info("测试集", X_test_serv, X_test_lat, y_test)

    if save_scalers:
        # 保存服务标准化器
        service_scaler_path = f"{save_dir}/service_scalers.pkl"
        joblib.dump(service_scalers, service_scaler_path)
        print(f"保存服务标准化器至 {service_scaler_path}")

        # 保存延迟标准化器
        latency_scaler_path = f"{save_dir}/latency_scaler.pkl"
        joblib.dump(latency_scaler, latency_scaler_path)
        print(f"保存延迟标准化器至 {latency_scaler_path}")

    return ((X_train_serv, X_train_lat, y_train), (X_val_serv, X_val_lat, y_val), (X_test_serv, X_test_lat, y_test),
            service_scalers, latency_scaler)


class ServiceBranch(nn.Module):
    """增强版服务特征提取模块，支持多尺度时空特征融合"""

    def __init__(
            self,
            feature_dim=26,
            time_steps=30,
            service_num=28,
            conv_channels=64,
            lstm_hidden=128,
            mode='hier_attention',  # 新增模式 ['hier_attention', 'multi_scale']
            attn_heads=4):
        super().__init__()
        self.mode = mode
        self.lstm_hidden = lstm_hidden

        # 多尺度卷积网络
        self.conv_blocks = nn.ModuleList([
            nn.Sequential(
                nn.Conv1d(feature_dim, conv_channels, kernel_size=k, padding=k // 2),
                nn.BatchNorm1d(conv_channels),
                nn.ReLU(),
                nn.AdaptiveMaxPool1d(16)  # 统一所有分支输出为固定长度
            ) for k in [3, 5, 7]  # 保持通道数一致，统一时间维度
        ])

        # 时空LSTM编码器
        self.lstm = nn.LSTM(
            input_size=conv_channels * 3,  # 多尺度特征拼接
            hidden_size=lstm_hidden,
            bidirectional=True,  # 使用双向LSTM
            batch_first=True)

        # 层次化注意力机制
        if 'hier' in mode:
            self.temporal_attn = nn.MultiheadAttention(embed_dim=2 * lstm_hidden,
                                                       num_heads=attn_heads,
                                                       batch_first=True)
            self.service_attn = nn.MultiheadAttention(embed_dim=2 * lstm_hidden, num_heads=attn_heads, batch_first=True)
        else:
            self.fusion = nn.Sequential(nn.Linear(2 * lstm_hidden * 16, 512), nn.ReLU(),
                                        nn.Linear(512, lstm_hidden))

        # 残差连接
        self.residual = nn.Sequential(nn.Conv1d(feature_dim, 2 * lstm_hidden, 1), nn.BatchNorm1d(2 * lstm_hidden))

        # 最终投影
        self.proj = nn.Linear(2 * lstm_hidden, lstm_hidden)

    def forward(self, x):
        """
        输入形状: (B, T, S, F)
        输出形状: (B, H)
        """
        B, T, S, F = x.size()

        # 多尺度特征提取 ----------------------------
        # 转换维度: (B, T, S, F) -> (B*S, F, T)
        x_reshaped = x.permute(0, 2, 1, 3).reshape(-1, F, T)

        # 并行多尺度卷积
        conv_features = []
        for conv in self.conv_blocks:
            feat = conv(x_reshaped)  # (B*S, C, 16)
            conv_features.append(feat)

        # 多尺度特征拼接 (B*S, 3C, 16)
        multi_scale = torch.cat(conv_features, dim=1)

        # 时空特征编码 ----------------------------
        # LSTM处理: (B*S, 3C, T//2) -> (B*S, T//2, 2H)
        lstm_out, _ = self.lstm(multi_scale.permute(0, 2, 1))

        # 残差连接
        residual = self.residual(x_reshaped).permute(0, 2, 1)  # (B*S, T, 2H)
        lstm_out = lstm_out + residual[:, :lstm_out.size(1), :]  # 对齐时间维度

        # 注意力融合 ----------------------------
        if 'hier' in self.mode:
            # 时间维度注意力
            temporal_attn, _ = self.temporal_attn(lstm_out, lstm_out, lstm_out)  # (B*S, T', 2H)

            # 服务维度注意力 (B, S, T', 2H)
            temporal_attn = temporal_attn.view(B, S, -1, 2 * self.lstm_hidden)
            service_attn, _ = self.service_attn(
                temporal_attn.mean(dim=2),  # (B, S, 2H)
                temporal_attn.mean(dim=2),
                temporal_attn.mean(dim=2))  # (B, S, 2H)
            out = service_attn.mean(dim=1)  # (B, 2H)
        else:
            # 全量特征融合
            flattened = lstm_out.reshape(B, S, -1)  # (B, S, T'*2H)
            out = self.fusion(flattened.mean(dim=1))  # (B, H)
            return out

        return self.proj(out)  # (B, H)


class LatencyBranch(nn.Module):
    """处理延迟数据的增强模块，使用时间序列全局信息"""

    def __init__(self, input_dim=6, time_steps=30, lstm_hidden=64, use_attention=True):
        super().__init__()
        self.use_attention = use_attention

        # 时序特征提取
        self.lstm = nn.LSTM(input_size=input_dim, hidden_size=lstm_hidden, batch_first=True,
                            bidirectional=True)  # 使用双向LSTM

        # 注意力机制
        if self.use_attention:
            self.attention = nn.Sequential(
                nn.Linear(2 * lstm_hidden, 64),  # 双向所以是2倍
                nn.Tanh(),
                nn.Linear(64, 1),
                nn.Softmax(dim=1))
            self.attn_proj = nn.Linear(2 * lstm_hidden, lstm_hidden)
        # 特征融合
        self.fc = nn.Sequential(
            nn.Linear(2 * lstm_hidden * time_steps, 256),  # 全量信息融合
            nn.ReLU(),
            nn.Linear(256, lstm_hidden))

    def forward(self, x):
        """
        输入形状: (B, T, D=6)
        输出形状: (B, H=64)
        """
        # LSTM特征提取 → (B, T, 2H)
        lstm_out, _ = self.lstm(x)  # 双向输出

        if self.use_attention:
            # 时序注意力加权
            attn_weights = self.attention(lstm_out)  # (B, T, 1)
            weighted = torch.sum(lstm_out * attn_weights, dim=1)  # (B, 2H)
            weighted = self.attn_proj(weighted)
        else:
            # 全量特征拼接
            batch_size = x.size(0)
            flattened = lstm_out.reshape(batch_size, -1)  # (B, T*2H)
            weighted = self.fc(flattened)  # (B, H)

        return weighted


class DynamicSLOPredictor(nn.Module):

    def __init__(self, service_mode='hier_attention'):
        super().__init__()
        # 特征提取模块（保持原始设计）
        self.service_net = ServiceBranch(mode=service_mode)  # 输出维度 (B, 128)
        self.latency_net = LatencyBranch()  # 输出维度 (B, 64)
        # 修改后的六分类器
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 256),  # 增加中间层维度
            nn.BatchNorm1d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.Linear(256, 64),
            nn.GELU(),
            nn.Linear(64, 6)  # 输出6个类别
        )

    def forward(self, service_data, latency_data):
        # 特征提取（无需修改）
        service_feat = self.service_net(service_data)  # (B, 128)
        latency_feat = self.latency_net(latency_data)  # (B, 64)

        # 特征拼接与分类
        combined = torch.cat([service_feat, latency_feat], dim=1)
        return self.classifier(combined)


class OnlineScaler:
    """支持时间步维度的增量标准化器，专为新增服务设计"""

    def __init__(self, epsilon=1e-8, warmup_steps=5):
        """
        参数:
            epsilon: 防止除零的小量
            warmup_steps: 热身样本数（时间步总数），此期间返回原始数据
        """
        self.epsilon = epsilon
        self.warmup_steps = warmup_steps

        # 统计量（按特征维度）
        self.count_ = 0  # 已处理时间步总数
        self.mean_ = None  # 特征均值
        self.M2_ = None  # 方差计算的中间量

    def partial_fit(self, X):
        """增量更新统计量，支持三维输入 (batch, time, features)"""
        if X.ndim == 3:
            X = X.reshape(-1, X.shape[-1])  # 展平时间步维度

        for x in X:
            # 初始化统计量
            if self.mean_ is None:
                self.mean_ = np.zeros_like(x)
                self.M2_ = np.zeros_like(x)

            self.count_ += 1
            delta = x - self.mean_
            self.mean_ += delta / self.count_
            delta2 = x - self.mean_
            self.M2_ += delta * delta2

    def transform(self, X):
        """应用标准化，自动处理三维结构"""
        original_shape = X.shape
        if X.ndim == 3:
            X = X.reshape(-1, X.shape[-1])  # (batch*time, features)

        # 热身期间返回原始数据
        if self.count_ < self.warmup_steps:
            return X.reshape(original_shape)

        # 计算标准差
        variance = self.M2_ / (self.count_ - 1) if self.count_ > 1 else 1.0
        std = np.sqrt(variance + self.epsilon)

        # 标准化
        normalized = (X - self.mean_) / std
        return normalized.reshape(original_shape)

    @property
    def variance_(self):
        """获取方差的无偏估计"""
        if self.count_ < 2:
            return np.zeros_like(self.M2_)
        return self.M2_ / (self.count_ - 1)

    def __repr__(self):
        return f"OnlineScaler(count={self.count_}, features={self.mean_.shape[0]})"


class SLOTrainer:

    def __init__(
            self,
            service_mode='hier_attention',
            device='cuda',
            class_weights=None,  # 新增参数：类别权重
            lr=2e-5,
            batch_size=64,
            weight_decay=1e-4):
        """
        参数:
            service_mode: 'hier_attention' 或 'mutil_scale'
            device: 计算设备
            class_weights: 类别权重张量 (长度6)
            batch_size: 批大小
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.service_mode = service_mode

        # 初始化模型组件
        self.model = self._init_model(service_mode).to(self.device)

        # 损失函数修改为CrossEntropyLoss
        self.criterion = nn.CrossEntropyLoss(
            weight=class_weights.to(self.device) if class_weights is not None else None)

        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=lr, weight_decay=weight_decay)
        self.scaler_manager = None

    def _init_model(self, mode):
        return DynamicSLOPredictor(service_mode=mode)

    class SLODataset(TensorDataset):

        def __init__(self, service_data, latency_data, labels):
            # 标签改为长整型
            super().__init__(
                torch.FloatTensor(service_data),
                torch.FloatTensor(latency_data),
                torch.LongTensor(labels)  # 修改为LongTensor
            )

    def train_epoch(self):
        self.model.train()
        total_loss = 0

        for service, latency, labels in self.train_loader:
            service = service.to(self.device)
            latency = latency.to(self.device)
            labels = labels.to(self.device)  # labels自动转为torch.long

            self.optimizer.zero_grad()

            # 输出形状 (B,6)
            outputs = self.model(service, latency)

            # 直接计算交叉熵损失
            loss = self.criterion(outputs, labels)

            loss.backward()
            # 可调参数
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def evaluate(self, loader):
        self.model.eval()
        all_preds = []
        all_labels = []
        total_loss = 0.0  # 添加损失累加器

        with torch.no_grad():
            for service, latency, labels in loader:
                service = service.to(self.device)
                latency = latency.to(self.device)
                labels = labels.to(self.device)  # 将标签也移至设备

                outputs = self.model(service, latency)
                preds = torch.argmax(outputs, dim=1)  # 取最大概率类别

                # 计算损失
                loss = self.criterion(outputs, labels)
                total_loss += loss.item()

                all_preds.append(preds.cpu().numpy())
                all_labels.append(labels.cpu().numpy())  # 确保标签也是CPU张量

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)

        metrics = {
            'loss': total_loss / len(loader),  # 添加平均损失
            'accuracy': accuracy_score(y_true, y_pred),
            'f1_macro': f1_score(y_true, y_pred, average='macro'),
            'f1_weighted': f1_score(y_true, y_pred, average='weighted')
        }
        return metrics

    def prepare_data(self, train_data, val_data, test_data):
        """数据预处理与加载器准备"""
        # 创建数据集
        self.train_dataset = self.SLODataset(*train_data)
        self.val_dataset = self.SLODataset(*val_data)
        self.test_dataset = self.SLODataset(*test_data)

        # 创建加载器
        self.train_loader = DataLoader(self.train_dataset, batch_size=self.batch_size, shuffle=True)
        self.val_loader = DataLoader(self.val_dataset, batch_size=self.batch_size)
        self.test_loader = DataLoader(self.test_dataset, batch_size=self.batch_size)

    def train(self, epochs=5000, early_stop=5000, callback=None):
        """使用验证集准确率作为早停标准"""
        best_acc = 0
        no_improve = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)

            print(f"Epoch {epoch+1:03d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Loss: {val_metrics['loss']:.4f} |"
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"Macro-F1: {val_metrics['f1_macro']:.4f} | "
                  f"Weighted-F1: {val_metrics['f1_weighted']:.4f}")

            # 以准确率为早停标准
            if val_metrics['accuracy'] > best_acc:
                best_acc = val_metrics['accuracy']
                self.save_checkpoint()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

            if callback:
                callback(epoch, train_loss, val_metrics, self.optimizer.param_groups[0]['lr'])

    def save_checkpoint(self, path=f'{save_dir}/best_model.pth'):
        """保存完整训练状态"""
        torch.save(
            {
                'model_state': self.model.state_dict(),
                'optimizer_state': self.optimizer.state_dict(),
                'scaler_manager': self.scaler_manager,
                'config': {
                    'service_mode': self.service_mode,
                    'batch_size': self.batch_size
                }
            }, path)

    def load_checkpoint(self, path=f'{save_dir}/best_model.pth'):
        """加载完整训练状态"""
        checkpoint = torch.load(path, map_location=self.device)
        self.model.load_state_dict(checkpoint['model_state'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state'])
        self.scaler_manager = checkpoint['scaler_manager']


def predict_example():
    """预测示例函数，处理两种输入情况"""
    # 加载预训练资源 -------------------------------------------------
    # 加载模型
    model = DynamicSLOPredictor(service_mode='attention')
    model.load_state_dict(torch.load(f"{save_dir}/best_model.pth"))
    model.eval()

    # 加载标准化器
    service_scalers = joblib.load(f"{save_dir}/service_scalers.pkl")  # 28个StandardScaler
    latency_scaler = joblib.load(f"{save_dir}/latency_scaler.pkl")

    # 维护新增服务的scaler {service_id: OnlineScaler}
    new_scalers = {}

    # 示例1: 正常输入 (batch=1, 10, 28, 25)
    def case_normal():
        # 生成模拟数据
        gathered = np.random.randn(1, 10, 28, 25)
        latency = np.random.randn(1, 10, 5)

        # 预处理服务数据
        processed_serv = np.zeros_like(gathered)
        for s in range(28):
            # 提取特征 (1,10,24)
            features = gathered[:, :, s, :24].reshape(-1, 24)
            # 使用预训练scaler
            processed_serv[:, :, s, :24] = service_scalers[s].transform(features).reshape(1, 10, 24)

        # 预处理延迟数据
        processed_lat = latency_scaler.transform(latency.reshape(-1, 5)).reshape(1, 10, 5)

        # 转换为Tensor并预测
        with torch.no_grad():
            prob = model(torch.FloatTensor(processed_serv), torch.FloatTensor(processed_lat)).item()

        print("\n案例1：正常输入")
        print(f"输入形状: 服务数据 {gathered.shape}, 延迟数据 {latency.shape}")
        print(f"预测违例概率: {prob:.4f}")

    # 示例2: 新增服务输入 (batch=1, 10, 29, 25)
    def case_new_services():
        # 生成模拟数据（新增第29个服务）
        gathered = np.random.randn(1, 10, 29, 25)
        latency = np.random.randn(1, 10, 5)

        processed_serv = np.zeros_like(gathered)

        # 处理前28个服务
        for s in range(28):
            features = gathered[:, :, s, :24].reshape(-1, 24)
            processed_serv[:, :, s, :24] = service_scalers[s].transform(features).reshape(1, 10, 24)

        # 处理新增服务（第29个）
        s = 28  # 索引从0开始，28对应第29个服务
        features = gathered[:, :, s, :24].reshape(-1, 24)

        # 初始化或获取OnlineScaler
        if s not in new_scalers:
            new_scalers[s] = OnlineScaler(warmup_steps=5)
        scaler = new_scalers[s]

        # 在线更新并转换
        scaler.partial_fit(features)
        processed_serv[:, :, s, :24] = scaler.transform(features).reshape(1, 10, 24)

        # 处理延迟数据
        processed_lat = latency_scaler.transform(latency.reshape(-1, 5)).reshape(1, 10, 5)

        # 预测
        with torch.no_grad():
            prob = model(torch.FloatTensor(processed_serv), torch.FloatTensor(processed_lat)).item()

        print("\n案例2：新增服务输入")
        print(f"输入形状: 服务数据 {gathered.shape}, 延迟数据 {latency.shape}")
        print(f"新增服务scaler状态: {new_scalers[s]}")
        print(f"预测违例概率: {prob:.4f}")

    # 执行案例 -------------------------------------------------
    case_normal()
    case_new_services()


def analyze_feature_importance(model, sample):
    import shap
    explainer = shap.DeepExplainer(model, sample)
    shap_values = explainer.shap_values(sample)
    shap.summary_plot(shap_values, sample)


def main(epochs, learning_rate, batch_size, service_mode, weight_decay):
    # 导入必要的模块
    import time
    import os
    import json
    import matplotlib.pyplot as plt
    from torch.utils.tensorboard import SummaryWriter

    # 创建基于时间戳的保存目录
    time_str = time.strftime("%Y%m%d_%H%M%S", time.localtime())
    save_dir = f"./exp_data/{time_str}"
    os.makedirs(save_dir, exist_ok=True)

    # 创建日志记录器
    log_file = f"{save_dir}/training_log.txt"
    tensorboard_dir = f"{save_dir}/tensorboard"
    writer = SummaryWriter(tensorboard_dir)

    # 记录函数
    def log_info(message):
        time_now = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_message = f"[{time_now}] {message}"
        print(log_message)
        with open(log_file, "a", encoding="utf-8") as f:
            f.write(log_message + "\n")

    # 记录训练开始
    log_info("开始训练过程")
    start_time = time.time()

    # 数据预处理
    log_info("开始数据预处理...")
    train_data, val_data, test_data, service_scalers, latency_scaler = process_data(window_size=30,
                                                                                    pred_window=5,
                                                                                    threshold=500)
    log_info(f"数据预处理完成，训练集大小: {len(train_data[0])}, 验证集大小: {len(val_data[0])}, 测试集大小: {len(test_data[0])}")

    # 保存标准化器
    log_info("保存标准化器...")
    import joblib
    scaler_dir = f"{save_dir}/scalers"
    os.makedirs(scaler_dir, exist_ok=True)
    joblib.dump(service_scalers, f"{scaler_dir}/service_scalers.pkl")
    joblib.dump(latency_scaler, f"{scaler_dir}/latency_scaler.pkl")

    # 计算类别权重（处理不平衡数据）
    log_info("计算类别权重...")
    from sklearn.utils.class_weight import compute_class_weight
    _, _, y_train = train_data
    class_weights = compute_class_weight('balanced', classes=np.arange(6), y=y_train)
    class_weights = torch.tensor(class_weights, dtype=torch.float32)

    # 记录类别分布
    unique, counts = np.unique(y_train, return_counts=True)
    class_distribution = dict(zip(unique, counts))
    log_info(f"类别分布: {class_distribution}")
    log_info(f"类别权重: {class_weights.numpy().tolist()}")

    # 保存训练配置
    config = {
        "window_size": 30,
        "pred_window": 5,
        "threshold": 500,
        "service_mode": service_mode,
        "batch_size": batch_size,
        "learning_rate": learning_rate,
        "epochs": epochs,
        "weight_decay": weight_decay,
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "class_distribution": {
            int(k): int(v)
            for k, v in class_distribution.items()
        },
        "class_weights": class_weights.numpy().tolist(),
    }
    with open(f"{save_dir}/config.json", "w") as f:
        json.dump(config, f, indent=4)

    # 初始化训练器
    log_info("初始化训练器...")
    trainer = SLOTrainer(
        service_mode=service_mode,
        device='cuda' if torch.cuda.is_available() else 'cpu',
        class_weights=class_weights,  # 传入类别权重
        batch_size=batch_size,
        lr=learning_rate,
        weight_decay=weight_decay)

    # 准备数据（添加维度验证）
    try:
        log_info("准备数据...")
        trainer.prepare_data(train_data, val_data, test_data)
        log_info("数据准备完成")
    except AssertionError as e:
        log_info(f"数据校验失败: {str(e)}")
        return

    # 记录模型结构
    log_info(f"模型结构:\n{trainer.model}")
    total_params = sum(p.numel() for p in trainer.model.parameters())
    trainable_params = sum(p.numel() for p in trainer.model.parameters() if p.requires_grad)
    log_info(f"总参数量: {total_params}, 可训练参数: {trainable_params}")

    # 训练历史记录
    history = {"train_loss": [], "val_loss": [], "val_accuracy": [], "val_f1_macro": [], "learning_rate": []}

    # 自定义回调函数
    def epoch_callback(epoch, train_loss, val_metrics, lr):
        # 记录到历史
        history["train_loss"].append(train_loss)
        history["val_loss"].append(val_metrics["loss"])
        history["val_accuracy"].append(val_metrics["accuracy"])
        history["val_f1_macro"].append(val_metrics["f1_macro"])
        history["learning_rate"].append(lr)

        # 记录到TensorBoard
        writer.add_scalar("Loss/train", train_loss, epoch)
        writer.add_scalar("Loss/val", val_metrics["loss"], epoch)
        writer.add_scalar("Accuracy/val", val_metrics["accuracy"], epoch)
        writer.add_scalar("F1/macro", val_metrics["f1_macro"], epoch)
        writer.add_scalar("F1/weighted", val_metrics["f1_weighted"], epoch)
        writer.add_scalar("LearningRate", lr, epoch)

        # 每10个epoch记录一次详细信息
        if epoch % 2 == 0:
            log_info(f"Epoch {epoch}: train loss={train_loss:.4f}, val loss={val_metrics['loss']:.4f}, "
                     f"acc={val_metrics['accuracy']:.4f}, F1={val_metrics['f1_macro']:.4f}, LR={lr:.6f}")

            # 每100个epoch保存一次检查点
            if epoch % 100 == 0 and epoch > 0:
                checkpoint_path = f"{save_dir}/checkpoint_epoch_{epoch}.pth"
                trainer.save_checkpoint(checkpoint_path)
                log_info(f"保存检查点: {checkpoint_path}")

    # 训练流程（增加学习率调度和回调）
    try:
        log_info("开始训练...")
        trainer.train(epochs=epochs, early_stop=500, callback=epoch_callback)  # 添加回调函数
    except KeyboardInterrupt:
        log_info("\n训练中断，保存临时模型...")
        trainer.save_checkpoint(f"{save_dir}/interrupted.pth")

        # 保存训练历史
        with open(f"{save_dir}/history_interrupted.json", "w") as f:
            json.dump(history, f, indent=4)

    # 分别绘制并保存每个训练曲线
    log_info("生成训练曲线图表...")

    # 1. 损失曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history["train_loss"], label="train_loss")
    plt.plot(history["val_loss"], label="val_loss", color="red")
    plt.legend()
    plt.title("loss curve")
    plt.xlabel("Epoch")
    plt.ylabel("loss")
    plt.grid(True, linestyle='--', alpha=0.6)
    loss_path = f"{save_dir}/loss_curve.png"
    plt.savefig(loss_path, dpi=300)
    plt.close()
    log_info(f"损失曲线已保存至 {loss_path}")

    # 2. 准确率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history["val_accuracy"], label="val ACC", color="green")
    plt.legend()
    plt.title("ACC curve")
    plt.xlabel("Epoch")
    plt.ylabel("accuracy")
    plt.grid(True, linestyle='--', alpha=0.6)
    acc_path = f"{save_dir}/accuracy_curve.png"
    plt.savefig(acc_path, dpi=300)
    plt.close()
    log_info(f"准确率曲线已保存至 {acc_path}")

    # 3. F1分数曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history["val_f1_macro"], label="Val F1 score", color="orange")
    plt.legend()
    plt.title("F1 score curve")
    plt.xlabel("Epoch")
    plt.ylabel("F1 score")
    plt.grid(True, linestyle='--', alpha=0.6)
    f1_path = f"{save_dir}/f1_curve.png"
    plt.savefig(f1_path, dpi=300)
    plt.close()
    log_info(f"F1分数曲线已保存至 {f1_path}")

    # 4. 学习率曲线
    plt.figure(figsize=(10, 6))
    plt.plot(history["learning_rate"], label="learning rate", color="purple")
    plt.legend()
    plt.title("lr curve")
    plt.xlabel("Epoch")
    plt.ylabel("lr")
    plt.grid(True, linestyle='--', alpha=0.6)
    lr_path = f"{save_dir}/learning_rate_curve.png"
    plt.savefig(lr_path, dpi=300)
    plt.close()
    log_info(f"学习率曲线已保存至 {lr_path}")

    # 最终测试集评估
    log_info("\n" + "=" * 50)
    log_info("测试集最终表现:")
    test_metrics = trainer.evaluate(trainer.test_loader)
    log_info(f"accuracy: {test_metrics['accuracy']:.4f}")
    log_info(f"宏平均F1 marco avg F1: {test_metrics['f1_macro']:.4f}")
    log_info(f"weighted F1: {test_metrics['f1_weighted']:.4f}")

    # 保存混淆矩阵
    if hasattr(test_metrics, 'confusion_matrix'):
        plt.figure(figsize=(10, 8))
        plt.imshow(test_metrics['confusion_matrix'], cmap='Blues')
        plt.colorbar()
        plt.xlabel('预测标签')
        plt.ylabel('真实标签')
        plt.title('混淆矩阵')
        plt.savefig(f"{save_dir}/confusion_matrix.png")
        log_info(f"混淆矩阵已保存至 {save_dir}/confusion_matrix.png")

    # 保存最终模型
    best_model_path = f"{save_dir}/best_model.pth"
    final_model_path = f"{save_dir}/final_model.pth"
    trainer.save_checkpoint(final_model_path)
    log_info(f"最终模型已保存至 {final_model_path}")

    # 如果有最佳模型，复制一份
    if hasattr(trainer, 'best_model_state'):
        torch.save(trainer.best_model_state, best_model_path)
        log_info(f"最佳模型已保存至 {best_model_path}")

    # 保存完整训练历史
    with open(f"{save_dir}/history.json", "w") as f:
        json.dump(history, f, indent=4)

    # 记录训练总时间
    end_time = time.time()
    training_time = end_time - start_time
    hours, remainder = divmod(training_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    log_info(f"训练完成！总用时: {int(hours)}小时 {int(minutes)}分钟 {int(seconds)}秒")

    # 关闭TensorBoard写入器
    writer.close()


import argparse


def parse_arguments():
    parser = argparse.ArgumentParser(description="Train the SLO Predictor model.")
    parser.add_argument("--epochs", type=int, default=300, help="Number of training epochs.")
    parser.add_argument("--learning-rate", type=float, default=1e-5, help="Learning rate.")
    parser.add_argument("--batch-size", type=int, default=64, help="Batch size.")
    parser.add_argument("--service-mode", type=str, default="hier_attention", help="Service mode.")
    parser.add_argument("--weight-decay", type=float, default=1e-4, help="Weight decay.")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_arguments()
    main(args.epochs, args.learning_rate, args.batch_size, args.service_mode, args.weight_decay)
