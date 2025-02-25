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

# ----------------------------------
# 步骤1：加载原始数据
# ----------------------------------
data_dir = f"{PROJECT_ROOT}/communication/data"

# 加载三个数据集
gathered = np.load(f"{data_dir}/gathered.npy")  # 形状 (n,28,6,4)
latency = np.load(f"{data_dir}/latency.npy")    # 形状 (n,5)
replicas = np.load(f"{data_dir}/replicas.npy")  # 形状 (28,)

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
    """处理第一类数据：融合副本数量（不标准化）并对统计量标准化"""
    n_timesteps = gathered.shape[0]
    
    # 展平最后两个维度 (n,28,6,4) → (n,28,24)
    flattened = gathered.reshape(n_timesteps, 28, -1)
    
    # 直接广播原始副本数（不标准化）到所有时间步：(28,) → (n,28,1)
    replicas_expanded = np.tile(
        replicas.reshape(1, -1, 1),  # 先变形为 (1,28,1)
        (n_timesteps, 1, 1)          # 沿时间步复制n次 → (n,28,1)
    ).astype(np.float32)  # 统一数据类型
    
    # 拼接特征 → (n,28,25)
    merged = np.concatenate([flattened, replicas_expanded], axis=-1)
    
    # 仅对前24列（统计量）进行服务级标准化
    service_scalers = []
    for i in range(28):
        scaler = StandardScaler()
        # 只标准化前24列，保留副本数原始值
        merged[:, i, :24] = scaler.fit_transform(merged[:, i, :24])
        service_scalers.append(scaler)
    
    return merged, service_scalers


def process_data(window_size=10, pred_window=5, threshold=500):
    """
    处理时序数据生成带滑动窗口的训练数据集
    
    参数:
    window_size (int): 输入模型的历史时间步数，默认用过去10个时间步预测未来
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
    4. 时序数据划分: 按70-15-15比例分割训练/验证/测试集，保持时间连续性
    5. 标签生成: 计算预测窗口内P99延迟的平均值，超过阈值标记为违例(1)
    """

    # 处理第一类数据
    X_service, service_scalers = process_gathered(gathered, replicas)
    
    # 处理第二类数据：保留原始延迟数据用于标签生成
    raw_latency = latency.copy()  # 原始未标准化的延迟数据
    percentile_idx = 3  # 假设'99%'延迟在latency数据的第3列
    
    # 标准化延迟指标（仅用于模型输入）
    latency_scaler = StandardScaler()
    X_latency = latency_scaler.fit_transform(latency)

    # ----------------------------------
    # 生成所有有效样本（全局处理）
    # ----------------------------------
    def create_full_dataset(X_serv, X_lat, raw_lat):
        samples_serv = []
        samples_lat = []
        labels = []
        
        n_timesteps = X_serv.shape[0]
        
        # 遍历所有可能的起始点（确保预测窗口不越界）
        for i in range(window_size, n_timesteps - pred_window + 1):
            # 输入窗口：i-window_size 到 i-1
            serv_window = X_serv[i - window_size:i]  # (window_size, 28, 25)
            lat_window = X_lat[i - window_size:i]    # (window_size, 5)
            
            # 预测窗口：i 到 i+pred_window-1
            pred_values = raw_lat[i:i + pred_window, percentile_idx]
            
            # 生成标签：平均P99延迟是否超过阈值
            avg_p99 = np.mean(pred_values)
            label = 1 if avg_p99 > threshold else 0
            
            samples_serv.append(serv_window)
            samples_lat.append(lat_window)
            labels.append(label)
        
        return (
            np.array(samples_serv),  # (num_samples, window_size, 28, 25)
            np.array(samples_lat),    # (num_samples, window_size, 5)
            np.array(labels)         # (num_samples,)
        )

    # 生成所有样本
    X_serv_all, X_lat_all, y_all = create_full_dataset(X_service, X_latency, raw_latency)
    
    # ----------------------------------
    # 按时间顺序划分数据集（样本已有序）
    # ----------------------------------
    num_samples = len(y_all)
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
        print(f"标签分布: 0={np.sum(labels==0)}, 1={np.sum(labels==1)}")
    
    print_dataset_info("训练集", X_train_serv, X_train_lat, y_train)
    print_dataset_info("验证集", X_val_serv, X_val_lat, y_val)
    print_dataset_info("测试集", X_test_serv, X_test_lat, y_test)

    return (
        (X_train_serv, X_train_lat, y_train),
        (X_val_serv, X_val_lat, y_val),
        (X_test_serv, X_test_lat, y_test),
        service_scalers,
        latency_scaler
    )





def main():
    # 使用示例
    train_data, val_data, test_data, service_scalers, latency_scaler = process_data()

     # 解包训练集数据
    (X_train_serv,  # 形状 (num_train, window_size, 28, 25)
     X_train_lat,   # 形状 (num_train, window_size, 5)
     y_train        # 形状 (num_train,)
    ) = train_data

     # 解包验证集数据
    (X_val_serv,    # 形状 (num_val, window_size, 28, 25)
     X_val_lat,     # 形状 (num_val, window_size, 5)
     y_val          # 形状 (num_val,)
    ) = val_data

    # 解包测试集数据
    (X_test_serv,   # 形状 (num_test, window_size, 28, 25)
     X_test_lat,    # 形状 (num_test, window_size, 5)
     y_test         # 形状 (num_test,)
    ) = test_data

     # 典型维度示例 (假设 window_size=10，总共有1000个样本):
    # --------------------------------------------------
    # | 数据集  | 服务数据形状         | 延迟数据形状     | 标签形状 |
    # |---------|---------------------|-----------------|----------|
    # | 训练集  | (800, 10, 28, 25)  | (800, 10, 5)    | (800,)   |
    # | 验证集  | (100, 10, 28, 25)  | (100, 10, 5)    | (100,)   |
    # | 测试集  | (100, 10, 28, 25)  | (100, 10, 5)    | (100,)   |
    # --------------------------------------------------

    # 预处理对象说明：
    # service_scalers: 列表长度28，每个元素是StandardScaler对象
    #   每个scaler对应一个服务的前24个特征的标准化参数
    # latency_scaler:  StandardScaler对象，用于延迟数据的5个特征


class ServiceBranch(nn.Module):
    """处理服务状态数据的动态网络模块"""
    def __init__(self, 
                 feature_dim=25, 
                 time_steps=10,
                 conv_channels=64,
                 lstm_hidden=128):
        super().__init__()
        
        # 时间维度1D卷积（每个服务独立处理）
        self.conv = nn.Conv1d(
            in_channels=feature_dim, 
            out_channels=conv_channels,
            kernel_size=3,
            padding=1
        )
        
        # 时序特征提取
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            batch_first=True
        )
        
        # 自适应池化（处理不同服务数量）
        self.pool = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x):
        """
        输入形状: (batch_size, time_steps, num_services, feature_dim)
        输出形状: (batch_size, lstm_hidden)
        """
        batch_size, T, S, F = x.size()
        
        # 合并时间步和服务维度： (B, T, S, F) → (B*S, T, F)
        x = x.permute(0, 2, 1, 3)       # (B, S, T, F)
        x = x.reshape(-1, T, F)         # (B*S, T, F)
        
        # 时间维度卷积 → (B*S, T, conv_channels)
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)
        
        # LSTM提取时序特征 → (B*S, T, lstm_hidden)
        x, _ = self.lstm(x)
        
        # 取最后一个时间步 → (B*S, lstm_hidden)
        x = x[:, -1, :]
        
        # 按服务聚合 → (B, S, lstm_hidden)
        x = x.view(batch_size, S, -1)
        
        # 全局平均池化 → (B, lstm_hidden)
        x = self.pool(x.permute(0, 2, 1)).squeeze(-1)
        
        return x
    
class LatencyBranch(nn.Module):
    """处理延迟数据的固定网络模块"""
    def __init__(self,
                 input_dim=5,
                 time_steps=10,
                 lstm_hidden=64):
        super().__init__()
        
        self.lstm = nn.LSTM(
            input_size=input_dim,
            hidden_size=lstm_hidden,
            batch_first=True
        )
        
    def forward(self, x):
        """
        输入形状: (batch_size, time_steps, 5)
        输出形状: (batch_size, lstm_hidden)
        """
        # LSTM处理 → (B, T, lstm_hidden)
        x, (h_n, _) = self.lstm(x)
        
        # 取最后隐藏状态 → (B, lstm_hidden)
        return h_n[-1]

    
class SLOViolationPredictor(nn.Module):
    """端到端预测模型"""
    def __init__(self, 
                 service_feature_dim=25,
                 latency_feature_dim=5,
                 num_classes=1):
        super().__init__()
        
        # 服务数据处理分支
        self.service_branch = ServiceBranch(feature_dim=service_feature_dim)
        
        # 延迟数据处理分支
        self.latency_branch = LatencyBranch(input_dim=latency_feature_dim)
        
        # 联合分类器
        self.classifier = nn.Sequential(
            nn.Linear(128 + 64, 64),
            nn.ReLU(),
            nn.Linear(64, num_classes),
            nn.Sigmoid() if num_classes == 1 else nn.Softmax(dim=1)
        )
        
    def forward(self, service_data, latency_data):
        # 服务特征提取
        service_feat = self.service_branch(service_data)  # (B, 128)
        
        # 延迟特征提取
        latency_feat = self.latency_branch(latency_data)  # (B, 64)
        
        # 特征融合
        combined = torch.cat([service_feat, latency_feat], dim=1)
        
        # 分类输出
        return self.classifier(combined)



if __name__ == "__main__":
    main()