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

save_dir = f"{PROJECT_ROOT}/predictor/data"

# 加载三个数据集
gathered = np.load(f"{data_dir}/gathered.npy")  # 形状 (n,28,6,4)
latency = np.load(f"{data_dir}/latency.npy")  # 形状 (n,5)
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
        (n_timesteps, 1, 1)  # 沿时间步复制n次 → (n,28,1)
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


def process_data(window_size=10,
                 pred_window=5,
                 threshold=500,
                 save_scalers=True):
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
            lat_window = X_lat[i - window_size:i]  # (window_size, 5)

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
            np.array(samples_lat),  # (num_samples, window_size, 5)
            np.array(labels)  # (num_samples,)
        )

    # 生成所有样本
    X_serv_all, X_lat_all, y_all = create_full_dataset(X_service, X_latency,
                                                       raw_latency)

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

    if save_scalers:
        # 保存服务标准化器
        service_scaler_path = f"{save_dir}/service_scalers.pkl"
        joblib.dump(service_scalers, service_scaler_path)
        print(f"保存服务标准化器至 {service_scaler_path}")

        # 保存延迟标准化器
        latency_scaler_path = f"{save_dir}/latency_scaler.pkl"
        joblib.dump(latency_scaler, latency_scaler_path)
        print(f"保存延迟标准化器至 {latency_scaler_path}")

    return ((X_train_serv, X_train_lat, y_train), (X_val_serv, X_val_lat,
                                                   y_val),
            (X_test_serv, X_test_lat, y_test), service_scalers, latency_scaler)


class ServiceBranch(nn.Module):
    """支持两种聚合模式的服务数据处理模块"""

    def __init__(
            self,
            feature_dim=25,
            time_steps=10,
            conv_channels=64,
            lstm_hidden=128,
            mode='attention',  # 新增模式参数 ['attention', 'pooling']
            attn_heads=4):  # 注意力模式专用参数
        super().__init__()
        self.mode = mode

        # 共享的时序特征提取层
        self.conv = nn.Conv1d(in_channels=feature_dim,
                              out_channels=conv_channels,
                              kernel_size=3,
                              padding=1)
        self.lstm = nn.LSTM(input_size=conv_channels,
                            hidden_size=lstm_hidden,
                            batch_first=True)

        # 模式分支
        if self.mode == 'attention':
            # 注意力机制
            self.service_query = nn.Parameter(
                torch.randn(1, attn_heads, lstm_hidden))
            self.attn = nn.MultiheadAttention(embed_dim=lstm_hidden,
                                              num_heads=attn_heads,
                                              batch_first=True)
            self.output_proj = nn.Linear(lstm_hidden, lstm_hidden)
        elif self.mode == 'pooling':
            # 自适应池化
            self.pool = nn.AdaptiveAvgPool1d(1)
        else:
            raise ValueError(f"Unsupported mode: {self.mode}")

    def forward(self, x):
        """
        输入形状: (B, T, S, F)
        输出形状: (B, lstm_hidden)
        """
        B, T, S, F = x.size()

        # 公共特征提取流程
        x = x.permute(0, 2, 1, 3)  # (B, S, T, F)
        x = x.reshape(-1, T, F)  # (B*S, T, F)
        x = self.conv(x.permute(0, 2, 1)).permute(0, 2, 1)  # (B*S, T, C)
        x, _ = self.lstm(x)  # (B*S, T, H)
        x = x[:, -1, :]  # (B*S, H)
        x = x.view(B, S, -1)  # (B, S, H)

        # 模式分支处理
        if self.mode == 'attention':
            # 注意力聚合
            query = self.service_query.expand(B, -1, -1)  # (B, heads, H)
            attn_out, _ = self.attn(query=query, key=x,
                                    value=x)  # (B, heads, H)
            out = self.output_proj(attn_out.mean(dim=1))  # (B, H)
        elif self.mode == 'pooling':
            # 池化聚合
            out = self.pool(x.permute(0, 2, 1)).squeeze(-1)  # (B, H)

        return out


class LatencyBranch(nn.Module):
    """处理延迟数据的固定网络模块"""

    def __init__(self, input_dim=5, time_steps=10, lstm_hidden=64):
        super().__init__()

        self.lstm = nn.LSTM(input_size=input_dim,
                            hidden_size=lstm_hidden,
                            batch_first=True)

    def forward(self, x):
        """
        输入形状: (batch_size, time_steps, 5)
        输出形状: (batch_size, lstm_hidden)
        """
        # LSTM处理 → (B, T, lstm_hidden)
        x, (h_n, _) = self.lstm(x)

        # 取最后隐藏状态 → (B, lstm_hidden)
        return h_n[-1]


class DynamicSLOPredictor(nn.Module):

    def __init__(self, service_mode='attention'):
        super().__init__()
        self.service_net = ServiceBranch(mode=service_mode)
        self.latency_net = LatencyBranch()

        self.classifier = nn.Sequential(nn.Linear(128 + 64, 64), nn.ReLU(),
                                        nn.Linear(64, 1), nn.Sigmoid())

    def forward(self, service_data, latency_data):
        service_feat = self.service_net(service_data)
        latency_feat = self.latency_net(latency_data)
        return self.classifier(torch.cat([service_feat, latency_feat], dim=1))


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

    def __init__(self,
                 service_mode='attention',
                 device='cuda',
                 pos_weight=2.0,
                 batch_size=64):
        """
        参数:
            service_mode: 'attention' 或 'pooling'
            device: 计算设备
            pos_weight: 正样本权重
            batch_size: 批大小
        """
        self.device = device if torch.cuda.is_available() else 'cpu'
        self.batch_size = batch_size
        self.service_mode = service_mode

        # 初始化模型组件
        self.model = self._init_model(service_mode).to(self.device)
        self.criterion = torch.nn.BCEWithLogitsLoss(
            pos_weight=torch.tensor([pos_weight]).to(self.device))
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=1e-4)
        self.scaler_manager = None  # 标准化管理器

    def _init_model(self, mode):
        """初始化双分支模型"""
        return DynamicSLOPredictor(service_mode=mode)

    def prepare_data(self, train_data, val_data, test_data):
        """数据预处理与加载器准备"""
        # 创建数据集
        self.train_dataset = self.SLODataset(*train_data)
        self.val_dataset = self.SLODataset(*val_data)
        self.test_dataset = self.SLODataset(*test_data)

        # 创建加载器
        self.train_loader = DataLoader(self.train_dataset,
                                       batch_size=self.batch_size,
                                       shuffle=True)
        self.val_loader = DataLoader(self.val_dataset,
                                     batch_size=self.batch_size)
        self.test_loader = DataLoader(self.test_dataset,
                                      batch_size=self.batch_size)

    class SLODataset(TensorDataset):
        """多模态时序数据集"""

        def __init__(self, service_data, latency_data, labels):
            super().__init__(torch.FloatTensor(service_data),
                             torch.FloatTensor(latency_data),
                             torch.FloatTensor(labels))

        def __getitem__(self, idx):
            return (
                self.tensors[0][idx],  # service (T, S, F)
                self.tensors[1][idx],  # latency (T, D)
                self.tensors[2][idx]  # label
            )

    def train_epoch(self):
        """修改后的训练周期"""
        self.model.train()
        total_loss = 0

        for service, latency, labels in self.train_loader:
            # 数据迁移到设备
            service = service.to(self.device)
            latency = latency.to(self.device)
            labels = labels.to(self.device)

            # 梯度清零
            self.optimizer.zero_grad()

            # 前向传播（移除autocast）
            outputs = self.model(service, latency).squeeze()
            loss = self.criterion(outputs, labels)

            # 反向传播（移除scaler）
            loss.backward()

            # 梯度裁剪
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)

            # 参数更新
            self.optimizer.step()

            total_loss += loss.item()

        return total_loss / len(self.train_loader)

    def evaluate(self, loader):
        """在指定数据集上评估模型"""
        self.model.eval()
        all_preds = []
        all_labels = []

        with torch.no_grad():
            for service, latency, labels in loader:
                service = service.to(self.device)
                latency = latency.to(self.device)

                outputs = self.model(service, latency).squeeze()
                probs = torch.sigmoid(outputs).cpu().numpy()

                all_preds.append(probs)
                all_labels.append(labels.numpy())

        y_true = np.concatenate(all_labels)
        y_pred = np.concatenate(all_preds)

        metrics = {
            'accuracy': accuracy_score(y_true, y_pred > 0.5),
            'auc': roc_auc_score(y_true, y_pred),
            'f1': f1_score(y_true, y_pred > 0.5)
        }
        return metrics

    def train(self, epochs=100, early_stop=10):
        """完整训练流程"""
        best_auc = 0
        no_improve = 0

        for epoch in range(epochs):
            train_loss = self.train_epoch()
            val_metrics = self.evaluate(self.val_loader)

            print(f"Epoch {epoch+1:03d} | "
                  f"Train Loss: {train_loss:.4f} | "
                  f"Val Acc: {val_metrics['accuracy']:.4f} | "
                  f"AUC: {val_metrics['auc']:.4f} | "
                  f"F1: {val_metrics['f1']:.4f}")

            # 保存最佳模型
            if val_metrics['auc'] > best_auc:
                best_auc = val_metrics['auc']
                self.save_checkpoint()
                no_improve = 0
            else:
                no_improve += 1
                if no_improve >= early_stop:
                    print(f"Early stopping at epoch {epoch+1}")
                    break

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


def train_example(epochs: int = 500,
                  batch_size: int = 64,
                  pos_weight: float = 2.0,
                  device: str = "cuda",
                  service_mode: str = "attention",
                  save_dir: str = f"{PROJECT_ROOT}/predictor/data",
                  window_size: int = 10,
                  pred_window: int = 5,
                  threshold: int = 500):
    """
    完整的训练流程示例
    
    参数:
        epochs (int): 训练轮数，默认500
        batch_size (int): 批大小，默认64
        pos_weight (float): 正样本权重，默认2.0
        device (str): 计算设备，默认"cuda"
        service_mode (str): 服务分支模式，["attention", "pooling"]
        save_dir (str): 模型和标准化器保存路径
        window_size (int): 输入时间窗口大小，默认10
        pred_window (int): 预测时间窗口，默认5
        threshold (int): SLO违例阈值(ms)，默认500
    """
    # 数据预处理
    train_data, val_data, test_data, service_scalers, latency_scaler = process_data(
        window_size=window_size, pred_window=pred_window, threshold=threshold)

    # 打印数据集信息
    def print_shape_info(name, data):
        print(
            f"{name}服务数据: {data[0].shape} | 延迟数据: {data[1].shape} | 标签: {data[2].shape}"
        )

    print("\n" + "=" * 40)
    print_shape_info("训练集", train_data)
    print_shape_info("验证集", val_data)
    print_shape_info("测试集", test_data)
    print("=" * 40 + "\n")

    # 初始化训练器
    trainer = SLOTrainer(service_mode=service_mode,
                         device=device,
                         pos_weight=pos_weight,
                         batch_size=batch_size)

    # 准备数据加载器
    trainer.prepare_data(train_data, val_data, test_data)

    # 训练流程
    try:
        print(f"开始训练，共{epochs}个epoch...")
        trainer.train(epochs=epochs)
    except KeyboardInterrupt:
        print("\n训练中断，保存临时模型...")
        trainer.save_checkpoint(f"{save_dir}/interrupted_model.pth")
    finally:
        # 始终保存最终模型
        trainer.save_checkpoint(f"{save_dir}/final_model.pth")
        print(f"模型已保存至 {save_dir}/final_model.pth")

    # 测试集评估
    test_metrics = trainer.evaluate(trainer.test_loader)
    print("\n" + "=" * 40)
    print("测试集最终表现:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"AUC-ROC:  {test_metrics['auc']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")
    print("=" * 40)

    # 保存标准化器
    joblib.dump(service_scalers, f"{save_dir}/service_scalers.pkl")
    joblib.dump(latency_scaler, f"{save_dir}/latency_scaler.pkl")
    print(f"\n标准化器已保存至 {save_dir}")

    return trainer  # 返回训练器对象以便后续使用


def predict_example():
    """预测示例函数，处理两种输入情况"""
    # 加载预训练资源 -------------------------------------------------
    # 加载模型
    model = DynamicSLOPredictor(service_mode='attention')
    model.load_state_dict(torch.load(f"{save_dir}/best_model.pth"))
    model.eval()

    # 加载标准化器
    service_scalers = joblib.load(
        f"{save_dir}/service_scalers.pkl")  # 28个StandardScaler
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
            processed_serv[:, :, s, :24] = service_scalers[s].transform(
                features).reshape(1, 10, 24)

        # 预处理延迟数据
        processed_lat = latency_scaler.transform(latency.reshape(-1,
                                                                 5)).reshape(
                                                                     1, 10, 5)

        # 转换为Tensor并预测
        with torch.no_grad():
            prob = model(torch.FloatTensor(processed_serv),
                         torch.FloatTensor(processed_lat)).item()

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
            processed_serv[:, :, s, :24] = service_scalers[s].transform(
                features).reshape(1, 10, 24)

        # 处理新增服务（第29个）
        s = 28  # 索引从0开始，28对应第29个服务
        features = gathered[:, :, s, :24].reshape(-1, 24)

        # 初始化或获取OnlineScaler
        if s not in new_scalers:
            new_scalers[s] = OnlineScaler(warmup_steps=5)
        scaler = new_scalers[s]

        # 在线更新并转换
        scaler.partial_fit(features)
        processed_serv[:, :,
                       s, :24] = scaler.transform(features).reshape(1, 10, 24)

        # 处理延迟数据
        processed_lat = latency_scaler.transform(latency.reshape(-1,
                                                                 5)).reshape(
                                                                     1, 10, 5)

        # 预测
        with torch.no_grad():
            prob = model(torch.FloatTensor(processed_serv),
                         torch.FloatTensor(processed_lat)).item()

        print("\n案例2：新增服务输入")
        print(f"输入形状: 服务数据 {gathered.shape}, 延迟数据 {latency.shape}")
        print(f"新增服务scaler状态: {new_scalers[s]}")
        print(f"预测违例概率: {prob:.4f}")

    # 执行案例 -------------------------------------------------
    case_normal()
    case_new_services()


def main():
    # 使用示例
    train_data, val_data, test_data, service_scalers, latency_scaler = process_data(
    )

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

    trainer = SLOTrainer(service_mode='attention',
                         device='cuda',
                         pos_weight=2.0)
    trainer.prepare_data(train_data, val_data, test_data)

    # 执行训练
    try:
        trainer.train(epochs=500)
    except KeyboardInterrupt:
        print("Training interrupted, saving latest model...")
        trainer.save_checkpoint('interrupted.pth')

    # 最终评估
    test_metrics = trainer.evaluate(trainer.test_loader)
    print(f"\nFinal Test Metrics:")
    print(f"Accuracy: {test_metrics['accuracy']:.4f}")
    print(f"AUC-ROC: {test_metrics['auc']:.4f}")
    print(f"F1 Score: {test_metrics['f1']:.4f}")


if __name__ == "__main__":
    main()
