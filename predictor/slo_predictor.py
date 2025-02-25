import os
import sys
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)


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
    # 生成滑动窗口数据集
    # ----------------------------------
    def create_dataset(X_serv, X_lat, raw_lat, start_idx, end_idx):
        samples_serv = []
        samples_lat = []
        labels = []
        
        # 计算有效范围（考虑预测窗口）
        effective_end = end_idx - pred_window
        
        for i in range(start_idx, effective_end + 1):
            # 输入窗口：t-window_size 到 t-1
            window_start = max(i - window_size, 0)
            window_end = i
            
            # 服务数据窗口（三维时序数据）
            serv_window = X_serv[window_start:window_end]  # (seq_len, 28, 25)
            
            # 延迟数据窗口（二维时序数据）
            lat_window = X_lat[window_start:window_end]     # (seq_len, 5)
            
            # 如果窗口不足长度，进行padding（前向填充）
            if serv_window.shape[0] < window_size:
                pad_size = window_size - serv_window.shape[0]
                serv_window = np.pad(serv_window, 
                                   ((pad_size,0), (0,0), (0,0)),
                                   mode='edge')
                lat_window = np.pad(lat_window,
                                  ((pad_size,0), (0,0)),
                                  mode='edge')
            
            # 预测窗口：t 到 t+pred_window-1
            pred_values = raw_lat[i:i+pred_window, percentile_idx]
            
            # 生成标签：平均P99延迟是否超过阈值
            avg_p99 = np.mean(pred_values)
            label = 1 if avg_p99 > threshold else 0
            
            samples_serv.append(serv_window)
            samples_lat.append(lat_window)
            labels.append(label)
        
        return (
            np.array(samples_serv),
            np.array(samples_lat),
            np.array(labels)
        )

    # ----------------------------------
    # 保持时序的数据划分
    # ----------------------------------
    n_timesteps = X_service.shape[0]
    
    # 计算划分点（保持原始比例）
    split_idx = int(n_timesteps * 0.7)
    val_idx = int(n_timesteps * 0.85)
    
    # 训练集（前70%）
    X_train_serv, X_train_lat, y_train = create_dataset(
        X_service, X_latency, raw_latency, 0, split_idx
    )
    
    # 验证集（中间15%）
    X_val_serv, X_val_lat, y_val = create_dataset(
        X_service, X_latency, raw_latency, split_idx, val_idx
    )
    
    # 测试集（后15%）
    X_test_serv, X_test_lat, y_test = create_dataset(
        X_service, X_latency, raw_latency, val_idx, n_timesteps
    )

    # ----------------------------------
    # 输出数据集信息
    # ----------------------------------
    def print_dataset_info(name, serv, lat, labels):
        print(f"\n{name}数据集:")
        print(f"服务数据形状: {serv.shape} (样本数, 时间步, 服务数, 特征数)")
        print(f"延迟数据形状: {lat.shape} (样本数, 时间步, 特征数)")
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

# 使用示例
train_data, val_data, test_data, service_scalers, latency_scaler = process_data()



def main():
    pass


if __name__ == "__main__":
    main()