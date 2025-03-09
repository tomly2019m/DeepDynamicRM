import numpy as np
from sklearn.preprocessing import StandardScaler


def test_data_pipeline():
    """端到端数据处理流程测试（完全使用模拟数据）"""
    print("\n=== 开始端到端数据处理测试 ===")
    np.random.seed(42)  # 固定随机种子

    # ----------------------------------
    # 生成模拟测试数据
    # ----------------------------------
    n_timesteps = 100  # 测试用时间步数量
    n_services = 28  # 与服务数量一致
    n_metrics = 6  # 6个指标
    n_stats = 4  # max/min/mean/std

    # 生成符合实际形状的模拟数据
    gathered = np.random.randn(n_timesteps, n_services, n_metrics, n_stats) * 10 + 50  # (100,28,6,4)
    latency = np.random.randn(n_timesteps, 5) * 2 + 10  # (100,5)
    replicas = np.random.randint(1, 20, size=(n_services, ))  # (28,)
    labels = np.random.randint(0, 2, size=(n_timesteps, ))  # (100,)

    print("Shape of gathered:", gathered.shape)
    print("Shape of latency:", latency.shape)
    print("Shape of replicas:", replicas.shape)
    print("Shape of labels:", labels.shape)

    # ----------------------------------
    # 执行完整处理流程
    # ----------------------------------
    # 步骤1：特征融合
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

    X_service, service_scalers = process_gathered(gathered, replicas)

    # 步骤2：延迟数据标准化
    latency_scaler = StandardScaler()
    X_latency = latency_scaler.fit_transform(latency)

    # 步骤3：数据划分
    split_idx = int(n_timesteps * 0.7)
    val_idx = int(n_timesteps * 0.85)

    X_train = X_service[:split_idx]
    X_val = X_service[split_idx:val_idx]
    X_test = X_service[val_idx:]

    # ----------------------------------
    # 验证测试项
    # ----------------------------------
    # 测试1：特征融合结果验证
    assert X_service.shape == (n_timesteps, n_services, 24 + 1), \
        f"特征融合形状错误，期望({n_timesteps},28,25)，实际得到{X_service.shape}"

    # 验证副本数量是否正确合并
    sample_idx = np.random.randint(0, n_timesteps)
    service_idx = np.random.randint(0, n_services)
    assert X_service[sample_idx, service_idx, 24] == replicas[service_idx], \
        "副本数量合并错误"

    # 测试2：标准化验证
    # 随机选取一个服务的特征检查标准化
    test_service = np.random.randint(0, n_services)
    service_data = X_service[:, test_service, :24]  # 排除副本数列
    assert np.allclose(service_data.mean(axis=0), 0, atol=1e-7), "标准化均值非零"
    assert np.allclose(service_data.std(axis=0), 1, atol=1e-7), "标准化标准差非1"

    # 测试3：数据划分验证
    assert len(X_train) == 70, f"训练集大小错误，期望70，实际{len(X_train)}"
    assert len(X_val) == 15, f"验证集大小错误，期望15，实际{len(X_val)}"
    assert len(X_test) == 15, f"测试集大小错误，期望15，实际{len(X_test)}"

    # 测试4：延迟数据处理验证
    assert np.allclose(X_latency.mean(axis=0), 0, atol=1e-7), "延迟数据标准化错误"
    assert X_latency.shape == latency.shape, "延迟数据形状变化"

    print("=== 所有测试通过 ===")


# 执行测试（可直接在代码中调用）
test_data_pipeline()
