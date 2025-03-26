import os
import sys
import time
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F

# 设置项目根目录
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.append(PROJECT_ROOT)

from predictor.slo_predictor_hotel import DynamicSLOPredictor as DynamicSLOPredictorHotel
from predictor.slo_predictor import DynamicSLOPredictor as DynamicSLOPredictor


def load_model(model_path, type="hotel", device="cuda"):
    """加载预测器模型"""
    if type == "hotel":
        model = DynamicSLOPredictorHotel(service_mode="hier_attention").to(device)
    else:
        model = DynamicSLOPredictor(service_mode="hier_attention").to(device)

    try:
        # 加载完整的检查点文件
        checkpoint = torch.load(model_path, map_location=device)

        # 检查是否包含model_state键
        if "model_state" in checkpoint:
            # 只加载模型状态部分
            model.load_state_dict(checkpoint["model_state"])
        else:
            # 尝试直接加载（兼容旧格式）
            model.load_state_dict(checkpoint)

        model.eval()
        print(f"模型 {model_path} 成功加载到 {device} 设备")
        return model
    except Exception as e:
        print(f"模型加载失败: {str(e)}")
        print("请确保模型文件格式正确")
        raise


def evaluate_inference_speed(model, input_state, input_latency, num_runs=1000, device="cuda"):
    """评估模型推理速度"""
    # 确保输入数据在正确设备上
    input_state = input_state.to(device)
    input_latency = input_latency.to(device)

    # 预热阶段
    warmup_runs = 100 if device == "cuda" else 10  # CPU预热次数较少
    print(f"{device.upper()} 预热中 ({warmup_runs}次)...")
    for _ in range(warmup_runs):
        with torch.no_grad():
            _ = model(input_state, input_latency)

    # 计时推理速度
    print(f"开始评估 {device.upper()} 推理速度，运行 {num_runs} 次...")
    start_time = time.time()

    for _ in range(num_runs):
        with torch.no_grad():
            predictions = model(input_state, input_latency)
            probs = F.softmax(predictions, dim=1)

    end_time = time.time()
    total_time = end_time - start_time
    avg_time = total_time / num_runs
    fps = num_runs / total_time

    return {"总时间(秒)": total_time, "平均推理时间(毫秒)": avg_time * 1000, "每秒推理次数": fps}


def main():
    # 模型路径和对应的服务数量
    model_configs = [{
        "path": "/home/tomly/DeepDynamicRM/predictor/model/hotel/best_model.pth",
        "service_num": 17,
        "type": "hotel"
    }, {
        "path": "/home/tomly/DeepDynamicRM/predictor/model/best_model.pth",
        "service_num": 28,
        "type": "model"
    }]

    # 公共参数
    batch_size = 1
    window_size = 30
    service_feature_dim = 26
    latency_feature_dim = 6

    # 设备列表 - 同时测试GPU和CPU
    devices = ["cuda", "cpu"]

    # 结果存储 - 按模型和设备分类
    all_results = {}

    # 评估每个模型在每个设备上的性能
    for i, model_config in enumerate(model_configs):
        model_path = model_config["path"]
        service_num = model_config["service_num"]
        model_type = model_config["type"]
        model_name = f"模型 {i+1} ({Path(model_path).parent.name})"

        print(f"\n=== 评估 {model_name} ===")
        print(f"服务数量: {service_num}")

        # 创建输入数据(初始在CPU上)
        input_state = torch.randn(batch_size, window_size, service_num, service_feature_dim)
        input_latency = torch.randn(batch_size, window_size, latency_feature_dim)

        all_results[model_name] = {}

        # 在不同设备上测试
        for device in devices:
            print(f"\n在 {device.upper()} 上测试")
            try:
                # 加载模型到指定设备
                model = load_model(model_path, type=model_type, device=device)

                # 评估速度
                num_runs = 1000 if device == "cuda" else 1000  # CPU测试运行次数较少
                model_results = evaluate_inference_speed(model,
                                                         input_state,
                                                         input_latency,
                                                         num_runs=num_runs,
                                                         device=device)
                all_results[model_name][device] = model_results

            except Exception as e:
                print(f"在 {device} 上测试失败: {str(e)}")
                continue

    # 打印结果
    print("\n======= 模型推理速度评估结果 =======")

    for model_name, device_results in all_results.items():
        print(f"\n{model_name}:")
        for device, results in device_results.items():
            print(f"  {device.upper()}:")
            for metric, value in results.items():
                print(f"    {metric}: {value:.4f}")

    # 比较GPU与CPU的性能差异
    print("\n======= GPU vs CPU 性能对比 =======")
    for model_name, device_results in all_results.items():
        if "cuda" in device_results and "cpu" in device_results:
            gpu_time = device_results["cuda"]["平均推理时间(毫秒)"]
            cpu_time = device_results["cpu"]["平均推理时间(毫秒)"]
            speedup = cpu_time / gpu_time

            print(f"\n{model_name}:")
            print(f"  GPU/CPU 加速比: {speedup:.2f}x")
            print(f"  加速百分比: {(speedup-1)*100:.2f}%")

    # 比较不同模型在相同设备上的性能差异
    if len(all_results) > 1:
        model_names = list(all_results.keys())

        for device in devices:
            print(f"\n======= {device.upper()} 上不同模型性能对比 =======")

            models_with_device = []
            for model_name in model_names:
                if device in all_results[model_name]:
                    models_with_device.append(model_name)

            if len(models_with_device) < 2:
                print(f"没有足够的模型在 {device.upper()} 上完成测试，无法比较")
                continue

            model1 = models_with_device[0]
            model2 = models_with_device[1]

            model1_time = all_results[model1][device]["平均推理时间(毫秒)"]
            model2_time = all_results[model2][device]["平均推理时间(毫秒)"]
            speedup = model1_time / model2_time

            faster_model = model1 if model1_time < model2_time else model2
            print(f"更快的模型: {faster_model}")
            print(f"速度比率: {speedup:.4f}")
            print(f"速度差异: {abs(1-speedup):.2%}")


if __name__ == "__main__":
    main()
