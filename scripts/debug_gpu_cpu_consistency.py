#!/usr/bin/env python3
"""
GPU和CPU计算一致性调试脚本
"""

import torch
import numpy as np
import time

def set_deterministic_mode():
    """设置确定性模式"""
    torch.manual_seed(42)
    torch.cuda.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    np.random.seed(42)
    
    # 设置确定性算法
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def check_tensor_consistency(gpu_tensor, cpu_tensor, name="Tensor"):
    """检查张量一致性"""
    print(f"\n=== {name} 一致性检查 ===")
    
    # 检查形状
    print(f"GPU形状: {gpu_tensor.shape}")
    print(f"CPU形状: {cpu_tensor.shape}")
    
    # 检查数据类型
    print(f"GPU数据类型: {gpu_tensor.dtype}")
    print(f"CPU数据类型: {cpu_tensor.dtype}")
    
    # 检查设备
    print(f"GPU设备: {gpu_tensor.device}")
    print(f"CPU设备: {cpu_tensor.device}")
    
    # 检查数值范围
    print(f"GPU范围: [{gpu_tensor.min():.6f}, {gpu_tensor.max():.6f}]")
    print(f"CPU范围: [{cpu_tensor.min():.6f}, {cpu_tensor.max():.6f}]")
    
    # 检查统计信息
    print(f"GPU均值: {gpu_tensor.mean():.6f}")
    print(f"CPU均值: {cpu_tensor.mean():.6f}")
    print(f"GPU标准差: {gpu_tensor.std():.6f}")
    print(f"CPU标准差: {cpu_tensor.std():.6f}")
    
    # 检查一致性 - 确保数据类型匹配
    cpu_from_gpu = gpu_tensor.cpu()
    
    # 如果数据类型不同，转换为相同类型进行比较
    if cpu_from_gpu.dtype != cpu_tensor.dtype:
        print(f"⚠️  数据类型不匹配，将GPU结果转换为 {cpu_tensor.dtype}")
        cpu_from_gpu = cpu_from_gpu.to(cpu_tensor.dtype)
    
    # 使用不同的容差
    tolerances = [1e-3, 1e-5, 1e-7, 1e-9]
    
    for tol in tolerances:
        try:
            is_close = torch.allclose(cpu_from_gpu, cpu_tensor, atol=tol, rtol=tol)
            print(f"容差 {tol}: {'✓ 一致' if is_close else '✗ 不一致'}")
        except RuntimeError as e:
            print(f"容差 {tol}: 比较失败 - {e}")
    
    # 检查最大差异
    try:
        max_diff = torch.max(torch.abs(cpu_from_gpu - cpu_tensor))
        print(f"最大差异: {max_diff:.2e}")
    except RuntimeError as e:
        print(f"计算最大差异失败: {e}")
    
    # 检查相对差异
    try:
        relative_diff = torch.max(torch.abs(cpu_from_gpu - cpu_tensor) / (torch.abs(cpu_tensor) + 1e-8))
        print(f"最大相对差异: {relative_diff:.2e}")
    except RuntimeError as e:
        print(f"计算相对差异失败: {e}")

def test_basic_operations():
    """测试基础操作的一致性"""
    print("=== 基础操作一致性测试 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建测试数据
    size = 1000
    a_cpu = torch.randn(size, size)
    b_cpu = torch.randn(size, size)
    
    a_gpu = a_cpu.to(device)
    b_gpu = b_cpu.to(device)
    
    # 测试矩阵乘法
    print("\n1. 矩阵乘法测试")
    c_cpu = torch.mm(a_cpu, b_cpu)
    c_gpu = torch.mm(a_gpu, b_gpu)
    check_tensor_consistency(c_gpu, c_cpu, "矩阵乘法结果")
    
    # 测试元素级运算
    print("\n2. 元素级运算测试")
    d_cpu = a_cpu * b_cpu + a_cpu
    d_gpu = a_gpu * b_gpu + a_gpu
    check_tensor_consistency(d_gpu, d_cpu, "元素级运算结果")
    
    # 测试激活函数
    print("\n3. 激活函数测试")
    e_cpu = torch.relu(a_cpu)
    e_gpu = torch.relu(a_gpu)
    check_tensor_consistency(e_gpu, e_cpu, "ReLU激活结果")

def test_neural_network_consistency():
    """测试神经网络的一致性"""
    print("\n=== 神经网络一致性测试 ===")
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # 创建模型
    model_cpu = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10)
    )
    
    model_gpu = torch.nn.Sequential(
        torch.nn.Linear(100, 50),
        torch.nn.ReLU(),
        torch.nn.Linear(50, 10)
    ).to(device)
    
    # 复制权重
    for cpu_param, gpu_param in zip(model_cpu.parameters(), model_gpu.parameters()):
        gpu_param.data.copy_(cpu_param.data)
    
    # 创建测试数据
    x_cpu = torch.randn(32, 100)
    x_gpu = x_cpu.to(device)
    
    # 前向传播
    print("\n1. 前向传播测试")
    model_cpu.eval()
    model_gpu.eval()
    
    with torch.no_grad():
        y_cpu = model_cpu(x_cpu)
        y_gpu = model_gpu(x_gpu)
    
    check_tensor_consistency(y_gpu, y_cpu, "前向传播结果")
    
    # 反向传播
    print("\n2. 反向传播测试")
    model_cpu.train()
    model_gpu.train()
    
    # 定义损失函数和优化器
    criterion = torch.nn.CrossEntropyLoss()
    optimizer_cpu = torch.optim.Adam(model_cpu.parameters(), lr=0.01)
    optimizer_gpu = torch.optim.Adam(model_gpu.parameters(), lr=0.01)
    
    # 创建标签
    target_cpu = torch.randint(0, 10, (32,))
    target_gpu = target_cpu.to(device)
    
    # 前向传播
    y_cpu = model_cpu(x_cpu)
    y_gpu = model_gpu(x_gpu)
    
    # 计算损失
    loss_cpu = criterion(y_cpu, target_cpu)
    loss_gpu = criterion(y_gpu, target_gpu)
    
    print(f"CPU损失: {loss_cpu.item():.6f}")
    print(f"GPU损失: {loss_gpu.item():.6f}")
    print(f"损失差异: {abs(loss_cpu.item() - loss_gpu.item()):.2e}")
    
    # 反向传播
    optimizer_cpu.zero_grad()
    optimizer_gpu.zero_grad()
    
    loss_cpu.backward()
    loss_gpu.backward()
    
    # 更新参数
    optimizer_cpu.step()
    optimizer_gpu.step()
    
    # 检查参数一致性
    print("\n3. 参数更新一致性")
    for i, (cpu_param, gpu_param) in enumerate(zip(model_cpu.parameters(), model_gpu.parameters())):
        check_tensor_consistency(gpu_param, cpu_param, f"参数 {i}")

def test_mixed_precision():
    """测试混合精度的一致性"""
    print("\n=== 混合精度一致性测试 ===")
    
    if not torch.cuda.is_available():
        print("CUDA不可用，跳过混合精度测试")
        return
    
    device = torch.device('cuda')
    
    # 创建测试数据
    x_cpu = torch.randn(100, 100)
    x_gpu = x_cpu.to(device)
    
    # 标准精度
    y_cpu = torch.mm(x_cpu, x_cpu)
    y_gpu = torch.mm(x_gpu, x_gpu)
    
    print("\n1. 标准精度 (FP32)")
    check_tensor_consistency(y_gpu, y_cpu, "标准精度结果")
    
    # 混合精度
    with torch.cuda.amp.autocast():
        y_gpu_mixed = torch.mm(x_gpu, x_gpu)
    
    print("\n2. 混合精度 (FP16)")
    # 将混合精度结果转换为FP32进行比较
    y_gpu_mixed_fp32 = y_gpu_mixed.float()
    check_tensor_consistency(y_gpu_mixed_fp32, y_cpu, "混合精度结果(转换为FP32)")
    
    # 直接比较FP16和FP32的差异
    print("\n3. FP16 vs FP32 直接比较")
    y_gpu_mixed_cpu = y_gpu_mixed.cpu()
    y_cpu_fp16 = y_cpu.half()
    
    print(f"FP16 GPU结果形状: {y_gpu_mixed_cpu.shape}")
    print(f"FP16 CPU结果形状: {y_cpu_fp16.shape}")
    print(f"FP16 GPU数据类型: {y_gpu_mixed_cpu.dtype}")
    print(f"FP16 CPU数据类型: {y_cpu_fp16.dtype}")
    
    # 转换为相同类型进行比较
    if y_gpu_mixed_cpu.dtype != y_cpu_fp16.dtype:
        y_gpu_mixed_cpu = y_gpu_mixed_cpu.to(y_cpu_fp16.dtype)
    
    try:
        is_close = torch.allclose(y_gpu_mixed_cpu, y_cpu_fp16, atol=1e-3, rtol=1e-3)
        print(f"FP16精度比较: {'✓ 一致' if is_close else '✗ 不一致'}")
    except RuntimeError as e:
        print(f"FP16精度比较失败: {e}")

def main():
    """主函数"""
    print("GPU和CPU计算一致性调试")
    print("=" * 50)
    
    # 设置确定性模式
    set_deterministic_mode()
    
    # 检查设备
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"使用设备: {device}")
    
    if torch.cuda.is_available():
        print(f"GPU: {torch.cuda.get_device_name(0)}")
        print(f"CUDA版本: {torch.version.cuda}")
    print()
    
    # 运行测试
    test_basic_operations()
    test_neural_network_consistency()
    test_mixed_precision()
    
    print("\n" + "=" * 50)
    print("调试完成！")
    print("\n常见问题解决建议:")
    print("1. 使用 torch.allclose() 而不是 == 比较")
    print("2. 设置合适的容差 (atol, rtol)")
    print("3. 确保数据类型一致")
    print("4. 使用确定性模式进行调试")
    print("5. 检查是否有NaN或Inf值")

if __name__ == "__main__":
    main() 